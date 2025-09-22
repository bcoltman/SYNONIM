import numpy as np
import random
import logging
import scipy.sparse as sp
from multiprocessing import Pool
from typing import Any, Dict, List, Optional, Tuple, Union
from scipy.optimize import minimize

import time

from threadpoolctl import threadpool_limits

from synonim.core import Profile, Model
from synonim.optimizers import AbundanceOptimizer, Solution
from synonim.optimizers.utils import generate_unified_taxonomy_matrix, clr_transform

logger = logging.getLogger(__name__)

def run_aga_wrapper(args: Tuple[Any, np.ndarray]) -> Dict[str, Any]:
    """
    Multiprocessing helper to seed the random number generators and invoke the GA.
    
    Parameters
    ----------
    args : tuple
        optimizer : AbundanceGenetic
            The GA optimizer instance.
        target_vector : np.ndarray
            CLR-transformed abundance target for one scenario.
        scenario_id : int
            scenario-specific offset for random seed.
            
    Returns
    -------
    Dict[str, Any]
        The dictionary returned by optimizer.genetic_algorithm.
    """
    optimizer, target_vector, scenario_id = args
    seed = optimizer.seed + scenario_id
    random.seed(seed)
    np.random.seed(seed)
    
    with threadpool_limits(limits=1):
        return optimizer.genetic_algorithm(
            target_vector,
            optimizer.G,
            optimizer.consortia_size, 
            scenario_id
        )

            
class AbundanceGenetic(AbundanceOptimizer):
    def __init__(
        self,
        model: Model,
        consortia_size: int,
        population_size: int = 100,
        generations: int = 10000,
        mutation_rate: float = 0.05,
        fitness_threshold: Optional[float] = None,
        max_unchanged_generations: int = 1000,
        required_genomes: Optional[List[Union[str, Profile]]] = None,
        taxonomy_constraints: Optional[Dict[str, Any]] = None,
        taxonomic_levels: Optional[List[str]] = None,
        processes: int = 1,
        tournament_size: int = 10,
        seed: int = 42,
        f_bounds: Union[Tuple[float, float], Dict[int, Tuple[float, float]]] = (0.1, 1.0),
        epsilon: float = 1e-6,
        p_total: float = 1.0,
        exploration_rate: float = 0.1
    ) -> None:
        super().__init__(
            model=model,
            consortia_size=consortia_size,
            taxonomy_constraints=taxonomy_constraints,
            taxonomic_levels=taxonomic_levels,
            required_genomes=required_genomes
        )
        # Set up abundance multiplier bounds
        num_genomes = self.G.shape[1]
        if isinstance(f_bounds, dict):
            f_min = np.array([f_bounds.get(i, (0.0, 0.0))[0] for i in range(num_genomes)])
            f_max = np.array([f_bounds.get(i, (0.0, 0.0))[1] for i in range(num_genomes)])
        else:
            f_min = np.full(num_genomes, f_bounds[0])
            f_max = np.full(num_genomes, f_bounds[1])
        self.is_fully_fixed = bool(np.all(f_min == f_max))
        self.f_const = f_min if self.is_fully_fixed else None
        
        # Store GA parameters
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.fitness_threshold = fitness_threshold
        self.max_unchanged_generations = max_unchanged_generations
        self.processes = processes
        self.tournament_size = tournament_size
        self.seed = seed
        self.f_bounds = f_bounds
        self.epsilon = epsilon
        self.p_total = p_total
        self.exploration_rate = exploration_rate
        
        random.seed(seed)
        np.random.seed(seed)
        # self.genome_names = self.model.genome_names
        
            # Build unified taxonomy matrix and precompute counts
        if taxonomy_constraints and self.genome_labels is not None:
            # Generate the sparse taxonomy matrix (taxa × genomes)
            self.T_unified, self.unified_taxon_to_index = generate_unified_taxonomy_matrix(
                self.genome_labels, taxonomic_levels
            )
            self.Tcsc = self.T_unified.tocsc()
            self.num_taxa = self.T_unified.shape[0]
            
            # Precompute a dense (taxa × genomes) matrix for fast indexing
            self.taxonomy_matrix_dense = self.T_unified.T.toarray().astype(int)
            
            # A zero‐vector to use when no genome is being removed
            self._zero_removal = np.zeros(self.num_taxa, dtype=int)
            
            # For each taxon row i, list of genomes belonging to that taxon
            self.taxon_to_genomes = [self.T_unified.getrow(i).indices for i in range(self.num_taxa)]
            # For each genome j, list of taxa to which it belongs
            self.genome_to_taxa = [self.Tcsc.getcol(j).indices for j in range(num_genomes)]
            
            # Initialize min/max/exact arrays (length = num_taxa)
            self._min_counts = np.zeros(self.num_taxa, dtype=float)
            self._max_counts = np.full(self.num_taxa, np.inf, dtype=float)
            self._exact_counts = np.full(self.num_taxa, np.nan, dtype=float)
            
            # Populate them from taxonomy_constraints
            for level, rules in taxonomy_constraints.items():
                default_rules = rules.get("default", {})
                # 1) Handle explicit taxa constraints
                for taxon, cons in rules.items():
                    if taxon == "default":
                        continue
                    key = f"{level}-{taxon}"
                    idx = self.unified_taxon_to_index.get(key)
                    if idx is None:
                        continue
                    # Assign min, max, exact if provided
                    self._min_counts[idx] = cons.get("min", self._min_counts[idx])
                    self._max_counts[idx] = cons.get("max", self._max_counts[idx])
                    if "exact" in cons:
                        self._exact_counts[idx] = cons["exact"]
                        
                # 2) Apply default rules to all other taxa at this level
                for composite_key, idx in self.unified_taxon_to_index.items():
                    lvl, t = composite_key.split("-", 1)
                    if lvl != level or t in rules:
                        continue
                    # If default min specified, assign
                    if "min" in default_rules:
                        self._min_counts[idx] = default_rules["min"]
                    # If default max specified, assign
                    if "max" in default_rules:
                        self._max_counts[idx] = default_rules["max"]
                        
        else:
            # No taxonomy constraints → build "empty" but valid arrays
            
            # A (0 taxa × num_genomes) sparse matrix
            self.T_unified = sp.csr_matrix((0, num_genomes))
            # self.taxonomy_matrix_dense = self.T_unified.T.toarray()
            self.taxonomy_matrix_dense = self.T_unified.T.toarray().astype(int)
            
            self.Tcsc = None
            self.num_taxa = 0
            # self.taxonomy_matrix_dense = self.T_unified.T.toarray()
            
            # No mapping of taxon→genome or genome→taxon
            self.taxon_to_genomes = []
            self.genome_to_taxa = [[] for _ in range(num_genomes)]
            
            # Empty float arrays for min/max/exact (length 0)
            self._min_counts = np.zeros(0, dtype=float)
            self._max_counts = np.full(0, np.inf, dtype=float)
            self._exact_counts = np.full(0, np.nan, dtype=float)
            
            # A zero‐vector of length 0 for removal
            self._zero_removal = np.zeros(0, dtype=int)
            
    def __getstate__(self) -> dict:
        """
        Exclude the model instance when pickling to avoid circular references.
        """
        state = self.__dict__.copy()
        state['model'] = None
        return state
        
    @property
    def descriptive_name(self) -> str:
        """
        Return a human-readable name summarizing key GA parameters.
        """
        return f"AbundanceGenetic_MutRate{self.mutation_rate}"
        
    def _get_counts(self, mask: np.ndarray) -> np.ndarray:
        """
        Compute the number of selected genomes per taxon.
        
        Parameters
        ----------
        mask : np.ndarray of int (0 or 1), shape (num_genomes,)
            Selection mask for genomes.
            
        Returns
        -------
        np.ndarray of shape (num_taxa,)
            Counts of genomes per taxon.
        """
        return self.T_unified.dot(mask)
    
    def _can_transition_batch(
        self,
        current_counts: np.ndarray,
        candidates_to_add: np.ndarray,
        candidate_to_remove: Optional[int],
        slots_left: int
    ) -> np.ndarray:
        M = self.taxonomy_matrix_dense
        adds = M[candidates_to_add, :]
        if candidate_to_remove is None:
            remove = self._zero_removal
        else:
            remove = M[candidate_to_remove]
        # broadcast to shape (n_cand, num_taxa)
        current = current_counts[None, :]
        removal = remove[None, :]
        new_counts = current + adds - removal
        within_max = np.all(new_counts <= self._max_counts[None, :], axis=1)
        shortfall = np.clip(self._min_counts[None, :] - new_counts, 0, None)
        can_fill = (shortfall.sum(axis=1) <= slots_left)
        return within_max & can_fill
    
    def get_candidate_bounds(self, candidate_index: int) -> Tuple[float, float]:
        """
        Retrieve (f_min, f_max) abundance bounds for a genome.
        
        Parameters
        ----------
        candidate_index : int
        Returns
        -------
        tuple of (min, max)
        """
        if isinstance(self.f_bounds, dict):
            return self.f_bounds.get(candidate_index, (0.1, 1.0))
        return self.f_bounds
        
    def evaluate_taxonomy_penalty(
        self,
        mask: np.ndarray
    ) -> float:
        """
        Calculate total penalty for taxonomy constraint violations.
        
        Parameters
        ----------
        mask : np.ndarray of int (0 or 1), shape (num_genomes,)
            Selection mask for genomes.
            
        Returns
        -------
        float
            Sum of minima undercounts, excess counts, and deviations from exact rules
        """
        if self.T_unified is None:
            return 0.0
        counts = self.T_unified.dot(mask)
        under_counts = np.clip(self._min_counts - counts, 0, None)
        over_counts = np.clip(counts - self._max_counts, 0, None)
        exact_deviation = np.where(
            np.isnan(self._exact_counts),
            0,
            np.abs(counts - self._exact_counts)
        )
        return (under_counts + over_counts + exact_deviation).sum()
        
    def is_feasible(
        self,
        mask: np.ndarray
    ) -> bool:
        """
        Check if a genome selection mask satisfies all taxonomy constraints.
        
        Parameters
        ----------
        mask : np.ndarray of int (0 or 1)
            Selection mask.
            
        Returns
        -------
        bool
            True if there are no constraint violations.
        """
        return self.evaluate_taxonomy_penalty(mask) == 0.0
    
    def _eval_mask(self, mask: np.ndarray, T: np.ndarray) -> float:
        """
        Evaluate a binary selection mask by fitting continuous abundances f to minimize
        CLR-transformed error to target T, using an analytic Jacobian for speed.
        Returns negative squared error (higher is better), or -np.inf if infeasible.
        """
        # 1) Taxonomy feasibility check
        if not self.is_feasible(mask):
            return -np.inf
            
        # 2) Fully-fixed-f branch (no inner solve)
        if self.is_fully_fixed:
            full_f = mask.astype(float) * self.f_const
            v = self.G @ full_f
            clr_v = clr_transform(v, self.epsilon)
            return -np.sum((clr_v - T) ** 2)
            
        # 3) Identify selected indices
        selected = np.nonzero(mask)[0]
        if selected.size == 0:
            raise ValueError("Mask selects no genomes.")
        # 4) Bounds and feasibility of p_total
        bounds = [self.get_candidate_bounds(j) for j in selected]
        lower = np.array([b[0] for b in bounds], dtype=float)
        upper = np.array([b[1] for b in bounds], dtype=float)
        sum_lower, sum_upper = lower.sum(), upper.sum()
        if self.p_total < sum_lower or self.p_total > sum_upper:
            raise ValueError(
                f"p_total {self.p_total} not in feasible range [{sum_lower}, {sum_upper}]."
            )
            
        # 5) Boundary-case shortcuts
        if np.isclose(self.p_total, sum_upper):
            full_f = np.zeros(self.G.shape[1], dtype=float)
            full_f[selected] = upper
            clr_v = clr_transform(self.G @ full_f, self.epsilon)
            return -np.sum((clr_v - T) ** 2)
        if np.isclose(self.p_total, sum_lower):
            full_f = np.zeros(self.G.shape[1], dtype=float)
            full_f[selected] = lower
            clr_v = clr_transform(self.G @ full_f, self.epsilon)
            return -np.sum((clr_v - T) ** 2)
            
        # 6) Initial guess summing to p_total
        m = selected.size
        init_val = self.p_total / m
        init_f = np.clip(init_val, lower, upper)
        diff = self.p_total - init_f.sum()
        if abs(diff) > 1e-8:
            slack = upper - lower
            total_slack = slack.sum()
            if total_slack > 0:
                init_f += diff * (slack / total_slack)
                init_f = np.clip(init_f, lower, upper)
                
        # 7) Pre-slice G and constants
        G_sel = self.G[:, selected]      # shape (d, m)
        d, m = G_sel.shape
        inv_d = 1.0 / d
        
        # 8) Objective in CLR-space
        def obj(f: np.ndarray) -> float:
            v = G_sel @ f                           # (d,)
            if np.any(v < 0) or np.any(np.isnan(v)) or np.any(np.isinf(v)):
                return 1e10    # (d,)
            v_eps = v + self.epsilon                # (d,)
            log_v = np.log(v_eps)                   # (d,)
            geom_log = inv_d * np.sum(log_v)        # scalar
            clr = log_v - geom_log                  # (d,)
            return np.sum((clr - T) ** 2)           # scalar
            
        # 9) Analytic Jacobian
        def jac(f: np.ndarray) -> np.ndarray:
            """
            Analytic gradient ∇F(f) for the objective
                F(f) = || clr(G_sel @ f + ε) – T ||²
                
            Parameters
            ----------
            f : (n,) ndarray
                Current value of the parameters.
                
            Returns
            -------
            g : (n,) ndarray
                Gradient with respect to f.
            """
            # ----- forward pass (reuse everything the objective already computes) -----
            v = G_sel @ f                             # (d,)
            if np.any(v < 0) or np.any(~np.isfinite(v)):
                # Outside the feasible region → push solver back inside.
                return np.zeros_like(f)
            
            z   = v + optimizer.epsilon               # (d,)
            b   = 1.0 / z                             # (d,)   inverse once, reuse twice
            l   = np.log(z)                           # (d,)
            clr = l - l.mean()                        # (d,)
            s   = clr - T                             # (d,)
            
            # ----- backward pass (formula from the derivation) ------------------------
            q   = b * (s - s.mean())                  # (d,)
            g   = 2.0 * (G_sel.T @ q)                 # (n,)
            
            return g
            
        # 10) Equality constraint: sum(f) == p_total
        cons = ({'type': 'eq', 'fun': lambda f: np.sum(f) - self.p_total},)
        
        # 11) Run SLSQP with analytic gradient
        # result = minimize(
            # obj,
            # init_f,
            # method='SLSQP',
            # jac=jac,
            # bounds=list(zip(lower, upper)),
            # constraints=cons,
            # options={'ftol': 1e-4, 'maxiter': 200}
        # )
        result = minimize(
            obj,
            init_f,
            jac=jac,
            method='trust-constr',
            bounds=list(zip(lower, upper)),
            constraints=cons,
            options={
                'gtol': 1e-4,         # Gradient tolerance
                'xtol': 1e-4,         # Change in x tolerance
                'initial_tr_radius': 5.0,  # Initial trust region radius
                'maxiter': 1000,      # Max iterations
                'verbose': 3           # Verbose output
            }
        )
        # result = minimize(
            # obj,
            # init_f,
            # jac=jac,
            # method='trust-constr',
            # bounds=list(zip(lower, upper)),
            # constraints=cons,
            # options={'gtol': 1e-6, 'finite_diff_rel_step': 1e-4, 'maxiter':1000}
        # )
        if not result.success:
            raise RuntimeError(f"Inner optimization failed: {result.message}")
            
        # 12) Build full f-vector, compute final CLR error
        full_f = np.zeros(self.G.shape[1], dtype=float)
        full_f[selected] = result.x
        clr_v = clr_transform(self.G @ full_f, self.epsilon)
        return -np.sum((clr_v - T) ** 2)
        
    def initialize_population(
        self,
        candidates: np.ndarray,
        consortia_size: int,
        required_indices: List[int]
    ) -> np.ndarray:
        """
        Generate an initial population of feasible individuals.
        
        Each individual is built by:
          1. Satisfying exact/min taxonomy requirements (re-checking feasibility after each pick).
          2. Filling remaining slots with any feasible genomes.
          
        Parameters
        ----------
        candidates : np.ndarray, shape (num_features, num_genomes)
            Abundance feature matrix.
        consortia_size : int
            Number of genomes per individual.
        required_indices : list of int
            Genomes that must be present in every individual.
            
        Returns
        -------
        np.ndarray of shape (population_size, consortia_size)
            Array of genome-index individuals.
        """
        population = np.zeros((self.population_size, consortia_size), dtype=int)
        num_genomes = candidates.shape[1]
        # desired per-taxon counts (min or exact)
        desired_counts = np.where(np.isnan(self._exact_counts),
                                  self._min_counts,
                                  self._exact_counts)
        filled = 0
        max_attempts = self.population_size * 10
        attempts = 0
        
        while filled < self.population_size:
            attempts += 1
            if attempts > max_attempts:
                raise RuntimeError(
                    f"Failed to initialize {self.population_size} feasible individuals after {max_attempts} attempts."
                )
            # start new individual
            individual = list(required_indices)
            mask = np.zeros(num_genomes, dtype=int)
            mask[individual] = 1
            taxon_counts = self._get_counts(mask)
            failed = False
            
            # Phase 1: satisfy exact/min requirements, one pick at a time
            for taxon_index, need in enumerate(desired_counts):
                # how many more needed for this taxon
                to_add = int(need - taxon_counts[taxon_index])
                for _ in range(to_add):
                    slots_left = consortia_size - len(individual)
                    # candidate pool for this taxon
                    pool = np.setdiff1d(self.taxon_to_genomes[taxon_index], individual, assume_unique=True)
                    # filter by feasibility given current counts and slots
                    feasible = pool[
                        self._can_transition_batch(
                            taxon_counts,
                            pool,
                            None,
                            slots_left
                        )
                    ]
                    if feasible.size == 0:
                        failed = True
                        break
                    # pick one genome, update state
                    pick = int(np.random.choice(feasible))
                    individual.append(pick)
                    taxon_counts += self.taxonomy_matrix_dense[pick]
                if failed:
                    break
            if failed:
                continue
                
            # Phase 2: fill remaining slots one at a time
            while len(individual) < consortia_size:
                slots_left = consortia_size - len(individual)
                pool = np.setdiff1d(np.arange(num_genomes), individual, assume_unique=True)
                feasible = pool[
                    self._can_transition_batch(
                        taxon_counts,
                        pool,
                        None,
                        slots_left
                    )
                ]
                if feasible.size == 0:
                    failed = True
                    break
                pick = int(np.random.choice(feasible))
                individual.append(pick)
                taxon_counts += self.taxonomy_matrix_dense[pick]
            if failed or len(individual) != consortia_size:
                continue
                
            population[filled, :] = individual
            filled += 1
            
        return population
        
    def constrained_order_crossover(
        self,
        parent1: np.ndarray,
        parent2: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform an order crossover (OX) that respects taxonomy feasibility.
        
        A slice from parent1 is copied, then parent2 genes are inserted only if
        they maintain feasibility; remaining slots are filled with any feasible genome.
        
        Parameters
        ----------
        parent1, parent2 : np.ndarray of int
            Parent individuals (genome indices).
            
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Two offspring individuals.
        """
        size = len(parent1)
        
        def build_offspring(p_a, p_b):
            # Select slice endpoints
            i, j = sorted(random.sample(range(size), 2))
            offspring = [-1] * size
            offspring[i : j + 1] = p_a[i : j + 1]
            
            # Track mask and counts
            mask_array = np.zeros(self.G.shape[1], dtype=int)
            taxon_counts = np.zeros(self.num_taxa, dtype=int)
            for genome in offspring[i : j + 1]:
                mask_array[genome] = 1
                # mask_array[i] = 1
                taxon_counts += self.taxonomy_matrix_dense[genome]
                
            insert_pos = (j + 1) % size
            
            # Try genes from second parent
            for genome in p_b:
                if insert_pos == i:
                    break
                if mask_array[genome]:
                    continue
                if self._can_transition_batch(
                    taxon_counts,
                    np.array([genome]),
                    None,
                    size - mask_array.sum()
                )[0]:
                    offspring[insert_pos] = genome
                    mask_array[genome] = 1
                    taxon_counts += self.taxonomy_matrix_dense[genome]
                    insert_pos = (insert_pos + 1) % size
                    
            # Fill remaining holes with any feasible genome
            all_genomes = np.arange(mask_array.shape[0])
            while insert_pos != i:
                pool = np.setdiff1d(all_genomes, np.where(mask_array)[0], assume_unique=True)
                feasible = pool[
                    self._can_transition_batch(
                        taxon_counts,
                        pool,
                        None,
                        size - mask_array.sum()
                    )
                ]
                if feasible.size == 0:
                    break
                chosen = int(np.random.choice(feasible))
                offspring[insert_pos] = chosen
                mask_array[chosen] = 1
                taxon_counts += self.taxonomy_matrix_dense[chosen]
                insert_pos = (insert_pos + 1) % size
                
            return np.array(offspring, dtype=int)
            
        child1 = build_offspring(parent1, parent2)
        child2 = build_offspring(parent2, parent1)
        return child1, child2
        
    def crossover(
        self,
        parent1: np.ndarray,
        parent2: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Wrapper for constraint-aware crossover.
        """
        return self.constrained_order_crossover(parent1, parent2)
        
    def select_parents(
        self,
        population: np.ndarray,
        fitness: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Tournament selection: choose two parents based on fitness.
        
        Parameters
        ----------
        population : np.ndarray
            Current GA population (rows are individuals).
        fitness : np.ndarray
            Fitness scores for each individual.
            
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Two selected parents.
        """
        pop_size = population.shape[0]
        if pop_size == 1:
            return population[0].copy(), population[0].copy()
            
        tournament_size = min(self.tournament_size, pop_size)
        contenders = np.random.choice(pop_size, tournament_size, replace=False)
        parentA = population[contenders[np.argmax(fitness[contenders])]]
        
        contenders = np.random.choice(pop_size, tournament_size, replace=False)
        parentB = population[contenders[np.argmax(fitness[contenders])]]
        
        return parentA.copy(), parentB.copy()
        
    def mutate(
        self,
        individual: np.ndarray
    ) -> np.ndarray:
        """
        Apply a single-gene mutation, sampling only feasible replacements.
        
        Parameters
        ----------
        individual : np.ndarray of int
            Parent genome indices.
            
        Returns
        -------
        np.ndarray of int
            Mutated individual.
        """
        mutated = individual.copy()
        num_required = len(self.required_genomes)
        if len(mutated) > num_required and random.random() < self.mutation_rate:
            mask = np.zeros(self.G.shape[1], dtype=int)
            mask[mutated] = 1
            counts = self._get_counts(mask)
            pool = np.setdiff1d(np.arange(self.G.shape[1]), mutated, assume_unique=True)
            feasible = pool[
                self._can_transition_batch(counts, pool, None, 0)
            ]
            if feasible.size:
                swap_position = random.randint(num_required, len(mutated) - 1)
                mutated[swap_position] = int(random.choice(feasible))
        return mutated
        
    def one_shot_repair(
        self,
        individual: np.ndarray
    ) -> np.ndarray:
        """
        Repair an individual by:
          1. Removing excess genomes for over-represented taxa.
          2. Adding genomes to meet under-represented taxa.
          
        Returns
        -------
        np.ndarray of int
            A feasible individual.
        """
        genomes = list(individual)
        num_genomes = self.G.shape[1]
        mask = np.zeros(num_genomes, dtype=int)
        mask[genomes] = 1
        counts = self._get_counts(mask)
        
        excess = counts - self._max_counts
        deficit = self._min_counts - counts
        
        # Remove excess
        for taxon_index in np.where(excess > 0)[0]:
            remove_count = int(excess[taxon_index])
            positions = [i for i, g in enumerate(genomes) if taxon_index in self.genome_to_taxa[g]]
            for pos in random.sample(positions, min(remove_count, len(positions))):
                old_genome = genomes[pos]
                candidates_to_add = np.hstack([
                    self.taxon_to_genomes[d]
                    for d in np.where(deficit > 0)[0]
                ])
                pool = np.setdiff1d(candidates_to_add, genomes, assume_unique=True)
                if pool.size:
                    new_genome = int(random.choice(pool))
                    genomes[pos] = new_genome
                    counts += self.taxonomy_matrix_dense[new_genome]
                    counts -= self.taxonomy_matrix_dense[old_genome]
                    excess = counts - self._max_counts
                    deficit = self._min_counts - counts
                    
        # Fill deficit
        for taxon_index in np.where(deficit > 0)[0]:
            needed = int(deficit[taxon_index])
            pool = np.setdiff1d(self.taxon_to_genomes[taxon_index], genomes, assume_unique=True)
            picks = np.random.choice(pool, size=min(needed, pool.size), replace=False)
            for g in picks:
                genomes.append(int(g))
                counts += self.taxonomy_matrix_dense[g]
                deficit = self._min_counts - counts
                
        return np.array(genomes, dtype=int)
        
    def evaluate_population_fitness(
        self,
        target_clr: np.ndarray,
        candidates: np.ndarray,
        population: np.ndarray
    ) -> np.ndarray:
        """
        Compute fitness for each individual in the population, with optional sharing.
        
        Fitness = base_score - penalty
        or = -CLR_error for fixed-f scenarios.
        
        Returns
        -------
        np.ndarray of float
            Fitness scores.
        """
        pop_size, cons_size = population.shape
        num_genomes = candidates.shape[1]
        binary_matrix = np.zeros((pop_size, num_genomes), dtype=float)
        for i in range(pop_size):
            binary_matrix[i, population[i]] = 1.0
            
        if self.is_fully_fixed:
            F = binary_matrix * self.f_const[np.newaxis, :]
            mixture = F @ candidates.T
            mixture += self.epsilon
            geometric = np.exp(np.mean(np.log(mixture), axis=1))
            clr_values = np.log(mixture / geometric[:, None])
            errors = np.sum((clr_values - target_clr[None, :]) ** 2, axis=1)
            fitness = -errors
            feasible_flags = np.array([
                self.is_feasible(row.astype(int))
                for row in binary_matrix
            ])
            fitness[~feasible_flags] = -np.inf
        else:
            fitness = np.array([
                self._eval_mask(row.astype(int), target_clr)
                for row in binary_matrix
            ])
            
        return fitness
        
    def genetic_algorithm(
        self,
        target_clr: np.ndarray,
        candidates: np.ndarray,
        num_candidates: int,
        scenario_id: int
    ) -> Dict[str, Any]:
        """
        Run the genetic algorithm for a single target scenario, injecting exploration.
        
        Returns
        -------
        Dict[str, Any]
            Contains X_opt, p_opt, best_indices, generations_run, best_fitness,
            archive_solutions, archive_scores, analysis.
        """
        start_time = time.time()
        
        # Initialize population
        population = self.initialize_population(
            candidates,
            num_candidates,
            self.required_genomes
        )
        best_individual = population[0].copy()
        best_fitness = -np.inf
        unchanged_generations = 0
        archive_solutions = []
        archive_scores = []
        seen_masks = set()
        
        for generation in range(self.generations):
            # Evaluate fitness
            fitness = self.evaluate_population_fitness(
                target_clr,
                candidates,
                population
            )
            current_best = np.max(fitness)
            best_index = np.argmax(fitness)
            
            # Log every 10 generations
            if generation % 10 == 0:
                logging.info(f"Evaluating generation {generation}")
                
            if current_best > best_fitness:
                best_fitness = current_best
                best_individual = population[best_index].copy()
                unchanged_generations = 0
            else:
                unchanged_generations += 1
                
            # Archive feasible unique solutions
            for i, individual in enumerate(population):
                mask_vector = np.zeros(candidates.shape[1], dtype=int)
                mask_vector[individual] = 1
                mask_key = tuple(mask_vector.tolist())
                if mask_key not in seen_masks and self.is_feasible(mask_vector):
                    seen_masks.add(mask_key)
                    archive_solutions.append(sp.csr_matrix(mask_vector))
                    archive_scores.append(fitness[i])
                    
            # Check stopping criteria
            if (
                unchanged_generations >= self.max_unchanged_generations or
                (self.fitness_threshold is not None and best_fitness >= self.fitness_threshold)
            ):
                break
                
            # Determine exploration vs offspring counts
            num_explore = int(self.population_size * self.exploration_rate)
            num_offspring = self.population_size - num_explore
            
            # Generate GA-driven offspring
            next_population = [best_individual.copy()]
            while len(next_population) < num_offspring:
                parent1, parent2 = self.select_parents(population, fitness)
                child1, child2 = self.crossover(parent1, parent2)
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                child1 = self.one_shot_repair(child1)
                child2 = self.one_shot_repair(child2)
                next_population.extend([child1, child2])
            next_population = next_population[:num_offspring]
            
            # Generate exploration individuals
            exploration_population = self.initialize_population(
                candidates,
                num_candidates,
                self.required_genomes
            )[:num_explore]
            
            # Combine offspring and exploration for next generation
            population = np.vstack((next_population, exploration_population))
            
        # Final evaluation
        final_fitness = self.evaluate_population_fitness(
            target_clr,
            candidates,
            population
        )
        best_final_index = np.argmax(final_fitness)
        best_final = population[best_final_index]
        
        # Build solution vector
        X_opt = np.zeros(candidates.shape[1], dtype=int)
        X_opt[best_final] = 1
        
        # Compute abundance multipliers
        if self.is_fully_fixed:
            p_opt = X_opt.astype(float) * self.f_const
        else:
            selected_genomes = best_final
            G_selected = self.G[:, selected_genomes]
            bounds = [self.get_candidate_bounds(idx) for idx in selected_genomes]
            initial_guess = np.array([(b[0] + b[1]) / 2 for b in bounds])
            equality_constraint = ({
                'type': 'eq',
                'fun': lambda f: np.sum(f) - self.p_total
            },)
            result = minimize(
                lambda f: np.sum((
                    clr_transform(G_selected @ f, self.epsilon) - target_clr
                )**2),
                initial_guess,
                method='SLSQP',
                bounds=bounds,
                constraints=equality_constraint
            )
            optimal_f = result.x if result.success else initial_guess
            p_opt = np.zeros(candidates.shape[1], dtype=float)
            p_opt[selected_genomes] = optimal_f
        
        analysis_metrics = self.analyze_solution(target_clr, p_opt, X_opt)
        
        elapsed_time = time.time() - start_time
        
        # Create a Solution object with pick scores and analysis details.
        sol = Solution(method=self.descriptive_name,
                       X_opt=X_opt,
                       objective=float(best_fitness),
                       genome_names=self.genome_names,
                       selection_order=None,
                       details={"scenario":scenario_id,
                                "runtime":elapsed_time,
                                'best_indices': best_final.tolist(),
                                'generations_run': generation + 1,
                                'archive_solutions': sp.vstack(archive_solutions) if archive_solutions else None,
                                'archive_scores': np.array(archive_scores) if archive_scores else None,
                                "analysis": analysis_metrics,
                                "p_opt": p_opt}
                        
                       )
        
        return sol
        
    def optimize(self) -> Union[Solution, List[Solution]]:
        """
        Execute the GA across all target scenarios in parallel.
        
        Returns
        -------
        Solution or List[Solution]
            Solution objects containing final selections and metrics.
        """           
            
        num_features, num_scenarios = self.M.shape
        tasks = [(self, self.M[:, i], i) for i in range(num_scenarios)]
        with Pool(processes=self.processes) as pool: #initializer=set_single_thread
            # results = pool.map(run_aga_wrapper, tasks)
            solutions = pool.map(run_aga_wrapper, tasks)
            
        for i, sol in enumerate(solutions):
            logger.info(
                f"Target {i}: GA completed after {sol.details['generations_run']} generations with best fitness {sol.objective:.4f}"
            )
        result: Union[Solution, List[Solution]] = solutions[0] if num_scenarios == 1 else solutions
        self.result = result
        return result
