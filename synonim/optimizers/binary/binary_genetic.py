import numpy as np
import random
import logging
import scipy.sparse as sp
from multiprocessing import Pool
from typing import Any, Dict, List, Optional, Tuple, Union

import time

from synonim.optimizers import BinaryOptimizer, Solution
from synonim.optimizers.utils import generate_unified_taxonomy_matrix
from synonim.core import Profile, Model

from threadpoolctl import threadpool_limits

logger = logging.getLogger(__name__)


def run_ga_wrapper(args: Tuple[Any, np.ndarray, np.ndarray]) -> Dict[str, Any]:
    """
    Wrapper function for multiprocessing that calls the genetic_algorithm method.

    Parameters
    ----------
    args : tuple
        A tuple (optimizer, target, weights).

    Returns
    -------
    dict
        The dictionary returned by optimizer.genetic_algorithm.
    """
    optimizer, target, weights, scenario_idx = args
    with threadpool_limits(limits=1):
        return optimizer.genetic_algorithm(target, weights, optimizer.G, optimizer.consortia_size, scenario_idx)


class BinaryGenetic(BinaryOptimizer):
    """
    Genetic Algorithm optimizer for multi-level candidate selection with taxonomic constraints.

    This optimizer performs a genetic algorithm (GA) search to select a consortia of candidate
    profiles that best meet a target metagenome profile. It uses a unified taxonomy matrix to
    quickly impose multi-level constraints. The GA employs standard genetic operators such as
    tournament selection, order crossover, mutation, repair, and elitism. It can run in parallel
    (one GA run per target sample).

    Attributes
    ----------
    model : Model
        The model instance supplying the necessary matrices and profile information.
    consortia_size : int
        The desired number of candidate profiles to select.
    taxonomy_constraints : dict, optional
        Mapping taxonomic level to per-taxon constraints. For example:
            {
                "domain": {
                    "bacteria": {"exact": 3},
                    "fungi": {"exact": 4},
                    "archaea": {"max": 3}
                },
                "family": {
                    "default": {"max": 3}
                }
            }
    taxonomic_levels : list, optional
        List of taxonomic levels to enforce. If not provided, the keys of taxonomy_constraints are used.
    required_genomes : list, optional
        List of candidate profiles (or their IDs) that must be included in the solution.
    absence_cover_penalty : float, optional
        Penalty multiplier applied when a candidate covers absent features.
    absence_match_reward : float, optional
        Reward multiplier applied when candidate features match absent features.
    population_size : int
        Number of individuals in the GA population.
    generations : int
        Maximum number of generations to run.
    mutation_rate : float
        Probability of mutating an individual.
    fitness_threshold : float, optional
        Early-stop threshold on fitness (if reached, GA terminates).
    max_unchanged_generations : int
        Maximum number of generations with no fitness improvement before termination.
    penalty_factor : float
        Weight factor used when penalizing constraint violations.
    processes : int
        Number of parallel processes for GA runs.
    tournament_size : int
        Number of individuals in each tournament for parent selection.
    seed : int
        Random seed for reproducibility.
    unified_taxon_to_index : dict, optional
        Mapping from unified taxon keys to indices in the unified taxonomy matrix.
    T_unified : scipy.sparse.csr_matrix, optional
        The unified taxonomy matrix.
    weighted : bool, optional
        Whether to weight function selection by the abundance of functions in the target metagenome
    """
    
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
        penalty_factor: float = 10.0,
        processes: int = 1,
        absence_cover_penalty: float = 1,
        absence_match_reward: float = 0,
        tournament_size: int = 10,
        seed: int = 42,
        weighted: bool = False,
        exploration_rate: float = 0.2
    ) -> None:
        """
        Initialize the BinaryGenetic optimizer.
        
        Parameters
        ----------
        model : Model
            The model instance supplying profiles and matrices.
        consortia_size : int
            Number of candidate profiles to select.
        population_size : int, optional
            Number of individuals in the GA population.
        generations : int, optional
            Maximum number of generations to run.
        mutation_rate : float, optional
            Probability of mutating an individual.
        fitness_threshold : float, optional
            Early-stop threshold on fitness (if reached, GA terminates).
        max_unchanged_generations : int, optional
            Maximum number of generations with no fitness improvement before termination.
        required_genomes : list of (str or Profile), optional
            Candidate profiles (or their IDs) that must be included.
        taxonomy_constraints : dict, optional
            Mapping of taxonomic level to per-taxon constraints.
        taxonomic_levels : list of str, optional
            List of taxonomic levels to enforce (defaults to taxonomy_constraints keys).
        penalty_factor : float, optional
            Weight factor used when penalizing constraint violations.
        weights : np.ndarray, optional
            Weights array for features; if provided, used in descriptive name.
        processes : int, optional
            Number of parallel processes for GA runs.
        absence_cover_penalty : float, optional
            Penalty multiplier for covering absent features.
        absence_match_reward : float, optional
            Reward multiplier for matching absent features.
        tournament_size : int, optional
            Number of individuals in each tournament for parent selection.
        seed : int, optional
            Random seed for reproducibility.
        exploration_rate : float, optional
            Fraction of each generation replaced by fresh random individuals.
        """
        # Initialize the base optimizer.
        super().__init__(model=model, 
                         consortia_size=consortia_size, 
                         weighted=weighted,
                         taxonomy_constraints=taxonomy_constraints,
                         taxonomic_levels=taxonomic_levels,
                         required_genomes=required_genomes,
                         absence_cover_penalty=absence_cover_penalty,
                         absence_match_reward=absence_match_reward)
                         
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.fitness_threshold = fitness_threshold
        self.max_unchanged_generations = max_unchanged_generations
        self.penalty_factor = penalty_factor
        self.tournament_size = tournament_size
        self.exploration_rate = exploration_rate
        
        self.processes = processes
        self.seed = seed
        
        # Set random seeds for reproducibility.
        random.seed(seed)
        np.random.seed(seed)
        
        # We need the total number of genomes. BinaryOptimizer should have created self.G:
        #   self.G is the candidate‐matrix (features × num_genomes).
        # So:
        num_genomes = self.G.shape[1]
        
        if self.taxonomy_constraints is not None and self.genome_labels is not None:
            # Build the unified taxonomy matrix (rows = taxa, cols = genomes).
            self.T_unified, self.unified_taxon_to_index = generate_unified_taxonomy_matrix(
                self.genome_labels, taxonomic_levels
            )
            self.Tcsc = self.T_unified.tocsc()
            num_taxa_levels, _ = self.T_unified.shape
            
            # Precompute dense arrays for fast indexing.
            self.taxonomy_matrix_dense = self.T_unified.T.toarray().astype(int)
            
            # Vector of zeros used when no genome is removed.
            self._zero_removal = np.zeros(num_taxa_levels, dtype=int)
            
            # Precompute mappings: taxon → genomes, genome → taxa.
            self.taxon_to_genomes = [self.T_unified.getrow(i).indices for i in range(num_taxa_levels)]
            self.genome_to_taxa = [self.Tcsc.getcol(j).indices for j in range(num_genomes)]
            
            # Arrays to store min, max, and exact per-taxon requirements.
            self._min_counts = np.zeros(num_taxa_levels, dtype=float)
            self._max_counts = np.full(num_taxa_levels, np.inf, dtype=float)
            self._exact_counts = np.full(num_taxa_levels, np.nan, dtype=float)
            
            # Populate arrays from taxonomy_constraints.
            # For "exact", we set min=max=exact so that all logic treats it as a hard requirement.
            for level, rules in self.taxonomy_constraints.items():
                default_rules = rules.get("default", {})
                # Handle explicitly defined taxa.
                for taxon, cons in rules.items():
                    if taxon == "default":
                        continue
                    key = f"{level}-{taxon}"
                    idx = self.unified_taxon_to_index.get(key)
                    if idx is None:
                        continue
                    if "exact" in cons:
                        val = float(cons["exact"])
                        self._exact_counts[idx] = val
                        self._min_counts[idx] = val
                        self._max_counts[idx] = val
                    else:
                        self._min_counts[idx] = cons.get("min", self._min_counts[idx])
                        self._max_counts[idx] = cons.get("max", self._max_counts[idx])
                # Apply default rules to all other taxa at this level.
                for composite_key, idx in self.unified_taxon_to_index.items():
                    lvl, t = composite_key.split("-", 1)
                    if lvl != level or t in rules:
                        continue
                    if "exact" in default_rules:
                        val = float(default_rules["exact"])
                        self._exact_counts[idx] = val
                        self._min_counts[idx] = val
                        self._max_counts[idx] = val
                    else:
                        if "min" in default_rules:
                            self._min_counts[idx] = default_rules["min"]
                        if "max" in default_rules:
                            self._max_counts[idx] = default_rules["max"]
        
        else:
            # No taxonomy constraints → create “dummy” arrays of length 0 so
            # np.isnan(self._exact_counts) still works when needed.
            
            # T_unified: a (0 taxa × num_genomes) sparse matrix
            self.T_unified = sp.csr_matrix((0, num_genomes))
            self.unified_taxon_to_index = {}
            # self.taxonomy_matrix_dense = self.T_unified.T.toarray()
            self.taxonomy_matrix_dense = self.T_unified.T.toarray().astype(int)
            
            # No taxon→genome or genome→taxon relationships
            self.taxon_to_genomes = []
            self.genome_to_taxa = [[] for _ in range(num_genomes)]
            
            # min_counts, max_counts, exact_counts are all length‐0 float arrays
            self._min_counts = np.zeros(0, dtype=float)
            self._max_counts = np.full(0, np.inf, dtype=float)
            self._exact_counts = np.full(0, np.nan, dtype=float)
            
            # A zero‐vector of length 0 to use when "no genome is removed"
            self._zero_removal = np.zeros(0, dtype=int)
        
        self.validate_constraints(self.consortia_size)
        
    def __getstate__(self) -> dict:
        """
        Customize pickling by removing non-pickleable objects.
        
        Returns
        -------
        dict
            The state dictionary for pickling.
        """
        state = self.__dict__.copy()
        state['model'] = None  # Remove the model to avoid circular references.
        return state
        
    @property
    def descriptive_name(self) -> str:
        """
        Build and return a descriptive name for the optimizer.
        
        Returns
        -------
        str
            A name constructed from key parameter flags.
        """
        parts = ["BinaryGenetic"]
        if self.weights is not None:
            parts.append("Weighted")
        parts.append(f"ACP-{self.absence_cover_penalty}")
        parts.append(f"AMR-{self.absence_match_reward}")
        return "_".join(parts)
    
    def validate_constraints(self, consortia_size: int) -> None:
        """
        Recursively validate hierarchical taxonomy constraints.
        
        Starting from the top level, check that exact/min/max allocations
        can be satisfied within the given consortia size. For sub-levels,
        recurse into each constrained taxon with its allocated share.
        
        Raises
        ------
        ValueError if the constraints are globally infeasible.
        """
        
        def check_level(level: str, size: int) -> None:
            """
            Validate constraints at one level of the hierarchy.
            
            Parameters
            ----------
            level : str
                Taxonomic level name (e.g. "domain", "phylum").
            size : int
                Number of genomes allocated to this level (from parent).
            """
            # Collect indices of all taxa at this level
            taxa = [(key.split("-", 1)[1], idx)
                    for key, idx in self.unified_taxon_to_index.items()
                    if key.startswith(f"{level}-")]
                        
            if not taxa:
                return  # nothing to check
                
            exact_sum = 0
            min_sum = 0
            max_sum = 0
            unconstrained = False
            
            for taxon, idx in taxa:
                ex = self._exact_counts[idx]
                mi = self._min_counts[idx]
                ma = self._max_counts[idx]
                
                if not np.isnan(ex):
                    exact_sum += int(ex)
                else:
                    min_sum += int(mi)
                    if np.isfinite(ma):
                        max_sum += int(ma)
                    else:
                        unconstrained = True
                        
            if exact_sum > size:
                raise ValueError(
                    f"Infeasible {level}-level constraints: exact={exact_sum} > allocated={size}"
                )
                
            lower = exact_sum + min_sum
            if size < lower:
                raise ValueError(
                    f"Infeasible {level}-level constraints: require ≥{lower}, allocated={size}"
                )
                
            if not unconstrained:
                upper = exact_sum + max_sum
                if size > upper:
                    raise ValueError(
                        f"Infeasible {level}-level constraints: allow ≤{upper}, allocated={size}"
                    )
                    
            # --- Recurse into sub-levels ---
            # For each exact taxon, its size is fixed = exact.
            # For others, we can only validate local min/max bounds.
            for taxon, idx in taxa:
                ex = self._exact_counts[idx]
                if not np.isnan(ex):
                    # Recurse with fixed allocation
                    sub_level = self._next_level(level)
                    if sub_level is not None:
                        check_level(sub_level, int(ex))
                        
        # Kick off recursion from the topmost level
        if self.taxonomic_levels:
            check_level(self.taxonomic_levels[0], consortia_size)
            
    def _next_level(self, level: str) -> Optional[str]:
        """
        Return the next taxonomic level after the given one,
        or None if this is the deepest level.
        """
        if self.taxonomic_levels is None:
            return None
        try:
            idx = self.taxonomic_levels.index(level)
            if idx + 1 < len(self.taxonomic_levels):
                return self.taxonomic_levels[idx + 1]
        except ValueError:
            pass
        return None
                    
    def _can_transition_batch(
        self,
        current_counts: np.ndarray,
        candidates_to_add: np.ndarray,
        genome_to_remove: Optional[int],
        slots_left: int
    ) -> np.ndarray:
        """
        Return a boolean array (len = len(candidates_to_add)).
        True at position i if adding candidates_to_add[i] (and optionally
        removing genome_to_remove) keeps the current selection within
        feasible bounds.
        
        Feasibility here means:
          - All taxon counts remain ≤ max (including exact-as-max).
          - There are still enough slots left to meet all min (including exact-as-min).
        """
        # Contribution of genomes we may add.
        adds = self.taxonomy_matrix_dense[candidates_to_add, :]
        
        # Contribution of genome we may remove (or zero-vector if none).
        remove = self._zero_removal if genome_to_remove is None else self.taxonomy_matrix_dense[genome_to_remove, :]
        
        # New counts per taxon if we add/remove these genomes.
        new_counts = current_counts[None, :] + adds - remove[None, :]
        
        # Effective min/max already include "exact" cases.
        eff_min = self._min_counts[None, :]
        eff_max = self._max_counts[None, :]
        
        # Check ≤ max for each candidate.
        within_max = np.all(new_counts <= eff_max, axis=1)
        
        # Check if remaining slots could still fill deficits in min requirements.
        shortfall = np.clip(eff_min - new_counts, 0, None)
        can_fill = shortfall.sum(axis=1) <= slots_left
        
        return within_max & can_fill
        
    def individual_to_binary(self, individual: np.ndarray, total_candidates: int) -> np.ndarray:
        """
        Convert an individual (array of candidate indices) to a binary selection vector.
        
        Parameters
        ----------
        individual : np.ndarray
            1D array of candidate indices.
        total_candidates : int
            Total number of candidate profiles.
            
        Returns
        -------
        np.ndarray
            Binary vector (length total_candidates) with 1s for selected candidates.
        """
        binary = np.zeros(total_candidates, dtype=int)
        binary[individual] = 1
        return binary
    
    def initialize_population(
        self,
        candidates: np.ndarray,         # shape = (num_features, num_genomes)
        consortia_size: int,
        required_genomes: List[int]     # list of genome‐indices that must appear
    ) -> np.ndarray:
        """
        Construct an initial population where every individual exactly
        satisfies all 'exact'/'min'/'max' taxonomy constraints.
        
        Returns an array of shape (population_size, consortia_size),
        where each row is a list of genome‐indices.
        """
        population = np.zeros((self.population_size, consortia_size), dtype=int)
        num_genomes = candidates.shape[1]
        
        # Desired per‐taxon counts: if exact is specified, use that; else use min.
        desired_counts = np.where(
            np.isnan(self._exact_counts),
            self._min_counts,
            self._exact_counts
        ).astype(int)
        
        filled = 0
        max_attempts = self.population_size * 10
        attempts = 0
        
        while filled < self.population_size:
            attempts += 1
            if attempts > max_attempts:
                raise RuntimeError(
                    f"Failed to build {self.population_size} feasible individuals "
                    f"after {max_attempts} tries."
                )
                
            # Start with required_genomes
            individual = list(required_genomes)
            mask = np.zeros(num_genomes, dtype=int)
            mask[individual] = 1
            taxon_counts = self.T_unified.dot(mask)  # current counts per taxon (dense)
            
            failed = False
            
            # Phase 1: Satisfy exact/min constraints for each taxon
            for taxon_idx, needed in enumerate(desired_counts):
                # How many more we need from this taxon?
                to_add = int(needed - taxon_counts[taxon_idx])
                for _ in range(to_add):
                    slots_left = consortia_size - len(individual)
                    # Pool of genomes that belong to this taxon, not already chosen
                    pool = np.setdiff1d(self.taxon_to_genomes[taxon_idx], individual, assume_unique=True)
                    # Filter to those that keep all future constraints
                    feasible = pool[
                        self._can_transition_batch(taxon_counts, pool, None, slots_left)
                    ]
                    if feasible.size == 0:
                        failed = True
                        break
                    pick = int(np.random.choice(feasible))
                    individual.append(pick)
                    # Update counts
                    mask[pick] = 1
                    taxon_counts += self.taxonomy_matrix_dense[pick, :]
                    
                if failed:
                    break
                    
            if failed:
                # Abort and retry a fresh individual
                continue
                
            # Phase 2: Fill any remaining slots with ANY feasible genome
            while len(individual) < consortia_size:
                slots_left = consortia_size - len(individual)
                pool = np.setdiff1d(np.arange(num_genomes), individual, assume_unique=True)
                feasible = pool[
                    self._can_transition_batch(taxon_counts, pool, None, slots_left)
                ]
                if feasible.size == 0:
                    failed = True
                    break
                pick = int(np.random.choice(feasible))
                individual.append(pick)
                mask[pick] = 1
                taxon_counts += self.taxonomy_matrix_dense[pick, :]
                
            if failed or len(individual) != consortia_size:
                # Couldn't finish a feasible individual—retry
                continue
                
            # At this point, 'individual' is exactly consortia_size long and fully feasible.
            population[filled, :] = individual
            filled += 1
            
        return population
        
    def select_parents(self, population: np.ndarray, fitness: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Select two parent individuals via tournament selection.
        
        Parameters
        ----------
        population : np.ndarray
            Current GA population (each row is an individual).
        fitness : np.ndarray
            Fitness scores corresponding to each individual.
            
        Returns
        -------
        tuple of np.ndarray
            Two selected parent individuals.
        """
        pop_size = population.shape[0]
        # Tournament selection: choose a subset and pick the fittest.
        tour = np.random.choice(pop_size, self.tournament_size, replace=False)
        parent1_idx = tour[np.argmax(fitness[tour])]
        tour = np.random.choice(pop_size, self.tournament_size, replace=False)
        parent2_idx = tour[np.argmax(fitness[tour])]
        return population[parent1_idx], population[parent2_idx]
        
    def constrained_order_crossover(
        self,
        parent1: np.ndarray,
        parent2: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform an order‐crossover (OX) that always produces
        offspring satisfying taxonomy constraints.
        """
        
        def build_offspring(p_a, p_b):
            size = len(p_a)
            # Choose two cut points
            i, j = sorted(random.sample(range(size), 2))
            offspring = [-1] * size
            
            # 1) Copy slice p_a[i:j+1] to offspring
            offspring[i : j + 1] = p_a[i : j + 1]
            
            # Build mask & counts for that slice
            mask_array = np.zeros(self.T_unified.shape[1], dtype=int)  # (num_genomes,)
            taxon_counts = np.zeros(self.T_unified.shape[0], dtype=int)  # (#taxa,)
            for genome in offspring[i : j + 1]:
                mask_array[genome] = 1
                taxon_counts += self.taxonomy_matrix_dense[genome]
                
            insert_pos = (j + 1) % size
            
            # 2) Attempt to take genes from p_b (in order) if they stay feasible
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
                    # taxon_counts += self.T_unified[genome, :].toarray().ravel()
                    taxon_counts += self.taxonomy_matrix_dense[genome]
                    insert_pos = (insert_pos + 1) % size
                    
            # 3) Fill any remaining slots with “any feasible genome”
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
                    # In pathological cases, you might fail here—though with well-constructed parents
                    # this shouldn’t happen. For safety, we break and return whatever we have.
                    break
                chosen = int(np.random.choice(feasible))
                offspring[insert_pos] = chosen
                mask_array[chosen] = 1
                taxon_counts += self.taxonomy_matrix_dense[chosen]
                # taxon_counts += self.T_unified[chosen, :].toarray().ravel()
                insert_pos = (insert_pos + 1) % size
                
            return np.array(offspring, dtype=int)
            
        child1 = build_offspring(parent1, parent2)
        child2 = build_offspring(parent2, parent1)
        return child1, child2
    
    def _order_crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform order crossover (OX) on two parent individuals.
        
        Parameters
        ----------
        parent1 : np.ndarray
            First parent's candidate indices.
        parent2 : np.ndarray
            Second parent's candidate indices.
            
        Returns
        -------
        tuple of np.ndarray
            Two offspring individuals produced by crossover.
        """
        size = len(parent1)
        if size < 2:
            return np.copy(parent1), np.copy(parent2)
        # Initialize children with placeholder -1.
        child1 = [-1] * size
        child2 = [-1] * size
        i, j = sorted(random.sample(range(size), 2))
        # Copy subsequence from one parent.
        child1[i:j+1] = parent1[i:j+1]
        child2[i:j+1] = parent2[i:j+1]
        
        def fill_child(child: List[int], parent: np.ndarray) -> List[int]:
            current = (j + 1) % size
            p_index = (j + 1) % size
            while -1 in child:
                if parent[p_index] not in child:
                    child[current] = parent[p_index]
                    current = (current + 1) % size
                p_index = (p_index + 1) % size
            return child
            
        child1 = fill_child(child1, parent2)
        child2 = fill_child(child2, parent1)
        return np.array(child1), np.array(child2)
        
    def crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply the crossover operator to two parents.
        
        Returns
        -------
        tuple of np.ndarray
            Two offspring individuals.
        """
        return self.constrained_order_crossover(parent1, parent2)
        # return self._order_crossover(parent1, parent2)
    
    def mutate(self, individual: np.ndarray, candidates: np.ndarray) -> np.ndarray:
        """
        Swap a single gene—but only to a genome that
        keeps the individual fully feasible.
        """
        mutated = individual.copy()
        num_req = len(self.required_genomes)
        
        # Only attempt mutation with probability self.mutation_rate
        if len(mutated) > num_req and random.random() < self.mutation_rate:
            num_genomes = candidates.shape[1]
            # Build mask & counts for current individual
            mask = np.zeros(num_genomes, dtype=int)
            mask[mutated] = 1
            counts = self.T_unified.dot(mask)
            
            # Pool = all genomes not currently selected
            pool = np.setdiff1d(np.arange(num_genomes), mutated, assume_unique=True)
            # Filter to those that keep feasibility if added
            feasible = pool[
                self._can_transition_batch(counts, pool, None, 0)
            ]
            
            if feasible.size > 0:
                swap_pos = random.randint(num_req, len(mutated) - 1)
                new_genome = int(np.random.choice(feasible))
                mutated[swap_pos] = new_genome
                
        return mutated
        
    def evaluate_taxonomy_penalty(self, individual: np.ndarray) -> float:
        """
        Compute a penalty for an individual based on taxonomy constraints using the unified taxonomy matrix.
        
        Parameters
        ----------
        individual : np.ndarray
            Array of candidate indices.
            
        Returns
        -------
        float
            Computed penalty value.
        """
        penalty = 0.0
        if self.taxonomy_constraints is None or self.T_unified is None:
            return penalty
            
        # Retrieve counts across unified taxonomy rows.
        counts = np.array(self.T_unified[:, individual].sum(axis=1)).ravel()
        # Iterate over each taxonomic level and its constraints.
        for level, constraints in self.taxonomy_constraints.items():
            explicit_keys = set()
            for taxon in constraints:
                if taxon != "default":
                    key = f"{level}-{taxon}"
                    explicit_keys.add(key)
                    if key in self.unified_taxon_to_index:
                        count = counts[self.unified_taxon_to_index[key]]
                        if "exact" in constraints[taxon]:
                            penalty += self.penalty_factor * abs(count - constraints[taxon]["exact"])
                        if "max" in constraints[taxon] and count > constraints[taxon]["max"]:
                            penalty += self.penalty_factor * (count - constraints[taxon]["max"])
                        if "min" in constraints[taxon] and count < constraints[taxon]["min"]:
                            penalty += self.penalty_factor * (constraints[taxon]["min"] - count)
                    else:
                        # If taxon key is not found, add a fixed penalty.
                        if "exact" in constraints[taxon]:
                            penalty += self.penalty_factor * constraints[taxon]["exact"]
                        if "min" in constraints[taxon]:
                            penalty += self.penalty_factor * constraints[taxon]["min"]
            # Evaluate default constraints for taxons not explicitly mentioned.
            if "default" in constraints:
                default_cons = constraints["default"]
                for key, row in self.unified_taxon_to_index.items():
                    if key.startswith(f"{level}-"):
                        taxon = key.split("-", 1)[1]
                        if taxon not in explicit_keys:
                            count = counts[row]
                            if "max" in default_cons and count > default_cons["max"]:
                                penalty += self.penalty_factor * (count - default_cons["max"])
                            if "min" in default_cons and count < default_cons["min"]:
                                penalty += self.penalty_factor * (default_cons["min"] - count)
        return penalty
    
    def repair_individual(self, individual: np.ndarray, candidates: np.ndarray) -> np.ndarray:
        """
        Final check to guarantee feasibility of an individual.
        
        Strategy:
          1) If already feasible (respecting min/max/exact), return unchanged.
          2) Otherwise:
             a) Remove excess genomes from overrepresented taxa.
             b) Add genomes from underrepresented taxa until target counts are met.
             c) Trim back to consortia_size if we overshot.
        """
        num_genomes = candidates.shape[1]
        mask = np.zeros(num_genomes, dtype=int)
        mask[individual] = 1
        counts = self.T_unified.dot(mask)
        
        # Effective bounds, with "exact" treated as both min and max.
        eff_min = self._min_counts.copy()
        eff_max = self._max_counts.copy()
        target = np.where(np.isnan(self._exact_counts), counts, self._exact_counts)
        
        # If already feasible, nothing to do.
        if np.all(counts >= eff_min) and np.all(counts <= eff_max) and np.all(
            np.isnan(self._exact_counts) | (counts == self._exact_counts)
        ):
            return individual
        
        genomes = list(individual)
        
        # --- Step 1: Remove excess genomes (over target or over max). ---
        excess = counts - np.minimum(target, eff_max)
        for tax_idx in np.where(excess > 0)[0]:
            remove_count = int(excess[tax_idx])
            # Candidate positions in the genome list that contribute to this taxon.
            positions = [i for i, g in enumerate(genomes) if tax_idx in self.genome_to_taxa[g]]
            # Remove enough of them to fix the excess.
            for pos in random.sample(positions, min(remove_count, len(positions))):
                old = genomes[pos]
                # Prefer replacements from taxa with deficits.
                deficit = np.maximum(np.maximum(eff_min, target) - counts, 0)
                deficit_taxa = np.where(deficit > 0)[0]
                if deficit_taxa.size == 0:
                    pool = np.setdiff1d(np.arange(num_genomes), genomes, assume_unique=True)
                else:
                    pool = np.setdiff1d(
                        np.concatenate([self.taxon_to_genomes[t] for t in deficit_taxa]),
                        genomes,
                        assume_unique=True
                    )
                if pool.size == 0:
                    continue
                feasible = pool[self._can_transition_batch(counts, pool, old, 0)]
                if feasible.size == 0:
                    continue
                newg = int(np.random.choice(feasible))
                genomes[pos] = newg
                counts += self.taxonomy_matrix_dense[newg] - self.taxonomy_matrix_dense[old]
                
        # --- Step 2: Add genomes to fill deficits. ---
        deficit = np.maximum(np.maximum(eff_min, target) - counts, 0)
        for tax_idx in np.where(deficit > 0)[0]:
            need = int(deficit[tax_idx])
            pool = np.setdiff1d(self.taxon_to_genomes[tax_idx], genomes, assume_unique=True)
            feasible = pool[self._can_transition_batch(counts, pool, None, 0)]
            picks = np.random.choice(feasible, size=min(need, feasible.size), replace=False) if feasible.size > 0 else []
            for g in picks:
                genomes.append(int(g))
                counts += self.taxonomy_matrix_dense[g]
                
        # --- Step 3: Trim back to original size if too many genomes were added. ---
        while len(genomes) > len(individual):
            over = counts - target
            over_taxa = np.where(over > 0)[0]
            # Choose genomes that belong to overrepresented taxa preferentially.
            removable_pos = [i for i, g in enumerate(genomes) if any(t in self.genome_to_taxa[g] for t in over_taxa)]
            if not removable_pos:
                removable_pos = list(range(len(genomes)))
            pos = random.choice(removable_pos)
            g = genomes.pop(pos)
            counts -= self.taxonomy_matrix_dense[g]
            
        return np.array(genomes, dtype=int)
        
    def evaluate_population_fitness(
        self, T: np.ndarray, W: np.ndarray, candidates: np.ndarray, population: np.ndarray
    ) -> np.ndarray:
        """
        Compute fitness for each individual in the population.
        
        Fitness is defined as a base score (from feature matching) minus any penalties from taxonomy constraint violations.
        
        Parameters
        ----------
        T : np.ndarray
            Target binary feature vector.
        W : np.ndarray
            Weight vector for features.
        candidates : np.ndarray
            Candidate matrix.
        population : np.ndarray
            Population array where each row is an individual.
            
        Returns
        -------
        np.ndarray
            Fitness score for each individual.
        """
        # Advanced indexing: shape becomes (d, population_size, consortia_size).
        pop_candidates = candidates[:, population]
        # Combine candidate profiles via bitwise OR along the consortia dimension.
        combined = np.bitwise_or.reduce(pop_candidates, axis=2).T  # shape: (population_size, d)
        # Create a coefficient vector for scoring.
        c = 2 * W * T - (self.absence_cover_penalty + self.absence_match_reward) * (1 - T)
        base_score = combined @ c  # Each individual's base score.
        fitness = np.empty(population.shape[0], dtype=float)
        for i, indiv in enumerate(population):
            penalty = self.evaluate_taxonomy_penalty(indiv)
            fitness[i] = base_score[i] - penalty
        return fitness
        
    def genetic_algorithm(
        self, T: np.ndarray, W: np.ndarray, candidates: np.ndarray, num_candidates: int, scenario_idx: int
    ) -> Dict[str, Any]:
        """
        Run the genetic algorithm for a single target vector.
        
        The GA uses crossover, mutation, repair, and elitism. It archives feasible individuals.
        Termination occurs when either the fitness threshold is reached or a maximum number
        of generations with no improvement is exceeded.
        
        Parameters
        ----------
        T : np.ndarray
            Target binary feature vector.
        W : np.ndarray
            Weight vector for features.
        candidates : np.ndarray
            Candidate matrix.
        num_candidates : int
            Consortia size (number of candidates to select).
        scenario_id : int
            Scenario ID
        
        Returns
        -------
        dict
            GA results including best solution, objective value, and metrics.
        """
        start_time = time.time()
        
        population = self.initialize_population(candidates, num_candidates, self.required_genomes)
        best_individual = population[0].copy()
        best_fitness = -np.inf
        unchanged_generations = 0
        generations_run = 0
        
        total_candidates = candidates.shape[1]
        archive_rows = []
        archive_scores = []
        seen_masks = set()
        
        for generation in range(self.generations):
            generations_run = generation + 1
            fitness = self.evaluate_population_fitness(T, W, candidates, population)
            current_best = np.max(fitness)
            best_idx = np.argmax(fitness)
            best_individual = population[best_idx]
            
            # Log every 10 generations
            if generation % 10 == 0:
                logging.info(f"Evaluating generation {generation}")
            
            if current_best > best_fitness:
                best_fitness = current_best
                unchanged_generations = 0
            else:
                unchanged_generations += 1
                
            # Archive individuals that satisfy taxonomy constraints (penalty == 0).
            for i, indiv in enumerate(population):
                if self.evaluate_taxonomy_penalty(indiv) == 0:
                    binary = self.individual_to_binary(indiv, total_candidates)
                    mask_key = tuple(binary.tolist())
                    if mask_key not in seen_masks:
                        seen_masks.add(mask_key)
                        archive_rows.append(sp.csr_matrix(binary))
                        archive_scores.append(fitness[i])
                    
            # Termination conditions.
            if unchanged_generations >= self.max_unchanged_generations:
                break
            if self.fitness_threshold is not None and best_fitness >= self.fitness_threshold:
                break
            
            # Determine exploration vs offspring counts
            num_explore = int(self.population_size * self.exploration_rate)
            num_offspring = self.population_size - num_explore
            
            # Create new population using elitism and genetic operators.
            new_population = [best_individual.copy()]
            while len(new_population) < self.population_size:
                parent1, parent2 = self.select_parents(population, fitness)
                child1, child2 = self.crossover(parent1, parent2)
                child1 = self.mutate(child1, candidates)
                child2 = self.mutate(child2, candidates)
                child1 = self.repair_individual(child1, candidates)
                child2 = self.repair_individual(child2, candidates)
                new_population.extend([child1, child2])
            new_population = new_population[:num_offspring]
            
            # Generate exploration individuals
            exploration_population = self.initialize_population(
                candidates,
                num_candidates,
                self.required_genomes
            )[:num_explore]
            
            # Combine offspring and exploration for next generation
            population = np.vstack((new_population, exploration_population))
        
        # --- Final evaluation after GA loop ---
        fitness = self.evaluate_population_fitness(T, W, candidates, population)
        
        # Add feasible individuals from the final population into the archive.
        for i, indiv in enumerate(population):
            if self.evaluate_taxonomy_penalty(indiv) == 0:
                binary = self.individual_to_binary(indiv, total_candidates)
                mask_key = tuple(binary.tolist())
                if mask_key not in seen_masks:
                    seen_masks.add(mask_key)
                    archive_rows.append(sp.csr_matrix(binary))
                    archive_scores.append(fitness[i])
        
        # Convert archive lists to arrays/matrices if not empty.
        archive_solutions = sp.vstack(archive_rows) if archive_rows else None
        archive_scores_arr = np.array(archive_scores) if archive_scores else None
        
        # Prefer the best feasible archived solution (archive now includes final population).
        if archive_solutions is not None and archive_solutions.shape[0] > 0:
            best_arch_idx = int(np.argmax(archive_scores_arr))
            X_opt = archive_solutions[best_arch_idx].toarray().ravel().astype(int)
            best_fitness = float(archive_scores_arr[best_arch_idx])
            best_individual = np.where(X_opt == 1)[0]
        else:
            # Fallback: no feasible individuals found at all (archive empty).
            best_idx = np.argmax(fitness)
            best_individual = population[best_idx]
            X_opt = np.zeros(total_candidates, dtype=int)
            X_opt[best_individual] = 1
        
        analysis_metrics = self.analyze_solution(T, X_opt)
        # details["analysis"] = analysis_metrics
        
        elapsed_time = time.time() - start_time
        
        # Create a Solution object with pick scores and analysis details.
        sol = Solution(method=self.descriptive_name,
                       X_opt=X_opt,
                       objective=float(best_fitness),
                       genome_names=self.genome_names,
                       selection_order=None,
                       details={"scenario":scenario_idx,
                                "runtime":elapsed_time,
                                'best_indices': best_individual.tolist(),
                                'generations_run': generations_run,
                                'archive_solutions': archive_solutions,
                                'archive_scores': archive_scores_arr,
                                "analysis": analysis_metrics,
                                }
                       )
        
        return sol
        
    def optimize(self) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Execute the genetic algorithm for each target sample in parallel.
        
        For each target column in the metagenome binary matrix, the GA is run (in parallel)
        to select candidate profiles. The results are converted into Solution objects (without a selection order,
        as GA does not enforce order).
        
        Returns
        -------
        Solution or list of Solution
            A Solution object for a single sample or a list of Solutions for multiple samples.
        """
        d, s = self.M.shape
        if self.weights is not None and self.weights.shape == (d, s):
            W = self.weights
        elif self.weights is not None:
            W = np.tile(self.weights[:, 0], (s, 1)).T
        else:
            W = np.ones((d, s))
        
        # Build argument list for parallel execution.
        args = [(self, self.M[:, i], W[:, i], i) for i in range(s)]
        with Pool(processes=self.processes) as pool:
            solutions = pool.map(run_ga_wrapper, args)
        
        for i, sol in enumerate(solutions):
            logger.info(
                f"Target {i}: GA completed after {sol.details['generations_run']} generations with best fitness {sol.objective:.4f}"
            )
        result: Union[Solution, List[Solution]] = solutions[0] if s == 1 else solutions
        self.result = result
        return result
