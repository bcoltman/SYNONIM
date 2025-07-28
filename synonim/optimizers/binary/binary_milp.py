import time
import numpy as np
import gurobipy as gp
from gurobipy import GRB
import scipy.sparse as sp

from typing import Any, Dict, List, Optional, Tuple, Union

from synonim.core import Profile, Model
from synonim.optimizers import BinaryOptimizer, Solution
from synonim.optimizers.utils import generate_unified_taxonomy_matrix

import logging

logger = logging.getLogger(__name__)

gurobi_logger = logging.getLogger('gurobipy')
gurobi_logger.setLevel(logging.INFO)#WARNING)

class BinaryMILP(BinaryOptimizer):
    """
    MILP optimizer with support for multi-level taxonomy constraints.
    
    For each feature i, given an observed indicator m_i and weight w_i, the effective
    contribution is computed as:
    
         c_i = 2 * w_i * m_i - (absence_cover_penalty + absence_match_reward) * (1 - m_i)
         
    The per-scenario objective is to maximize:
    
         ∑ c_i * B_i
         
    where B_i is a binary indicator showing whether feature i is covered. When taxonomy
    constraints (and genome_labels) are provided, a unified taxonomy matrix is generated and
    additional constraints are added per taxonomic level.
    """
    
    def __init__(
        self,
        model: Model,
        consortia_size: int,
        processes: int = 1,
        required_genomes: Optional[List[Union[str, Profile]]] = None,
        absence_cover_penalty: float = 1,
        absence_match_reward: float = 0,
        taxonomy_constraints: Optional[Dict[str, Any]] = None,
        taxonomic_levels: Optional[List[str]] = None,
        time_limit: float = 3600,
        weighted: bool = False
    ) -> None:
        """
        Initialize the MILP optimizer.
        
        Parameters
        ----------
        model : Model
            The model object supplying profiles and matrices.
        consortia_size : int
            The number of candidate profiles to select.
        processes : int, optional
            Number of parallel threads for the MILP solver.
        required_genomes : list of (str or Profile), optional
            Candidate profiles (or their IDs) that must be selected.
        absence_cover_penalty : float, optional
            Penalty multiplier for covering absent features.
        absence_match_reward : float, optional
            Reward multiplier for matching absent features.
        taxonomy_constraints : dict, optional
            Mapping from taxonomic level to per-taxon constraints.
        taxonomic_levels : list of str, optional
            List of taxonomic levels to enforce. If None and taxonomy_constraints is given,
            defaults to the keys of taxonomy_constraints.
        time_limit : float, optional
            Time limit (in seconds) for the solver.
        weighted : bool, optional
            Whether to weight function selection by the abundance of functions in the target metagenome
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
        
        self.time_limit = time_limit
        self.processes = processes
        
        if self.taxonomy_constraints is not None and self.genome_labels is not None:
            self.T_unified, self.unified_taxon_to_index = generate_unified_taxonomy_matrix(
                self.genome_labels, taxonomic_levels
            )
        else:
            self.T_unified = None
            self.unified_taxon_to_index = None
        
        # Build the base MILP model with common constraints and decision variables.
        self.model_gp, self.variables = self._initialize_base_model()
        
        # Initialize scenario-specific objectives.
        self._initialize_scenarios()
        
    @property
    def descriptive_name(self) -> str:
        """
        Build and return a descriptive name for the MILP optimizer.
        
        Returns
        -------
        str
            A descriptive name constructed from key parameter flags.
        """
        parts = ["MILPOptimizer"]
        if hasattr(self, "weights") and self.weights is not None:
            parts.append("Weighted")
        parts.append(f"ACP-{self.absence_cover_penalty}")
        parts.append(f"AMR-{self.absence_match_reward}")
        return "_".join(parts)
        
    def _initialize_base_model(self) -> Tuple[gp.Model, Dict[str, Any]]:
        """
        Build the base MILP model (common to all scenarios) and add constraints in bulk.
        
        Returns
        -------
        tuple
            (model_gp, variables), where model_gp is a Gurobi model and variables is a dict
            of decision variables.
        """
        start_time = time.time()
        # Get dimensions from the target matrix M: d features and s scenarios.
        d, s = self.M.shape
        k = self.G.shape[1]
        self.num_scenarios = s
        logger.info(f"Setting up multi-scenario MILP model with {s} scenario(s).")
        
        model_gp = gp.Model("MILP")
        # Set Gurobi solver parameters.
        model_gp.Params.LogToConsole = 0
        model_gp.Params.LogFile = ''  
        model_gp.Params.Threads = self.processes
        model_gp.Params.Seed = 42
        model_gp.Params.Presolve = 0
        model_gp.Params.PrePasses = 0
        model_gp.Params.TimeLimit = self.time_limit
        model_gp.Params.MIPFocus = 3 #1
        model_gp.Params.Cuts = 0
        model_gp.Params.CutPasses = 3
        model_gp.Params.Heuristics = 0.05
        model_gp.Params.NoRelHeurWork = self.time_limit * 0.01
        model_gp.Params.Method = 4
        model_gp.Params.DisplayInterval = 900
        model_gp.Params.NodeMethod = 1
        model_gp.Params.PreSparsify = 1
        model_gp.Params.Symmetry = 2
        model_gp.Params.PoolSolutions = 10000
        
        # Decision variables:
        #   x: binary selection vector for candidates (length k)
        #   B: binary indicators for gene coverage (length d)
        x = model_gp.addMVar(shape=k, vtype=GRB.BINARY, name="x")
        B = model_gp.addMVar(shape=d, vtype=GRB.BINARY, name="B")
        
        # Add constraints for required candidates.
        if self.required_genomes:
            for idx in self.required_genomes:
                model_gp.addConstr(x[idx] == 1, name=f"required_genome_{idx}")
                
        # --- 1. Coverage Constraints ---
        # For each gene i, compute the set of candidate profiles that cover it.
        coverage_indices: Dict[int, List[int]] = {
            i: np.where(self.G[i, :] == 1)[0].tolist() for i in range(d)
        }
        # Concatenate B and x so that matrix constraints can reference both.
        v_Bx = gp.concatenate((B, x))
        
        # Build sparse matrices for the "upper" and "lower" coverage constraints.
        rows_up, cols_up, data_up = [], [], []
        rows_low, cols_low, data_low = [], [], []
        row_low_counter = 0
        
        # Upper bound: B[i] - sum_{j in coverage(i)} x[j] <= 0.
        # Lower bound: For each candidate j that covers gene i: -B[i] + x[j] <= 0.
        for i in range(d):
            # Add coefficient for B[i].
            rows_up.append(i)
            cols_up.append(i)
            data_up.append(1.0)
            # For each candidate j that covers gene i.
            for j in coverage_indices[i]:
                # Upper constraint: subtract x[j]
                rows_up.append(i)
                cols_up.append(d + j)
                data_up.append(-1.0)
                # Lower constraint: enforce -B[i] + x[j] <= 0.
                rows_low.append(row_low_counter)
                cols_low.append(i)
                data_low.append(-1.0)
                rows_low.append(row_low_counter)
                cols_low.append(d + j)
                data_low.append(1.0)
                row_low_counter += 1
                
        A_up = sp.coo_matrix((data_up, (rows_up, cols_up)), shape=(d, d + k))
        b_up = np.zeros(d)
        A_low = sp.coo_matrix((data_low, (rows_low, cols_low)), shape=(row_low_counter, d + k))
        b_low = np.zeros(row_low_counter)
        
        model_gp.addMConstr(A_up, v_Bx, '<=', b_up, name="coverage_up")
        model_gp.addMConstr(A_low, v_Bx, '<=', b_low, name="coverage_down")
        
        # --- 2. Consortia-size Constraint ---
        model_gp.addConstr(gp.quicksum(x) == self.consortia_size, name="consortia_size_constraint")
        
        # --- 3. Taxonomy Constraints ---
        self._add_taxonomy_constraints(model_gp, x)
        
        # Set a dummy objective (scenario-specific objectives are set later).
        model_gp.setObjective(0.0, GRB.MAXIMIZE)
        model_gp.update()
        
        variables = {"x": x, "B": B, "coverage_indices": coverage_indices}
        elapsed_time = time.time() - start_time
        logger.info(f"Base model initialization took {elapsed_time:.2f} seconds.")
        return model_gp, variables
        
    def _add_taxonomy_constraints(self, model: gp.Model, x_vars: gp.MVar) -> None:
        """
        Add taxonomy constraints to the model using a unified taxonomy matrix and matrix constraints.
        
        Each constraint controls the number of selected genomes from a given taxon or taxonomic level.
        """
        if self.taxonomy_constraints is None or self.genome_labels is None:
            logger.info("No taxonomy constraints: either taxonomy_constraints or genome_labels is None.")
            return
        
        k = x_vars.shape[0]
        # Loop over each taxonomic level and its constraints.
        for level, constraints in self.taxonomy_constraints.items():
            explicit_taxa = [taxon for taxon in constraints if taxon != "default"]
            # Process explicit taxa.
            for taxon in explicit_taxa:
                key = f"{level}-{taxon}"
                if key in self.unified_taxon_to_index:
                    row_idx = self.unified_taxon_to_index[key]
                    row = self.T_unified.getrow(row_idx)
                    indices = row.indices.tolist()
                    if indices:
                        # Build a row-vector constraint using the sparse matrix formulation.
                        A_lab = sp.coo_matrix(([1.0] * len(indices), ([0] * len(indices), indices)), shape=(1, k))
                        if "exact" in constraints[taxon]:
                            b = np.array([constraints[taxon]["exact"]])
                            model.addMConstr(A_lab, x_vars, '=', b, name=f"tax_{key}_exact")
                        if "max" in constraints[taxon]:
                            b = np.array([constraints[taxon]["max"]])
                            model.addMConstr(A_lab, x_vars, '<=', b, name=f"tax_{key}_max")
                        if "min" in constraints[taxon]:
                            b = np.array([constraints[taxon]["min"]])
                            model.addMConstr(A_lab, x_vars, '>=', b, name=f"tax_{key}_min")
                else:
                    logger.warning(f"Taxon key '{key}' not found in unified taxonomy matrix.")
            # Process default constraints for unspecified taxa.
            if "default" in constraints:
                default_cons = constraints["default"]
                for key, row_idx in self.unified_taxon_to_index.items():
                    if not key.startswith(f"{level}-"):
                        continue
                    taxon = key.split("-", 1)[1]
                    if taxon in explicit_taxa:
                        continue
                    row = self.T_unified.getrow(row_idx)
                    indices = row.indices.tolist()
                    if indices:
                        A_lab = sp.coo_matrix(([1.0] * len(indices), ([0] * len(indices), indices)), shape=(1, k))
                        if "max" in default_cons:
                            b = np.array([default_cons["max"]])
                            model.addMConstr(A_lab, x_vars, '<=', b, name=f"tax_{key}_default_max")
                        if "min" in default_cons:
                            b = np.array([default_cons["min"]])
                            model.addMConstr(A_lab, x_vars, '>=', b, name=f"tax_{key}_default_min")
                            
    def _initialize_scenarios(self) -> None:
        """
        Initialize scenario-specific modifications.
        
        For each scenario (each column in the target matrix M), set the scenario objective
        for gene coverage. For scenario j, the objective coefficients are computed as:
            
            c = 2 * w_obs * m_obs - (absence_cover_penalty + absence_match_reward) * (1 - m_obs)
            
        and assigned to B.ScenNObj.
        """
        start_time = time.time()
        d, s = self.M.shape
        model_gp = self.model_gp
        B = self.variables["B"]
        
        model_gp.NumScenarios = self.num_scenarios
        # Use each column index as the scenario index.
        for scenario in range(s):
            model_gp.Params.ScenarioNumber = scenario
            model_gp.ScenNName = f"Scenario_{scenario}"
            m_obs = self.M[:, scenario]
            w_obs = np.ones(d) if self.weights is None else self.weights[:, scenario]
            c = 2 * w_obs * m_obs - (self.absence_cover_penalty + self.absence_match_reward) * (1 - m_obs)
            # Set scenario-specific objective coefficients for gene coverage.
            B.ScenNObj = np.array(c)
        model_gp.ModelSense = GRB.MAXIMIZE
        model_gp.update()
        elapsed_time = time.time() - start_time
        logger.info(f"Scenario initialization took {elapsed_time:.2f} seconds.")
        
    def warmup(self, solution: np.ndarray) -> None:
        """
        Set a warm start solution for the MILP model.
        
        Parameters
        ----------
        solution : np.ndarray
            A binary vector (of length equal to the number of candidate profiles) used as a warm start.
        """
        self.model_gp.Params.StartNumber = -1
        x = self.variables["x"]
        if solution.shape[0] != x.shape[0]:
            raise ValueError("Initial solution length does not match number of candidate variables.")
        if not np.all(np.isin(solution, [0, 1])):
            raise TypeError("Warm start solution must be binary.")
        # Set the warm start solution for the candidate selection variable.
        x.Start = solution.flatten().tolist()
        self.model_gp.update()
        logger.info("Warm start solution has been set.")
        
    def optimize(self) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Solve the MILP model for all scenarios.
        
        For each scenario (each column in the target matrix), after solving the MILP,
        the decision variable values are retrieved and a Solution object is created.
        The unified analysis method is called for each scenario to generate evaluation
        metrics, which are then included in the solution details.
        
        Returns
        -------
        dict or list of dict
            A Solution object if one scenario exists; otherwise, a list of Solution objects for each scenario.
        """
        start_time = time.time()
        self.model_gp.optimize()
        if self.model_gp.Status not in [GRB.OPTIMAL, GRB.WORK_LIMIT, GRB.TIME_LIMIT]:
            raise RuntimeError(f"MILP optimization failed with status {self.model_gp.Status}")
            
        x_var = self.variables["x"]
        d, s = self.M.shape
        solution_list = []
        
        for scenario in range(s):
            self.model_gp.Params.ScenarioNumber = scenario
            obj_val = self.model_gp.ScenNObjVal
            
            sol_count = self.model_gp.SolCount  # Total number of pool solutions
            
            if obj_val == -np.inf:
                logger.warning(f"No objective value for scenario: {scenario}")
                # If the scenario yields no feasible solution, return a zero-vector.
                x_sol = np.zeros(self.G.shape[1])
            else:
                x_sol = x_var.ScenNX  # Retrieve the binary decision vector for the current scenario.
            
            analysis_metrics = self.analyze_solution(self.M[:, scenario], x_sol)
            elapsed_time = time.time() - start_time

            sol = Solution(method=self.descriptive_name,
                           X_opt=x_sol,
                           objective=obj_val,
                           genome_names=self.genome_names,
                           selection_order=None,
                           details={"scenario": scenario,
                                    "runtime":elapsed_time,
                                    "analysis": analysis_metrics}
                          )
            
            if sol_count != 0:
                archive_rows = []
                archive_scores = []
                for i in range(1, sol_count):
                    self.model_gp.Params.SolutionNumber = i
                    xn_sol = x_var.Xn  # Pool solution
                    obj_val_i = self.model_gp.PoolObjVal
                    
                    archive_rows.append(sp.csr_matrix(xn_sol))
                    archive_scores.append(obj_val_i)
                    
                # Archive matrix and score array
                archive_solutions = sp.vstack(archive_rows) if archive_rows else None
                archive_scores_arr = np.array(archive_scores) if archive_scores else None
                sol.details["archive_solutions"] = archive_solutions
                sol.details["archive_scores"] = archive_scores_arr
            
            solution_list.append(sol)
        
        result = solution_list[0] if s == 1 else solution_list
        self.result = result
        return result
        
    def __getstate__(self) -> dict:
        # copy everything except the live Gurobi stuff
        state = self.__dict__.copy()
        # these hold PyCapsule / C‑objects
        state.pop('model_gp', None)
        state.pop('variables', None)
        return state
        
    def __setstate__(self, state: dict) -> None:
        # restore all the simple attributes
        self.__dict__.update(state)
        # rebuild the solver from the stored parameters
        # (this re‑runs _initialize_base_model and _initialize_scenarios)
        #
        print("Gurobi model is not-pickleable and will need to either be re-initialised or loaded from other files")
        # self.model_gp, self.variables = self._initialize_base_model()
        # self._initialize_scenarios()
