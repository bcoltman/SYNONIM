import time
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from typing import Any, Dict, List, Optional, Tuple, Union
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

import scipy.sparse as sp

from synonim.core import Profile, Model
from synonim.optimizers import AbundanceOptimizer, Solution
from synonim.optimizers.utils import generate_unified_taxonomy_matrix

import tempfile
import threading
import time
import os

logger = logging.getLogger(__name__)

def _gurobi_log_callback(scenario_idx: int, model: gp.Model, where: int):
    if where == GRB.Callback.MESSAGE:
        # This is the raw text of the log line
        text = model.cbGet(GRB.Callback.MSG_STRING).rstrip()
        logger.info(f"[Scenario {scenario_idx}] {text}")

def run_minlp_scenario_thread(scenario_idx: int, base_optimizer: "AbundanceMINLP") -> Solution:
    """
    Thread-compatible function to solve a single scenario using a copied model,
    with live logging of Gurobi output.
    
    Parameters
    ----------
    scenario_idx : int
        The scenario index.
    base_optimizer : AbundanceMINLP
        The pre-built optimizer instance with the base model.
        
    Returns
    -------
    Solution
        A Solution object for the scenario.
    """
    # self.start_time = time.time()
    # 1) Copy & apply the per-scenario objective
    model_copy, vars_copy = base_optimizer.copy_with_scenario(scenario_idx)

    # 2) Suppress ALL direct console output
    model_copy.setParam("LogToConsole", 0)
    model_copy.setParam("OutputFlag", 0)
    
        # 3) Attach our callback, binding the scenario index
    from functools import partial
    cb = partial(_gurobi_log_callback, scenario_idx)
    
    # 4) Solve (callback will be called for every log line)
    model_copy.optimize(cb)
    
    
    if model_copy.Status not in [GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.WORK_LIMIT]:
        raise RuntimeError(f"Scenario {scenario_idx} failed with status {model_copy.Status}")
    
    logger.info(f"Scenario {scenario_idx} optimisation finished, now extracting results")
    
    p_vars = vars_copy.get("p")
    x_vars = vars_copy.get("x")
    
    # Extract results
    x_sol = np.array([v.X for v in x_vars], dtype=int)
    
    if p_vars is not None and all(v is not None for v in p_vars):
        p_opt = np.array([v.X for v in p_vars])
    else:
        f_const = base_optimizer.f_const
        p_opt = f_const * x_sol
    selection_order = [
        base_optimizer.model.genome_names[j]
        for j in range(len(x_sol)) if x_sol[j] == 1
    ]
    
    analysis_metrics = base_optimizer.analyze_solution(base_optimizer.M[:, scenario_idx], p_opt, x_sol)
    
    elapsed_time = time.time() - base_optimizer.start_time
    
    sol = Solution(method=base_optimizer.descriptive_name,
                   X_opt=x_sol,
                   objective=model_copy.ObjVal,
                   genome_names=base_optimizer.genome_names,
                   selection_order=selection_order,
                   details={"p_opt": p_opt,
                            "scenario": scenario_idx,
                            "runtime": elapsed_time,
                            "analysis": analysis_metrics}
                  )
    
    sol_count = model_copy.SolCount
    
    if sol_count != 0:
        archive_rows = []
        archive_scores = []
        for i in range(1, sol_count):
            model_copy.Params.SolutionNumber = i
            xn_sol = np.array([v.Xn for v in x_vars], dtype=int) # Pool solution
            obj_val_i = model_copy.PoolObjVal
            
            archive_rows.append(sp.csr_matrix(xn_sol))
            archive_scores.append(obj_val_i)
            
        # Archive matrix and score array
        archive_solutions = sp.vstack(archive_rows) if archive_rows else None
        archive_scores_arr = np.array(archive_scores) if archive_scores else None
        sol.details["archive_solutions"] = archive_solutions
        sol.details["archive_scores"] = archive_scores_arr
    
    
    return sol

class AbundanceMINLP(AbundanceOptimizer):
    """
    Mixed-Integer Nonlinear Program (MINLP) optimizer for abundance data using CLR transformation.
    
    This optimizer selects a fixed-size consortia of candidates and assigns abundance contributions
    to minimize the squared Euclidean distance between a CLR-transformed prediction and a given
    target profile. Each scenario is solved independently using parallel processing with threads.
    """
    def __init__(
        self,
        model: Model,
        consortia_size: int,
        f_bounds: Union[Tuple[float, float], List[Tuple[float, float]]] = (0.1, 1.0),
        P_max: float = 1,
        epsilon: float = 0.1, #1e-6,
        taxonomy_constraints: Optional[Dict[str, Any]] = None,
        taxonomic_levels: Optional[List[str]] = None,
        time_limit: float = 3600,
        processes: int = 4,
        required_genomes: Optional[List[Union[str, Profile]]] = None,
    ):
        super().__init__(model=model, 
                         consortia_size=consortia_size, 
                         taxonomy_constraints=taxonomy_constraints,
                         taxonomic_levels=taxonomic_levels,
                         required_genomes=required_genomes)
        
        self.f_bounds = f_bounds
        self.P_max = P_max
        self.epsilon = epsilon
        
        self.genome_names = self.model.genome_names
        self.time_limit = time_limit
        self.processes = processes
        self.model_gp: Optional[gp.Model] = None
        self.variables: Dict[str, Any] = {}
        self._initialize_base_model()
        
    def _initialize_base_model(self):
        """
        Constructs the shared base MINLP model structure (variables and constraints),
        with an optional branch for fully fixed f via x only.
        """
        
        # ------------------------------------------------------------------
        # Build parameter vectors
        # ------------------------------------------------------------------
    
        d, _ = self.M.shape
        k = self.G.shape[1]
        
        # Build f_min and f_max
        if isinstance(self.f_bounds, tuple):
            f_min = np.full(k, self.f_bounds[0])
            f_max = np.full(k, self.f_bounds[1])
        else:
            f_min = np.array([b[0] for b in self.f_bounds])
            f_max = np.array([b[1] for b in self.f_bounds])
            
        # Detect fully fixed-f case
        is_fully_fixed = bool(np.all(f_min == f_max))
        
        f_const = f_min if is_fully_fixed else None
        self.f_const = f_const  # Save fixed abundances, or None
        
        # ------------------------------------------------------------------
        # Create model & global parameters
        # ------------------------------------------------------------------
    
        # Initialize model
        model = gp.Model("AbundanceMINLP_Base")
        model.Params.Seed = 42
        
        model.Params.NonConvex = 2
        model.Params.TimeLimit = self.time_limit
        model.Params.Threads = 2
        model.Params.Presolve = 2
        
        model.Params.MIPFocus = 1
        model.Params.Heuristics = 0.05
        model.Params.NoRelHeurWork = self.time_limit * 0.01
        model.Params.PoolSolutions = 10000
        model.Params.DisplayInterval = 900
        
        model.Params.ScaleFlag = 2   # aggressive automatic scaling
        model.Params.NumericFocus = 1
        model.Params.Method = 1    # barrier at root
        model.Params.Crossover = 1    # quick basis
        model.Params.DegenMoves   = 0      # cut degenerate pivots
        model.Params.PerturbValue = 1e-4 
        
        # ------------------------------------------------------------------
        # Binary selection
        # ------------------------------------------------------------------
        
        x = model.addMVar(k, vtype=GRB.BINARY, name="x")
        # Required genomes
        if self.required_genomes:
            for idx in self.required_genomes:
                model.addConstr(x[idx] == 1, name=f"required_genome_{idx}")
        model.addConstr(x.sum() == self.consortia_size, name="consortia_size")
        
        # ------------------------------------------------------------------
        # Continuous part
        # ------------------------------------------------------------------
    
        # Continuous abundance or folded fixed-f
        if is_fully_fixed:
            # ----- compute tight upper bounds on v_i -----
            # for each i, sum the top-consortia_size values of G[i,j]*f_const[j]
            prod = self.G * f_const[np.newaxis, :]
            # partition so that the largest consortia_size are at the end
            idx = np.argpartition(prod, -self.consortia_size, axis=1)[:, -self.consortia_size:]
            v_max = np.sum(np.take_along_axis(prod, idx, axis=1), axis=1)
            
            # No p variables: embed f_const into v = G @ (f_const * x), with a tight ub
            v = model.addMVar(d, lb=0.0, ub=v_max.tolist(), name="v")
            for i in range(d):
                model.addConstr(
                    v[i] == gp.quicksum(self.G[i, j] * f_const[j] * x[j]
                                       for j in range(k)),
                    name=f"v_const_f_{i}"
                )
            
            # Tell Gurobi the true maximum inside the LOG cone
            model.Params.FuncMaxVal = float(v_max.max() * 1.1)
        else:
            # ---- 4b. p variables + v_max with f_max
            p = model.addMVar(k, lb=0.0, ub=f_max, name="p")
            model.addConstr(p <= f_max * x, name="p_ub")
            model.addConstr(p >= f_min * x, name="p_lb")
            model.addConstr(p.sum() <= self.P_max, name="total_abundance")
            
            # v upper bounds
            prod   = self.G * f_max[np.newaxis, :]
            idx    = np.argpartition(prod, -self.consortia_size, axis=1)[:, -self.consortia_size:]
            v_max  = np.sum(np.take_along_axis(prod, idx, axis=1), axis=1)
            
            v = model.addMVar(d, lb=0.0, ub=v_max.tolist(), name="v")
            model.addConstr(v == self.G @ p, name="v_constr")
            
            model.setParam("FuncMaxVal", float(v_max.max() * 1.1))
        
        # # Continuous abundance or folded fixed-f
        # if is_fully_fixed:
            # # No p variables: embed f_const into v = G @ (f_const * x)
            # # v = model.addMVar(d, lb=-GRB.INFINITY, name="v")
            # v = model.addMVar(d, lb=0, name="v")
            # for i in range(d):
                # model.addConstr(
                    # v[i] == gp.quicksum(self.G[i, j] * f_const[j] * x[j]
                                       # for j in range(k)),
                    # name=f"v_const_f_{i}"
                # )
        # else:
            # # Standard p variables
            # p = model.addMVar(k, lb=0.0, ub=f_max, name="p")
            # model.addConstr(p <= f_max * x, name="p_ub")
            # model.addConstr(p >= f_min * x, name="p_lb")
            # model.addConstr(p.sum() <= self.P_max, name="total_abundance")
            # # v = model.addMVar(d, lb=-GRB.INFINITY, name="v")
            # v = model.addMVar(d, lb=0, name="v")
            # model.addConstr(v == self.G @ p, name="v_constr")
            
        # ------------------------------------------------------------------
        # LOG-cone and mean-centering
        # ------------------------------------------------------------------
        
        vshift = model.addMVar(d, lb=self.epsilon, name="vshift")
        model.addConstr(vshift == v + self.epsilon, name="vshift_constr")
        y = model.addMVar(d, lb=-GRB.INFINITY, name="y")
        for i in range(d):
            model.addGenConstrLog(vshift[i], y[i], name=f"log_constr_{i}")
            
        avg_y = model.addVar(lb=-GRB.INFINITY, name="avg_y")
        model.addConstr(avg_y == y.sum() / d, name="avg_y_constr")
        
        # Placeholder objective
        model.setObjective(0.0, GRB.MINIMIZE)
        # model.update()
        
        # ------------------------------------------------------------------
        # Taxonomy constraints and variable collection
        # ------------------------------------------------------------------
        self._add_taxonomy_constraints(model, x)
        
        # Store
        self.model_gp = model
        self.variables = {"x": x, "v": v, "vshift": vshift, "y": y, "avg_y": avg_y}
        if not is_fully_fixed:
            self.variables["p"] = p
        
        model.update()
    
    @property
    def descriptive_name(self) -> str:
        """
        Build and return a descriptive name for the MINLP optimizer.
        
        Returns
        -------
        str
            A descriptive name constructed from key parameter flags.
        """
        parts = ["MINLPOptimizer"]
        return "_".join(parts)
    
    def _add_taxonomy_constraints(self, model: gp.Model, x: gp.MVar):
        """
        Adds taxonomic selection constraints to the base model.
        """
        if self.taxonomy_constraints is None:
            return
        
                
        T, idx_map = generate_unified_taxonomy_matrix(
            self.genome_labels,
            self.taxonomic_levels
        )
        
        k = x.shape[0]
        
        for level, cons in self.taxonomy_constraints.items():
            explicit = [t for t in cons if t != "default"]
            
            for taxon in explicit:
                key = f"{level}-{taxon}"
                if key not in idx_map:
                    continue
                row = T.getrow(idx_map[key])
                indices = row.indices.tolist()
                if not indices:
                    continue
                A = sp.coo_matrix(([1.0] * len(indices), ([0] * len(indices), indices)), shape=(1, k))
                if "min" in cons[taxon]:
                    model.addMConstr(A, x, '>=', [cons[taxon]["min"]], name=f"tax_min_{key}")
                if "max" in cons[taxon]:
                    model.addMConstr(A, x, '<=', [cons[taxon]["max"]], name=f"tax_max_{key}")
                if "exact" in cons[taxon]:
                    model.addMConstr(A, x, '=', [cons[taxon]["exact"]], name=f"tax_eq_{key}")
                    
            # Default constraints
            if "default" in cons:
                for key, row_idx in idx_map.items():
                    if not key.startswith(f"{level}-"):
                        continue
                    taxon = key.split("-", 1)[1]
                    if taxon in explicit:
                        continue
                    row = T.getrow(row_idx)
                    indices = row.indices.tolist()
                    if not indices:
                        continue
                    A = sp.coo_matrix(([1.0] * len(indices), ([0] * len(indices), indices)), shape=(1, k))
                    if "min" in cons["default"]:
                        model.addMConstr(A, x, '>=', [cons["default"]["min"]], name=f"tax_dmin_{key}")
                    if "max" in cons["default"]:
                        model.addMConstr(A, x, '<=', [cons["default"]["max"]], name=f"tax_dmax_{key}")
    
    def warmup(self, solutions, update_model=True) -> None:
        """
        Set a warm start solution for the MINLP model.
        
        Parameters
        ----------
        solutions : np.ndarray
            A binary vector/matrix used as a warm start.
        """
            
        # If solutions is a 2D numpy array, check orientation and process
        if solutions.ndim == 2:
            # Check if the number of columns matches the number of candidate variables
            x = self.variables["x"]
            if solutions.shape[1] != x.shape[0]:
                # If the number of rows exceeds the number of columns, transpose
                if solutions.shape[0] == x.shape[0]:
                    solutions = solutions.T
                else:
                    raise ValueError("Number of rows or columns in the solutions matrix does not match the number of candidate variables.")
            
            for solution in solutions:
                solution = solution.flatten()
                
                self.model_gp.Params.StartNumber = -1
                if solution.shape[0] != x.shape[0]:
                    raise ValueError("Initial solution length does not match number of candidate variables.")
                if not np.all(np.isin(solution, [0, 1])):
                    raise TypeError("Warm start solution must be binary.")
                x.Start = solution.tolist()
                
        else:
            solution = solutions.flatten()
            self.model_gp.Params.StartNumber = -1
            if solution.shape[0] != x.shape[0]:
                raise ValueError("Initial solution length does not match number of candidate variables.")
            if not np.all(np.isin(solution, [0, 1])):
                raise TypeError("Warm start solution must be binary.")
            x.Start = solution.tolist()
        
        self.model_gp.update()
        logger.info(f"Added {self.model_gp.NumStart} warm starts to MINLP model")
        
    def copy_with_scenario(self, scenario_idx: int) -> Tuple[gp.Model, Dict[str, Any]]:
        """
        Create a deep copy of the base model and apply the objective for a specific scenario.
        
        Parameters
        ----------
        scenario_idx : int
            The index of the target scenario column in M.
            
        Returns
        -------
        Tuple[gp.Model, Dict[str, Any]]
            The copied model and a dictionary of its variables.
        """
        model_copy = self.model_gp.copy()
        # Here, rather than relying on slicing by index (which can be fragile),
        # it is preferable to retrieve each variable by name.
        # For example:
        k = self.G.shape[1]
        vars_copy = {
            "x": [model_copy.getVarByName(f"x[{j}]") for j in range(k)],
            "v": [model_copy.getVarByName(f"v[{i}]") for i in range(self.M.shape[0])],
            "vshift": [model_copy.getVarByName(f"vshift[{i}]") for i in range(self.M.shape[0])],
            "y": [model_copy.getVarByName(f"y[{i}]") for i in range(self.M.shape[0])],
            "avg_y": model_copy.getVarByName("avg_y")
        }
        
        if "p" in self.variables:
            vars_copy["p"] = [
                model_copy.getVarByName(f"p[{j}]") for j in range(k)
            ]
        
        # Set the scenario-specific objective: min sum_i (y_i - avg_y - T_i)^2,
        # where T is the target vector for the scenario.
        T = self.M[:, scenario_idx]
        obj = gp.QuadExpr()
        for i in range(self.M.shape[0]):
            diff = vars_copy["y"][i] - vars_copy["avg_y"] - T[i]
            obj.add(diff * diff)
        model_copy.setObjective(obj, GRB.MINIMIZE)
        model_copy.update()
        return model_copy, vars_copy
        
    def optimize(self) -> Union[Solution, List[Solution]]:
        """
        Optimize the model for each target scenario using threading.
        
        Returns
        -------
        Union[Solution, List[Solution]]
            A single Solution or a list of Solutions.
        """
        logger.info("Optimizing MINLP per scenario using a thread pool...")
        
        self.start_time = time.time()
        
        # logging.getLogger("gurobipy").setLevel(logging.WARNING)
        
        scenario_indices = list(range(self.M.shape[1]))
        results = []
        with ThreadPoolExecutor(max_workers=self.processes) as executor:
            future_to_idx = {executor.submit(run_minlp_scenario_thread, idx, self): idx for idx in scenario_indices}
            for future in as_completed(future_to_idx):
                try:
                    sol = future.result()
                    results.append(sol)
                except Exception as e:
                    logger.error(f"Scenario {future_to_idx[future]} failed: {e}")
                    raise
        results.sort(key=lambda s: s.details["scenario"])  # sort if order matters
        
        self.result = results[0] if len(results) == 1 else results
        return self.result
