#!/usr/bin/env python3
"""
Base classes for optimizers.

These classes define common functionality for optimizers that operate on a Model
instance. Optimizers aim to select a consortia of profiles that meet specific targets,
based on binary presence/absence data. Specialized solvers should inherit from these
base classes.
"""

from abc import ABC, abstractmethod
from collections.abc import Mapping
import time
import numpy as np
import logging
from typing import Any, Dict, List, Union, Optional
from synonim.core import Model, Profile

logger = logging.getLogger(__name__)


# in __init__, after reading taxonomic_levels & taxonomy_constraints:
DEFAULT_RANKS = ["kingdom", "domain", "phylum", "class", "order", "family", "genus", "species"]

class BaseOptimizer(ABC):
    """
    Abstract base class for all optimizers.
    
    This class provides a common interface and shared methods for processing the model
    data such as aggregating features across profiles and analyzing optimization results.
    It also manages the consortia_size parameter.
    
    Attributes
    ----------
    model : Model
        The model instance that supplies profile and feature data.
    consortia_size : int
        The desired number of profiles to select.
    """
    
    def __init__(self, model: Model, consortia_size: int) -> None:
        """
        Initialize the BaseOptimizer.
        
        Parameters
        ----------
        model : Model
            The model instance containing profiles and feature data.
        consortia_size : int
            The desired number of candidate profiles to select.
        """
        self.model = model
        
        self.metagenome_names = self.model.metagenome_names
        
        self.genome_names = self.model.genome_names
        
        self.consortia_size = consortia_size
    
    def _get_all_features(self) -> List[Any]:
        """
        Aggregate and return a sorted list of unique features across all profiles.
        
        Returns
        -------
        List[Any]
            A sorted list of features (sorted by each feature's id).
        """
        feature_set = set()
        for profile in self.model.profiles:
            feature_set.update(profile.features)
        return sorted(feature_set, key=lambda f: f.id)
        
    @abstractmethod
    def optimize(self) -> Any:
        """
        Perform the optimization procedure.
        
        Subclasses must override this method with a specific optimization algorithm that
        uses self.model and self.consortia_size to generate a solution.
        
        Returns
        -------
        Any
            The result of the optimization procedure.
        """
        pass
    
    def _resolve_required_genomes(
        self, 
        required_genomes: Optional[List[Union[str, Profile]]]
    ) -> List[int]:
        """
        Convert the required_genomes input (list of profiles or IDs) to a list of candidate indices.
        Assumes that the candidate profiles are stored in self.model.genome_profiles,
        which is a DictList.
        """
        candidate_profiles = self.model.genome_profiles  # This is the ordered list of genome profiles
        if required_genomes is None:
            return []
        indices = []
        for item in required_genomes:
            if isinstance(item, str):
                # If a string, treat it as an ID and use get_by_id to retrieve the candidate
                candidate = candidate_profiles.get_by_id(item)
                idx = candidate_profiles.index(candidate)
            elif isinstance(item, Profile):
                # If it is a Profile instance, look it up in candidate_profiles
                idx = candidate_profiles.index(item)
            else:
                raise TypeError("Elements in required_genomes must be either profile IDs (str) or Profile objects.")
            indices.append(idx)
        return indices
        
    def __repr__(self) -> str:
        """
        Return a string representation of the optimizer.
        
        Returns
        -------
        str
            A string indicating the class name and consortia size.
        """
        return f"<{self.__class__.__name__}(consortia_size={self.consortia_size})>"


class BinaryOptimizer(BaseOptimizer):
    """
    Base class for optimizers that work with binary (presence/absence) data.
    
    This class defines methods specific to evaluating and reporting solutions in a
    binary optimization context. It is meant to be further subclassed by specific
    optimization algorithms (e.g., heuristic, genetic, MILP).
    """
    
    def __init__(self, 
                 model: Model, 
                 consortia_size: int,
                 weights: Optional[Any] = None,
                 taxonomy_constraints: Optional[Dict[str, Any]] = None,
                 taxonomic_levels: Optional[List[str]] = None,
                 required_genomes: Optional[List[Union[str, Profile]]] = None,
                 absence_cover_penalty: float = 0,
                 absence_match_reward: float = 1) -> None:
        """
        Initialize the BinaryOptimizer.
        
        Parameters
        ----------
         model : Model
            The model instance supplying profiles and matrices.
        consortia_size : int
            Number of candidate profiles to select.
        weights : optional
            Explicit feature weights for the binary objective. Accepts a mapping keyed by
            feature ID, a 1-D array-like of length ``n_features``, or a 2-D array-like
            with shape ``(n_features, n_metagenomes)``. Missing mapping keys default to 1.
        taxonomy_constraints : dict, optional
            Mapping of taxonomic level to per-taxon constraints.
        taxonomic_levels : list of str, optional
            List of taxonomic levels to enforce (defaults to taxonomy_constraints keys).
        required_genomes : list of (str or Profile), optional
            Candidate profiles (or their IDs) that must be included.
        absence_cover_penalty : float, optional
            Penalty multiplier for covering absent features.
        absence_match_reward : float, optional
            Reward multiplier for matching absent features.
        """
        super().__init__(model, consortia_size)
        self._optimizer_type = "binary"
        
        self.taxonomy_constraints = taxonomy_constraints
        if taxonomy_constraints:
            # take user‐provided list if any, otherwise the dict keys
            initial = taxonomic_levels if taxonomic_levels is not None else list(taxonomy_constraints.keys())
            # only keep those in the canonical hierarchy, in the correct order
            self.taxonomic_levels = [lvl for lvl in DEFAULT_RANKS if lvl in initial]
        else:
            self.taxonomic_levels = None
        
        # Convert the user-passed required_genomes to candidate indices.
        self.required_genomes = self._resolve_required_genomes(required_genomes)
        
        # Retrieve taxonomy labels and genome names from the model.
        self.genome_labels = self.model.get_genome_labels(self.taxonomic_levels)
        # self.genome_names = self.model.genome_names
        
        self.absence_cover_penalty = absence_cover_penalty
        self.absence_match_reward = absence_match_reward
        
        if self.absence_cover_penalty == 0 and self.absence_match_reward:
            logger.warning(
                "'absence_cover_penalty' is 0 while 'absence_match_reward' is non-zero; please check parameter settings."
            )
        
        
        # Retrieve binary matrices from the model.
        self.M = self.model.metagenome_binary_matrix.copy()  # Target matrix: features x samples
        self.G = self.model.genome_binary_matrix.copy()       # Candidate matrix: features x candidates
        self.weighted = weights is not None
        self.weights = self._normalize_weights(weights)

    def _normalize_weights(self, weights: Optional[Any]) -> np.ndarray:
        """
        Return a non-negative feature-by-sample weight matrix aligned to model features.

        Weights are intentionally independent of any particular biological quantity: callers may
        supply measurement-derived weights, confidence weights, benchmarking weights, or plain
        feature priorities as long as they align with the model feature order.
        """
        d, s = self.M.shape
        if weights is None:
            return np.ones((d, s), dtype=float)

        feature_ids = [feature.id for feature in self.model.features]
        sample_names = self.metagenome_names

        if hasattr(weights, "reindex") and hasattr(weights, "columns"):
            aligned = weights.reindex(index=feature_ids, columns=sample_names)
            values = aligned.fillna(1.0).to_numpy(dtype=float)
        elif hasattr(weights, "reindex") and hasattr(weights, "index") and not hasattr(weights, "columns"):
            aligned = weights.reindex(feature_ids)
            values = aligned.fillna(1.0).to_numpy(dtype=float)
        elif isinstance(weights, Mapping):
            values = np.array([float(weights.get(feature_id, 1.0)) for feature_id in feature_ids], dtype=float)
        else:
            values = np.asarray(weights, dtype=float)

        if values.ndim == 1:
            if values.shape[0] != d:
                raise ValueError(f"Feature weights must have length {d}; received {values.shape[0]}.")
            values = np.tile(values.reshape(-1, 1), (1, s))
        elif values.ndim == 2:
            if values.shape == (d, 1):
                values = np.tile(values, (1, s))
            elif values.shape != (d, s):
                raise ValueError(f"Feature weights must have shape {(d, s)}; received {values.shape}.")
        else:
            raise ValueError("Feature weights must be one- or two-dimensional.")

        if not np.all(np.isfinite(values)):
            raise ValueError("Feature weights must be finite numeric values.")
        if np.any(values < 0):
            raise ValueError("Feature weights must be non-negative.")

        return values.astype(float, copy=False)
        
    @property
    def optimizer_type(self) -> str:
        """
        Get the type of the optimizer.
        
        Returns
        -------
        str
            A string indicating the optimizer type (e.g., "binary").
        """
        return self._optimizer_type
    
    def analyze_solution(self, T: np.ndarray, x_opt: np.ndarray) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Analyze the optimization solution(s) given a target vector and a binary selection vector.
        
        Parameters
        ----------
        T : np.ndarray
            Target binary vector (shape (d,)) or matrix (shape (d, s)).
        x_opt : np.ndarray
            Binary solution vector (shape (n,)) or matrix (shape (n, s)).
        
        Returns
        -------
        Union[Dict[str, Any], List[Dict[str, Any]]]
            Analysis metrics for the provided solution(s).
        """
        
        def compute_metrics_single(T_single: np.ndarray, x_single: np.ndarray) -> Dict[str, Any]:
            def safe_divide(numerator: float, denominator: float) -> float:
                return float(numerator / denominator) if denominator else 0.0

            # Convert to boolean for logical operations.
            T_bool = T_single.astype(bool)
            # 'combined' is the aggregated candidate coverage.
            combined = np.sign(np.dot(self.G, x_single)).astype(bool)
            
            results = {}
            
            results["P"] = int(np.sum(T_bool))
            results["N"] = int(T_bool.shape[0]) - results["P"]
            
            # Compute confusion matrix counts.
            results["TP"] = int(np.sum(T_bool & combined))
            results["FN"] = int(np.sum(T_bool & ~combined))
            results["FP"] = int(np.sum(~T_bool & combined))
            results["TN"] = int(np.sum(~T_bool & ~combined))
            
            results["TPR/recall"] = safe_divide(results["TP"], results["P"])
            results["FPR"] = safe_divide(results["FP"], results["N"])
            results["FNR"] = safe_divide(results["FN"], results["P"])
            results["TNR/specificity"] = safe_divide(results["TN"], results["N"])
            
            results["Jaccard"] = safe_divide(results["TP"], results["TP"] + results["FN"] + results["FP"])
            
            results["PPV/precision"] = safe_divide(results["TP"], results["TP"] + results["FP"])
            results["NPV"] = safe_divide(results["TN"], results["TN"] + results["FN"])
            results["FDR"] = safe_divide(results["FP"], results["TP"] + results["FP"])
            results["FOR"] = safe_divide(results["FN"], results["TN"] + results["FN"])
            
            results["ACC"] = safe_divide(results["TP"] + results["TN"], results["P"] + results["N"])
            results["BA"] = (results["TPR/recall"] + results["TNR/specificity"]) / 2
            results["F1_score"] = safe_divide(
                2 * results["PPV/precision"] * results["TPR/recall"],
                results["PPV/precision"] + results["TPR/recall"],
            )
            
            p1 = np.sqrt(results["TPR/recall"] * results["TNR/specificity"] * results["PPV/precision"] * results["NPV"])
            p2 = np.sqrt(results["FNR"] * results["FPR"] * results["FOR"] * results["FDR"])
            results["MCC"] =  p1 - p2
            
            # Compute custom score as a linear combination of counts.
            results["custom_score"] = (results["TP"]
                            - results["FN"] 
                            - results["FP"] * self.absence_cover_penalty 
                            + results["TN"] * self.absence_match_reward)
            
            
            # Optionally, add taxonomic counts if available.
            if self.taxonomy_constraints and hasattr(self.model, "get_genome_labels"):
                selected_taxa = {}
                labels = self.model.get_genome_labels(self.taxonomic_levels)
                for idx in np.where(x_single == 1)[0]:
                    label = labels[idx]
                    for level in self.taxonomic_levels:
                        taxon = label.get(level, "Unknown") if isinstance(label, dict) else "Unknown"
                        selected_taxa.setdefault(level, {}).setdefault(taxon, 0)
                        selected_taxa[level][taxon] += 1
                results["Taxonomic_counts"] = selected_taxa
                
            # === Extension: Robustness & Redundancy metrics ===
            sel_idx = np.where(x_single == 1)[0]
            G_sel = self.G[:, sel_idx] if len(sel_idx) > 0 else np.zeros((self.G.shape[0], 0))
            m = np.sum(G_sel, axis=1)  
            
            
            if len(sel_idx) > 0:
                # Redundancy metrics
                if np.sum(T_bool) > 0:
                    redundancy = np.mean(np.maximum(m[T_bool] - 1, 0))
                    spf = np.mean(m[T_bool] == 1)
                else:
                    redundancy, spf = 0.0, 0.0
                results["Redundancy_index"] = float(redundancy)
                results["Single_point_failure_frac"] = float(spf)
                
                # Robustness metrics (single deletions of strains)
                robustness_vals = []
                for i in range(G_sel.shape[1]):
                    m_minus_i = m - G_sel[:, i]
                    covered_after_del = np.sum((m_minus_i > 0) & T_bool)
                    robustness_vals.append(covered_after_del / np.sum(T_bool) if np.sum(T_bool) > 0 else 0.0)
                results["Robustness_avg_del"] = float(np.mean(robustness_vals))
                results["Robustness_min_del"] = float(np.min(robustness_vals))
                
                # Contribution evenness (strain-level Shannon & Gini-Simpson)
                contrib = np.sum(G_sel[T_bool, :], axis=0)  # contributions per strain
                if np.sum(contrib) > 0:
                    p_contrib = contrib / np.sum(contrib)
                    shannon = -np.sum(p_contrib * np.log(p_contrib + 1e-12))
                    gini_simpson = 1 - np.sum(p_contrib ** 2)
                else:
                    shannon, gini_simpson = 0.0, 0.0
                results["Shannon_strain_contrib"] = float(shannon)
                results["GiniSimpson_strain_contrib"] = float(gini_simpson)
                
                # Probability of Failure (PoF, approximate up to lethal pairs)
                p_fail = getattr(self, "strain_fail_prob", 0.01)  # per-strain fail probability
                essentials = [
                    i for i in range(G_sel.shape[1])
                    if np.any((G_sel[:, i] == 1) & (m == 1) & T_bool)
                ]
                PoF = len(essentials) * p_fail
                for i in range(G_sel.shape[1]):
                    for j in range(i + 1, G_sel.shape[1]):
                        shared_loss = np.any(((m - G_sel[:, i] - G_sel[:, j]) == 0) & T_bool)
                        if shared_loss:
                            PoF += p_fail ** 2
                results["PoF"] = min(PoF, 1.0)
                results["Robustness_prob"] = 1 - results["PoF"]
                
            return results
        
        
        # Handle both single and multiple-solution cases.
        if x_opt.ndim == 1:
            return compute_metrics_single(T, x_opt)
        else:
            return [compute_metrics_single(T[:, i], x_opt[:, i]) for i in range(x_opt.shape[1])]
        
    @abstractmethod
    def optimize(self) -> Any:
        """
        Optimize the model using a binary (presence/absence) approach.
        
        This method should be implemented by subclasses with a specific optimization
        strategy (e.g., heuristic, genetic algorithms, or MILP).
        
        Returns
        -------
        Any
            The result of the binary optimization procedure.
        """
        pass
    
        
    def __repr__(self) -> str:
        """
        Return a string representation of the BinaryOptimizer.
        
        Returns
        -------
        str
            A string indicating the optimizer type and consortia size.
        """
        return f"<{self.__class__.__name__}(consortia_size={self.consortia_size}, optimizer_type={self.optimizer_type})>"
