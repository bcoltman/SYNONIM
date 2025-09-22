#!/usr/bin/env python3
"""
Base classes for optimizers.

These classes define common functionality for optimizers that operate on a Model
instance. Optimizers aim to select a consortia of profiles that meet specific targets,
whether based on binary presence/absence or abundance data. Specialized solvers should
inherit from these base classes.
"""

from abc import ABC, abstractmethod
import time
import numpy as np
import logging
from typing import Any, Dict, Iterable, List, Union, Optional
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
                 weighted: bool = False,
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
        weighted : bool, optional
            Whether to weight function selection by the abundance of functions in the target metagenome
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
        
        
        # Retrieve binary matrices and weights from the model.
        self.M = self.model.metagenome_binary_matrix.copy()  # Target matrix: features x samples
        self.G = self.model.genome_binary_matrix.copy()       # Candidate matrix: features x candidates
        
        self.weights = None
        if weighted:
            self.weights = self.model.metagenome_abundance_matrix.copy()  # Abundance weights: features x samples
        
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
            
            results["TPR/recall"] = results["TP"] / results["P"]
            results["FPR"] = results["FP"] / results["N"]
            results["FNR"] = results["FN"] / results["P"]
            results["TNR/specificity"] = results["TN"] / results["N"]
            
            results["Jaccard"] = results["TP"] / (results["TP"] + results["FN"] + results["FP"])
            
            results["PPV/precision"] = results["TP"] / (results["TP"] + results["FP"])
            results["NPV"] = results["TN"] / (results["TN"] + results["FN"])
            results["FDR"] = results["FP"] / (results["TP"] + results["FP"])
            results["FOR"] = results["FN"] / (results["TN"] + results["FN"])
            
            results["ACC"] = (results["TP"] + results["TN"]) / (results["P"] + results["N"])
            results["BA"] = (results["TPR/recall"] + results["TNR/specificity"]) / 2
            results["F1_score"] = (2 * results["PPV/precision"] *results["TPR/recall"])/ (results["PPV/precision"] + results["TPR/recall"])
            
            p1 = np.sqrt(results["TPR/recall"] * results["TNR/specificity"] * results["PPV/precision"] * results["NPV"])
            p2 = np.sqrt(results["FNR"] * results["FPR"] * results["FOR"] * results["FDR"])
            results["MCC"] =  p1 - p2
            
            # Compute custom score as a linear combination of counts.
            results["custom_score"] = (results["TP"]
                            - results["FN"] 
                            - results["FP"] * self.absence_cover_penalty 
                            + results["TN"] * self.absence_match_reward)
            
            
            if self.weights is not None:
                if self.weights.ndim == 1 or self.weights.shape[1] == 1:
                    W_sample = self.weights.flatten()
                else:
                    W_sample = self.weights[:, 0]
                # Compute weighted version of the counts
                weighted_TP = np.sum((T_bool.astype(int) & np.sign(np.dot(self.G, x_single)).astype(int)) * W_sample)
                weighted_FP = np.sum(((~T_bool).astype(int) & np.sign(np.dot(self.G, x_single)).astype(int)) * W_sample)
                results["weighted_precision"] = weighted_TP / (weighted_TP + weighted_FP) if (weighted_TP + weighted_FP) != 0 else 0.0
            
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


class AbundanceOptimizer(BaseOptimizer):
    """
    Base class for optimizers that work with abundance data.
    
    This class defines methods specific to evaluating and reporting solutions in an
    abundance-based optimization context. It serves as a foundation for solvers that
    leverage continuous abundance values.
    """
    
    def __init__(self, 
                 model: Model, 
                 consortia_size: int,
                 taxonomy_constraints: Optional[Dict[str, Any]] = None,
                 taxonomic_levels: Optional[List[str]] = None,
                 required_genomes: Optional[List[Union[str, Profile]]] = None) -> None:
        """
        Initialize the AbundanceOptimizer.
        
        Parameters
        ----------
        model : Model
            The model instance supplying genome and metagenome profiles.
        consortia_size : int
            The desired number of candidate profiles to select.
        """
        super().__init__(model, consortia_size)
        
        self._optimizer_type = "abundance"
        
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
        
        # Retrieve matrices from the model.
        self.M = self.model.metagenome_abundance_matrix.copy()  # Target matrix: features x samples
        self.G = self.model.genome_abundance_matrix.copy()       # Candidate matrix: features x candidates
        
    @property
    def optimizer_type(self) -> str:
        """
        Get the type of the optimizer.
        
        Returns
        -------
        str
            A string indicating the optimizer type (e.g., "abundance").
        """
        return self._optimizer_type
    
    def analyze_solution(
        self,
        T: np.ndarray,
        p_opt: np.ndarray,
        x_opt: np.ndarray
    ) -> Dict[str, Any]:
        """
        Analyze an abundance-based optimization solution.
        
        Parameters
        ----------
        T : np.ndarray
            CLR-transformed target profile (shape d,).
        p_opt : np.ndarray
            Optimized continuous abundance vector (length n).
        x_opt : np.ndarray
            Optimized binary vector indicating selected profiles (length n).
            
        Returns
        -------
        dict
            Dictionary of analysis metrics: distance, correlation, abundance use, taxonomy.
        """
        
        v = self.G @ p_opt.reshape(-1, 1)
        v = v.flatten()
        v_shift = v + self.epsilon
        y = np.log(v_shift)
        avg_y = np.mean(y)
        c = y - avg_y
        
        mse = np.sum((c - T) ** 2)
        mae = np.mean(np.abs(c - T))
        corr = np.corrcoef(c, T)[0, 1] if np.std(c) > 0 and np.std(T) > 0 else 0.0
        total_abundance = np.sum(p_opt)
        
        results: Dict[str, Any] = {
            "MSE": float(mse),
            "MAE": float(mae),
            "Pearson_correlation": float(corr),
            "Total_abundance": float(total_abundance),
        }
        
        
        if self.taxonomy_constraints and self.genome_labels:
            selected_taxa = {}
            labels = self.genome_labels
            
            for idx in np.where(x_opt == 1)[0]:
                label = labels[idx]
                for level in self.taxonomic_levels:
                    key = label.get(level, "Unknown") if isinstance(label, dict) else "Unknown"
                    selected_taxa.setdefault(level, {})
                    selected_taxa[level][key] = selected_taxa[level].get(key, 0) + 1
            
            results["Taxonomic_counts"] = selected_taxa
            
        return results
        
    @abstractmethod
    def optimize(self) -> Any:
        """
        Optimize the model using an abundance-based approach.
        
        This method should be implemented by subclasses with the specific abundance
        optimization logic.
        
        Returns
        -------
        Any
            The result of the abundance optimization procedure.
        """
        pass
        
    def __repr__(self) -> str:
        """
        Return a string representation of the AbundanceOptimizer.
        
        Returns
        -------
        str
            A string indicating the optimizer type and consortia size.
        """
        return f"<{self.__class__.__name__}(consortia_size={self.consortia_size}, optimizer_type={self.optimizer_type})>"
