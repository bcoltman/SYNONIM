import numpy as np
import logging
from typing import Any, Dict, List, Optional, Union

import time

# Import the base BinaryOptimizer and the Solution container.
from synonim.optimizers import BinaryOptimizer, Solution
from synonim.core import Profile, Model

logger = logging.getLogger(__name__)


class BinaryHeuristic(BinaryOptimizer):
    """
    Heuristic optimizer for multi-level genome selection under taxonomy constraints.
    
    This optimizer selects candidate genome profiles one-by-one via a heuristic search.
    It relies on binary matrices describing the metagenome (targets) and genome (candidate)
    feature presence, as well as an abundance matrix used for weighting.
    
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
    mask_covered_absent_features : bool, optional
        Whether to mask features with absent coverage after a candidate is selected.
    mask_covered_present_features : bool, optional
        Whether to mask features with present coverage after a candidate is selected.
    mask_covered_isolate_features : bool, optional
        Whether to remove candidate signals for entire features when a candidate covers them.
    weighted : bool, optional
        Whether to weight function selection by the abundance of functions in the target metagenome
    """
    
    def __init__(
        self,
        model: Model,
        consortia_size: int,
        weighted: bool = False,
        taxonomy_constraints: Optional[Dict[str, Any]] = None,
        taxonomic_levels: Optional[List[str]] = None,
        required_genomes: Optional[List[Union[str, Profile]]] = None,
        absence_cover_penalty: float = 0,
        absence_match_reward: float = 1,
        mask_covered_absent_features: bool = False,
        mask_covered_present_features: bool = False,
        mask_covered_isolate_features: bool = False
    ) -> None:
        """
        Initialize the BinaryHeuristic optimizer.
        
        Parameters
        ----------
        model : Model
            The model instance supplying profiles and binary/abundance matrices.
        consortia_size : int
            The target number of candidate profiles to select.
        taxonomy_constraints : dict, optional
            Mapping of taxonomic level to per-taxon constraints.
        taxonomic_levels : list, optional
            List of taxonomic levels to enforce; if None, defaults to the keys of taxonomy_constraints.
        required_genomes : list of (str or Profile), optional
            Candidate profiles (or their IDs) that must be included.
        absence_cover_penalty : float, optional
            Multiplier penalty for candidate features that cover absent features.
        absence_match_reward : float, optional
            Multiplier reward for candidate features that match absent features.
        mask_covered_absent_features : bool, optional
            Whether to zero out absent feature signals once covered.
        mask_covered_present_features : bool, optional
            Whether to zero out present feature signals once covered.
        mask_covered_isolate_features : bool, optional
            Whether to mask entire candidate features after selection.
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
        
        self.mask_covered_absent_features = mask_covered_absent_features
        self.mask_covered_present_features = mask_covered_present_features
        self.mask_covered_isolate_features = mask_covered_isolate_features
        
    @property
    def descriptive_name(self) -> str:
        """
        Construct a descriptive name for the optimizer.
        
        Returns
        -------
        str
            A concatenation of flags and parameters (e.g., "BinaryHeuristic_Weighted_MA_MP_AMR-1_ACP-0").
        """
        name_parts = ["BinaryHeuristic"]
        # Optionally, if weighted analysis is applied.
        if hasattr(self, "weights") and self.weights is not None:
            name_parts.append("Weighted")
        if self.mask_covered_absent_features:
            name_parts.append("MA")  # mask absent
        if self.mask_covered_present_features:
            name_parts.append("MP")  # mask present
        if self.mask_covered_isolate_features:
            name_parts.append("MI")  # mask isolate
        if self.absence_match_reward:
            name_parts.append(f"AMR-{self.absence_match_reward}")  # absence reward
        if self.absence_cover_penalty:
            name_parts.append(f"ACP-{self.absence_cover_penalty}")  # absence penalty
        return "_".join(name_parts)
        
    def optimize(self) -> Union[Solution, List[Solution]]:
        """
        Execute the heuristic optimization to select candidate profiles.
        
        For each sample (column in the target matrix), the algorithm:
          - Determines candidate weights.
          - Applies required selections (if any).
          - Iteratively selects additional candidates based on computed scores,
            enforcing taxonomic constraints.
            
        Returns
        -------
        Solution or list of Solution
            A Solution object for a single sample or a list of Solutions for multiple samples.
        """
        
        
        # Get dimensions: d = number of features, s = number of samples, k = number of candidates.
        d, s = self.M.shape
        k = self.G.shape[1]
        
        # Preallocate solution and score arrays.
        X_opt = np.zeros((k, s), dtype=int)
        pick_scores = np.zeros((self.consortia_size, s))
        solutions = []  # To collect a Solution for each sample.
        
        for sample_idx in range(s):
            
            start_time = time.time()
            
            # Determine per-feature weights for the current sample.
            if self.weights is not None and self.weights.shape == (d, s):
                current_weights = self.weights[:, sample_idx]
            elif self.weights is not None:
                current_weights = self.weights[:, 0]
            else:
                current_weights = np.ones(d)
                
            # Create a working copy of the candidate matrix locally.
            isolates = self.G.copy()
            
            # Build the status vector for features:
            # 1 indicates the target feature is present; 2 indicates absent.
            status = np.where(self.M[:, sample_idx] == 1, 1, 2)
            
            # Initialize selection tracking.
            selected_indices = set(self.required_genomes)
            selected_order = list(self.required_genomes)
            taxonomy_selected_counts = self._init_taxonomy_counts()
            
            # Process required candidate profiles.
            if self.required_genomes:
                for j_req in self.required_genomes:
                    X_opt[j_req, sample_idx] = 1
                    self._update_taxonomy_counts(taxonomy_selected_counts, j_req)
                    self._apply_coverage_vectorized(status, isolates, j_req)
                    
            # Determine how many additional candidates need to be picked.
            picks_needed = self.consortia_size - len(selected_indices)
            
            for itr in range(picks_needed):
                # Create a mask that excludes already selected candidates.
                mask = np.ones(k, dtype=bool)
                mask[list(selected_indices)] = False
                
                # Determine which candidates are feasible given the taxonomy constraints.
                feasible = self._feasible_taxonomy_vectorized(mask, taxonomy_selected_counts)
                # Compute candidate scores.
                scores = self._compute_scores_vectorized(status, current_weights, isolates)
                # Mark infeasible or already-selected candidates with -infinity.
                scores[~mask | ~feasible] = -np.inf
                
                best_j = int(np.argmax(scores))
                if scores[best_j] == -np.inf:
                    # No feasible candidate found.
                    break
                    
                # Record the candidate selection.
                X_opt[best_j, sample_idx] = 1
                selected_indices.add(best_j)
                selected_order.append(best_j)
                pick_scores[itr, sample_idx] = scores[best_j]
                
                self._update_taxonomy_counts(taxonomy_selected_counts, best_j)
                self._apply_coverage_vectorized(status, isolates, best_j)
                
            # Compute a simple objective as the sum of pick scores.
            objective_value = float(np.sum(pick_scores[:, sample_idx]))
            # Build the selection order names.
            selection_order_names = (
                [self.model.genome_names[i] for i in selected_order]
                if self.model.genome_names is not None
                else selected_order
            )
            
            # Analyze the resulting solution.
            analysis_metrics = self.analyze_solution(
                self.M[:, sample_idx], X_opt[:, sample_idx]
            )
            
            elapsed_time = time.time() - start_time
            
            # Create a Solution object with pick scores and analysis details.
            sol = Solution(method=self.descriptive_name,
                           X_opt=X_opt[:, sample_idx],
                           objective=objective_value,
                           genome_names=self.genome_names,
                           selection_order=selection_order_names,
                           details={"scenario": sample_idx, 
                                    "runtime":elapsed_time,
                                    "pick_scores": pick_scores[:, sample_idx],
                                    "analysis": analysis_metrics
                                    }
                           )
            solutions.append(sol)
            
        # If only one sample, return a single Solution; otherwise, return a list.
        result: Union[Solution, List[Solution]] = solutions[0] if s == 1 else solutions
        self.result = result
        return result
        
    def _init_taxonomy_counts(self) -> Dict[str, Dict[Any, int]]:
        """
        Initialize taxonomy counts for each taxonomic level.
        
        Returns
        -------
        dict
            A dictionary mapping each taxonomic level to an empty dictionary.
        """
        counts: Dict[str, Dict[Any, int]] = {}
        if self.taxonomy_constraints is not None and self.genome_labels is not None:
            for level in self.taxonomy_constraints.keys():
                counts[level] = {}
        return counts
        
    def _update_taxonomy_counts(
        self, taxonomy_counts: Dict[str, Dict[Any, int]], j: int
    ) -> None:
        """
        Update taxonomy counts for a selected candidate.
        
        Parameters
        ----------
        taxonomy_counts : dict
            The current taxonomy counts.
        j : int
            The index of the selected candidate.
        """
        if self.taxonomy_constraints is not None and self.genome_labels is not None:
            candidate = self.genome_labels[j]
            for level in self.taxonomy_constraints.keys():
                # Get the taxon for the candidate at the given level.
                taxon = candidate.get(level) if isinstance(candidate, dict) else None
                if taxon is not None:
                    taxonomy_counts[level][taxon] = taxonomy_counts[level].get(taxon, 0) + 1
                    
    def _feasible_taxonomy_vectorized(
        self, mask: np.ndarray, taxonomy_counts: Dict[str, Dict[Any, int]]
    ) -> np.ndarray:
        """
        Determine which candidates satisfy the taxonomy constraints.
        
        For each candidate where mask[j] is True, check that the candidate's taxon at each level
        does not exceed the allowed threshold.
        
        Parameters
        ----------
        mask : np.ndarray
            Boolean array indicating available candidates.
        taxonomy_counts : dict
            Current counts of selected candidates per taxon.
            
        Returns
        -------
        np.ndarray
            Boolean array indicating feasibility for each candidate.
        """
        feasible = mask.copy()
        if self.taxonomy_constraints is None or self.genome_labels is None:
            return feasible
            
        for j in np.where(mask)[0]:
            candidate = self.genome_labels[j]
            for level, constraints in self.taxonomy_constraints.items():
                candidate_taxon = candidate.get(level) if isinstance(candidate, dict) else None
                if candidate_taxon is None:
                    continue
                # Determine the allowed count for this taxon.
                allowed = None
                if candidate_taxon in constraints:
                    cons = constraints[candidate_taxon]
                    allowed = cons.get("exact", cons.get("max"))
                if allowed is None and "default" in constraints:
                    allowed = constraints["default"].get("max", float("inf"))
                if allowed is None:
                    allowed = float("inf")
                current_count = taxonomy_counts.get(level, {}).get(candidate_taxon, 0)
                if current_count >= allowed:
                    feasible[j] = False
                    break
        return feasible
        
    def _compute_scores_vectorized(
        self, status: np.ndarray, weights: np.ndarray, isolates: np.ndarray
    ) -> np.ndarray:
        """
        Compute candidate scores based on feature matching.
        
        Scores are determined by:
          - The dot product of the candidate matrix (isolates) with the weighted indicator of present target features.
          - Adding a reward for matching absent features.
          - Subtracting a penalty for covering absent features.
          
        Parameters
        ----------
        status : np.ndarray
            Vector (length d) indicating for each feature if it is present (1) or absent (2) in the target.
        weights : np.ndarray
            Weights for each feature (from the abundance matrix).
        isolates : np.ndarray
            The working copy of the candidate matrix.
            
        Returns
        -------
        np.ndarray
            Computed scores for each candidate (length k).
        """
        # Explicitly cast boolean results to float for the dot products.
        present_indicator = (status == 1).astype(float)
        absent_indicator = (status == 2).astype(float)
        
        # Compute matching scores.
        present_matches = np.dot(isolates.T, present_indicator * weights)
        absent_matches = (
            self.absence_match_reward * np.dot((1 - isolates).T, absent_indicator)
            if self.absence_match_reward > 0
            else 0
        )
        absence_penalty_score = (
            self.absence_cover_penalty * np.dot(isolates.T, absent_indicator)
            if self.absence_cover_penalty > 0
            else 0
        )
        return present_matches + absent_matches - absence_penalty_score
        
    def _apply_coverage_vectorized(
        self, status: np.ndarray, isolates: np.ndarray, j: int
    ) -> None:
        """
        Update feature coverage after a candidate is selected.
        
        Depending on the configuration, features covered by the selected candidate may be
        masked in the status vector or removed from the candidate matrix (isolates).
        
        Parameters
        ----------
        status : np.ndarray
            Current status vector for features.
        isolates : np.ndarray
            The current working copy of the candidate matrix.
        j : int
            Index of the selected candidate.
        """
        coverage = isolates[:, j]
        if self.mask_covered_present_features:
            # Mask features that are present in the target and covered by the candidate.
            status[:] = np.where((coverage == 1) & (status == 1), 0, status)
        if self.mask_covered_absent_features:
            # Mask features that are absent in the target but now covered.
            status[:] = np.where((coverage == 1) & (status == 2), 0, status)
        if self.mask_covered_isolate_features:
            # Completely remove candidate signals for features covered by the selected candidate.
            isolates[coverage.astype(bool), :] = 0
    
    def update_consortia_size(self, consortia_size: int) -> None:
        """
        Update the consortia size parameter.
        
        Parameters
        ----------
        consortia_size : int
            The new desired consortia size.
            
        Raises
        ------
        ValueError
            If consortia_size is not a positive integer or exceeds available candidate profiles.
        """
        if not isinstance(consortia_size, int) or consortia_size <= 0:
            raise ValueError("Consortia size must be a positive integer.")
        
        # If the model has profiles, check that consortia_size does not exceed the number of candidates.
        if hasattr(self.model, "profiles"):
            available_candidates = len(self.model.profiles)
            if consortia_size > available_candidates:
                raise ValueError(
                    f"Consortia size {consortia_size} cannot exceed the number of available candidate profiles ({available_candidates})."
                )
        self.consortia_size = consortia_size
