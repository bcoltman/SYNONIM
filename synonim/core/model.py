import logging
from copy import deepcopy, copy
from functools import partial
from typing import Any, Dict, Iterable, List, Optional, Union
import numpy as np

from ..util.context import HistoryManager, get_context, resettable
from .object import Object
from .dictlist import DictList
from .profile import Profile
from .feature import Feature

logger = logging.getLogger(__name__)


class Model(Object):
    """
    Represents the main model that aggregates and manages profiles and features.
    
    The Model acts as a container that dynamically manages Profile objects and a global
    registry of Feature objects. When profiles are added, their features are automatically
    checked against the global registry and added if missing. Similarly, when profiles are removed,
    the model may (optionally) remove orphan features.
    
    Candidate matrices and metadata (e.g., genome_names and genome_labels) are derived from the 
    ordered list of profiles and features. Matrix properties (binary and abundance) are cached
    after first computation. When the model is modified (e.g. adding/removing profiles or features),
    the caches are invalidated automatically.
    
    This version supports context-based reversible modifications. When the model is used as a context
    (via a with-statement), any changes that register reversal actions (for example, adding or removing
    profiles) are automatically undone upon exiting the context.
    """
    
    def __init__(
        self,
        id_or_model: Union[str, "Model", None] = None,
        name: Optional[str] = None
    ) -> None:
        """
        Initialize an empty Model.
        
        Parameters
        ----------
        id_or_model : str or Model, optional
            A unique identifier for the model or an existing model from which to copy state.
        name : str, optional
            A human-readable name for the model.
        
        Notes
        -----
        This constructor creates an empty model with no profiles or features.
        Profiles and features can be added later using add_profile() and add_features().
        """
        if isinstance(id_or_model, Model):
            Object.__init__(self, name=name)
            self.__setstate__(id_or_model.__dict__)
        else:
            super().__init__(id=id_or_model, name=name)
            # Global registries.
            self.profiles: DictList[Profile] = DictList()          # All profiles.
            self.features: DictList[Feature] = DictList()            # Global feature registry.
            self.genome_profiles: DictList[Profile] = DictList()     # Subset of profiles (genomes).
            self.metagenome_profiles: DictList[Profile] = DictList()   # Subset of profiles (metagenomes).
            self._contexts: List[Any] = []                           # For reversible changes.
            # Cached matrices; set to None to indicate cache invalidation.
            self._genome_binary_matrix_cache: Optional[np.ndarray] = None
            self._genome_abundance_matrix_cache: Optional[np.ndarray] = None
            self._metagenome_binary_matrix_cache: Optional[np.ndarray] = None
            self._metagenome_abundance_matrix_cache: Optional[np.ndarray] = None
            
    def _invalidate_cache(self) -> None:
        """
        Invalidate cached matrices. This is called automatically whenever the model state changes.
        """
        self._genome_binary_matrix_cache = None
        self._genome_abundance_matrix_cache = None
        self._metagenome_binary_matrix_cache = None
        self._metagenome_abundance_matrix_cache = None
        
    def __getstate__(self) -> Dict:
        """
        Prepare the Model for pickling.
        
        Clears transient attributes (such as the context stack and caches) to avoid pickling issues.
        
        Returns
        -------
        dict
            A dictionary representing the state of the model.
        """
        odict = self.__dict__.copy()
        odict["_contexts"] = []
        # Do not pickle the caches.
        # odict["_genome_binary_matrix_cache"] = None
        # odict["_genome_abundance_matrix_cache"] = None
        # odict["_metagenome_binary_matrix_cache"] = None
        # odict["_metagenome_abundance_matrix_cache"] = None
        return odict
        
    def __setstate__(self, state: Dict) -> None:
        """
        Restore the Model state from a pickled state dictionary.
        
        Re-establishes relationships between profiles and features.
        
        Parameters
        ----------
        state : dict
            The state dictionary.
        """
        self.__dict__.update(state)
        # Reinitialize transient attributes.
        self._contexts = []
        self._invalidate_cache()
        # Re-link each profile's model reference and ensure each feature knows its profile.
        for profile in self.profiles:
            profile._model = self
            for feature in profile._features.keys():
                feature._profiles.add(profile)
                
    # --- Profile Management ---
    def add_profiles(self, profiles: Iterable[Profile]) -> None:
        """
        Add multiple profiles to the model.
        
        For each profile added, its features are validated against the global registry.
        Missing features are added. Each profile's _model attribute is set to this model.
        Reversible actions are recorded if an active context exists.
        
        Parameters
        ----------
        profiles : Iterable[Profile]
            An iterable of Profile objects.
        """
        for profile in profiles:
            self.add_profile(profile)
        self._invalidate_cache()
    
    def add_profile(self, profile: Profile) -> None:
        """
        Add a single profile to the model.
        
        Before adding, the profile's features are registered globally if missing.
        The profile’s _model attribute is set to this model, and the profile is appended
        to the model’s profile list. The profiles are then reclassified.
        
        If an active HistoryManager context is found, reversal actions are recorded so that
        the addition can be undone.
        
        Parameters
        ----------
        profile : Profile
            The Profile object to add.
        """
        if profile.id in self.profiles:
            logger.warning(f"Ignoring profile '{profile.id}' since it already exists.")
            return
        
            # Efficiently check which features are not yet in the model using set logic.
        existing_ids = set(f.id for f in self.features)
        features_to_add = [feat for feat in profile.features if feat.id not in existing_ids]
        if features_to_add:
            self.add_features(features_to_add)
        
        
        # Record a reversible action (if in a context) to remove the profile.
        context = get_context(self)
        if context:
            context(partial(self.profiles.remove, profile))
            context(partial(setattr, profile, "_model", None))
            context(self._classify_profiles)
        
        profile._model = self
        self.profiles.append(profile)
        self._classify_profiles()
        self._invalidate_cache()
        
    def remove_profiles(
        self,
        profiles: Union[str, Profile, Iterable[Union[str, Profile]]],
        remove_orphans: bool = False,
    ) -> None:
        """
        Remove one or more profiles from the model.
        
        Clears the _model attribute in each removed profile. Optionally, orphan features (no longer
        present in any profile) can also be removed.
        
        If an active HistoryManager context is found, reversal actions are recorded so that the removal
        can be undone.
        
        Parameters
        ----------
        profiles : str, Profile, or Iterable[str or Profile]
            The profile(s) to remove.
        remove_orphans : bool, optional
            If True, features not present in any remaining profile are also removed.
        """
        if isinstance(profiles, (str, Profile)):
            profiles = [profiles]
            
        context = get_context(self)
        for profile in profiles:
            if isinstance(profile, str):
                profile = self.profiles.get_by_id(profile)
            if profile in self.profiles:
                # Record reversal action if in a context.
                if context:
                    context(partial(self.profiles.append, profile))
                    context(partial(setattr, profile, "_model", self))
                    context(self._classify_profiles)
                self.profiles.remove(profile)
                profile._model = None
            else:
                logger.warning(f"Profile '{profile.id}' not found in the model.")
        self._classify_profiles()
        if remove_orphans:
            self._remove_orphaned_features()
        self._invalidate_cache()
        
    def _classify_profiles(self) -> None:
        """
        Classify profiles into genome and metagenome groups.
        
        Rebuilds the genome_profiles and metagenome_profiles lists from self.profiles.
        """
        self.genome_profiles = DictList()
        self.metagenome_profiles = DictList()
        for profile in self.profiles:
            profile._model = self
            if profile.profile_type == "genome":
                self.genome_profiles.append(profile)
            elif profile.profile_type == "metagenome":
                self.metagenome_profiles.append(profile)
        self._invalidate_cache()
    
    def add_features(self, features: Union[Feature, List[Feature]]) -> None:
        """
        Add one or more Feature objects to the model's global registry.
        
        Only features with valid, non-empty string identifiers that are not already
        present (by id) are added. For each added feature, its _model attribute is set to this model.
        
        Parameters
        ----------
        features : Feature or list of Feature
            The feature(s) to add.
        """
        if not isinstance(features, list):
            features = [features]
        features = [feat for feat in features if isinstance(feat.id, str) and feat.id]
        features_to_add = [feat for feat in features if feat.id not in [f.id for f in self.features]]
        for feat in features_to_add:
            feat._model = self
            self.features.append(feat)
        logger.debug(f"Added {len(features_to_add)} feature(s) to the model.")
        self._invalidate_cache()
        
    def remove_features(self, features: Union[Feature, List[Feature]]) -> None:
        """
        Remove one or more Feature objects from the model.
        
        The feature is removed from the global registry as well as from any profile that includes it.
        
        Parameters
        ----------
        features : Feature or list of Feature
            The feature(s) to remove.
        """
        if not isinstance(features, list):
            features = [features]
        features_to_remove = [feat for feat in features if feat.id in [f.id for f in self.features]]
        for feat in features_to_remove:
            for profile in self.profiles:
                if feat in profile._features:
                    profile.remove_feature(feat)
            self.features.remove(feat)
            feat._model = None
        logger.info(f"Removed {len(features_to_remove)} feature(s) from the model.")
        self._invalidate_cache()
        
    def _remove_orphaned_features(self) -> None:
        """
        Remove features that are not present in any profile.
        
        Checks the global feature registry and removes any feature that is absent from all profiles.
        """
        features_in_profiles = set()
        for profile in self.profiles:
            features_in_profiles.update(feat.id for feat in profile.features)
        features_to_remove = [feat for feat in self.features if feat.id not in features_in_profiles]
        for feat in features_to_remove:
            self.features.remove(feat)
            feat._model = None
        logger.info(f"Removed {len(features_to_remove)} orphan feature(s) from the model.")
        self._invalidate_cache()
    
    @property
    def genome_binary_matrix(self) -> np.ndarray:
        """
        Generate and cache the binary matrix for genome profiles.
        
        Rows correspond to features; columns to genome profiles.
        
        Returns
        -------
        np.ndarray
            A binary matrix of shape (num_features, num_genome_profiles).
        """
        if self._genome_binary_matrix_cache is not None:
            return self._genome_binary_matrix_cache
        
        n_features = len(self.features)
        n_candidates  = len(self.genome_profiles)
        
        
        feat_to_idx = {f: i for i, f in enumerate(self.features)}
        prof_to_idx = {p: j for j, p in enumerate(self.genome_profiles)}
        
        # Pre-allocate the binary matrix
        mat = np.zeros((n_features, n_candidates), dtype=int)
        
        rows, cols, data = [], [], []
        
        for profile in self.genome_profiles:
            j = prof_to_idx[profile]
            for feat, feat_dict in profile.features.items():
                i = feat_to_idx.get(feat)
                if i is not None:
                    presence = feat_dict.get("presence", 0)
                    if presence:
                        rows.append(i)
                        cols.append(j)
                        data.append(presence)
        
        
        mat[rows, cols] = data
                    
        self._genome_binary_matrix_cache = mat
        return mat
        
    @property
    def genome_abundance_matrix(self) -> np.ndarray:
        """
        Generate and cache the abundance matrix for genome profiles.
        
        Rows correspond to features; columns to genome profiles.
        Each value corresponds to the 'abundance' stored in the profile's feature data.
        
        Returns
        -------
        np.ndarray
        A matrix of shape (num_features, num_genome_profiles).
        """
        if self._genome_abundance_matrix_cache is not None:
            return self._genome_abundance_matrix_cache
        
        n_features = len(self.features)
        n_profiles = len(self.genome_profiles)
        
        feat_to_idx = {f: i for i, f in enumerate(self.features)}
        prof_to_idx = {p: j for j, p in enumerate(self.genome_profiles)}
        
        rows, cols, data = [], [], []
        
        for profile in self.genome_profiles:
            j = prof_to_idx[profile]
            for feat, feat_dict in profile.features.items():
                i = feat_to_idx.get(feat)
                if i is not None:
                    abundance = feat_dict.get("abundance", 0.0)
                    if abundance:
                        rows.append(i)
                        cols.append(j)
                        data.append(abundance)
        
        mat = np.zeros((n_features, n_profiles), dtype=float)
        mat[rows, cols] = data
        
        self._genome_abundance_matrix_cache = mat
        return mat
        
        
    @property
    def metagenome_binary_matrix(self) -> np.ndarray:
        """
        Generate and cache the binary matrix for metagenome profiles.
        
        Rows correspond to features; columns to metagenome profiles.
        
        Returns
        -------
        np.ndarray
            A binary matrix of shape (num_features, num_metagenome_profiles).
        """
        if self._metagenome_binary_matrix_cache is not None:
            return self._metagenome_binary_matrix_cache
        
        n_features = len(self.features)
        n_samples  = len(self.metagenome_profiles)
        
        
        feat_to_idx = {f: i for i, f in enumerate(self.features)}
        prof_to_idx = {p: j for j, p in enumerate(self.metagenome_profiles)}
        
        # Pre-allocate the binary matrix
        mat = np.zeros((n_features, n_samples), dtype=int)
        
        rows, cols, data = [], [], []
        
        for profile in self.metagenome_profiles:
            j = prof_to_idx[profile]
            for feat, feat_dict in profile.features.items():
                i = feat_to_idx.get(feat)
                if i is not None:
                    presence = feat_dict.get("presence", 0)
                    if presence:
                        rows.append(i)
                        cols.append(j)
                        data.append(presence)
            
        mat[rows, cols] = data
                    
        self._metagenome_binary_matrix_cache = mat
        return mat
        
    @property
    def metagenome_abundance_matrix(self) -> np.ndarray:
        """
        Generate and cache the abundance matrix for metagenome profiles.
        
        Rows correspond to features; columns to metagenome profiles.
        Each value corresponds to the 'abundance' stored in the profile's feature data.
        
        Returns
        -------
        np.ndarray
            A matrix of shape (num_features, num_metagenome_profiles).
        """
        if self._metagenome_abundance_matrix_cache is not None:
            return self._metagenome_abundance_matrix_cache
            
        n_features = len(self.features)
        n_samples = len(self.metagenome_profiles)
        
        feat_to_idx = {f: i for i, f in enumerate(self.features)}
        prof_to_idx = {p: j for j, p in enumerate(self.metagenome_profiles)}
        
        rows, cols, data = [], [], []
        
        for profile in self.metagenome_profiles:
            j = prof_to_idx[profile]
            for feat, feat_dict in profile.features.items():
                i = feat_to_idx.get(feat)
                if i is not None:
                    abundance = feat_dict.get("abundance", 0.0)
                    if abundance:
                        rows.append(i)
                        cols.append(j)
                        data.append(abundance)
                        
        mat = np.zeros((n_features, n_samples), dtype=float)
        mat[rows, cols] = data
        
        self._metagenome_abundance_matrix_cache = mat
        return mat
        
    @property
    def genome_names(self) -> List[str]:
        """
        Return the names of genome profiles.
        
        Returns
        -------
        List[str]
            A list of profile names.
        """
        return [profile.name for profile in self.genome_profiles]
        
    @property
    def genome_labels(self) -> List[Dict[str, Any]]:
        """
        Return taxonomy labels for genome profiles.
        
        Returns
        -------
        List[Dict[str, Any]]
            A list of taxonomy dictionaries.
        """
        return [profile.taxonomy for profile in self.genome_profiles]
        
    def get_genome_labels(self, taxonomic_levels: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Return the taxonomy labels for genome profiles, optionally filtered to specified levels.
        
        Parameters
        ----------
        taxonomic_levels : list of str, optional
            If provided, each profile's taxonomy is filtered to include only these levels.
            
        Returns
        -------
        List[Dict[str, Any]]
            A list of taxonomy dictionaries.
        """
        if taxonomic_levels is None:
            return self.genome_labels
        else:
            return [
                {level: profile.taxonomy.get(level) for level in taxonomic_levels if level in profile.taxonomy}
                for profile in self.genome_profiles
            ]
            
    # --- Context manager methods ---
    def __enter__(self) -> "Model":
        """
        Enter a context for reversible modifications.
        
        Creates a new HistoryManager and adds it to the context stack.
        
        Returns
        -------
        Model
            This model instance.
        """
        try:
            self._contexts.append(HistoryManager())
        except AttributeError:
            self._contexts = [HistoryManager()]
        return self
        
    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """
        Exit the context and revert any recorded changes.
        
        Pops the last HistoryManager from the stack and calls its reset() method.
        """
        context = self._contexts.pop()
        context.reset()
        self._invalidate_cache()
        
    def __repr__(self) -> str:
        """
        Return a string representation of the model.
        
        Returns
        -------
        str
            A summary including the model id and counts of genome and metagenome profiles.
        """
        return f"<Model {self.id}, genomes={len(self.genome_profiles)}, metagenomes={len(self.metagenome_profiles)}>"
        
    def __str__(self) -> str:
        """
        Return a user-friendly string representation of the model.
        
        Includes the model name or ID, and the number of genome profiles,
        metagenome profiles, and features.
        
        Returns
        -------
        str
            A human-readable summary of the model contents.
        """
        name = self.name or self.id or "Unnamed Model"
        return (
            f"Model '{name}'\n"
            f" - Genomes:     {len(self.genome_profiles)}\n"
            f" - Metagenomes: {len(self.metagenome_profiles)}\n"
            f" - Features:    {len(self.features)}"
        )