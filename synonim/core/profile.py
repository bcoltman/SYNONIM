from typing import Dict, Optional, List, Union
from functools import partial
from .object import Object
from .feature import Feature
from ..util.context import HistoryManager, get_context, resettable

class Profile(Object):
    """
    Represents a genome or metagenome profile with per-feature data.
    
    All feature data is stored in a dictionary (_features) where keys are Feature objects
    and values are dictionaries containing:
        - "presence": int (typically 0 or 1)
        - "abundance": Optional[float] (None indicates no abundance data)
        
    This design allows merging of feature data and supports operations such as scaling.
    Reversible (context-based) modifications are supported: for instance, modifications to
    the taxonomy, metadata, or feature data can be reversed upon exit from a context.
    """
    
    def __init__(
        self,
        id: str,
        name: str,
        profile_type: str = "genome",
        taxonomy: Optional[dict] = None,
        metadata: Optional[dict] = None,
    ) -> None:
        """
        Initialize a Profile.
        
        Parameters
        ----------
        id : str
            Unique identifier for the profile.
        name : str
            Human-readable name for the profile.
        profile_type : str, optional
            Type of profile (e.g., 'genome' or 'metagenome'). Default is 'genome'.
        taxonomy : dict, optional
            Taxonomic classification information.
        metadata : dict, optional
            Any additional metadata.
        """
        super().__init__(id=id, name=name)
        self._model = None
        self.profile_type = profile_type
        # Store taxonomy and metadata in private attributes so that their setters can be reversible.
        self._taxonomy = taxonomy if taxonomy is not None else {}
        self._metadata = metadata if metadata is not None else {}
        # Each feature maps to a dict with keys "presence" (int) and "abundance" (Optional[float])
        self._features: Dict[Feature, Dict[str, Union[int, Optional[float]]]] = {}
        
    @property
    def taxonomy(self) -> dict:
        """Return the taxonomy information for the profile."""
        return self._taxonomy
        
    @taxonomy.setter
    @resettable
    def taxonomy(self, new_taxonomy: dict) -> None:
        """
        Set the taxonomy in a reversible manner.
        
        Parameters
        ----------
        new_taxonomy : dict
            The new taxonomy information.
        """
        self._taxonomy = new_taxonomy
        
    @property
    def metadata(self) -> dict:
        """Return the metadata associated with the profile."""
        return self._metadata
        
    @metadata.setter
    @resettable
    def metadata(self, new_metadata: dict) -> None:
        """
        Set the metadata in a reversible manner.
        
        Parameters
        ----------
        new_metadata : dict
            The new metadata.
        """
        self._metadata = new_metadata
        
    def add_features(
        self, 
        features_to_add: Dict[Union[Feature, str], Dict[str, Union[int, float, None]]],
        combine: bool = True,
        reversibly: bool = True
    ) -> None:
        """
        Add or update feature data for the profile.
        
        Each entry in the input dictionary maps a feature (or feature id) to a dictionary
        containing (optionally) "presence" and/or "abundance" values. When provided as a string,
        the feature is looked up in the profile.
        
        When `reversibly` is True, the entire state of the feature mapping is saved and a reversal
        action is registered with the active HistoryManager.
        
        Parameters
        ----------
        features_to_add : dict
            Dictionary mapping Feature objects (or feature ids) to dictionaries with keys
            "presence" and/or "abundance". Missing keys default to 0 for presence and None for abundance.
        combine : bool, optional
            If True and the feature already exists, then:
              - "presence" is combined using logical OR.
              - "abundance" values are summed, treating None as 0.
            If False, the new values replace any existing data.
        reversibly : bool, optional
            Whether the change should be made reversible via an active HistoryManager. Default is True.
        """
        # Save current state if reversible modifications are desired.
        if reversibly:
            old_state = {feat: data.copy() for feat, data in self._features.items()}
        
        for feature_key, props in features_to_add.items():
            # Resolve feature key to a Feature object.
            if isinstance(feature_key, str):
                found_feature = None
                for f in self._features.keys():
                    if f.id == feature_key:
                        found_feature = f
                        break
                if found_feature is None:
                    raise KeyError(f"Feature with id '{feature_key}' not found in profile.")
                feature = found_feature
            elif isinstance(feature_key, Feature):
                feature = feature_key
            else:
                raise TypeError("Key must be a Feature object or a feature id (str).")
                
            new_presence = int(props.get("presence", 0))
            # If abundance is not provided, default to None.
            new_abundance_raw = props.get("abundance", None)
            if new_abundance_raw is None:
                new_abundance = None
            else:
                try:
                    new_abundance = float(new_abundance_raw)
                except (TypeError, ValueError):
                    new_abundance = None
                
            if feature in self._features:
                if combine:
                    current_presence = self._features[feature]["presence"]
                    # Use 0.0 when current abundance or new abundance is None.
                    current_abundance = self._features[feature].get("abundance")
                    current_abundance_val = current_abundance if current_abundance is not None else 0.0
                    new_abundance_val = new_abundance if new_abundance is not None else 0.0
                    combined_presence = 1 if (current_presence or new_presence) else 0
                    combined_abundance = current_abundance_val + new_abundance_val
                    self._features[feature]["presence"] = combined_presence
                    # If both abundance values are None, keep it as None.
                    if current_abundance is None and new_abundance is None:
                        self._features[feature]["abundance"] = None
                    else:
                        self._features[feature]["abundance"] = combined_abundance
                else:
                    self._features[feature] = {"presence": new_presence, "abundance": new_abundance}
            else:
                self._features[feature] = {"presence": new_presence, "abundance": new_abundance}
                # Link profile to the feature.
                feature._profiles.add(self)
        
        # Register a reversal action if within an active context.
        context = get_context(self)
        if context and reversibly:
            context(partial(setattr, self, "_features", old_state))
    
    def remove_feature(self, feature: Union[Feature, str], reversibly: bool = True) -> None:
        """
        Remove a feature from the profile in a reversible way.
        
        If an active HistoryManager context is present and reversibly is True,
        the current data associated with the feature is recorded so that the removal
        can be reverted upon context exit.
        
        Parameters
        ----------
        feature : Feature or str
            The Feature object or its id (str) to remove.
        reversibly : bool, optional
            Whether the removal should be reversible (default True).
            
        Raises
        ------
        KeyError
            If the specified feature is not found in the profile.
        """
        # Resolve feature key to a Feature object.
        if isinstance(feature, str):
            found_feature = None
            for f in list(self._features.keys()):
                if f.id == feature:
                    found_feature = f
                    break
            if found_feature is None:
                raise KeyError(f"Feature with id '{feature}' not found in profile.")
            feature = found_feature
            
        # If the feature exists and reversible removal is desired, record the reversal.
        if feature in self._features and reversibly:
            context = get_context(self)
            if context:
                # Save the current data (make a copy of the inner dictionary)
                old_data = self._features[feature].copy()
                # Register the reversal so that the feature is re-added with the saved data.
                # Use combine=False to replace any current state upon reversal, and set reversibly=False to avoid recursion.
                context(partial(self.add_features, {feature: old_data}, combine=False, reversibly=False))
                
        # Proceed with the removal.
        if feature in self._features:
            del self._features[feature]
            feature._profiles.discard(self)
    
    @property
    def model(self) -> Optional["Model"]:
        """
        Retrieve the model associated with this profile.
        
        Returns
        -------
        Model or None
            The model instance, or None if not set.
        """
        return self._model
    
    @property
    def features(self) -> Dict[Feature, Dict[str, Union[int, Optional[float]]]]:
        """
        Retrieve all features associated with the profile.
        
        Returns
        -------
        dict
            A copy of the feature mapping.
        """
        return self._features.copy()
        
    @property
    def presence_vector(self) -> List[int]:
        """
        Get the presence vector for the profile.
        
        Returns
        -------
        list of int
            A list with the presence (0 or 1) of each feature,
            using the canonical order from model.features if available,
            otherwise the order in the profile.
        """
        if self._model is not None:
            canonical_features = self._model.features
        else:
            canonical_features = list(self._features.keys())
        return [self._features.get(feat, {"presence": 0})["presence"] for feat in canonical_features]
        
    @property
    def abundance_vector(self) -> List[float]:
        """
        Get the abundance vector for the profile.
        
        Returns
        -------
        list of float
            A list with the abundance values of each feature,
            using the canonical order from model.features if available.
            Missing abundance data (None) is converted to 0.0.
        """
        if self._model is not None:
            canonical_features = self._model.features
        else:
            canonical_features = list(self._features.keys())
        vector = []
        for feat in canonical_features:
            # Default to 0.0 if feature data is missing or abundance is None.
            abundance = self._features.get(feat, {"abundance": 0.0})["abundance"]
            vector.append(0.0 if abundance is None else abundance)
        return vector
        
    def __add__(self, other: "Profile") -> "Profile":
        """
        Combine two profiles by merging their feature data.
        
        For features present in both profiles, "presence" is combined using logical OR
        and "abundance" values are summed (treating None as 0).
        
        Parameters
        ----------
        other : Profile
            The other profile to merge.
            
        Returns
        -------
        Profile
            A new Profile instance with merged feature data.
        """
        new_profile = self.copy()
        for feature, data in other._features.items():
            new_data = {"presence": data["presence"], "abundance": data["abundance"]}
            new_profile.add_features({feature: new_data}, combine=True, reversibly=False)
        return new_profile
        
    def __iadd__(self, other: "Profile") -> "Profile":
        """
        Merge another profile into this one in place.
        
        Parameters
        ----------
        other : Profile
            The profile to merge from.
            
        Returns
        -------
        Profile
            Self, with updated feature data.
        """
        combined = self + other
        self._features = combined._features
        return self
        
    def __mul__(self, coefficient: float) -> "Profile":
        """
        Generate a new profile with feature abundances scaled by a coefficient.
        
        Parameters
        ----------
        coefficient : float
            Factor by which to scale the abundance values.
            
        Returns
        -------
        Profile
            A new Profile with scaled feature abundances.
        """
        new_profile = self.copy()
        for feature in new_profile._features:
            current_abundance = new_profile._features[feature].get("abundance")
            if current_abundance is not None:
                new_profile._features[feature]["abundance"] = current_abundance * coefficient
        return new_profile
        
    def __imul__(self, coefficient: float) -> "Profile":
        """
        Scale feature abundances in the profile in place.
        
        Parameters
        ----------
        coefficient : float
            Factor to scale the abundance values.
            
        Returns
        -------
        Profile
            Self, with scaled feature abundances.
        """
        for feature in self._features:
            current_abundance = self._features[feature].get("abundance")
            if current_abundance is not None:
                self._features[feature]["abundance"] = current_abundance * coefficient
        
        context = get_context(self)
        if context:
            context(partial(self.__imul__, 1.0 / coefficient))
            
        return self
        
    def copy(self) -> "Profile":
      """
      Create a deep copy of the profile with independent feature data.
      
      Feature objects remain shared; inner dictionaries are duplicated.
      
      Returns
      -------
      Profile
          A deep copy of the current profile.
      """
      new_profile = Profile(
          id=self.id,
          name=self.name,
          profile_type=self.profile_type,
          taxonomy=self.taxonomy.copy(),
          metadata=self.metadata.copy(),
      )
      new_profile._features = {feat: data.copy() for feat, data in self._features.items()}
      for feature in new_profile._features.keys():
          feature._profiles.add(new_profile)
      return new_profile
        
    def __repr__(self) -> str:
        """
        Return a detailed string representation of the profile.
        
        Returns
        -------
        str
            A string including the profile id, type, number of features, and taxonomy.
        """
        return (f"<Profile {self.id}, type={self.profile_type}, "
                f"features={len(self.features)}, taxonomy={self.taxonomy}>")
                
    def __str__(self) -> str:
        """
        Return a user-friendly string representation of the profile.
        
        Returns
        -------
        str
            A summary that includes the profile id, name, type, and number of features.
        """
        return (f"Profile {self.name} (ID: {self.id}, Type: {self.profile_type}, "
                f"Features: {len(self.features)}")
                
    def __setstate__(self, state: Dict) -> None:
        """
        Restore the profile from a pickled state.
        
        Re-establishes the relationship so that each contained Feature's _profiles set
        includes this profile.
        
        Parameters
        ----------
        state : dict
            The state dictionary.
        """
        self.__dict__.update(state)
        # for feature in self._features.keys():
        for feature in state["_features"]:
            feature._model = self._model
            feature._profiles.add(self)
    
    def __getstate__(self) -> Dict:
        """Get state for reaction.
        
        This serializes the reaction object. The GPR will be converted to a string
        to avoid unneccessary copies due to interdependencies of used objects.
        
        Returns
        -------
        dict
            The state/attributes of the reaction in serilized form.
            
        """
        state = self.__dict__.copy()
        return state
