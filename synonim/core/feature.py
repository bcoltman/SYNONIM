# feature.py

from typing import Optional, Set, TYPE_CHECKING, FrozenSet, Dict
from .object import Object

if TYPE_CHECKING:
    from .profile import Profile  # for type checking


class Feature(Object):
    """
    Represents a canonical feature shared among profiles.
    
    Features have immutable IDs, and track the profiles in which they appear.
    """
    
    def __init__(self, id: str, name: str = "", **kwargs) -> None:
        """
        Initialize a Feature.
        
        Parameters
        ----------
        id : str
            The unique identifier for the feature.
        name : str, optional
            A human-readable name.
        **kwargs
            Additional attributes to attach dynamically.
        """
        super().__init__(id=id, name=name)
        
        self._model = None
        
        self._profiles: Set["Profile"] = set()
        
        # Dynamically add additional attributes
        for key, value in kwargs.items():
            setattr(self, key, value)
            
    def __setattr__(self, name, value):
        """
        Enforce immutability of `_id` after initialization.
        
        Raises
        ------
        AttributeError
            If attempting to change `_id` after it's been set.
        """
        if name == "_id" and hasattr(self, "_id"):
            raise AttributeError("Feature id is immutable and cannot be modified after initialization.")
        super().__setattr__(name, value)
        
    def __repr__(self) -> str:
        """
        Return a concise debug representation.
        
        Returns
        -------
        str
            Feature class, ID, and name.
        """
        return f"<Feature {self.id}, name={self.name}>"
        
    def __str__(self) -> str:
        """
        Return a user-friendly string representation.
        
        Returns
        -------
        str
            Feature description with ID and name.
        """
        return f"Feature {self.id} ({self.name})"
        
    def __hash__(self) -> int:
        """
        Hash based on feature ID.
        
        Returns
        -------
        int
            Hash of the feature's ID.
        """
        return hash(self.id)
        
    def __eq__(self, other: object) -> bool:
        """
        Equality comparison based on ID.
        
        Parameters
        ----------
        other : object
            Another object.
            
        Returns
        -------
        bool
            True if other is a Feature with the same ID.
        """
        return isinstance(other, Feature) and self.id == other.id
        
    @property
    def profiles(self) -> FrozenSet:
        """
        Get the set of profiles that reference this feature.
        
        Returns
        -------
        FrozenSet
            Immutable set of Profile objects.
        """
        return frozenset(self._profiles)
        
    @property
    def model(self) -> Optional["Model"]:
        """
        Retrieve the model associated with this object.
        
        Returns
        -------
        Model or None
            The model instance that the object is associated with, or None if not set.
        """
        return self._model
    
    def __getstate__(self) -> Dict:
        """
        Prepare the Feature for pickling.
        
        Removes the _profiles set to avoid circular references.
        
        Returns
        -------
        dict
            The state dictionary.
        """
        # state = self.__dict__.copy()
        state = Object.__getstate__(self)
        # Clear _profiles to prevent circular references during pickling.
        state["_profiles"] = set()
        return state
        
    def __setstate__(self, state: Dict) -> None:
        """
        Restore the Feature state from a pickled state.
        
        Ensures _profiles is reinitialized.
        
        Parameters
        ----------
        state : dict
            The state dictionary.
        """
        # Ensure _profiles is a clean empty set upon unpickling.
        state["_profiles"] = set()
        self.__dict__.update(state)
