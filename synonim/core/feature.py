from typing import Optional
from .object import Object

class Feature(Object):
    """Represents a single feature in a profile."""
    
    def __init__(
        self,
        id: str,
        name: str = "",
        presence: int = 0,
        abundance: Optional[float] = None,
        **kwargs,
    ) -> None:
        """
        Initialize a Feature object.
        
        Parameters
        ----------
        id : str
            The unique identifier for the feature (e.g., PfamID).
        name : str, optional
            A human-readable name for the feature.
        presence : int, optional
            Presence (1) or absence (0) of the feature. Only 0 or 1 are allowed.
        abundance : float, optional
            Quantitative measure of the feature's abundance.
        **kwargs
            Additional attributes to set on the feature.
        """
        super().__init__(id=id, name=name)
        
        if presence not in (0, 1):
            raise ValueError("The 'presence' attribute must be 0 or 1.")
        self.presence: int = presence
        self.abundance: Optional[float] = abundance
        
        # Set any additional attributes from kwargs.
        for key, value in kwargs.items():
            setattr(self, key, value)
            
    def __repr__(self) -> str:
        """Return a detailed string representation of the Feature."""
        return (
            f"<Feature {self.id}, presence={self.presence}, abundance={self.abundance}>"
        )
    
    def __str__(self) -> str:
        """Return a user-friendly string representation of the Feature."""
        return f"Feature {self.id} ({self.name}): presence={self.presence}, abundance={self.abundance}"

# from typing import Any, Optional

# from .object import Object
# from .dictlist import DictList

# class Feature(Object):
    # """Represents a single feature in a profile."""
    
    # def __init__(
        # self,
        # id: str,
        # name: str = "",
        # presence: int = 0,
        # abundance: Optional[float] = None,
        # **kwargs,
    # ) -> None:
        # """Initialize a Feature object.
        
        # Parameters
        # ----------
        # id : str
            # The unique identifier for the feature (e.g., PfamID).
        # name : str, optional
            # A human-readable name for the feature.
        # presence : int, optional
            # Presence (1) or absence (0) of the feature (default is 0).
        # abundance : float, optional
            # Quantitative measure of the feature's abundance.
        # **kwargs
            # Additional attributes to set on the feature.
        # """
        # super().__init__(id=id, name=name)
        # self.presence = presence
        # self.abundance = abundance
        # # Set any additional attributes
        # for key, value in kwargs.items():
            # setattr(self, key, value)
            
    # def __repr__(self) -> str:
        # """Return a string representation of the Feature."""
        # return (
            # f"<Feature {self.id}, presence={self.presence}, "
            # f"abundance={self.abundance}>"
        # )
