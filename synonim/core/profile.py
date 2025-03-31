from typing import List, Optional
from .object import Object
from .feature import Feature
from .dictlist import DictList

class Profile(Object):
    """Represents a genome or metagenome profile."""
    
    def __init__(
        self,
        id: str,
        name: str,
        features: Optional[List[Feature]] = None,
        profile_type: str = "genome",
        taxonomy: Optional[dict] = None,
        metadata: Optional[dict] = None
    ) -> None:
        """Initialize a Profile object.
        
        Parameters
        ----------
        id : str
            The unique identifier for the profile.
        name : str
            A human-readable name for the profile.
        features : list of Feature, optional
            A list of Feature instances to include in the profile.
        profile_type : str, optional
            The type of profile ('genome' or 'metagenome').
        taxonomy : dict, optional
            A dictionary of taxonomic information (e.g., {"domain": "bacteria", "family": "Enterobacteriaceae"}).
        metadata : dict, optional
            Additional information that may be used by constraints or other parts of the system.
        """
        super().__init__(id=id, name=name)
        self.profile_type = profile_type  # renamed from self.type to avoid shadowing the built-in.
        self.features = DictList(features) if features else DictList()
        self.taxonomy = taxonomy if taxonomy is not None else {}
        self.metadata = metadata if metadata is not None else {}
    
    def add_feature(self, feature: Feature) -> None:
        """Add a feature to the profile.
        
        Parameters
        ----------
        feature : Feature
            The Feature instance to add.
        """
        self.features.append(feature)
    
    @property
    def presence_vector(self) -> List[int]:
        """Return a list of presence values for all features."""
        return [feature.presence for feature in self.features]
    
    @property
    def abundance_vector(self) -> List[Optional[float]]:
        """Return a list of abundance values for all features."""
        return [feature.abundance for feature in self.features]
    
    def __repr__(self) -> str:
        """Return a string representation of the Profile."""
        return (f"<Profile {self.id}, type={self.profile_type}, "
                f"features={len(self.features)}, taxonomy={self.taxonomy}>")
    
    def __str__(self) -> str:
        """Return a human-friendly string representation of the Profile."""
        return (f"Profile {self.name} (ID: {self.id}, Type: {self.profile_type}, "
                f"Taxonomy: {self.taxonomy})")

# from typing import List, Optional


# from .object import Object
# from .feature import Feature
# from .dictlist import DictList

# class Profile(Object):
    # """Represents a genome or metagenome profile."""

    # def __init__(
        # self,
        # id: str,
        # name: str,
        # features: Optional[List[Feature]] = None,
        # profile_type: str = "genome",
    # ) -> None:
        # """Initialize a Profile object.

        # Parameters
        # ----------
        # id : str
            # The unique identifier for the profile.
        # name : str
            # A human-readable name for the profile.
        # features : list of Feature, optional
            # A list of Feature instances to include in the profile.
        # profile_type : str, optional
            # The type of profile ('genome' or 'metagenome').
        # """
        # super().__init__(id=id, name=name)
        # self.type = profile_type
        # self.features = DictList(features) if features else DictList()

    # def add_feature(self, feature: Feature) -> None:
        # """Add a feature to the profile.

        # Parameters
        # ----------
        # feature : Feature
            # The Feature instance to add.
        # """
        # self.features.append(feature)

    # def get_presence_vector(self) -> List[int]:
        # """Get a list of presence values for all features."""
        # return [feature.presence for feature in self.features]

    # def get_abundance_vector(self) -> List[Optional[float]]:
        # """Get a list of abundance values for all features."""
        # return [feature.abundance for feature in self.features]

    # def __repr__(self) -> str:
        # """Return a string representation of the Profile."""
        # return (
            # f"<Profile {self.id}, type={self.type}, "
            # f"features={len(self.features)}>"
        # )
