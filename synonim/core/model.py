import logging
from typing import Any, Dict, Iterable, List, Optional, Union
from functools import partial

from .object import Object
from .dictlist import DictList
from .feature import Feature
from .profile import Profile

# Assume that Object and DictList have been defined as before

logger = logging.getLogger(__name__)

class Model(Object):
    """Represents the main model containing profiles and optimization logic."""
    
    def __init__(
        self,
        id_or_model: Union[str, "Model", None] = None,
        name: Optional[str] = None,
        profiles: Optional[List["Profile"]] = None,
    ) -> None:
        """Initialize the Model object.
        
        Parameters
        ----------
        id_or_model : str or Model, optional
            String to use as model id, or actual model to base new model on.
            If string, it is used as id. If model, a new model object is
            instantiated with the same properties as the original model.
        name : str, optional
            A human-readable name for the model.
        profiles : list of Profile, optional
            A list of Profile instances to include in the model.
        """
        if isinstance(id_or_model, Model):
            Object.__init__(self, name=name)
            self.__setstate__(id_or_model.__dict__)
            self._optimizer = id_or_model.optimizer
        else:
            super().__init__(id=id_or_model, name=name)
            self.profiles = DictList(profiles) if profiles else DictList()
            self.genome_profiles = DictList()
            self.metagenome_profiles = DictList()
            self._classify_profiles()
            self._contexts = []
            self._optimizer = None  # Placeholder for an optimizer instance
            
    def __setstate__(self, state: Dict) -> None:
        """Set the state of the model from a state dictionary.
        
        Parameters
        ----------
        state : dict
            A dictionary representing the state to set.
        """
        self.__dict__.update(state)
        for profile in self.profiles:
            profile._model = self
            
    def __getstate__(self) -> Dict:
        """Get the state of the model for serialization.
        
        Returns
        -------
        dict
            A dictionary representing the current state of the model.
        """
        odict = self.__dict__.copy()
        odict["_contexts"] = []
        return odict
        
    def _classify_profiles(self) -> None:
        """Classify profiles into genomes and metagenomes."""
        for profile in self.profiles:
            profile._model = self
            if profile.type == "genome":
                self.genome_profiles.append(profile)
            elif profile.type == "metagenome":
                self.metagenome_profiles.append(profile)
                
    def add_profiles(self, profiles: Iterable["Profile"]) -> None:
        """Add profiles to the model.
        
        Parameters
        ----------
        profiles : iterable of Profile
            An iterable of Profile instances to add to the model.
        """
        for profile in profiles:
            self.add_profile(profile)
            
    def add_profile(self, profile: "Profile") -> None:
        """Add a single profile to the model.
        
        Parameters
        ----------
        profile : Profile
            The Profile instance to add.
        """
        if profile.id in self.profiles:
            logger.warning(f"Ignoring profile '{profile.id}' since it already exists.")
            return
        profile._model = self
        self.profiles.append(profile)
        if profile.type == "genome":
            self.genome_profiles.append(profile)
        elif profile.type == "metagenome":
            self.metagenome_profiles.append(profile)
            
    def remove_profiles(
        self,
        profiles: Union[str, "Profile", Iterable[Union[str, "Profile"]]],
        remove_orphans: bool = False,
    ) -> None:
        """Remove profiles from the model.
        
        Parameters
        ----------
        profiles : str, Profile, or iterable of str or Profile
            Profiles to remove from the model.
        remove_orphans : bool, optional
            Remove features that are no longer associated with any profiles (default False).
        """
        if isinstance(profiles, (str, Profile)):
            profiles = [profiles]
        for profile in profiles:
            if isinstance(profile, str):
                profile = self.profiles.get_by_id(profile)
            if profile in self.profiles:
                self.profiles.remove(profile)
                profile._model = None
                if profile.type == "genome":
                    self.genome_profiles.remove(profile)
                elif profile.type == "metagenome":
                    self.metagenome_profiles.remove(profile)
                # Optionally remove orphaned features
                if remove_orphans:
                    self._remove_orphaned_features()
            else:
                logger.warning(f"Profile '{profile.id}' not found in the model.")
                
    def _remove_orphaned_features(self) -> None:
        """Remove features that are no longer associated with any profiles."""
        feature_usage = {}
        for profile in self.profiles:
            for feature in profile.features:
                feature_usage[feature.id] = feature_usage.get(feature.id, 0) + 1
        for profile in self.profiles:
            for feature in profile.features:
                if feature_usage.get(feature.id, 0) == 0:
                    profile.features.remove(feature)
                    
    def copy(self) -> "Model":
        """Create a deep copy of the model.
        
        Returns
        -------
        Model
            A new instance of the Model class with copied data.
        """
        new_model = self.__class__()
        do_not_copy = {
            "profiles",
            "genome_profiles",
            "metagenome_profiles",
            "_contexts",
            "_optimizer",
        }
        for attr in self.__dict__:
            if attr not in do_not_copy:
                new_model.__dict__[attr] = self.__dict__[attr]
        new_model.profiles = DictList()
        for profile in self.profiles:
            new_profile = profile.copy()
            new_profile._model = new_model
            new_model.profiles.append(new_profile)
        new_model.genome_profiles = DictList()
        new_model.metagenome_profiles = DictList()
        new_model._classify_profiles()
        # Copy optimizer if necessary
        if self._optimizer is not None:
            new_model._optimizer = deepcopy(self._optimizer)
        return new_model
        
    def optimize(self, **kwargs) -> Any:
        """Optimize the model using the assigned optimizer.
        
        Parameters
        ----------
        **kwargs
            Additional keyword arguments for the optimizer.
            
        Returns
        -------
        Any
            The result of the optimization.
            
        Raises
        ------
        ValueError
            If no optimizer is set for the model.
        """
        if self._optimizer is None:
            raise ValueError("No optimizer is set for the model.")
        return self._optimizer.optimize(model=self, **kwargs)
        
    def set_optimizer(self, optimizer) -> None:
        """Set the optimizer for the model.
        
        Parameters
        ----------
        optimizer : Any
            An optimizer instance that can work with the model.
        """
        self._optimizer = optimizer
        
    def __enter__(self) -> "Model":
        """Enter a context, recording changes to the model.
        
        Returns
        -------
        Model
            The model instance with context management enabled.
        """
        self._contexts.append(HistoryManager())
        return self
        
    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """Exit the context, reverting changes if necessary."""
        context = self._contexts.pop()
        context.reset()
        
    def __repr__(self) -> str:
        """Return a string representation of the Model."""
        return (
            f"<Model {self.id}, genomes={len(self.genome_profiles)}, "
            f"metagenomes={len(self.metagenome_profiles)}>"
        )



# class Model(Object):
    # """Represents the main model containing profiles and optimization logic."""
    
    # def __init__(
        # self,
        # id: str,
        # name: str = "",
        # profiles: Optional[List[Profile]] = None,
    # ) -> None:
        # """Initialize the Model object.
        
        # Parameters
        # ----------
        # id : str
            # The unique identifier for the model.
        # name : str, optional
            # A human-readable name for the model.
        # profiles : list of Profile, optional
            # A list of Profile instances to include in the model.
        # """
        # super().__init__(id=id, name=name)
        # self.profiles = DictList(profiles) if profiles else DictList()
        # self.genome_profiles = DictList()
        # self.metagenome_profiles = DictList()
        # self._classify_profiles()
        
    # def _classify_profiles(self) -> None:
        # """Classify profiles into genomes and metagenomes."""
        # for profile in self.profiles:
            # profile._model = self
            # if profile.type == "genome":
                # self.genome_profiles.append(profile)
            # elif profile.type == "metagenome":
                # self.metagenome_profiles.append(profile)
                
    # def add_profiles(self, profiles: Iterable["Profile"]) -> None:
        # """Add profiles to the model.
        
        # Parameters
        # ----------
        # profiles : iterable of Profile
            # An iterable of Profile instances to add to the model.
        # """
        # for profile in profiles:
            # self.add_profile(profile)
            
    # def add_profile(self, profile: "Profile") -> None:
        # """Add a single profile to the model.
        
        # Parameters
        # ----------
        # profile : Profile
            # The Profile instance to add.
        # """
        # if profile.id in self.profiles:
            # logger.warning(f"Ignoring profile '{profile.id}' since it already exists.")
            # return
        # profile._model = self
        # self.profiles.append(profile)
        # if profile.type == "genome":
            # self.genome_profiles.append(profile)
        # elif profile.type == "metagenome":
            # self.metagenome_profiles.append(profile)
            
    # def remove_profiles(
        # self,
        # profiles: Union[str, "Profile", Iterable[Union[str, "Profile"]]],
        # remove_orphans: bool = False,
    # ) -> None:
        # """Remove profiles from the model.
        
        # Parameters
        # ----------
        # profiles : str, Profile, or iterable of str or Profile
            # Profiles to remove from the model.
        # remove_orphans : bool, optional
            # Remove features that are no longer associated with any profiles (default False).
        # """
        # if isinstance(profiles, (str, Profile)):
            # profiles = [profiles]
        # for profile in profiles:
            # if isinstance(profile, str):
                # profile = self.profiles.get_by_id(profile)
            # if profile in self.profiles:
                # self.profiles.remove(profile)
                # profile._model = None
                # if profile.type == "genome":
                    # self.genome_profiles.remove(profile)
                # elif profile.type == "metagenome":
                    # self.metagenome_profiles.remove(profile)
                # # Optionally remove orphaned features
                # if remove_orphans:
                    # self._remove_orphaned_features()
            # else:
                # logger.warning(f"Profile '{profile.id}' not found in the model.")
            
    # def optimize(self, optimizer) -> Any:
        # """Run optimization using the provided optimizer.
        
        # Parameters
        # ----------
        # optimizer : Any
            # An optimizer instance that has an 'optimize' method.
            
        # Returns
        # -------
        # Any
            # The result of the optimization.
        # """
        # return optimizer.optimize()
        
    # def analyze_results(self, results) -> None:
        # """Analyze the optimization results.
        
        # Parameters
        # ----------
        # results : Any
            # The results from the optimization.
        # """
        # # Implement analysis logic here
        # pass
        
    # def __repr__(self) -> str:
        # """Return a string representation of the Model."""
        # return (
            # f"<Model {self.id}, genomes={len(self.genome_profiles)}, "
            # f"metagenomes={len(self.metagenome_profiles)}>"
        # )
