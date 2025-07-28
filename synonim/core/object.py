# object.py

from typing import Optional, Dict


class Object:
    """
    Base class for objects in the synthetic community design system.
    
    Provides common properties such as id, name, annotations, and notes.
    Supports optional model association and serialization safety.
    """
    
    def __init__(self, id: Optional[str] = None, name: str = "") -> None:
        """
        Initialize a base Object.
        
        Parameters
        ----------
        id : str, optional
            A unique identifier for the object.
        name : str, optional
            A human-readable name for the object.
        """
        self._id = id
        self.name = name
        self.notes = {}
        self._annotation = {}
        self._model = None  # Optional reference to a model
        
    @property
    def id(self) -> Optional[str]:
        """
        Get the object's unique identifier.
        
        Returns
        -------
        str or None
            The unique identifier of the object, or None if not set.
        """
        return getattr(self, "_id", None)
        
    @id.setter
    def id(self, value: str) -> None:
        """
        Set the object's unique identifier.
        
        Parameters
        ----------
        value : str
            The new unique identifier.
            
        Raises
        ------
        TypeError
            If the value is not a string.
        """
        if value == self.id:
            return
        if not isinstance(value, str):
            raise TypeError("ID must be a string")
        if self._model is not None:
            self._set_id_with_model(value)
        else:
            self._id = value
            
    def _set_id_with_model(self, value: str) -> None:
        """
        Override this in subclasses to update the ID in the context of a model.
        
        Parameters
        ----------
        value : str
            New ID value.
        """
        self._id = value
        
    @property
    def annotation(self) -> dict:
        """
        Get or set the object's annotation dictionary.
        
        Returns
        -------
        dict
            The annotation dictionary.
        """
        return self._annotation
        
    @annotation.setter
    def annotation(self, value: dict) -> None:
        """
        Set the object's annotation dictionary.
        
        Parameters
        ----------
        value : dict
            A dictionary containing annotations.
            
        Raises
        ------
        TypeError
            If the value is not a dictionary.
        """
        if not isinstance(value, dict):
            raise TypeError("Annotation must be a dict")
        self._annotation = value
        
    def __getstate__(self) -> Dict:
        """
        Prepare object state for pickling.
        
        Removes non-serializable or circular references (e.g. _model).
        
        Returns
        -------
        dict
            Dictionary of state attributes.
        """
        state = self.__dict__.copy()
        # Do not pickle the model reference.
        state["_model"] = None
        
        return state
        
    def __repr__(self) -> str:
        """
        Return a detailed string representation of the object.
        
        Returns
        -------
        str
            Class name, ID, and memory address.
        """
        return f"<{self.__class__.__name__} {self.id} at {hex(id(self))}>"
        
    def __str__(self) -> str:
        """
        Return a simplified string representation.
        
        Returns
        -------
        str
            Object's ID as a string.
        """
        return str(self.id)