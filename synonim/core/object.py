from typing import Optional

class Object:
    """Defines common behavior of objects in the syncom design system."""
    
    def __init__(self, id: Optional[str] = None, name: str = "") -> None:
        """Initialize an object with an identifier and a name.

        Parameters
        ----------
        id : str, optional
            The unique identifier for the object.
        name : str, optional
            A human-readable name for the object.
        """
        self._id = id
        self.name = name
        self.notes = {}
        self._annotation = {}
    
    @property
    def id(self) -> str:
        """Get or set the object's unique identifier."""
        return self._id

    @id.setter
    def id(self, value: str) -> None:
        if not isinstance(value, str):
            raise TypeError("ID must be a string")
        self._id = value

    @property
    def annotation(self) -> dict:
        """Get or set the object's annotation dictionary."""
        return self._annotation

    @annotation.setter
    def annotation(self, value: dict) -> None:
        if not isinstance(value, dict):
            raise TypeError("Annotation must be a dict")
        self._annotation = value

    def __repr__(self) -> str:
        """Return a string representation of the object."""
        return f"<{self.__class__.__name__} {self.id} at {hex(id(self))}>"

    def __str__(self) -> str:
        """Return a string representation of the object's ID."""
        return str(self.id)
