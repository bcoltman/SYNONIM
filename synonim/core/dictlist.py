import re
from typing import (
    Any,
    Callable,
    Iterable,
    Iterator,
    List,
    Optional,
    Pattern,
    Tuple,
    Type,
    Union,
)

from .object import Object


class DictList(list):
    """
    Combined list and dict: maintains list behavior with O(1) lookup by item.id.
    """
    
    def __init__(self, *args):
        # Always start with a clean list and index
        super().__init__()
        self._dict: dict = {}
        # If initialized from a single iterable, populate
        if len(args) == 1:
            other = args[0]
            if isinstance(other, DictList):
                # fast copy existing list and index
                list.extend(self, other)
                self._dict = other._dict.copy()
            else:
                for obj in other:
                    self.append(obj)
                    
    def __reduce__(self) -> Tuple[Type["DictList"], Tuple[List[Object]]]:
        # Pickle as a plain Python list; __init__ will rebuild index
        return (self.__class__, (list(self),))
        
    def _generate_index(self) -> None:
        """Rebuild the internal id->position mapping."""
        self._dict = {v.id: i for i, v in enumerate(self)}
        
    def has_id(self, id: Union[Object, str]) -> bool:
        """Check whether an id or Object is in the list."""
        lookup = id.id if isinstance(id, Object) else id
        return lookup in self._dict
        
    def _check(self, id: Union[Object, str]) -> None:
        """Ensure no duplicate ids are added."""
        if id in self._dict:
            raise ValueError(f"id '{id}' is already present in the DictList")
            
    def get_by_id(self, id: Union[Object, str]) -> Object:
        """Retrieve an item by its id."""
        return list.__getitem__(self, self._dict[id])
        
    def list_attr(self, attribute: str) -> List[Any]:
        """Return a list of the given attribute for every element."""
        return [getattr(item, attribute) for item in self]
        
    def query(
        self,
        search_function: Union[str, Pattern, Callable],
        attribute: Optional[str] = None,
    ) -> "DictList":
        """
        Return a new DictList of items matching the search criteria.
        Supports regex strings, compiled patterns, or callables.
        """
        def select(x: Any) -> Any:
            return getattr(x, attribute) if attribute else x
            
        try:
            regex = re.compile(search_function)
            matches = (i for i in self if regex.search(str(select(i))))
        except TypeError:
            matches = (i for i in self if search_function(select(i)))
            
        results = self.__class__()
        results._extend_nocheck(matches)
        return results
        
    def _extend_nocheck(self, iterable: Iterable[Object]) -> None:
        """Internal extend without uniqueness checks."""
        start = len(self)
        list.extend(self, iterable)
        if start == 0:
            self._generate_index()
            return
        for idx, obj in enumerate(self[start:], start):
            self._dict[obj.id] = idx
            
    def append(self, obj: Object) -> None:
        """Append an object, updating the index."""
        self._check(obj.id)
        self._dict[obj.id] = len(self)
        list.append(self, obj)
        
    def extend(self, iterable: Iterable[Object]) -> None:
        """Extend by appending each object, updating the index."""
        for obj in iterable:
            self.append(obj)
            
    def __contains__(self, entity: Union[str, Object]) -> bool:
        """Membership test by id or object."""
        if isinstance(entity, Object):
            key = entity.id
        else:
            key = entity
        return key in self._dict
        
    def __getitem__(
        self,
        index: Union[int, slice, str]
    ) -> Union[Object, "DictList"]:
        """Index by position, slice, or id."""
        if isinstance(index, (int, slice)):
            result = list.__getitem__(self, index)
            if isinstance(result, list):
                new = self.__class__()
                new.extend(result)
                return new
            return result
        if isinstance(index, str):
            return self.get_by_id(index)
        raise TypeError("Index must be int, slice, or str")
        
    def __delitem__(self, index: Union[int, slice]) -> None:
        """Delete item(s) and rebuild index."""
        if isinstance(index, int):
            removed = self[index]
            list.__delitem__(self, index)
            del self._dict[removed.id]
        else:
            removed = self[index]
            list.__delitem__(self, index)
            for item in removed:
                del self._dict[item.id]
        self._generate_index()
        
    def __setitem__(self, index: Union[int, slice], value: Object) -> None:
        """Set single item, updating the index."""
        if isinstance(index, int):
            old = self[index]
            del self._dict[old.id]
            self._check(value.id)
            list.__setitem__(self, index, value)
            self._dict[value.id] = index
        else:
            raise NotImplementedError("Setting slices not supported.")
            
    def __repr__(self) -> str:
        return f"DictList({list.__repr__(self)})"
        
    def __getattr__(self, attr: Any) -> Any:
        """Allow lookup by id as attribute, ignore magic names."""
        if not (attr.startswith("__") and attr.endswith("__")) and attr in self._dict:
            return self.get_by_id(attr)
        raise AttributeError(f"{self.__class__.__name__!r} has no attribute or entry {attr!r}")