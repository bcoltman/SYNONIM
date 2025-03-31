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
    Define a combined dict and list.

    This object behaves like a list, but has the O(1) speed
    benefits of a dict when looking up elements by their id.
    """

    def __init__(self, *args):
        """Instantiate a combined dict and list.

        Parameters
        ----------
        args : iterable
            Iterable as a single argument to create a new DictList from.
        """
        super(DictList, self).__init__()
        self._dict = {}
        if len(args) == 1:
            other = args[0]
            if isinstance(other, DictList):
                list.extend(self, other)
                self._dict = other._dict.copy()
            else:
                self.extend(other)

    def has_id(self, id: Union[Object, str]) -> bool:
        """Check if an id is in the DictList."""
        return id in self._dict

    def _check(self, id: Union[Object, str]) -> None:
        """Ensure that duplicate ids are not added."""
        if id in self._dict:
            raise ValueError(f"id '{str(id)}' is already present in the DictList")

    def _generate_index(self) -> None:
        """Rebuild the _dict index."""
        self._dict = {v.id: k for k, v in enumerate(self)}

    def get_by_id(self, id: Union[Object, str]) -> Object:
        """Return the element with a matching id."""
        return list.__getitem__(self, self._dict[id])

    def list_attr(self, attribute: str) -> list:
        """Return a list of the given attribute for every object."""
        return [getattr(i, attribute) for i in self]

    def query(
        self,
        search_function: Union[str, Pattern, Callable],
        attribute: Optional[str] = None,
    ) -> "DictList":
        """Query the list.

        Parameters
        ----------
        search_function : str, Pattern, or Callable
            Used to find matching elements in the list.
        attribute : str, optional
            The attribute of the objects to compare against.

        Returns
        -------
        DictList
            A new DictList of objects that match the query.
        """
        def select_attribute(x: Any) -> Any:
            return getattr(x, attribute) if attribute else x

        try:
            # If the search_function is a regular expression
            regex_searcher = re.compile(search_function)
            matches = (
                i for i in self if regex_searcher.search(str(select_attribute(i)))
            )
        except TypeError:
            matches = (i for i in self if search_function(select_attribute(i)))

        results = self.__class__()
        results._extend_nocheck(matches)
        return results

    def _extend_nocheck(self, iterable: Iterable[Object]) -> None:
        """Extend without checking for uniqueness."""
        current_length = len(self)
        list.extend(self, iterable)
        _dict = self._dict
        if not current_length:
            self._generate_index()
            return
        for i, obj in enumerate(self[current_length:], current_length):
            _dict[obj.id] = i

    def append(self, obj: Object) -> None:
        """Append an object to the end of the DictList."""
        self._check(obj.id)
        self._dict[obj.id] = len(self)
        list.append(self, obj)

    def extend(self, iterable: Iterable[Object]) -> None:
        """Extend the DictList by appending elements from the iterable."""
        for obj in iterable:
            self.append(obj)

    def __contains__(self, entity: Union[str, Object]) -> bool:
        """Check if the DictList contains an entity."""
        if isinstance(entity, Object):
            the_id = entity.id
        else:
            the_id = entity
        return the_id in self._dict

    def __getitem__(
        self, index: Union[int, slice, str]
    ) -> Union[Object, "DictList"]:
        """Get item by index or id."""
        if isinstance(index, (int, slice)):
            result = list.__getitem__(self, index)
            if isinstance(result, list):
                new_list = self.__class__()
                new_list.extend(result)
                return new_list
            return result
        elif isinstance(index, str):
            return self.get_by_id(index)
        else:
            raise TypeError("Index must be an int, slice, or str")

    def __delitem__(self, index: Union[int, slice]) -> None:
        """Delete item by index."""
        if isinstance(index, int):
            obj = self[index]
            list.__delitem__(self, index)
            del self._dict[obj.id]
            self._generate_index()
        elif isinstance(index, slice):
            objs = self[index]
            list.__delitem__(self, index)
            for obj in objs:
                del self._dict[obj.id]
            self._generate_index()
        else:
            raise TypeError("Index must be an int or slice")

    def __setitem__(self, index: Union[int, slice], value: Object) -> None:
        """Set item at index."""
        if isinstance(index, int):
            old_obj = self[index]
            del self._dict[old_obj.id]
            self._check(value.id)
            list.__setitem__(self, index, value)
            self._dict[value.id] = index
        elif isinstance(index, slice):
            raise NotImplementedError("Setting slices is not supported.")
        else:
            raise TypeError("Index must be an int")

    def __repr__(self) -> str:
        """Return string representation."""
        return f"DictList({list.__repr__(self)})"
