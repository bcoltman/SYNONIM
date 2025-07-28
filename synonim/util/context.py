"""
HistoryManager module: provides a context manager to support reversible,
temporary modifications on objects. It records a stack of reversal operations,
which are executed when the context is exited.
"""

from functools import partial
from typing import Any, Callable, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    # Replace 'Object' with your base class if needed.
    from your_project.object import Object


class HistoryManager:
    """
    Context manager for reversible modifications.

    This class records a list of operations (as callables) that reverse changes
    made to an object's state. Once the context is exited, the recorded operations
    are executed in reverse order to restore the original state.
    """
    
    def __init__(self, **kwargs) -> None:
        """Initialize the HistoryManager with an empty history stack."""
        self._history = []
    
    def __call__(self, operation: Callable[[Any], Any]) -> None:
        """
        Add a reversal operation to the history stack.

        Parameters
        ----------
        operation : Callable[[Any], Any]
            A function to reverse a change; it is called without arguments when resetting.
        """
        self._history.append(operation)
    
    def reset(self) -> None:
        """
        Execute all recorded reversal operations in reverse order.

        This method is typically called upon exiting the context to revert any changes.
        """
        while self._history:
            op = self._history.pop()
            op()  # Execute the operation to revert the change.
    
    def size(self) -> int:
        """
        Return the number of recorded operations.
        
        Returns
        -------
        int
            The size of the history stack.
        """
        return len(self._history)


def get_context(obj: Any) -> Optional[HistoryManager]:
    """
    Retrieve the active HistoryManager context for an object.

    It first checks whether the object itself has a _contexts attribute. If not,
    it looks for the object's model (via _model) and attempts to retrieve the active context.

    Parameters
    ----------
    obj : Any
        The object for which to retrieve the current HistoryManager.

    Returns
    -------
    HistoryManager or None
        The active HistoryManager if available; otherwise, None.
    """
    # Try to get _contexts directly from the object.
    try:
        return obj._contexts[-1]
    except (AttributeError, IndexError):
        pass
    # Fall back: try to get _contexts from the object's model.
    try:
        return obj._model._contexts[-1]
    except (AttributeError, IndexError):
        return None


def resettable(func: Callable[[Any, Any], None]) -> Callable[[Any, Any], None]:
    """
    Decorator to provide reversible attribute setters.

    This decorator wraps a setter method so that it saves the current value of the attribute
    (using get_context to obtain the active HistoryManager) before updating it. The old value is
    recorded via a partial function that can be called later to revert the change.

    Parameters
    ----------
    func : Callable[[Any, Any], None]
        The original setter function.

    Returns
    -------
    Callable[[Any, Any], None]
        The decorated setter function that handles reversible modifications.
    """
    def wrapper(self, new_value):
        context = get_context(self)
        if context:
            old_value = getattr(self, func.__name__)
            # Only store the reversal operation if the new value is different.
            if old_value != new_value:
                context(partial(func, self, old_value))
        func(self, new_value)
    return wrapper
