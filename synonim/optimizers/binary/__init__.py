from synonim.optimizers.binary.binary_genetic import BinaryGenetic
from synonim.optimizers.binary.binary_heuristic import BinaryHeuristic

__all__ = ["BinaryGenetic", "BinaryHeuristic"]


def __getattr__(name):
    if name != "BinaryMILP":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from synonim.optimizers.binary.binary_milp import BinaryMILP

    globals()[name] = BinaryMILP
    return BinaryMILP
