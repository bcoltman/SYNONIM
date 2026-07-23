__author__ = "Benjamin Coltman."

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("synonim")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"


from synonim.core import (
    
    DictList,
    Model,
    Object,
    Feature,
    Profile,

)

from synonim import io
