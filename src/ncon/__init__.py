# Making this whole thing a package is a bit silly, and seems like an
# overkill. The reason is, that in some setups, Python 3's implicit
# namespace packages (PEP 420) mess things up when the folder and the
# module have the same name.
from .ncon import ncon
