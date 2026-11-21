"""NorOpCal — Normal-Only Selective Operating-Point Calibration.

Public facade over the frozen PB7-P1 scientific implementation. See
``docs/METHOD.md`` and ``docs/reference/PUBLICATION_GROUND_TRUTH.md``.
"""

from .api import NorOpCal, load_frozen_spec
from ._bridge import frozen_spec_path
from .version import __version__

__all__ = ["NorOpCal", "load_frozen_spec", "frozen_spec_path", "__version__"]
