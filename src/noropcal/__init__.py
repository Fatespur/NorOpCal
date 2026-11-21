"""NorOpCal — Normal-Only Selective Operating-Point Calibration.

Recalibrates decision thresholds of frozen industrial anomaly detectors
under acquisition shift using only normal observations.
"""

from .api import NorOpCal, load_frozen_spec
from .version import __version__

__all__ = ["NorOpCal", "load_frozen_spec", "__version__"]
