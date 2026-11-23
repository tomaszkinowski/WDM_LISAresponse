"""Comparisons of the WDM wavelet response against external codes.

- ``config``: the locked parameters every code runs on.
- ``tdi``, ``tdi_reference``: our second-generation TDI, wavelet and time
  domain.
- ``common``: the conversion, velocity factor and report every comparison
  shares.
- ``conventions_flr``, ``conventions_gwr``: running ``fastlisaresponse``, and
  ``lisagwresponse`` with ``pytdi``, on our source and grid.

No external code is imported here: the caller passes the classes in, so the
package stays importable without them. Only
``examples/response_comparison.ipynb`` imports them.
"""

from . import (
    common, config, conventions_flr, conventions_gwr, tdi, tdi_reference,
)

__all__ = ["common", "config", "conventions_flr", "conventions_gwr", "tdi",
           "tdi_reference"]
