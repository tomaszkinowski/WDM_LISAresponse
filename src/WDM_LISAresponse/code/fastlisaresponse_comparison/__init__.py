"""Comparison of the WDM wavelet response against ``fastlisaresponse``.

Everything in this subpackage exists to support one study: a second-generation
TDI comparison between this package's wavelet-domain response and the
time-domain response of ``fastlisaresponse``, on a single galactic binary.

What is here is what is *specific* to the study: a TDI combination, a
convention adapter and a set of locked parameters, none of which the rest of
the package wants yet, and all of which would raise "why is this here?" if
they were scattered into ``utils`` and a general ``tdi`` module. Keeping them
together means the folder name is the explanation. If the TDI combination
later earns a place as a package feature it moves out as a unit, and nothing
else has to move with it.

The waveform is the counter-example, and the line is worth stating because it
was once drawn in the wrong place. A galactic binary is a waveform model like
any other - it is ``monochromatic`` with a chirp and a source parametrisation
- so it lives in ``waveforms.waveforms`` behind the usual registry, and a
local copy here only duplicated the windowing that
``generate_windowed_waveform`` already dispatches. What *is* study-specific is
the choice of parameters, and those are in ``config``:
``ComparisonConfig.waveform()`` generates the binary this study runs on.

``fastlisaresponse`` is never imported from inside this subpackage. It follows
the convention already set for FEW in ``waveforms.waveforms``: the optional
dependency is constructed by the caller and passed in, so the package stays
installable and importable without it. Only the notebook touches it.
"""

from . import config, conventions, tdi, tdi_reference

__all__ = ["config", "conventions", "tdi", "tdi_reference"]
