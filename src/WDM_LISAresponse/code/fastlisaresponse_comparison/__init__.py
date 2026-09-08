"""Comparison of the WDM wavelet response against ``fastlisaresponse``.

Everything in this subpackage exists to support one study: a second-generation
TDI comparison between this package's wavelet-domain response and the
time-domain response of ``fastlisaresponse``, on a single galactic binary.

It is deliberately self-contained. The comparison needs a galactic binary
waveform, a TDI combination, a convention adapter and a set of locked
parameters, none of which the rest of the package wants yet, and all of which
would raise "why is this here?" if they were scattered into ``waveforms``,
``utils`` and a general ``tdi`` module. Keeping them together means the folder
name is the explanation. If the TDI combination later earns a place as a
package feature it moves out as a unit, and nothing else has to move with it.

``fastlisaresponse`` is never imported from inside this subpackage. It follows
the convention already set for FEW in ``waveforms.waveforms``: the optional
dependency is constructed by the caller and passed in, so the package stays
installable and importable without it. Only the notebook touches it.
"""

from . import config, gb_waveform, tdi, tdi_reference

__all__ = ["config", "gb_waveform", "tdi", "tdi_reference"]
