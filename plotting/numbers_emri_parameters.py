"""The EMRI's derived parameters in Table I, from FEW's trajectory.

The trajectory of the waveform model (FastKerrEccentricEquatorialFlux's
KerrEccEqFlux) for the EMRI of parameters.py, starting at the start of the record:
the time to plunge after the end of the record, where FEW ends the inspiral, and
the semi-latus rectum and eccentricity at the end of the record (N_SIGNAL dt).
Printed and written to numbers/emri_parameters.txt.
"""

from pathlib import Path

from parameters import DT, EMRI_PARAMS, N_SIGNAL

HERE = Path(__file__).parent


if __name__ == "__main__":
    from few.trajectory.inspiral import EMRIInspiral
    from few.trajectory.ode import KerrEccEqFlux
    from few.utils.constants import YRSID_SI

    traj = EMRIInspiral(func=KerrEccEqFlux)
    args = tuple(EMRI_PARAMS[k] for k in ("m1", "m2", "a", "p0", "e0", "x0"))
    record = N_SIGNAL * DT                                  # seconds
    t, *_ = traj(*args, T=10.0, dt=DT)                      # ends at the plunge
    t_end, p_end, e_end, *_ = traj(*args, T=record / YRSID_SI, dt=DT)
    lines = [f"record: {record / 86400:.2f} d",
             f"time to plunge after the end of the record: {(t[-1] - record) / YRSID_SI:.3f} yr",
             f"semi-latus rectum at the end of the record: {p_end[-1]:.4f} GM/c^2",
             f"eccentricity at the end of the record: {e_end[-1]:.4f}"]
    text = "\n".join(lines)
    print(text)
    (HERE / "numbers").mkdir(exist_ok=True)
    (HERE / "numbers" / "emri_parameters.txt").write_text(text + "\n")
