"""Every script's cache rule: load the arrays if they are cached, otherwise compute
them, save them and say so."""

import numpy as np


def load_or_compute(path, compute):
    """The dict of arrays cached at `path`; when it is missing, ``compute()`` is run
    and its result saved there first. With `path` None nothing is cached."""
    if path is not None and path.exists():
        return dict(np.load(path))
    if path is not None:
        print(f"{path.name} is not cached: computing it", flush=True)
    out = compute()
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, **out)
    return out
