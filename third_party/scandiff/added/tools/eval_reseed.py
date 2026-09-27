"""src/eval.py, but re-seeding AFTER the checkpoint is loaded (2026-09-20).

src/callbacks/model_checkpoint.py::resume_checkpoint restores the python /
numpy / torch RNG states stored in the checkpoint, which silently overrides
the `seed=` hydra option of src/eval.py: generation is then identical for
every call on the same checkpoint. For an independent second sampling draw
we wrap resume_checkpoint and seed again afterwards. No source file is
modified (the ScanDiff test manifest hashes src/).
Usage: RESEED=<int> python tools/eval_reseed.py --config-path <abs configs> <hydra overrides as for src/eval.py>"""
import os, random
import numpy as np, torch
from src.callbacks import model_checkpoint as _mc

_SEED = int(os.environ["RESEED"])
_orig = _mc.ModelCheckpoint.resume_checkpoint
def _patched(self, *a, **k):
    out = _orig(self, *a, **k)
    random.seed(_SEED); np.random.seed(_SEED); torch.manual_seed(_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(_SEED)
    print(f"eval_reseed: RNG re-seeded with {_SEED} after checkpoint load", flush=True)
    return out
_mc.ModelCheckpoint.resume_checkpoint = _patched

from src.eval import main   # noqa: E402
if __name__ == "__main__":
    main()
