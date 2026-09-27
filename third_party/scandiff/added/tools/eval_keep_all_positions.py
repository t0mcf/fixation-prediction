"""src/eval.py with the token-validity cut disabled (control for the
length-oracle asymmetry, 2026-09-17): every generated position is exported,
so the saved generations can later be cut to the paired reference length
(tools/scandiff_gen_to_npy_lenmatch.py). No source file is modified — the
validity predictor is monkey-patched after model construction to always
vote 'valid'. Usage: identical to src/eval.py (hydra overrides)."""
import sys, torch, torch.nn as nn
from src.model.components import dit_model as _dm

def _always_valid(x):                          # (N, T, D) -> logits favouring class 1 = valid
    out = torch.zeros(*x.shape[:-1], 2, device=x.device, dtype=x.dtype); out[..., 1] = 1.0
    return out

_orig_init = _dm.DiTModel.__init__
def _patched_init(self, *a, **k):
    _orig_init(self, *a, **k)
    # keep the nn.Linear (its weights must still load from the checkpoint, 2026-09-19 fix);
    # only its forward is overridden on the instance
    self.token_validity_predictor.forward = _always_valid
    print("eval_keep_all_positions: token_validity_predictor.forward patched -> all 16 positions kept", flush=True)
_dm.DiTModel.__init__ = _patched_init

from src.eval import main   # noqa: E402  (hydra-decorated)
if __name__ == "__main__":
    main()
