import pytest
import imaging.patchmatch_inpaint as mod

CASES = [
    ("fixed_seed", (1,)),
    ("coarse_converge", (1,)),
    ("window_clip", (1,)),
    ("feather_decouple", (1,)),
    ("adaptive_iter", (1,)),
    ("block_writeback", (1,)),
    ("determinism", (1,)),
    ("validation", (1,)),
    ("audit", (1,)),
    ("fallback", (1,)),
    ("patch_search", (1,)),
    ("metrics", (1,)),
    ("propagate", (1,)),
    ("mask_dilate", (1,)),
    ("inpaint", (1,)),
]

@pytest.mark.parametrize("fn_name,args", CASES)
def test_contract(fn_name, args):
    fn = getattr(mod, fn_name)
    assert fn(*args) is not None
