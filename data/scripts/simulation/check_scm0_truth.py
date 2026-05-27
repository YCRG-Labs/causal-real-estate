"""Is the scm0 'bias' a real DML bias, or a mis-set truth?

run_simulation enforces truth=0 for scm0 by fiat. But the DGP's actual
large-sample DML estimand (effect of PC1 of E on Y, partialling out W) may not
be exactly 0. This estimates it at N=20,000 (where finite-sample bias has
washed out), so we can tell which is going on:

  large-n estimand ~ +0.01  -> the n<=2000 '+0.01 bias' is the TRUE estimand;
                               enforcing truth=0 is the bug. Recalibrate scm0
                               truth and the null-cell coverage should recover.
                               Does NOT affect the real-data headline.
  large-n estimand ~ 0       -> the +0.01 at n<=2000 is a genuine finite-sample
                               DML bias that would also bias the headline point.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from simulation.dgp import fit_generator, load_real_pairs, sample_scm0  # noqa: E402
from causal_inference import dml_continuous_treatment  # noqa: E402


def main() -> int:
    real_E, real_z = load_real_pairs(None, n_subsample=None, seed=20260429)
    gen = fit_generator(real_E, real_z, low_rank=10, min_bin_n=10)

    ins, cf = [], []
    for s in range(8):
        rng = np.random.default_rng(1000 + s)
        E, _, W, Y = sample_scm0(gen, None, 20000, n_W=5, rng=rng)
        ins.append(dml_continuous_treatment(E, W, Y, verbose=False)["theta"])
        cf.append(dml_continuous_treatment(E, W, Y, cross_fit_pca=True,
                                           verbose=False)["theta"])
    ins = np.array(ins)
    cf = np.array(cf)
    print("\nscm0 large-n (N=20000, 8 draws) DML estimand vs enforced truth 0:")
    print(f"  in-sample PCA : mean={ins.mean():+.5f}  sd={ins.std(ddof=1):.5f}")
    print(f"  cross-fit PCA : mean={cf.mean():+.5f}  sd={cf.std(ddof=1):.5f}")
    print()
    m = float(ins.mean())
    if abs(m) > 0.004:
        print(f"  VERDICT: large-n estimand is {m:+.4f}, NOT 0. The scm0 'bias' is a")
        print(f"  mis-set truth (enforced 0). Recalibrate scm0 truth to ~{m:+.4f} and the")
        print(f"  null-cell coverage should recover. The real-data headline is unaffected.")
    else:
        print(f"  VERDICT: large-n estimand is ~0 ({m:+.4f}). The +0.01 at n<=2000 is a")
        print(f"  genuine finite-sample DML bias that would also bias the headline point.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
