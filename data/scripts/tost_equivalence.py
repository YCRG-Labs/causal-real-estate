"""
Two One-Sided Tests (TOST) equivalence testing with pre-committed equivalence
margin delta. Replaces the "CI contains zero" framing in the paper with a
positive equivalence claim when justified.

For an equal-tailed interval, TOST at level alpha is algebraically equivalent
to: the (1 - 2 alpha) two-sided CI lies entirely inside [-delta, +delta].
The reported CI in the paper at the 95% level corresponds to TOST at alpha=2.5%;
report the 90% CI to read off the equivalence claim at the standard alpha=5%.

Margin discipline:
  delta must be PRE-COMMITTED, justified by a substantive economic anchor,
  and never reverse-engineered from the realized CI. Recommended anchor for
  log-price hedonic effects: delta = 0.01 (1% in price), which sits below
  Case-Shiller monthly noise and below one-quarter of the smallest amenity
  coefficient in Bayer, Ferreira & McMillan 2007 JPE.

References:
  Schuirmann (1987) JPB 15:657-680 — TOST origin.
  Romano (2005) Annals of Statistics 33(3) — optimal equivalence testing.
  Wellek (2010) Testing Statistical Hypotheses of Equivalence (CRC).
  Hartman & Hidalgo (2018) AJPS — equivalence approach to balance/placebo.
  Lakens (2017) SPPS 8(4) — practical exposition.
  Fitzgerald (2024) I4R DP-125 — "The Need for Equivalence Testing in
    Economics" (Stata/R commands, calibration examples).
"""
from __future__ import annotations
from dataclasses import dataclass
import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats


@dataclass
class TOSTResult:
    theta_hat: float
    se: float
    delta: float
    alpha: float
    df: float | None
    p_lower: float
    p_upper: float
    tost_p_value: float
    ci_two_sided: tuple[float, float]
    ci_traditional: tuple[float, float]
    rejects_equivalence_null: bool
    interpretation: str


def tost(theta_hat: float, se: float, delta: float,
         alpha: float = 0.05, df: float | None = None) -> TOSTResult:
    """Two one-sided tests for equivalence: H0: |theta| >= delta vs H1: |theta| < delta.

    Uses t-distribution if df given (e.g. bootstrap df = B for cheap bootstrap),
    else standard normal.
    """
    if se <= 0:
        raise ValueError("se must be positive")
    if delta <= 0:
        raise ValueError("delta must be positive (pre-committed economic margin)")

    if df is None:
        dist = stats.norm
        q_low = dist.ppf(alpha)
        q_high = dist.ppf(1 - alpha)
        q_two = dist.ppf(1 - alpha / 2)
        q_trad = dist.ppf(1 - alpha / 2)
        z_l = (theta_hat - (-delta)) / se
        z_u = (theta_hat - delta) / se
        p_lower = 1 - dist.cdf(z_l)
        p_upper = dist.cdf(z_u)
    else:
        dist = stats.t(df=df)
        q_low = dist.ppf(alpha)
        q_high = dist.ppf(1 - alpha)
        q_two = dist.ppf(1 - alpha / 2)
        q_trad = dist.ppf(1 - alpha / 2)
        t_l = (theta_hat - (-delta)) / se
        t_u = (theta_hat - delta) / se
        p_lower = 1 - dist.cdf(t_l)
        p_upper = dist.cdf(t_u)

    tost_p = float(max(p_lower, p_upper))
    rejects = tost_p < alpha

    if df is None:
        q = stats.norm.ppf(1 - alpha)
    else:
        q = stats.t(df=df).ppf(1 - alpha)
    ci_two_sided = (theta_hat - q * se, theta_hat + q * se)

    ci_trad = (theta_hat - q_trad * se, theta_hat + q_trad * se)

    if rejects:
        interp = (f"Equivalence demonstrated: |theta| < delta={delta:.4f} at "
                  f"alpha={alpha:.3f} (TOST p={tost_p:.4f}). The "
                  f"{(1-2*alpha)*100:.0f}% CI {ci_two_sided} lies inside "
                  f"[-delta, delta].")
    else:
        interp = (f"Equivalence not established at alpha={alpha:.3f} "
                  f"(TOST p={tost_p:.4f}). The {(1-2*alpha)*100:.0f}% CI "
                  f"{ci_two_sided} extends outside [-delta, delta].")

    return TOSTResult(
        theta_hat=float(theta_hat), se=float(se), delta=float(delta),
        alpha=float(alpha), df=float(df) if df is not None else None,
        p_lower=float(p_lower), p_upper=float(p_upper),
        tost_p_value=tost_p,
        ci_two_sided=(float(ci_two_sided[0]), float(ci_two_sided[1])),
        ci_traditional=(float(ci_trad[0]), float(ci_trad[1])),
        rejects_equivalence_null=bool(rejects),
        interpretation=interp,
    )


def tost_from_dml_result(dml_result: dict, delta: float = 0.01,
                         alpha: float = 0.05) -> TOSTResult:
    """Run TOST on a DML headline output. Expects keys 'theta', 'se' (or 'se_cb').

    The cheap-bootstrap SE (Lam 2022) is the right SE to feed in for our pipeline.
    """
    theta = dml_result.get("theta")
    se = dml_result.get("se_cb") or dml_result.get("se") or dml_result.get("se_if")
    df = dml_result.get("B")
    if theta is None or se is None:
        raise KeyError("dml_result must contain 'theta' and 'se'/'se_cb'/'se_if'")
    return tost(theta, se, delta=delta, alpha=alpha, df=df)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", help="JSON file with DML results "
                    "(list of dicts with theta + se + B)")
    ap.add_argument("--delta", type=float, default=0.01,
                    help="pre-committed equivalence margin on log-price scale "
                         "(default 0.01 = 1% in price; below Case-Shiller noise)")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--out", default="results/dedup_rerun/tost.json")
    args = ap.parse_args()

    if args.inp is None:
        demo = tost(theta_hat=-0.001, se=0.025, delta=args.delta,
                    alpha=args.alpha, df=50)
        print(json.dumps(demo.__dict__, indent=2, default=str))
        return

    rows = json.loads(Path(args.inp).read_text())
    if isinstance(rows, dict):
        rows = [rows]
    out = []
    for r in rows:
        try:
            res = tost_from_dml_result(r, delta=args.delta, alpha=args.alpha)
            out.append({"city": r.get("city"), **res.__dict__})
        except KeyError as e:
            out.append({"city": r.get("city"), "error": str(e)})
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"Saved -> {out_path}")
    for r in out:
        print(json.dumps(r, indent=2, default=str))


if __name__ == "__main__":
    main()
