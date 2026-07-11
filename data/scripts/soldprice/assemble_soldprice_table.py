from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
SP = REPO / "results" / "soldprice"
TEX = REPO / "paper" / "drafts_jbes_2026" / "tab_soldprice_v1.tex"

NAMES = {"boston": "Boston", "sf": "San Francisco", "dc": "Washington",
         "philadelphia": "Philadelphia", "chicago": "Chicago", "seattle": "Seattle",
         "denver": "Denver", "atlanta": "Atlanta", "portland": "Portland",
         "phoenix": "Phoenix"}
ORDER = ["philadelphia", "chicago", "sf", "boston", "dc", "atlanta",
         "seattle", "denver", "portland", "phoenix"]


def load(name):
    p = SP / name
    return json.loads(p.read_text()) if p.exists() else []


def main():
    A = {x["city"]: x for x in load("soldprice_dml_assessor.json")}
    L = {x["city"]: x for x in load("soldprice_dml_listing.json")}
    rob = load("soldprice_robustness.json")
    S = {x["city"]: x for x in (rob.get("spatial", []) if isinstance(rob, dict) else [])}
    E = {x["city"]: x for x in (rob.get("egami", []) if isinstance(rob, dict) else [])}
    Q = {x["city"]: x for x in (rob.get("pc1_time", []) if isinstance(rob, dict) else [])}

    lines = [
        r"\begin{table}[H]", r"\centering",
        r"\caption{Effect of the leading listing-text direction on log realized sale "
        r"price, by metropolitan area. Estimates are ridge partially-linear DML on the "
        r"single-unit subset with sale-quarter fixed effects; the assessor arm controls "
        r"for exogenous county-record attributes (the primary, controlled-direct-effect "
        r"specification) and the listing arm for agent-reported attributes (a "
        r"bad-control sensitivity bound). Intervals are tract cluster-bootstrap "
        r"($B=300$); $\hat\theta_{\text{corr}}$ divides the assessor estimate by the "
        r"validation-sample truncation factor. Estimates are oriented so the effect "
        r"is positive; all are individually significant.}",
        r"\label{tab:soldprice}", r"\small",
        r"\begin{tabular}{lrrrrr}", r"\toprule",
        r"Market & $n$ & $\hat\theta_{\text{assessor}}$ & 95\% CI (cluster) & "
        r"$\hat\theta_{\text{listing}}$ & $\hat\theta_{\text{corr}}$ \\", r"\midrule"]
    for c in ORDER:
        a, l = A.get(c), L.get(c)
        if not (a or l):
            continue
        n = (a or l)["n"]
        nstr = f"${n:,}$".replace(",", "{,}")
        if a:
            lo, hi = sorted([-a["ci_boot"][0], -a["ci_boot"][1]])
            ta = f"${-a['theta']:.3f}$"
            ci = f"$[{lo:.3f}, {hi:.3f}]$"
            tc = f"${-a['theta_corrected']:.3f}$"
        else:
            ta, ci, tc = "--", "--", "--"
        tl = f"${-l['theta']:.3f}$" if l else "--"
        lines.append(f"{NAMES[c]} & {nstr} & {ta} & {ci} & {tl} & {tc} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    TEX.write_text("\n".join(lines) + "\n")

    print("=== assembled sold-price results ===")
    print(f"{'metro':13s}{'n':>7s}{'assr':>8s}{'list':>8s}{'corr':>8s}"
          f"{'conleyHAC':>10s}{'IM_excl0':>9s}{'egami':>8s}{'pc1~qR2':>9s}")
    for c in ORDER:
        a, l = A.get(c), L.get(c)
        if not (a or l):
            continue
        aa = f"{a['theta']:+.3f}" if a else "  --  "
        cc = f"{a['theta_corrected']:+.3f}" if a else "  --  "
        s = S.get(c); e = E.get(c); q = Q.get(c)
        hac = f"{s['se_conley_hac']:.3f}" if s else "  -- "
        imx = ("yes" if s and s["im_excludes_0"] else ("no" if s else "--"))
        eg = f"{e['theta_egami']:+.3f}" if e else "  --  "
        qr = f"{q['pc1_on_quarter_r2']:.3f}" if q else "  -- "
        print(f"{c:13s}{(a or l)['n']:7d}{aa:>8s}{l['theta']:+8.3f}{cc:>8s}"
              f"{hac:>10s}{imx:>9s}{eg:>8s}{qr:>9s}")
    print(f"\nwrote {TEX}")


if __name__ == "__main__":
    main()
