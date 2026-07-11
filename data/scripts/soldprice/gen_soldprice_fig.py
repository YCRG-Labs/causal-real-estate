from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
SP = REPO / "results" / "soldprice"
FIG = REPO / "paper" / "drafts_jbes_2026" / "fig_soldprice_v1.tex"

NAMES = {"boston": "Boston", "sf": "San Francisco", "dc": "Washington",
         "philadelphia": "Philadelphia", "chicago": "Chicago", "seattle": "Seattle",
         "denver": "Denver", "atlanta": "Atlanta", "portland": "Portland",
         "phoenix": "Phoenix"}


def main():
    A = {x["city"]: x for x in json.loads((SP / "soldprice_dml_assessor.json").read_text())}
    L = {x["city"]: x for x in json.loads((SP / "soldprice_dml_listing.json").read_text())}

    assr = sorted(A.values(), key=lambda x: -abs(x["theta"]))
    listing_only = sorted([L[c] for c in L if c not in A], key=lambda x: -abs(x["theta"]))
    order = [x["city"] for x in assr] + [x["city"] for x in listing_only]
    ypos = {c: len(order) - i for i, c in enumerate(order)}

    def flip(v):
        return -v

    yt = ",".join(str(ypos[c]) for c in order)
    yl = ",".join(NAMES[c] for c in order)

    lines = [r"\begin{figure}[H]", r"\centering", r"\begin{tikzpicture}",
             r"\begin{axis}[",
             r"    causalre, width=0.62\textwidth, height=7.4cm,",
             f"    ytick={{{yt}}}, yticklabels={{{yl}}},",
             r"    xlabel={$\hat\theta$ per $\sigma$ (log sale price)},",
             r"    xmin=-0.008, xmax=0.185, xtick={0,0.05,0.10,0.15},",
             r"    scaled x ticks=false, xticklabels={$0$,$0.05$,$0.10$,$0.15$},",
             r"    enlarge y limits=0.07,",
             r"    extra x ticks={0}, extra x tick style={grid=major,"
             r" grid style={black, dashed, thin}}, extra x tick labels={},",
             r"    legend style={at={(0.97,0.06)}, anchor=south east}, legend cell align=left,",
             r"]"]

    off = 0.17
    aw, ap = [], []
    for x in assr:
        c = x["city"]
        y = ypos[c] + off
        lo, hi = sorted([flip(x["ci_boot"][0]), flip(x["ci_boot"][1])])
        aw.append(f"\\draw[thin,gray] (axis cs:{lo:.3f},{y:.2f})--(axis cs:{hi:.3f},{y:.2f});")
        ap.append(f"({flip(x['theta']):.3f},{y:.2f})")
    lw, lp = [], []
    for c in order:
        x = L[c]
        y = ypos[c] - (off if c in A else 0.0)
        se = x["se_boot"]
        lo, hi = flip(x["theta"]) - 1.96 * se, flip(x["theta"]) + 1.96 * se
        lo, hi = sorted([lo, hi])
        lw.append(f"\\draw[thin,gray] (axis cs:{lo:.3f},{y:.2f})--(axis cs:{hi:.3f},{y:.2f});")
        lp.append(f"({flip(x['theta']):.3f},{y:.2f})")

    lines += aw + lw
    lines.append(r"\addplot[only marks, mark=*, mark size=1.9pt, color=cNYC] coordinates {"
                 + " ".join(ap) + "};")
    lines.append(r"\addlegendentry{assessor controls}")
    lines.append(r"\addplot[only marks, mark=o, mark size=2.1pt, line width=0.8pt, color=cYEL]"
                 r" coordinates {" + " ".join(lp) + "};")
    lines.append(r"\addlegendentry{listing controls}")
    lines += [r"\end{axis}", r"\end{tikzpicture}",
              r"\caption{Per-market effect of the leading listing-text direction on log "
              r"realized sale price, oriented so the effect is positive. Filled blue markers "
              r"use exogenous county-assessor structural controls (the primary, "
              r"controlled-direct-effect specification); open amber markers use "
              r"agent-reported listing attributes (the bad-control sensitivity bound). Gray "
              r"whiskers are $95\%$ tract cluster-bootstrap intervals ($B=300$); the dashed "
              r"line marks the null. Washington and Atlanta lack a county structural source "
              r"and carry only the listing specification. The listing effect lies inside the "
              r"assessor effect in every market where both are estimated, the signature of "
              r"controlling for text-entangled attributes. Numerical values appear in "
              r"Table~\ref{tab:soldprice}.}",
              r"\label{fig:soldprice-forest}", r"\end{figure}"]
    FIG.write_text("\n".join(lines) + "\n")
    print(f"wrote {FIG}")
    print("assessor order (top->bottom):", [x["city"] for x in assr])
    print("listing-only:", [x["city"] for x in listing_only])


if __name__ == "__main__":
    main()
