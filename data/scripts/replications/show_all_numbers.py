"""Consolidated readout of every current headline number (full 12-city corpus).

Reads the per-city Baur and Shen DML tables, the pooled meta + Portland
drop-sensitivity, LEACE erasure, and the Hotelling joint-null, and prints
one digestible block. Read-only; never recomputes. Path-anchored to the
repo root so it runs from any working directory.
"""
import csv
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]


def rows(rel):
    with open(REPO / rel) as fh:
        return list(csv.DictReader(fh))


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


print("=" * 74)
print("  EVERY CURRENT NUMBER  --  full 12-city corpus")
print("=" * 74)

try:
    print("\n--- BAUR  (pooled-PCA BERT-PC1, DML theta per sigma) ---")
    for r in rows("results/replications/baur_pooled_pca/baur_pooled_pca_table.csv"):
        th, se = num(r["dml_theta"]), num(r["dml_se"])
        tag = ("  <-- DEGENERATE (drop)" if abs(th) > 0.5
               else "  *sig" if r.get("dml_excludes_zero") == "True" else "")
        print(f"  {r['city']:<13} n={int(float(r['n'])):>6}  theta={th:+.4f}  se={se:.4f}{tag}")
except Exception as e:
    print("  [baur unavailable:", e, "]")

try:
    print("\n--- SHEN  (doc2vec uniqueness, DML theta per sigma) ---")
    for r in rows("results/replications/shen_12city_table.csv"):
        th = num(r["dml_theta"])
        tag = ("  <-- DEGENERATE (drop)" if abs(th) > 0.5
               else "  *sig" if r.get("dml_excludes_zero") == "True" else "")
        print(f"  {r['city']:<13} n={int(float(r['n'])):>6}  theta={th:+.4f}{tag}")
except Exception as e:
    print("  [shen unavailable:", e, "]")

try:
    s = json.load(open(REPO / "results/replications/portland_sensitivity.json"))
    print("\n--- POOLED META (PM+HKSJ): full k=12  vs  drop-Portland k=11 ---")
    for m in ("baur", "shen"):
        a, b = s[m]["full_panel_k12"], s[m]["drop_portland_k11"]
        print(f"  {m.upper():<5} k=12: theta={a['theta']:+.4f} p={a['p_two_sided']:.3f}"
              f"   |   k=11: theta={b['theta']:+.4f} "
              f"CI[{b['ci_low']:+.3f},{b['ci_high']:+.3f}] p={b['p_two_sided']:.3f}")
except Exception as e:
    print("  [pooled sensitivity unavailable:", e, "]")

try:
    print("\n--- LEACE  (ZIP-classifier accuracy raw -> erased; ~chance = erased) ---")
    lr = rows("results/leace12/leace_12city_table.csv")
    cols = list(lr[0].keys())
    acc_raw = next((c for c in cols if "raw" in c.lower() and "acc" in c.lower()), None)
    acc_er = next((c for c in cols if "eras" in c.lower() and "acc" in c.lower()), None)
    for r in lr:
        city = r.get("city") or list(r.values())[0]
        if acc_raw and acc_er:
            ar, ae = num(r[acc_raw]), num(r[acc_er])
            print(f"  {city:<13} {ar:.3f} -> {ae:.3f}")
        else:
            print("  " + "  ".join(str(v) for v in list(r.values())[:6]))
except Exception as e:
    print("  [leace unavailable:", e, "]")

try:
    hr = rows("results/replications/hotelling_t2_cross_method.csv")
    rej = sum(1 for r in hr
              if num(r.get("p_chi2_3df")) is not None and num(r.get("p_chi2_3df")) < 0.05)
    print(f"\n--- HOTELLING joint-null: {rej}/{len(hr)} markets reject at raw p<0.05 ---")
except Exception as e:
    print("\n  [hotelling unavailable:", e, "]")

print("\n" + "=" * 74)
