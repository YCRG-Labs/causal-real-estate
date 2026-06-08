#!/bin/bash
set -u
cd ~/causal-real-estate
C="sf boston nyc dc philadelphia chicago seattle denver atlanta portland phoenix dallas"

echo "=== Baur per city (valid_mask alignment fix) === $(date)"
python3 data/scripts/replications/baur_pooled_pca.py --all_12 --fast 2>&1 | tail -16

echo "=== Shen per city (winsorized confounders + valid_mask fix) === $(date)"
for c in $C; do
    echo -n "  shen $c: "
    python3 data/scripts/replications/shen_2021.py --city "$c" --doc2vec --fast \
        --out results/replications/shen_$c.json 2>&1 | grep -E "DML .* causal|wrote" | tail -1
done
echo "=== rollup shen ==="
python3 data/scripts/replications/rollup_shen_12.py 2>&1 | tail -18

echo "=== confounder sensitivity ==="
python3 data/scripts/replications/confounder_sensitivity_12.py --all_12 2>&1 | tail -6

echo "=== meta-regression (Baur + Shen) ==="
python3 data/scripts/replications/meta_regression_pooled.py --source baur_pooled 2>&1 | tail -6
python3 data/scripts/replications/meta_regression_pooled.py --source shen 2>&1 | tail -6

echo "=== Hotelling joint-null ==="
python3 data/scripts/replications/hotelling_t2_cross_method.py 2>&1 | tail -8

echo "=== Type-S / Type-M ==="
python3 data/scripts/replications/type_s_type_m.py 2>&1 | tail -8

echo "=== cross-method concordance + dollarized effects ==="
python3 data/scripts/replications/cross_method_concordance.py 2>&1 | tail -6
python3 data/scripts/replications/dollarize_effects.py 2>&1 | tail -6

echo "=== STATS CASCADE DONE === $(date)"
