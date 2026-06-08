#!/bin/bash
set -u
cd ~/causal-real-estate
mkdir -p logs/cf
echo "=== CF n=200 rerun start $(date) ==="
python3 data/scripts/counterfactual/run_counterfactual.py --all_12 --n_listings 200 \
    --use_vllm --skip_perplexity --force_redo --out_dir results/counterfactual
echo "=== CF ROLLUP ==="
python3 data/scripts/replications/rollup_counterfactual_12.py --results_dir results/counterfactual 2>&1 | tail -25
echo "=== CF n=200 DONE $(date) ==="
