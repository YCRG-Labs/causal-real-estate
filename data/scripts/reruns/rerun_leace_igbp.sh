#!/bin/bash
set -u
cd ~/causal-real-estate
mkdir -p results/leace12_igbp
echo "=== LEACE+IGBP rerun start $(date) ==="
python3 -c "import torch; print('CUDA available:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
for c in sf boston nyc dc philadelphia chicago seattle denver atlanta portland phoenix dallas; do
    echo "=== leace+igbp $c ==="
    python3 data/scripts/leace_deconfound.py --city "$c" --postnl_igbp --force_continuous --out_dir results/leace12_igbp 2>&1 | tail -6
done
echo "=== ROLLUP ==="
python3 data/scripts/replications/rollup_leace_12.py --results_dir results/leace12_igbp 2>&1 | tail -30
echo "=== LEACE+IGBP DONE $(date) ==="
