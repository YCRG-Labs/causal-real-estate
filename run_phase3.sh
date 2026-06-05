#!/bin/bash
# Phase 3 master runner. Launches after the 12-city ZIP-shard scrape produced
# fresh listings parquets (75,480 listings vs the original 3,914-cap corpus).
# Each step's stdout+stderr is captured per-step under logs/phase3/; the master
# run.log holds START/PASS/FAIL banners plus tail-5 of any failure.
# CLI surface verified for each downstream script (see pre-flight audit, 2026-06-05).
set -u
cd ~/causal-real-estate
mkdir -p logs/phase3

C="sf boston nyc dc philadelphia chicago seattle denver atlanta portland phoenix dallas"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a logs/phase3/run.log; }

step() {
    local n=$1; shift
    log "START $n"
    local t0=$SECONDS
    "$@" > logs/phase3/$n.log 2>&1
    local rc=$?
    local dt=$((SECONDS - t0))
    if [ $rc -eq 0 ]; then
        log "PASS  $n (${dt}s)"
    else
        log "FAIL  $n (${dt}s, rc=$rc)"
        tail -5 logs/phase3/$n.log | sed 's/^/    | /' | tee -a logs/phase3/run.log
    fi
}

log "=== PHASE 3 START ==="

# 1. Re-embed all 12 cities in a single process (positional CLI). 75K listings
#    at batch=128 on A100-80GB should run in ~30-45 min total.
step embed_all python3 data/scripts/generate_embeddings.py $C

# 2. Pooled within-city-centered PCA on stacked embeddings.
step pooled_pca python3 data/scripts/replications/pooled_pca_treatment.py

# 3. Baur DML per city.
for c in $C; do step baur_$c python3 data/scripts/replications/baur_pooled_pca.py --city $c; done

# 4. Shen Doc2Vec uniqueness per city. workers=8 fix → ~3x faster Doc2Vec.
for c in $C; do step shen_$c python3 data/scripts/replications/shen_2021.py --city $c --doc2vec; done

# 5. LEACE per city, then roll up.
for c in $C; do step leace_$c python3 data/scripts/leace_deconfound.py --city $c; done
step leace_rollup python3 data/scripts/replications/rollup_leace_12.py

# 6. Confounder-set sensitivity across all 12 cities in one call.
step sensitivity python3 data/scripts/replications/confounder_sensitivity_12.py --all_12

# 7. Random-effects meta-regression (PM + HKSJ) for both Baur and Shen tables.
step meta_baur python3 data/scripts/replications/meta_regression_pooled.py --source baur_pooled
step meta_shen python3 data/scripts/replications/meta_regression_pooled.py --source shen

# 8. Hotelling joint-null and Type-S / Type-M retrodesign.
step hotelling python3 data/scripts/replications/hotelling_t2_cross_method.py
step typesm    python3 data/scripts/replications/type_s_type_m.py

# 9. Counterfactual rewrites via vLLM (longest single step, ~4-6 h on A100).
#    Uses Qwen 2.5 32B AWQ by default; pass --vllm_model to override.
step counterfactual python3 data/scripts/counterfactual/run_counterfactual.py --all_12 --use_vllm

log "=== PHASE 3 DONE ==="
