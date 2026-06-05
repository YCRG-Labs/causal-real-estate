#!/bin/bash
# Phase 3 RESUME runner. Picks up after embed_all + pooled_pca + parquet
# adapter have already PASSed (verified by inspecting their outputs). Skips
# the 10-min embedding step so a retry doesn't re-burn it; otherwise this
# is identical in shape to run_phase3.sh.
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

log "=== PHASE 3 RESUME (starts at baur step) ==="

# Sanity: confirm the prerequisites are on disk before we start chewing.
for c in $C; do
    if [ ! -f "data/processed/${c}_embeddings.parquet" ]; then
        log "ABORT missing embeddings for $c; rerun run_phase3.sh from scratch"
        exit 1
    fi
done
log "INFO  all 12 embedding parquets present"

# Baur, Shen, LEACE per city. --fast switches Baur and Shen to
# influence-function-based SE instead of B=500 percentile bootstrap; on
# the new 988-15,360 listing corpus the bootstrap pushes per-city wall
# clock from ~3 min to ~9 hours, which is unworkable for a 12-city run.
# Per-city SE only feeds inverse-variance weighting in the panel
# meta-regression, which provides the headline HKSJ CI; IF-SE is
# documented as the small-G default in §6.2 already.
for c in $C; do step baur_$c python3 data/scripts/replications/baur_pooled_pca.py --city $c --fast; done
for c in $C; do step shen_$c python3 data/scripts/replications/shen_2021.py --city $c --doc2vec --fast; done
for c in $C; do step leace_$c python3 data/scripts/leace_deconfound.py --city $c; done
step leace_rollup python3 data/scripts/replications/rollup_leace_12.py

# Confounder-set sensitivity, meta-regression, joint tests.
step sensitivity python3 data/scripts/replications/confounder_sensitivity_12.py --all_12
step meta_baur python3 data/scripts/replications/meta_regression_pooled.py --source baur_pooled
step meta_shen python3 data/scripts/replications/meta_regression_pooled.py --source shen
step hotelling python3 data/scripts/replications/hotelling_t2_cross_method.py
step typesm    python3 data/scripts/replications/type_s_type_m.py

# Counterfactual rewrites via vLLM.
step counterfactual python3 data/scripts/counterfactual/run_counterfactual.py --all_12 --use_vllm

log "=== PHASE 3 DONE ==="
