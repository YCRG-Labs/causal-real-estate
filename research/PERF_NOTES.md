# Performance plan (researched 2026-06-06, 4 parallel agents, repo-grounded)

Profile FIRST, then optimize the measured bottleneck. Biggest wins are free code
changes to the two jobs running now (counterfactual vLLM, LEACE+IGBP).

## ⚠️ Safety first — do these BEFORE the n=200 counterfactual run
The current `run_counterfactual.py` will, at `--n_listings 200`:
- **reload the 32B-AWQ model 12 times** (engine built inside the per-city loop,
  `run_counterfactual.py:372`, called per city `:598`) — throws away the cross-city
  prefix cache and risks a double-resident-LLM OOM on a 48GB card.
- **cap in-flight sequences at 4** — it batches the 4 arms/listing but loops 200×
  per city (`:378`), so 200 separate `llm.generate()` calls of 4 prompts. vLLM's
  continuous batching is wasted; this is the throughput bottleneck.
- **write the city JSON once at the very end** (`:542`) — a crash at listing
  14k/15k loses the whole city (hours). The `baur_sf` bootstrap run already did
  this: 9h → OOM kill (rc=137) → nothing saved (`logs/phase3/run.log`).

## A. Counterfactual / vLLM — ranked
1. **Batch ALL ~800 prompts/city in one `generate()` call**, validate/score in a
   second pass. `generate_blocks_batch` already accepts a flat list
   (`generator.py:484`). Raises concurrency 4 → KV-limited; removes ~200 GPU-idle
   validation gaps. **~5–15× generation wall-clock.** Risk: low (identical output).
2. **Hoist the vLLM engine out of the per-city loop** — build once in `main`, pass
   into `run_pipeline`. Saves 11 reloads, keeps prefix cache, kills the OOM class.
   **Tens of min/run.** Risk: low.
3. **Intra-city checkpointing** — append each listing to JSONL instead of one final
   `json.dump`. Bounds re-do on crash. Risk: low. (Do this or risk the 9h-wipeout.)
4. **`max_tokens` 2048 → ~512** (`generator.py:418`) — rewrites are short; lets the
   big batch admit more concurrent seqs. Validator already salvages truncation.
5. Keep: `enable_prefix_caching=True` (already on; system prompt is a true shared
   prefix — `prompts.py:324/392` take no args), `awq_marlin`, structured outputs OFF.
   `--skip_perplexity` already passed (good; GPT-2 check is CPU, `validator.py:52`).

## B. LEACE + IGBP — ranked
1. **float64 → float32** (`leace_deconfound.py:299, 888`, `.double()` at
   `578,667,703,787,809`). ~2× linalg, ~1.5–2× the MLP/MDL/IGBP probes that DOMINATE
   runtime, half memory, unlocks GPU. concept-erasure runs float32 natively (its
   default `dtype=None` inherits input; `svd_tol=0.01`). **Caveat: loosen the tight
   guardedness assertion `:345` from 1e-4 → ~1e-3** — float32 residual over d=768 can
   reach ~1e-5–1e-4. Risk: medium-low.
2. **Randomized SVD for the IGBP top-1 direction** (`:622`, only `Vt[0]` used) →
   `sklearn.utils.extmath.randomized_svd(G, n_components=1, n_iter=7)`. **~20–50× that
   step** (~5–15% of IGBP). Risk: low (exact to sign).
3. **Move probes/IGBP adversary to CUDA float32** — currently all CPU float64, zero
   `.cuda()` in the file. On Brev's GPU **10–50× on the probe loops** (measure first).
   Requires #1. Risk: low correctness, big payoff on GPU box.
4. IGBP `n_outer` 5 → 3 (`:590`) — leakage curve plateaus; ~40% less IGBP. METHODOLOGY
   knob — validate erased R² unchanged before lowering; keep 5 if it moves the number.
5. The closed-form LEACE SVD and `_verify_linear_guardedness` are already cheap —
   do NOT touch.

## C. Phase-3 statistical pipeline — ranked
1. **REAL BUG: `run_phase3.sh:51` runs baur WITHOUT `--fast`** → the B=500 GBM
   bootstrap path that measured 9h then OOM-killed. The resume runner has `--fast`
   (24s). Add `--fast` to run_phase3.sh. (Also `:54` shen.)
2. **Wire the v2 slim-parquet loader into the shared loader** — `causal_inference.py:26`
   re-reads the parcels `.gpkg` (SF 364MB, ~12.7s) once per process × 24 processes.
   `_load_parcels_slim`/`_build_slim_cache` already exist in `fast_bootstrap_dml_v2.py:140-175`
   (0.12s read). Risk: low (v2 proves equivalence).
3. **Parallelize the two serial bootstrap loops** — `causal_inference.py:454` (GBM →
   loky, inner_max_num_threads=1) and `compare_to_dml.py:93` (ridge → joblib
   **threading** backend; LAPACK releases GIL, no IPC). ~min(B,cores)×.
4. **Lazy-import torch** in `causal_inference.py:6` — ~3–5s × 24 processes, code path
   never uses torch outside adversarial_deconfounding.
5. **Memoize load+spatial-join** (`joblib.Memory`, key on mtimes) — recomputed 4–5×/city
   across baur/leace/counterfactual/sensitivity.
6. **Drop the unused MiniLM embed pass + load SBERT once** (`generate_embeddings.py:67,162-173`) —
   ~half the 592s embed step. fp16 on GPU (`model_kwargs={"torch_dtype":"float16"}`) ~2×.
7. **GNU-parallel the independent per-city loops** (shen/leace; baur already fast):
   `printf '%s\n' $C | parallel -j 3 python3 .../shen_2021.py --city {} --doc2vec --fast ...`
   (cap -j vs the bootstrap parallelism — do one or the other, not both).

REJECTED (don't chase): HistGBR at n≈300, polars on 12-row tables, numba on the IF
math, caching embeddings across cities (not re-read). All verified dead ends.

## D. Profile-first playbook (copy-paste)
```bash
pgrep -af run_counterfactual.py ; pgrep -af leace_deconfound.py     # PIDs
sudo py-spy top --pid $(pgrep -f run_counterfactual.py|head -1)      # live hot frames
sudo py-spy record --pid <PID> --subprocesses --duration 60 -o /tmp/cf.svg
sudo py-spy dump --pid $(pgrep -f leace_deconfound.py|head -1)       # where LEACE is stuck
sudo py-spy record --native --pid <leace PID> --duration 60 -o /tmp/leace.svg
scalene run --cpu --gpu --memory --html --outfile /tmp/leace.html --- data/scripts/leace_deconfound.py --city sf
nvidia-smi dmon -s pucvmet -d 1 -o DT | tee /tmp/gpu.log             # GPU util during vLLM
```
Read GPU: low SM util + near-cap memory ⇒ KV/bandwidth-bound ⇒ a bigger card helps.

## E. Hardware (Brev) — only after the free fixes
- **Counterfactual vLLM (Job A): a single A100-80GB (~2 TB/s, ~2× bandwidth + ~2× KV)
  ≈ ~2× throughput; H100-80GB if you want max.** NOT tensor-parallel (32B-AWQ fits on
  one card; TP adds comms for no capacity gain). Confirm with `dmon` before buying.
- **LEACE (Job B): a bigger GPU does NOTHING as-written** (pure CPU float64). Leverage =
  high-core CPU + GNU-parallel the 12 cities, and/or port probes to CUDA (B#3).
- **SBERT encode: no upgrade needed** — fix the redundant MiniLM pass instead.

Every external claim above was WebFetch-verified by the research agents; items they
could not confirm (exact speedup multipliers, GPU bandwidth sheets) are flagged
UNVERIFIED in the session log — treat magnitudes as approximate, confirm by profiling.
