# RESULTS: an independent evaluation for SOP use

Branch `sop-eval`, based on `main` at `8e68bc6`.
- **Unchanged:** `src/` and `dashboard/`. No existing test was modified.
- **tests/:** one new file, `tests/test_known_defects.py` — 9 reproduction tests pinning the three section-9 defects so
  they cannot regress, or be fixed, silently. Suite goes 73 → 82 passed.
- **README.md:** two claims qualified (change log #6). The generated part of that change comes from `scripts/readme_metrics.py`.
- **New:** all evaluation code and outputs are in `eval_sop/`.
- **Agent evaluation (section 4):** four arms on a local 7B model with an 8k context and scored programmatically. No paid API
  and no LLM judge were used.

## 0. Setup and reproduction

| Item | Value |
|---|---|
| Churn data | Cell2Cell (`cell2celltrain.csv`, 51,047 labelled rows, 28.8% churn). I used the committed pipeline artifact `data/processed/segmented.parquet` (the exact input to Stage 3) so the segments match the shipped models. |
| Uplift data | Hillstrom MineThatData e-mail challenge, 64,000 rows, randomized (2/3 got an e-mail). Source: `http://www.minethatdata.com/Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv`, sha256 `0e5893…aece`, checked in code. Raw data is not committed (`eval_sop/.gitignore`). |
| Python env | Python 3.12.3. Pinned in `eval_sop/requirements-eval.txt`: numpy 2.4.6, pandas 2.3.3, scikit-learn 1.8.0, catboost 1.2.10, xgboost 3.2.0, causalml 0.16.0, numba 0.65.1. |
| Seeds | Churn: split and model seeds {42, 7, 13, 21, 99}. Hillstrom: split seeds {42, 7, 13, 21, 99}. The project's XGBoost learners keep their hard-coded `random_state=42`. |
| CIs | 95% percentile bootstrap over held-out rows: 1,000 resamples (churn) and **5,000 (uplift — raised from 500 in round-2 #12, because a conclusion rested on a bound of 2e-5 that was inside the Monte-Carlo error of 500)**. Agent CIs bootstrap over the 40 customers, clustered on customer. The CIs capture test-set noise only, and **none is corrected for multiple comparisons**. The ± values are std across seeds, which are overlapping splits of the same rows, so they are **not** standard errors (§5). |
| Threads | All runs used `OMP_NUM_THREADS=2 MKL_NUM_THREADS=2`. `churn_eval.py` pins CatBoost `thread_count=2`; CatBoost output depends on it (see #5 below). |

Reproduce (from the repo root, with the env above):

```
OMP_NUM_THREADS=2 python eval_sop/churn_eval.py           # ~9 min; byte-identical CSVs across two runs
OMP_NUM_THREADS=2 python eval_sop/uplift_hillstrom.py     # ~53 min at N_BOOT=5000 (downloads + verifies Hillstrom if missing)
OMP_NUM_THREADS=2 python eval_sop/positivity_signflip.py  # <1 min
OMP_NUM_THREADS=2 python eval_sop/direction_check_random.py   # <1 min
OMP_NUM_THREADS=2 python eval_sop/agent_eval.py --scenarios-only   # writes scenario set, no LLM
OMP_NUM_THREADS=2 python eval_sop/agent_eval.py --model qwen2.5:7b --seeds 0 1 2   # ~52 min, local Ollama
OMP_NUM_THREADS=2 python eval_sop/agent_parser_check.py --agent             # <1 min, offline; agent-arm parser symmetry
OMP_NUM_THREADS=2 python eval_sop/agent_parser_check.py --no-tools --model qwen2.5:7b   # ~10 min, no-tools arm
OMP_NUM_THREADS=2 python eval_sop/agent_third_arm.py --arm B --model qwen2.5:7b --seeds 0 1 2   # ~10 min, local Ollama (arm B)
OMP_NUM_THREADS=2 python eval_sop/agent_third_arm.py --arm D --model qwen2.5:7b --seeds 0 1 2   # ~12 min, local Ollama (arm D)
OMP_NUM_THREADS=2 python eval_sop/agent_section4_figures.py   # <1 min, offline; every section-4 figure + CI + decomposition
python -m pytest -q                                           # 82 passed (73 existing + 9 defect-reproduction)
```

**The README figures reproduce exactly.** With the project's own `churn_model.train_segment_model` and seed 42, every
per-segment holdout AUC and Brier pair matches the README table to the printed precision
(`eval_sop/results/churn_reproduction_seed42.csv`). Examples: At-Risk AUC 0.6923, Brier 0.2197→0.1778; Lapsed 0.5755,
0.2453→0.2197. This path keeps the project's `thread_count=-1`, exactly as the README numbers were produced.

## 1. Churn model (Cell2Cell, pooled held-out set: n = 10,211 per seed)

The test set is the union of the per-segment 80/20 stratified holdouts. Every model is trained on the same training
rows and scored on the same test rows. Values are mean ± std over 5 seeds — a std over five overlapping splits of the
same rows, not a standard error (section 5). The 95% bootstrap CIs come from seed 42 and are **uncorrected** for
multiple comparisons.
Columns:
- **Skill vs base rate:** Brier skill score against the constant training churn rate.
- **Skill vs segment base rate:** a sensitivity check, against each segment's own training churn rate.

| Model | AUC | PR-AUC (base 0.288) | Brier | Skill vs base rate | Skill vs segment base rate | ECE |
|---|---|---|---|---|---|---|
| Base rate (constant 0.288) | 0.500 | 0.288 | 0.2051 | 0 | −0.5% | 0 |
| Logistic regression (22 project features) | 0.594 ± 0.003 | 0.351 ± 0.003 | 0.2008 | 2.1% ± 0.2 | 1.6% ± 0.2 | 0.009 |
| **Per-segment CatBoost + isotonic (project)** | **0.629 ± 0.006** [0.619, 0.642] | 0.391 ± 0.008 [0.371, 0.403] | 0.1970 | **4.0% ± 0.5** [3.0, 4.8] | **3.5% ± 0.5** | 0.015 [0.012, 0.027] |
| Per-segment CatBoost, uncalibrated | 0.620 ± 0.004 | 0.388 ± 0.007 | 0.2360 | −15.0% ± 0.6 | −15.6% ± 0.6 | 0.196 |
| Global CatBoost + isotonic (same recipe, same features) | **0.640 ± 0.004** [0.626, 0.650] | 0.402 ± 0.005 | 0.1948 | 5.1% ± 0.2 [4.0, 5.9] | 4.6% ± 0.2 | 0.009 |
| Global CatBoost + isotonic + segment id | 0.640 ± 0.003 | 0.403 ± 0.004 | 0.1945 | 5.2% ± 0.3 | 4.7% ± 0.3 | 0.009 |
| *Reference:* global CatBoost on all raw Cell2Cell columns | 0.673 ± 0.005 [0.665, 0.687] | 0.443 ± 0.005 | 0.1890 | 7.9% ± 0.3 [6.9, 9.0] | 7.4% ± 0.3 | 0.011 |

> **Re-measured 2026-10-07** after the three feature defects in §11 were fixed, so
> these differ slightly from the figures this section carried before. The project
> model moved 0.631 → 0.629 and the logistic-regression row now has 22 features
> rather than 23. The reference arm reproduced at 0.6734 against the 0.673 it was
> published at, which is what confirmed the §10a fix to how that arm selects its
> columns. Every figure here comes from a single run of
> `eval_sop/churn_eval.py` whose per-seed outputs are in `eval_sop/results/`; a
> second run of the whole pipeline and eval reproduced all of them exactly. The
> bracketed 95% bootstrap CIs are seed 42's, as before, and remain uncorrected for
> multiple comparisons; the full set for every arm and seed, including the
> per-segment within-AUCs, is in `eval_sop/results/churn_metrics_by_seed.csv`.

**Per-segment vs global, paired on the same test rows** (`churn_perseg_vs_global_paired.csv`):
- ΔAUC (per-segment minus global) is −0.0081, −0.0071, −0.0128, −0.0165 and −0.0096 for seeds 42, 7, 13, 21 and 99.
  **Every 95% CI excludes 0** (uncorrected, and the five seeds share most of their test rows). The fixes in §11 made this
  gap slightly *wider*, not narrower: the conclusion did not depend on the defects.
- ΔBrier is positive (per-segment worse) in 5/5 seeds, and every CI excludes 0 (uncorrected).
- Within each of the 5 segments, the global model's mean AUC is higher than that segment's own model: mean within-segment
  AUC 0.6300 global vs 0.6189 per-segment, in 5/5 seeds.

What this supports:
- The pipeline is reproducible, and its holdout numbers are honest.
- Isotonic calibration is real and fixes a large miscalibration caused by class weighting (ECE 0.197 → 0.015).
- The model ranks better than chance and better than logistic regression (+0.04 AUC).

What it does not support:
- **"Per-segment beats global."** On this data a single global model is slightly but consistently better.
- **The 17% Brier reduction as a measure of skill.** The README reports it correctly as calibrated vs uncalibrated
  (`scripts/readme_metrics.py`). It measures calibration, not predictive skill. Against a constant base-rate forecast
  (Brier 0.2051) the skill is **4.1% ± 0.5**. Against per-segment base rates it is 3.7% ± 0.5.
- Strong prediction. AUC is about 0.63. The project's feature mapping also gives up some signal: the raw columns reach about 0.67.

Reliability (seed 42, `churn_reliability_project_model.csv`) is good between 0.1 and 0.5, where 95% of rows sit. Above
0.6 the predictions are over-confident: the bin with mean prediction 0.65 has an observed rate of 0.49 (n = 47). The README's
"High Risk ≥ 0.6" tier is the least calibrated region.

## 2. Uplift learners on randomized data (Hillstrom, held-out 30%, n_test = 19,200)

The project's own `compute_uplift_scores_causalml` and `compute_uplift_scores_custom` are fitted on the 70% train split
and scored on the test split. Outcome = website visit. Test ATE = 0.061.

The Qini coefficient is the area between the Qini curve and the random line, divided by n, in incremental visits per test
customer averaged over targeting depths. **Random targeting has an expected Qini of 0.** The "random" row below is one
random score per seed, shown only as a noise reference.

**This normalisation is non-standard — do not compare these numbers to published Qini coefficients.** `common.py:113-122`
divides the area between the curves by `n` (the test-set size), which keeps the units interpretable as "incremental
visits per targeted customer, averaged over depths" and makes the figure comparable across the five seeds and the eight
scores in the table — all of which share one `n`. It is not either common convention: published Qini coefficients are
usually the *unnormalised* area (in whole incremental outcomes, so they scale with dataset size) or that area divided by
the area of the *perfect-targeting* curve (giving a 0–1 efficiency ratio). An 0.0028 here is therefore roughly
0.0028 × 19,200 ≈ 54 incremental visits of area on the unnormalised scale, and is **not** "a Qini of 0.0028" in the sense
any paper or library reports. Every comparison drawn in this section is internal — between rows of this table and
against the 0 of random targeting — which is what this scaling supports. All intervals below are **uncorrected** for
multiple comparisons (80 of them in this section alone; see section 5).

All figures below are from the **5,000-resample** re-run (`n_boot: 5000` in `uplift_hillstrom_run_info.json`, runtime
3,208 s). The point estimates are byte-identical to the earlier 500-resample run — verified, max |Δqini| = 0.0 across all
80 rows — which also re-confirms that run; only the interval bounds moved.

| Targeting score | Qini (mean ± std, 5 seeds) | Qini seed 42 [95% CI] | Seeds with CI excluding 0 | Observed uplift in top 10% (seed 42) |
|---|---|---|---|---|
| Project T+S ensemble (as shipped) | **0.0028 ± 0.0008** | 0.0020 [−0.0000, 0.0040] | **3/5** (seeds 42 and 21 include 0) | 0.087 [0.052, 0.123] |
| Project S-learner (CausalML) | 0.0034 ± 0.0007 | 0.0030 [0.0010, 0.0049] | 5/5 | 0.108 [0.072, 0.145] |
| Project T-learner (CausalML) | 0.0020 ± 0.0010 | 0.0013 [−0.0008, 0.0033] | 3/5 | 0.085 [0.051, 0.119] |
| Project T-learner (custom fallback) | 0.0024 ± 0.0009 | 0.0013 [−0.0007, 0.0034] | 3/5 | 0.071 [0.034, 0.108] |
| Random (expected 0) | observed −0.0005 ± 0.0005 | −0.0005 [−0.0024, 0.0014] | 0/5 | 0.050 [0.020, 0.083] |
| "Churn-score" targeting (highest P(no visit)) | −0.0020 ± 0.0008 | −0.0008 [−0.0027, 0.0012] | 3/5 (all below 0) | 0.051 [0.031, 0.068] |
| Response model (highest P(visit)) | 0.0020 ± 0.0006 | 0.0011 [−0.0008, 0.0030] | 2/5 | 0.076 [0.038, 0.117] |
| Sign-flipped ensemble (= −ensemble, pre-fix bug) | −0.0028 ± 0.0009 | −0.0020 [−0.0040, 0.0001] | 3/5 (all below 0) | 0.065 [0.032, 0.096] |

**What the re-run changed, and why it was run.** At 500 resamples the ensemble's seed-42 lower bound was +2.5e-5 — a
bound two orders of magnitude inside the Monte-Carlo error of 500 draws, which the table then rendered as `0.0000`. A
conclusion ("4 of 5 seeds") rested on it, so the resample count was raised tenfold to settle it. **It settled against the
earlier claim:** at 5,000 resamples that bound is **−2.6e-5**, i.e. the interval now includes zero, and the ensemble's
Qini CI excludes zero in **3 of 5 seeds, not 4 of 5**. The response model likewise fell from 3/5 to 2/5. Nothing else in
the table moved materially. The lesson is in the direction of the change: a bound that small was never evidence either
way, and the "4 of 5" phrasing had given it the weight of a finding.

**Two of the eight rows are not independent tests, and both are restatements of a row above them.**
- The sign-flipped ensemble is `−ensemble` by construction (noted again in section 3).
- The "churn-score" row is `1 − p_resp` (`uplift_hillstrom.py:84`), i.e. the **exact reverse ranking of the response
  model**. Qini depends only on the ranking, so this row is the response-model row read from the bottom up; that it
  scores below zero whenever the response model scores above zero is arithmetic, not evidence. It is kept because it is
  the operational policy the README argues against ("e-mail everyone above 0.7"), and it does show that policy loses on
  this data — but it is not an independent confirmation of the response-model result, and the two cannot be counted as
  two findings.

Ensemble vs response model (`uplift_hillstrom_by_seed.csv`): the ensemble's Qini is above the response model's in
every seed by point estimate, but only marginally. Per seed the gap is +0.0009 (42), +0.0007 (7), +0.0010 (13),
**+0.0001 (21, effectively a tie)** and +0.0015 (99). No paired CI was computed for this difference, so I claim no
significant advantage over a response model.

Uplift by decile (ensemble, mean over seeds, `uplift_hillstrom_deciles.csv`): the observed uplift falls from 0.085
(decile 1) to 0.046 (decile 10). Ranking is monotone in coarse terms, but the spread is modest (1.85×) and
non-monotone in places (deciles 9 and 10 sit above 7 and 8).

**On the secondary outcome (conversion, 0.9% base rate), no method's Qini CI excludes 0 — 0 of 40, in either
direction.** This statement was *false* of the 500-resample artifact, where exactly one interval cleared zero: the
response-model baseline at seed 99, lower bound +2.2e-5. At 5,000 resamples that bound is no longer above zero. It was
the same kind of artefact as the ensemble's seed-42 bound — one marginal crossing out of 40 uncorrected intervals is
precisely what noise produces — and the tenfold resample increase removed it. No project learner's CI excluded zero
under either resample count.

What this supports:
- The project's meta-learners, run unchanged, recover some real treatment-effect heterogeneity on a randomized
  experiment. The ensemble's Qini is above 0 in all 5 seeds by point estimate, with the CI excluding 0 in **3 of 5**
  (uncorrected). The S-learner is the stronger of the two components: 5 of 5.
- They roughly match a response model, whose own CI excludes 0 in only 2 of 5 seeds.
- Targeting the highest-risk customers (the "email everyone above 0.7" approach the README argues against) scores below 0
  in all 5 seeds **by point estimate**. Its 95% CI lies entirely below 0 in only 3 of 5. This score is the reverse
  ranking of the response model, so it is not an independent test of it.

What it does not support:
- A claim that the learners beat a response model.
- Anything about the size of uplift on Cell2Cell. Hillstrom is a different domain, a different treatment and a positive outcome.
- A Criteo uplift subsample was **not run**.

## 3. Cell2Cell "treatment": positivity, and the sign flip

**The positivity violation is confirmed** (`cell2cell_positivity_signflip.json`). Treatment = `Complain | CouponUsed`
(src/uplift_model.py:72). On Cell2Cell, Complain = `CustomerCareCalls > 3` and CouponUsed = `RespondsToMailOffers`.
- P(T=1 | Complain=1) = **1.000**: all 8,472 complainers (16.6% of rows) are treated, and none are controls.
- `Complain` is also an uplift covariate. A 5-fold cross-validated propensity model on the 9 uplift features gives
  e(x) > 0.99 for exactly those 16.6% of rows (minimum e = 0.999) and AUC(T | X) = 0.83. 38–39% of rows fall outside
  [0.1, 0.9]. SMD for SupportRiskScore is 1.02.
- Uplift is fitted and predicted on the same rows (src/uplift_model.py:185-192), with no held-out split and no Qini/AUUC.
- Mean UpliftScore is 0.0246. 20.2% of customers clear the 0.05 Persuadable threshold.
- The pipeline's `validate_uplift_direction` check passes for −ChurnProbability, a pure risk score with no causal content.
  It also passes for uniform random noise in **98 of 200 independent draws (49%, Wilson 95% CI 42–56%)**
  (`direction_check_random_noise.json`). `positivity_signflip.py` reports a single such draw. The check can catch a
  flipped sign on a *fixed* score, but it cannot validate an uplift model.

So the Cell2Cell "uplift" is an observational contrast between treated and control groups. For 1/6 of the population there are no
comparable controls. It should not be described as a causal estimate.

**Sign-flip counterfactual**: buggy score = −UpliftScore, which is exact given the code. Recomputed on today's learners:

| | Fixed (shipped) | Buggy (pre-fix) |
|---|---|---|
| Persuadables | 7,602 | 552 |
| Overlap of the two Persuadable sets | 0 (Jaccard 0.00) | |
| Overlap of the top-100 priority lists | 0 | |
| Pipeline-reported net ROI of its own Persuadables (CLV $500, cost $15) | $166,768 | $13,131 |
| Same set, valued with the fixed model | $166,768 | **−$29,691** |

On Hillstrom the sign-flipped score is −(ensemble) **by construction**, so its Qini is approximately the ensemble's mirrored around 0 (not exactly, because the Qini curve rescales by the treated/control ratio of each prefix).
It is a restatement of the ensemble result, not independent evidence.

What the sign-flip analysis shows on its own: the buggy pipeline still reported positive ROI for every customer it
selected. Its own reported numbers could not reveal the bug.

## 4. Agent vs no-tools LLM vs rule policy (40 customers × 3 seeds)

**Setup.**
- **Agent:** the project's **Python** agent, `src/agent_loop.generate_retention_action_agentic`, with its system prompt,
  6 tools from `src/agent_tools.py`, max 5 rounds and its own JSON parsing, all unmodified. Only the client is swapped:
  `agent_loop.Groq` is replaced by a shim that calls the native Ollama `/api/chat`.
- **Model and sampling:** `qwen2.5:7b` (Ollama tag, Q4), `num_ctx=8192`, `temperature=0.7`, seeds {0, 1, 2}. One request at a time on :11434.
  The model was unloaded afterwards (`keep_alive: 0`; `/api/ps` was empty).
- **Context budget:** the full 6-tool schema fits, so no tool subset or output truncation was needed. The harness estimates every prompt at
  chars/3 tokens, which over-counts: on cold calls Ollama's `prompt_eval_count` was 1,353 tokens where the estimate said 2,036.
  It counts a run as a failure if estimate + `max_tokens` (1,500) > 8,192. The largest prompt estimate was 3,993 tokens,
  and 0/240 LLM runs overflowed (`agent_eval_token_check.json`).
- **Label leak stripped:** the harness drops the `actually_churned`, `churn` and `Churn` keys from every tool result. The segment-level
  `actual_churn_rate` aggregate is kept.
- **No-tools LLM (arm C):** the same model and JSON contract, and the same facts the agent's first user message gets
  (segment, churn probability, uplift, customer type, CLV), but no tools. Its system prompt is `SYSTEM_PROMPT_BATCH`
  **truncated** at "Use the available tools" (`agent_eval.py:190-193`), which also drops the numbered 5-step
  gather-then-recommend sequence (`src/agent_loop.py:36-41`). So arm C differs from the agent in tool access *and* in
  prompt content.
- **Control arms B and D** (`eval_sop/agent_third_arm.py`) strip the tool loop apart one layer at a time. Both run
  `SYSTEM_PROMPT_BATCH` **verbatim** (5-step tool sequence intact, which still instructs the model to call tools that are
  not supplied), one shot, same 40 customers × 3 seeds, same model, same lenient parser as arm C, and both retain full
  untruncated replies.

  | Arm | System prompt | Tool calls | Tool output in context | Rounds |
  |---|---|---|---|---|
  | **A** `agent` | full `SYSTEM_PROMPT_BATCH` | 6 live tools | yes, from its own calls | up to 5 (~4.7 used) |
  | **D** `tool_output_no_calls` | full, verbatim | **none** | **yes, pasted in** | 1 |
  | **B** `full_prompt_no_tools` | full, verbatim | none | no | 1 |
  | **C** `no_tools_llm` | truncated (5-step sequence dropped) | none | no | 1 |

  **A vs B** isolates the whole tool loop; **B vs C** the prompt; **D vs B** having tool output in context; **A vs D**
  function-calling itself, with the same information in both.

  **How arm D's tool output was built.** The original run stored `(tool, args)` but not tool *results*, so they are
  **recomputed**. Every tool in `src/agent_tools.py` is a deterministic pure function of `(df, playbook, args)`, so this
  reproduces them exactly rather than approximating them. Arm D pastes the prompt's own prescribed 5-step sequence
  (`src/agent_loop.py:36-41`) evaluated for that customer, through the harness's own leak-key stripping (verified: 0
  leak-key violations across all 40 customers, and `actually_churned` is absent while the segment-level
  `actual_churn_rate` is kept, exactly as in arm A). The playbook is queried for the customer's **true** top signed SHAP
  driver and ROI is computed at the pipeline's own flat $15 cost. Largest prompt was 1,941 estimated tokens, well inside
  the 8k budget; 0/120 overflows.
  - **Caveat, stated because it cuts one way:** arm D's context is *correct by construction*, whereas the live agent
    queried the playbook for the true top driver in only 71% of runs. Arm D is a mildly **favourable** reconstruction of
    arm A's context, not a replay of it. If clean, correct tool output still produces the behaviour, noisy real output is
    not the exculpating factor.
- **Rule policy:** intervene iff Persuadable and NetROI > 0, using the playbook entry for the top signed SHAP driver.
- **Never-intervene:** a trivial baseline that always outputs "do not intervene".
- **Scenario set:** `eval_sop/agent_scenarios.csv`, 10 customers per CustomerType.
- **Scoring:** programmatic. **There is no ground-truth retention outcome**, because Cell2Cell has no randomized intervention.
  "Agrees with rule" therefore means consistency with the project's own decision rule, not business value.
- **Parsing (asymmetric, and only partly repairable).** The agent is scored with the project's **strict** parser
  (`src/agent_loop.py:243` — strip fences, then `json.loads` the whole reply), the no-tools arm with the **lenient**
  `parse_json` (`agent_eval.py:156-170` — strip `<think>`, then regex-extract the first `{...}`). Both arms are now
  measured both ways:
  - *No-tools arm* — re-run keeping the raw reply: **100% valid under both** parsers
    (`agent_eval_no_tools_parser_check.csv`). The asymmetry costs this arm nothing.
  - *Agent arm* — re-parsed offline from the committed replies
    (`agent_parser_check.py --agent` → `agent_eval_agent_arm_parser_check.{csv,json}`): of the 20 strict-parser
    failures, the lenient parser recovers **1**, and **19 of 120 runs cannot be decided at all**. The project's parser
    stores only `raw_response = raw[:500]`, so all 20 recorded failures are truncated at exactly 500 characters; a reply
    whose JSON object had not closed by then is unknown under the lenient parser, not invalid. The symmetric figure is
    therefore a **bound, not a point estimate: agent lenient-parser validity is between 101/120 (84.2%) and 120/120**,
    against 120/120 for the no-tools arm. Closing the gap needs a re-run that retains full replies, not a re-parse.
  Every agreement figure below is computed under the strict parser for the agent and the lenient one for the no-tools
  arm, i.e. as originally scored; the 19 indeterminate runs mean the agent's valid-JSON rate could be anywhere from
  83.3% to 100% had it been scored leniently.
- **Infrastructure errors:** none (0 excluded).

**Headline: Sleeping Dogs.** These are customers the pipeline labels "do not contact", and the system prompt says not to intervene on them.
- **A** agent, tools: **20 of 30 Sleeping Dog runs (0.67; clustered-bootstrap CI 0.43–0.87, uncorrected)**.
- **D** full prompt, no calls, tool output pasted in: **9 of 30 (0.30; CI 0.00–0.60)**.
- **B** full prompt, no tools, no tool output: **5 of 30 (0.17; CI 0.03–0.33)**.
- **C** truncated prompt, no tools: **1 of 30 (0.03; CI 0.00–0.10)**.
- The rule policy and never-intervene: 0 of 30.

**The prompt is ruled out and the tool loop is confirmed. The split of the tool effect into function-calling versus
tool-output-in-context is *not* resolved: the two outcome definitions disagree.** Paired on customer, 2,000 clustered
bootstrap resamples, shared draws (`agent_sleeping_dog_decomposition.csv`, from `agent_section4_figures.py`):

| Contrast | Isolates | Sleeping Dog only (n=10) | Lost Cause **or** Sleeping Dog (n=20) |
|---|---|---|---|
| A − B | the whole tool loop | **+0.500 [0.233, 0.733]** ✓ | **+0.283 [0.117, 0.467]** ✓ |
| D − B | having tool output in context | +0.133 [−0.200, 0.467] ✗ | **+0.300 [0.067, 0.533]** ✓ |
| A − D | function-calling itself | **+0.367 [0.133, 0.633]** ✓ | −0.017 [−0.267, 0.233] ✗ |
| B − C | the prompt | +0.133 [0.000, 0.300] ✗ | +0.067 [0.000, 0.150] ✗ |
| A − C | both together (original contrast) | **+0.633 [0.400, 0.833]** ✓ | **+0.350 [0.183, 0.533]** ✓ |

(✓ = 95% interval excludes zero, uncorrected.)

**Robust across both definitions:** the whole tool loop raises do-not-contact intervention (A−B excludes zero either
way), and the prompt difference does not (B−C fails either way). That is the finding.

**Not robust — and I am not going to pick the convenient one.** On the headline Sleeping-Dog metric the resolved
component is **function-calling itself** (+0.367), with tool output in context contributing an unresolved +0.133: arm D
recovers only about a third of the gap, so merely *having* the information is not what does it. On the broader
do-not-contact set the ordering reverses exactly — tool output accounts for +0.300 and function-calling for −0.017.
Two reasons for the contradiction, and the first is a measurement artifact in the *agent's* favour:
- **The agent's Lost-Cause rate is deflated by its own parse failures.** 13 of its 30 Lost-Cause replies are unparseable,
  and `score()` records an unparseable reply as `intervene = False`. So the agent is credited with non-intervention on
  43% of Lost-Cause runs simply for failing to emit JSON (2/30 = 0.067 all runs; 2/17 = 0.118 parseable only). Arm D
  parses 120/120 and so gets no such credit. The Sleeping-Dog metric is far cleaner — only 3 of 30 unparseable — which is
  why I treat it as the headline.
- **Arm D is genuinely anomalous on Lost Causes**, at 14/30 = 0.467 against the agent's 0.118 parseable-only. Pasting a
  complete, correct tool block appears to push this model toward intervening on Lost Causes *more* than the live loop
  does. That is a real effect in arm D, not an artifact, and it is unexplained.

So: the tool loop is the driver (robust); on the headline metric the active ingredient looks like the act of calling
rather than the information returned (+0.367, resolved); and that last attribution rests on 10 customers and does not
survive a change of outcome definition, so it is reported as the better-supported of two readings rather than settled.

### Finding: the 17% invalid-JSON rate is a tool-loop defect — not the prompt, and not context length

**Arms B and D each produced valid JSON in 120/120 runs; the agent manages 100/120.** All three run the **identical**
system prompt and JSON contract, so the agent's 17% parse-failure rate is not about how the output format is specified.
Arm D sharpens this further: its prompts are the longest of the three single-shot arms (up to 1,941 estimated tokens,
against 667 for arm B) because they carry the whole pasted tool block, and it still parsed **120/120**. So the failure is
not "long context" generically either — it is the **multi-round loop structure** specifically.

This is a separable, independently fixable defect: it needs a tolerant parser and a higher `raw_response` retention limit
(§9), not prompt engineering or a bigger context. It also distorts the agent's scores in two opposite directions at once
— it drags the agreement rate down, while *flattering* the agent on do-not-contact metrics, because an unparseable reply
is scored as a non-intervention (see the decomposition above). Fixing the parser would make the agent's Lost-Cause
behaviour look worse, not better.

### Finding: the Sure-Thing and Sleeping-Dog failures are two distinct defects the agreement rate was averaging together

The single "agrees with project rule" number hides two unrelated failure modes with different causes, and this is why the
two no-tool arms sit near the never-intervene baseline overall while behaving nothing like each other by customer type:

| Failure | Who shows it | Cause | Tools change it? |
|---|---|---|---|
| **Intervenes on Sleeping Dogs** (do-not-contact) | agent 20/30, arm D 9/30, arm B 5/30, arm C 1/30 | the tool loop | **yes — this is the +0.50** |
| **Intervenes on every Sure Thing** | **all four** LLM arms: B 30/30, D 30/30, C 29/30, agent 24/28 parseable | model + prompt | **no — identical with and without tools** |

Removing the tools repairs the Sleeping-Dog column almost completely (arm B: 83.3% agreement, against 25.9% with tools)
and leaves the Sure-Thing column at **0% in every single arm, including the one with the full tool block pasted in**. So a
reader who sees only "75% for the no-tools arm, 53% for the agent" would conclude the no-tools configuration is simply
better; in fact it trades one failure for the other, and its 75% is the arithmetic of getting three customer types right
and one entirely wrong. Reporting the aggregate alone would have obscured both the real tool effect and a prompt-level
defect that no amount of tool work will fix — the Sure-Thing failure is invariant across all four LLM configurations
tested, which is the strongest evidence in this section that it lives in the model and prompt rather than the plumbing.

**Estimators, stated explicitly.** Two are in play, and on the parseable-only subset they differ by 5.8 points, so each
row below names the one it uses:
- **ratio** — `sum(numerator) / sum(denominator)` over runs. The run is the unit, so a customer with more parseable
  replies gets more weight.
- **per-customer** — the unweighted mean over the 40 customers of each customer's own rate. This is what `agent_eval.py`
  writes to `agent_eval_summary_ci.csv`.

They coincide wherever every customer has the same denominator (all the 30-run and 120-run rows). They diverge only on
the parseable-only rows, where the denominator varies by customer. All CIs are 95% percentile bootstrap, 2,000
resamples, **clustered on customer** (a resample draws customers and keeps all their runs; the three seeds are reruns of
the same 40 customers, so the customer is the independent unit). **No multiple-comparison correction is applied
anywhere.** Every figure in this section is recomputed by `eval_sop/agent_section4_figures.py` →
`agent_section4_figures.csv` + `_info.json`; where a row also exists in `agent_eval_summary_ci.csv` the two bootstraps
agree to within ±0.01 on the bounds.

| Metric | Estimator | A: Agent (tools) | D: Output pasted, no calls | B: Full prompt, no tools | C: No-tools LLM | Rule policy | Never-intervene |
|---|---|---|---|---|---|---|---|
| Intervened on a Sleeping Dog (/30 SD runs) | ratio = per-cust. | **0.67** (20/30) [0.43, 0.87] | **0.30** (9/30) [0.00, 0.60] | **0.17** (5/30) [0.03, 0.33] | 0.03 (1/30) [0.00, 0.10] | 0 (0/30) | 0 (0/30) |
| Intervened on Lost Cause or Sleeping Dog (/60 LC+SD runs) | ratio = per-cust. | **36.7%** (22/60) [20.0, 55.0] | 38.3% (23/60) [18.3, 60.0] | 8.3% (5/60) [1.7, 18.3] | 1.7% (1/60) [0.0, 5.0] | 0 (0/60) | 0 (0/60) |
| *Same event, /120 all runs (as `summary_ci.csv` reports it)* | ratio = per-cust. | 18.3% (22/120) [9.2, 29.2] | 19.2% (23/120) | 4.2% (5/120) | 0.8% (1/120) [0, 2.5] | 0 | 0 |
| Valid final JSON (parser as scored) | ratio = per-cust. | 83.3% (100/120) [75.0, 90.8] | **100% (120/120)** | **100% (120/120)** | 100% (120/120) | 100% | 100% |
| Agrees with project rule, **parseable replies only** | **ratio** | **53.0%** (53/100) [39.4, 66.7] | 55.8% (67/120) [40.8, 70.0] | 70.8% (85/120) [56.7, 83.3] | 75.0% (90/120) [62.5, 87.5] | 100% (by definition) | **75.0%** (90/120) [60.0, 87.5] |
| Agrees with project rule, **parseable replies only** | **per-customer** | **58.8%** (53/100) [45.0, 71.7] | 55.8% (67/120) [40.8, 70.0] | 70.8% (85/120) [56.7, 83.3] | 75.0% (90/120) [60.8, 87.5] | 100% (by definition) | **75.0%** (90/120) [60.0, 87.5] |
| Agrees with project rule, all runs (unparseable = disagree) | ratio = per-cust. | 44.2% (53/120) [33.3, 55.0] | 55.8% (67/120) [40.8, 70.0] | 70.8% (85/120) [56.7, 83.3] | 75.0% | 100% | 75.0% |
| Intervention rate | ratio | 60.8% | 69.2% [54.2, 82.5] | 54.2% [39.2, 68.3] | 50.0% | 25.0% | 0% |
| Recommended cost tier > model-expected value (uplift × $500) | per-customer | 21.7% [13.3, 31.7] | 26.7% (32/120) | 1.7% (2/120) | 0% | 15.0% [5.0, 27.5] | 0% |
| Playbook query names the customer's true top SHAP driver | per-customer | 70.8% [58.3, 83.3] | 100% by construction | n/a (no tools) | n/a | n/a | n/a |
| ROI tool called with the customer's true uplift (±0.01) | per-customer | 99.2% [97.5, 100] | 100% by construction | n/a (no tools) | n/a | n/a | n/a |
| Tool calls / latency | mean | 4.7 / 21.1 s | 0 / 6.3 s | 0 / 5.0 s | 0 / 5.2 s | 0 / 0 s | 0 / 0 s |
| Largest prompt (est. tokens, 8k budget) | max | 3,993 | 1,941 | 667 | 667 | n/a | n/a |

Two notes on the table itself, both corrections to the previous version of this file:
- The Lost-Cause-or-Sleeping-Dog row previously printed 18.3% while labelling the row "(60 runs)". 18.3% is 22/**120** —
  `agent_eval.py` scores that flag over every run, not over the 60 eligible ones. Both denominators are now shown and
  labelled; 22/60 = 36.7% is the one comparable to the 20/30 Sleeping-Dog row above it.
- The parseable-only agreement row previously mixed estimators with the row beneath it: 53% is the ratio estimator,
  while the 44.2% below it is the committed per-customer mean. Under the per-customer estimator the parseable-only
  figure is 58.8%. Both are now shown.

How to read the agreement rows:
- The rule intervenes on only the 10/40 Persuadables, so never intervening already agrees 75% of the time.
- The no-tools LLM's 75% therefore does **not** beat the trivial baseline. Its disagreements are almost all Sure Things
  (it intervened on 29/30).
- The agent is below both on parseable replies under **either** estimator (53.0% ratio, 58.8% per-customer), and its CI
  overlaps the 75% baseline in both cases, so this is a direction, not a significant difference.

By customer type, the agent's agreement on parseable replies (ratio estimator, with the per-customer value beside it;
CIs uncorrected and wide — each cell rests on 10 customers):

| Customer type | A: Agrees with rule | A: Ratio [95% CI] | A: Per-customer [95% CI] | D: Output pasted [95% CI] | B: No tool output [95% CI] |
|---|---|---|---|---|---|
| Persuadable | 27/28 | 96.4% [89.3, 100] | 96.7% [90.0, 100] | 30/30 = 100% | 30/30 = 100% |
| Lost Cause | 15/17 | 88.2% [75.0, 100] | 91.7% [80.0, 100] | 16/30 = 53.3% [23.3, 83.3] | 30/30 = 100% |
| Sleeping Dog | 7/27 | 25.9% [9.9, 46.2] | 31.7% [11.7, 55.0] | 21/30 = 70.0% [40.0, 100] | 25/30 = 83.3% [66.7, 96.7] |
| Sure Thing | 4/28 | 14.3% [3.6, 25.9] | 15.0% [3.3, 26.7] | **0/30 = 0%** | **0/30 = 0%** |

The by-type pattern carries most of this section's information. Removing the tools while keeping the prompt fixed (arm B)
repairs the Lost Cause and Sleeping Dog columns almost completely — 100% and 83.3%, against 88.2% and 25.9% with tools —
but leaves Sure Things at **0/30**: arm B intervenes on every Sure Thing, as arm C does (29/30) and as arm D does (30/30).
Arm D, which holds the same information as the agent but makes no calls, sits between A and B on Sleeping Dogs (70.0%) and
is the **worst of every arm on Lost Cause (53.3%)** — the anomaly discussed in the decomposition above. So the Sure-Thing
failure is a property of the *model and prompt*, untouched by tool access or tool information, while the Sleeping-Dog
failure is the tool-driven one. Both no-tool arms land near the never-intervene baseline overall precisely because they
trade a fixed Sure-Thing error for near-perfect do-not-contact behaviour. Read the agent's Lost-Cause 88.2% against its 13
unparseable Lost-Cause replies, which this parseable-only column excludes and which the rate columns score as
non-intervention — a good part of why the agent looks compliant there and arm D does not.

13 of the agent's 20 unparseable replies are on Lost Cause customers (Sleeping Dog 3, Persuadable 2, Sure Thing 2), so
the Lost Cause row rests on the fewest parseable replies of any type. Across-seed std: agent valid-JSON 0.113,
agreement 0.104 — a std over three correlated reruns of the same 40 customers, not a standard error.

What this supports:
- **Giving this 7B model its tool loop is what makes it contact Sleeping Dogs.** With the project's full prompt held
  fixed, adding the tools raises the Sleeping-Dog contact rate from 5 of 30 to 20 of 30 — **+0.50, 95% CI [0.23, 0.73]**,
  paired on customer, and the same contrast excludes zero on the broader do-not-contact set too (+0.283 [0.117, 0.467]).
  The prompt difference that confounded the original two-arm comparison accounts for +0.13 at most, with an interval
  reaching zero on both outcome definitions. **The prompt is ruled out; the tool loop is confirmed.**
- On the headline Sleeping-Dog metric, the active ingredient looks like **the act of calling rather than the information
  returned**: function-calling alone is +0.367 [0.133, 0.633] while pasting the identical tool output in gets only
  +0.133 [−0.200, 0.467]. This reverses on the Lost-Cause-inclusive metric and rests on 10 customers, so it is the
  better-supported of two readings, not a settled result.
- The agent's 17% invalid-JSON rate is tool-loop-specific — not the prompt and not context length. Arms B **and D** share
  the identical system prompt and parsed 120/120 each, and arm D's prompts are three times longer than arm B's.
- The tool loop does carry the right numbers into the reasoning: the true uplift reaches the ROI tool in 99% of runs, and
  the playbook is queried for the true top driver in 71%. So the failure is not that the tools deliver bad data — they
  deliver the right data, and the model then acts against the instruction it was given.
- Tools did not improve agreement with the project's rule. On parseable replies the agent agrees less often than a
  trivial never-intervene policy.
- One plausible mechanism, **not verified**: Sleeping Dogs often have a small positive uplift, and the ROI tool then
  returns "Intervene — positive ROI" at the low cost the agent itself chooses.

What it does not support:
- Any claim about the deployed agent (TypeScript, 12 tools, Supabase, a hosted larger model) or about retention outcomes.
- Generalisation beyond one 7B model, 40 customers and three seeds.
- Small cells: the per-type rates rest on 30 runs (10 customers) each, and the decomposition on 10 Sleeping-Dog customers.
- **A settled split of the +0.50 into function-calling versus tool output in context.** Arm D was run to do exactly this
  and **did not resolve it**: the two outcome definitions give opposite answers (function-calling +0.367 vs tool output
  −0.017 on Sleeping Dogs; +−reversed on Lost Cause ∪ Sleeping Dog). With 10–20 customers per contrast, and with the
  agent's Lost-Cause rate deflated by 13 unparseable replies scored as non-intervention, the decomposition is
  underpowered and partly confounded by the parser defect. Settling it needs more customers and a tolerant parser first.
- **A size for the prompt effect.** +0.13 [0.000, 0.300] on 10 customers is consistent with zero and with a real effect
  about a quarter the size of the tool effect. It is not evidence of no prompt effect.
- **An explanation for arm D's Lost-Cause behaviour.** Pasting a complete, correct tool block makes this model intervene
  on Lost Causes more than the live loop does (0.467 vs 0.118 parseable-only). That is unexplained and is the clearest
  open question in this section.

Other notes:
- The rule baseline picks "High ($50–100)" interventions for customers whose model-expected value is under $50,
  because NetROI assumes a flat $15 cost.
- `agent_eval_human_validation.csv` (seed 0, 120 rows) has blank columns for human rating. It has not been human-validated.

**Label leakage, found by inspection:** `lookup_customer_details` returns `actually_churned`, the realized label
(src/agent_tools.py:204). The deployed route returns `select("*")` from `customers`, which includes `churn`
(route.ts:422-429, migrate_to_supabase.py:66,104). The harness strips these keys; src/ is unchanged. Before the strip
was in place, no agent run reached the model, so no result here was produced with the leak.

**Paid-API needs: none required.** For reference, an optional run of the deployed-style agent on a hosted model is
about 40 customers × 3 seeds × ~20k input + ~1.5k output tokens, roughly 2.4M input and 0.2M output tokens. At typical
open-model hosted prices ($0.1–0.6/M in) that is **≈ $0.5–2**. At frontier prices (~$3/M in, ~$15/M out) it is **≈ $10**.
These are estimates from list-price ranges, not quotes, and nothing was spent.

## 5. Threats to validity

- Segmentation (K-Means) was fit on all rows, holdout included. It is unsupervised and uses no labels, but it is mildly transductive.
- One dataset for churn. The Cell2Cell feature mapping (e.g. CreditRating → "SatisfactionScore") is the project's, and I kept it as is.
- The bootstrap CIs ignore training variance. The seed std covers part of it. Five seeds is few.
- **The seeds are not independent, so the ± values are not standard errors.** Both eval scripts redraw a split of the
  *same* rows per seed (`churn_eval.py:72`, `uplift_hillstrom.py:118`) rather than drawing fresh data. With a 20% churn
  holdout, two seeds' test sets share about 20% of their rows in expectation; with Hillstrom's 30% holdout, about 30%.
  Pooled across five seeds, nearly every row appears in several test sets. The across-seed std therefore measures
  split-and-fit sensitivity on one fixed dataset, not sampling variability of the population, and it is biased low as an
  estimate of the latter. Read every `± x` in this file as a stability indicator; do not divide it by √5, and do not
  build a t-interval from it.
- **No multiple-comparison control anywhere in this file.** Section 2 alone reports 80 nominal 95% intervals (8 scores ×
  5 seeds × 2 outcomes); sections 1 and 4 add dozens more. At the nominal level, 5% of them are expected to exclude zero
  by chance even if every underlying effect were null, which is roughly 4 of section 2's 80. Each interval is valid on
  its own; the family is not, and no interval below is corrected. Where a conclusion rests on a single marginal CI, that
  is said in place.
- **No temporal validation is possible.** Neither dataset carries a calendar date, timestamp or cohort column. All 90
  columns of `data/processed/segmented.parquet` are numeric or categorical with no datetime dtype; the only time-flavoured
  fields are *durations* measured backwards from one unstated snapshot (`MonthsInService`, `DaySinceLastOrder`,
  `CurrentEquipmentDays`), which order customers by tenure but give no common calendar axis to split on. Hillstrom's
  columns (`recency, history_segment, history, mens, womens, zip_code, newbie, channel, segment, visit, conversion,
  spend`) are likewise a single cross-section. Every split in this file is therefore stratified random, and each holdout
  AUC is an *interpolation* estimate: how well the model scores customers
  drawn from the same period as its training rows. It is not a forward-in-time estimate, and it is silent on drift,
  seasonality, or the rolling retrain the deployed system would actually need. A production churn model should be
  back-tested on later periods; nothing here does that, or can.
- CatBoost results depend on `thread_count`. The eval pins it to 2. At the project default (-1) the global baselines
  differed by up to 0.0016 AUC (#5), which does not change any conclusion.
- Hillstrom uses a different treatment, outcome and domain from churn retention. It validates the learners, not the Cell2Cell targeting.
- The sign-flip result is a counterfactual recomputation on today's learners, not a replay of the historical buggy run.
- Environment: the base anaconda env has numpy 2.5, which breaks numba, so `import causalml` fails. `src/uplift_model.py`
  then **silently** switches to its custom T-learner (it logs only a warning). The committed artifacts say "CausalML", so they were
  built in a working env. I ran everything in a separate venv with numpy 2.4.6 (`eval_sop/requirements-eval.txt`).
  `requirements.txt` pins `numpy<2.1`; I did not test that exact pin. **Corrected 2026-10-07:** it is tested now —
  everything in §11 ran at numpy 2.0.2 with causalml, pacmap and numba all importable, so the uplift stage went
  through CausalML rather than the fallback. On the ambient numpy 2.5 env `src/pipeline.py` does not reach the
  fallback at all: it dies at `import pacmap` in stage 2.

## 6. SOP-ready sentences (each is backed by a file in `eval_sop/results/`)

1. "I re-evaluated my churn pipeline against baselines. On 10,211 held-out Cell2Cell customers, its calibrated
   per-segment CatBoost models reached AUC 0.63 (95% CI 0.62–0.64), but a single global model trained the same way was
   slightly and consistently better (ΔAUC −0.006 to −0.012 across five seeds, every paired CI excluding zero,
   uncorrected for multiple comparisons). I have corrected my project README accordingly."
   - Precise scope: the README correction is on branch `sop-eval`, not yet merged. The five seeds are overlapping splits
     of the same rows, so the across-seed spread is not a standard error.
2. "On the randomized Hillstrom e-mail experiment (19,200 held-out customers), my uplift learners' Qini was positive in
   all five seeds (0.0028 ± 0.0008 — a spread across correlated splits of the same rows, not a standard error), and the
   95% bootstrap CI excluded zero in three of five seeds (5,000 resamples, uncorrected for multiple comparisons). They
   were only marginally above a plain response model, and tied with it in one seed. Targeting by predicted risk — the
   exact reverse of the response-model ranking, so not an independent test — scored below zero in all five seeds."
3. "Isotonic calibration cut expected calibration error from 0.20 to 0.015. The 17% Brier reduction I reported was
   against uncalibrated, class-weighted outputs. Against a base-rate forecast the skill is ~4% (3.7% against
   segment-specific base rates)."
4. (Agent, optional) "In a programmatic evaluation of my tool-using retention agent (40 customers × 3 seeds, local 7B
   model) I ran four arms to find out what actually drives its failures: the full agent; the same prompt with the tools
   removed but their output pasted in; the same prompt with neither; and a shorter prompt with no tools. **The tool loop
   is what makes the model contact customers the pipeline marks 'do not contact'** — with the prompt held fixed, the loop
   raised Sleeping-Dog contact from 5 of 30 runs to 20 of 30, +0.50 (95% CI 0.23–0.73, paired on customer), and the
   contrast held on the wider do-not-contact set (+0.28, CI 0.12–0.47). The prompt difference that I had originally
   confounded with tool access accounts for +0.13 at most, with an interval reaching zero. The tools delivered correct
   data — the true uplift reached the ROI tool in 99% of runs — so the model was overriding its instruction on good
   inputs. Two further findings separate defects the single agreement rate had merged: the agent's 17% invalid-JSON rate
   is specific to the multi-round loop (both single-shot arms on the identical prompt parsed 120/120, including one with
   three times the context), and its habit of intervening on every Sure Thing is a model-and-prompt defect untouched by
   tools — 0% agreement on Sure Things in all four LLM arms. On parseable replies the agent agreed with the system's own
   targeting rule 53% of the time (58.8% weighting customers equally rather than runs), below a trivial never-intervene
   baseline (75%)."
   - Precise scope: one 7B model, 40 customers, 3 seeds, uncorrected for multiple comparisons; the decomposition rests on
     10–20 customers per contrast.
   - **What this does not settle:** whether the tool effect is function-calling itself or merely having tool output in
     context. The fourth arm was run to answer that and the two outcome definitions disagree (function-calling +0.37
     [0.13, 0.63] on Sleeping Dogs; tool output +0.30 [0.07, 0.53] on the Lost-Cause-inclusive set, where
     function-calling is ~0). The agent's Lost-Cause rate is also deflated by 13 unparseable replies that score as
     non-intervention, so that half of the decomposition is confounded by the parser defect. Do not claim the split.

## 7. README corrections

Applied on this branch (#6):
- Per-segment bullet (README "Why it's shaped this way"): added the measured result that a global model was better.
- Calibration section (generated): the 17% now says it is relative to the uncalibrated, class-weighted outputs. Brier
  skill vs base rate is added beside it: 4.0%, and 3.5% vs per-segment base rates. These are computed from the committed
  artifacts at seed 42 (thread_count=-1), so they match this file's seed-42 values, not the 5-seed means.

Proposed, not applied:
- State that the Cell2Cell uplift has a positivity violation (all `Complain=1` customers are treated). It is neither
  validated out of sample nor causal. Cite the Hillstrom held-out Qini for the learners instead.
- Say that the direction check passes for random scores about half the time, so it cannot validate an uplift model.

## 8. Change log

| # | Commit | Change | Why | Evidence | Preserved |
|---|---|---|---|---|---|
| 1 | `612d984`, `00b3e59` | Added `eval_sop/` (scripts, `results/`, `agent_scenarios.csv`, `.gitignore`) and this file | Deliverable | Each script names its inputs and outputs. Seed-42 reproduction matches the README exactly | No existing file modified |
| 2 | `94e80ca` | `agent_eval.py`: native-Ollama shim (`num_ctx=8192`, sequential), token-budget accounting that counts overflow as failure, leak-key stripping, `--limit`/`--scenarios-only`, resumable checkpoint, `keep_alive: 0` unload, two trace-grounding metrics | Coordinator constraints. The OpenAI-compatible endpoint cannot set `num_ctx` | `agent_eval_token_check.json` (max est. 3,993 tokens, 0 overflows). Direct check: the unwrapped `lookup_customer_details` returns `actually_churned`, the wrapped one returns no churn-label key | `src/agent_loop.py`, `src/agent_tools.py` unmodified |
| 3 | `94e80ca` | Added `agent_parser_check.py` | The agent was scored with the project's strict parser and the no-tools arm with a lenient one | `agent_eval_no_tools_parser_check.csv`: 100% valid under both | — |
| 4 | `6263f3e` | Fixed the `agent_eval.py` docstring (native `/api/chat`, not the OpenAI-compatible endpoint). Stopped tracking `agent_eval_rows_including_infra_errors.csv` and the checkpoint jsonl | Reviewer: stale docstring; duplicate artifacts | `cmp` showed the infra CSV byte-identical to `agent_eval_rows.csv` (0 infra errors); it is now written only when rows are excluded. The 468 KB checkpoint has the same 360 rows and raw outputs as `agent_eval_rows.csv` + `agent_eval_raw_outputs.jsonl` (verified equal); it is now a gitignored local resume file | Results unchanged |
| 5 | `39db206` | `churn_eval.py`: pinned CatBoost `thread_count=2`, added the segment-base-rate skill column, re-ran | Reviewer reproduced global AUC 0.6381 → 0.6389 at thread_count=2 | Confirmed: global seed 42 0.6381 → 0.6389. Max \|ΔAUC\| across seeds and models is 0.0016 (globals and raw reference only). Per-segment and logistic numbers are identical. ΔBrier (per-segment − global) CI now excludes 0 in 5/5 seeds (was 4/5). Two consecutive runs gave byte-identical CSVs | README-reproduction path still uses the project's `thread_count=-1` |
| 6 | `83e0148` | README: qualified the per-segment bullet. In `scripts/readme_metrics.py`, the calibration template now names the 17% baseline and adds the base-rate skill (computed from artifacts). Section regenerated | Reviewer: the SOP sentence said "retracted" while the README still made the claim. The 17% was correct as calibration-vs-uncalibrated (readme_metrics.py:69-70) and is kept | `readme_metrics.py --check` fails after the template edit and passes after `--write`. pytest 73 passed | All other README text, including the rationale for per-segment models |
| 7 | `0e098db` | Added `direction_check_random.py` | Reviewer: "passes for random noise" was one draw | 98/200 draws pass (49%, Wilson 42–56%) | — |
| 8 | `53bc9e3` | Added `eval_sop/requirements-eval.txt` | Reviewer: no lock file for the eval env | Versions read from the venv used for every run | Repo `requirements.txt` untouched |
| 9 | `57d133e` | `eval_sop/*.py`: per-file `# ruff: noqa: I001` and `zip(strict=True)` | CI runs `ruff check .`. Main passes, but eval_sop added 7 errors, which would have failed CI on merge | `ruff check .` → all passed. Re-running `positivity_signflip.py` and `agent_eval.py --scenarios-only` produced byte-identical outputs | Import order kept (it is load-bearing) |
| 10 | `5057d59` | RESULTS.md rewritten for the review fixes | Reviewer items 2, 3, 4, 6 | Numbers above, from the committed CSVs | — |

### Round-2 review (independent reviewer: PASS-WITH-FIXES). Every claim was re-verified against the committed artifacts before the wording was touched.

| # | Commit | Change | Why | Evidence | Preserved |
|---|---|---|---|---|---|
| 11 | `c7c1159` | §2 conversion sentence. **The reviewer was right about the committed 500-resample artifact and the re-run then superseded the finding**, so the prescribed replacement wording was *not* applied verbatim — it would have been false of the new artifacts | R2 #1 | At 500 boots exactly one of the 40 conversion rows had `qini_lo > 0`: `baseline_response_model`, seed 99, `qini_lo` = **+2.196e-5** — reviewer verified correct. At 5,000 boots that bound is no longer above zero and **0 of 40 conversion CIs exclude zero in either direction**, which makes the original sentence true as written. §2 now states the figure *and* records that it was false at 500 resamples, so the correction is visible rather than silently reverted | The original sentence, now re-earned rather than assumed |
| 12 | `c7c1159` | **Raised `N_BOOT` 500 → 5,000** in `uplift_hillstrom.py` and re-ran the whole uplift analysis — **the reviewer's preferred route, taken; the fallback wording was not needed**. Runtime 3,208 s | R2 #2. The "4 of 5 seeds" claim rested on seed 42's ensemble `qini_lo` = +2.487e-5, two orders of magnitude inside the Monte-Carlo error of 500 resamples, and the table rendered it as `0.0000` | Verified the bound first (+2.487e-5 at 500 boots). **The re-run settled it against the earlier claim:** that bound is **−2.6e-5** at 5,000 resamples, so the ensemble's CI excludes zero in **3 of 5 seeds, not 4 of 5**; the response model fell 3/5 → 2/5. All 80 point estimates reproduced **byte-identically** (max \|Δqini\| = 0.0), which independently re-verifies the original run and confirms only the bounds moved. §2, SOP sentence 2 and the "what this supports" bullets all updated to 3/5 | Seeds, splits, features, learners and model order unchanged; the 500-boot CSV was kept outside the repo for the comparison only |
| 13 | `c7c1159` | **Reworded the §4 tools claim and SOP sentence 4** to a configuration contrast, not a tool ablation | R2 #3. **Verified, reviewer correct** | `agent_eval.py:190-193` builds the no-tools prompt as `SYSTEM_PROMPT_BATCH.split("Use the available tools")[0]`, which also drops the numbered 5-step sequence at `src/agent_loop.py:36-41`; the agent arm additionally carries ~4.7 rounds of tool output. Three differences, one comparison | The 20/30 vs 1/30 counts, which are correct as counts |
| 14 | `c7c1159` | Extended `agent_parser_check.py` with an **offline `--agent` half** applying the lenient parser to the agent arm's 20 strict-parser failures | R2 #4. **Verified, reviewer correct about the asymmetry; the repair is only partly possible** | `src/agent_loop.py:243` is strict, `agent_eval.py:156-170` lenient. New result: 1 of 20 recovered, **19 of 120 runs indeterminate** — all 20 recorded `raw_response` values are exactly 500 chars because the project's parser stores `raw[:500]`, so a reply whose JSON had not closed is *unknown*, not invalid. Reported as a bound (101/120 to 120/120), not a point. `agent_eval_agent_arm_parser_check.{csv,json}` | The no-tools half and its CSV, unchanged |
| 15 | `c7c1159` | Added **`agent_section4_figures.py`**, which recomputes every §4 figure (clustered bootstrap over customers) and writes `agent_section4_figures.csv` + `_info.json` | R2 #5. **Verified, reviewer correct** — the Sleeping-Dog rate, the no-tools rate, the parseable-only agreement, the never-intervene CI and the by-type table had no committed script (`agent_eval_summary_ci.csv` holds neither the subset-restricted nor the parseable-only rows, and has no never-intervene arm) | Reproduces every quoted figure: 0.67 [0.433, 0.867], 0.033 [0.000, 0.100], 53.0% [0.394, 0.667], never-intervene [0.600, 0.875], by-type 27/28, 15/17, 7/27, 4/28. Rows that also exist in `agent_eval_summary_ci.csv` agree to within ±0.01 on the bounds (independent bootstrap draws) | Original CSVs untouched; the new script only reads them |
| 16 | `c7c1159` | **Labelled the estimator on every §4 row** and showed both where they differ | R2 #5, second half. **Verified, reviewer correct, including the 0.5875 figure** | Confirmed: §4's parseable-only row was the ratio estimator (53/100 = 0.530) directly above the committed per-customer mean (0.4417), undisclosed. Under the per-customer estimator the parseable-only figure is **0.5875**. Both now appear, each labelled, and the conclusion holds under either | Both numbers; neither is dropped in favour of the other |
| 17 | `c7c1159` | **Correction the review did not flag:** the "Lost Cause or Sleeping Dog" row printed 18.3% while labelling itself "(60 runs)" | Found while verifying R2 #5. 18.3% is 22/**120** — `agent_eval.py` averages that flag over all runs, not the 60 eligible ones, so the row was inconsistent with the 20/30 row above it | 22/60 = 36.7% [20.0, 55.0]; 22/120 = 18.3% [9.2, 29.2]. Both denominators now shown and labelled | The committed 18.3% is kept, as the figure `summary_ci.csv` reports |
| 18 | `c7c1159` | §5: **the seeds are not independent**, so `± std` is not a standard error | R2 #6. **Verified** | `churn_eval.py:72` and `uplift_hillstrom.py:118` both `train_test_split` the *same* rows per seed. Overlap is ~20% (churn, 20% holdout) and ~30% (Hillstrom) between any two seeds' test sets | All `±` values; only their interpretation changed |
| 19 | `c7c1159` | Added **"uncorrected"** wherever many CIs are reported (§1, §2, §4, §5, SOP sentence 1) | R2 #7. **Verified: no multiple-comparison control exists anywhere in the eval code.** §2 alone is 80 nominal intervals (8 scores × 5 seeds × 2 outcomes), of which ~4 would exclude zero by chance under a global null | Counted directly from `uplift_hillstrom_by_seed.csv` (80 rows) | No interval was recomputed or widened; they are valid individually |
| 20 | `c7c1159` | §5: **no temporal validation is possible** | R2 #8. **Verified** | No datetime dtype in any of the 90 columns of `data/processed/segmented.parquet`; the time-flavoured fields (`MonthsInService`, `DaySinceLastOrder`, `CurrentEquipmentDays`) are durations from one unstated snapshot, not calendar dates. Hillstrom's 12 columns are a single cross-section | — |
| 21 | `c7c1159` | §2 + SOP sentence 2: the **churn-score row is the reverse ranking of the response model**, so not an independent test | R2 #9. **Verified** | `uplift_hillstrom.py:84`: `"baseline_churn_score (highest P(no visit))": 1 - p_resp`. Qini depends only on the ranking, so this row is the response-model row inverted | The row itself, which is still the operational policy the README argues against |
| 22 | `c7c1159` | §2: the **Qini normalisation is non-standard**, so 0.0028 is not comparable to published coefficients | R2 #10. **Verified** | `common.py:113-122` divides the area between the curves by `n`. Published Qini is normally the unnormalised area or the area ÷ perfect-targeting area. Added the conversion (≈54 incremental visits of area at n = 19,200) | The metric and every internal comparison, which the scaling supports |
| 23a | `c7c1159` | §2 decile sentence: 0.086 → **0.085**, "1.9×" → **1.85×** | Found in a fidelity pass over every §2 number against the CSVs; not raised by the review | `uplift_hillstrom_deciles.csv`, ensemble, visit, mean over seeds: decile 1 = 0.08548, decile 10 = 0.04621, ratio 1.850. The CSV is unchanged by the re-run, so this was a pre-existing rounding slip | The substantive point (modest, partly non-monotone spread), now with the non-monotonicity named |
| 23 | `c7c1159` | Refreshed `eval_sop/results/uplift_hillstrom.log` and `uplift_hillstrom_run_info.json` from the 5,000-resample run | The tracked log and run-info must describe the committed CSVs, not the superseded run | `run_info.json` now records `n_boot: 5000`, `runtime_s: 3207.6`. `uplift_hillstrom_deciles.csv` and `uplift_hillstrom_summary.csv` are **unchanged**, which is the expected signature of a CI-only change | Same 29-line log format |
| 24 | `c7c1159` | Added **`tests/test_known_defects.py`** — 9 reproduction tests pinning the three §9 defects | R2 #11. **Verified: the branch previously added no tests at all** (`git diff --stat 8e68bc6 HEAD -- tests/` was empty at `5057d59`) | Label leak (2 tests, one of which guards the eval harness's strip list against a new label spelling), silent CausalML fallback (2), noise-passing direction check (3, incl. a deterministic negated-risk-score case and a 35–65% band for the stochastic one so it is not flaky), plus the one working guard. `pytest -q` → **82 passed** | `src/` still unmodified; no existing test changed |

| 25 | `1cfa648` | **Ran the third agent arm** — `eval_sop/agent_third_arm.py`, full `SYSTEM_PROMPT_BATCH` verbatim with the tools removed, 40 customers × 3 seeds, local `qwen2.5:7b`, 597 s, 0 infra errors. Added the paired decomposition to `agent_section4_figures.py`. **Rewrote §4's headline and SOP sentence 4, and dropped the "does not separate" caveat** | §4's strongest claim was confounded three ways (round-2 #13 could only label the confound, not resolve it). This arm holds the prompt fixed and removes only the tools, so A−B isolates tools and B−C isolates prompt | **It is the tools.** Arm B contacts Sleeping Dogs in 5/30 (0.17 [0.03, 0.33]) — near arm C's 1/30, far from arm A's 20/30. Paired on the 10 SD customers: tools **+0.500 [0.233, 0.733]** (excludes 0); prompt +0.133 [0.000, 0.300] (does not); both together +0.633 [0.400, 0.833]. Secondary finding: arm B parsed **120/120** against the agent's 100/120 on the *identical* prompt, so the 17% parse-failure rate is tool-loop-specific too. `agent_eval_third_arm_rows.csv`, `_raw_outputs.jsonl` (full replies, untruncated), `_run_info.json`, `agent_sleeping_dog_decomposition.csv` | Arms A and C untouched — the new arm is a separate script and separate files, so no existing artifact was re-run or altered. `src/` still unmodified |

| 26 | `9f225f3` | **Ran the fourth arm** — `agent_third_arm.py --arm D`: full prompt, no tool calls, but the prescribed 5-step tool **results pasted into the user message**. 40 × 3, local `qwen2.5:7b`, 725 s, 0 infra errors, 0/120 context overflows (largest prompt 1,941 est. tokens). Extended the decomposition to **two outcome definitions** because they disagree | Round-2 #25 left one ambiguity inside the +0.50: tool access vs tool output in context. Arm D holds the information fixed and removes only the calling | **The split did not resolve, and the write-up says so rather than picking the convenient reading.** Sleeping Dog (n=10): function-calling **+0.367 [0.133, 0.633]** ✓, tool output +0.133 [−0.200, 0.467] ✗. Lost Cause ∪ Sleeping Dog (n=20): the reverse — tool output **+0.300 [0.067, 0.533]** ✓, function-calling −0.017 ✗. Two causes identified: the agent's Lost-Cause rate is deflated because 13/30 of its Lost-Cause replies are unparseable and `score()` records those as `intervene=False`; and arm D is genuinely anomalous on Lost Causes (0.467 vs the agent's 0.118 parseable-only). **Robust across both definitions:** whole tool loop ✓ (+0.500 / +0.283), prompt ✗. Tool output recomputed, not replayed — deterministic pure functions; verified 0 leak-key violations across all 40 customers and `actually_churned` absent. `agent_eval_third_arm_armD_*`, `agent_sleeping_dog_decomposition.csv` | Arms A, B, C untouched. Arm B re-verified: replayed from its checkpoint and reproduced byte-identical rows and raw outputs |
| 27 | `9f225f3` | Strengthened two §4 findings into named subsections, per review follow-up, and corrected arm B's `runtime_s` | Both are separable defects, not footnotes; and the checkpoint replay had overwritten arm B's runtime with 2.1 s | (a) **Invalid-JSON rate is tool-loop-specific, not prompt and not context length** — arms B *and* D parsed 120/120 on the identical prompt, and arm D carries 3× arm B's context. Also notes the defect *flatters* the agent on do-not-contact metrics. (b) **Sure-Thing and Sleeping-Dog failures are two distinct defects** the aggregate was averaging: Sure Thing is 0% in **all four** LLM arms (model+prompt, tool-invariant), Sleeping Dog is the tool-driven one. Arm B's `runtime_s` restored to 596.6 with a `runtime_note` recording that the 2.1 s replay was a cache-hit reproduction check, not a timing | Both findings' underlying numbers unchanged |

No change was made to `src/` or `dashboard/`, and no existing test was modified. `tests/test_known_defects.py` is new
(round-2 #23), so the suite is now **82 passed** (`python -m pytest -q`), up from 73. `ruff check .` → all passed. Both
were run before every commit in this phase.

## 9. Proposed, not done

**All three defects below are now pinned by reproduction tests** in `tests/test_known_defects.py` (9 tests, the only
tests this branch adds). They assert the *current, defective* behaviour, so a silent regression is impossible in either
direction: if someone fixes a defect, the matching test fails and has to be updated in the same commit, which is the
intent. `src/` is still unmodified. This replaces "reproduced by inspection" with "reproduced in CI".

- Remove `actually_churned` from `src/agent_tools.py:204`, and select explicit columns in `route.ts` `lookup_customer_details`
  (label leakage to the LLM). Pinned by `test_lookup_customer_details_leaks_realized_label`, plus
  `test_lookup_customer_details_has_no_other_label_key`, which fails if a future change adds a second spelling of the
  label that would slip past the eval harness's strip list.
- Make `src/uplift_model.py` fail loudly, not fall back, when CausalML cannot be imported. Reproduced: importing it in an
  env with numpy 2.5 logs "CausalML not available — using custom T-learner fallback" and continues. Pinned by
  `test_causalml_fallback_is_silent_in_the_return_value` (the metrics dict never names the estimator that ran, so a
  caller cannot tell which one produced the numbers) and `test_causalml_import_failure_only_warns`.
- The direction check's lack of power is pinned by `test_direction_check_passes_for_pure_noise_about_half_the_time`
  (asserts a 35–65% band, not a point, so it is not flaky) and the deterministic
  `test_direction_check_passes_for_negated_churn_probability`. Its one working guard — abstaining below 100 treated rows
  — is pinned too, so it is not lost in a rewrite.
- Replace the Cell2Cell treatment proxy, or drop `Complain` from it, and add a held-out Qini to the pipeline.
- Paired bootstrap CI for the ensemble vs response-model Qini difference on Hillstrom.
- Run the Criteo subsample. Run the agent eval on a second model and on the deployed TypeScript route (needs a hosted
  model; see the cost estimate). Human-rate `agent_eval_human_validation.csv`.
- ~~Add the missing third agent arm~~ and ~~a fourth arm to split tool access from tool output~~ — **both done** (round-2
  #25 and #26, `eval_sop/agent_third_arm.py --arm B|D`). The tool-loop effect is +0.50 [0.23, 0.73] with the prompt held
  fixed. The function-calling-vs-tool-output split is **still open**: the two outcome definitions disagree and the
  Lost-Cause half is confounded by the agent's 13 unparseable replies scoring as non-intervention. Settling it needs (a)
  the tolerant parser below, so the agent is not credited for failing to answer, and (b) more than 10 customers per cell.
  Re-running the decomposition after the parser fix is the cheapest next step and still free.
- Explain arm D's Lost-Cause anomaly: with the complete correct tool block pasted in, this model intervenes on Lost Causes
  at 0.467 against the live agent's 0.118 (parseable-only). Inspect the full arm D replies — they are committed
  untruncated in `agent_eval_third_arm_armD_raw_outputs.jsonl`, with the exact pasted block per run.
- Make the agent's JSON parsing tolerant of trailing text (src/agent_loop.py, the `json.loads(raw)` path). 17% of agent
  replies fail it. Not changed, because I have not inspected full replies and have no failing test. **Also raise or remove
  the `raw[:500]` truncation on the same line**: it is what makes 19 of the 20 failures impossible to re-adjudicate
  offline (section 4, Parsing), so the symmetric parser comparison for the agent arm is a bound rather than a number. The
  truncation costs more in lost evidence than it saves in artifact size.
- Cleanup: the evaluation venv lives at `%TEMP%\sopvenv` (outside the repo, because of the Windows MAX_PATH limit under the
  scratchpad). It can be deleted after review.

## 10. The tracked parquets were redistributing the source dataset

`.gitignore` records a deliberate decision: `data/processed/` and `models/` are
tracked so the deployed dashboard runs from a clone without re-running the
pipeline. That decision stands. What was never decided is **which columns** came
along with it.

Every parquet write handed `to_parquet` whatever frame the stage happened to be
holding, so all four tracked files carried the Cell2Cell source columns:

| Tracked file | Columns before → after | Source columns before → after | Bytes |
| --- | --- | --- | --- |
| `features.parquet` | 81 → 28 | 55 → 2 | 3,469,815 → 1,555,889 |
| `scored.parquet` | 94 → 41 | 55 → 2 | 7,907,735 → 5,993,809 |
| `segmented.parquet` | 90 → 37 | 55 → 2 | 5,883,593 → 3,969,667 |
| `uplift.parquet` | 103 → 50 | 55 → 2 | 9,451,315 → 7,537,389 |

53 columns dropped from each file, 51,047 rows preserved, and every surviving
column asserted byte- and dtype-identical to what it held before
(`pd.testing.assert_frame_equal`). The two source columns that remain are
`CustomerID` (the join key Supabase and the dashboard need) and `Churn` (the
label — without it the committed artifacts cannot be checked against the
reported AUC/Brier/churn-rate figures, which is the reason they are tracked).

Cell2Cell was compiled by the Teradata Center for CRM at Duke University (Neslin
et al., 2006); the circulating copy is a Kaggle mirror whose licence field reads
"Unknown". The raw CSV was correctly gitignored, so the README's "we do not ship
the data" was true of `data/raw/` and false of `data/processed/`.

**Fix.** `src/published_columns.py` declares the source schema and the two-entry
allowlist; all seven `to_parquet` sites in `src/` project through
`restrict_for_publication`, which a test now counts rather than taking on trust.
`cell2cell_features.py` maps the raw telecom columns onto the e-commerce schema's
names before anything is modelled, so all 23 churn features, all 13 clustering
features and all 9 uplift features survive the trim. `readme_metrics.py --check`
still passes, `migrate_to_supabase.load_data()` still yields all 25 dashboard
fields, every feature in `models/segment_models.pkl` and `models/scaler.pkl` is
still present, and a cache-resume through stages 2–4 still has everything it
needs.

**The e-commerce path is deliberately not enforced.** There the source column
names *are* the feature names: enforcing the same rule would drop 17 of the 26
churn features, 5 of 13 clustering and 3 of 9 uplift. `ECOMMERCE_SOURCE_COLUMNS`
is declared but kept out of `SOURCE_COLUMNS`, and a test pins that it stays out
so the gap reads as a choice. A consequence worth stating plainly: the guard is
therefore **dataset-conditional**. If someone commits e-commerce-built parquets,
`test_tracked_parquet_publishes_no_source_feature_columns` passes vacuously
because `forbidden_columns("ecommerce")` is empty.

### 10a. What an adversarial review found wrong with the above

The first version of this section was wrong in five ways and the change broke
something. The review is in the change log; the findings are here rather than
quietly edited away, because the pattern in them is the useful part.

**(i) The trim silently destroyed a published number, and the method used to
justify it was structurally unable to notice.** `eval_sop/churn_eval.py` took its
raw-column reference set *by position*:

```python
first_mapped = list(df.columns).index("Tenure")
raw_cols = [c for c in df.columns[:first_mapped] if c not in ("CustomerID", "Churn")]
```

Before the trim that slice was 53 columns. After it, `index("Tenure")` is 3 and
the slice is **one** column. Eval arm (5),
`reference_global_catboost_all_raw_cols`, published at `RESULTS.md` §1 as
**0.673 ± 0.005**, would have silently re-run at **AUC 0.500** — no exception, no
warning, and `churn_run_info.json` would have recorded
`raw_cols_reference: ["MaritalStatus"]`. CI does not run `churn_eval.py`, so
nothing would have caught it.

The justification for the trim was a scan for each column name as a *string
literal* across every tracked file. A positional access contains no column name,
so the method could not see this consumer however carefully it was run. That is
the finding worth keeping: the confidence was in proportion to the scan's
coverage, not to its blind spot.

Worse, the evidence was in hand and misread. The scan did flag
`eval_sop/results/churn_run_info.json`, which contains `raw_cols_reference` with
exactly those 53 names, and it was dismissed as "records column names a past run
saw — provenance, not a consumer". It is the *output of* the consumer.

**Fixed** by `load_raw_reference()` in `eval_sop/churn_eval.py`, which builds the
reference columns **by name** from `data/raw/` rather than by position from a
committed artifact. It refuses to run on fewer than 40 columns instead of
degrading, and when `data/raw/` is absent it omits arm (5) from the output with a
printed note rather than reporting a number. Re-measured after the fix: 53 raw
columns, seed 42 **AUC 0.6754**, consistent with the published 0.673 ± 0.005.
Pinned by `test_eval_harness_does_not_select_raw_columns_by_position`.

**(ii) "Four source signals still ship" understated reality by roughly 3×.**
`engineer_features` is largely a rename-and-clip table, not a set of
transformations, so dropping a source column by name does not remove its values.
Measured against the raw CSV:

| Dropped source column | Recoverable from | Rows exactly recoverable |
| --- | --- | --- |
| `MonthsInService` | `Tenure`, `OrderCount` | 100.00% |
| `RespondsToMailOffers` | `CouponUsed` | 100.00% |
| `Occupation` | `PreferedOrderCat` | 100.00% (bijective encode) |
| `HandsetWebCapable` | `PreferredLoginDevice` | 100.00% |
| `MonthlyRevenue` | `AvgOrderValue` | 99.99% |
| `CurrentEquipmentDays` | `DaySinceLastOrder` | 99.85% |
| `RoamingCalls` | `WarehouseToHome` | 99.73% |
| `UniqueSubs` | `NumberOfAddress` | 99.54% |
| `Handsets` | `NumberOfDeviceRegistered` | 97.75% |
| `CreditRating` | `SatisfactionScore` | 93.60% |
| `MonthlyMinutes` | `HourSpendOnApp` | 85.40% |

So source information still present covers about **13 of the 58 source
columns** — the 2 allowlisted plus the 11 above, 8 of them near-exactly — not 4.
`MonthsInService` was disclosed as the lone exception when it is the pattern;
the analysis stopped at the first hit instead of enumerating. Lossy aliases are
not counted: `Complain` keeps 54.89% of `CustomerCareCalls` (a `> 3` threshold),
`OrderAmountHikeFromlastYear` 43.95% of `PercChangeRevenues` (negatives clipped),
`CityTier` 12.20% of `IncomeGroup`.

The aliases cannot be removed — they are the models' features. So the claim is
now stated as what it is: a large reduction, not a clean break. 53 of 55 source
columns gone by name; source information for ~13 of 58 still derivable.

**(iii) `MaritalStatus` was allowlisted on a reason that does not exist.** The
stated justification was "read by `app.py` and the retention prompts". Neither
file mentions it; `api/serve.py` has it as a request-body field with a default
and never reads it from a parquet. It was dropped, which is why the figures above
say 2 source columns and 53 dropped rather than 3 and 52. Pinned by
`test_marital_status_is_not_published`, which also fails if `app.py` starts
naming it.

**(iv) "All seven `to_parquet` sites project through `restrict_for_publication`"
was false — six of seven did.** `src/olist_features.py` was missed. It is now
routed through the projection too (a no-op for Olist, which has no declared
schema), and `test_every_to_parquet_site_goes_through_the_projection` counts the
call sites so the claim cannot rot.

**(v) Two stated figures did not reproduce.** The table claimed 52 source columns
and "0 of the 52 read" for three of the four files, with 55 and 3 only for
`uplift.parquet`. All four carried 55, and all four carried the same read ones.
And "the only other hit was `churn_run_info.json`" was not what the scan returns
— `eval_sop/positivity_signflip.py` names two dropped columns (harmlessly, in a
descriptive run-info string), as do `src/cell2cell_features.py` and `RESULTS.md`.
The conclusion held; the statement of it was stronger than the evidence.

The pattern across (ii), (iii) and (v) is the same one this branch found
elsewhere: **everything mechanically re-derivable from a file was exact — the
column counts, the byte counts, the row counts, the feature-set survival — and
everything that required a judgement about what some code does was not.**

### Found while doing this, fixed in section 11

Three defects of the same class were found here and recorded rather than quietly
changed, on the grounds that each moves every reported figure: `Tenure` and
`OrderCount` are the same column and both were in the churn feature list;
`PreferredPaymentMode` was constant; `Gender` was hardcoded to 0.

**All three are now fixed and everything they touched was re-measured — see
section 11**, which also records two further defects that fixing them exposed, in
how the project reports on itself rather than in the model.

### Tests

`tests/test_published_columns.py`, 18 tests. **Five** of them go red if the
pre-trim parquets are copied back in place — four parametrizations of
`test_tracked_parquet_publishes_no_source_feature_columns` plus
`test_marital_status_is_not_published`. Counted by doing exactly that and
re-running (`5 failed, 93 passed, 2 skipped`), not by reasoning about it; an
earlier draft of this section said four, from before the `MaritalStatus` test
existed. Two guard the opposite failure,
over-trimming: one pins the columns `scripts/readme_metrics.py` needs, one pins
all 25 fields in `migrate_to_supabase.py`'s column map. That map is intersected
with whatever is present, so a dropped column would not raise — it would silently
become a missing dashboard field. Two more re-read the real raw headers when
`data/raw/` is present and assert the hardcoded schemas agree; both verified
against the real files (58 columns for Cell2Cell, 20 for the e-commerce
workbook), and they skip in a clone that has no raw data.

Suite **82 → 98 passed**, `ruff check .` clean.

## 11. Three dead or duplicated features, and two false numbers about them

§10 recorded three defects and declined to fix them, on the grounds that each one
moves every figure the README reports. That was the right call at the time and
the wrong place to stop: the figures were wrong either way, and "wrong but
unchanged" is not a property worth preserving. All three are fixed here, the
pipeline was re-run, and every affected number was re-measured rather than
adjusted.

Fixing them surfaced two more defects of the same family — not in the model, in
how the project reports on itself. Those are §11d and §11e, and they are the more
interesting pair.

### 11a. `PreferredPaymentMode` was constant because of a string that never matched

`clean_cell2cell` encoded homeownership as:

```python
df["Homeownership"] = (df["Homeownership"].astype(str).str.lower() == "known homeowner").astype(int)
```

The column holds **`"Known"`** and **`"Unknown"`**. Nothing is ever equal to
`"known homeowner"`, so all 51,047 rows became 0, and `PreferredPaymentMode` —
which `engineer_features` maps from it — was a constant inside the `churn_model`
feature list. The real split is **33,987 Known / 17,060 Unknown**.

Fixed by comparing against `"known"`. `clean_cell2cell` now **raises** if the
column collapses to a single value after encoding, because the original failure
was silent and that is what let it survive a full pipeline run, a CI pass and an
independent review.

### 11b. `OrderCount` was `Tenure`, and the model was fed both

`engineer_features` set `df["OrderCount"] = df["Tenure"]` outright, and **both
names were in the `churn_model` list**, so every per-segment model received the
same 56-valued column twice.

Two different fixes, because the two feature sets had two different problems:

- `churn_model` had both, so this was a real model change: `OrderCount` dropped,
  **23 → 22 features**.
- `clustering` had `OrderCount` and *not* `Tenure`, so it was not duplicated
  there — just misnamed. Renamed to `Tenure`, which is numerically inert: the
  clustering input matrix is bit-identical, asserted with
  `np.array_equal` before the re-run and confirmed after it (identical segment
  sizes 12,966 / 11,975 / 9,553 / 8,400 / 8,153, identical stability ARI
  0.9152 ± 0.136).

The column is still emitted, because the Supabase `customers` table and the
dashboard both have an order-count field. On this dataset its documented meaning
is tenure in months; Cell2Cell subscribers place no orders.

### 11c. `Gender` was a published constant

`df["Gender"] = 0` for all 51,047 rows, with the honest comment "not in
Cell2Cell" — and then published in all four tracked parquets while being named in
no Cell2Cell feature set. No longer emitted on this path. The e-commerce path has
a real `Gender` column and is untouched.

### 11d. The README reported a 1-tree model as 500 trees

This one is worth more than the three above.

```python
best_iteration = int(getattr(base_clf, "best_iteration_", None) or params["iterations"])
```

CatBoost sets `best_iteration_` to **0** when the first iteration is the best.
`0 or 500` is `500`. After 11a and 11b, the **Lapsed** segment began stopping at
iteration 0 — it trains a single tree, `tree_count_ == 1` — and the README's
"Trees" column printed **500**.

A 500× overstatement, generated automatically by `scripts/readme_metrics.py` and
**verified by CI on every push**. CI compares the README against
`segment_models.pkl`'s recorded number, so it confirmed the README faithfully
reported a figure that the model contradicts. The check was real; its subject was
wrong.

Fixed in three places:

- `train_segment_model` no longer defaults with `or`: `0` stays `0`.
- It records `tree_count` as well, and `readme_metrics.py` reports that in the
  column headed "Trees" — which is what the heading claims. The other segments
  shift by one as a result (61→62, 41→42, 35→36, 167→168) because
  `tree_count_ == best_iteration_ + 1`.
- It emits a **warning** when a segment produces ≤ 1 tree, saying that the
  holdout AUC should be read as "no usable signal found" rather than as a score.
  A warning and not an exception: the pipeline should still finish and publish
  the number, just not quietly.

So Lapsed's 0.557 is now labelled for what it is. A stump that early-stops on
iteration 0 has found nothing in that segment, and its AUC was previously
presented in the same table, in the same style, as At-Risk's 0.689 from a
62-tree model.

### 11e. The reproduction check compared against a copy of the README

`eval_sop/churn_eval.py` had:

```python
README_HOLDOUT_AUC = {  # copied from README.md "Per-segment churn models" table
    "At-Risk": 0.692, "Price Sensitive": 0.645, ...
}
```

The README is **generated** from the artifacts, so this dict was stale the moment
any model changed — and the comparison is only *printed*, never asserted. It duly
printed `README 0.576` beside `re-run 0.557` with no comment, in a section headed
"reproduce the committed numbers".

It now parses the README's generated table, raises if the table has changed shape
rather than reproducing against a partial set of segments, and prints a loud
**REPRODUCTION WARNING** when any segment deviates by more than 0.01 AUC — a
threshold set above the measured 0.0016 `thread_count` sensitivity and far below
anything that changes a conclusion. Current deviations: **+0.0001, −0.0001,
+0.0003, −0.0004, −0.0001**.

### What changed in the numbers

The honest summary is that **fixing the features made the headline model slightly
worse.**

| | Before | After |
| --- | --- | --- |
| Mean holdout AUC (per segment) | 0.6236 | **0.6210** |
| Mean train AUC | 0.7015 | 0.6954 |
| Mean holdout Brier, raw → calibrated | 0.2342 → 0.1946 | 0.2343 → 0.1947 |
| Churn-model features | 23 | 22 |
| Persuadables | 7,602 | **7,788** |
| High-risk customers | 322 | 222 |
| Cluster stability (mean ARI) | 0.9152 ± 0.136 | 0.9152 ± 0.136 (unchanged by construction) |

Per segment:

| Segment | AUC before | AUC after | Trees before (as published) | Trees after (true) |
| --- | --- | --- | --- | --- |
| At-Risk | 0.692 | 0.689 | 64 | 62 |
| Price Sensitive | 0.645 | 0.646 | 39 | 42 |
| Loyal Customers | 0.614 | 0.615 | 26 | 36 |
| Champions | 0.591 | 0.598 | 95 | 168 |
| Lapsed | 0.576 | **0.557** | 77 | **1** |

**The central finding of §1 survives and is slightly stronger.** Per-segment
remains worse than a single global model on the same test rows in 5/5 seeds, with
ΔAUC now −0.0081, −0.0071, −0.0128, −0.0165, −0.0096 (was −0.0062 to −0.0120) and
every CI still excluding 0. The conclusion did not rest on the defects.

**Do not read this as "the fixes hurt the model."** Removing a duplicated feature
and making a dead one informative changed what CatBoost's early stopping sees;
the mean moved by 0.0026, which is inside the seed-to-seed spread of ±0.006. The
defensible statement is that the model's measured quality is unchanged within
noise, and that two of its 23 features were previously contributing nothing.

### Reproducibility, and an environment claim this corrects

§5 said: *"`requirements.txt` pins `numpy<2.1`; I did not test that exact pin."*
It is tested now. Everything in this section ran in a fresh venv at **numpy
2.0.2** with causalml 0.16.0, pacmap and numba importable — which also means the
uplift stage ran through CausalML rather than the silent custom-T-learner
fallback (`uplift_metrics.pkl` records
`"T-Learner + S-Learner ensemble (CausalML)"`).

The ambient Anaconda environment on this machine has numpy 2.5, which breaks
numba and therefore pacmap, so `src/pipeline.py` cannot run there at all — it
fails at `import pacmap` in stage 2 before reaching the silent causalml fallback.

The whole pipeline and the 5-seed eval were run **twice**, and the second run
reproduced the first exactly: same segment sizes, same stability ARI, same 7,788
persuadables, same risk tiers, and the same figures to four decimal places in
every row of §1.

### Tests

`tests/test_cell2cell_feature_defects.py` (10) and
`tests/test_reporting_defects.py` (9). **13 of the 19 go red** when the four
source files are reverted to `48ffe08` with the regenerated artifacts left in
place — counted by doing exactly that and re-running, not by reasoning about it.
Suite **98 → 119 passed**, `ruff check .` clean.

Two of them assert the general form rather than the instance, because the
specific defects are less interesting than their shape:

- `test_churn_features_contain_no_duplicated_column` fails if **any** two
  features in the churn list hold identical values, under any names. It caught a
  false positive in its own fixture first — `IncomeGroup` and `UniqueSubs` both
  cycled mod 3, so `CityTier` and `NumberOfAddress` came out identical on
  synthetic data while the real data has no such coincidence. The fixture was
  fixed, not the assertion.
- `test_no_churn_feature_is_constant` fails if any feature cannot affect a
  prediction, which is 11a and 11c at once.

And one guards the reverse direction: `test_order_count_is_still_emitted_for_the_dashboard`
fails if dropping the column from the model also drops it from the data, which
would silently empty a Supabase field.
