# RESULTS: an independent evaluation for SOP use

Branch `sop-eval`, based on `main` at `8e68bc6`. I changed nothing under `src/`, `dashboard/`, `tests/` or `README.md`.
All new code and outputs are in `eval_sop/`. The **agent evaluation has not been run yet**. The harness and
scenario set exist, but no LLM was called (see section 4).

## 0. Setup and reproduction

| Item | Value |
|---|---|
| Churn data | Cell2Cell (`cell2celltrain.csv`, 51,047 labelled rows, 28.8% churn). I used the committed pipeline artifact `data/processed/segmented.parquet` (the exact input to Stage 3) so the segments match the shipped models. |
| Uplift data | Hillstrom MineThatData e-mail challenge, 64,000 rows, randomized (2/3 got an e-mail). Source: `http://www.minethatdata.com/Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv`, sha256 `0e5893…aece`, checked in code. Raw data is not committed (`eval_sop/.gitignore`). |
| Python env | Python 3.12.3. numpy 2.4.6, pandas 2.3.3, scikit-learn 1.8.0, catboost 1.2.10, xgboost 3.2.0, causalml 0.16.0. |
| Seeds | Churn: split and model seeds {42, 7, 13, 21, 99}. Hillstrom: split seeds {42, 7, 13, 21, 99}. The project's XGBoost learners keep their hard-coded `random_state=42`. |
| CIs | 95% percentile bootstrap over held-out rows: 1,000 resamples (churn) and 500 (uplift). The CIs capture test-set noise. The ± values are std across the 5 seeds. |
| Threads | Uplift and agent runs used `OMP_NUM_THREADS=2 MKL_NUM_THREADS=2`. The churn run used CatBoost's default `thread_count=-1`. |

Reproduce (from the repo root, with the env above):

```
python eval_sop/churn_eval.py           # ~17 min on a shared laptop
python eval_sop/uplift_hillstrom.py     # ~4 min with 2 threads (downloads + verifies Hillstrom if missing)
python eval_sop/positivity_signflip.py  # <1 min
python eval_sop/agent_eval.py --scenarios-only   # writes scenario set, no LLM
python eval_sop/agent_eval.py --model qwen2.5:7b --seeds 0 1 2   # NOT YET RUN (needs a local Ollama slot)
```

**The README figures reproduce exactly.** With the project's own `churn_model.train_segment_model` and seed 42, every
per-segment holdout AUC and Brier pair matches the README table to the printed precision
(`eval_sop/results/churn_reproduction_seed42.csv`). Examples: At-Risk AUC 0.6923, Brier 0.2197→0.1778; Lapsed 0.5755, 0.2453→0.2197.

## 1. Churn model (Cell2Cell, pooled held-out set: n = 10,211 per seed)

The test set is the union of the per-segment 80/20 stratified holdouts. Every model is trained on the same training
rows and scored on the same test rows. Values are mean ± std over 5 seeds. The 95% bootstrap CIs come from seed 42.

| Model | AUC | PR-AUC (base 0.288) | Brier | Brier skill vs base rate | ECE |
|---|---|---|---|---|---|
| Base rate (constant 0.288) | 0.500 | 0.288 | 0.2051 | 0 | 0 |
| Logistic regression (23 project features) | 0.594 ± 0.003 | 0.352 ± 0.003 | 0.2008 | 2.1% ± 0.2 | 0.008 |
| **Per-segment CatBoost + isotonic (project)** | **0.631 ± 0.006** [0.621, 0.643] | 0.393 ± 0.009 [0.372, 0.405] | 0.1966 | **4.1% ± 0.5** [3.1, 4.9] | 0.015 [0.009, 0.024] |
| Per-segment CatBoost, uncalibrated | 0.623 ± 0.006 | 0.392 ± 0.008 | 0.2360 | −15.1% ± 1.0 | 0.197 |
| Global CatBoost + isotonic (same recipe, same features) | **0.640 ± 0.003** [0.626, 0.650] | 0.402 ± 0.002 | 0.1946 | 5.1% ± 0.2 [4.1, 5.9] | 0.008 |
| Global CatBoost + isotonic + segment id | 0.640 ± 0.004 | 0.401 ± 0.003 | 0.1947 | 5.1% ± 0.3 | 0.010 |
| *Reference:* global CatBoost on all raw Cell2Cell columns | 0.673 ± 0.005 | 0.443 ± 0.007 | 0.1889 | 7.9% ± 0.4 | 0.009 |

**Per-segment vs global, paired on the same test rows** (`churn_perseg_vs_global_paired.csv`): ΔAUC (per-segment minus
global) is −0.006, −0.006, −0.013, −0.012 and −0.011 across the 5 seeds, and **every 95% CI excludes 0**. ΔBrier is
positive (worse) in 5/5 seeds, and its CI excludes 0 in 4/5. Within each of the 5 segments, the global model's mean AUC is
higher than that segment's own model (for example At-Risk 0.684 vs 0.674, Lapsed 0.577 vs 0.567).

What this supports:
- The pipeline is reproducible, and its holdout numbers are honest.
- Isotonic calibration is real and fixes a large miscalibration caused by class weighting (ECE 0.197 → 0.015).
- The model ranks better than chance and better than logistic regression (+0.04 AUC).

What it does not support:
- **"Per-segment beats global."** On this data a single global model is slightly but consistently better.
- **"17% Brier improvement" as a measure of skill.** That 17% compares calibrated with *uncalibrated, class-weighted*
  probabilities. Against the trivial base-rate forecast (Brier 0.2051) the skill is **~4%**. The prior review's "~5%"
  is close; the measured value is 4.1% ± 0.5.
- Strong prediction. AUC is about 0.63. The project's feature mapping also gives up some signal: the raw columns reach about 0.67.

Reliability (seed 42, `churn_reliability_project_model.csv`) is good between 0.1 and 0.5, where 95% of rows sit. Above
0.6 the predictions are over-confident: the bin with mean prediction 0.65 has an observed rate of 0.49 (n = 47). The README's
"High Risk ≥ 0.6" tier is the least calibrated region.

## 2. Uplift learners on randomized data (Hillstrom, held-out 30%, n_test = 19,200)

The project's own `compute_uplift_scores_causalml` and `compute_uplift_scores_custom` are fitted on the 70% train split
and scored on the test split. Outcome = website visit. Test ATE = 0.061.

The Qini coefficient is the area between the Qini curve and the random line, divided by n, in incremental visits per test
customer averaged over targeting depths. 0 means random targeting.

| Targeting score | Qini (mean ± std, 5 seeds) | Qini seed 42 [95% CI] | Seeds with CI > 0 | Observed uplift in top 10% (seed 42) |
|---|---|---|---|---|
| Project T+S ensemble (as shipped) | **0.0028 ± 0.0008** | 0.0020 [0.0000, 0.0041] | 4/5 | 0.087 [0.053, 0.122] |
| Project S-learner (CausalML) | 0.0034 ± 0.0008 | 0.0030 [0.0011, 0.0049] | 5/5 | 0.108 |
| Project T-learner (CausalML) | 0.0021 ± 0.0010 | 0.0013 [−0.0008, 0.0033] | 3/5 | 0.085 |
| Project T-learner (custom fallback) | 0.0024 ± 0.0009 | 0.0013 [−0.0007, 0.0033] | 3/5 | 0.071 |
| Random | −0.0005 ± 0.0005 | −0.0005 [−0.0023, 0.0013] | 0/5 | 0.050 |
| "Churn-score" targeting (highest P(no visit)) | −0.0020 ± 0.0008 | −0.0008 | 0/5 | 0.051 |
| Response model (highest P(visit)) | 0.0020 ± 0.0006 | 0.0011 | 3/5 | 0.076 |
| **Sign-flipped ensemble (pre-fix bug)** | **−0.0028 ± 0.0009** | −0.0020 [−0.0040, 0.0000] | 0/5 | 0.065 |

Uplift by decile (ensemble, mean over seeds, `uplift_hillstrom_deciles.csv`): the observed uplift falls from 0.086
(decile 1) to 0.046 (decile 10). Ranking is monotone in coarse terms, but the spread is modest (1.9×) and
non-monotone in places. The ensemble beats the response-model baseline in all 5 seeds, by +0.0001 to +0.0015 Qini.
It beats random by +0.0019 to +0.0046.

On the secondary outcome (conversion, 0.9% base rate), no method's Qini CI excludes 0. The point estimates still rank
the same way.

What this supports: the project's meta-learners, run unchanged, recover real treatment-effect heterogeneity on a
randomized experiment. They beat random and churn-risk targeting, and roughly match or slightly beat a response model.
Targeting the highest-risk customers (the "email everyone above 0.7" approach the README argues against) scores below
random on Hillstrom: the point estimate is below random in 5/5 seeds, and the 95% CI lies entirely below 0 in 3/5.

What it does not support: anything about the size of uplift on Cell2Cell. Hillstrom is a different domain, a different
treatment and a positive outcome. A Criteo uplift subsample was **not run** (time and laptop-load limits).

## 3. Cell2Cell "treatment": positivity, and the sign flip

**The positivity violation is confirmed** (`cell2cell_positivity_signflip.json`). Treatment = `Complain | CouponUsed`
(src/uplift_model.py:72). On Cell2Cell, Complain = `CustomerCareCalls > 3` and CouponUsed = `RespondsToMailOffers`.
- P(T=1 | Complain=1) = **1.000**: all 8,472 complainers (16.6% of rows) are treated, and none are controls.
- `Complain` is also an uplift covariate. A 5-fold cross-validated propensity model on the 9 uplift features gives
  e(x) > 0.99 for exactly those 16.6% of rows (minimum e = 0.999) and AUC(T | X) = 0.83. 38–39% of rows fall outside
  [0.1, 0.9]. SMD for SupportRiskScore is 1.02.
- Uplift is fitted and predicted on the same rows (src/uplift_model.py:185-192), with no held-out split and no Qini/AUUC.
- Mean UpliftScore is 0.0246. 20.2% of customers clear the 0.05 Persuadable threshold.
- The pipeline's `validate_uplift_direction` check **also passes for random noise and for −ChurnProbability**. It can catch
  a flipped sign on a *fixed* score, but it cannot validate an uplift model.

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

On randomized Hillstrom data, the sign bug turns a Qini of +0.0028 into −0.0028: below random in 5/5 seeds (point estimate), with the CI entirely below 0 in 3/5.
The buggy pipeline still reported positive ROI for every customer it selected. Its own reported numbers could not reveal
the bug.

## 4. Agent: harness built, not run

- `eval_sop/agent_eval.py` drives the project's **Python** agent (`src/agent_loop.generate_retention_action_agentic`,
  unmodified) against a local Ollama model through the OpenAI-compatible API. It compares the agent with (a) the same
  LLM without tools and (b) a rule policy (intervene iff Persuadable and NetROI > 0, playbook entry from the top SHAP
  driver).
- Scoring is **programmatic, with no LLM judge**: valid JSON, agreement with the rule, interventions on Lost Cause or Sleeping Dog,
  whether the true top-SHAP driver is cited, cost greater than model-expected value, tool calls and latency. Bootstrap CIs are over customers.
  The run also writes `agent_eval_human_validation.csv` with blank human-rating columns.
- Scenario set: `eval_sop/agent_scenarios.csv`, 40 customers, 10 per CustomerType (`sample(random_state=0)`), with
  rule-based outputs. The rule baseline already shows one problem: it picks "High ($50–100)" interventions for
  customers whose model-expected value is under $50, because NetROI assumes a flat $15 cost.
- **Not run.** On the first attempt the shared local Ollama server returned HTTP 500 on every chat request for more than
  an hour while other jobs loaded it. I produced no agent numbers and claim none.
- Limits even once run: this is the Python agent (6 tools, local DataFrame), not the deployed TypeScript route (12 tools,
  Supabase). A local 7B model is not the deployed model. "Agreement with rule" is a consistency check, not a retention outcome.
- **Label leakage, found by inspection:** `lookup_customer_details` returns `actually_churned`, the realized label
  (src/agent_tools.py:204). The deployed route returns `select("*")` from `customers`, which includes `churn`
  (route.ts:422-429, migrate_to_supabase.py:66,104). The harness strips the key before the LLM sees it. src/ is not changed.

**Paid-API needs: none required.** For reference, an optional run of the deployed-style agent on a hosted model is
about 40 customers × 3 seeds × ~20k input + ~1.5k output tokens, roughly 2.4M input and 0.2M output tokens. At typical
open-model hosted prices ($0.1–0.6/M in) that is **≈ $0.5–2**. At frontier prices (~$3/M in, ~$15/M out) it is **≈ $10**.
These are estimates from list-price ranges, not quotes, and nothing was spent.

## 5. Threats to validity

- Segmentation (K-Means) was fit on all rows, holdout included. It is unsupervised and uses no labels, but it is mildly transductive.
- One dataset for churn. The Cell2Cell feature mapping (e.g. CreditRating → "SatisfactionScore") is the project's, and I kept it as is.
- The bootstrap CIs ignore training variance. The seed std covers part of it. Five seeds is few.
- Hillstrom uses a different treatment, outcome and domain from churn retention. It validates the learners, not the Cell2Cell targeting.
- The sign-flip result is a counterfactual recomputation on today's learners, not a replay of the historical buggy run.
- Environment: the base anaconda env has numpy 2.5, which breaks numba, so `import causalml` fails. `src/uplift_model.py`
  then **silently** switches to its custom T-learner (it logs only a warning). The committed artifacts say "CausalML", so they were
  built in a working env. I ran everything in a separate venv with numpy 2.4.6. `requirements.txt` pins `numpy<2.1`; I did
  not test that exact pin.

## 6. SOP-ready sentences (each is backed by a file in `eval_sop/results/`)

1. "I re-evaluated my churn pipeline against baselines. On 10,211 held-out Cell2Cell customers, its calibrated
   per-segment CatBoost models reached AUC 0.63 (95% CI 0.62–0.64), but a single global model was slightly and
   consistently better (ΔAUC −0.006 to −0.013 across five seeds). I therefore retracted my claim that per-segment
   modelling helps."
2. "On the randomized Hillstrom e-mail experiment (19,200 held-out customers), my uplift learners beat random targeting
   in all five seeds (Qini 0.0028 ± 0.0008 versus −0.0005). Targeting by predicted risk, and the sign-inverted score from a bug I had shipped, both scored below random in all five seeds."
3. "Isotonic calibration cut expected calibration error from 0.20 to 0.015. Measured against a base-rate forecast, though,
   the model's Brier skill is about 4%, not the 17% I had reported."

## 7. Proposed README corrections (not applied)

- Replace "17% Brier reduction" with "Brier skill ≈ 4% vs. base rate; calibration reduced ECE 0.20 → 0.015".
- Remove or reverse "per-segment models over a global model". The measured difference favours global.
- State that the Cell2Cell uplift has a positivity violation (all `Complain=1` customers are treated). It is neither
  validated out of sample nor causal. Cite the Hillstrom held-out Qini for the learners instead.
- Say that the direction check passes for random scores, so it cannot validate an uplift model.

## 8. Change log

| # | Change | Why | Evidence | Preserved |
|---|---|---|---|---|
| 1 | Added `eval_sop/` (scripts, `results/`, `agent_scenarios.csv`, `.gitignore` for `data/`) and this file | Deliverable | Each script names its inputs and outputs. Seed-42 reproduction matches the README exactly | No existing file modified |
| — | **No change to `src/`, `tests/`, `dashboard/`, `README.md`** | Every issue below is either already confirmed by a script in `eval_sop/` or not needed for a valid eval. The harness works around the label leak without editing src | `git diff main --stat` shows only additions | All comments and docs |

The existing test suite was run before committing: `python -m pytest -q` → 73 passed.

## 9. Proposed, not done

- Remove `actually_churned` from `src/agent_tools.py:204`, and select explicit columns in `route.ts` `lookup_customer_details`
  (label leakage to the LLM). I reproduced it by inspection only; there is no failing test yet.
- Make `src/uplift_model.py` fail loudly, not fall back, when CausalML cannot be imported. Reproduced: importing it in an
  env with numpy 2.5 logs "CausalML not available — using custom T-learner fallback" and continues.
- Replace the Cell2Cell treatment proxy, or drop `Complain` from it, and add a held-out Qini to the pipeline.
- Run the agent eval (section 4) once an Ollama slot is available. Run the Criteo subsample.
- Cleanup: the evaluation venv lives at `%TEMP%\sopvenv` (outside the repo, because of the Windows MAX_PATH limit under the
  scratchpad). It can be deleted after review.
