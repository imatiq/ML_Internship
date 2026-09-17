# Capstone Report — Structured Content Archetype Clustering

- **Author:** Atiq Ur Rehman
- **Lane:** Structured Content Archetype Clustering (Lane 3, unsupervised)
- **Repo:** https://github.com/imatiq/ML_Internship
- **Date:** 2026-09-17

## 0. Abstract

Can FlyRank content pages be grouped into a small number of interpretable archetypes from
observable search/engagement metrics alone, well enough to prioritize which pages a content
strategist reviews first? Using a 30,000-row anonymized slice of FlyRank's content-performance
data, we cluster pages with K-Means (k=6, chosen by silhouette score) into six archetypes and
rank them by decline rate. The resulting archetype queue does not beat a simple, transparent
rule-based baseline (precision@20: 0.50 vs 0.65; precision@50: 0.46 vs 0.50), and degrades
further under an honest client-held-out split (precision@20: 0.35). This is decision-support
evidence, not a production-ready score: the archetypes are moderately stable (ARI=0.59 across
reseeds) and useful for grouping pages into standing review buckets, but a reviewer with
limited time should lean on the rule baseline's reason codes over the archetype ranking alone.

## 1. Problem framing

**Decision:** instead of reviewing 30,000 pages one at a time, a content strategist decides
per-archetype (e.g. "protect champions," "rewrite thin/stale pages") and assigns each of six
clusters a standard playbook — refresh, monitor, deprioritize — then routes pages by cluster
membership.

**Unit of analysis:** one content page, described by a 90-day rolling snapshot of search and
engagement metrics. **Output:** a cluster/archetype label plus a ranked review queue.
**Action a human takes:** review the highest-decline-rate archetypes first, apply the
archetype's standing playbook, and use the reason for that page's placement to decide what to
fix.

**Cost of a wrong call:** a misclassified declining page grouped with a low-priority archetype
keeps losing visibility unreviewed; a healthy page wrongly grouped into a high-priority
archetype wastes review time. Clustering has no ground-truth label, so the safeguard is manual
inspection of real pages per archetype plus disclosing the honest, degraded, held-out numbers
rather than the flattering in-sample ones.

**Why ML/data helps here:** a fixed rule can only check conditions chosen in advance, one
dimension at a time. Single metrics are weak predictors on their own — `search_volume` vs
`impressions_90d` correlation is 0.001 — so archetypes likely emerge from combinations of
several metrics moving together that would be hard to guess by hand. Clustering finds these
combinations from the data's structure instead of a hard-coded guess.

## 2. Data safety

**Data used:** `data/raw/content_refresh_anonymized.csv` — 30,000 rows, 44 columns, one row
per pseudonymized content page, 32 pseudonymous clients. Single 90-day snapshot, not a time
series.

**Features used for clustering:** `log1p(impressions_90d)`, `ctr`, `avg_position`,
`engagement_rate`, `word_count`, `content_age_days`, `days_since_last_update` — all
standardized before clustering.

**Deliberately excluded, and why:**
- `trend_direction`, `trend_pct` — these define the evaluation label
  (`is_declining_label = trend_direction == "down"`); using them as features would be reading
  the answer key.
- `search_volume` — near-zero correlation (0.001) with actual traffic; adds noise, not signal.
- Pre-bucketed tier columns (`position_tier`, `impression_tier`, etc.) — double-count raw
  numeric fields already included.
- `provider_used`, `model_used` — product/process flags, never used in any score.

**Leakage risk considered:** four of the seven clustering features (`log_impressions`, `ctr`,
`avg_position`, `engagement_rate`) are built from the same 90-day window that structurally
overlaps the label's own last-30-day window. A confession test (refit without those four
features) showed the decline-rate spread across clusters drops from 0.505 to 0.364 — a real
but partial dependency, disclosed rather than hidden.

**Pseudonymous IDs:** `content_id` and `client_id` are used for grouping and the client-holdout
split only, never as clustering features, and never mapped back to real identities.

**Confirmed:** no client names, domains, URLs, page titles, or raw search queries appear
anywhere in `work/`.

## 3. Baseline

**The rule (ML-07):** a page is worth reviewing first if it's still pulling real search
visibility, and at least one of: hasn't been touched in a long time, is thin for how much it's
seen, is already ranking well enough that a push could move it further, or is ranking decently
but not getting clicks. `trend_direction`/`trend_pct` are deliberately kept out of the rule and
its reason codes, since those are what the evaluation label is built from.

**Reason codes:** `stale_but_visible`, `thin_but_visible`, `rankable_position`,
`ranking_low_ctr`, `low_priority` (no codes fired).

**Fairness of the comparison:** transparent, no fitted weights, evaluated on the same data and
metric (precision@K, base rate) as the model.

**Numbers:** base rate 0.542. Precision@20 = 0.65 (lift +0.109 over base rate). Precision@50 =
0.50 — actually *below* the base rate (lift −0.041), meaning past the top ~20–30 picks the
rule does no better than random. The weakness traces to the `log1p(impressions)` multiplier
being strong enough that the top of the queue is effectively "biggest pages with lowest CTR,"
crowding out `thin_but_visible`/`stale_but_visible` pages from ever reaching the top 20.

## 4. Model / analysis

**Method:** K-Means, `k` picked by silhouette score, clusters named after inspecting real
pages in each one. This fits the lane because it is a grouping question, not a yes/no
classification — there is no pre-existing archetype label.

**Feature list:** `log1p(impressions_90d)`, `ctr`, `avg_position` (0 = "no data," imputed to
median rather than treated as rank zero), `engagement_rate`, `word_count`,
`content_age_days`, `days_since_last_update` — all `StandardScaler`-scaled.

**Target/proxy:** none — clustering is unsupervised. Cluster membership itself is the proxy,
evaluated post hoc against `is_declining_label` (never used during fitting).

**Picking k:** swept k=2..8; silhouette rose from 0.225 (k=2) to a local high of 0.304 at k=6,
dipped at k=7, climbed again at k=8. k=6 chosen for the local peak and because six is a
manageable, nameable number of archetypes for a content team.

## 5. Evaluation

**Split:** clustering has no future value to hold out, so two checks replace a conventional
train/test split:
1. **Reseed check** — refit with a different random seed; Adjusted Rand Index = 0.59
   (moderate agreement — roughly a third of pages land in a different cluster between runs).
2. **Client-holdout check** — split the 32 pseudonymous clients 70/30 (22 train, 10 test), fit
   the scaler + K-Means on train clients only, assign test rows to the nearest fitted cluster,
   and rank using train-derived decline rates (test labels never touched during fitting or
   ordering).

**Metrics, model vs. baseline, same metric (precision@K, base rate visible):**

| Evaluation | Base rate | Precision@20 | Precision@50 |
|---|---|---|---|
| ML-07 rule baseline | 0.542 | **0.65** | **0.50** |
| K-Means archetype queue — in-sample | 0.542 | 0.50 | 0.46 |
| K-Means archetype queue — honest, client-grouped holdout | 0.502 | 0.35 | 0.32 |

**Error analysis:** the top of the in-sample archetype queue is entirely `Stale Heavyweights`
pages, and 10 of the top 20 are not actually declining. The wrong picks average ~154,000
impressions and ~6,000 words at a decent-not-great position (~21) — the same failure mode as
the rule baseline: sorting by raw visibility inside a group re-surfaces "biggest pages," not
genuinely at-risk ones, because ranking within an archetype still has to break ties on
something, and impressions is the strongest available signal in both methods.

## 6. Interpretation

**What the clusters found** (profile, sorted by decline rate):

| Archetype | n | Median impressions | Decline rate |
|---|---|---|---|
| Stale Heavyweights | 2,887 | 4,599 | 0.65 |
| Aging Page-One (Engagement Gap) | 5,925 | 1,159 | 0.62 |
| Young & Slipping | 11,058 | 555.5 | 0.61 |
| Established Performers | 7,352 | 565.5 | 0.41 |
| Low-Demand Long-Tail | 2,638 | 256 | 0.37 |
| Near-Zero-Traffic | 140 | 3 | 0.14 |

Cluster centers show `word_count` and `days_since_last_update` carry the most separation for
`Stale Heavyweights` (loadings 2.33 and 1.35); `days_since_last_update` alone drives `Aging
Page-One` (1.45); `content_age_days` distinguishes `Established Performers` from `Young &
Slipping`. Length and staleness, not raw visibility, do most of the clustering work — even
though visibility (impressions) is what dominates the *ranking within* each archetype. Those
are two different jobs the features are doing, worth not conflating.

**Negative result, stated plainly:** the archetype queue does not beat the simpler rule
baseline on either precision cut, and the gap widens under honest evaluation. That is a valid,
useful finding — it tells a content team the simpler rule may be the better first filter, with
archetypes as a complementary lens for interpretability rather than a superior ranking.

## 7. Recommendation

**Ranked archetype playbook** (from `work/notebooks/w07_action_playbook.ipynb`, ML-10),
ordered by decline rate:

1. **Stale Heavyweights** (n=2,887, 0.65) → `refresh` — long, old-updated, still-visible pages;
   refresh content and update dates first.
2. **Aging Page-One (Engagement Gap)** (n=5,925, 0.62) → `refresh_title_and_meta` — ranking
   near page 1 but stale with thin engagement; cheapest high-leverage fix.
3. **Young & Slipping** (n=11,058, 0.61) → `monitor_closely` — newer pages already at risk;
   watch, don't rewrite yet.
4. **Established Performers** (n=7,352, 0.41) → `monitor` — below base rate; routine check-ins.
5. **Low-Demand Long-Tail** (n=2,638, 0.37) → `low_priority` — deep position, low impressions.
6. **Near-Zero-Traffic** (n=140, 0.14) → `deprioritize` — almost no impressions; quiet, not
   failing.

**How a FlyRank editor uses this tomorrow:** pull the top of
`work/outputs/archetype_action_playbook.csv`, start with `Stale Heavyweights` and `Aging
Page-One`, and cross-check each pick against the simpler ML-07 rule queue before committing
review time — the rule caught more true declines at the very top (P@20=0.65) even though it
can't explain *why* a page is grouped the way the archetype profile can.

**Confidence:** moderate for archetype-level triage buckets as a grouping/interpretability
tool; low for trusting any single row's rank in isolation, and low for using the archetype
queue as the *sole* ranking ahead of the simpler rule baseline.

## 8. Reproducibility

**Re-run from a fresh clone:**
```bash
git clone https://github.com/imatiq/ML_Internship
cd ML_Internship
pip install -r requirements.txt
# open and run top-to-bottom, in order:
#   work/notebooks/w01_research_question.ipynb
#   work/notebooks/w02_ml_task_framing.ipynb
#   work/notebooks/w03_data_contract.ipynb
#   work/notebooks/w04_baseline_score.ipynb
#   work/notebooks/w04_signal_audit.ipynb
#   work/notebooks/w05_model.ipynb
#   work/notebooks/w06_validation_audit.ipynb
#   work/notebooks/w07_action_playbook.ipynb
#   work/notebooks/capstone.ipynb
```

**Seeds:** `random_state=42` for the primary K-Means fit and the client-holdout split;
`random_state=7` for the reseed stability check; `random_state=0`/`42` for the two client-half
splits in ML-08/ML-09.

**Environment:** `requirements.txt` — `pandas>=2.2`, `numpy>=1.26`, `scikit-learn>=1.4`,
`matplotlib>=3.8`.

**Committed evaluation artifacts** (the receipts the numbers above trace back to):
- `work/outputs/baseline_action_score.csv` — ML-07 rule-scored, ranked queue.
- `work/outputs/archetype_action_playbook.csv` — ML-10 archetype-scored, ranked queue.
- `work/outputs/playbook_summary.json` — archetype profile, action mix, precision@K numbers.
- `work/outputs/charts/decline_rate_by_archetype.svg`, `work/outputs/charts/action_mix.svg`.

The honest client-holdout evaluation (Section 5) is rebuilt directly in
`work/notebooks/w06_validation_audit.ipynb`, Section 2 — the exact split, fit, and scoring code
is in that notebook, not just the resulting numbers.

## 9. Acknowledgments & data credit

Built on the FlyRank ML Internship dataset — https://flyrank.ai
