# APEX-R GNN proxy experiment v1 — RAM-safe execution report

Conclusion: `EXPLORATORY_PROXY_GAIN`. This remains a fixed-pair classified-order boundary proxy, not verified overtaking, attack-conditioned probability, or strategy evidence.

## Scope and population

- Disk-backed cache graphs: **15404** of 15423 labelled positive/negative windows; excluded 19.
- Eligible graph support: train 10671 (613 positive), validation 2533 (103 positive), development-test 2200 (106 positive).
- The proposed chronology-only split is 26/6/5 races. All are previously accessed development data; development-test is not an untouched final holdout.
- Protected session `11353` is excluded and no remote data was requested.

## RAM repair

The legacy cache was not loaded with `torch.load()`. Raw graph arrays are float64 memory-mapped; graph metadata is small; each Data object is created lazily with the original float32 tensor boundary and original DataLoader settings. Streaming preprocessing uses development-train graphs only and the same mean/std formulas mathematically.

Measured load/preprocessing/one-batch gate: **498.5 MiB peak RSS**, finite forward/backward loss and gradients, and graph equivalence exact for 15,404 graphs with max numeric difference 0.0. Stage measurements are in `ram_safe_measurement.json`. The failed legacy process reached 7.4 GB RSS and was kernel-OOM-killed, documented in the preceding diagnostic response.

Configuration unchanged: **True**. The canonical configuration is `EXPERIMENT_SPEC.md`; `ram_fix_config_after.json` records the comparison.

## Frozen experiment

- Two GINEConv layers, hidden 32, dropout 0.2; Adam lr 0.001, weight decay 0.0001; batch 32; unweighted BCE-with-logits; max 100 epochs; patience 15; seeds 17, 23, 42.
- Current as-of joins are backward-only with 1.0s tolerance; trailing lookback is 10s; no interpolation, centred smoothing, whole-lap aggregate, or classified-adjacency edge.
- Validation AP selects GNN checkpoints. Thresholds are validation-only maximum-F1 descriptive thresholds and were frozen before development-test access.

## Results

| Model | Split | Seed | n | Pos | Prevalence | AP | ROC-AUC | Brier | Log loss | Val-selected threshold |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| constant | development_train |  | 10671 | 613 | 0.057445 | 0.057445 | 0.500000 | 0.054145 | 0.219880 | t=0.0574454; P=0.0574; R=1.0000; F1=0.1086; 613/10058/0/0 |
| constant | development_validation |  | 2533 | 103 | 0.040663 | 0.040663 | 0.500000 | 0.039291 | 0.172927 | t=0.0574454; P=0.0407; R=1.0000; F1=0.0781; 103/2430/0/0 |
| logistic_c1 | development_train |  | 10671 | 613 | 0.057445 | 0.136170 | 0.707279 | 0.052220 | 0.204406 | t=0.125603; P=0.1684; R=0.2594; F1=0.2042; 159/785/454/9273 |
| logistic_c1 | development_validation |  | 2533 | 103 | 0.040663 | 0.089998 | 0.692301 | 0.038380 | 0.161098 | t=0.125603; P=0.1472; R=0.2330; F1=0.1805; 24/139/79/2291 |
| gnn_seed_17 | development_train | 17 | 10671 | 613 | 0.057445 | 0.323501 | 0.861135 | 0.045961 | 0.166582 | t=0.14239; P=0.2781; R=0.5171; F1=0.3617; 317/823/296/9235 |
| gnn_seed_17 | development_validation | 17 | 2533 | 103 | 0.040663 | 0.120806 | 0.709765 | 0.039257 | 0.164988 | t=0.14239; P=0.1446; R=0.3495; F1=0.2045; 36/213/67/2217 |
| gnn_seed_23 | development_train | 23 | 10671 | 613 | 0.057445 | 0.274835 | 0.844741 | 0.047321 | 0.172767 | t=0.189104; P=0.2817; R=0.4095; F1=0.3338; 251/640/362/9418 |
| gnn_seed_23 | development_validation | 23 | 2533 | 103 | 0.040663 | 0.106054 | 0.661193 | 0.040798 | 0.171819 | t=0.189104; P=0.1421; R=0.2718; F1=0.1867; 28/169/75/2261 |
| gnn_seed_42 | development_train | 42 | 10671 | 613 | 0.057445 | 0.308854 | 0.857302 | 0.046692 | 0.169556 | t=0.174238; P=0.3134; R=0.3752; F1=0.3415; 230/504/383/9554 |
| gnn_seed_42 | development_validation | 42 | 2533 | 103 | 0.040663 | 0.117812 | 0.694195 | 0.038718 | 0.166329 | t=0.174238; P=0.1770; R=0.1942; F1=0.1852; 20/93/83/2337 |
| constant | development_test |  | 2200 | 106 | 0.048182 | 0.048182 | 0.500000 | 0.045946 | 0.193963 | t=0.0574454; P=0.0482; R=1.0000; F1=0.0919; 106/2094/0/0 |
| logistic_c1 | development_test |  | 2200 | 106 | 0.048182 | 0.077608 | 0.631805 | 0.045776 | 0.188596 | t=0.125603; P=0.0876; R=0.1132; F1=0.0988; 12/125/94/1969 |
| gnn_seed_17 | development_test | 17 | 2200 | 106 | 0.048182 | 0.113141 | 0.686706 | 0.046788 | 0.193665 | t=0.14239; P=0.0973; R=0.2736; F1=0.1436; 29/269/77/1825 |
| gnn_seed_23 | development_test | 23 | 2200 | 106 | 0.048182 | 0.102951 | 0.667059 | 0.048933 | 0.198637 | t=0.189104; P=0.0879; R=0.1981; F1=0.1217; 21/218/85/1876 |
| gnn_seed_42 | development_test | 42 | 2200 | 106 | 0.048182 | 0.106208 | 0.702501 | 0.046231 | 0.187104 | t=0.174238; P=0.1210; R=0.1792; F1=0.1445; 19/138/87/1956 |

Full PR curves: `precision_recall_curves.csv`; reliability bins and counts: `reliability_bins.csv`; per-window predictions: `per_window_predictions.csv`.

Training runs:
- seed 17: best epoch 24, epochs 39, validation AP 0.120806, 126.27s
- seed 23: best epoch 21, epochs 36, validation AP 0.106054, 114.14s
- seed 42: best epoch 22, epochs 37, validation AP 0.117812, 117.07s

## Test protocol and limitations

`FROZEN_BEFORE_TEST.json` was written before test predictions. Each GNN seed was evaluated once; no revision followed test access. Whole-race bootstrap AP deltas and degenerate resamples are in `whole_race_bootstrap.csv`; five development-test races make these intervals limited. The GNN has other-car nodes and physical-neighbour edges while logistic regression has pair-level fields, so a gain does not isolate message passing.

## Verification

Automated checks: **PASSED**; see `experiment_tests.json`, `inference_parity.json`, `ram_fix_cache_conversion.json`, and input hash files. No model is connected to ATTACK/HOLD/HARVEST.
