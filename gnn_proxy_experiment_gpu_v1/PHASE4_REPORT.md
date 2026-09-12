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
| gnn_seed_17 | development_train | 17 | 10671 | 613 | 0.057445 | 0.385431 | 0.883273 | 0.043978 | 0.160264 | t=0.302112; P=0.5506; R=0.2398; F1=0.3341; 147/120/466/9938 |
| gnn_seed_17 | development_validation | 17 | 2533 | 103 | 0.040663 | 0.123615 | 0.692489 | 0.039326 | 0.182786 | t=0.302112; P=0.2542; R=0.1456; F1=0.1852; 15/44/88/2386 |
| gnn_seed_23 | development_train | 23 | 10671 | 613 | 0.057445 | 0.244156 | 0.827195 | 0.048411 | 0.178970 | t=0.212208; P=0.2891; R=0.3197; F1=0.3036; 196/482/417/9576 |
| gnn_seed_23 | development_validation | 23 | 2533 | 103 | 0.040663 | 0.108146 | 0.685377 | 0.040267 | 0.169460 | t=0.212208; P=0.1449; R=0.1942; F1=0.1660; 20/118/83/2312 |
| gnn_seed_42 | development_train | 42 | 10671 | 613 | 0.057445 | 0.334535 | 0.869245 | 0.045671 | 0.164052 | t=0.184664; P=0.3351; R=0.4078; F1=0.3679; 250/496/363/9562 |
| gnn_seed_42 | development_validation | 42 | 2533 | 103 | 0.040663 | 0.127602 | 0.689272 | 0.038754 | 0.166081 | t=0.184664; P=0.1776; R=0.1845; F1=0.1810; 19/88/84/2342 |
| constant | development_test |  | 2200 | 106 | 0.048182 | 0.048182 | 0.500000 | 0.045946 | 0.193963 | t=0.0574454; P=0.0482; R=1.0000; F1=0.0919; 106/2094/0/0 |
| logistic_c1 | development_test |  | 2200 | 106 | 0.048182 | 0.077608 | 0.631805 | 0.045776 | 0.188596 | t=0.125603; P=0.0876; R=0.1132; F1=0.0988; 12/125/94/1969 |
| gnn_seed_17 | development_test | 17 | 2200 | 106 | 0.048182 | 0.099680 | 0.667189 | 0.047411 | 0.219294 | t=0.302112; P=0.1707; R=0.0660; F1=0.0952; 7/34/99/2060 |
| gnn_seed_23 | development_test | 23 | 2200 | 106 | 0.048182 | 0.088548 | 0.652745 | 0.049294 | 0.199800 | t=0.212208; P=0.0863; R=0.1604; F1=0.1122; 17/180/89/1914 |
| gnn_seed_42 | development_test | 42 | 2200 | 106 | 0.048182 | 0.104338 | 0.700857 | 0.045965 | 0.190365 | t=0.184664; P=0.1304; R=0.1415; F1=0.1357; 15/100/91/1994 |

Full PR curves: `precision_recall_curves.csv`; reliability bins and counts: `reliability_bins.csv`; per-window predictions: `per_window_predictions.csv`.

Training runs:
- seed 17: best epoch 33, epochs 48, validation AP 0.123615, 171.07s
- seed 23: best epoch 16, epochs 31, validation AP 0.108146, 108.31s
- seed 42: best epoch 28, epochs 43, validation AP 0.127602, 152.35s

## Test protocol and limitations

`FROZEN_BEFORE_TEST.json` was written before test predictions. Each GNN seed was evaluated once; no revision followed test access. Whole-race bootstrap AP deltas and degenerate resamples are in `whole_race_bootstrap.csv`; five development-test races make these intervals limited. The GNN has other-car nodes and physical-neighbour edges while logistic regression has pair-level fields, so a gain does not isolate message passing.

## Verification

Automated checks: **PASSED**; see `experiment_tests.json`, `inference_parity.json`, `ram_fix_cache_conversion.json`, and input hash files. No model is connected to ATTACK/HOLD/HARVEST.
