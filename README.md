# Methods for Detecting Vulnerabilities in Open-Source Software
### VulProtoCL - Prototype Contrastive Learning on Slice Property Graphs for Function-Level Vulnerability Detection


Function-level C/C++ vulnerability detection.
A BiLSTM over tokens is mixed with an R-GCN over a tree-sitter slice graph
(dual views, prototypes, motif features). 

Main ReVeal result (full dump, seeds 42–44):

| Method | F1 | PR-AUC | ROC-AUC |
|--------|----|--------|---------|
| **VulProtoCL fusion** | **0.433 ± 0.011** | **0.325 ± 0.034** | 0.827 ± 0.021 |
| BiLSTM | 0.413 ± 0.018 | 0.292 ± 0.031 | 0.824 ± 0.009 |
| ProtoCL-GNN | 0.340 ± 0.032 | 0.320 ± 0.050 | 0.765 ± 0.019 |
| Snapshot fusion (prior hybrid) | 0.417 ± 0.013 | 0.317 ± 0.031 | 0.829 ± 0.006 |

On a 2026 CVE probe the ReVeal-tuned fusion ROC ≈ 0.65. See `results/`.

## Setup

Python 3.10, CUDA 12.4.

```bash
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

Data is at `datasets`.

## Reproduce the main table

Train VulProtoCL on full ReVeal, three seeds:

```bash
python -u experiment_multiseed_run_protocl.py --max-train 0 --max-val 0 --max-test 0 --seeds 42,43,44 --epochs-gnn 8 --epochs-seq 8 --batch-gnn 6 --lambda-proto 0.25 --lambda-view 0.15
```

Same-split snapshot fusion (BiLSTM + VulCPGPlus):

```bash
python -u run_late_fusion.py --max-train 0 --max-val 0 --max-test 0 --epochs-seq 8 --epochs-gnn 6 --batch-gnn 6 --seed 42
```

Ablations (one flag per job; writes `results/ablations/<tag>/`):

```bash
python -u experiment_multiseed_run_protocl.py --max-train 0 --max-val 0 --max-test 0 --seeds 42,43,44 --epochs-gnn 8 --epochs-seq 8 --batch-gnn 6 --lambda-proto 0.25 --lambda-view 0.15 --no-prototypes
```

Flags: `--no-dual-view` · `--no-prototypes` · `--no-motifs` · `--no-fusion`.
`--no-dual-view` is already in `results/ablations/nodual/` (fusion F1 0.426 ± 0.009).

2026 CVE probe (needs ReVeal checkpoints `results/seed_*/protocl_best.pt`):

```bash
python -u run_wild_ood_protocl.py --seeds 42,43,44
```

## Layout

```
cpg_vuln/          graphs, slicer, motifs, R-GCN models
vulprotocl/        data, metrics, BiLSTM
run_protocl.py     one seed of VulProtoCL + fusion
experiment_multiseed_run_protocl.py
run_late_fusion.py snapshot hybrid
run_bilstm_baseline.py
run_improved_experiments.py   VulCPGGNN / VulCPGPlus
results/           reported CSVs (main evidence)
```


