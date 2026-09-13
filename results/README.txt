VulProtoCL multi-seed outputs
================================
seed_<N>/metrics.csv     per-seed test metrics (protocl_gnn, bilstm_ref, protocl_late_fusion)
seed_<N>/hparams.json    argv, best_epoch, split file paths
seed_<N>/protocl_best.pt  final fused checkpoint
seed_<N>/ckpt_gnn.pt     GNN weights (resume skips GNN train)
seed_<N>/ckpt_bilstm.pt  BiLSTM weights (resume skips BiLSTM train)
seed_<N>/ckpt_scores.pt  val/test probabilities (resume skips forwards)
splits/reveal_seed<N>_*.txt  fingerprints (reveal_hash + code SHA-256)
  no suffix = full ReVeal; _capXXXX = subsampled debug run
reveal_full_seeds.csv    mean±std across seeds that currently have matching metrics.csv


