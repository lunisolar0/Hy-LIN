# Hy-LIN

Hybrid latent interaction network for bug triage. A frozen text encoder fixes a sparse set of bug–bug candidates, a valve learns how much each candidate contributes, and two paths score developers: an explicit path on the augmented assignment graph and an implicit path on a developer graph built from textual centroids. A frequency gate mixes the two standardized scores.

The locked configuration is `configs/locked_model.json`.

## Data

`data/processed/` holds the cleaned GC, MC, and MF records used by the paper (`owner`, `issue_title`, `description`), stored compressed because two of the files exceed GitHub's 100MB limit.

```bash
gzip -dk data/processed/gc.json.gz data/processed/mc.json.gz data/processed/mf.json.gz
```

## Train the locked model

The encoder is `sentence-transformers/all-MiniLM-L6-v2`. The code loads it with `local_files_only=True`, so the weights have to be in the local Hugging Face cache before training. From `code/`:

```bash
python -m hy_lin.train \
  --data ../data/processed/mf.json \
  --encoder minilm \
  --dev-graph centroid \
  --k-bug 20 --k-dev 15 \
  --exp-layers 2 --imp-layers 2 \
  --dropout 0.1 --lr 0.001 --weight-decay 0.0001 \
  --batch-size 2048 --epochs 40 --patience 10 \
  --tau 0.0 --freq-exponent 0.5 \
  --fusion zscore --story 0.5 --low-coef 0.35 \
  --imp-query prop --gate-mode freq --valve bug \
  --seed 42
```

Replace `mf.json` with `gc.json` or `mc.json` for the other two collections. `scripts/baselines/run_hylin.py` launches this locked setting on all three datasets and seeds 42, 43, and 44. The other scripts in `scripts/baselines/` are the comparison methods under the same release-order split.

`tables/` contains the result tables reported in the paper.
