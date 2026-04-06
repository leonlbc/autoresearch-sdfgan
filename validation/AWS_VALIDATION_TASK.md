# AWS Validation Task — Handoff Document

## Context

This repo implements an SDF-GAN (Stochastic Discount Factor via adversarial training)
for asset pricing. An autonomous agent ran 55 experiments modifying `train.py`, improving
`valid_sharpe` from 0.08 to 1.63 on a single 60-month validation split.

We just ran `validate.py` locally on the final model (exp55, commit `e68c340`). This
script re-trains the model on 6 rolling 240-month windows and evaluates each on the next
60 months OOS, producing 360 months of robust out-of-sample data. The result: **aggregate
OOS Sharpe of 0.985** — the single-split 1.63 was inflated by ~65%.

We now need to run the same validation on **two earlier checkpoints** to complete the
comparison for the thesis (TCC). This will be done on AWS to free the local GPU.

## The two validation runs needed

### 1. Baseline (commit `a1fc226`)
- The original unmodified model before any experiments
- Single-split valid_sharpe was 0.081
- Purpose: prove the improvements are real, not inherent in the method

### 2. Exp 42 (commit `0525cc8`)
- Hidden [256, 192], dropout 0.12, GELU, gradient clipping — before FiLM/input noise
- Single-split valid_sharpe=1.502, **test_sharpe=0.736** (best test of any experiment)
- Purpose: test whether exps 43-55 (FiLM, input noise) genuinely helped or just fit
  validation noise. The methodological review suspects this was the true OOS peak.

### Important: how to run each

`validate.py` and `prepare.py` must be the **current versions** (from HEAD). Only
`train.py` changes between runs — it contains the model architecture and `Config` that
`validate.py` imports. So for each run:

1. Use current `validate.py` and `prepare.py`
2. Extract `train.py` from the target commit: `git show <commit>:train.py > train.py`
3. Upload to instance and run `python -u validate.py`

## AWS setup

Scripts are in `aws/`. The infrastructure is NOT yet created — everything needs to run
from scratch.

### Step-by-step

```bash
cd aws

# 1. Upload datasets to S3 (one-time, ~5 min from local, 2.3 GB)
bash upload-data.sh

# 2. Launch spot instance (g4dn.xlarge, T4 16GB, sa-east-1, ~$0.12/hr)
bash launch.sh

# 3. Upload code + pull data from S3 on instance + install PyTorch
bash upload.sh

# 4. Run baseline validation
#    First, swap train.py to baseline version:
#    git show a1fc226:train.py > ../train.py
#    Then re-upload code:
#    bash upload.sh
#    Then run:
bash run-job.sh validate
#    Save output, then...

# 5. Run exp42 validation
#    Swap train.py to exp42 version:
#    git show 0525cc8:train.py > ../train.py
#    Re-upload code:
#    bash upload.sh
bash run-job.sh validate

# 6. Download results
bash download.sh

# 7. ALWAYS terminate when done
bash terminate.sh
```

### Expected timing and cost

- Each validate.py run: ~30 min on T4 (vs 102 min on local RTX 3050)
- Total instance time: ~1.5 hours (including setup, both runs)
- Spot cost: ~$0.12/hr × 1.5h = **~$0.18**
- S3 storage: ~$0.06/month (negligible)

## Recording results

Save results to `validation/` following the same format as `validate_exp55.txt`:

- `validation/validate_baseline.txt`
- `validation/validate_exp42.txt`

Also update `TRANSFORMATIONS.md` (in `../TCC-Leon/`) with the comparison table.

## AWS credentials

- User: `admin-leon` (account 730335252491)
- Region: `sa-east-1` (São Paulo)
- AWS CLI is already configured and authenticated

## What NOT to do

- Do not forget to run `bash terminate.sh` — spot instance bills per second
- Do not use the current `train.py` from HEAD — that's exp55, already validated locally
