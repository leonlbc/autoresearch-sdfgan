# SDF-GAN Autoresearch

This is an experiment to have an LLM autonomously improve a deep learning asset pricing model.

The model learns a Stochastic Discount Factor (SDF) via adversarial training (GAN), following Chen, Pelger & Zhu (2019) — *"Deep Learning in Asset Pricing"*.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar9`). The branch `autoresearch/<tag>` must not already exist.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current main.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `prepare.py` — fixed data loading and evaluation harness. **Do not modify.**
   - `train.py` — the file you modify. Model architecture, optimizer, training loop.
4. **Verify data exists**: Check that `datasets/` contains `char/Char_train.npz` etc. If not, tell the human to download the data.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment trains the SDF-GAN model end-to-end (3 phases: unconditional → moment update → conditional). You launch it as: `python train.py`.

**What you CAN do:**
- Modify `train.py` — this is the only file you edit. Everything is fair game: model architecture, optimizer, hyperparameters, training schedule, loss functions, regularization, activation functions, etc.

**What you CANNOT do:**
- Modify `prepare.py`. It is read-only. It contains the fixed evaluation, data loading, and loss function definitions.
- Install new packages or add dependencies.
- Modify the evaluation harness. The `evaluate()` and `evaluate_all_splits()` functions in `prepare.py` are the ground truth metrics.

**The goal is simple: get the highest valid_sharpe while generalizing well.** The primary metric is:

```
score = valid_sharpe
```

Higher is better. Everything is fair game: change the architecture, the optimizer, the hyperparameters, the training schedule. The only constraint is that the code runs without crashing. The test set is never used for model selection — test_sharpe is recorded for reporting only.

**Training time** is a soft constraint. Each run should complete in a reasonable time (~2-30 minutes depending on architecture). If a run takes more than 30 minutes, kill it and treat it as a failure.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. A 0.001 Sharpe improvement from deleting code? Definitely keep. A 0.001 improvement that adds 20 lines of hacky code? Probably not worth it.

**The first run**: Your very first run should always be to establish the baseline, so run the training script as is.

## Output format

Once the script finishes it prints a summary like this:

```
---
valid_sharpe:     0.452300
test_sharpe:      0.389100
train_sharpe:     0.523400
valid_loss:       0.001234
test_loss:        0.001456
train_loss:       0.000987
valid_ev:         0.031200
test_ev:          0.028500
train_ev:         0.040100
peak_vram_mb:     1234.5
```

You can extract the key metric from the log file:

```bash
grep "^valid_sharpe:" run.log
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated).

The TSV has a header row and 7 columns:

```
commit	valid_sharpe	test_sharpe	train_sharpe	valid_ev	status	description
```

1. git commit hash (short, 7 chars)
2. valid_sharpe achieved — use 0.000000 for crashes
3. test_sharpe achieved (reporting only, not used for selection) — use 0.000000 for crashes
4. train_sharpe achieved — use 0.000000 for crashes
5. valid_ev (explained variation) — use 0.000000 for crashes
6. status: `keep`, `discard`, `discard (overfit)`, or `crash`
7. short text description of what this experiment tried

Example:

```
commit	valid_sharpe	test_sharpe	train_sharpe	valid_ev	status	description
a1b2c3d	0.452300	0.389100	0.523400	0.031200	keep	baseline
b2c3d4e	0.480100	0.410200	0.550000	0.035000	keep	increase hidden dims to [128 64]
c3d4e5f	0.420000	0.350000	0.800000	0.028000	discard	switch to GRU
d4e5f6g	0.000000	0.000000	0.000000	0.000000	crash	attention model (shape mismatch)
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/mar9`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Modify `train.py` with an experimental idea by directly hacking the code.
3. git commit
4. Run the experiment: `python train.py > run.log 2>&1` (redirect everything)
5. Read out the results: `grep "^valid_sharpe:\|^test_sharpe:\|^train_sharpe:\|^valid_ev:" run.log`
6. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the stack trace and attempt a fix. If you can't fix it after a few attempts, give up on that idea.
7. Record `score = valid_sharpe` and all results in the TSV (do NOT commit results.tsv — leave it untracked)
8. Apply the **keep criteria** (see below) to decide keep vs discard
9. If kept, you "advance" the branch, keeping the git commit
10. If discarded, you `git reset --hard HEAD~1` back to where you started

### Keep criteria

An experiment is **kept** only if ALL of the following hold:
- **score improved** (higher than the previous best `valid_sharpe`)
- **train_sharpe / valid_sharpe < 2.0** (if the ratio exceeds 2x, the model is overfitting — try reducing capacity or increasing regularization instead of keeping)

### Generalization audits

Every **10 experiments**, pause and review the last 10 entries in results.tsv:
- If valid_sharpe has been increasing but the train/valid Sharpe ratio has been growing, **stop scaling up** (wider layers, more epochs) and shift focus to regularization or simplification.
- Look for discarded experiments with interesting approaches that might combine well with the current best.

### Combinatorial exploration

The default strategy is one-change-at-a-time greedy search. However, every **15 experiments**, revisit near-misses:
- Look for discarded experiments that were close to the best score or had interesting properties
- Try combining 2-3 of their ideas with the current best model
- This prevents discarding changes that might synergize but scored slightly lower individually

### Rolling-window validation (`validate.py`)

`validate.py` re-trains the current best model on 6 rolling 240-month windows and evaluates each on the next 60 months out-of-sample. It produces a robust, multi-window OOS Sharpe that is far less noisy than the single 60-month validation split used in the main loop. **It is expensive** (~6× a normal run), so do not run it routinely — only when one of the following conditions is met:

1. **Big valid_sharpe jump**: The current best `valid_sharpe` has improved by ≥ 0.05 since the last validation run (or since baseline, if never validated). A single 60-month window is noisy; confirm the gain is real before building further on it.
2. **Generalization audit concern**: A generalization audit (every 10 experiments) reveals a rising train/valid Sharpe ratio or stalling valid_sharpe despite train improvements. Use the rolling window to check whether the current best is genuinely robust or is starting to overfit.
3. **Pre-commitment check**: You are about to make a major architectural change (new layer type, new conditioning mechanism, fundamentally different structure). Validate the current foundation first so you have a reliable baseline to compare against after the change.
4. **Long discard streak**: 5+ consecutive experiments have been discarded without improving. Validate the current best — if the rolling-window Sharpe is weak, consider reverting further back rather than continuing to micro-optimize a fragile checkpoint.

**How to run:** `python validate.py > validate_run.log 2>&1`, then inspect the output. The script prints per-window OOS Sharpe and an aggregate summary. Do NOT log validation results in `results.tsv` — that file is only for the main experiment loop. Instead, note the rolling-window Sharpe in your commit message or print it for the human.

**Timeout**: Each experiment should take ~2-30 minutes. If a run exceeds 30 minutes, kill it and treat it as a failure (discard and revert).

**Crashes**: If a run crashes, use your judgment: if it's a typo or easy fix, fix and re-run. If the idea is fundamentally broken, skip it, log "crash", and move on.

**NEVER STOP**: Once the experiment loop has begun, do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep or away and expects you to continue working *indefinitely* until manually stopped. You are autonomous. If you run out of ideas, think harder — re-read the code, try combining previous near-misses, try more radical changes. The loop runs until the human interrupts you, period.

## Domain knowledge: Asset Pricing with SDF-GAN

### The SDF framework
The Stochastic Discount Factor $M_t = 1 + \sum_i R_{t,i} \cdot w_{t,i}$ prices all assets. The model learns weights $w$ that minimize the moment condition $E[R \cdot M \cdot h] = 0$ for all conditioning instruments $h$.

### Architecture
- **Model Layer** (generator): maps features → per-stock weights → SDF
- **Moment Layer** (adversary): finds the hardest conditioning instruments $h \in [-1, 1]$
- Phase 1: train against unconditional (constant) instruments
- Phase 2: update adversary to find harder tests
- Phase 3: train against the harder adversary → better model

### Things that tend to work (confirmed by experiments)
- Dropout at 0.12 is the sweet spot (0.05 paper default was too low, 0.15 too high)
- GELU activation outperforms ReLU, ELU, and SiLU
- Wider hidden layers help up to a point: [256, 192] is the current optimum
- Gradient clipping at max_norm=1.0 stabilizes training
- Longer Phase 3 training helps up to P3=768 (diminishing returns beyond)
- Group-wise FiLM conditioning (4 groups) for macro→individual modulation
- LSTM for macro features (GRU had good test_sharpe but worse valid_sharpe)
- The 3-phase training schedule matters — don't skip phases
- Validation Sharpe is noisy with only 60 months — changes < 0.01 may be noise

### Research directions (ordered by expected impact, given 46 experiments of prior work)

**High priority — unexplored or promising:**
1. Macro-individual conditioning: FiLM (4 groups) was the best late-stage win. Try cross-attention between macro and stock features, hypernetworks, or adaptive gating mechanisms
2. Skip / residual connections in the dense block (never tried, low-risk)
3. Attention over stocks: cross-sectional attention to capture inter-stock structure (never tried)
4. LR warmup + cosine decay: warmup was never tried (only constant LR and pure cosine were)
5. Mixture of SDFs / multi-factor models (never tried, potentially high impact)
6. Bidirectional RNN for macro features (never tried)
7. Weight clipping on generator (never tried)
8. Curriculum learning: phase 3 with gradually increasing adversary difficulty (never tried)

**Medium priority — partially explored, room left:**
9. sub_epoch count: only 4 (kept) and 8 (timed out) tried — try 6
10. Moment layer depth: hidden [32] didn't help, but deeper (e.g. [16, 16]) untested
11. Combining near-miss discards: GRU had best test_sharpe of any early discard; FiLM groups=2 had test=0.733 — try combining good-test ideas

**Settled — do not re-explore without strong reason:**
- Activation functions: GELU is the winner (ReLU, ELU, SiLU all tried)
- Hidden width: [256, 192] optimal; wider ([384, 128]) and deeper ([256, 128, 64]) both hurt
- Dropout: 0.12 optimal (0.05, 0.08, 0.10, 0.13, 0.15 all tried)
- L1/L2/spectral/batch/layer norm: all hurt — avoid explicit regularization beyond dropout
- Phase epochs: P3=768 saturated (384, 512, 768, 1024 all tried)
- Optimizer: Adam at lr=1e-3 is best (AdamW, lr=5e-4, lr=2e-3 all tried)
- Gradient clipping: 1.0 optimal (0.5 hurt)
- Kaiming init: hurt (stick with Glorot/Xavier)

### Things to AVOID
- Removing the 3-phase structure entirely
- Very large models that overfit 240 months of training data
- Training for too many epochs without early stopping
- Changing the evaluation metric or data preprocessing
- Repeatedly scaling up (wider, longer) without checking the train/valid Sharpe ratio

### Interpreting results
- Train Sharpe > 2x Valid Sharpe indicates overfitting — reduce capacity or increase regularization
- `score = valid_sharpe` is the primary selection metric — test_sharpe is recorded for reporting only
- Explained Variation (EV) ~3-5% is typical for a 1-factor model
- Validation set is only 60 months — changes < 0.01 in valid_sharpe may be noise
