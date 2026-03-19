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

**The goal is simple: get the highest valid_sharpe.** The primary metric is the monthly Sharpe ratio of the SDF portfolio on the validation set. Higher is better. Everything is fair game: change the architecture, the optimizer, the hyperparameters, the training schedule. The only constraint is that the code runs without crashing.

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

The TSV has a header row and 6 columns:

```
commit	valid_sharpe	test_sharpe	valid_ev	status	description
```

1. git commit hash (short, 7 chars)
2. valid_sharpe achieved — use 0.000000 for crashes
3. test_sharpe achieved — use 0.000000 for crashes
4. valid_ev (explained variation) — use 0.000000 for crashes
5. status: `keep`, `discard`, or `crash`
6. short text description of what this experiment tried

Example:

```
commit	valid_sharpe	test_sharpe	valid_ev	status	description
a1b2c3d	0.452300	0.389100	0.031200	keep	baseline
b2c3d4e	0.480100	0.410200	0.035000	keep	increase hidden dims to [128 64]
c3d4e5f	0.420000	0.350000	0.028000	discard	switch to GRU
d4e5f6g	0.000000	0.000000	0.000000	crash	attention model (shape mismatch)
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/mar9`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Modify `train.py` with an experimental idea by directly hacking the code.
3. git commit
4. Run the experiment: `python train.py > run.log 2>&1` (redirect everything)
5. Read out the results: `grep "^valid_sharpe:\|^test_sharpe:\|^valid_ev:" run.log`
6. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the stack trace and attempt a fix. If you can't fix it after a few attempts, give up on that idea.
7. Record the results in the TSV (do NOT commit results.tsv — leave it untracked)
8. If valid_sharpe improved (higher), you "advance" the branch, keeping the git commit
9. If valid_sharpe is equal or worse, you `git reset --hard HEAD~1` back to where you started

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

### Things that tend to work
- Dropout is critical (0.05 drop prob = paper default)
- LSTM for macro features captures business cycle dynamics
- L1 penalty encourages sparse portfolios (economically interpretable)
- The 3-phase training schedule matters — don't skip phases
- Validation Sharpe is noisy with only 60 months — changes < 0.01 may be noise

### Research directions (roughly ordered by expected impact)
1. Hidden layer architecture: width, depth, skip connections
2. Regularization: dropout, L1, L2, spectral norm, weight clipping
3. Activation functions: ReLU, ELU, GELU, SiLU
4. Optimizer: AdamW vs Adam, learning rate schedules, warmup/cooldown
5. Moment layer: more conditions (K>8), deeper adversary
6. RNN variant: GRU vs LSTM, hidden size, bidirectional
7. Training schedule: more/fewer epochs per phase, sub_epoch count
8. Novel: attention over stocks, feature cross-interactions
9. Novel: mixture of SDFs, multi-factor models
10. Novel: curriculum learning, learning rate scheduling

### Things to AVOID
- Removing the 3-phase structure entirely
- Very large models that overfit 240 months of training data
- Training for too many epochs without early stopping
- Changing the evaluation metric or data preprocessing

### Interpreting results
- Train Sharpe >> Valid Sharpe indicates overfitting
- Test Sharpe is reported but should NOT guide model selection (look-ahead bias)
- Explained Variation (EV) ~3-5% is typical for a 1-factor model
