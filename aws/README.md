# AWS GPU Runner for SDF-GAN

Run validation and training on a spot g4dn.xlarge (T4 16GB) in sa-east-1.
Spot price: ~$0.12/hr.

## Prerequisites

- AWS CLI configured (`aws sts get-caller-identity` works)
- `ssh` and `scp` available in PATH

## Usage

### 1. Upload data to S3 (one-time)

```bash
cd aws
bash upload-data.sh
```

Syncs `datasets/` (2.3 GB) to an S3 bucket. Only needs to run once -- subsequent
instances pull data from S3 internally (fast, no re-upload).

### 2. Launch instance

```bash
bash launch.sh
```

Creates a key pair, security group, IAM role (for S3 access), and spot instance.
Writes instance details to `instance.env` (sourced by other scripts).

### 3. Upload code and pull data

```bash
bash upload.sh
```

SCPs code files to the instance, pulls datasets from S3 (within AWS, fast),
and installs PyTorch with CUDA.

### 4. Run a job

```bash
bash run-job.sh validate          # runs validate.py
bash run-job.sh train             # runs train.py
bash run-job.sh "python train.py" # arbitrary command
```

Output streams to your terminal. Log is also saved remotely as `~/sdfgan/run.log`.

### 5. Download results

```bash
bash download.sh
```

Copies `run.log` and any `*.log` files back to `aws/results/`.

### 6. Terminate instance

```bash
bash terminate.sh
```

Cancels the spot request and terminates the instance. **Always run this when done.**

## Cleanup

To remove all AWS resources (key pair, security group, IAM role, S3 bucket):

```bash
bash cleanup.sh            # removes everything including S3 data
bash cleanup.sh --keep-s3  # keeps the S3 bucket to avoid re-uploading
```

## Cost control

- Spot instance auto-terminates if price exceeds your bid (set to on-demand price)
- `terminate.sh` kills it manually
- S3 storage: ~$0.06/month for 2.3 GB (negligible)
- There is no auto-termination timer -- remember to terminate when done
