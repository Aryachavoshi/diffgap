#!/bin/bash
# ---------------------------------------------------------
# Multicity one_config_inference sweep via PyLauncher on LS6
# ---------------------------------------------------------

#SBATCH -J multicity_hps_pyl
#SBATCH -o multicity_hps_pyl.%j.out
#SBATCH -e multicity_hps_pyl.%j.err
#SBATCH -p gpu-a100
#SBATCH -N 8                        # 8 nodes
#SBATCH --ntasks-per-node=3         # 3 tasks per node (one per GPU)
#SBATCH --exclusive
#SBATCH -t 15:00:00                 # adjust walltime as needed
#SBATCH -A ATM23014

set -euo pipefail

# ---- Environment ----
module purge
module load intel/19.1.1
module load python3/3.9.7

# Activate your venv (has torch, accelerate, etc.)
source /work/09461/arya_chavoshi/ls6/myenvs/torch_env/bin/activate

# Ensure paramiko is present (PyLauncher sometimes needs it)
python - <<'PY'
import importlib, sys, subprocess
try:
    importlib.import_module("paramiko")
    print("paramiko present")
except Exception:
    print("Installing paramiko into current venv...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "--upgrade", "pip"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "paramiko==3.4.0"])
PY

module load pylauncher

# ---- CUDA / threading niceties ----
export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128

# ---- Paths ----
WORK=/work/09461/arya_chavoshi/ls6
cd $WORK/Multicity_inference

DATA_PATH="$WORK/Multicity_inference/Multicity_dict.joblib"
CKPT_PATH="$WORK/Multicity_inference/unet_epoch050.pth"

# Job-scoped base output dir
export OUTBASE=$WORK/Multicity_inference/metrics_sweep_${SLURM_JOB_ID}
mkdir -p "$OUTBASE"

# Unique work dir for PyLauncher
export PYL_WORKDIR=pylauncher_out_${SLURM_JOB_ID}
rm -rf "$PYL_WORKDIR"

# ---- Generate commands.txt for all hyperparam combos ----
python - <<'PY'
import itertools
import os

WORK = "/work/09461/arya_chavoshi/ls6"
data_path = f"{WORK}/Multicity_inference/Multicity_dict.joblib"
ckpt_path = f"{WORK}/Multicity_inference/unet_epoch050.pth"
outbase   = os.environ["OUTBASE"]

# Hyperparameter grids
cloud_coverages   = [0.30, 0.55, 0.85]       # 30%, 55%, 85%
octaves_list      = [2, 6, 10]
wind_degrees      = [135]
timesteps_list    = [80, 100, 120, 150,170]
grad_steps_list   = [1, 2, 3]
stride_list       = [1, 2, 3,4]                # guidance_stride

N_test     = 500
batch_size = 50
N_samples  = 10
seed       = 1337

cmd_file = "commands.txt"

os.makedirs(outbase, exist_ok=True)

with open(cmd_file, "w") as f:
    for (cloud, octaves, wind,
         timesteps, grad_steps, stride) in itertools.product(
            cloud_coverages,
            octaves_list,
            wind_degrees,
            timesteps_list,
            grad_steps_list,
            stride_list,
    ):
        # subdir name encodes all hyperparams
        cloud_pct = int(round(cloud * 100))
        subdir = (
            f"cloud{cloud_pct}_oct{octaves}_wind{wind}"
            f"_T{timesteps}_gs{grad_steps}_str{stride}"
        )
        outdir = os.path.join(outbase, subdir)
        os.makedirs(outdir, exist_ok=True)

        cmd = (
            f"{WORK}/myenvs/torch_env/bin/python -m accelerate.commands.launch "
            f"--num_processes 1 "
            f"--mixed_precision no "
            f"--dynamo_backend no "
            f"one_config_inference_hpsweep.py "
            f"--data_path \"{data_path}\" "
            f"--ckpt \"{ckpt_path}\" "
            f"--output_dir \"{outdir}\" "
            f"--seed {seed} "
            f"--N_test {N_test} "
            f"--cloud_coverage {cloud} "
            f"--octaves {octaves} "
            f"--batch_size {batch_size} "
            f"--N_samples {N_samples} "
            f"--wind_degree {wind} "
            f"--timesteps {timesteps} "
            f"--grad_steps {grad_steps} "
            f"--guidance_stride {stride}"
        )

        f.write(cmd + "\n")

print("Wrote commands.txt with all hyperparam combos.")
PY

echo "Preview of commands.txt:"
head -n 5 commands.txt
echo "Total commands: $(wc -l < commands.txt)"

# ---- Launch PyLauncher across allocated nodes/GPUs ----
python - <<'PY'
import os, pylauncher
print("PyLauncher module:", pylauncher.__file__)
print("Workdir:", os.environ.get("PYL_WORKDIR"))
# Each A100 node has 3 GPUs; run one command per GPU.
pylauncher.GPULauncher("commands.txt", gpuspernode=3, workdir=os.environ["PYL_WORKDIR"])
PY

echo "All PyLauncher tasks completed."
echo "Outputs stored under: $OUTBASE"
