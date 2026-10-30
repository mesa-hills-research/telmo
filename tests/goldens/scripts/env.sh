# Environment for the reference runs. Set DATA to the data folder first.
# Temporary files and caches stay inside DATA, the runs are offline, and torch uses 6 threads at most.
# RUN_LOCKED may name a wrapper that limits concurrent heavy jobs. It is empty by default.
: "${DATA:?set DATA to the data folder}"
export TMPDIR=$DATA/tmp
export HF_HOME=$DATA/tmp/hf
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export XDG_CACHE_HOME=$DATA/tmp/cache
export TORCH_HOME=$DATA/tmp/torch
export NUMBA_CACHE_DIR=$DATA/tmp/numba
export MPLCONFIGDIR=$DATA/tmp/mpl
export WANDB_MODE=disabled
export OMP_NUM_THREADS=6
export MKL_NUM_THREADS=6
mkdir -p "$TMPDIR"
PY="$DATA/venv/bin/python -I"
RL=${RUN_LOCKED:-}
