#!/bin/sh

set -e

# Defaults (mirrors validate_multitarget.sh but for single-target absorption)
MASTER_PORT=10022
MASTER_IP=127.0.0.1
n_gpu=$(nvidia-smi -L | wc -l)
nnodes=1
node_rank=0

exp_name=singletarget
run_name=bs_2_unfreeze_backbone

data_path="../xtb_to_dft_implicit/split_1"
user_dir="./unimol_plus"
arch="uniprop_small"
subset="valid"

save_dir_set=0
results_path_set=0
weight_path_set=0

batch_size=16
num_workers=8

usage() {
  echo "Usage: $0 [--data-path PATH] [--weight-path FILE] [--subset NAME] [--results-path DIR] \
               [--arch NAME] [--user-dir DIR] [--exp-name NAME] [--run-name NAME] \
               [--nproc-per-node N] [--nnodes N] [--node-rank RANK] [--master-addr IP] [--master-port PORT] \
               [--batch-size N] [--num-workers N]"
  exit 1
}

while [ $# -gt 0 ]; do
  case "$1" in
    --data-path) data_path="$2"; shift 2;;
    --weight-path) weight_path="$2"; weight_path_set=1; shift 2;;
    --subset) subset="$2"; shift 2;;
    --results-path) results_path="$2"; results_path_set=1; shift 2;;
    --arch) arch="$2"; shift 2;;
    --user-dir) user_dir="$2"; shift 2;;
    --exp-name) exp_name="$2"; shift 2;;
    --run-name) run_name="$2"; shift 2;;
    --nproc-per-node) n_gpu="$2"; shift 2;;
    --nnodes) nnodes="$2"; shift 2;;
    --node-rank) node_rank="$2"; shift 2;;
    --master-addr) MASTER_IP="$2"; shift 2;;
    --master-port) MASTER_PORT="$2"; shift 2;;
    --batch-size) batch_size="$2"; shift 2;;
    --num-workers) num_workers="$2"; shift 2;;
    -h|--help) usage;;
    *) echo "Unknown argument: $1"; usage;;
  esac
done

# Derive save_dir if not provided explicitly
if [ $save_dir_set -eq 0 ]; then
  save_dir="../results/checkpoints_unimol/exp_${exp_name}/run_${run_name}-ema0.999"
fi

# Derive results_path if not provided explicitly
if [ $results_path_set -eq 0 ]; then
  results_path="${save_dir}/eval_${subset}"
fi
mkdir -p "$results_path"

# Derive weight_path if not provided explicitly
if [ $weight_path_set -eq 0 ]; then
  weight_path="${save_dir}/checkpoint_best_exp${exp_name}_run${run_name}.pt"
fi

export NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=1

echo "torchrun --nproc_per_node=$n_gpu --nnodes=$nnodes --node_rank=$node_rank --master_addr=$MASTER_IP --master_port=$MASTER_PORT \
      ./validate.py --user-dir $user_dir $data_path --valid-subset $subset \
      --results-path $results_path \
      --num-workers $num_workers --ddp-backend=c10d --batch-size $batch_size \
      --task pcq --loss unimol_plus --arch $arch \
      --path $weight_path \
      --fp16-init-scale 4 --fp16-scale-window 256 \
      --log-interval 50 --log-format simple --label-prob 0.0 --required-batch-size-multiple 1"

torchrun --nproc_per_node=$n_gpu --nnodes=$nnodes  --node_rank=$node_rank  --master_addr=$MASTER_IP --master_port=$MASTER_PORT \
      ./validate.py --user-dir $user_dir $data_path --valid-subset $subset \
      --results-path $results_path \
      --num-workers $num_workers --ddp-backend=c10d --batch-size $batch_size \
      --task pcq --loss unimol_plus --arch $arch \
      --path $weight_path \
      --fp16-init-scale 4 --fp16-scale-window 256 \
      --log-interval 50 --log-format simple --label-prob 0.0 --required-batch-size-multiple 1



