[ -z "${MASTER_PORT}" ] && MASTER_PORT=10087
[ -z "${MASTER_IP}" ] && MASTER_IP=127.0.0.1
[ -z "${n_gpu}" ] && n_gpu=$(nvidia-smi -L | wc -l)
[ -z "${OMPI_COMM_WORLD_SIZE}" ] && OMPI_COMM_WORLD_SIZE=1
[ -z "${OMPI_COMM_WORLD_RANK}" ] && OMPI_COMM_WORLD_RANK=0

[ -z "${arch}" ] && arch=uniprop_small
[ -z "${task}" ] && task=pcq
[ -z "${batch_size}" ] && batch_size=128
[ -z "${num_workers}" ] && num_workers=8

# Directory containing {subset}.lmdb
[ -z "${data_path}" ] && data_path="./conformations/xtb_to_dft_vacuum/"
# Path to model weights (override by exporting weight_path before running this script)
[ -z "${weight_path}" ] && weight_path="../results/checkpoints_unimol/uniprop_xtb_to_vacuum.pt"
# Output directory (override by exporting results_path before running this script)
[ -z "${results_path}" ] && results_path="../results/checkpoints_unimol/inference_xtb_to_dft_vacuum"
mkdir -p $results_path

# Check if multitarget is set
if [ -n "${multitarget}" ]; then
    multitarget_arg="--multitarget"
else
    multitarget_arg=""
fi

# Match validate.py defaults: EMA on (when checkpoint has it), FP32 by default.
# Enable toggles only when explicitly set to 1/true/yes.
case "${no_ema}" in
  1|true|TRUE|yes|YES) ema_arg="--no-ema" ;;
  *) ema_arg="" ;;
esac

case "${fp16}" in
  1|true|TRUE|yes|YES) fp16_arg="--fp16" ;;
  *) fp16_arg="" ;;
esac

# Pick launcher: torchrun (preferred) or python -m torch.distributed.run (fallback)
if command -v torchrun >/dev/null 2>&1; then
    launcher="torchrun"
else
    launcher="python -m torch.distributed.run"
fi

$launcher --nproc_per_node=$n_gpu --nnodes=$OMPI_COMM_WORLD_SIZE  --node_rank=$OMPI_COMM_WORLD_RANK  --master_addr=$MASTER_IP --master_port=$MASTER_PORT \
       inference.py --user-dir ./unimol_plus/ $data_path --valid-subset $1 \
       --results-path $results_path \
       --num-workers $num_workers --ddp-backend=c10d --batch-size $batch_size \
       --task $task --loss unimol_plus --arch $arch \
       --path $weight_path \
       $fp16_arg --fp16-init-scale 4 --fp16-scale-window 256 \
       --log-interval 50 --log-format simple --label-prob 0.0 $multitarget_arg \
       $ema_arg