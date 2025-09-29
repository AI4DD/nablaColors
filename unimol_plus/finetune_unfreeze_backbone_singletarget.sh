#!/bin/sh

MASTER_PORT=10015
MASTER_IP=127.0.0.1
n_gpu=1

exp_name=singletarget
run_name=bs_64_unfreeze_backbone

OMPI_COMM_WORLD_SIZE=1
OMPI_COMM_WORLD_RANK=0
data_path="/home/jovyan/potapov/nablaColors/conformations/xtb_to_dft_implicit/"

user_dir="/home/jovyan/potapov/nablaColors/nablaColors/unimol_plus/"
train_set="train"
valid_sets="valid,test"
chemprop_pretrain="/home/jovyan/potapov/nablaColors/nablaColors/models/chemprop/fold_0/model_1/model.pt"

# Defaults (can be overridden by CLI)
pretrained_model="/home/jovyan/potapov/nablaColors/results/checkpoints_unimol/checkpoint_best_expsingletarget_runbs_64_head_pretrain.pt"

batch_size=4
batch_size_valid=4
# Parse CLI flags (required: --lr)
# Parse CLI flags (required: --lr)
lr=5e-4
while [ $# -gt 0 ]; do
    case "$1" in
        --lr)
            lr="$2"; shift 2 ;;
        --data-path)
            data_path="$2"; shift 2 ;;
        --pretrained-model)
            pretrained_model="$2"; shift 2 ;;
        *)
            echo "Unknown argument: $1"; exit 1 ;;
    esac
done

end_lr=1e-9
run_name=${run_name}_lr${lr}
save_dir="/home/jovyan/potapov/nablaColors/results/checkpoints_unimol/exp_${exp_name}/run_${run_name}/"

warmup_steps=10000
total_steps=100000
update_freq=16
seed=1
clip_norm=5
weight_decay=0.0
pos_loss_weight=0.3
dist_loss_weight=1.5
min_pos_loss_weight=0.06
min_dist_loss_weight=0.3

# Target-specific loss weights (placeholders for single task)
abs_loss_weight=1.0
emm_loss_weight=1.0
plqy_loss_weight=10.0

noise=0.2
label_prob=0.8
mid_prob=0.1
mid_lower=0.4
mid_upper=0.6
ema_decay=0.999

log_interval=100
save_interval_updates=1000
validate_interval_updates=500
validate_interval=5
keep_best_checkpoints=3

arch="uniprop_small"

export NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=1

# Parse CLI flags
while [ $# -gt 0 ]; do
    case "$1" in
        --data-path)
            data_path="$2"; shift 2 ;;
        --pretrained-model)
            pretrained_model="$2"; shift 2 ;;
        *)
            echo "Unknown argument: $1"; exit 1 ;;
    esac
done

more_args="--finetune-from-model $pretrained_model
--checkpoint-suffix _exp${exp_name}_run${run_name} --wandb-project UniMol --wandb-name finetune_all_exp${exp_name}_run${run_name} --load-from-ema "

more_args=$more_args" --ema-decay $ema_decay --validate-with-ema"
save_dir=$save_dir"-ema"$ema_decay


mkdir -p $save_dir

export NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=1

echo "torchrun --nproc_per_node=1 --nnodes=1  --node_rank=$OMPI_COMM_WORLD_RANK  --master_addr=$MASTER_IP --master_port=$MASTER_PORT \
      /home/user/.local/bin/unicore-train $data_path --user-dir $user_dir --train-subset $train_set --valid-subset $valid_sets \
      --num-workers 8 --ddp-backend=c10d \
      --task pcq --loss unimol_plus --arch $arch --chemprop-weight-path $chemprop_pretrain  \
      --fp16 False --fp16-init-scale 4 --fp16-scale-window 256 --tensorboard-logdir $save_dir/tsb \
      --log-interval $log_interval --log-format simple \
      --save-interval-updates $save_interval_updates --validate-interval-updates $validate_interval_updates --keep-interval-updates 50 --no-epoch-checkpoints  \
      --save-dir $save_dir --validate-interval $validate_interval --keep-best-checkpoints $keep_best_checkpoints \
      --batch-size $batch_size \
      --data-buffer-size 32 --fixed-validation-seed 11 --batch-size-valid $batch_size_valid \
      --optimizer adam --adam-betas '(0.9, 0.999)' --adam-eps 1e-8 --clip-norm $clip_norm \
      --lr $lr --end-learning-rate $end_lr --lr-scheduler polynomial_decay --power 1 \
      --warmup-updates $warmup_steps --total-num-update $total_steps --max-update $total_steps --update-freq $update_freq \
      --weight-decay $weight_decay \
      --dist-loss-weight $dist_loss_weight --pos-loss-weight $pos_loss_weight \
      --min-dist-loss-weight $min_dist_loss_weight --min-pos-loss-weight $min_pos_loss_weight \
      --label-prob $label_prob --noise-scale $noise  \
      --mid-prob $mid_prob --mid-lower $mid_lower --mid-upper $mid_upper --seed $seed $more_args " 

torchrun --nproc_per_node=$n_gpu --nnodes=$OMPI_COMM_WORLD_SIZE  --node_rank=$OMPI_COMM_WORLD_RANK  --master_addr=$MASTER_IP --master_port=$MASTER_PORT \
      /home/user/.local/bin/unicore-train $data_path --user-dir $user_dir --train-subset $train_set --valid-subset $valid_sets \
      --num-workers 4 --ddp-backend=c10d \
      --task pcq --loss unimol_plus --arch $arch  --chemprop-weight-path $chemprop_pretrain \
      --fp16-init-scale 4 --fp16-scale-window 256 --tensorboard-logdir $save_dir/tsb \
      --log-interval $log_interval --log-format simple \
      --save-interval-updates $save_interval_updates --validate-interval-updates $validate_interval_updates --keep-interval-updates 50 --no-epoch-checkpoints  \
      --save-dir $save_dir --validate-interval $validate_interval --keep-best-checkpoints $keep_best_checkpoints \
      --batch-size $batch_size \
      --data-buffer-size 32 --fixed-validation-seed 11 --batch-size-valid $batch_size_valid \
      --optimizer adam --adam-betas '(0.9, 0.999)' --adam-eps 1e-8 --clip-norm $clip_norm \
      --lr $lr --end-learning-rate $end_lr --lr-scheduler polynomial_decay --power 1 \
      --warmup-updates $warmup_steps --total-num-update $total_steps --max-update $total_steps --update-freq $update_freq \
      --weight-decay $weight_decay \
      --dist-loss-weight $dist_loss_weight --pos-loss-weight $pos_loss_weight \
      --min-dist-loss-weight $min_dist_loss_weight --min-pos-loss-weight $min_pos_loss_weight \
      --label-prob $label_prob --noise-scale $noise  \
      --mid-prob $mid_prob --mid-lower $mid_lower --mid-upper $mid_upper --seed $seed $more_args



