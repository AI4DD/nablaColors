#!/bin/sh

MASTER_PORT=10011
MASTER_IP=127.0.0.1
n_gpu=2

exp_name=multitarget
run_name=bs_2_head_pretrain

OMPI_COMM_WORLD_SIZE=1
OMPI_COMM_WORLD_RANK=0
data_path="../xtb_to_dft_implicit/split_1"
save_dir="../results/checkpoints_unimol/exp_${exp_name}/run_${run_name}"
user_dir="./unimol_plus"
train_set="train"
valid_sets="valid"
chemprop_pretrain="../models/chemprop/fold_0/model_1/model.pt"

# Defaults (can be overridden by CLI)
pretrained_model="../unimol_plus_pcq_small.pt"

batch_size=2
batch_size_valid=2
lr=8e-5
end_lr=1e-9

warmup_steps=30000
total_steps=300000
update_freq=1
seed=1
clip_norm=5
weight_decay=0.0
pos_loss_weight=0.3
dist_loss_weight=1.5
min_pos_loss_weight=0.06
min_dist_loss_weight=0.3

# Target-specific loss weights
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
--checkpoint-suffix _exp${exp_name}_run${run_name} --wandb-project UniMol 
--wandb-name finetune_all_exp${exp_name}_run${run_name} --load-from-ema --head-pretrain --multitarget"

more_args=$more_args" --ema-decay $ema_decay --validate-with-ema"
save_dir=$save_dir"-ema"$ema_decay


mkdir -p $save_dir

export NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=1

echo "torchrun --nproc_per_node=$n_gpu --nnodes=$OMPI_COMM_WORLD_SIZE  --node_rank=$OMPI_COMM_WORLD_RANK  --master_addr=$MASTER_IP --master_port=$MASTER_PORT \
      $(which unicore-train) $data_path --user-dir $user_dir --train-subset $train_set --valid-subset $valid_sets \
      --num-workers 4 --ddp-backend=c10d \
      --task pcq --loss unimol_plus --arch $arch --chemprop-weight-path $chemprop_pretrain  \
      --fp16-init-scale 4 --fp16-scale-window 256 --tensorboard-logdir $save_dir/tsb \
      --log-interval $log_interval --log-format simple \
      --save-interval-updates $save_interval_updates --validate-interval-updates $validate_interval_updates --keep-interval-updates 50 --no-epoch-checkpoints  \
      --save-dir $save_dir --validate-interval $validate_interval \
      --batch-size $batch_size \
      --data-buffer-size 32 --fixed-validation-seed 11 --batch-size-valid $batch_size_valid \
      --optimizer adam --adam-betas '(0.9, 0.999)' --adam-eps 1e-8 --clip-norm $clip_norm \
      --lr $lr --end-learning-rate $end_lr --lr-scheduler polynomial_decay --power 1 \
      --warmup-updates $warmup_steps --total-num-update $total_steps --max-update $total_steps --update-freq $update_freq \
      --weight-decay $weight_decay \
      --dist-loss-weight $dist_loss_weight --pos-loss-weight $pos_loss_weight \
      --min-dist-loss-weight $min_dist_loss_weight --min-pos-loss-weight $min_pos_loss_weight \
      --abs-loss-weight $abs_loss_weight --emm-loss-weight $emm_loss_weight --plqy-loss-weight $plqy_loss_weight \
      --label-prob $label_prob --noise-scale $noise  \
      --mid-prob $mid_prob --mid-lower $mid_lower --mid-upper $mid_upper --seed $seed $more_args"

torchrun --nproc_per_node=$n_gpu --nnodes=$OMPI_COMM_WORLD_SIZE  --node_rank=$OMPI_COMM_WORLD_RANK  --master_addr=$MASTER_IP --master_port=$MASTER_PORT \
      $(which unicore-train) $data_path --user-dir $user_dir --train-subset $train_set --valid-subset $valid_sets \
      --num-workers 4 --ddp-backend=c10d \
      --task pcq --loss unimol_plus --arch $arch  --chemprop-weight-path $chemprop_pretrain \
      --fp16-init-scale 4 --fp16-scale-window 256 --tensorboard-logdir $save_dir/tsb \
      --log-interval $log_interval --log-format simple \
      --save-interval-updates $save_interval_updates --validate-interval-updates $validate_interval_updates --keep-interval-updates 50 --no-epoch-checkpoints  \
      --save-dir $save_dir --validate-interval $validate_interval \
      --batch-size $batch_size \
      --data-buffer-size 32 --fixed-validation-seed 11 --batch-size-valid $batch_size_valid \
      --optimizer adam --adam-betas '(0.9, 0.999)' --adam-eps 1e-8 --clip-norm $clip_norm \
      --lr $lr --end-learning-rate $end_lr --lr-scheduler polynomial_decay --power 1 \
      --warmup-updates $warmup_steps --total-num-update $total_steps --max-update $total_steps --update-freq $update_freq \
      --weight-decay $weight_decay \
      --dist-loss-weight $dist_loss_weight --pos-loss-weight $pos_loss_weight \
      --min-dist-loss-weight $min_dist_loss_weight --min-pos-loss-weight $min_pos_loss_weight \
      --abs-loss-weight $abs_loss_weight --emm-loss-weight $emm_loss_weight --plqy-loss-weight $plqy_loss_weight \
      --label-prob $label_prob --noise-scale $noise  \
      --mid-prob $mid_prob --mid-lower $mid_lower --mid-upper $mid_upper --seed $seed $more_args 
