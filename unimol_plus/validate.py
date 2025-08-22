#!/usr/bin/env python3 -u
import os
import sys
import json
import pickle
import logging
from typing import Dict, Any, List, Tuple

import numpy as np
import torch

from unicore import checkpoint_utils, distributed_utils, options
from unicore import tasks
from unicore.logging import progress_bar

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("unimol_plus.validate")


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def compute_metrics(pred: np.ndarray, label: np.ndarray, ids: np.ndarray) -> Dict[str, float]:
    pred_abs = pred[:, 0]
    pred_ems = pred[:, 1]
    pred_plqy = _sigmoid(pred[:, 2])
    label_abs = label[:, 0]
    label_ems = label[:, 1]
    label_plqy = label[:, 2]

    import pandas as pd

    df = pd.DataFrame(
        {
            "id": ids,
            "pred_abs": pred_abs,
            "pred_ems": pred_ems,
            "pred_plqy": pred_plqy,
            "label_abs": label_abs,
            "label_ems": label_ems,
            "label_plqy": label_plqy,
        }
    )
    df_grouped = df.groupby(["id"])
    df_mean = df_grouped.agg("mean")
    df_median = df_grouped.agg("median")

    def get_mae_losses(df_):
        abs_non_nan = ~df_["label_abs"].isna()
        ems_non_nan = ~df_["label_ems"].isna()
        plqy_non_nan = ~df_["label_plqy"].isna()
        abs_loss = np.abs(df_["pred_abs"][abs_non_nan] - df_["label_abs"][abs_non_nan]).mean()
        ems_loss = np.abs(df_["pred_ems"][ems_non_nan] - df_["label_ems"][ems_non_nan]).mean()
        plqy_loss = np.abs(df_["pred_plqy"][plqy_non_nan] - df_["label_plqy"][plqy_non_nan]).mean()
        return abs_loss, ems_loss, plqy_loss

    def get_mae_log_plqy(df_):
        plqy_non_nan = ~df_["label_plqy"].isna()
        return np.abs(np.log(df_["pred_plqy"][plqy_non_nan] + 1e-5) - np.log(df_["label_plqy"][plqy_non_nan] + 1e-5)).mean()

    abs_mean, ems_mean, plqy_mean = get_mae_losses(df_mean)
    abs_med, ems_med, plqy_med = get_mae_losses(df_median)
    plqy_log_mean = get_mae_log_plqy(df_mean)
    plqy_log_med = get_mae_log_plqy(df_median)

    return {
        "abs_MAE_by_mean": float(abs_mean),
        "ems_MAE_by_mean": float(ems_mean),
        "plqy_MAE_by_mean": float(plqy_mean),
        "abs_MAE_by_median": float(abs_med),
        "ems_MAE_by_median": float(ems_med),
        "plqy_MAE_by_median": float(plqy_med),
        "plqy_log_MAE_by_mean": float(plqy_log_mean),
        "plqy_log_MAE_by_median": float(plqy_log_med),
    }


def _barrier_small():
    if distributed_utils.get_data_parallel_world_size() > 1:
        _ = distributed_utils.all_gather_list(
            [torch.tensor(0)],
            max_size=10000,
            group=distributed_utils.get_data_parallel_group(),
        )


def run_validate(args) -> Dict[str, Any]:
    use_cuda = torch.cuda.is_available() and not args.cpu
    if use_cuda:
        torch.cuda.set_device(args.device_id)

    # Setup distributed
    if args.distributed_world_size > 1:
        data_parallel_world_size = distributed_utils.get_data_parallel_world_size()
        data_parallel_rank = distributed_utils.get_data_parallel_rank()
    else:
        data_parallel_world_size = 1
        data_parallel_rank = 0

    # Load model
    logger.info(f"Loading EMA weights from: {args.path}")
    state = checkpoint_utils.load_checkpoint_to_cpu(args.path)
    task = tasks.setup_task(args)
    model = task.build_model(args)
    model.load_state_dict(state["ema"]["params"], strict=True)
    if use_cuda:
        model.cuda()
    model.eval()

    all_ids: List[np.ndarray] = []
    all_preds: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []

    # Only support a single split name passed in via --valid-subset (like "valid" or "test")
    assert "," not in args.valid_subset, "Provide a single subset for validation"
    subset = args.valid_subset
    task.load_dataset(subset, combine=False, epoch=1, force_valid=True)
    dataset = task.dataset(subset)
    itr = task.get_batch_iterator(
        dataset=dataset,
        batch_size=args.batch_size,
        ignore_invalid_inputs=True,
        required_batch_size_multiple=args.required_batch_size_multiple,
        seed=args.seed,
        num_shards=data_parallel_world_size,
        shard_id=data_parallel_rank,
        num_workers=args.num_workers,
        data_buffer_size=args.data_buffer_size,
    ).next_epoch_itr(shuffle=False)

    progress = progress_bar.progress_bar(
        itr,
        log_format=args.log_format,
        log_interval=args.log_interval,
        prefix=f"valid on '{subset}' subset",
        default_log_format=("tqdm" if not args.no_progress_bar else "simple"),
    )

    with torch.no_grad():
        for i, sample in enumerate(progress):
            if use_cuda:
                from unicore import utils as u
                sample = u.move_to_cuda(sample)
            if len(sample) == 0:
                continue
            graph_output, _pos_pred = model(**sample)[:2]
            ids = sample["batched_data"]["id"].cpu().numpy()
            preds = graph_output.float().view(-1, 3).cpu().numpy()
            labels = sample["batched_data"]["target"].float().view(-1, 3).cpu().numpy()
            all_ids.append(ids)
            all_preds.append(preds)
            all_labels.append(labels)
            progress.log({}, step=i)

    # Stack local
    ids_np = np.concatenate(all_ids) if len(all_ids) else np.empty((0,), dtype=np.int64)
    preds_np = np.concatenate(all_preds) if len(all_preds) else np.empty((0, 3), dtype=np.float32)
    labels_np = np.concatenate(all_labels) if len(all_labels) else np.empty((0, 3), dtype=np.float32)

    # Save per-rank temporary outputs
    os.makedirs(args.results_path, exist_ok=True)
    rank = distributed_utils.get_data_parallel_rank()
    tmp_path = os.path.join(args.results_path, f"{subset}_{rank}.pkl")
    with open(tmp_path, "wb") as f:
        pickle.dump((ids_np, preds_np, labels_np), f)
    logger.info(f"Saved rank {rank} outputs to {tmp_path}")

    # Barrier to ensure all ranks saved
    _barrier_small()

    # Only rank 0 merges and computes metrics
    if rank == 0:
        # find all per-rank files
        files = [
            os.path.join(args.results_path, name)
            for name in os.listdir(args.results_path)
            if name.startswith(f"{subset}_") and name.endswith(".pkl")
        ]
        files.sort()
        merged_ids = []
        merged_preds = []
        merged_labels = []
        for fp in files:
            with open(fp, "rb") as f:
                i_, p_, l_ = pickle.load(f)
            merged_ids.append(i_)
            merged_preds.append(p_)
            merged_labels.append(l_)
        if merged_ids:
            ids_all = np.concatenate(merged_ids)
            preds_all = np.concatenate(merged_preds)
            labels_all = np.concatenate(merged_labels)
        else:
            ids_all = np.empty((0,), dtype=np.int64)
            preds_all = np.empty((0, 3), dtype=np.float32)
            labels_all = np.empty((0, 3), dtype=np.float32)

        metrics_dict = compute_metrics(preds_all, labels_all, ids_all)
        out_base = os.path.join(args.results_path, f"{subset}")
        with open(out_base + ".metrics.json", "w") as f:
            json.dump(metrics_dict, f, indent=2)
        with open(out_base + ".preds.pkl", "wb") as f:
            pickle.dump({"id": ids_all, "pred": preds_all, "label": labels_all}, f)
        logger.info("Validation metrics: " + json.dumps(metrics_dict))

        # Pretty table to stdout
        order = [
            "abs_MAE_by_mean",
            "ems_MAE_by_mean",
            "plqy_MAE_by_mean",
            "abs_MAE_by_median",
            "ems_MAE_by_median",
            "plqy_MAE_by_median",
            "plqy_log_MAE_by_mean",
            "plqy_log_MAE_by_median",
        ]
        w = max(len("Metric"), max(len(k) for k in order if k in metrics_dict))
        print()
        print(f"Validation metrics (subset: {subset})")
        print(f"{'Metric'.ljust(w)}  Value")
        print(f"{'-'*w}  {'-'*12}")
        for k in order:
            if k in metrics_dict:
                v = metrics_dict[k]
                if isinstance(v, float):
                    v_str = f"{v:.6f}"
                else:
                    v_str = str(v)
                print(f"{k.ljust(w)}  {v_str}")
        print()

    # Final barrier so ranks don't exit before rank 0 writes metrics
    _barrier_small()

    return {}


def cli_main():
    parser = options.get_validation_parser()
    options.add_model_args(parser)
    args = options.parse_args_and_arch(parser)
    distributed_utils.call_main(args, run_validate)


if __name__ == "__main__":
    cli_main()


