import torch
import numpy as np
import pandas as pd
from unicore import metrics
from unicore.losses import UnicoreLoss, register_loss


@register_loss("unimol_plus")
class UnimolPlusLoss(UnicoreLoss):
    """
    Implementation for the loss used in masked graph model (MGM) training.
    """

    @staticmethod
    def add_args(parser):
        parser.add_argument(
            "--multitarget",
            action="store_true",
            help="Enable multitarget training (dataset provides multiple targets)",
        )

    def __init__(self, task):
        super().__init__(task)
        # self.e_thresh = 0.02
        self.args = task.args

        def get_loss_weight(max_loss_weight, min_loss_weight):
            weight_range = max(0, max_loss_weight - min_loss_weight)
            return max_loss_weight, weight_range

        self.pos_loss_weight, self.pos_loss_weight_range = get_loss_weight(
            self.args.pos_loss_weight, self.args.min_pos_loss_weight
        )
        self.dist_loss_weight, self.dist_loss_weight_range = get_loss_weight(
            self.args.dist_loss_weight, self.args.min_dist_loss_weight
        )
        if self.args.multitarget:
            self.abs_loss_weight, self.emm_loss_weight, self.plqy_loss_weight = (
                self.args.abs_loss_weight,
                self.args.emm_loss_weight,
                self.args.plqy_loss_weight,
            )

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """

        with torch.no_grad():
            sample_size = sample["batched_data"]["atom_mask"].shape[0]
            natoms = sample["batched_data"]["atom_mask"].shape[1]

        (
            graph_output,
            pos_pred,
            pos_target_mask,
            dist_pred,
            update_num,
        ) = model(**sample)
        if self.training:
            max_update = self.args.max_update
            assert update_num >= 0 and max_update >= 0
            ratio = float(update_num) / max_update
            delta = self.pos_loss_weight_range * ratio
            pos_loss_weight = self.pos_loss_weight - delta
            delta = self.dist_loss_weight_range * ratio
            dist_loss_weight = self.dist_loss_weight - delta
        else:
            pos_loss_weight = self.pos_loss_weight
            dist_loss_weight = self.dist_loss_weight
        if self.args.multitarget:
            targets = sample["batched_data"]["target"].float().view(-1, 3)
            def get_plqy_logit_transform(targets):
                targets_mod = targets.clone()
                plqy_target = targets[:, 2]
                plqy_target = torch.clamp(plqy_target, min=3e-5, max=0.999 + 1e-5)
                plqy_target = torch.log(plqy_target / (1 - plqy_target))
                targets_mod[:, 2] = plqy_target
                return targets_mod

            targets = get_plqy_logit_transform(targets)
            NaN_target_mask = torch.isnan(targets)
            targets[NaN_target_mask] = 0.0
            per_data_loss = None
            if graph_output is not None:
                graph_output = graph_output.float().view(-1, 3)

                # Calculate L1 loss for each target separately
                per_data_loss = torch.nn.L1Loss(reduction="none")(
                    graph_output.float(), targets_mod
                ) * (1 - NaN_target_mask.float())  # mask out NaN targets

                # Apply different weights to each target column
                # Column 0: absorption, Column 1: emission, Column 2: PLQY
                target_weights = torch.tensor(
                    [self.abs_loss_weight, self.emm_loss_weight, self.plqy_loss_weight],
                    device=per_data_loss.device,
                    dtype=per_data_loss.dtype,
                )

                # # Average loss over non-NaN targets
                # abs_sample_size, ems_sample_size, plqy_sample_size = (
                #     1 - NaN_target_mask.float()
                # ).sum(dim=0)
                # Sum only over available (non-NaN) labels per column
                valid_abs = (~NaN_target_mask[:, 0]).float()
                valid_ems = (~NaN_target_mask[:, 1]).float()
                valid_plq = (~NaN_target_mask[:, 2]).float()
                loss_abs = (per_data_loss[:, 0] * valid_abs).sum()
                loss_ems = (per_data_loss[:, 1] * valid_ems).sum()
                loss_plqy = (per_data_loss[:, 2] * valid_plq).sum()

                # Apply weights to each column
                weighted_loss = per_data_loss * target_weights.unsqueeze(0)

                # energy_within_threshold = (per_data_loss < self.e_thresh).sum()
                loss = weighted_loss.sum()  # Is this correct?
        else:
            targets = sample["batched_data"]["target"].float().view(-1)
            per_data_loss = None
            if graph_output is not None:
                graph_output = graph_output.float().view(-1)
                per_data_loss = torch.nn.L1Loss(reduction="none")(
                    graph_output.float(), targets
                )
                # energy_within_threshold = (per_data_loss < self.e_thresh).sum()
                loss = per_data_loss.sum()
                
        per_data_pred = graph_output
        per_data_label = targets
        if per_data_loss is None:
            loss = torch.tensor(0.0, device=targets.device)

        atom_mask = sample["batched_data"]["atom_mask"].float()
        if pos_target_mask is not None:
            atom_mask = atom_mask * pos_target_mask.float()
        pos_target = sample["batched_data"]["pos_target"].float() * atom_mask.unsqueeze(
            -1
        )

        def get_pos_loss(pos_pred):
            pos_pred = pos_pred.float() * atom_mask.unsqueeze(-1)
            pos_loss = torch.nn.L1Loss(reduction="none")(
                pos_pred,
                pos_target,
            ).sum(dim=(-1, -2))
            pos_cnt = atom_mask.sum(dim=-1) + 1e-10
            pos_loss = (pos_loss / pos_cnt).sum()
            return pos_loss

        pos_loss = get_pos_loss(pos_pred)

        pair_mask = atom_mask.unsqueeze(-1) * atom_mask.unsqueeze(-2).float()
        dist_target = (pos_target.unsqueeze(-2) - pos_target.unsqueeze(-3)).norm(dim=-1)
        dist_target = dist_target * pair_mask
        dist_cnt = pair_mask.sum(dim=(-1, -2)) + 1e-10

        def get_dist_loss(dist_pred, return_sum=True):
            dist_pred = dist_pred.float() * pair_mask
            dist_loss = torch.nn.L1Loss(reduction="none")(
                dist_pred,
                dist_target,
            ).sum(dim=(-1, -2))
            if return_sum:
                return (dist_loss / dist_cnt).sum()
            else:
                return dist_loss / dist_cnt

        dist_loss = get_dist_loss(dist_pred)

        total_loss = loss + dist_loss_weight * dist_loss + pos_loss_weight * pos_loss
        logging_output = {
            "loss": loss.item(),
            # "ewt_metric": energy_within_threshold,
            "dist_loss": dist_loss.item(),
            "pos_loss": pos_loss.item(),
            "total_loss": total_loss.item(),
            "sample_size": sample_size,
            "nsentences": sample_size,
            "ntokens": natoms,
            "bsz": sample_size,
            "n_atoms": natoms * sample_size,
        }
        if self.args.multitarget:
            logging_output["loss_abs"] = loss_abs.item()
            logging_output["loss_ems"] = loss_ems.item()
            logging_output["loss_plqy"] = loss_plqy.item()
        if not torch.is_grad_enabled():
            logging_output["id"] = sample["batched_data"]["id"].cpu().numpy()
            logging_output["pred"] = per_data_pred.detach().cpu().numpy()
            logging_output["label"] = per_data_label.detach().cpu().numpy()
        # Removed redundant tensor assignment to prevent accumulating GPU tensors in logging dict
        logging_output["total_loss"] = total_loss.item()
        return total_loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs, split="valid") -> None:
        """Aggregate logging outputs from data parallel training."""
        if split != "train":
            id = np.concatenate([log["id"] for log in logging_outputs])
            pred = np.concatenate([log["pred"] for log in logging_outputs])
            label = np.concatenate([log["label"] for log in logging_outputs])
            if self.args.multitarget:
                def sigmoid(x):
                    return 1 / (1 + np.exp(-x))

                pred_abs = pred[:, 0]
                pred_ems = pred[:, 1]
                pred_plqy = sigmoid(pred[:, 2])
                label_abs = label[:, 0]
                label_ems = label[:, 1]
                label_plqy = label[:, 2]
                res_dict = {
                    "pred_abs": pred_abs,
                    "pred_ems": pred_ems,
                    "pred_plqy": pred_plqy,
                    "label_abs": label_abs,
                    "label_ems": label_ems,
                    "label_plqy": label_plqy,
                }
            else:
                res_dict = {
                    "pred": pred,
                    "label": label,
                }
            df = pd.DataFrame(res_dict)
            df_grouped = df.groupby(["id"])
            df_mean = df_grouped.agg("mean")
            df_median = df_grouped.agg("median")
            if self.args.multitarget:
                def get_mae_losses(df):
                    abs_non_nan_mask = ~df["label_abs"].isna()
                    ems_non_nan_mask = ~df["label_ems"].isna()
                    plqy_non_nan_mask = ~df["label_plqy"].isna()
                    abs_loss = np.abs(
                        df["pred_abs"][abs_non_nan_mask] - df["label_abs"][abs_non_nan_mask]
                    ).mean()
                    ems_loss = np.abs(
                        df["pred_ems"][ems_non_nan_mask] - df["label_ems"][ems_non_nan_mask]
                    ).mean()
                    plqy_loss = np.abs(
                        df["pred_plqy"][plqy_non_nan_mask]
                        - df["label_plqy"][plqy_non_nan_mask]
                    ).mean()
                    return abs_loss, ems_loss, plqy_loss

                def get_mae_log_plqy(df):
                    plqy_non_nan_mask = ~df["label_plqy"].isna()
                    log_plqy_loss = np.abs(
                        np.log(df["pred_plqy"][plqy_non_nan_mask] + 1e-5)
                        - np.log(df["label_plqy"][plqy_non_nan_mask] + 1e-5)
                    ).mean()
                    return log_plqy_loss

                # Compute per-target metrics only if that target has any non-NaN labels
                abs_has = (~df_mean["label_abs"].isna()).sum() > 0
                ems_has = (~df_mean["label_ems"].isna()).sum() > 0
                plq_has = (~df_mean["label_plqy"].isna()).sum() > 0

                abs_loss_by_mean, ems_loss_by_mean, plqy_loss_by_mean = get_mae_losses(df_mean)
                abs_loss_by_median, ems_loss_by_median, plqy_loss_by_median = get_mae_losses(df_median)
                log_plqy_loss_by_mean = get_mae_log_plqy(df_mean)
                log_plqy_loss_by_median = get_mae_log_plqy(df_median)

                if abs_has:
                    metrics.log_scalar("abs_MAE_by_mean", abs_loss_by_mean, 1, round=6)
                    metrics.log_scalar("abs_MAE_by_median", abs_loss_by_median, 1, round=6)
                if ems_has:
                    metrics.log_scalar("ems_MAE_by_mean", ems_loss_by_mean, 1, round=6)
                    metrics.log_scalar("ems_MAE_by_median", ems_loss_by_median, 1, round=6)
                if plq_has:
                    metrics.log_scalar("plqy_MAE_by_mean", plqy_loss_by_mean, 1, round=6)
                    metrics.log_scalar("plqy_MAE_by_median", plqy_loss_by_median, 1, round=6)
                    metrics.log_scalar(
                        "plqy_log_MAE_by_mean", log_plqy_loss_by_mean, 1, round=6
                    )
                    metrics.log_scalar(
                        "plqy_log_MAE_by_median", log_plqy_loss_by_median, 1, round=6
                    )
            else:

                def get_mae_loss(df):
                    return np.abs(df["pred"] - df["label"]).mean()

                metrics.log_scalar("loss_by_mean", get_mae_loss(df_mean), 1, round=6)
                metrics.log_scalar("loss_by_median", get_mae_loss(df_median), 1, round=6)


        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        n_atoms = sum(log.get("n_atoms", 0) for log in logging_outputs)
        for key in logging_outputs[0].keys():
            if "loss" in key or "metric" in key:
                total_loss_sum = sum(log.get(key, 0) for log in logging_outputs)
                metrics.log_scalar(
                    key, total_loss_sum / sample_size, sample_size, round=6
                )
        metrics.log_scalar("n_atoms", n_atoms / sample_size, sample_size, round=6)

    @staticmethod
    def logging_outputs_can_be_summed(is_train) -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return is_train
    