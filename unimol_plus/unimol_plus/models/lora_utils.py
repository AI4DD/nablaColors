import math
import torch
import torch.nn as nn


class Identity(nn.Module):
    def forward(self, x):
        return x


class LinearWithLoRA(nn.Module):
    """
    Wraps an existing nn.Linear-like module and adds a LoRA branch.

    The wrapped base module is stored as `base`, and its parameters are left as-is
    (you may freeze them externally). LoRA parameters are `lora_A` and `lora_B`.
    """

    def __init__(
        self,
        base_linear: nn.Linear,
        r: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if not isinstance(base_linear, nn.Linear):
            # support custom Linear subclass as it inherits from nn.Linear
            if not isinstance(base_linear, nn.modules.linear.Linear):
                raise TypeError("LinearWithLoRA expects an nn.Linear-compatible module")

        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features
        self.base = base_linear

        self.r = int(r)
        self.lora_alpha = float(lora_alpha)
        self.scaling = self.lora_alpha / max(self.r, 1)
        self.lora_dropout = nn.Dropout(p=lora_dropout) if lora_dropout and lora_dropout > 0 else Identity()

        if self.r > 0:
            # LoRA matrices
            self.lora_A = nn.Linear(self.in_features, self.r, bias=False)
            self.lora_B = nn.Linear(self.r, self.out_features, bias=False)

            # init: B to zeros, A with kaiming
            nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B.weight)
        else:
            self.lora_A = None
            self.lora_B = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.base(x)
        if self.r > 0 and self.lora_A is not None and self.lora_B is not None:
            lora_out = self.lora_B(self.lora_A(self.lora_dropout(x))) * self.scaling
            # ensure dtype/device alignment
            if lora_out.dtype != result.dtype:
                lora_out = lora_out.to(result.dtype)
            result = result + lora_out
        return result


def _should_wrap_module(module_name: str, target_names: set) -> bool:
    # exact match on leaf module attribute names
    return module_name in target_names if target_names else True


def inject_lora_adapters(
    root_module: nn.Module,
    target_module_names: list = None,
    r: int = 8,
    lora_alpha: float = 16.0,
    lora_dropout: float = 0.0,
    include_all_linear: bool = False,
    exclude_modules: list = None,
    exclude_name_substrings: list = None,
) -> None:
    """
    In-place replace target Linear modules with LinearWithLoRA wrappers.
    - If include_all_linear=True, wrap all nn.Linear under root_module (except excluded),
      otherwise wrap only leaf modules whose attribute name is in target_module_names.
    - exclude_modules: list of module objects to skip their subtrees entirely.
    - exclude_name_substrings: skip any child whose immediate name contains any of these substrings.
    """
    targets = set(target_module_names or [])
    exclude_modules = set(exclude_modules or [])
    exclude_name_substrings = list(exclude_name_substrings or [])

    def _recursive_replace(module: nn.Module, parent_excluded: bool = False):
        for name, child in list(module.named_children()):
            is_excluded_here = parent_excluded or (child in exclude_modules) or any(
                (s in name) for s in exclude_name_substrings
            )

            # Recurse into children first
            _recursive_replace(child, parent_excluded=is_excluded_here)

            if is_excluded_here:
                continue

            # Skip if already wrapped
            if isinstance(child, LinearWithLoRA):
                continue

            should_wrap = False
            if include_all_linear and isinstance(child, nn.Linear):
                should_wrap = True
            elif targets and _should_wrap_module(name, targets) and isinstance(child, nn.Linear):
                should_wrap = True

            if should_wrap:
                wrapped = LinearWithLoRA(child, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
                setattr(module, name, wrapped)

    _recursive_replace(root_module, parent_excluded=False)


def mark_only_lora_trainable(model: nn.Module, train_bias: bool = False) -> None:
    """
    Freeze all parameters except LoRA adapter weights (and optionally biases).
    """
    for p in model.parameters():
        p.requires_grad = False

    for m in model.modules():
        if isinstance(m, LinearWithLoRA):
            for p in m.lora_A.parameters():
                p.requires_grad = True
            for p in m.lora_B.parameters():
                p.requires_grad = True
            if train_bias and hasattr(m.base, "bias"):
                m.base.bias.requires_grad = True


