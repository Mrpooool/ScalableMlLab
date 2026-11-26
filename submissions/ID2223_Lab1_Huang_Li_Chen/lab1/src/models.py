"""From-scratch CIFAR-10 CNN architectures aligned with ID2223 Part A9.

No pretrained weights or torchvision.models constructors are used.
"""

from __future__ import annotations

from typing import Any, Sequence

import torch
from torch import nn


class AlexNetCIFAR(nn.Module):
    """Part A9 CIFAR-10 adapted AlexNet (3x32x32 input).

    Default configuration matches the Part A9 table:
      Conv1(3->64) + Pool -> Conv2(64->192) + Pool -> Conv3(192->384)
      -> Conv4(384->256) -> Conv5(256->256) + Pool -> Flatten(4096)
      -> FC1(4096->1024) + Drop -> FC2(1024->512) + Drop -> FC3(512->10)
    """

    def __init__(
        self,
        num_classes: int = 10,
        dropout: float = 0.5,
        width_mult: float = 1.0,
        fc_hidden1: int = 1024,
        fc_hidden2: int = 512,
        use_bn: bool = False,
    ) -> None:
        super().__init__()
        c1 = max(8, int(round(64 * width_mult)))
        c2 = max(16, int(round(192 * width_mult)))
        c3 = max(32, int(round(384 * width_mult)))
        c4 = max(32, int(round(256 * width_mult)))
        c5 = max(32, int(round(256 * width_mult)))

        def conv_block(in_ch: int, out_ch: int) -> list[nn.Module]:
            layers: list[nn.Module] = [
                nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=not use_bn)
            ]
            if use_bn:
                layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.ReLU(inplace=True))
            return layers

        self.stage1 = nn.Sequential(*conv_block(3, c1), nn.MaxPool2d(kernel_size=2, stride=2))
        self.stage2 = nn.Sequential(*conv_block(c1, c2), nn.MaxPool2d(kernel_size=2, stride=2))
        self.stage3 = nn.Sequential(
            *conv_block(c2, c3),
            *conv_block(c3, c4),
            *conv_block(c4, c5),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c5 * 4 * 4, fc_hidden1),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(fc_hidden1, fc_hidden2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(fc_hidden2, num_classes),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        return self.classifier(x)


class BasicBlock(nn.Module):
    """Standard ResNet basic residual block (2x 3x3 Conv + BN, ReLU after skip addition)."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.shortcut: nn.Module = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return self.relu(out)


class ResNetCIFAR(nn.Module):
    """Part A9 CIFAR-10 adapted ResNet (default ResNet-18: layers=(2, 2, 2, 2), base_width=64).

    Uses a 3x3 stride-1 stem without initial max pooling to preserve 32x32 resolution.
    Configurable layers, base_width, and head dropout support Task 3 ablations cleanly.
    """

    def __init__(
        self,
        layers: Sequence[int] = (2, 2, 2, 2),
        base_width: int = 64,
        dropout: float = 0.0,
        num_classes: int = 10,
    ) -> None:
        super().__init__()
        if len(layers) != 4:
            raise ValueError(f"Expected 4 stage block counts, got {layers}")

        widths = (base_width, base_width * 2, base_width * 4, base_width * 8)
        self.stem = nn.Sequential(
            nn.Conv2d(3, widths[0], kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(widths[0]),
            nn.ReLU(inplace=True),
        )
        self.stage1 = self._make_stage(widths[0], widths[0], num_blocks=layers[0], stride=1)
        self.stage2 = self._make_stage(widths[0], widths[1], num_blocks=layers[1], stride=2)
        self.stage3 = self._make_stage(widths[1], widths[2], num_blocks=layers[2], stride=2)
        self.stage4 = self._make_stage(widths[2], widths[3], num_blocks=layers[3], stride=2)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(widths[3], num_classes),
        )
        self._init_weights()

    @staticmethod
    def _make_stage(
        in_channels: int,
        out_channels: int,
        num_blocks: int,
        stride: int,
    ) -> nn.Sequential:
        blocks = [BasicBlock(in_channels, out_channels, stride=stride)]
        for _ in range(1, num_blocks):
            blocks.append(BasicBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*blocks)

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        return self.head(x)


def build_model(
    name: str,
    options: dict[str, Any] | None = None,
    device: torch.device | str | None = None,
) -> tuple[nn.Module, int]:
    """Instantiate a CIFAR-10 model by name and return (model, input_size)."""
    opts = dict(options) if options else {}
    if name == "alexnet_cifar":
        model: nn.Module = AlexNetCIFAR(**opts)
    elif name in ("resnet18_cifar", "resnet_cifar"):
        model = ResNetCIFAR(**opts)
    else:
        raise ValueError(f"Unknown model name: {name!r}")

    if device is not None:
        model = model.to(device)
    return model, 32


def count_parameters(model: nn.Module) -> dict[str, int]:
    """Return total and trainable parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total_params": int(total), "trainable_params": int(trainable)}


@torch.inference_mode()
def trace_shapes(
    model: nn.Module,
    input_size: int = 32,
    device: torch.device | str | None = None,
) -> list[dict[str, Any]]:
    """Trace top-level stage output shapes using forward hooks (aligned with Part A4)."""
    if device is None:
        device = next(model.parameters()).device

    rows: list[dict[str, Any]] = []
    hooks = []

    def make_hook(stage_name: str):
        def _hook(_module, _inputs, output):
            rows.append({"stage": stage_name, "shape": list(output.shape)})

        return _hook

    for stage_name, layer in model.named_children():
        hooks.append(layer.register_forward_hook(make_hook(stage_name)))

    was_training = model.training
    model.eval()
    try:
        dummy = torch.zeros(1, 3, input_size, input_size, device=device)
        logits = model(dummy)
        assert logits.shape == (1, 10) and torch.isfinite(logits).all()
    finally:
        for hook in hooks:
            hook.remove()
        model.train(was_training)

    return rows
