"""Torchvision ResNet adapter with intermediate-layer feature hooks.

This backbone provides the patch-feature extraction required by PatchCore.
It wraps a frozen torchvision ResNet (or Wide ResNet) and captures
intermediate activations from ``layer1`` through ``layer4`` using forward
hooks, then reshapes them into the ``[B, N, C]`` token format expected by
:class:`~shiftad.backbones.base.FeatureBatch`.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from shiftad.backbones.base import BackboneError, FeatureBatch, validate_image_batch

SUPPORTED_MODELS = frozenset({"resnet18", "resnet50", "wide_resnet50_2"})

#: Mapping from integer layer indices to torchvision ResNet submodule names.
LAYER_ATTRIBUTES: dict[int, str] = {
    1: "layer1",
    2: "layer2",
    3: "layer3",
    4: "layer4",
}

DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


class ResNetBackbone(nn.Module):
    """Extract intermediate patch features from a frozen torchvision ResNet.

    Args:
        model_name: Supported torchvision model name.
        layers: ResNet layer indices in ``{1, 2, 3, 4}``.
        device: Inference device.
        dtype: Floating-point inference dtype string.
        checkpoint: Optional local state-dict checkpoint path.
        allow_download: Permit torchvision network access for ImageNet weights.
        use_cache: Permit reuse of cached torchvision weights.
        model: Injected compatible module for isolated tests only.
        expected_input_size: Optional configured height and width contract.

    Raises:
        BackboneError: If model configuration or weight loading is unavailable.
    """

    def __init__(
        self,
        *,
        model_name: str,
        layers: tuple[int, ...],
        device: torch.device | str = "cpu",
        dtype: str = "float32",
        checkpoint: Path | None = None,
        allow_download: bool = False,
        use_cache: bool = True,
        model: nn.Module | None = None,
        expected_input_size: tuple[int, int] | None = None,
    ) -> None:
        super().__init__()
        if model_name not in SUPPORTED_MODELS:
            raise BackboneError(
                f"Unsupported ResNet model {model_name!r}; expected one of "
                f"{sorted(SUPPORTED_MODELS)}."
            )
        if not layers or len(set(layers)) != len(layers):
            raise BackboneError("ResNet layers must be unique non-empty indices.")
        invalid = [layer for layer in layers if layer not in LAYER_ATTRIBUTES]
        if invalid:
            raise BackboneError(
                f"ResNet layer indices {invalid} are invalid; expected values from "
                f"{sorted(LAYER_ATTRIBUTES)}."
            )
        if dtype not in DTYPES:
            raise BackboneError(
                f"Unsupported ResNet dtype {dtype!r}; expected float32, float16, or bfloat16."
            )
        self.model_name = model_name
        self.layers = layers
        self.checkpoint = checkpoint
        self.allow_download = allow_download
        self.use_cache = use_cache
        self.inference_dtype = DTYPES[dtype]
        self.dtype_name = dtype
        self.expected_input_size = expected_input_size
        self.model = model or self._load_model()
        self.model.eval()
        self.model.requires_grad_(False)
        self.model.to(device=device, dtype=self.inference_dtype)

        self._hook_handles: list[Any] = []
        self._captured: dict[str, Tensor] = {}
        self._register_hooks()

    @property
    def device(self) -> torch.device:
        """Return the device of the frozen model."""

        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    @property
    def is_real_dinov2(self) -> bool:
        """Return ``False`` because this is a ResNet, not DINOv2."""

        return False

    def _loading_error(self, detail: str) -> BackboneError:
        checkpoint_state = (
            "not provided"
            if self.checkpoint is None
            else f"{self.checkpoint} ({'exists' if self.checkpoint.is_file() else 'missing'})"
        )
        return BackboneError(
            f"Cannot load ResNet model {self.model_name!r}: {detail}. "
            f"Local checkpoint: {checkpoint_state}; allow_download={self.allow_download}; "
            f"use_cache={self.use_cache}. Provide a valid local checkpoint or set "
            "allow_download: true to fetch ImageNet pretrained weights."
        )

    def _load_model(self) -> nn.Module:
        """Load a torchvision ResNet with configurable weight policy.

        Weight loading priority:
        1. Local checkpoint (if provided, always used)
        2. ``allow_download=True`` or ``use_cache=True``: load torchvision
           ``DEFAULT`` weights, which use the cache when available and download
           only when the cache is empty.
        3. Otherwise: raise a clear configuration error.
        """

        if self.checkpoint is not None and not self.checkpoint.is_file():
            raise self._loading_error("the configured checkpoint does not exist")

        factory = self._model_factory()
        if self.checkpoint is not None:
            model = factory(weights=None)
        elif self.allow_download or self.use_cache:
            try:
                model = factory(weights="DEFAULT")
            except (OSError, RuntimeError, ValueError) as exc:
                if not self.allow_download:
                    raise self._loading_error(
                        f"Cannot load pretrained weights from cache: {exc}. "
                        "Set allow_download: true to download, or provide a checkpoint path."
                    ) from exc
                raise self._loading_error(f"torchvision weight download failed: {exc}") from exc
        else:
            raise self._loading_error("downloads are disabled and no local checkpoint was provided")

        if self.checkpoint is not None:
            try:
                payload: Any = torch.load(
                    self.checkpoint,
                    map_location="cpu",
                    weights_only=True,
                )
            except (OSError, RuntimeError, ValueError) as exc:
                raise self._loading_error(f"checkpoint loading failed: {exc}") from exc
            if isinstance(payload, dict) and "state_dict" in payload:
                payload = payload["state_dict"]
            if not isinstance(payload, dict) or not all(
                isinstance(key, str) and isinstance(value, Tensor) for key, value in payload.items()
            ):
                raise self._loading_error("checkpoint is not a tensor state dictionary")
            try:
                model.load_state_dict(payload, strict=True)
            except RuntimeError as exc:
                raise self._loading_error(f"checkpoint is incompatible: {exc}") from exc
        return model

    def _model_factory(self) -> Callable[..., nn.Module]:
        """Return the torchvision model constructor for the configured name."""

        import torchvision.models as tv_models

        factories: dict[str, Callable[..., nn.Module]] = {
            "resnet18": tv_models.resnet18,
            "resnet50": tv_models.resnet50,
            "wide_resnet50_2": tv_models.wide_resnet50_2,
        }
        return factories[self.model_name]

    def _register_hooks(self) -> None:
        """Attach forward hooks to capture intermediate layer activations."""

        for layer_index in self.layers:
            attr_name = LAYER_ATTRIBUTES[layer_index]
            if not hasattr(self.model, attr_name):
                raise BackboneError(
                    f"ResNet model {self.model_name!r} lacks submodule {attr_name!r}."
                )
            module = getattr(self.model, attr_name)
            name = f"layer{layer_index}"

            def _hook(
                _module: nn.Module,
                _inputs: tuple[Tensor, ...],
                output: Tensor,
                *,
                _name: str = name,
            ) -> None:
                self._captured[_name] = output

            handle = module.register_forward_hook(_hook)
            self._hook_handles.append(handle)

    @torch.inference_mode()
    def extract_features(self, images: Tensor) -> FeatureBatch:
        """Extract spatial patch features from configured ResNet layers.

        Args:
            images: Floating-point RGB tensor shaped ``[B, 3, H, W]``.

        Returns:
            Named patch features reshaped to ``[B, N, C]`` per layer.

        Raises:
            BackboneError: If inputs are invalid or hook capture fails.
        """

        _, _, height, width = validate_image_batch(images)
        if self.expected_input_size is not None and (height, width) != self.expected_input_size:
            raise BackboneError(
                f"ResNet input {(height, width)} does not match configured input size "
                f"{self.expected_input_size}."
            )
        inputs = images.to(device=self.device, dtype=self.inference_dtype)
        self._captured.clear()
        try:
            self.model(inputs)
        except (RuntimeError, TypeError, ValueError) as exc:
            raise BackboneError(f"ResNet forward pass failed: {exc}") from exc

        layer_features: dict[str, Tensor] = {}
        patch_grid_shapes: dict[str, tuple[int, int]] = {}
        for layer_index in self.layers:
            name = f"layer{layer_index}"
            if name not in self._captured:
                raise BackboneError(f"Forward hook did not capture ResNet layer {name!r}.")
            activation = self._captured[name]
            if activation.ndim != 4:
                raise BackboneError(
                    f"ResNet layer {name!r} output has shape {tuple(activation.shape)}; "
                    "expected [B, C, H, W]."
                )
            _batch_size, _channels, grid_h, grid_w = activation.shape
            # Convert [B, C, H, W] to [B, N, C] where N = H * W.
            tokens = activation.flatten(2).transpose(1, 2)
            tokens = tokens.to(torch.float32)
            layer_features[name] = tokens
            patch_grid_shapes[name] = (grid_h, grid_w)

        result = FeatureBatch(
            layer_features=layer_features,
            patch_grid_shapes=patch_grid_shapes,
            input_size=(height, width),
            metadata={
                "backbone": self.model_name,
                "layers": list(self.layers),
                "weight_source": (
                    "checkpoint" if self.checkpoint is not None else "torchvision_imagenet"
                ),
                "dtype": self.dtype_name,
            },
        )
        result.validate()
        return result

    def __del__(self) -> None:
        """Remove forward hooks on destruction to avoid reference leaks."""

        for handle in getattr(self, "_hook_handles", []):
            with contextlib.suppress(Exception):
                handle.remove()
