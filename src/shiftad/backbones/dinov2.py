"""Official DINOv2 adapter with explicit offline and weight-loading policy."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import torch
from torch import Tensor, nn

from shiftad.backbones.base import BackboneError, FeatureBatch, validate_image_batch

SUPPORTED_MODELS = frozenset({"dinov2_vits14", "dinov2_vitb14"})


class DinoV2Backbone(nn.Module):
    """Extract intermediate official DINOv2 patch tokens.

    Args:
        model_name: Supported official torch-hub model name.
        layers: Transformer block indices passed to the official intermediate API.
        device: Inference device.
        dtype: Floating-point inference dtype.
        checkpoint: Optional local state-dict checkpoint.
        allow_download: Permit official torch-hub network access when required.
        use_cache: Permit an existing official torch-hub repository cache.
        model: Injected compatible module for isolated adapter tests only.
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
        dtype: torch.dtype = torch.float32,
        checkpoint: Path | None = None,
        allow_download: bool = False,
        use_cache: bool = True,
        model: nn.Module | None = None,
        expected_input_size: tuple[int, int] | None = None,
    ) -> None:
        super().__init__()
        if model_name not in SUPPORTED_MODELS:
            raise BackboneError(
                f"Unsupported DINOv2 model {model_name!r}; expected one of "
                f"{sorted(SUPPORTED_MODELS)}."
            )
        if not layers or min(layers) < 0 or len(set(layers)) != len(layers):
            raise BackboneError("DINOv2 layers must be unique non-negative indices.")
        if dtype not in {torch.float32, torch.float16, torch.bfloat16}:
            raise BackboneError(f"Unsupported DINOv2 dtype {dtype}.")
        self.model_name = model_name
        self.layers = layers
        self.checkpoint = checkpoint
        self.allow_download = allow_download
        self.use_cache = use_cache
        self.inference_dtype = dtype
        self.expected_input_size = expected_input_size
        self.model = model or self._load_model()
        self.model.eval()
        self.model.requires_grad_(False)
        self.model.to(device=device, dtype=dtype)

    @property
    def device(self) -> torch.device:
        """Return the device of the frozen model."""

        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    @property
    def is_real_dinov2(self) -> bool:
        """Return ``True`` for the official adapter."""

        return True

    def _loading_error(self, detail: str) -> BackboneError:
        checkpoint_state = (
            "not provided"
            if self.checkpoint is None
            else f"{self.checkpoint} ({'exists' if self.checkpoint.is_file() else 'missing'})"
        )
        return BackboneError(
            f"Cannot load DINOv2 model {self.model_name!r}: {detail}. "
            f"Local checkpoint: {checkpoint_state}; allow_download={self.allow_download}; "
            f"use_cache={self.use_cache}. Provide a valid local checkpoint with the "
            "official torch-hub repository already cached, or explicitly set "
            "allow_download: true."
        )

    def _load_model(self) -> nn.Module:
        if self.checkpoint is not None and not self.checkpoint.is_file():
            raise self._loading_error("the configured checkpoint does not exist")
        hub_cache = Path(torch.hub.get_dir())
        cached_repositories = sorted(hub_cache.glob("facebookresearch_dinov2_*"))
        model: nn.Module
        hub_load = cast(Callable[..., nn.Module], torch.hub.load)
        try:
            if self.checkpoint is not None and self.use_cache and cached_repositories:
                model = hub_load(
                    str(cached_repositories[0]),
                    self.model_name,
                    source="local",
                    pretrained=False,
                )
            elif self.allow_download:
                model = hub_load(
                    "facebookresearch/dinov2",
                    self.model_name,
                    pretrained=self.checkpoint is None,
                )
            else:
                raise self._loading_error(
                    "downloads are disabled and loading local weights also requires "
                    "an existing official torch-hub repository cache"
                )
        except BackboneError:
            raise
        except (OSError, RuntimeError, ValueError) as exc:
            raise self._loading_error(str(exc)) from exc
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

    def _patch_size(self) -> tuple[int, int]:
        patch_size = getattr(self.model, "patch_size", None)
        if patch_size is None and hasattr(self.model, "patch_embed"):
            patch_size = getattr(self.model.patch_embed, "patch_size", None)
        if isinstance(patch_size, int):
            return patch_size, patch_size
        if (
            isinstance(patch_size, (tuple, list))
            and len(patch_size) == 2
            and all(isinstance(value, int) for value in patch_size)
        ):
            return int(patch_size[0]), int(patch_size[1])
        raise BackboneError(
            f"DINOv2 model {self.model_name!r} does not expose a usable patch size."
        )

    @torch.inference_mode()
    def extract_features(self, images: Tensor) -> FeatureBatch:
        """Extract patch tokens while excluding CLS tokens from spatial output."""

        _, _, height, width = validate_image_batch(images)
        if (
            self.expected_input_size is not None
            and (
                height,
                width,
            )
            != self.expected_input_size
        ):
            raise BackboneError(
                f"DINOv2 input {(height, width)} does not match configured input size "
                f"{self.expected_input_size}."
            )
        patch_height, patch_width = self._patch_size()
        if height % patch_height or width % patch_width:
            raise BackboneError(
                f"DINOv2 input {(height, width)} must be divisible by patch size "
                f"{(patch_height, patch_width)}."
            )
        if not hasattr(self.model, "get_intermediate_layers"):
            raise BackboneError(
                f"DINOv2 model {self.model_name!r} lacks get_intermediate_layers()."
            )
        inputs = images.to(device=self.device, dtype=self.inference_dtype)
        extractor = cast(
            Callable[..., list[tuple[Tensor, Tensor]]],
            self.model.get_intermediate_layers,
        )
        try:
            outputs = extractor(
                inputs,
                n=list(self.layers),
                reshape=False,
                return_class_token=True,
            )
        except (RuntimeError, TypeError, ValueError, IndexError) as exc:
            raise BackboneError(
                f"DINOv2 layer extraction failed for layers {self.layers}: {exc}"
            ) from exc
        if len(outputs) != len(self.layers):
            raise BackboneError(
                f"DINOv2 returned {len(outputs)} layers for requested {self.layers}."
            )
        grid = (height // patch_height, width // patch_width)
        layer_features: dict[str, Tensor] = {}
        class_token_shapes: dict[str, list[int]] = {}
        for layer, output in zip(self.layers, outputs, strict=True):
            if not isinstance(output, (tuple, list)) or len(output) != 2:
                raise BackboneError("DINOv2 intermediate output must contain patch and CLS tokens.")
            patch_tokens, class_token = output
            if not isinstance(patch_tokens, Tensor) or not isinstance(class_token, Tensor):
                raise BackboneError("DINOv2 intermediate outputs must be tensors.")
            name = f"blocks.{layer}"
            layer_features[name] = patch_tokens
            class_token_shapes[name] = list(class_token.shape)
        result = FeatureBatch(
            layer_features=layer_features,
            patch_grid_shapes={name: grid for name in layer_features},
            input_size=(height, width),
            metadata={
                "backbone": self.model_name,
                "layers": list(self.layers),
                "patch_size": [patch_height, patch_width],
                "class_token_shapes": class_token_shapes,
                "weight_source": "official_dinov2",
            },
        )
        result.validate()
        return result
