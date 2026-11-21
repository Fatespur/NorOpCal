"""Backbone contract, DINOv2 adapter, and feature-fusion tests."""

from pathlib import Path

import pytest
import torch
from torch import Tensor, nn

from shiftad.backbones.base import BackboneError
from shiftad.backbones.dinov2 import DinoV2Backbone
from shiftad.backbones.feature_extractor import FeatureFuser
from shiftad.backbones.fixture import DeterministicFixtureBackbone


class StubDino(nn.Module):
    """Small deterministic implementation of the official intermediate API."""

    patch_size = 4

    def __init__(self, depth: int = 4) -> None:
        super().__init__()
        self.depth = depth
        self.scale = nn.Parameter(torch.tensor(1.0))

    def get_intermediate_layers(
        self,
        images: Tensor,
        *,
        n: list[int],
        reshape: bool,
        return_class_token: bool,
    ) -> list[tuple[Tensor, Tensor]]:
        """Return deterministic patch and CLS tokens for requested blocks."""

        assert not reshape and return_class_token
        batch, _, height, width = images.shape
        pooled = torch.nn.functional.avg_pool2d(images, 4, 4)
        tokens = pooled.flatten(2).transpose(1, 2)
        outputs = []
        for layer in n:
            if layer >= self.depth:
                raise IndexError(layer)
            patch_tokens = tokens * self.scale + layer
            class_token = torch.full(
                (batch, 3),
                float(layer),
                device=images.device,
            )
            assert patch_tokens.shape[1] == height // 4 * (width // 4)
            outputs.append((patch_tokens, class_token))
        return outputs


def test_fixture_backbone_is_frozen_batched_and_cpu_safe() -> None:
    """Fixture extraction preserves batch and stable layer names on CPU."""

    backbone = DeterministicFixtureBackbone(layers=(0, 2), patch_size=4)
    assert all(not parameter.requires_grad for parameter in backbone.parameters())
    features = backbone.extract_features(torch.ones(2, 3, 8, 12))
    assert list(features.layer_features) == ["fixture.0", "fixture.2"]
    assert features.layer_features["fixture.0"].shape == (2, 6, 4)
    assert features.patch_grid_shapes["fixture.0"] == (2, 3)
    assert features.layer_features["fixture.0"].device.type == "cpu"


def test_fixture_backbone_rejects_invalid_input() -> None:
    """Rank, channels, and patch divisibility produce actionable errors."""

    backbone = DeterministicFixtureBackbone(patch_size=4)
    with pytest.raises(BackboneError, match=r"\[B, C, H, W\]"):
        backbone.extract_features(torch.ones(3, 8, 8))
    with pytest.raises(BackboneError, match="RGB"):
        backbone.extract_features(torch.ones(1, 1, 8, 8))
    with pytest.raises(BackboneError, match="not divisible"):
        backbone.extract_features(torch.ones(1, 3, 9, 8))


def test_dinov2_adapter_excludes_cls_and_restores_patch_grid() -> None:
    """Injected official-style outputs retain only spatial tokens."""

    model = StubDino()
    backbone = DinoV2Backbone(
        model_name="dinov2_vits14",
        layers=(0, 2),
        model=model,
    )
    output = backbone.extract_features(torch.ones(2, 3, 8, 12))
    assert output.patch_grid_shapes == {"blocks.0": (2, 3), "blocks.2": (2, 3)}
    assert output.layer_features["blocks.0"].shape == (2, 6, 3)
    assert output.metadata["class_token_shapes"] == {
        "blocks.0": [2, 3],
        "blocks.2": [2, 3],
    }
    assert all(not parameter.requires_grad for parameter in model.parameters())


def test_dinov2_adapter_reports_layer_and_weight_errors(tmp_path: Path) -> None:
    """Bad layers and absent local weights never trigger a silent random model."""

    backbone = DinoV2Backbone(
        model_name="dinov2_vits14",
        layers=(9,),
        model=StubDino(),
    )
    with pytest.raises(BackboneError, match="layers"):
        backbone.extract_features(torch.ones(1, 3, 8, 8))
    missing = tmp_path / "missing.pth"
    with pytest.raises(BackboneError, match="allow_download=False"):
        DinoV2Backbone(
            model_name="dinov2_vits14",
            layers=(0,),
            checkpoint=missing,
            allow_download=False,
            use_cache=False,
        )


def test_dinov2_default_loading_policy_never_calls_network(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Disabled downloads fail before invoking the official remote loader."""

    called = False

    def _unexpected_load(*args: object, **kwargs: object) -> nn.Module:
        nonlocal called
        called = True
        raise AssertionError((args, kwargs))

    monkeypatch.setattr(torch.hub, "get_dir", lambda: str(tmp_path))
    monkeypatch.setattr(torch.hub, "load", _unexpected_load)
    with pytest.raises(BackboneError, match="downloads are disabled"):
        DinoV2Backbone(
            model_name="dinov2_vits14",
            layers=(0,),
            allow_download=False,
            use_cache=True,
        )
    assert called is False


def test_feature_fusion_known_values_alignment_and_normalization() -> None:
    """Concat and mean fusion match analytically known tensors."""

    layer_a = torch.tensor([[[1.0], [2.0], [3.0], [4.0]]])
    layer_b = torch.tensor([[[10.0]]])
    concat = FeatureFuser(mode="concat", normalize=False)(
        layer_features={"a": layer_a, "b": layer_b},
        patch_grid_shapes={"a": (2, 2), "b": (1, 1)},
    )
    assert concat.grid_shape == (2, 2)
    assert torch.equal(
        concat.features,
        torch.tensor([[[1.0, 10.0], [2.0, 10.0], [3.0, 10.0], [4.0, 10.0]]]),
    )
    mean = FeatureFuser(mode="mean", normalize=False)(
        layer_features={"a": layer_a, "b": torch.ones_like(layer_a) * 3},
        patch_grid_shapes={"a": (2, 2), "b": (2, 2)},
    )
    assert torch.equal(mean.features, torch.tensor([[[2.0], [2.5], [3.0], [3.5]]]))
    normalized = FeatureFuser(mode="concat", normalize=True)(
        layer_features={"a": layer_a, "b": layer_b},
        patch_grid_shapes={"a": (2, 2), "b": (1, 1)},
    )
    assert torch.allclose(
        torch.linalg.vector_norm(normalized.features, dim=-1),
        torch.ones(1, 4),
    )
