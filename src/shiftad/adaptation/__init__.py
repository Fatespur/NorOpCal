"""ShiftAD reliability-gated online adaptation (Phase 6).

Public API for the adaptation package: core boundary types, versioned target
state, append-only audit log, causal stream engine, and the component
interfaces (shift estimator, reliability gate, target memory, score fusion,
safety monitor) with their FIXTURE_SCOPE deterministic implementations.
"""

from shiftad.adaptation.audit import AuditLog, AuditRecord
from shiftad.adaptation.base import (
    FIXTURE_SCOPE,
    PRODUCTION_SCOPE,
    PROTOCOL_VERSION,
    REASON_ADAPTATION_OFF,
    REASON_CONSISTENCY,
    REASON_IDENTITY,
    REASON_INPUT_QUALITY,
    REASON_RATE_LIMIT,
    REASON_ROLLBACK,
    REASON_SCORE_LIMIT,
    REASON_SHIFT_MEANINGFUL,
    REASON_SUSPICIOUS_AREA,
    AdaptationError,
    AdaptationPolicy,
    FusedScore,
    GateDecision,
    LeakageError,
    SchemaRejectionError,
    ShiftEstimate,
    SourceAnchor,
    SourceAnchorError,
)
from shiftad.adaptation.fusion import (
    W_C_MAX,
    W_S_MIN,
    W_T_MAX,
    AdaptationOffFusion,
    ConservativeScoreFusion,
    ScoreFusion,
)
from shiftad.adaptation.image_stats import (
    IMAGE_STATISTIC_NAMES,  # noqa: F401  (re-exported via __all__)
    brightness,
    channel_means,
    color_temperature,
    compute_image_stats,
    contrast,
    hf_noise,
    laplacian_sharpness,
    resolution_energy,
)
from shiftad.adaptation.reliability_gate import (
    ACCEPTED_CODE,
    FixtureReliabilityGate,
    ReliabilityGate,
    V1ReliabilityGate,
)
from shiftad.adaptation.safety import (
    FixtureSafetyMonitor,
    SafetyMonitor,
    SafetyVerdict,
    V1SafetyMonitor,
    require_valid_snapshot,
)
from shiftad.adaptation.shift_estimator import (
    FixtureShiftEstimator,
    ShiftEstimator,
    SourceNormalStatistics,
    V1ShiftEstimator,
    WindowStatistics,
)
from shiftad.adaptation.state import (
    TargetState,
    TargetStateSnapshot,
)
from shiftad.adaptation.streaming import (
    CausalStreamEngine,
    StreamEvaluator,
    StreamTransaction,
)
from shiftad.adaptation.target_memory import FixtureTargetMemory, TargetMemory, V1TargetMemory

__all__ = [
    "ACCEPTED_CODE",
    "FIXTURE_SCOPE",
    "PRODUCTION_SCOPE",
    "PROTOCOL_VERSION",
    "REASON_ADAPTATION_OFF",
    "REASON_CONSISTENCY",
    "REASON_IDENTITY",
    "REASON_INPUT_QUALITY",
    "REASON_RATE_LIMIT",
    "REASON_ROLLBACK",
    "REASON_SCORE_LIMIT",
    "REASON_SHIFT_MEANINGFUL",
    "REASON_SUSPICIOUS_AREA",
    "W_C_MAX",
    "W_S_MIN",
    "W_T_MAX",
    "AdaptationError",
    "AdaptationOffFusion",
    "AdaptationPolicy",
    "AuditLog",
    "AuditRecord",
    "CausalStreamEngine",
    "ConservativeScoreFusion",
    "FixtureReliabilityGate",
    "FixtureSafetyMonitor",
    "FixtureShiftEstimator",
    "FixtureTargetMemory",
    "FusedScore",
    "GateDecision",
    "LeakageError",
    "ReliabilityGate",
    "SafetyMonitor",
    "SafetyVerdict",
    "SchemaRejectionError",
    "ScoreFusion",
    "ShiftEstimate",
    "ShiftEstimator",
    "SourceAnchor",
    "SourceAnchorError",
    "SourceNormalStatistics",
    "StreamEvaluator",
    "StreamTransaction",
    "TargetMemory",
    "TargetState",
    "TargetStateSnapshot",
    "V1ReliabilityGate",
    "V1SafetyMonitor",
    "V1ShiftEstimator",
    "V1TargetMemory",
    "WindowStatistics",
    "brightness",
    "channel_means",
    "color_temperature",
    "compute_image_stats",
    "contrast",
    "hf_noise",
    "laplacian_sharpness",
    "require_valid_snapshot",
    "resolution_energy",
]
