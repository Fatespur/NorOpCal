# PB7-P0 Normal-Only Threshold Correctability Identifiability Audit — Final Report

> 阶段：PB7-P0（Normal-Only Threshold Correctability Identifiability Audit）
> 日期：2026-08-20
> 状态：**COMPLETE** — 机制裁决已封存（F4）
> 本报告为当前审计的最终汇总；各阶段原始证据见 `reports/pb7_p0_threshold_correctability/`。

---

## 1. 协议（Protocol）

本审计遵循硬顺序 `P0 → A → B → C → D → E → F → FINAL`，每个阶段均产出正式 artifact、机器可读 JSON/CSV、自检记录，并通过 gate 后才进入下一阶段。完整协议见 `P0_PROTOCOL.json`，输入冻结见 `P0_INPUT_MANIFEST.csv`（176 个输入，SHA256 绑定），数据角色见 `P0_DATA_ROLES.json`，estimand 注册见 `P0_ESTIMAND_REGISTRY.json`，假设注册见 `P0_HYPOTHESIS_REGISTRY.json`。

### 1.1 冻结的科学背景（不可重新解释）

- PB6 terminal 已封存：`PB1DS_PB6_COMPLETE_REAL_THIRD_DETECTOR_DOES_NOT_SUPPORT_CORE_SELECTIVE_RECALIBRATION_CLAIM`，本审计不修改该 terminal。
- PB6_POSTHOC_ACTUATOR_AUDIT 的 H1–H7 全部 SUPPORTED：QM 正 ΔBA 属于 operating-point rebalancing，而非 representation/ranking capability improvement。
- 冻结 BENEFIT：`ΔBA ≥ +0.02 AND ΔFPR ≤ -0.05 AND ΔFNR ≤ +0.02`。
- 冻结 source threshold：`τ_source = Q95(source-normal scores)`。
- 总体：39 units × 3 detectors = 117 entities（MVTec AD 2，8 categories）。

### 1.2 数据角色与防火墙

- **DEPLOYABLE**：source-normal、target-normal、detector identity、category identity。
- **ORACLE_DIAGNOSTIC_ONLY**：target anomaly scores/labels、BA/FPR/FNR 结果、oracle thresholds。
- 代码级 `Firewall`（`pb7_p0_common.py`）：deployable 计算路径读取任何 anomaly/oracle 文件即抛 `FIREWALL_VIOLATION`。P0 Gate 验证通过。

### 1.3 Estimand（O1/O2/O3）

- **O1 τ\*_BA**：冻结候选 threshold grid 上 BA 最大的 threshold（tie-break 与 endpoint 处理见 `P0_ESTIMAND_REGISTRY.json`）。
- **O2 τ\*_SAFE**：在 `ΔFNR ≤ +0.02` 约束内最大化 BA；并列时选择离 source threshold 最近者。
- **O3 CORRECTABLE_SAFE**：`ΔBA ≥ +0.02 AND ΔFPR ≤ -0.05 AND ΔFNR ≤ +0.02`（相对 source operating point）。仅作 diagnostic label，禁止作为 deployment input。

### 1.4 假设（H1–H6）

见 `P0_HYPOTHESIS_REGISTRY.json`。裁决汇总见 §8。

---

## 2. Oracle Threshold Headroom（Stage A — PASS）

A 阶段只做诊断，不开发 deployable method。结果见 `A1_SOURCE_OPERATING_POINT.csv`、`A2_ORACLE_HEADROOM.csv`、`A2_ORACLE_THRESHOLD_CURVES.csv`、`A3_DETECTOR_HEADROOM_SUMMARY.csv`、`A4_CORRECTABILITY_TAXONOMY.csv`。

### 2.1 总体 headroom

| 指标 | 值 |
|---|---|
| 总体 safe-correctable（CORRECTABLE_SAFE=1） | **22 / 117（18.8%）** |
| 总体 ΔBA_safe ≥ +0.02 的 entity | **103 / 117（88.0%）** |
| Oracle 上界（DIAGNOSTIC ONLY） | mean ΔBA = +0.110，CI_lo = +0.092，mean ΔFNR = −0.327，harm_rate = 0.0，coverage = 0.889 |

### 2.2 三 detector 分层（A3）

| detector | n | safe-correctable | frac | mean ΔBA_safe | mean Δτ_safe | mean ΔFPR_safe | mean ΔFNR_safe |
|---|---|---|---|---|---|---|---|
| POOLED | 117 | 22 | 0.188 | +0.109 | −0.049 | +0.107 | −0.325 |
| DINOv2 Memory | 39 | **16** | **0.410** | +0.124 | −0.022 | −0.028 | −0.219 |
| PatchCore (WRN50-2) | 39 | 1 | 0.026 | +0.103 | −0.051 | +0.207 | −0.414 |
| EfficientAD | 39 | 5 | 0.128 | +0.101 | −0.075 | +0.141 | −0.344 |

**H1 SUPPORTED**：DINOv2 16/39（41.0%）≥ 20% 且 ≥ 8 units，满足最低继续门槛。A Gate = PASS → PROCEED_TO_B。

### 2.3 Correctability taxonomy（A4，primary 互斥标签）

| 状态 | POOLED | DINOv2 | PatchCore | EfficientAD |
|---|---|---|---|---|
| SAFE_THRESHOLD_CORRECTABLE | 22 | 16 | 1 | 5 |
| UNSAFE_ONLY | 6 | 1 | 4 | 1 |
| SOURCE_ALREADY_NEAR_OPTIMAL | 7 | 3 | 2 | 2 |
| AMBIGUOUS_FLAT_CURVE | 20 | 9 | 4 | 7 |
| NO_HEADROOM | 62 | 10 | 28 | 24 |

注：A3 汇总中的 `n_ambiguous_flat=27` 为可重叠 flag 计数（部分 entity 同时满足 safe-correctable 与 flat-curve flag）；上表为互斥的 primary taxonomy 分布（合计 117）。

关键观察：
- **DINOv2 是唯一具有实质 safe-correctable 比例的 detector**；PatchCore 与 EfficientAD 的 safe headroom 极小。
- 大量 entity 属于 NO_HEADROOM（62/117）或 AMBIGUOUS_FLAT_CURVE（20/117），说明多数 operating point 没有可安全利用的阈值调整空间。

---

## 3. Normal-Only Deformation vs Oracle Correctability（Stage B — PASS via PASS-B）

B 阶段开始询问 oracle correctability 能否从 normal-only evidence 被识别，但仍不构造复杂方法。结果见 `B_FEATURES.csv`、`B_UNIVARIATE_IDENTIFIABILITY.csv`、`B_TRIVIAL_BASELINES.csv`。

### 3.1 冻结的 normal-only features

- BSEL 已注册 deformation features：median_shift、mean_shift、mad_ratio、std_ratio、d_q90、d_q95、d_q99、q99_q95_change、wasserstein、ks。
- 预注册 threshold-specific features：T1（source-threshold percentile under target-normal）、T2（target q95 − source q95）、T3（target exceedance rate at τ_source）。
- 无 outcome-driven 增删（B 自检确认）。

### 3.2 单变量 identifiability（全量输出，未挑最优）

| feature | Spearman ρ(Δτ\*) | ρ CI | sign_auc | correctable_auc | correctable AUC CI | rank-biserial |
|---|---|---|---|---|---|---|
| median_shift | 0.064 | [−0.141, 0.329] | 0.710 | **0.712** | [0.634, 0.830] | 0.425 |
| mean_shift | −0.022 | [−0.207, 0.229] | 0.652 | 0.665 | [0.572, 0.804] | 0.329 |
| mad_ratio | 0.174 | [0.029, 0.331] | 0.536 | 0.507 | [0.353, 0.636] | 0.014 |
| std_ratio | −0.096 | [−0.225, 0.042] | 0.341 | 0.344 | [0.169, 0.477] | −0.311 |
| d_q90 | −0.014 | [−0.198, 0.217] | 0.589 | 0.603 | [0.505, 0.719] | 0.206 |
| d_q95 | −0.072 | [−0.276, 0.148] | 0.564 | 0.575 | [0.496, 0.646] | 0.150 |
| d_q99 | −0.099 | [−0.331, 0.150] | 0.544 | 0.557 | [0.470, 0.634] | 0.115 |
| q99_q95_change | −0.203 | [−0.409, 0.041] | 0.413 | 0.408 | [0.233, 0.534] | −0.184 |
| wasserstein | −0.062 | [−0.259, 0.163] | 0.634 | 0.645 | [0.536, 0.800] | 0.290 |
| ks | 0.176 | [−0.060, 0.405] | 0.696 | 0.681 | [0.555, 0.845] | 0.362 |
| T1 | −0.029 | [−0.239, 0.137] | 0.331 | 0.315 | [0.232, 0.377] | −0.369 |
| T2 | −0.072 | [−0.276, 0.148] | 0.564 | 0.575 | [0.496, 0.646] | 0.150 |
| T3 | 0.029 | [−0.137, 0.239] | 0.669 | 0.685 | [0.623, 0.768] | 0.369 |

### 3.3 方向识别（PASS-A）— **NOT SUPPORTED**

- 没有任何 feature 达到 `|ρ(Δτ\*_SAFE)| ≥ 0.30`（最大 |ρ| = 0.203，q99_q95_change）。
- 自然 sign 规则准确率 0.37–0.53，**低于 trivial baseline B3（detector-only prior，0.79）**。
- 主导方向是**降低阈值**（82/117 为负 Δτ\*）；自然规则“正 shift → 提高阈值”对 PatchCore（23/24 反例）与 EfficientAD（19/24 反例）**方向错误**。
- detector 分层 ρ 符号不一致：DINOv2 +0.094、PatchCore −0.242、EfficientAD +0.014。

**H2（方向识别）与 H3（幅度识别）均 NOT SUPPORTED。** 方向/幅度不能由 target-normal deformation 稳定识别。

### 3.4 Correctability 识别（PASS-B）— **SUPPORTED**

- median_shift：correctable_auc = **0.712**（rank-biserial 0.425，mean LCO AUC 0.714）。
- 其他有效 feature：T3 0.685、ks 0.681、mean_shift 0.665、wasserstein 0.645。
- per-detector median_shift AUC：DINOv2 0.617、PatchCore 0.763、EfficientAD 0.676。
- per-category LCO AUC 0.64–0.91（跨 6 个 category 有变化，**非单一 category 驱动**）。
- **不是 FNR proxy**：AUC(fnr_before vs correctable) = 0.494（接近随机）；correctable 与 fpr_before 相关（AUC 0.685）；median_shift 在高 FNR regime 内仍可区分 correctability（AUC 0.765）。

**B Gate = PASS（PASS-B）→ PROCEED_TO_C。** 关键 caveat（已写入 B 自检）：Stage D 的 actuator 必须是 **SELECTIVE（决定是否修正）** 且修正保守；精确幅度修正不受 normal-only evidence 支持。

---

## 4. Few-Normal Robustness（Stage C — PASS）

C 阶段评估 K=4 / K=8 / FULL 下的稳定性。结果见 `C_FULL_FEATURES.csv`、`C_SAMPLED_FEATURES.csv`、`C_K_STABILITY.csv`。K8 为 principal practical setting；K4/K8 各 30 次嵌套 subsampling（S4 ⊂ S8 ⊆ FULL），固定 seed。

### 4.1 K-stability（principal feature = median_shift）

| K | bias vs FULL | within-std | between-std | swamp ratio | rank corr w/ FULL | sign consistency | corr_auc |
|---|---|---|---|---|---|---|---|
| 4 | +0.0006 | 0.0166 | 0.0440 | 2.64 | 0.879 | 0.872 | 0.649 |
| 8 | +0.0005 | 0.0124 | 0.0553 | **4.46** | 0.969 | **0.933** | **0.802** |

### 4.2 C3 K8 gate — PASS

| 检查 | 阈值 | K8 值 | 结果 |
|---|---|---|---|
| sign_consistency_k8 | ≥ 0.85 | 0.933 | OK |
| swamp_ratio_k8 | ≥ 2.0 | 4.46 | OK |
| detector_pattern_rho_k8 | ≥ 0.8 | 1.0 | OK |
| corr_auc_k8 | ≥ 0.6 | 0.802 | OK |

**H5 SUPPORTED**：K=8 下 deformation signal 未被 sampling noise 淹没，detector 级模式保留。C Gate = PASS → PROCEED_TO_D。

---

## 5. Pre-Registered Simple Threshold Actuators（Stage D）

候选在 D outcome 生成前全部冻结（`D_CANDIDATES.json`），无参数拟合、无 outcome-driven 修改：

| 候选 | 规则 | 角色 |
|---|---|---|
| T0 | τ_t = τ_source | mandatory baseline |
| T1 | τ_t = τ_source + (Q95_t − Q95_s) | Quantile Offset（q=0.95 冻结） |
| T2 | τ_t = median_t + (MAD_t/MAD_s)·(τ_source − median_s) | Robust Location-Scale（MAD≈0 保护） |
| T3 | τ_t = Q95(target-normal) | Tail Exceedance（α_s=0.05） |
| T4 | if T3_feature > 0.10: apply T3 else NoAdapt | Abstaining（δ=0.10 预注册，=2×nominal exceedance） |

T3 与 T1 在给定 τ_source=Q95(source) 下数学等价（协议中分别注册、按同一有效规则评估）。T4 仅在 B 阶段证明 T3_feature 与 correctability 稳定关系（AUC 0.685）后授权。

---

## 6. Strict OOF / Held-Out Evaluation（Stage E — FAIL on G3）

E 阶段才评价 deployable actuator。所有候选均为参数无关的预注册公式，直接对全部 117 entities 评估即严格 OOF（每个 entity 的 target scores 从不用于拟合任何候选）。结果见 `E_PER_ENTITY.csv`、`E_DETECTOR_SUMMARY.csv`、`E_BOOTSTRAP.csv`、`E_RANKING.csv`。

### 6.1 主结果（FULL，POOLED）

| 候选 | mean ΔBA | mean ΔFPR | mean ΔFNR | benefit_rate | harm_rate | catastrophic_rate | coverage |
|---|---|---|---|---|---|---|---|
| T0 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| T1/T3 | +0.035 | −0.112 | +0.041 | 0.094 | 0.530 | 0.410 | 1.000 |
| T2 | **+0.040** | −0.137 | **+0.056** | 0.094 | **0.573** | 0.479 | 1.000 |
| T4 | +0.043 | −0.142 | +0.055 | 0.094 | **0.376** | 0.376 | 0.615 |
| ORACLE (DIAGNOSTIC) | +0.110 | +0.107 | −0.327 | 0.188 | 0.000 | 0.000 | 0.889 |

### 6.2 Bootstrap（B=5000，entity/category-aware block bootstrap）

| 候选 | K | boot mean ΔBA | CI_lo | CI_hi | mean ΔFNR | harm CI |
|---|---|---|---|---|---|---|
| T2 | FULL | +0.0403 | +0.0210 | +0.0571 | +0.0554 | [0.464, 0.694] |
| T2 | 8 | +0.0378 | +0.0087 | +0.0660 | +0.0215 | [0.399, 0.628] |
| T4 | FULL | +0.0434 | +0.0250 | +0.0610 | +0.0549 | [0.250, 0.500] |
| T4 | 8 | +0.0529 | +0.0247 | +0.0814 | +0.0359 | [0.209, 0.531] |
| ORACLE | FULL | +0.1103 | +0.0925 | +0.1276 | −0.3267 | [0, 0] |

### 6.3 Detector-stratified（FULL，Always-On 最佳 T2）

| detector | mean ΔBA | mean ΔFNR | harm_rate |
|---|---|---|---|
| DINOv2 | **+0.0999** | +0.075 | 0.564 |
| PatchCore | +0.009 | +0.056 | 0.667 |
| EfficientAD | +0.012 | +0.036 | 0.487 |

### 6.4 E Gate 逐项

| Gate | 结果 | 证据 |
|---|---|---|
| G1 Benefit（ΔBA ≥ +0.02） | **PASS** | T2 mean ΔBA = +0.040 |
| G2 Confidence（CI_lo > 0） | **PASS** | CI_lo = +0.021 |
| G3 Safety（ΔFNR ≤ +0.02 且 harm 显著低于 Always-On） | **FAIL** | Always-On mean ΔFNR = +0.056 > +0.02；T4 harm 0.376 < Always-On 0.573 但 ΔFNR 仍超界 |
| G4 Detector validity | PASS（detector-dependent） | DINOv2 9、PatchCore 1、EfficientAD 1 |
| G5 K8 retains effect | **PASS** | K8 mean ΔBA = +0.037 |
| G6 Selectivity value | **PASS** | T4 harm 0.376 vs Always-On 0.573，T4 ΔBA +0.043 ≥ Always-On +0.040 |

**E Gate = FAIL（G3）→ terminal = THRESHOLD_CORRECTABILITY_IDENTIFIED_BUT_SIMPLE_ACTUATORS_INSUFFICIENT。**

### 6.5 Ranking capability invariance

T0–T4 只修改 operating threshold，不重映射 anomaly scores；`E_RANKING.csv` 中每个 entity 的 AUROC/AUPR 跨候选完全一致（构造上不变）。本审计**不**将任何 threshold-only 的 BA 增益称为 ranking/detection capability improvement。

---

## 7. 机制裁决（Stage F）

综合 A–E，进入 mutually exclusive terminal 判定：

| Terminal | 判定 | 依据 |
|---|---|---|
| F1 INSUFFICIENT_ORACLE_HEADROOM | 否 | A PASS：oracle headroom 存在（pooled 22/117 safe-correctable；DINOv2 16/39） |
| F2 SIGNAL_NOT_IDENTIFIABLE | 否 | B PASS-B：correctability 可由 normal-only feature 显著区分（median_shift AUC 0.712，LCO 稳定） |
| F3 FEW_NORMAL_SIGNAL_UNSTABLE | 否 | C PASS：K8 sign consistency 0.933、swamp ratio 4.46、corr AUC 0.802 |
| **F4 IDENTIFIED_BUT_SIMPLE_ACTUATORS_INSUFFICIENT** | **是** | E gate FAIL（G3）：简单预注册 actuator 的 mean ΔFNR 超 +0.02 安全界，harm rate 高 |
| F5 DETECTOR_DEPENDENT_SAFE_ADAPTATION_SUPPORTED | 否 | E gate FAIL：安全 gate 未通过，不能宣称 safe adaptation 已支持 |
| F6 DETECTOR_AGNOSTIC_SAFE_ADAPTATION_SUPPORTED | 否 | 仅 DINOv2 有实质 headroom；禁止强行 pooled |

**Terminal（封存）：`THRESHOLD_CORRECTABILITY_IDENTIFIED_BUT_SIMPLE_ACTUATORS_INSUFFICIENT`**

该 terminal 的含义：
- 核心科学问题的回答是**限定性 YES**：target-normal evidence 包含关于 safe threshold correction 的**可识别、稳定、detector-dependent** 信息（correctability 识别 PASS-B + K8 稳定性 PASS）。
- 但**方向/幅度**（sign/magnitude of Δτ\*_SAFE）**不可由 normal-only deformation 稳定识别**（PASS-A 失败，trivial detector-only prior 已超过所有 normal-only feature）。
- 预注册的简单解析 actuator（T0–T4）**尚未兑现安全可部署的修正**：降低阈值换取 FPR 下降的同时，FNR 上升超出 +0.02 安全界，harm rate 高（Always-On 0.573，T4 0.376）。
- 该 terminal **允许**后续 actuator-development cycle，但**不授权**任何方法开发。

---

## 8. 假设裁决汇总

| 假设 | 裁决 | 关键证据 |
|---|---|---|
| H1 Threshold headroom exists | **SUPPORTED** | DINOv2 16/39（41.0%）safe-correctable；pooled 103/117 ΔBA_safe ≥ +0.02 |
| H2 Direction identifiability | **NOT SUPPORTED** | 最大 \|ρ\|=0.203；自然 sign 规则 0.37–0.53 << detector-only 0.79；PatchCore/EfficientAD 方向反 |
| H3 Magnitude identifiability | **NOT SUPPORTED** | 无稳定 monotonic/regression relation |
| H4 Detector dependence | **SUPPORTED** | correctable prevalence DINOv2 16/39 vs PatchCore 1/39 vs EfficientAD 5/39；sign 模式按 detector 显著不同 |
| H5 Few-normal stability | **SUPPORTED** | K8 sign consistency 0.933、swamp ratio 4.46、corr AUC 0.802 |
| H6 Safe deployable actuator exists | **NOT SUPPORTED** | 简单 actuator T0–T4 未通过 G3 安全 gate（mean ΔFNR +0.056 > +0.02） |

---

## 9. 已知限制与 NOT_VERIFIED

1. **方向/幅度不可识别**：H2/H3 明确 NOT SUPPORTED。任何未来 actuator 必须基于“是否修正”（selective/abstaining），不能依赖 normal-only 精确方向/幅度估计。
2. **简单 actuator 安全性不达标**：T1–T4 的 FNR 上升超出安全界。T4（abstaining）显著降低 harm rate（0.573→0.376）并保持 ΔBA，但未完全满足 G3。
3. **detector-dependent**：正收益主要由 DINOv2 驱动；PatchCore/EfficientAD safe headroom 极小。任何正 claim 必须限定为 detector-dependent。
4. **oracle 上界仅诊断**：ORACLE（mean ΔBA +0.110）为 DIAGNOSTIC ONLY，不是 deployable method。
5. **K8 边际**：T2 在 K8 的 mean ΔFNR = +0.0214 仍略超 +0.02 界；K8 下安全边际不足。
6. **MVTec AD 2 为外部确认证据**：本审计仅使用已冻结的 PB6 score caches，未新增数据访问。
7. **NOT_VERIFIED**：未评估任何复杂 classifier（本阶段禁止）；未评估跨数据集泛化；未评估真实部署中的 acquisition metadata 可用性。

---

## 10. 下一授权工作

**AUTHORIZATION_FROM_THIS_DOCUMENT = NONE。**

本审计的 terminal（F4）允许后续 actuator-development cycle，但任何后续工作（例如 detector-aware sign rules、保守/选择性修正 + 显式 FNR bound、abstention policy、新的 actuator 设计）都需要：
1. 用户的明确授权；
2. 一份新的冻结协议（新 candidate 必须在 outcome 前注册）；
3. 继续遵守 DEPLOYABLE / ORACLE 防火墙与禁止事项（不得使用 target anomaly label 构造 deployable predictor、不得 outcome-driven 调参、不得修改 BENEFIT 定义）。

---

## 11. 证据清单

| 阶段 | Artifact | Gate |
|---|---|---|
| P0 | `P0_PROTOCOL.json`、`P0_INPUT_MANIFEST.csv`、`P0_DATA_ROLES.json`、`P0_ESTIMAND_REGISTRY.json`、`P0_HYPOTHESIS_REGISTRY.json`、`P0_GATE.json`、`P0_GIT_STATE.txt` | PASS |
| A | `A1_SOURCE_OPERATING_POINT.csv`、`A2_ORACLE_THRESHOLD_CURVES.csv`、`A2_ORACLE_HEADROOM.csv`、`A3_DETECTOR_HEADROOM_SUMMARY.csv`、`A4_CORRECTABILITY_TAXONOMY.csv`、`A_GATE.json`、`A_SELFCHECK.json` | PASS |
| B | `B_FEATURES.csv`、`B_UNIVARIATE_IDENTIFIABILITY.csv`、`B_TRIVIAL_BASELINES.csv`、`B_GATE.json`、`B_SELFCHECK.json` | PASS（PASS-B） |
| C | `C_FULL_FEATURES.csv`、`C_SAMPLED_FEATURES.csv`、`C_K_STABILITY.csv`、`C_GATE.json` | PASS |
| D | `D_CANDIDATES.json` | 冻结 |
| E | `E_PER_ENTITY.csv`、`E_DETECTOR_SUMMARY.csv`、`E_BOOTSTRAP.csv`、`E_RANKING.csv`、`E_GATE.json` | **FAIL（G3）** |
| F | `F_THRESHOLD_MECHANISM_VERDICT.json` | **F4 封存** |
| FINAL | `PB7_P0_THRESHOLD_CORRECTABILITY_FINAL_REPORT.md`、`P0_EXECUTION_LEDGER.csv` | COMPLETE |
