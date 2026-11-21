# Method — NorOpCal

NorOpCal (**Nor**mal-Only **S**elective **Op**erating-Point **Cal**ibration)
recalibrates only the decision threshold of a frozen anomaly detector under
acquisition shift using $K$ known-normal target observations ($K \approx 8$),
without requiring target anomalies, labels, pixel masks, or fine-tuning the detector.

> [!NOTE]
> **公开版本与科研资产说明 (Public Release Notice)**  
> 历史研究全量工程中的内部参考记录（如 `docs/reference/PUBLICATION_GROUND_TRUTH.md`）以及内部重放脚本（如 `tools/pb7_p1_safety_actuator/pb7_p1_common.py`、`tools/pb7_p0_threshold_correctability/pb7_p0_common.py`）属于内部科研冻结资产，**未随精简开源发布包分发**。  
> 本开源仓库通过 `src/noropcal` 提供了核心算法的完全自包含实现，并内嵌了权威冻结参数规范（可通过 `noropcal.load_frozen_spec()` 加载验证）。

---

## Deployment Information Set

NorOpCal 严格遵循工业在线低开销部署约束，其可用信息集仅为：
$$(\tau_s, S_s^N, S_t^N)$$
- $\tau_s$：源域标定的静态决策阈值
- $S_s^N$：源域正常样本异常得分集合
- $S_t^N$：目标域仅有的 $K$ 张已知正常样本异常得分（支持集，$K \approx 8$）

系统不使用任何目标域缺陷样本、像素掩码、先验分布或在线测试反馈。

---

## Frozen Policy Specification (权威冻结规范)

1. **分布漂移感知 (Median-Shift Sensing)**：  
   $$m_s = \text{median}(S_s^N), \quad m_t = \text{median}(S_t^N), \quad \Delta m = m_t - m_s$$
2. **可纠正性门控 (Correctability Gate, $Q_{50}$)**：  
   仅当中位数漂移量满足 $\Delta m \ge Q_{50}$ 时触发阈值校准，其中冻结分位阈值：
   $$Q_{50} = 0.015890181064605713$$
   若 $\Delta m < Q_{50}$，则**主动弃权 (Abstain)**，保持原静态阈值 $\tau_t = \tau_s$。
3. **保守有界执行器 (Conservative Bounded Actuator)**：  
   门控通过后，实施受约束的有界收缩平移：
   $$\tau_t = \tau_s + \lambda \cdot \Delta m \quad (\text{冻结收缩系数 } \lambda = 0.25)$$
4. **安全保护理念**：  
   在微弱漂移或无漂移工况下，门控主动拦截并保持源域阈值是精心设计的安全保障行为，可有效防止过度调整带来的负迁移（Negative Transfer）。

---

## What NorOpCal is NOT

- **不是领域自适应 (Domain Adaptation)**：不微调特征提取主干，不改变特征空间。
- **不是检测器重训练**：主干权重、记忆库与异常打分函数完全冻结。
- **不改变排序性能**：阈值平移属于单调后处理，不改变测试样本的相对排名，因此受试者工作特征曲线下面积（AUROC/AUPR）在数学上严格保持不变。

---

## Detector Scope & Scientific Findings (检测器适用边界)

根据论文正式冻结评测，阈值自适应校准的有效性**强依赖于底层特征空间的几何与漂移特性**，并非所有模型普遍有效：

| 检测器主干 (Detector) | 评测角色 (Role) | 科学评测结论 (Verdict) | 详细原因 |
|---|---|---|---|
| **DINOv2** (`dinov2_vits14`) | 核心主评测主干 | **USEFUL_SAFE_SELECTABLE** | 特征分布呈现刚性平移，门控与有界收缩协同生效，取得确定性正向收益 |
| **PatchCore** (`wide_resnet50_2`) | 机制对照主干 | **ACTUATOR_MISMATCH** | 特征空间漂移非刚性，单纯中位数平移产生动作器失配 |
| **EfficientAD** (70,000 steps) | 机制对照主干 | **NO_HEADROOM** | 基线模型本身在目标工况下无显著阈值调整余量 |

> [!WARNING]
> 正向校准收益属于**特定检测器与特征空间下的实证发现**，绝不能作为跨检测器普适的通用声明。

---

## Principal Result (核心实证结论)

在 DINOv2 主干网络、$K=8$ 目标域正常样本支持集、**10 个符合条件的 TEST 评估单元**（fabric 5 个工况 + walnuts 5 个工况）上的冻结评测数据为：
- 平衡准确率净增益 ($\Delta\text{BA}$)：**+3.23 pp**（类别感知 Block-Bootstrap 95% 置信区间 $[+2.70, +3.76]\text{ pp}$）
- 漏检率增量 ($\Delta\text{FNR}$)：**+0.24 pp**（严格控制在预注册 $+2.0\text{ pp}$ 安全预算内）
- 决策危害率 (Harm Rate)：**3.67%**（相比无门控的 42.1% 得到显著压制）
- 策略覆盖率 (Coverage)：**40.33%**

---

## Known Scientific Limitations

1. **残余危害集中性**：残余的 3.67% 决策恶化主要集中在源域初始 FNR 偏高的个别 walnuts 工况单元；算法不允许引入任何后验品类补丁规则。
2. **样本量资格门槛**：$K=8$ 要求目标域具备至少 8 张独立的正常品图像；在 MVTec AD 2 的 26 个 TEST 单元中，仅有 10 个单元满足此条件，其余 16 个单元存在真实的数据量短缺。
3. **外部数据集诚实弃权**：在外部数据集（如 AeBAD-S）上，由于工况极端偏移导致门控退化为全触发，研究明确记录了诚实放弃（Abstention），不宣称外部泛化。
