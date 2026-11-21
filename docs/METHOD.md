# Method — NorOpCal

NorOpCal (**Nor**mal-Only **S**elective **Op**erating-Point **Cal**ibration)
recalibrates only the decision threshold of a frozen anomaly detector under
acquisition shift using $K$ known-normal target observations ($K \approx 8$),
without requiring target anomalies, labels, pixel masks, or fine-tuning the detector.

> [!NOTE]
> **公开版本与科研资产说明 (Public Release Notice)**  
> 历史研究全量工程中的内部参考记录（如 `docs/reference/PUBLICATION_GROUND_TRUTH.md`）以及内部重放脚本（如 `tools/pb7_p1_safety_actuator/pb7_p1_common.py`）属于内部科研封闭存档，**未随精简开源发布包分发**。  
> 本开源仓库通过 `src/noropcal` 提供了核心算法的完全自包含实现，并内嵌了权威冻结参数规范（可通过 `noropcal.load_frozen_spec()` 载入）。

---

## 科研复现层级与资产边界

为维护学术严谨性，本项目明确区分三类复现范畴：

1. **Level 1：核心算法与 API 复现（独立运行）**  
   依托本仓库自包含代码，通过 `python -m pytest -v tests/` 验证核心类 `NorOpCal`、冻结超参数载入以及边界异常拦截。
2. **Level 2：原理机制与合成验证（独立运行）**  
   依托 `scripts/demo.py` 与 `assets/figures/generate_all_figures.py`，在合成高斯分布下验证中位数感知、有界收缩与门控弃权机制。
3. **Level 3：论文全量实验复现（依赖外部资产）**  
   涉及 MVTec AD 2 原始高分辨率图像、DINOv2/PatchCore/EfficientAD 的全量特征抽取与重算。由于数据属于受控资产，本仓库未包含全部原始数据与权重，通过 Level 1/2 不等同于直接完成 Level 3 复现。

---

## 形式化数学定义与操作流程

工业在线部署的信息集严格限定为：
$$(\tau_s, S_s^N, S_t^N)$$
- $\tau_s$：源域标定的静态决策阈值
- $S_s^N$：源域正常样本异常得分集合
- $S_t^N$：目标域仅有的 $K$ 张已知正常样本异常得分（支持集，$K \approx 8$）

### 三阶段校准流程

1. **分布漂移感知 (Median-Shift Sensing)**：  
   $$m_s = \text{median}(S_s^N), \quad m_t = \text{median}(S_t^N), \quad \Delta m = m_t - m_s$$
2. **可纠正性门控 (Correctability Gate, $Q_{50}$)**：  
   仅当中位数漂移量满足 $\Delta m \ge Q_{50}$ 时触发阈值校准，其中冻结分位阈值：
   $$Q_{50} = 0.015890181064605713$$
   若 $\Delta m < Q_{50}$，则**主动弃权 (Abstain)**，保持原静态阈值 $\tau_t = \tau_s$。
3. **保守有界执行器 (Conservative Bounded Actuator)**：  
   门控通过后，实施受约束的有界收缩平移：
   $$\tau_t = \tau_s + \lambda \cdot \Delta m \quad (\text{冻结收缩系数 } \lambda = 0.25)$$

---

## 数学保证 vs. 统计假设 vs. 实测现象

为避免机理归因泛化与夸大，特别强调以下三层区别：

| 范畴 | 内容 | 边界与严谨表述 |
| :--- | :--- | :--- |
| **数学保证 (Mathematical Invariant)** | 单调阈值平移 | 阈值调整仅改变样本的正负判决界限，样本间的相对异常排序严格保持不变，因此受试者工作特征曲线下面积（AUROC）在数学上严格守恒。 |
| **统计设计 (Statistical Policy)** | 安全预算约束 | 预注册设计要求 $\mathbb{E}[\Delta FNR] \le +0.02$。该约束属于**指定测试总体上的平均 FNR 退化预算**，**不构成对单个工况单元或单个孤立样本的个体安全保证**。 |
| **实测现象与假说 (Observation & Hypothesis)** | 检测器适用性差异 | 实测现象表明校准有效性强依赖于主干架构；机制解释属于学术假设，不代表确定性因果证明（见下表）。 |

### 检测器适用性：实测现象 vs. 机制假设

| 检测器主干 (Detector) | 评测角色 | 判定结论 (Verdict) | 实测现象 (Observation) | 机制假设 (Hypothesis) |
|---|---|---|---|---|
| **DINOv2** (`dinov2_vits14`) | 核心主干 | **USEFUL_SAFE_SELECTABLE** | 在 10 个测试单元获得 +3.23 pp 显著净增益，危害率受控在 3.67%。 | 推测其深层自监督特征空间在成像漂移下表现出近似刚性平移，标量中位数对齐有效。 |
| **PatchCore** (`wide_resnet50_2`) | 机制对照 | **ACTUATOR_MISMATCH** | 阈值平移导致漏检或误报剧烈波动，表现出机制失配。 | 推测记忆库最近邻检索受局部邻域非各向同性形变影响，简单的标量平移无法匹配真实分界面。 |
| **EfficientAD** (70,000 steps) | 机制对照 | **NO_HEADROOM** | 目标域下基线模型无可调增益空间。 | 推测学生-教师网络在该工况下的预测残差分布具有特有收敛特性，阈值调节无增益空间。 |

---

## 核心消融实证与机制归因

在 DINOv2 主干网络、$K=8$ 正常支持集、10 个符合条件的 TEST 单元上的正式消融结果：

| 策略方案 (Policy) | $\Delta$ 平衡准确率 (BA, pp) | $\Delta$ 漏检率 (FNR, pp) | 危害率 (Harm Rate, %) | 覆盖率 (Coverage, %) |
|---|---|---|---|---|
| **不校准 (NoAdapt, $\tau_s$)** | 0.00 (基准; BA 0.5411) | 0.00 | 0.00 | 0.00 |
| **仅门控机制 ($\lambda=1.0$, Q50)** | +4.25 [+3.96, +4.55] | +7.49 | 37.67 | 40.33 |
| **仅有界收缩 ($\lambda=0.25$, 无门控)** | +3.64 [+3.52, +3.76] | −0.16 | 3.67 | 100.00 |
| **NorOpCal 全策略 ($\lambda=0.25$, Q50)** | **+3.23 [+2.70, +3.76]** | **+0.24** | **3.67** | **40.33** |

### 机制归因关键结论：
1. **有界收缩是控制决策危害率的核心**：将激进更新（$\lambda=1.0$）高达 37.67% 的危害率大幅压制到 3.67%（降低 34 个百分点）。
2. **门控机制并未进一步降低危害率**：仅有界收缩与全策略的危害率均为 3.67%。门控的核心价值在于**选择性触发（40.33% 覆盖率）与弃权保护**，避免在微弱漂移工况下无效扰动决策阈值，并将平均漏检率控制在预注册安全预算内。
