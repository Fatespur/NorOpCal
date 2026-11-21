# NorOpCal

**工业视觉异常检测在采集域偏移下的选择性操作点（阈值）自适应校准框架**  
*Normal-Only Selective Operating-Point Calibration for Industrial Visual Anomaly Detection under Acquisition Shift*

<p align="left">
    <a href="README_EN.md">English</a> •
    <a href="README.md"><b>简体中文</b></a>
</p>

[![CI](https://github.com/Fatespur/NorOpCal/actions/workflows/ci.yml/badge.svg)](https://github.com/Fatespur/NorOpCal/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

NorOpCal 是一种轻量、即插即用的工业异常检测决策阈值校准机制。针对部署后的工业无监督异常检测模型在产线**采集域分布偏移**（如成像工况改变、传感器老化、环境光照漂移）下导致的虚警与漏检激增问题，NorOpCal **仅需目标域极少量的正常样本**（$K \approx 8$ 张无缺陷图像），即可保守、选择性地自适应重校准模型的**操作点（决策阈值）**。

整个过程**无需目标域缺陷样本、无需人工标签、无需像素级掩码，且完全无需微调或重新训练检测器主干网络**。

![NorOpCal 框架总览](assets/figures/Fig1_系统框架图.png)

---

## 核心痛点与方法原理

在工业视觉检测中，部署后的异常检测器阈值 $\tau_s$ 通常在源域（标准工况）下标定。当产线环境出现**采集域偏移**时，目标域样本的得分分布整体漂移，导致漏检率（FNR）或误报率陡增。而重新采集大量缺陷样本进行微调成本极高且严重影响产线节拍。

NorOpCal 将该问题形式化为**仅基于正常样本的操作点安全校准机制**，核心包含三个连续阶段：

1. **中位数漂移感知（Median-Shift Sensing）**：  
   计算源域与目标域正常样本得分分布的中位数位移信号：
   $$\Delta m = \text{median}(S_{\text{target, normal}}) - \text{median}(S_{\text{source, normal}})$$
2. **可纠正性门控（Correctability Gate, $Q_{50}$）**：  
   仅当检测到的分布漂移显著大于预注册的稳健阈值时才触发校准（$\Delta m \ge Q_{50}$，其中冻结分位数 $Q_{50} = 0.01589$）；若未达标，则**主动放弃校准（Abstain）**，保持源域静态阈值 $\tau_s$，彻底防止微弱漂移下的盲目校准带来负迁移危害。
3. **保守有界执行器（Conservative Bounded Actuator）**：  
   在门控通过后，实施受约束的收缩自适应更新：
   $$\tau_t = \tau_s + \lambda \cdot \Delta m \quad (\text{预注册冻结系数 } \lambda = 0.25)$$

> **核心设计理念**：检测器的特征提取器、记忆库与样本异常度排序完全保持冻结。**“主动放弃校准”是本系统精心设计的安全保护行为，而非算法失效。**

---

## 极速上手

### 1. 环境安装

```bash
git clone https://github.com/Fatespur/NorOpCal.git
cd NorOpCal

# 创建虚拟环境（可选）
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
```

### 2. Python API 快速调用

```python
import numpy as np
from noropcal import NorOpCal

# 初始化校准器（自动加载冻结推荐参数: lambda=0.25, Q50 门控）
calibrator = NorOpCal()

# 准备源域正常得分与目标域少量正常得分 (例如 K=8)
source_normal_scores = np.array([0.12, 0.15, 0.14, 0.18, 0.11])
target_normal_scores = np.array([0.31, 0.33, 0.29, 0.35])
tau_source = 0.25  # 源域初始操作阈值

# 自适应计算目标域最佳决策阈值
tau_target = calibrator.threshold(source_normal_scores, target_normal_scores, tau_source)
print(f"自适应决策阈值校准结果: {tau_source:.4f} -> {tau_target:.4f}")
```

### 3. 运行交互式演示

```bash
python scripts/demo.py
```

### 4. 自动化单元测试

```bash
pytest tests/
```

---

## 核心实验结论与适用边界

### 1. 核心实证结论 (Principal Result)

依据论文正式冻结的研究结果，NorOpCal 的核心性能增益是在 **DINOv2 主干网络**、**$K=8$ 正常样本支持集**、以及 **10 个符合评估资格的 TEST 单元**（`fabric` 5 个工况 + `walnuts` 5 个工况）上测得，**并非跨所有检测器的普遍收益**：

| 策略方案 (Policy) | $\Delta$ 平衡准确率 (BA, pp) | $\Delta$ 漏检率 (FNR, pp) | 危害率 (Harm Rate, %) | 覆盖率 (Coverage, %) |
|---|---|---|---|---|
| **不校准 (NoAdapt, $\tau_s$)** | 0.00 (基准; BA 0.5411) | 0.00 | 0.00 | 0.00 |
| **仅有界收缩 ($\lambda=0.25$, 无门控)** | +3.64 [+3.52, +3.76] | −0.16 | 3.67 | 100.00 |
| **仅门控机制 ($\lambda=1.0$, Q50)** | +4.25 [+3.96, +4.55] | +7.49 | 37.67 | 40.33 |
| **NorOpCal 全策略 ($\lambda=0.25$, Q50)** | **+3.23 [+2.70, +3.76]** | **+0.24** | **3.67** | **40.33** |

*注：区间为类别感知 Block-Bootstrap (B=10,000) 95% 置信区间。漏检率增量严格控制在预注册 $+2.0\text{ pp}$ 安全预算内。*

### 2. 检测器适用性边界 (Detector Scope)

实验表明，阈值自适应校准的有效性高度取决于特征空间的几何结构与漂移规律：

| 检测器 (Detector) | 评测角色 | 判定结论 (Verdict) | 科学解释 |
|---|---|---|---|
| **DINOv2** (`dinov2_vits14`) | 核心主干 | **USEFUL_SAFE_SELECTABLE** | 特征分布呈现刚性平移，中位数位移有效表征工况漂移，取得确定性安全增益 (+3.23 pp)。 |
| **PatchCore** (`wide_resnet50_2`) | 机制对照 | **ACTUATOR_MISMATCH** | 记忆库邻域漂移具有非各向同性，标量平移执行器存在机制失配。 |
| **EfficientAD** (70,000 steps) | 机制对照 | **NO_HEADROOM** | 基线模型在偏移工况下由于网络结构特性，无可提升余量。 |

### 3. 数据集与样本量资格说明

- **基准数据集**：MVTec AD 2 共包含 8 个品类（`can`, `fabric`, `fruit_jelly`, `rice`, `sheet_metal`, `vial`, `wallplugs`, `walnuts`）合计 39 个工况单元。其中 DEV 集包含 13 个单元，TEST 集包含 26 个单元。
- **$K=8$ 资格限制**：开展 $K=8$ 校准要求目标域至少包含 8 张独立无缺陷正常图像。在 26 个 TEST 单元中，仅 `fabric`（5 个单元）与 `walnuts`（5 个单元）共 **10 个单元**满足资格；其余 16 个单元存在真实的数据短缺，不满足评估条件。

---

## 机制分析与概念示意

| 策略消融分析 (DINOv2 实测) | 安全门控消融与危害率压制 (实测) |
|:---:|:---:|
| ![Fig 2 策略消融权衡](assets/figures/Fig2_帕累托风险增益前沿.png) | ![Fig 4 安全门控消融](assets/figures/Fig4_安全门控消融与危害率压制.png) |

| 工况漂移与决策阈值校准原理【概念示意图】 |
|:---:|
| ![Fig 5 决策边界概念示意](assets/figures/Fig5_决策边界与工况漂移对比.png) |

> *说明：Fig 2 与 Fig 4 展示 DINOv2 在 10 个测试单元上的冻结消融实测数据；Fig 5 为基于合成高斯分布的方法原理概念示意图（已剔除任何夸大绝对安全断言）。原包含虚构跨模型数据的图表已被撤下，以维护学术严谨性。*

---

## 项目目录架构

```text
NorOpCal/
├── .github/
│   └── workflows/ci.yml  # 自动化持续集成流水线 (CI)
├── assets/
│   └── figures/          # 系统框架图与概念示意图
├── configs/              # 检测器与协议配置文件
├── docs/                 # 详细学术与数据协议规范
│   ├── METHOD.md         # 算法数学推导、门控设计与适用边界
│   ├── DATASETS.md       # 数据集协议、工况划分与资格审查
│   └── KNOWN_ISSUES.md   # 科研边界与工程考量
├── scripts/
│   └── demo.py           # 30 秒开箱即用演示脚本
├── src/
│   ├── noropcal/         # NorOpCal 核心自包含校准算法与 API 接口
│   └── shiftad/          # 异常检测器适配与主干支撑模块
├── tests/                # 自动化单元测试套件 (pytest)
├── pyproject.toml        # 标准 Python 打包与构建配置
└── requirements.txt      # 依赖清单
```

---

## 引用声明

如果您在科研或项目中参考了本项目，请按如下格式引用：

```bibtex
@software{noropcal2026,
  author = {Fatespur},
  title = {NorOpCal: Normal-Only Selective Operating-Point Calibration for Industrial Visual Anomaly Detection under Acquisition Shift},
  url = {https://github.com/Fatespur/NorOpCal},
  year = {2026}
}
```
