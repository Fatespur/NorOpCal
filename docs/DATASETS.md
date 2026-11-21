# Datasets Protocol & Specification

## 1. MVTec AD 2 (AD2) 工业基准数据集

论文评测基于工业无监督异常检测基准数据集 **MVTec AD 2**（涵盖 8 类工业品、39 个采集工况单元）：

- **8 大工业品类**：`can`, `fabric`, `fruit_jelly`, `rice`, `sheet_metal`, `vial`, `wallplugs`, `walnuts`。
- **工况单元 (Acquisition Unit)**：按“品类 × 成像工况”划分（如 `vial` 包含 6 种工况，`fruit_jelly` 包含 3 种，其他品类各 5 种，合计 39 单元）。
- **评估切分**：从公开测试切分读取图像级异常得分，`test_public/good` 提供目标域正常样本支持集，`test_public/bad` 提供缺陷测试样本池。

> [!IMPORTANT]
> **数据集分发与获取说明**  
> **本仓库不附带分发任何原始图像数据集**。用户需遵循原数据集许可协议自行获取 MVTec AD 2 数据，并通过环境变量指定本地路径：
> ```bash
> # Windows PowerShell
> $env:NOROPCAL_MVTEC_AD2_ROOT = "D:\path\to\mvtec_ad_2"
> 
> # Linux / macOS Bash
> export NOROPCAL_MVTEC_AD2_ROOT="/path/to/mvtec_ad_2"
> ```

### 预期数据目录结构规范

```text
<NOROPCAL_MVTEC_AD2_ROOT>/
  <category>/
    train/good/*.png                    # 源域正常训练样本 (标准工况)
    test_public/good/<cond>_<idx>.png   # 目标域偏移工况正常样本
    test_public/bad/<cond>_<idx>.png    # 目标域偏移工况缺陷样本
```

> [!NOTE]
> 历史研究中用于自动枚举工况前缀的数据分析脚本（如 `tools/pb6/pb6_entry_determination.py`）属于内部研发历史资产，未包含在精简开源发布包中。

---

## 2. 冻结 DEV / TEST 切分协议

数据集品类级划分采用权威冻结协议（基于品类名 SHA256 校验和确定性切分）：

- **开发集 (DEV Split, 仅供超参选定)**：`can`, `fruit_jelly`, `wallplugs`（共 13 个工况单元）。
- **测试集 (TEST Split, 仅供最终评测)**：`fabric`, `rice`, `sheet_metal`, `vial`, `walnuts`（共 26 个工况单元）。

### $K=8$ 支持集资格限制 (Eligibility Census)

- 依据协议，开展 $K=8$ 自适应校准的前提是目标域至少包含 8 张独立的无缺陷正常品图像。
- 在 TEST 集的 26 个工况单元中，仅有 **10 个工况单元**具备充足的正常品图像：
  - `fabric`：5 个工况单元均符合资格
  - `walnuts`：5 个工况单元均符合资格
- 其余 16 个工况单元存在真实的数据量短缺（正常样本 $<8$ 张），按预注册协议不计入 $K=8$ 最终评估池。

---

## 3. 外部数据集声明 (External Datasets)

在早期的路线探索中曾尝试引入 AeBAD / AeBAD-S 工业机载数据集。但研究发现该数据集的背景与光照漂移极大，导致固定门控 $Q_{50}$ 退化为常开状态。  
为坚持严谨科研原则，论文明确记录了**对外部数据集的诚实弃权 (Honest Abstention)**，因此开源包中不包含 AeBAD 数据适配与模型缓存。
