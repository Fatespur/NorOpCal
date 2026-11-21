# Configuration Specifications

本目录保留了科研开发过程中检测器与实验配置的声明结构，方便审查与扩展。

| 配置路径 | 逻辑作用 | 公开状态说明 |
|---|---|---|
| `models/*.yaml` | **检测器模型构建参数**：`dino_memory.yaml` (DINOv2 主评测主干)、`patchcore_default.yaml`、`efficientad_default.yaml` | 已开源公开 |
| `experiments/*_fixture.yaml` | 用于轻量级单元测试的合成夹具配置（如 `dino_memory_fixture.yaml`） | 已开源公开 |
| `protocols/noropcal_frozen_protocol.yaml` | NorOpCal 冻结科研规范与超参数的**声明式镜像记录** | 已开源公开（权威值可通过 `noropcal.load_frozen_spec()` 载入） |
| `examples/dataset_roots.example.yaml` | 本地自备数据集路径的环境变量配置示例 | 已开源公开 |

> [!NOTE]
> **历史内部路径说明 (Internal Provenance Notice)**  
> 历史研究中涉及的部分全量数据集适配脚本、全量历史权重以及内部训练模板（如历史上引用的 `tools/pb6/*`、`tools/pb7_*`、`reports/pb7_*` 等）属于内部封闭科研档案，未包含在精简开源发布包中。开源发布包专注于 NorOpCal 校准算法本身及其可直接测试的轻量级资产。
