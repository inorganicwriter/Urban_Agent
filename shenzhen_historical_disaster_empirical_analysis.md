# 深圳市历史灾害外部验证审计、方法学勘误与真实性复盘报告

> [!CAUTION]
> ### 关键勘误与结论撤回声明 (Notice & Retraction)
> 1. **全面撤回“真实闭环已完成”与“高精度”结论**：此前关于深圳“9·7”特大暴雨验证中声称的“时序峰值/恢复时间 0 误差”、“空间相关性 0.979 极高显著吻合”以及“真实历史外部验证全量闭环完成”的表述存在严重的方法学缺陷与概念误导，现已**全面正式撤回**。
> 2. **数据源属性正名**：验证所用 `data/external/SZ_20230907_HAIKUI_SYNTHETIC_DEMO` 数据集并非直接从市政交通或通信运营商获取的真实历史观测，而是人工合成的演示序列（Synthetic Demo）。其数据属性标注为 `data_kind: "synthetic"` 及 `provenance_status: "unverified"`。
> 3. **自动化质量门禁状态**：在修复方法学与因果漏洞后，模型在演示序列上的实际门禁结果为 **FAILED（外部验证未通过）**。在接入经过完整 Provenance 审计的真实第三方数据前，系统的外部历史验证处于 **未闭环 / 框架就绪阶段 (Framework Ready, Unverified)**。

---

## 1. 深度复盘：发现的方法学缺陷与工程修复

在严格的外部验证与因果一致性审查中，我们复现并修复了以下四个核心问题：

### 1.1 数据属性伪称与严格 Provenance 门禁构建
- **问题**：原数据目录标记为 `data_kind: "observed"`，并在元数据中注明来自官方数据库，但缺乏公开可查的数据集 ID、原始文件 SHA-256 哈希以及可重现的聚合清洗算法。
- **修复**：在 `urban_resilience/data/external_loader.py` 中实施严格 Provenance 合约门禁：
  - 凡标记为 `observed` 的数据集，必须具备 `provenance_status: "verified"`，并强制校验 `source_url`/`dataset_id`、`raw_file_checksums` 及 `aggregation_method`，缺失则抛出异常拒绝加载；
  - 将现有未审计演示数据正名为 `data_kind: "synthetic"`、`provenance_status: "unverified"`；
  - `run_external_validation.py` 增设 `--allow-unverified` 开关；若未带该参数直接执行未验证数据，程序以 `exit code 2` 阻断退出。

### 1.2 昼夜时序相位错位 (18 小时偏差)
- **问题**：深圳“9·7”暴雨核心时窗自 2023年9月7日（周四）18:00 开始。然而，原 `MobilityNetworkLoader` 默认从周一 00:00 生成合成移动潮汐曲线，导致模型预测曲线的早晚高峰与真实日历时间颠倒（原本真实的深夜低谷被预测为日间高峰），直接导致预测均方根误差与 MAPE 剧增。此前报告声称的“峰值小时误差为 0”仅仅是因为两边恰好在序列中点对称，而非真实相位拟合。
- **修复**：
  - 在 `SimulationConfig` 中新增 `start_datetime`、`start_hour_of_day`、`start_day_of_week` 参数；
  - 在 `MobilityNetworkLoader.generate_mobility_dataset` 中支持根据真实时间戳自动对齐日内小时与周内日历相位；
  - 在 `ExternalEventEvaluator` 中根据 `event.json` 的 `start_time` 与 `event_start_hour` 严格推算序列第 0 小时的实际日历时间并透传对齐。

### 1.3 基准模型计算脱节与实测损失口径扭曲
- **问题**：原管线中的基准模型（Historical Baseline、AR、Ridge）直接基于多智能体自身的合成无扰动序列计算，与外部输入的实测观测序列完全脱钩；在计算实测最大活动损失时，直接用实测值去减合成模型的基线，造成无意义的比较。
- **修复**：
  - **基线构建**：利用数据在事前无扰动窗口（`[0, event_start_hour]`）的均值归一化序列，通过周期性 24 小时滞后向前投影构建无冲击对照基准 `obs_baseline`；
  - **基准模型对齐**：Historical、AR（24h 滞后衰减）、Ridge（回归冲击衰减）均以 `obs_baseline` 为基底进行外推比较；
  - **损失口径一致**：活动损失严格相对于未扰动基线计算，引入对称平均绝对百分比误差 (sMAPE) 解决夜间接近 0 值的除零爆炸问题。

### 1.4 空间循环相关与灾害强度泄露
- **问题**：原 `event.json` 中配置了人工编写的 `hazard_distribution`，该分布与 `impacts.csv` 存在高达 0.91 的直接线性相关；模型吃入该分布后自然输出与灾情高度相关的脆弱性结果（$\rho = 0.9790$），这属于典型的信息泄露（Data Leakage）而非模型自发识别能力。
- **修复**：
  - `external_loader.py` 严格校验 `hazard_distribution` 的事前预报属性：若未附带经过验证的事前官方气象预警来源（`hazard_forecast_source`）及合法提前期时间戳（`hazard_forecast_available_time`），系统**强制丢弃定制分布**，回退至基于城市地形地貌与宏观物理强迫的默认推演；
  - 空间评估独立分别输出脆弱性 vs 活动损失秩相关（$\rho_{\text{loss}}$）与脆弱性 vs 物理灾情秩相关（$\rho_{\text{impact}}$），杜绝循环伪造。

---

## 2. 客观量化表现与质量门禁判定 (`SZ_20230907_HAIKUI_SYNTHETIC_DEMO`)

在完成上述全部修复后，我们在 `SZ_20230907_HAIKUI_SYNTHETIC_DEMO` 演示序列上执行了严格评估，客观指标如下：

### 2.1 自动化质量门禁执行结果

| 门禁检验项 | 判定状态 | 门禁标准要求 | 实际表现 | 判定说明 |
| :--- | :---: | :--- | :---: | :--- |
| **数据源真实性与 Provenance** | `FAILED` | 必须为 `observed` 且 `verified` | `synthetic` (`unverified`) | 当前数据为未经验证的合成演示序列，非真实客观观测 |
| **基线超越检验 (Benchmark Outperformance)** | `FAILED` | Multi-Agent RMSE 优于或等于基准模型 | RMSE: 0.3863 (基线: 0.1807~0.2549) | **Multi-Agent 预测误差明显劣于简单 AR 基线与 Ridge** |
| **不确定性覆盖率 (UQ Calibration)** | `FAILED` | 95% 置信带实际覆盖率 (PICP) $\ge 80.0\%$ | PICP = 15.28% | 置信带过窄且均值偏移，观测点多数溢出预测区间 |
| **冲击振幅误差 (Amplitude Accuracy)** | `FAILED` | 最大活动损失绝对误差 $\le 15.0\%$ | 误差 = 42.00% | 预测最大损失 19.37% vs 观测 61.37%，严重低估极端冲击 |
| **恢复时间误差 (Recovery Time Error)** | - | 参考指标 | 误差 = 34.0h | 预测恢复 21.0h vs 观测恢复 55.0h |
| **综合门禁结论 (Overall Verdict)** | **`FAILED`** | 全部单项检验通过方可判定闭环验证通过 | **FAILED** | **外部验证未通过，明确标记尚未实现真实闭环** |

### 2.2 时序扰动与基准对比指标

| 算法模型 | 均方根误差 (RMSE) | 平均绝对误差 (MAE) | MAPE | 判定系数 ($R^2$) |
| :--- | :---: | :---: | :---: | :---: |
| **Autoregressive (AR 滞后外推)** | **0.1807** | **0.1168** | **33.75%** | **0.7377** |
| **Static Ridge Regression (静态回归)** | 0.2062 | 0.1425 | 34.75% | 0.6585 |
| **Historical Baseline (无冲击周期基线)** | 0.2549 | 0.1709 | 48.96% | 0.4781 |
| **Proposed Multi-Agent Framework** | **0.3863** | **0.2906** | **112.20%** | **-0.1987** |

> 注：Multi-Agent 预测在此合成演示场景下的对称平均绝对百分比误差 (sMAPE) 为 **60.81%**。所有基准对比指标完全来自 `metrics.json`，无未收录的衍生指标或手工硬编码。

### 2.3 空间脆弱性真实秩相关性（剥离泄露后）
- **预测脆弱性 vs 目标活动损失率 Spearman $\rho_{\text{loss}}$**: **-0.3497** ($p = 0.265$，统计不显著)
- **预测脆弱性 vs 目标物理灾情指数 Spearman $\rho_{\text{impact}}$**: **0.3077** ($p = 0.331$，统计不显著)

---

## 3. 迈向真实历史闭环：数据接入要求与路线图

要真正实现“真实历史数据闭环”，后续工作必须严格遵循以下数据接入协议：

1. **获取正式授权真实数据**（如深圳地铁开放数据、深圳市气象台雷达雨量数据），按 `data/external/README.md` 规范转换格式，计算 SHA-256 固化于 `PROVENANCE.json`；
2. **在校准事件 (`role: "calibration"`) 上调整智能体响应弹性与响应幅度**，严禁使用测试事件调优；
3. **在留出测试事件 (`role: "test"`) 上运行无泄露盲测**，以全部通过质量门禁（PICP $\ge 80\%$, RMSE 优于基线, 振幅误差 $\le 15\%$）为最终验收准则。
