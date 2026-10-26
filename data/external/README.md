# 外部真实事件与移动数据验证数据规范 (External Validation Data Contract)

> [!CAUTION]
> ### 重要声明与事实勘误 (Notice & Retraction)
> 1. **数据性质勘误**：此前演示所使用的 `data/external/SZ_20230907_HAIKUI` 序列为**合成演示数据 (Synthetic Demo)**，非真实经过审计的历史观测。现已规范重命名为 `data/external/SZ_20230907_HAIKUI_SYNTHETIC_DEMO`，标注为 `data_kind: "synthetic"` 及 `provenance_status: "unverified"`。
> 2. **结论撤回**：收回任何声称“已完成真实历史数据高精度验证/外部闭环成功完成”的表述。在未接入真实可信的第三方历史观测数据前，外部验证结论严格处于 **未验证 / 测试桩阶段 (Unverified / Demo Stage)**。
> 3. **防循环泄露约束**：外部事件严禁在缺乏权威气象预报源与发布时间戳的前提下输入人工定制的 `hazard_distribution`；实测灾情数据 (`impacts.csv`) 仅用于后验空间秩相关计算，严禁输入模型。

---

## 1. 核心治理原则

1. **真实性与严禁伪造 (No Synthetic Masquerading)**：
   - 严禁将合成生成的数据伪装为真实观测数据。
   - 数据属性 (`data_kind`) 严格分为：
     - `"observed"`：真实世界历史观测。**必须**具有 `provenance_status: "verified"` 并附带公开数据源 URL/数据集 ID、原始文件哈希及聚合方法；否则将被系统直接拒绝加载。
     - `"synthetic"`：合成测试或演示数据。在 CLI 中运行必须显式指定 `--allow-unverified`（或 `--dry-run`），否则以 `exit code 2` 拒绝执行。
     - `"fixture"`：极简测试桩数据。在 CLI 中运行必须显式指定 `--dry-run`，否则以 `exit code 2` 拒绝执行。
2. **因果时序与事前性保证 (Ex-ante Causality)**：
   - 模型推演输入**仅允许使用灾害发生前（提前期 `lead_time_hours`）已发布的气象预警与预报**。
   - 若只能获取事后实际观测雨量作为气象强迫，元数据中必须显式标记 `input_mode: "observed_hindcast"`，并在报告中明确提示属于后报验证 (Hindcast)，而非真实事前推演。
3. **防泄露与防循环相关 (Anti-Leakage & Provenance)**：
   - 严禁使用与实测灾情循环相关的局部灾害强度输入。若 `event.json` 包含 `hazard_distribution` 但未提供可验证的事前预报源与发布时间戳，系统将自动丢弃该输入并采用模型默认空间分布。
   - `impacts.csv` 绝不参与模型推演，仅用于事后评估预测脆弱度与实测灾情的 Spearman 相关性。
4. **严格事件级留出 (Strict Event-Level Holdout)**：
   - 测试事件集合 (`role: "test"`) 严禁参与模型权重调优、超参数优化或智能体微调。
   - 调优仅允许在明确标记为 `role: "calibration"` 的事件上进行，且测试集与校准集事件 ID 必须严格互斥。
5. **自动化质量门禁标准 (Quality Gate Standards)**：
   - 验证报告和 CLI 输出均由代码自动判定门禁状态：
     - **Provenance 审计**：必须为 `observed` 且 `verified`。
     - **基线超越检验**：Multi-Agent 预测误差 (RMSE) 必须优于或等于各基准模型（Historical Baseline、AR、Ridge）。
     - **不确定性覆盖**：95% 置信带实际覆盖率 (PICP) $\ge 80.0\%$。
     - **振幅误差精度**：最大活动损失绝对误差 $\le 15.0\%$。
   - 任意一项未达标，综合门禁自动判定为 **`FAILED`**，禁止宣称闭环完成。

---

## 2. 目录结构

每个外部验证事件独立存放在 `data/external/<event_id>/` 目录下：

```text
data/external/
├── README.md                          # 本规范说明文档
└── <event_id>/                        # 事件目录（如 SZ_20230907_HAIKUI_SYNTHETIC_DEMO）
    ├── event.json                     # 事件元数据与预警输入
    ├── zones.csv                      # 外部数据空间单元到模型12个韧性分区的映射关系
    ├── activity.csv                   # 小时级区域活动代理观测序列
    ├── impacts.csv                    # (可选) 区域实际物理灾情/积水观测
    └── PROVENANCE.json                # 数据来源、授权及 SHA-256 校验清单
```

---

## 3. 文件格式与字段定义

### 3.1 `event.json`（事件元数据）

| 字段 | 类型 | 必填 | 约束与说明 | 示例 |
| :--- | :--- | :--- | :--- | :--- |
| `event_id` | string | 是 | 唯一事件标识符 | `"SZ_20230907_HAIKUI_SYNTHETIC_DEMO"` |
| `city_name` | string | 是 | 城市名称，需与模型城市匹配 | `"Shenzhen"` |
| `hazard_type` | string | 是 | 灾害类型，枚举值：`"rainstorm"`, `"typhoon"`, `"heatwave"`, `"flood"` | `"rainstorm"` |
| `start_time` | string | 是 | 灾害事件起始时间（ISO 8601，含时区） | `"2023-09-07T18:00:00+08:00"` |
| `end_time` | string | 是 | 灾害事件结束时间（ISO 8601，含时区） | `"2023-09-08T12:00:00+08:00"` |
| `forecast_available_time` | string | 是 | 预报发布时间，必须 `<= start_time - lead_time_hours` | `"2023-09-07T06:00:00+08:00"` |
| `lead_time_hours` | integer | 是 | 预报提前期（小时），`>= 1` | `12` |
| `duration_hours` | integer | 是 | 灾害持续时间（小时），`>= 1` | `18` |
| `total_hours` | integer | 是 | 总验证窗口时长（小时），`>= duration_hours + 12` | `96` |
| `event_start_hour` | integer | 是 | 事件在总窗口中的相对小时索引（从0开始） | `24` |
| `peak_intensity` | float | 是 | 归一化灾害强度峰值，区间 `[0.0, 1.0]` | `0.95` |
| `hazard_distribution` | object | 否 | 分区局部冲击强度（需附带事前预报源和时间戳，否则将被丢弃） | `null` |
| `forecast_source` | string | 是 | 气象预报或预警发布权威机构 | `"深圳市气象局历史预警记录"` |
| `role` | string | 是 | 数据用途，`"test"` 或 `"calibration"` | `"test"` |
| `data_kind` | string | 是 | 数据性质，`"observed"`、`"synthetic"` 或 `"fixture"` | `"synthetic"` |
| `provenance_status` | string | 否 | 审计状态，`"verified"` 或 `"unverified"`（`data_kind: observed` 强制要求 `"verified"`） | `"unverified"` |
| `input_mode` | string | 是 | 输入模式，`"ex_ante_forecast"` 或 `"observed_hindcast"` | `"observed_hindcast"` |

### 3.2 `zones.csv`（空间映射表）

将外部数据中的源区域（如行政区、街道办事处、交通小区）映射至系统的 12 个标准韧性评估区：
`SZ_FT_CBD`, `SZ_NS_TECH`, `SZ_NS_QH`, `SZ_NS_SK`, `SZ_LH_COMM`, `SZ_BA_CTR`, `SZ_LG_BT`, `SZ_LH_RES`, `SZ_YT_PORT`, `SZ_GM_SCI`, `SZ_PS_IND`, `SZ_DP_ECO`。

| 字段 | 类型 | 说明 |
| :--- | :--- | :--- |
| `source_zone_id` | string | 外部数据中的原始区域标识符 |
| `model_zone_id` | string | 对应的模型分区 ID（必须在上述12个标准分区中） |
| `weight` | float | 该源区域计入模型分区的权重比例（`> 0.0`），同一源区域的所有权重和应为 1.0 |

### 3.3 `activity.csv`（小时级活动观测表）

| 字段 | 类型 | 说明 |
| :--- | :--- | :--- |
| `timestamp` | string | 小时观测时间戳（ISO 8601，如 `2023-09-07T00:00:00+08:00`），时区必须与 `event.json` 一致 |
| `zone_id` | string | 区域标识（对应 `zones.csv` 中的 `source_zone_id` 或直接为 `model_zone_id`） |
| `activity_metric` | float | 小时聚合活动绝对指标（如进出站客流量、定位活跃人次、通过车次等，`>= 0.0`） |
| `metric_type` | string | 活动代理类型（如 `"metro_ridership"`, `"cellular_activity"`, `"traffic_flow"`） |

**时序连续性要求**：
- 每个区域的时间序列必须严格连续（步长 1 小时），不得缺失、跳跃或重复。
- 总小时数必须与 `event.json` 的 `total_hours` 完全一致。

### 3.4 `impacts.csv`（可选：区域物理灾情表）

用于空间维度的 Rank 相关性验证（如易涝点数量、抢险出警记录、路段积水阻断次数等）：

| 字段 | 类型 | 说明 |
| :--- | :--- | :--- |
| `zone_id` | string | 区域标识符（`model_zone_id` 或 `source_zone_id`） |
| `impact_score` | float | 归一化影响得分 `[0.0, 1.0]` 或实测灾情绝对计数值 |
| `impact_type` | string | 灾情类型（如 `"waterlogging_points"`, `"traffic_disruption_events"`） |

### 3.5 `PROVENANCE.json`（数据来源与校验表）

```json
{
  "event_id": "SZ_20230907_HAIKUI_SYNTHETIC_DEMO",
  "data_kind": "synthetic",
  "provenance_status": "unverified",
  "source_name": "Synthetic Demonstration Dataset (Unverified Demo)",
  "license": "CC0-1.0 (Public Domain)",
  "retrieved_at": "2026-10-18T00:00:00+08:00",
  "time_range": "2023-09-06T18:00:00+08:00 to 2023-09-10T17:00:00+08:00",
  "checksums": {
    "event.json": "...",
    "zones.csv": "...",
    "activity.csv": "...",
    "impacts.csv": "..."
  }
}
```

> **注意**：当 `data_kind: "observed"` 或 `provenance_status: "verified"` 时，`PROVENANCE.json` 还必须包含以下字段：
> - `source_url` 或 `dataset_id`：可追溯的数据源公开链接或数据库登记号。
> - `raw_file_checksums`：未经聚合的原始数据文件 SHA-256 哈希字典。
> - `aggregation_method`：原始数据至 `activity.csv` 的清洗聚合算法描述。

---

## 4. 运行与验证指引

1. 运行合成测试桩（Dry-run 冒烟测试）：
   ```powershell
   python run_external_validation.py --event-id SZ_20230907_MINIMAL_FIXTURE --dry-run
   ```
2. 运行未验证合成演示序列（需显式授权）：
   ```powershell
   python run_external_validation.py --event-id SZ_20230907_HAIKUI_SYNTHETIC_DEMO --allow-unverified
   ```
3. 运行已审计真实历史观测数据集（放置于 `data/external/<verified_event_id>/` 后）：
   ```powershell
   python run_external_validation.py --event-id <verified_event_id>
   ```
