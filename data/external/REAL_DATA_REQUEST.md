# 真实历史事件与多源移动数据采集与接入需求清单 (Real-World Data Request)

> [!IMPORTANT]
> **当前验证状态：BLOCKED（受阻于真实数据缺失）**
> 
> 当前仓库内仅提供合成演示数据集 (`data/external/SZ_20230907_HAIKUI_SYNTHETIC_DEMO/`，`data_kind: "synthetic"`, `provenance_status: "unverified"`)，**尚未完成真实历史灾害事件的实测验证**。根据系统核心治理原则，系统**严禁使用合成数据冒充真实验证**，亦严禁伪造数据源或哈希。
> 
> **注意**：即使接入了满足最低门槛（24 小时灾前基线）的真实数据，系统中的 `PredictabilityAgent` 因无法计算 48 小时自相关及日间余弦规律性，仍会自动降级为中性值回退（0.50）。因此，强烈建议数据提供方提供 $\ge 48$ 小时（推荐 72 小时）连续灾前观测序列以实现完整未降级的 Multi-Agent 评估。
> 
> 本文档定义接入已审计真实历史观测数据所需的最小完整数据契约与合规要求。

---

## 0. 真实历史数据可行性审计与候选源评估 (Feasibility Audit & Sourcing Assessment)

> [!NOTE]
> 审计评估时间：2026-10-21  
> 审计目标事件：深圳市 2023-09-07 “9·7” 极端特大暴雨（台风“海葵”残余环流）及其他历史灾害候选事件

针对接入“具备可追溯性、公开许可、空间覆盖 12 区、小时级连续观测”的真实外部验证数据集，系统开展了专题可行性审计，审计检查结果如下：

### 0.1 审计检查项核验结果
1. **空间粒度（能否映射至 12 个评估分区）**：
   - **目标**：能够将城市全域观测映射至模型 12 个标准韧性区（如福田 CBD、南山科技园、前海、罗湖商业区、龙岗宝龙工业区等）。
   - **核验**：部分公开研究（如 Zenodo 上的 PassengerRequestDataset）仅覆盖福田、罗湖等局部中心城区，缺乏宝安、光明、坪山、大鹏等外围城区的同步实测，空间完整度无法满足全域 12 区评估。
2. **时间粒度（连续小时级时序）**：
   - **目标**：灾前 $\ge 24\text{h}$（完整经验规律性推荐 $\ge 48\text{h}$）、灾害期与恢复期至少 96 小时连续小时级时序。
   - **核验**：官方统计年鉴及公开政务通报多为“单日客运总量”（如单日 853 万/1017 万人次）或定性应急响应通报，缺少连续 96 小时无断点的逐小时站点/分区时序。
3. **观测指标（人类活动代理）**：
   - **目标**：真实反映人群流动与活动的物理指标（地铁刷卡出入站、网约车上下车点、浮动车轨迹衍生流量、公开城市热力指数）。
   - **核验**：深圳地铁集团与市交通运输局具备此类底层数据，但属于关键市政基础设施核心遥测数据，未对公众无限制开放逐小时站点级刷卡矩阵。
4. **数据源与许可合规**：
   - **深圳市政府数据开放平台 (`opendata.sz.gov.cn`)**：提供交通运行统计与部分实时 API，但历史长周期多站点逐小时全量刷卡/客流矩阵不对未鉴权公众提供静态直接下载，需科研或企业法人实名申请与合规审批。
   - **学术开放数据仓（Zenodo, ScienceDB, Figshare, Harvard Dataverse）**：暂无公开发布的涵盖 2023-09-05 至 2023-09-10 期间深圳全域 12 区连续逐小时实测客流数据集。
   - **运营商信令与互联网位置大数据**：受《中华人民共和国数据安全法》与《中华人民共和国个人信息保护法》(PIPL) 严格约束，严禁未经合规脱敏与授权公开分发原始移动轨迹。
5. **可审计性与校验**：
   - **目标**：必须具备公开稳定的源 URL、原始文件 SHA-256 校验和、明确采集时间戳和开源/开放许可协议。
   - **核验**：目前不存在无需特权认证即可下载并满足上述所有校验标准的公开数据集。

### 0.2 候选数据源逐项核实清单 (Candidate Sourcing Audit Matrix)

针对 5 类候选数据源，本次审计核查了数据 URL、许可证、空间/时间粒度及可获取性：

| 候选来源类别 | 候选机构与数据源 (source_name / dataset_id) | 参考链接 (source_url) | 授权许可 (license / url) | 时间范围与时间粒度 | 空间粒度与 12 区映射 | 个人/受限数据 | 获取方式与阻断原因 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1. 气象与预警** | 深圳市气象台 (SZMB) / 广东省气象局历史预警与自动站降水 | `http://weather.sz.gov.cn/` | 政府公开信息 (Government Public Info) | 2023-09-07~09-08 逐小时/逐 5 分钟雨量 | 覆盖深圳各区自动气象站，可映射至 12 区 | 否 | **部分可得**：官方灾情通报与日/小时降雨量极值公开，但空间精细网格雷达产品 (`nc`/`grib2`) 需气象科研专线申请。 |
| **2. 政务与轨道客流** | 深圳市政府数据开放平台 / 深圳地铁运营客流统计 (`SZ-METRO-AFC`) | `https://opendata.sz.gov.cn/` | 深圳市公共数据开放利用协议 (ODbL 类似) | 2023 年 9 月日级客运量 / 实时 API 采样 | 288+ 地铁站，可按权重聚合至 12 区 | 否 (已聚合) | **受阻 (BLOCKED)**：平台仅开放静态月度统计与实时限流 API，未开放 2023-09-05~10 连续 96h 无断点的逐小时全线网出入站历史矩阵下载，需企业实名审批。 |
| **3. 交通指数与速度** | 高德/百度城市交通出行大数据报告与实时交通指数 API | `https://report.amap.com/` | 商业专有 (Proprietary / Free Web View) | 2023 年 Q3 拥堵延误指数 (日均/高峰) | 行政区级 (区级/全市)，未细化至街道网格 | 否 (聚合指数) | **受阻 (BLOCKED)**：公开网页端仅提供日均或高峰拥堵排名与折线图，历史小时级 12 区道路运行速度矩阵需定制购买商业商业数据包。 |
| **4. 学术公开数据集** | Zenodo / ScienceDB 城市出行数据集 (如 Zhu et al. PassengerRequestDataset) | `https://zenodo.org/record/...` | CC-BY-4.0 / ODbL-1.0 | 局部时段或 2023 年非灾害期采样 | 仅覆盖福田、罗湖核心区，缺少外围 8 区 | 否 (去标识化) | **受阻 (BLOCKED)**：公开发布的数据集空间不完整，无法覆盖深圳 12 个评估分区，且不涵盖 9·7 极端特大暴雨 96 小时完整窗口。 |
| **5. 应急与内涝灾情** | 深圳市应急管理局 / 水务局 / 消防救援支队官方警情通报 | `http://yjj.sz.gov.cn/` | 政府政务通报 (Public Notices) | 2023-09-07 18:00 至 09-08 12:00 灾情汇总 | 罗湖、福田、龙岗等重点积水点与抢险案例 | 否 | **仅定性/半定量**：官方通报记录了抢险出警总数、重点停运区段，但未发布覆盖全域 12 区标准化网格的结构化 impacts.csv 矩阵。 |

### 0.3 需要用户/合作方协助提供的真实数据清单 (Requested File Deliverables)
若需推进真实历史外部验证，需数据提供方提供经合法脱敏的以下 4 个文件（或原始数据并提供计算哈希）：
1. `activity.csv`：涵盖 2023-09-05T00:00:00+08:00 至 2023-09-09T23:00:00+08:00 连续 120 小时（灾前 $\ge 48\text{h}$）、覆盖 12 个评估分区的小时级人类活动代理指标（如地铁刷卡量、卡口车流或网约车订单数）；
2. `zones.csv`：原始站点/单元 ID 到模型 12 个韧性分区的空间归属权重映射表；
3. `event.json`：标明官方预警发布时间戳 (`forecast_available_time`)、事件起始与持续时长的事件元数据；
4. `PROVENANCE.json`：包含数据源机构全称、授权使用协议、原始文件 SHA-256 校验哈希的合规清单。

### 0.4 审计结论与阻断状态
- **当前结论**：**BLOCKED（受阻于真实数据不可得）**。
- **治理纪律**：系统恪守科研诚信与因果透明性，**严禁为了“假装完成验证”而合成伪造观测数据、使用平滑插值冒充实测、或将合成演示数据集改名为 observed**。
- **后续行动路径**：
  1. 维持当前外部验证结论为真实客观的 `BLOCKED`，合成演示管线保留为 `SZ_20230907_HAIKUI_SYNTHETIC_DEMO`；
  2. 待合作研究机构或政府/企业主体提供经授权合规脱敏的逐小时实测数据集（含原始文件 SHA-256 签名）后，按照本需求清单规范填入 `data/external/<verified_event_id>/` 目录并触发端到端验证。

---

## 1. 目标事件推荐与入选标准

优先选择具备完整公开或授权监测记录的极端历史灾害事件（例如深圳 **2023-09-07 “9·7” 极端特大暴雨**，或台风“苏拉”、“山竹”等）。

入选事件必须具备以下前置条件：
1. **气象/灾害预警权威可信**：具备灾害发生前（至少提前 `lead_time >= 12h`）深圳市气象台等官方机构发布的暴雨/台风预警信号及发布时间戳；
2. **人类活动/移动代理连续可得**：覆盖灾前基线期、灾害发生期与灾后恢复期的小时级活动观测代理。灾前观测窗口区分两种模式：
   - **最低可行窗口 (Degraded Minimum)**: 灾前至少 24 小时（仅能提取日内负荷轮廓与重力流，Predictability 降级为中性值 0.50）；
   - **推荐完整窗口 (Recommended Full)**: 灾前 $\ge 48$ 小时（推荐 72 小时 / 3 天完整观测，支持自相关与日间余弦规律性经验计算）；
3. **空间覆盖完整**：能够映射至深圳市 12 个评估分区；
4. **合规许可清晰**：具备公开科研或企业授权使用协议，允许在开源或评估环境中复现。

---

## 2. 必需数据文件与格式清单

数据集需存放在 `data/external/<verified_event_id>/` 目录下，包含以下 5 个文件：

### 2.1 `event.json`（事件元数据与事前预警强迫）
- `event_id`: 唯一标识符（如 `SZ_20230907_HAIKUI_VERIFIED`）
- `city_name`: `"Shenzhen"`
- `hazard_type`: `"rainstorm"` / `"typhoon"` / `"flood"`
- `start_time`: ISO 8601 时间戳（如 `"2023-09-07T18:00:00+08:00"`）
- `end_time`: ISO 8601 时间戳
- `forecast_available_time`: 必须 `<= start_time - lead_time_hours`（严守事前因果）
- `lead_time_hours`: `>= 1`（推荐 12~24 小时）
- `event_start_hour`: 灾害在总窗口中的小时索引。
  - **最低要求（降级运行）**: `>= 24`（仅支持 24h 日内周期外推，Predictability 降级为中性回退 `neutral_fallback`）；
  - **推荐标准（完整运行）**: `>= 48`（推荐 72 小时，解锁完整 `observed_pre_event` 经验可预测性特征）；
- `duration_hours`: 灾害影响持续小时数
- `total_hours`: 评估总窗口（必须 `>= event_start_hour + duration_hours + 12`）
- `peak_intensity`: 归一化强度 `[0.0, 1.0]`
- `forecast_source`: 官方发布机构全称
- `data_kind`: `"observed"`
- `provenance_status`: `"verified"`
- `input_mode`: `"ex_ante_forecast"`（事前预警）或 `"observed_hindcast"`（后报雷达）

### 2.2 `zones.csv`（空间映射表）
将外部原始空间单元（交通小区 TAZ、街道办、网格、传感器站点）映射至模型 12 个标准韧性区：
- `source_zone_id`: 原始数据中的区域/站点 ID
- `model_zone_id`: 模型 12 分区之一（`SZ_FT_CBD`, `SZ_NS_TECH`, `SZ_NS_QH`, `SZ_NS_SK`, `SZ_LH_COMM`, `SZ_BA_CTR`, `SZ_LG_BT`, `SZ_LH_RES`, `SZ_YT_PORT`, `SZ_GM_SCI`, `SZ_PS_IND`, `SZ_DP_ECO`）
- `weight`: 归属权重（`> 0.0`），同一 `source_zone_id` 权重和为 1.0

### 2.3 `activity.csv`（小时级移动活动代理）
- `timestamp`: ISO 8601 格式，带时区，严格 1 小时间隔无跳变、无重复
- `zone_id`: 区域 ID（对应 `zones.csv` 的 `source_zone_id` 或 `model_zone_id`）
- `activity_metric`: 活动量（地铁客流、刷卡量、手机信令活跃数、网约车上车点数等，`>= 0.0`）
- `metric_type`: 代理类型（如 `"metro_ridership"`, `"cellular_density"`, `"traffic_index"`）
- **时序长度要求**：每个分区观测点数量必须恰好等于 `event.json` 的 `total_hours`，且 `event_start_hour >= 24`（推荐 `event_start_hour >= 48`）。

### 2.4 `impacts.csv`（可选：实测物理灾情表）
- `zone_id`: 分区 ID
- `impact_score`: 实测灾情统计（积水点数量、抢险出警数、道路中断时长）
- `impact_type`: 灾情指标类型
- *注：此表仅用于事后空间秩相关（Spearman $\rho$）验证，绝对严禁作为模型推演输入。*

### 2.5 `PROVENANCE.json`（可审计数据源与校验清单）
```json
{
  "event_id": "SZ_20230907_HAIKUI_VERIFIED",
  "data_kind": "observed",
  "provenance_status": "verified",
  "source_name": "深圳市气象局 / 深圳地铁运营客流公开报告",
  "source_url": "https://opendata.sz.gov.cn/...",
  "dataset_id": "SZ-METRO-20230907-HOURLY",
  "license": "ODbL-1.0 / Open Government Data License",
  "retrieved_at": "2026-10-20T10:00:00+08:00",
  "time_range": "2023-09-06T18:00:00+08:00 to 2023-09-10T17:00:00+08:00",
  "aggregation_method": "将原始地铁 288 个站点客流按 zones.csv 距离权重聚合至 12 个韧性评估分区",
  "raw_file_checksums": {
    "raw_metro_trips_202309.parquet": "<sha256_hash>",
    "raw_weather_radar_grid.nc": "<sha256_hash>"
  },
  "checksums": {
    "event.json": "<sha256_hash>",
    "zones.csv": "<sha256_hash>",
    "activity.csv": "<sha256_hash>",
    "impacts.csv": "<sha256_hash>"
  }
}
```

---

## 3. 防泄漏与事前后报约束校验准则

1. **时序隔离与特征降级**：
   - 真实移动数据接入管线后，系统通过 `ObservedMobilityAdapter` **仅截取 `timestamp < event_start` 的灾前观测**驱动 Agent 计算韧性指标。
   - 灾害期间与恢复期的真实观测数据**绝不输入**模型。
   - 若灾前观测窗口 $24 \le \text{pre\_event\_hours} < 48$，系统透明标记 `predictability_source: "neutral_fallback"` 并在报告中输出降级告警；若 $\text{pre\_event\_hours} \ge 48$，系统标记 `predictability_source: "observed_pre_event"`，计算完整经验规律性。
2. **基线外推**：
   - 推演使用的基线由灾前 24 小时自然日周期外推重复生成，严禁使用事件期真实值充当基线。
3. **质量门禁**：
   - 真实数据接入后，系统自动运行基线超越检验（RMSE 优于 AR/Ridge）、置信带覆盖检验（PICP $\ge 80\%$）和振幅误差检验（$\le 15\%$）。全项达标后方可将综合结论评定为 `PASS`。
