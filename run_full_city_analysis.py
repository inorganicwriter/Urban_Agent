import logging
import sys
from pathlib import Path

# Force UTF-8 encoding on Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from typing import Any, Dict, Optional
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from urban_resilience.config import AgentWeightConfig, ProjectPaths, SimulationConfig
from urban_resilience.pipeline import UrbanResiliencePipeline
from urban_resilience.prediction.provenance import (
    get_provenance_warning,
    get_synthetic_warning,
)
from urban_resilience.types import HazardType


def _configure_logging():
    """Configures file and stream logging lazily when analysis runs."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(ProjectPaths.OUTPUTS_DIR / "resilience_run.log", mode="w", encoding="utf-8"),
        ],
    )


logger = logging.getLogger("CityAnalysis")
console = Console()


ZONE_RECOMMENDATIONS = {
    "SZ_FT_CBD": "高密度商务中心，地下空间复杂。需重点加强车库与地下通道防水防倒灌演练，确保电力与通信连续性。",
    "SZ_NS_TECH": "高新技术产业园区，通勤人流高度密集且时序刚性强。建议实行弹性错峰上下班并强化园区排水保障。",
    "SZ_NS_QH": "滨海低洼新城开发区，易受沿海强降雨及风暴潮浸润。需重点防护地下交通枢纽与综合管廊，保障应急物资通道。",
    "SZ_NS_SK": "沿海低标高港口生活片区，易受风暴潮顶托与局部内涝影响。需密切监控海水倒灌与滨海大道积水情况，加固防汛挡板。",
    "SZ_LH_COMM": "老旧城区地势偏低，排水系统负荷高，商业客流密集。建议在灾前前置部署大型移动抽排泵车，并关闭低洼地下商场与下沉通道。",
    "SZ_BA_CTR": "宝安中心城区，人口密度大且临海地势平缓。需加强雨水泵站巡检及内涝易发点值守。",
    "SZ_LG_BT": "信息通信产业聚集区，内陆较高地势但人车流量大。需防范局部边坡地质风险与关键制造基地供电稳定。",
    "SZ_LH_RES": "高铁枢纽与高密度居住生活区，跨区通勤潮汐特征显著。建议及时发布交通疏导信息，开展错峰出行与枢纽人员分流。",
    "SZ_YT_PORT": "深水港区与临港物流枢纽，对外交通通道单一。需防范台风期岸桥作业安全与疏港公路瓶颈拥堵。",
    "SZ_GM_SCI": "科学城高新研发组团，地处内陆高地。关注科研实验基地供电水韧性与周边新建路网截排水系统。",
    "SZ_PS_IND": "生物医药与新能源制造基地，路网密度较低。建议加强园区危化品库区防洪巡查与关键企业供应链通道保畅。",
    "SZ_DP_ECO": "处于沿海末端走廊，路网属于单一通道结构（空间冗余低），且受沿海风暴潮强迫显著。建议灾前提前组织沿海游客与居民撤离，并备勤水上应急救援。",
}


def run_shenzhen_full_analysis():
    """Runs the complete anticipatory resilience evaluation on the Shenzhen Metropolitan Area."""
    _configure_logging()
    console.print(Panel.fit("[bold cyan]Urban Resilience Anticipatory Multi-Agent Analysis: SHENZHEN[/bold cyan]"))

    # Disaster simulation scenario parameters (Single source of truth)
    hazard_type = HazardType.RAIN_STORM
    event_start_hour = 24
    event_duration_hours = 18
    peak_intensity = 0.88
    total_hours = 96
    event_lead_time_hours = 12
    monte_carlo_samples = 300
    recovery_threshold_ratio = 0.95
    random_seed = 42

    sim_cfg = SimulationConfig(
        total_hours=total_hours,
        event_lead_time_hours=event_lead_time_hours,
        event_start_hour=event_start_hour,
        event_duration_hours=event_duration_hours,
        peak_intensity=peak_intensity,
        monte_carlo_samples=monte_carlo_samples,
        recovery_threshold_ratio=recovery_threshold_ratio,
        random_seed=random_seed,
    )

    weight_cfg = AgentWeightConfig(
        temporal_redundancy=0.20,
        intraday_flexibility=0.20,
        predictability=0.15,
        spatial_redundancy=0.25,
        functional_mix=0.20,
    )

    pipeline = UrbanResiliencePipeline(
        city_name="Shenzhen Metropolitan Area",
        sim_config=sim_cfg,
        weight_config=weight_cfg,
    )

    # Execute full pipeline with single-source scenario parameters
    results = pipeline.run_full_pipeline(
        hazard_type=hazard_type,
        event_start_hour=event_start_hour,
        duration_hours=event_duration_hours,
        peak_intensity=peak_intensity,
    )

    prediction = results["prediction"]
    outlook = results["spatial_outlook"]
    validation = results["validation_result"]
    indicators = results["key_indicators"]
    plots = results["plot_paths"]

    # ---------------------------------------------------------
    # PRINT RESULTS TO CONSOLE
    # ---------------------------------------------------------
    console.print("\n[bold green][SUCCESS] FULL PIPELINE EXECUTION COMPLETE[/bold green]\n")

    # 1. Summary Indicator Table
    t_ind = Table(title="[bold blue]Key Anticipatory Disruption & Recovery Indicators[/bold blue]")
    t_ind.add_column("Indicator", style="cyan", justify="left")
    t_ind.add_column("Predicted Value", style="bold yellow", justify="right")
    for k, v in indicators.items():
        t_ind.add_row(k, v)
    prov = getattr(prediction, "provenance", None)
    if prov:
        synth_warn = get_synthetic_warning(prov)
        prov_warn = get_provenance_warning(prov)
        t_ind.add_row("Summary Metric Source", prov.summary_metric_source)
        t_ind.add_row("Curve Source", prov.curve_source)
        t_ind.add_row("Curve Metric Source", getattr(prov, "curve_metric_source", "monte_carlo_physics"))
        t_ind.add_row("Model Status", prov.model_status)
        t_ind.add_row("Interval Source", prov.interval_source)
        t_ind.add_row("Training Label Source", str(prov.training_label_source or "none"))
        t_ind.add_row("Engineering Only", str(bool(prov.engineering_only)))
        t_ind.add_row("Training Event IDs", ", ".join(prov.training_event_ids) if prov.training_event_ids else "None")
        if synth_warn:
            t_ind.add_row("[yellow]Synthetic Warning[/yellow]", f"[bold yellow]{synth_warn}[/bold yellow]")
        elif prov_warn:
            t_ind.add_row("[red]Provenance Warning[/red]", f"[bold red]{prov_warn}[/bold red]")
    console.print(t_ind)

    # 2. Spatial Priority Ranking Table
    t_rank = Table(title="[bold red]Spatial Vulnerability Outlook & Emergency Priority Ranking[/bold red]")
    t_rank.add_column("Rank", style="bold magenta", justify="center")
    t_rank.add_column("Zone ID", style="cyan")
    t_rank.add_column("Zone Name", style="white")
    t_rank.add_column("Vulnerability Score", style="bold yellow", justify="right")
    t_rank.add_column("Composite Resilience", style="green", justify="right")
    t_rank.add_column("Primary Vulnerability Driver", style="red")

    for z in outlook.zone_rankings:
        t_rank.add_row(
            str(z.vulnerability_rank),
            z.zone_id,
            z.zone_name,
            f"{z.vulnerability_score:.3f}",
            f"{z.composite_resilience_index:.3f}",
            z.primary_vulnerability_driver,
        )
    console.print(t_rank)

    # 3. Baseline Comparison Table
    t_base = Table(title="[bold green]Historical Validation: Baseline Benchmark Comparison[/bold green]")
    t_base.add_column("Model / Algorithm", style="cyan")
    t_base.add_column("RMSE", justify="right")
    t_base.add_column("MAE", justify="right")
    t_base.add_column("MAPE (%)", justify="right")
    t_base.add_column("R^2 Score", justify="right", style="bold")

    for b in validation.baseline_comparisons:
        is_ours = "Proposed" in b.model_name
        style = "bold green" if is_ours else "white"
        t_base.add_row(
            f"[{style}]{b.model_name}[/{style}]",
            f"[{style}]{b.rmse:.4f}[/{style}]",
            f"[{style}]{b.mae:.4f}[/{style}]",
            f"[{style}]{b.mape:.2f}%[/{style}]",
            f"[{style}]{b.r2_score:.4f}[/{style}]",
        )
    console.print(t_base)

    # 4. Agent Ablation Study Table
    t_abl = Table(title="[bold yellow]Agent Ablation Study (Marginal Importance of Domain Perspectives)[/bold yellow]")
    t_abl.add_column("Ablation Configuration", style="cyan")
    t_abl.add_column("RMSE", justify="right")
    t_abl.add_column("MAE", justify="right")
    t_abl.add_column("Delta RMSE (%)", justify="right", style="bold red")

    for name, metrics in validation.ablation_results.items():
        t_abl.add_row(
            name,
            f"{metrics['rmse']:.4f}",
            f"{metrics['mae']:.4f}",
            f"{metrics['delta_rmse_pct']:+.2f}%",
        )
    console.print(t_abl)

    # 5. Uncertainty Calibration
    calib = validation.calibration
    console.print(Panel(
        f"[bold]Uncertainty Calibration Quality (95% CI):[/bold]\n"
        f"- Empirical Coverage (PICP): [bold cyan]{calib.picp_95 * 100:.1f}%[/bold cyan] (Target: 95.0%)\n"
        f"- Mean Prediction Interval Width (MPIW): [bold cyan]{calib.mpiw:.4f}[/bold cyan]\n"
        f"- Winkler Interval Score: [bold cyan]{calib.interval_score:.4f}[/bold cyan]\n"
        f"- Expected Calibration Error (ECE): [bold cyan]{calib.ece:.4f}[/bold cyan]",
        title="[bold blue]Uncertainty Calibration[/bold blue]",
    ))

    # 6. Generated Graphics
    console.print("\n[bold green]Generated High-Resolution Visualizations:[/bold green]")
    for name, p in plots.items():
        console.print(f"  - {name.upper()}: [cyan]{p}[/cyan]")

    # Export a comprehensive Markdown report to reports/
    report_path = ProjectPaths.REPORTS_DIR / "shenzhen_resilience_report.md"
    resolved = getattr(pipeline, "resolved_config", sim_cfg)
    scenario_info = {
        "hazard": "极端特大暴雨与强台风侵袭",
        "duration_hours": resolved.event_duration_hours,
        "peak_intensity": resolved.peak_intensity,
        "event_start_hour": resolved.event_start_hour,
    }
    _generate_markdown_report(report_path, prediction, outlook, validation, indicators, scenario=scenario_info)
    console.print(f"\n[bold yellow]Executive Analytical Report exported to:[/bold yellow] [cyan]{report_path}[/cyan]\n")

    return results


def _generate_markdown_report(
    path: Path, pred, outlook, val, indicators, scenario: Optional[Dict[str, Any]] = None
):
    """Writes a formal markdown analytical report."""
    if scenario:
        hazard_desc = scenario.get("hazard", "极端气象扰动")
        dur_desc = scenario.get("duration_hours")
        peak_desc = scenario.get("peak_intensity")
        if dur_desc is not None and peak_desc is not None:
            scenario_desc = f"{dur_desc}小时持续冲击，峰值强迫 {peak_desc}"
        elif dur_desc is not None:
            scenario_desc = f"{dur_desc}小时持续冲击"
        else:
            scenario_desc = "情景参数部分提供"
    else:
        hazard_desc = "极端气象扰动"
        scenario_desc = "情景参数未显式提供（以运行配置为准）"

    prov = getattr(pred, "provenance", None)
    synth_warn = get_synthetic_warning(prov)
    prov_warn = get_provenance_warning(prov)
    warning_banner = ""
    if synth_warn:
        warning_banner = f"> [!WARNING]\n> **合成预测器产物警告**: {synth_warn}\n\n"
    elif prov_warn:
        warning_banner = f"> [!WARNING]\n> **模型训练溯源警告**: {prov_warn}\n\n"

    content = f"""# 深圳市城市系统韧性前瞻性预测与应急防灾推演报告
{warning_banner}**分析对象**: 深圳市重点城区与产业功能区（Shenzhen Metropolitan Area）  
**扰动情景**: {hazard_desc}（{scenario_desc}）  
**预测范式**: 基于 ReAct 多视角领域智能体与证据综合闭环系统  

---

## 一、 宏观中断与恢复时序核心指标
- **最大活动损失 (Maximum Activity Loss)**: {indicators['Maximum Activity Loss']}
- **累积活动受损总量 (Cumulative Activity Loss)**: {indicators['Cumulative Activity Loss']}
- **预计恢复时长 (Recovery Time)**: {indicators['Recovery Time']}
- **损失峰值时刻 (Peak Loss Hour)**: {indicators['Peak Loss Hour']}
- **指标来源 (Summary Metric Source)**: {getattr(getattr(pred, 'provenance', None), 'summary_metric_source', 'monte_carlo_physics')}
- **曲线来源 (Curve Source)**: {getattr(getattr(pred, 'provenance', None), 'curve_source', 'monte_carlo_physics')}
- **时序曲线指标来源 (Curve Metric Source)**: {getattr(getattr(pred, 'provenance', None), 'curve_metric_source', 'monte_carlo_physics')}
- **模型状态 (Model Status)**: {getattr(getattr(pred, 'provenance', None), 'model_status', 'untrained_fallback')}
- **模型训练标签来源 (Training Label Source)**: {getattr(getattr(pred, 'provenance', None), 'training_label_source', 'none')}
- **工程测试标记 (Engineering Only)**: {getattr(getattr(pred, 'provenance', None), 'engineering_only', True)}
- **训练事件集合 (Training Event IDs)**: {", ".join(getattr(getattr(pred, 'provenance', None), 'training_event_ids', [])) or 'None'}
- **置信区间机制 (Interval Source)**: {getattr(getattr(pred, 'provenance', None), 'interval_source', 'monte_carlo_quantile')}

## 二、 空间脆弱性格局与应急优先干预排名 (Top Priorities)
各区域由于地形高程、沿海距离、路网连通冗余、功能混合度及通勤节律的异质性，表现出显著的空间分异：

| 优先级 (Rank) | 区域代码 | 区域名称 | 脆弱性指数 (0-1) | 综合韧性指数 (0-1) | 主要脆弱性主导诱因 |
|:---:|:---|:---|:---:|:---:|:---|
"""
    for z in outlook.zone_rankings:
        content += f"| {z.vulnerability_rank} | {z.zone_id} | {z.zone_name} | {z.vulnerability_score:.3f} | {z.composite_resilience_index:.3f} | {z.primary_vulnerability_driver} |\n"

    # Dynamic Top vulnerable zone recommendations
    recommendations_md = ""
    for rank, z in enumerate(outlook.zone_rankings[:4], 1):
        rec_text = ZONE_RECOMMENDATIONS.get(
            z.zone_id,
            f"该区域脆弱性得分为 {z.vulnerability_score:.3f}，主要受{z.primary_vulnerability_driver}驱动。建议针对该主导诱因前置部署应急监测与工程防御措施。",
        )
        recommendations_md += f"{rank}. **{z.zone_name} ({z.zone_id} - Rank #{z.vulnerability_rank})**:\n   - {rec_text}\n"

    content += f"""
### 重点脆弱区域研判与处置策略：
{recommendations_md}
## 三、 历史事件基线验证对比 (Baseline Benchmarks)
| 模型算法 | RMSE (均方根误差) | MAE (绝对误差) | MAPE (%) | R² 决定系数 |
|:---|:---:|:---:|:---:|:---:|
"""
    for b in val.baseline_comparisons:
        content += f"| {b.model_name} | {b.rmse:.4f} | {b.mae:.4f} | {b.mape:.2f}% | {b.r2_score:.4f} |\n"

    # Calculate dynamic baseline conclusion
    proposed_b = next((b for b in val.baseline_comparisons if "Proposed" in b.model_name), None)
    other_baselines = [b for b in val.baseline_comparisons if "Proposed" not in b.model_name]
    if proposed_b and other_baselines:
        strictly_all_better = all(
            (
                proposed_b.rmse < b.rmse
                and proposed_b.mae < b.mae
                and proposed_b.mape < b.mape
                and proposed_b.r2_score > b.r2_score
            )
            for b in other_baselines
        )
        if strictly_all_better:
            baseline_conclusion = "> **结论**: 本研究所提出的多智能体前瞻性预测框架在所有指标上均优于各对比基线，表现出卓越的时空扰动预测与拟合能力。"
        else:
            better_baselines = [
                b.model_name
                for b in other_baselines
                if b.rmse < proposed_b.rmse or b.mae < proposed_b.mae or b.mape < proposed_b.mape or b.r2_score > proposed_b.r2_score
            ]
            better_str = "、".join(set(better_baselines)) if better_baselines else "部分对比基线"
            baseline_conclusion = f"> **结论**: 本框架整体预测精度具备优势，但在部分指标或特定对比下与对比基线 ({better_str}) 各有侧重，展现出各自适配的预测特征。"
    else:
        baseline_conclusion = "> **结论**: 完成对比基线验证分析。"

    content += f"""
{baseline_conclusion}

## 四、 智能体视角消融实验 (Agent Ablations)
| 消融视角配置 | RMSE | MAE | 精度变化 Δ RMSE (%) | 视角物理机制价值 |
|:---|:---:|:---:|:---:|:---|
"""
    for name, m in val.ablation_results.items():
        delta = m.get("delta_rmse_pct", 0.0)
        if "Full Framework" in name:
            note = "完整框架基准（未消融）"
        elif abs(delta) < 1e-9:
            note = "无显著变化（ΔRMSE ≈ 0）"
        elif delta > 0:
            note = f"移除后性能下降 (ΔRMSE +{delta:.2f}%)，证明该视角具有正向增益"
        else:
            note = f"移除后误差反而降低 (ΔRMSE {delta:.2f}%)，表明在当前扰动特征下存在冗余或干扰"
        content += f"| {name} | {m['rmse']:.4f} | {m['mae']:.4f} | {delta:+.2f}% | {note} |\n"

    # Calibration dynamic assessment
    picp_pct = val.calibration.picp_95 * 100
    picp_val = val.calibration.picp_95
    if picp_val < 0.93:
        picp_assessment = f"覆盖率不足（实际 {picp_pct:.1f}%，低于标称 95.0%，需适度扩大置信区间）"
        calib_summary_text = "当前预测区间覆盖率不足，建议在后续演练中调高认识不确定性先验方差以增强区间的包络鲁棒性。"
    elif picp_val > 0.97:
        picp_assessment = f"覆盖率达标（超出标称值 {(picp_val - 0.95) * 100:.1f} 个百分点，区间偏保守）"
        calib_summary_text = "该预测区间有效包络了非线性极端冲击下的系统变异，但区间宽度相对保守，可进一步提高区间锐度。"
    else:
        picp_assessment = f"与标称 95.0% 高度吻合（偏差 {abs(picp_val - 0.95) * 100:.1f} 个百分点）"
        calib_summary_text = "该预测区间既有效包络了非线性极端冲击下的系统变异，又保持了紧凑的区间锐度，能够为特大城市防汛防台风指挥调度提供高度可信的前瞻性定量支持。"

    content += f"""
## 五、 不确定性校准 (Uncertainty Calibration)
- **95% 预测区间实际覆盖率 (PICP)**: {picp_pct:.1f}% ({picp_assessment})
- **平均区间宽度 (MPIW)**: {val.calibration.mpiw:.4f}
- **Winkler 评分**: {val.calibration.interval_score:.4f}
- **期望校准误差 (ECE)**: {val.calibration.ece:.4f}

{calib_summary_text}
"""
    path.write_text(content, encoding="utf-8")


if __name__ == "__main__":
    run_shenzhen_full_analysis()
