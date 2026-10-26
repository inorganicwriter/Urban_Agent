"""Tests for FIX-7: Event Agent bulletin discovery, parsing, and official authority verification."""

from urban_resilience.event import (
    EventDiscoveryAgent,
    EventQuery,
    HazardBulletin,
    HazardSeverityLevel,
    OfficialSourceVerifier,
)


def test_official_source_verifier_accepts_government_authorities():
    """Verifier accepts official meteorological and emergency authorities."""
    verifier = OfficialSourceVerifier()

    ok, reason = verifier.verify_source("Shenzhen Meteorological Bureau", "http://weather.sz.gov.cn")
    assert ok is True
    assert "Verified" in reason

    ok, reason = verifier.verify_source("深圳市应急管理局", "http://yjgl.sz.gov.cn")
    assert ok is True

    ok, reason = verifier.verify_source("China Meteorological Administration", "http://www.cma.gov.cn")
    assert ok is True


def test_official_source_verifier_rejects_unauthorized_sources():
    """Verifier rejects social media, blogs, or unauthorized random channels."""
    verifier = OfficialSourceVerifier()

    # Unauthorized domain
    ok, reason = verifier.verify_source("Anonymous Blogger", "https://medium.com/@weatherguru")
    assert ok is False
    assert "Rejected" in reason

    # Social channel
    ok, reason = verifier.verify_source("Shenzhen Weather Enthusiasts", "https://twitter.com/sz_storms")
    assert ok is False
    assert "unverified external source" in reason


def test_event_discovery_agent_parses_bulletin_text():
    """EventDiscoveryAgent parses unstructured warning notice into structured HazardBulletin."""
    agent = EventDiscoveryAgent()

    raw_notice = (
        "【深圳市气象局红色暴雨预警】受强降雨云系影响，预计未来3小时我市福田、南山、罗湖"
        "累积雨量将达100毫米以上（局地最大雨量达145.5mm，最大风速21.5m/s）。请进入紧急防御状态。"
    )

    bulletin = agent.parse_bulletin_text(
        bulletin_id="BLT_20230907_RED_01",
        raw_text=raw_notice,
        source_authority="深圳市气象局",
        source_channel="weather.sz.gov.cn",
        published_at="2023-09-07T18:30:00",
    )

    assert bulletin.bulletin_id == "BLT_20230907_RED_01"
    assert bulletin.severity_level == HazardSeverityLevel.RED
    assert bulletin.is_verified_official is True
    assert bulletin.peak_rainfall_mm == 100.0 or bulletin.peak_rainfall_mm == 145.5
    assert bulletin.max_wind_speed_ms == 21.5
    assert "福田" in bulletin.affected_zones
    assert "南山" in bulletin.affected_zones
    assert "罗湖" in bulletin.affected_zones


def test_event_discovery_agent_filters_unverified_sources():
    """EventDiscoveryAgent filters out unverified rumors/blogs from official queries."""
    agent = EventDiscoveryAgent()

    # 1. Official bulletin
    b_official = HazardBulletin(
        bulletin_id="b_sz_official",
        source_authority="Shenzhen Meteorological Bureau",
        source_channel="weather.sz.gov.cn",
        published_at="2023-09-07T18:00:00",
        hazard_type="rainstorm",
        severity_level=HazardSeverityLevel.RED,
        raw_text="Official red rainstorm warning",
    )
    # 2. Rumor bulletin
    b_rumor = HazardBulletin(
        bulletin_id="b_social_rumor",
        source_authority="Internet User",
        source_channel="https://twitter.com/rumor",
        published_at="2023-09-07T18:05:00",
        hazard_type="rainstorm",
        severity_level=HazardSeverityLevel.RED,
        raw_text="Heard there is red rainstorm",
    )

    agent.register_bulletin(b_official)
    agent.register_bulletin(b_rumor)

    query = EventQuery(city="Shenzhen", keywords=["rainstorm"])

    # Query with require_verified=True only returns official bulletin
    verified_results = agent.discover_events(query, require_verified=True)
    assert len(verified_results) == 1
    assert verified_results[0].bulletin_id == "b_sz_official"
    assert verified_results[0].is_verified_official is True
