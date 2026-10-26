"""Offline tools for querying and filtering disaster bulletins without network dependency."""

from __future__ import annotations

import logging
from typing import List

from urban_resilience.event.schemas import EventQuery, HazardBulletin, HazardSeverityLevel

logger = logging.getLogger(__name__)

# Severity hierarchy for threshold filtering
SEVERITY_ORDER = {
    HazardSeverityLevel.BLUE: 1,
    HazardSeverityLevel.YELLOW: 2,
    HazardSeverityLevel.ORANGE: 3,
    HazardSeverityLevel.RED: 4,
    HazardSeverityLevel.UNKNOWN: 0,
}


def search_bulletins_in_store(
    query: EventQuery,
    store: List[HazardBulletin],
) -> List[HazardBulletin]:
    """Filters offline store of bulletins matching query parameters.

    Args:
        query: Search parameters (keywords, city, time bounds, min_severity).
        store: Pre-loaded list of HazardBulletin records.

    Returns:
        Filtered list of matching HazardBulletin instances.
    """
    results: List[HazardBulletin] = []
    min_rank = SEVERITY_ORDER.get(query.min_severity, 0) if query.min_severity else 0

    for item in store:
        # Check severity threshold
        if SEVERITY_ORDER.get(item.severity_level, 0) < min_rank:
            continue

        # Check time window if specified
        if query.time_start and item.published_at < query.time_start:
            continue
        if query.time_end and item.published_at > query.time_end:
            continue

        # Check keywords if specified
        if query.keywords:
            combined_text = (item.raw_text + " " + item.source_authority + " " + item.hazard_type).lower()
            if not any(kw.lower() in combined_text for kw in query.keywords):
                continue

        results.append(item)

    logger.info("Search executed for query keywords=%s, matched %d bulletins", query.keywords, len(results))
    return results
