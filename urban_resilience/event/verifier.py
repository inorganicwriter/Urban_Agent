"""Verification rules and validators for authenticating official disaster bulletin sources."""

from __future__ import annotations

import logging
from typing import List, Set, Tuple

logger = logging.getLogger(__name__)

# Recognized official meteorological & emergency authorities
OFFICIAL_AUTHORITIES: Set[str] = {
    "shenzhen meteorological bureau",
    "深圳市气象局",
    "cma",
    "china meteorological administration",
    "中国气象局",
    "shenzhen emergency management bureau",
    "深圳市应急管理局",
    "mem",
    "ministry of emergency management",
    "中华人民共和国应急管理部",
    "shenzhen water authority",
    "深圳市水务局",
    "national meteorological center",
    "国家气象中心",
    "guangdong meteorological bureau",
    "广东省气象局",
}

# Recognized government official TLDs and trusted portal domains
OFFICIAL_DOMAINS: List[str] = [
    ".gov.cn",
    "weather.sz.gov.cn",
    "yjgl.sz.gov.cn",
    "swj.sz.gov.cn",
    "cma.gov.cn",
    "mem.gov.cn",
    "gd.gov.cn",
]

# Explicitly unauthorized unverified channels (blogs, personal social media, forums)
UNAUTHORIZED_CHANNELS: List[str] = [
    "twitter.com",
    "x.com",
    "blog",
    "personal_weibo",
    "reddit.com",
    "tieba.baidu.com",
    "medium.com",
    "unverified_telegram",
]


class OfficialSourceVerifier:
    """Validates that event bulletins strictly originate from authorized government agencies."""

    def __init__(
        self,
        allowed_authorities: Set[str] | None = None,
        allowed_domains: List[str] | None = None,
    ):
        self.allowed_authorities = (
            {a.lower().strip() for a in allowed_authorities}
            if allowed_authorities is not None
            else OFFICIAL_AUTHORITIES
        )
        self.allowed_domains = allowed_domains or OFFICIAL_DOMAINS

    def verify_source(self, authority: str, channel: str = "") -> Tuple[bool, str]:
        """Validates the authority name and channel.

        Args:
            authority: The declared publishing authority.
            channel: The website URL, API endpoint, or feed channel.

        Returns:
            Tuple of (is_verified: bool, reason: str).
        """
        clean_auth = authority.lower().strip()
        clean_chan = channel.lower().strip()

        # 1. Check for explicitly banned / unverified social channels
        for bad_chan in UNAUTHORIZED_CHANNELS:
            if bad_chan in clean_chan:
                msg = f"Rejected: Channel '{channel}' is an unverified external source ({bad_chan})."
                logger.warning(msg)
                return False, msg

        # 2. Check authority match
        authority_match = any(
            allowed in clean_auth or clean_auth in allowed for allowed in self.allowed_authorities
        )

        # 3. Check domain match (if channel is provided)
        domain_match = False
        if clean_chan:
            domain_match = any(dom in clean_chan for dom in self.allowed_domains)

        if authority_match:
            return True, f"Verified: Authority '{authority}' is recognized official agency."

        if domain_match:
            return True, f"Verified: Channel '{channel}' belongs to official government domain."

        msg = (
            f"Rejected: Authority '{authority}' is not in the official emergency agency whitelist, "
            f"and channel '{channel}' does not resolve to an official government domain."
        )
        logger.warning(msg)
        return False, msg
