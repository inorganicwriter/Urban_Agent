"""LLM Supervisor Reviewer for evidence audit, targeted inquiry delegation, and round budgeting."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
import pandas as pd

from urban_resilience.llm.client import LLMClient
from urban_resilience.llm.schemas import PlannerDecision

logger = logging.getLogger(__name__)

ALLOWED_SUBAGENTS = ["temporal", "flexibility", "predictability", "spatial", "functional"]
ALLOWED_ACTIONS = ["targeted_inquiry", "reconcile_conflict", "verify_anomaly"]


class LLMSupervisorReviewer:
    """Evaluates cross-agent domain evidence and decides whether targeted follow-up tasks are required."""

    def __init__(
        self,
        llm_client: LLMClient,
        max_rounds: int = 2,
        budget_token_limit: int = 15000,
        prompt_version: str = "v1.0.0",
    ):
        self.llm_client = llm_client
        self.max_rounds = max_rounds
        self.budget_token_limit = budget_token_limit
        self.prompt_version = prompt_version
        self.decision_history: List[Dict[str, Any]] = []

    def _build_system_prompt(self) -> str:
        return (
            "You are the LLM Supervisor Reviewer for urban resilience assessment.\n"
            f"Prompt version: '{self.prompt_version}'. Max follow-up rounds: {self.max_rounds}.\n\n"
            "GOVERNANCE RULES:\n"
            "1. You must review domain evidence, localized cross-domain conflicts, and anomalies from the EvidenceBus.\n"
            "2. Prioritize delegating follow-up tasks to resolve localized cross-domain conflicts (which specify zone_id and multiple conflicting domains).\n"
            "3. You can decide action='delegate' to request targeted follow-up analysis, or action='stop' to proceed to synthesis.\n"
            f"4. You are STRICTLY RESTRICTED to whitelisted agents: {ALLOWED_SUBAGENTS}.\n"
            f"5. You can only request actions from the predefined catalog: {ALLOWED_ACTIONS}.\n"
            "6. You MUST NOT invent arbitrary new features or unapproved computational actions.\n"
            "7. Conflicting evidence between perspectives must be preserved if unresolved; do not force consensus.\n"
            "8. Maximum 2 rounds allowed. Output must strictly conform to PlannerDecision schema."
        )

    def review_and_decide(
        self,
        round_num: int,
        evidence_summary: Dict[str, Any],
        conflicts: List[Any],
        zone_features: pd.DataFrame,
        tokens_spent: int = 0,
        structured_conflicts: Optional[List[Any]] = None,
    ) -> PlannerDecision:
        """Reviews evidence and returns a sanitized PlannerDecision."""
        if round_num > self.max_rounds:
            logger.info("LLMSupervisorReviewer: Reached max rounds (%d); stopping follow-up loop.", self.max_rounds)
            dec = PlannerDecision(
                action="stop",
                reasoning=f"Round limit reached: current round {round_num} > max_rounds {self.max_rounds}",
                budget_consumed=float(tokens_spent),
            )
            self.decision_history.append({"round": round_num, "decision": dec.model_dump()})
            return dec

        if tokens_spent >= self.budget_token_limit:
            logger.info("LLMSupervisorReviewer: Budget limit (%d) reached; stopping follow-up loop.", self.budget_token_limit)
            dec = PlannerDecision(
                action="stop",
                reasoning=f"Token budget exhausted: {tokens_spent} >= {self.budget_token_limit}",
                budget_consumed=float(tokens_spent),
            )
            self.decision_history.append({"round": round_num, "decision": dec.model_dump()})
            return dec

        s_conf = structured_conflicts or []
        cross_domain_items = []
        anomaly_items = []
        unlocalized_items = []
        for c in s_conf:
            k = c.get("kind") if isinstance(c, dict) else getattr(c, "kind", None)
            zid = c.get("zone_id") if isinstance(c, dict) else getattr(c, "zone_id", None)
            doms = list(c.get("domains", [])) if isinstance(c, dict) else list(getattr(c, "domains", []))
            msg = str(c.get("message", "")) if isinstance(c, dict) else str(getattr(c, "message", ""))
            if k == "cross_domain" and zid:
                cross_domain_items.append({"zone_id": zid, "domains": doms, "message": msg})
            elif k == "cross_domain" and not zid:
                unlocalized_items.append({"domains": doms, "message": msg})
            elif k == "anomaly":
                anomaly_items.append({"zone_id": zid, "domains": doms, "message": msg})

        evidence_brief = {
            "registered_domains": list(evidence_summary.keys()),
            "localized_conflict_count": len(cross_domain_items),
            "cross_domain_conflicts": cross_domain_items[:5],
            "anomalies": anomaly_items[:5],
            "unlocalized_conflicts": unlocalized_items[:5],
            "all_conflicts": [str(c) for c in conflicts[:5]],
            "feature_columns": list(zone_features.columns),
            "zones_count": len(zone_features),
        }

        messages = [
            {"role": "system", "content": self._build_system_prompt()},
            {
                "role": "user",
                "content": (
                    f"Round #{round_num} Review:\n"
                    f"Evidence brief:\n{evidence_brief}\n\n"
                    f"Decide whether to 'delegate' targeted follow-up tasks to resolve uncertainties/anomalies, or 'stop'."
                ),
            },
        ]

        try:
            raw = self.llm_client.complete_json(
                messages=messages,
                schema=PlannerDecision.model_json_schema(),
            )
            decision = PlannerDecision.model_validate(raw)
        except Exception as exc:
            logger.warning("LLMSupervisorReviewer failed to parse decision: %s. Defaulting to 'stop'.", exc)
            decision = PlannerDecision(
                action="stop",
                reasoning=f"Fallback stop due to LLM error: {exc}",
                budget_consumed=float(tokens_spent),
            )

        # Sanitize tasks against whitelist
        sanitized_tasks = []
        if decision.action == "delegate":
            for t in decision.tasks:
                agent = t.get("agent_name", "")
                action = t.get("action", "")
                if agent in ALLOWED_SUBAGENTS and action in ALLOWED_ACTIONS:
                    payload = t.setdefault("payload", {})
                    payload.setdefault("follow_up_source", "synthetic_refined_window")
                    payload.setdefault("query_context", {"refined_window_hours": 72, "baseline_type": "7day_profile"})
                    sanitized_tasks.append(t)
                else:
                    logger.warning(
                        "LLMSupervisorReviewer: Rejected unauthorized task (agent='%s', action='%s')",
                        agent,
                        action,
                    )

            if not sanitized_tasks:
                decision.action = "stop"
                decision.reasoning = (decision.reasoning or "") + " [No valid tasks remained after whitelist filtering; stopping]"
            else:
                decision.tasks = sanitized_tasks

        self.decision_history.append({"round": round_num, "decision": decision.model_dump()})
        return decision
