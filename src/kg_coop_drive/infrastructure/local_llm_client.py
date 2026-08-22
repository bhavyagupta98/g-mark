from __future__ import annotations

"""Local OpenAI-compatible LLM client for planning-awareness reranking.

This client is designed for self-hosted local model servers that expose an
OpenAI-compatible chat completion endpoint, such as vLLM or llama.cpp-based
servers.
"""

import json
import re
from dataclasses import dataclass
from typing import Any
from urllib import request

from kg_coop_drive.application.qa.planning_awareness import (
    PlanningAwarenessBatchLLMClient,
    PlanningAwarenessLLMRankItem,
    PlanningAwarenessLLMRankedItem,
)
from kg_coop_drive.application.qa.v2vgotqa_router import (
    NotableObjectLLMRankItem,
    NotableObjectLLMRankedItem,
    NotableObjectsBatchLLMClient,
    OccludingObjectLLMRankItem,
    OccludingObjectLLMRankedItem,
    OccludingObjectsBatchLLMClient,
)


@dataclass(frozen=True)
class LocalOpenAICompatibleLLMConfig:
    """Connection/configuration for a local OpenAI-compatible server."""

    base_url: str
    model: str
    api_key: str = "local-token"
    timeout_seconds: float = 180.0
    temperature: float = 0.0
    max_tokens: int = 192


class LocalOpenAICompatibleLLMClient(
    PlanningAwarenessBatchLLMClient,
    NotableObjectsBatchLLMClient,
    OccludingObjectsBatchLLMClient,
):
    """Calls a local OpenAI-compatible chat completion endpoint."""

    def __init__(self, config: LocalOpenAICompatibleLLMConfig) -> None:
        self._config = config

    def rerank_candidates(
        self,
        asker_agent_id: str,
        raw_question: str,
        candidates: tuple[PlanningAwarenessLLMRankItem, ...],
    ) -> tuple[PlanningAwarenessLLMRankedItem, ...]:
        if not candidates:
            return ()

        payload = {
            "model": self._config.model,
            "temperature": self._config.temperature,
            "max_tokens": self._config.max_tokens,
            "response_format": {"type": "json_object"},
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a driving-safety reranker. "
                        "Given a benchmark question and a shortlist of already-grounded "
                        "graph candidates, return JSON only. "
                        "Return only a JSON object with a `ranked_candidates` list. "
                        "Each item must contain only `object_id` and `score`. "
                        "Score each candidate in [0,1] by how relevant it is to the "
                        "planning-awareness question. Favor objects that are near the "
                        "planned trajectory, visible or occluded in a safety-relevant way, "
                        "and plausible hazards for the asking vehicle."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "task": "rerank_planning_awareness_candidates",
                            "asker_agent_id": asker_agent_id,
                            "question": raw_question,
                            "candidates": [
                                {
                                    "object_id": candidate.object_id,
                                    "object_type": candidate.object_type,
                                    "visibility_state": candidate.visibility_state,
                                    "distance_to_trajectory": round(candidate.distance_to_trajectory, 3),
                                    "base_score": round(candidate.base_score, 3),
                                    "status": candidate.status,
                                }
                                for candidate in candidates
                            ],
                            "required_output_schema": {
                                "ranked_candidates": [
                                    {
                                        "object_id": "string",
                                        "score": "float in [0,1]",
                                    }
                                ]
                            },
                        }
                    ),
                },
            ],
        }
        response_payload = self._post_json("/v1/chat/completions", payload)
        message_content = self._extract_message_content(response_payload)
        parsed = self._parse_ranking_payload(message_content)
        ranked_candidates = parsed.get("ranked_candidates", [])
        if not isinstance(ranked_candidates, list):
            raise ValueError("Local LLM response missing `ranked_candidates` list.")

        results: list[PlanningAwarenessLLMRankedItem] = []
        for item in ranked_candidates:
            if not isinstance(item, dict):
                continue
            object_id = str(item.get("object_id", "")).strip()
            if not object_id:
                continue
            score = float(item.get("score", 0.0))
            score = max(0.0, min(1.0, score))
            results.append(
                PlanningAwarenessLLMRankedItem(
                    object_id=object_id,
                    score=score,
                    rationale=(),
                )
            )
        return tuple(results)

    def rerank_occluding_candidates(
        self,
        asker_agent_id: str,
        raw_question: str,
        candidates: tuple[OccludingObjectLLMRankItem, ...],
    ) -> tuple[OccludingObjectLLMRankedItem, ...]:
        if not candidates:
            return ()

        payload = {
            "model": self._config.model,
            "temperature": self._config.temperature,
            "max_tokens": self._config.max_tokens,
            "response_format": {"type": "json_object"},
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a driving-occlusion reranker. "
                        "Given a shortlist of visible blocker candidates and the hidden "
                        "objects they may occlude, return JSON only. "
                        "Return only a JSON object with a `ranked_candidates` list. "
                        "Each item must contain only `object_id` and `score`. "
                        "Favor visible objects that plausibly block important hidden "
                        "objects from the asking vehicle's viewpoint. Prefer blockers "
                        "with strong angular alignment, multiple hidden targets, and "
                        "hidden targets that are near the planned trajectory."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "task": "rerank_occluding_object_candidates",
                            "asker_agent_id": asker_agent_id,
                            "question": raw_question,
                            "candidates": [
                                {
                                    "object_id": candidate.object_id,
                                    "object_type": candidate.object_type,
                                    "distance_to_asker": round(candidate.distance_to_asker, 3),
                                    "distance_to_trajectory": round(candidate.distance_to_trajectory, 3),
                                    "base_score": round(candidate.base_score, 3),
                                    "status": candidate.status,
                                    "support_count": candidate.support_count,
                                    "best_alignment_radians": round(candidate.best_alignment_radians, 4),
                                    "aligned_hidden_object_ids": list(candidate.aligned_hidden_object_ids),
                                    "aligned_hidden_distances_to_trajectory": [
                                        round(value, 3)
                                        for value in candidate.aligned_hidden_distances_to_trajectory
                                    ],
                                }
                                for candidate in candidates
                            ],
                            "required_output_schema": {
                                "ranked_candidates": [
                                    {
                                        "object_id": "string",
                                        "score": "float in [0,1]",
                                    }
                                ]
                            },
                        }
                    ),
                },
            ],
        }
        response_payload = self._post_json("/v1/chat/completions", payload)
        message_content = self._extract_message_content(response_payload)
        parsed = self._parse_ranking_payload(message_content)
        ranked_candidates = parsed.get("ranked_candidates", [])
        if not isinstance(ranked_candidates, list):
            raise ValueError("Local LLM response missing `ranked_candidates` list.")

        results: list[OccludingObjectLLMRankedItem] = []
        for item in ranked_candidates:
            if not isinstance(item, dict):
                continue
            object_id = str(item.get("object_id", "")).strip()
            if not object_id:
                continue
            score = float(item.get("score", 0.0))
            score = max(0.0, min(1.0, score))
            results.append(
                OccludingObjectLLMRankedItem(
                    object_id=object_id,
                    score=score,
                )
            )
        return tuple(results)

    def rerank_notable_candidates(
        self,
        asker_agent_id: str,
        raw_question: str,
        candidates: tuple[NotableObjectLLMRankItem, ...],
    ) -> tuple[NotableObjectLLMRankedItem, ...]:
        if not candidates:
            return ()

        payload = {
            "model": self._config.model,
            "temperature": self._config.temperature,
            "max_tokens": self._config.max_tokens,
            "response_format": {"type": "json_object"},
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a driving-scene reranker. "
                        "Given a shortlist of visible objects, return JSON only. "
                        "Return only a JSON object with a `ranked_candidates` list. "
                        "Each item must contain only `object_id` and `score`. "
                        "Favor visible objects that are most notable and relevant to the "
                        "asking vehicle's near-term planned path. Prefer objects near the "
                        "planned trajectory or first waypoint, with strong support and lower "
                        "conflict or uncertainty."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "task": "rerank_notable_object_candidates",
                            "asker_agent_id": asker_agent_id,
                            "question": raw_question,
                            "candidates": [
                                {
                                    "object_id": candidate.object_id,
                                    "object_type": candidate.object_type,
                                    "distance_to_asker": round(candidate.distance_to_asker, 3),
                                    "distance_to_trajectory": round(candidate.distance_to_trajectory, 3),
                                    "distance_to_first_waypoint": round(candidate.distance_to_first_waypoint, 3),
                                    "base_score": round(candidate.base_score, 3),
                                    "status": candidate.status,
                                    "support_count": candidate.support_count,
                                    "conflict_score": round(candidate.conflict_score, 3),
                                    "uncertainty_score": round(candidate.uncertainty_score, 3),
                                }
                                for candidate in candidates
                            ],
                            "required_output_schema": {
                                "ranked_candidates": [
                                    {
                                        "object_id": "string",
                                        "score": "float in [0,1]",
                                    }
                                ]
                            },
                        }
                    ),
                },
            ],
        }
        response_payload = self._post_json("/v1/chat/completions", payload)
        message_content = self._extract_message_content(response_payload)
        parsed = self._parse_ranking_payload(message_content)
        ranked_candidates = parsed.get("ranked_candidates", [])
        if not isinstance(ranked_candidates, list):
            raise ValueError("Local LLM response missing `ranked_candidates` list.")

        results: list[NotableObjectLLMRankedItem] = []
        for item in ranked_candidates:
            if not isinstance(item, dict):
                continue
            object_id = str(item.get("object_id", "")).strip()
            if not object_id:
                continue
            score = float(item.get("score", 0.0))
            score = max(0.0, min(1.0, score))
            results.append(
                NotableObjectLLMRankedItem(
                    object_id=object_id,
                    score=score,
                )
            )
        return tuple(results)

    def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self._config.base_url.rstrip('/')}{path}"
        data = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self._config.api_key:
            headers["Authorization"] = f"Bearer {self._config.api_key}"
        req = request.Request(
            url,
            data=data,
            headers=headers,
            method="POST",
        )
        with request.urlopen(req, timeout=self._config.timeout_seconds) as response:
            body = response.read().decode("utf-8")
        parsed = json.loads(body)
        if not isinstance(parsed, dict):
            raise ValueError("Local LLM response was not a JSON object.")
        return parsed

    @staticmethod
    def _extract_message_content(response_payload: dict[str, Any]) -> str:
        choices = response_payload.get("choices", [])
        if not isinstance(choices, list) or not choices:
            raise ValueError("Local LLM response missing `choices`.")
        first_choice = choices[0]
        if not isinstance(first_choice, dict):
            raise ValueError("Local LLM response `choices[0]` was not an object.")
        message = first_choice.get("message", {})
        if not isinstance(message, dict):
            raise ValueError("Local LLM response missing `message` object.")
        content = message.get("content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            text_parts = [
                str(part.get("text", ""))
                for part in content
                if isinstance(part, dict) and part.get("type") == "text"
            ]
            return "".join(text_parts)
        raise ValueError("Unsupported `message.content` format from local LLM response.")

    @staticmethod
    def _parse_ranking_payload(message_content: str) -> dict[str, Any]:
        try:
            parsed = json.loads(message_content)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

        match = re.search(r"\{.*\}", message_content, re.DOTALL)
        if match:
            candidate_json = match.group(0)
            try:
                parsed = json.loads(candidate_json)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                pass

        ranked_candidates: list[dict[str, Any]] = []
        for object_id, score_text in re.findall(
            r'"object_id"\s*:\s*"([^"]+)"[^}]*?"score"\s*:\s*([0-9]*\.?[0-9]+)',
            message_content,
            flags=re.DOTALL,
        ):
            try:
                score = max(0.0, min(1.0, float(score_text)))
            except ValueError:
                continue
            ranked_candidates.append({"object_id": object_id, "score": score})

        if ranked_candidates:
            return {"ranked_candidates": ranked_candidates}

        raise ValueError(f"Could not parse local LLM ranking payload: {message_content!r}")
