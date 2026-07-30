"""
query_engine.py — 统一查询协议

所有参考模型和目标模型在同一协议下采样：
- max_tokens=1, temperature=0, top_p=1
- presence_penalty=0, frequency_penalty=0
- 极简 system prompt

支持 OpenAI-compatible 和 Anthropic 两类 API。
"""

from __future__ import annotations

import asyncio
import copy
import json
import os
import time
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import httpx

from .schema import Probe, SingleTokenResult

logger = logging.getLogger(__name__)

# ── 默认查询参数 ────────────────────────────────────────────────

DEFAULT_SYSTEM_PROMPT = "直接续写，不要解释，不要补充说明。"

DEFAULT_QUERY_PARAMS = {
    "max_tokens": 1,
    "temperature": 0,
    "top_p": 1,
    "presence_penalty": 0,
    "frequency_penalty": 0,
}

COMPLETIONS_ENDPOINT_MODELS = {
    "Qwen/Qwen2.5-7B",
    "Qwen/Qwen3-8B-Base",
    "LLM-Research/Meta-Llama-3-8B",
    "LLM-Research/Meta-Llama-3.1-8B",
}

COMPLETIONS_ENDPOINT_PORTS = {
    ":18103/",
}

THINKING_MARKER_OUTPUTS = {
    "<think>",
    "Thinking",
    "thinking",
    "嗯",
    "首先",
    "好的",
    "Hmm",
    "我们",
    "We",
    "Okay",
}

DEEPSEEK_CHAT_PREFIX_MESSAGE_MODES = {
    "deepseek_chat_prefix",
    "chat_prefix_completion",
}


def _is_local_qwen_thinking_model(model: str, base_url: str) -> bool:
    normalized_url = base_url.rstrip("/")
    return (
        model.startswith("Qwen/Qwen3")
        and (
            normalized_url.startswith("http://127.0.0.1:")
            or normalized_url.startswith("http://localhost:")
        )
    )


@dataclass
class APIConfig:
    """模型 API 配置"""
    provider: str         # "openai" | "anthropic"
    model: str
    api_key: str
    base_url: str = "https://api.openai.com/v1"
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    timeout: float = 30.0
    max_retries: int = 3
    retry_delay: float = 1.0
    request_interval: float = 0.0
    empty_output_retries: int = 0
    empty_retry_delay: float = 0.0
    message_mode: str = "user_prompt"
    endpoint: str = "chat_completions"
    extra_body: dict[str, Any] = field(default_factory=dict)
    output_normalization: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(
        cls,
        d: dict,
        provider: str = "openai",
        defaults: Optional[dict] = None,
    ) -> "APIConfig":
        if defaults:
            merged = copy.deepcopy(defaults)
            merged.update(d)
            d = merged
        api_key = d.get("api_key", "")
        # Resolve env vars like ${OPENAI_API_KEY}
        if api_key.startswith("${") and api_key.endswith("}"):
            env_var = api_key[2:-1]
            api_key = os.environ.get(env_var, "")
        timeout = d.get("timeout", d.get("request_timeout", 30.0))
        model = d.get("model", "")
        endpoint = d.get("endpoint", d.get("api_endpoint", "chat_completions"))
        force_completions = os.environ.get("TOKENIZER_FP_FORCE_COMPLETIONS", "1")
        if model in COMPLETIONS_ENDPOINT_MODELS and force_completions != "0":
            endpoint = "completions"
        extra_body = copy.deepcopy(d.get("extra_body", {}))
        if _is_local_qwen_thinking_model(model, d.get("base_url", "")):
            extra_body.setdefault("enable_thinking", False)
            chat_template_kwargs = extra_body.setdefault("chat_template_kwargs", {})
            if isinstance(chat_template_kwargs, dict):
                chat_template_kwargs.setdefault("enable_thinking", False)
        return cls(
            provider=provider,
            model=model,
            api_key=api_key,
            base_url=d.get("base_url", "https://api.openai.com/v1"),
            system_prompt=d.get("system_prompt", DEFAULT_SYSTEM_PROMPT),
            timeout=float(timeout),
            max_retries=int(d.get("max_retries", 3)),
            retry_delay=float(d.get("retry_delay", 1.0)),
            request_interval=float(d.get("request_interval", 0.0)),
            empty_output_retries=int(
                d.get("empty_output_retries", d.get("max_empty_retries", 0))
            ),
            empty_retry_delay=float(d.get("empty_retry_delay", 0.0)),
            message_mode=d.get("message_mode", "user_prompt"),
            endpoint=endpoint,
            extra_body=extra_body,
            output_normalization=d.get("output_normalization", {}),
        )


class QueryEngine:
    """
    统一查询引擎。

    对所有模型使用相同的 query protocol，
    确保特征分布的可比性。
    """

    def __init__(
        self,
        config: APIConfig,
        model_name: str = "",
        concurrency: int = 5,
    ):
        self.config = config
        self.model_name = model_name or config.model
        self.concurrency = concurrency
        self._semaphore = asyncio.Semaphore(concurrency)
        self._rate_lock = asyncio.Lock()
        self._last_request_at = 0.0
        self._use_completions = (
            config.endpoint in {"completion", "completions"}
            or self._should_force_completions()
        )
        logger.info(
            "QueryEngine initialized for %s using endpoint=%s base_url=%s",
            self.model_name,
            "completions" if self._use_completions else "chat_completions",
            self.config.base_url,
        )

    async def query_single(
        self,
        probe: Probe,
        client: httpx.AsyncClient,
    ) -> SingleTokenResult:
        """发送单条 probe 查询，返回单 token 结果"""
        async with self._semaphore:
            await self._wait_for_request_slot()
            return await self._dispatch(probe, client)

    async def _wait_for_request_slot(self):
        if self.config.request_interval <= 0:
            return
        async with self._rate_lock:
            now = time.monotonic()
            wait = self.config.request_interval - (now - self._last_request_at)
            if wait > 0:
                await asyncio.sleep(wait)
            self._last_request_at = time.monotonic()

    async def _dispatch(
        self,
        probe: Probe,
        client: httpx.AsyncClient,
    ) -> SingleTokenResult:
        if self.config.provider == "anthropic":
            return await self._query_anthropic(probe, client)
        else:
            return await self._query_openai_compatible(probe, client)

    async def _query_openai_compatible(
        self,
        probe: Probe,
        client: httpx.AsyncClient,
    ) -> SingleTokenResult:
        """OpenAI / OpenAI-compatible API 查询"""
        if self._should_force_completions():
            self._use_completions = True
        if self._use_completions:
            return await self._query_openai_completion(probe, client)

        url = f"{self.config.base_url.rstrip('/')}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.config.model,
            "messages": self._build_openai_messages(probe),
            "max_tokens": DEFAULT_QUERY_PARAMS["max_tokens"],
            "temperature": DEFAULT_QUERY_PARAMS["temperature"],
            "top_p": DEFAULT_QUERY_PARAMS["top_p"],
            "presence_penalty": DEFAULT_QUERY_PARAMS["presence_penalty"],
            "frequency_penalty": DEFAULT_QUERY_PARAMS["frequency_penalty"],
        }
        payload.update(self.config.extra_body)
        payload.update(probe.metadata.get("_query_extra_body", {}))

        error_attempt = 0
        request_attempt_count = 0
        empty_response_summaries: list[dict[str, Any]] = []
        max_error_attempts = max(1, self.config.max_retries)

        while True:
            try:
                request_attempt_count += 1
                t0 = time.monotonic()
                resp = await client.post(
                    url,
                    json=payload,
                    headers=headers,
                    timeout=self.config.timeout,
                )
                latency = (time.monotonic() - t0) * 1000

                if resp.status_code == 429:
                    if error_attempt >= max_error_attempts - 1:
                        error = self._format_http_error(resp)
                        logger.error(f"Query failed for probe {probe.id}: {error}")
                        return self._build_result(probe, "", 0.0, {"error": error})
                    wait = self._retry_wait(resp, error_attempt)
                    error_attempt += 1
                    logger.warning(f"Rate limited, waiting {wait}s...")
                    await asyncio.sleep(wait)
                    continue
                if resp.status_code in {500, 502, 503, 504}:
                    if error_attempt >= max_error_attempts - 1:
                        error = self._format_http_error(resp)
                        logger.error(f"Query failed for probe {probe.id}: {error}")
                        return self._build_result(probe, "", 0.0, {"error": error})
                    wait = self.config.retry_delay * (2 ** error_attempt)
                    error_attempt += 1
                    logger.warning(
                        f"Server error HTTP {resp.status_code}, waiting {wait}s..."
                    )
                    await asyncio.sleep(wait)
                    continue
                if 400 <= resp.status_code < 500:
                    error = self._format_http_error(resp)
                    if self._is_missing_chat_template_error(resp):
                        self._use_completions = True
                        logger.warning(
                            "Chat completions rejected for probe %s because the "
                            "server has no chat template; retrying /completions.",
                            probe.id,
                        )
                        return await self._query_openai_completion(probe, client)
                    logger.error(f"Query failed for probe {probe.id}: {error}")
                    return self._build_result(probe, "", 0.0, {"error": error})

                resp.raise_for_status()
                data = resp.json()

                output_text = ""
                choices = data.get("choices", [])
                if choices:
                    msg = choices[0].get("message", {})
                    output_text = msg.get("content", "") or ""

                if (
                    output_text == ""
                    and len(empty_response_summaries)
                    < self.config.empty_output_retries
                ):
                    empty_response_summaries.append(
                        self._summarize_empty_response(data)
                    )
                    logger.warning(
                        "Empty output for probe %s from %s; retrying (%s/%s)",
                        probe.id,
                        self.model_name,
                        len(empty_response_summaries),
                        self.config.empty_output_retries,
                    )
                    await self._sleep_before_empty_retry(
                        len(empty_response_summaries)
                    )
                    continue

                raw = self._with_empty_retry_metadata(
                    data,
                    request_attempt_count=request_attempt_count,
                    empty_response_summaries=empty_response_summaries,
                    final_output_text=output_text,
                )
                return self._build_result(probe, output_text, latency, raw)

            except Exception as e:
                if error_attempt >= max_error_attempts - 1:
                    error = self._format_error(e)
                    logger.error(f"Query failed for probe {probe.id}: {error}")
                    return self._build_result(probe, "", 0.0, {"error": error})
                wait = self.config.retry_delay * (2 ** error_attempt)
                error_attempt += 1
                await asyncio.sleep(wait)

        return self._build_result(probe, "", 0.0, {"error": "max_retries_exceeded"})

    async def _query_openai_completion(
        self,
        probe: Probe,
        client: httpx.AsyncClient,
    ) -> SingleTokenResult:
        """OpenAI-compatible completions API for base models."""
        url = f"{self.config.base_url.rstrip('/')}/completions"
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.config.model,
            "prompt": probe.text,
            "max_tokens": DEFAULT_QUERY_PARAMS["max_tokens"],
            "temperature": DEFAULT_QUERY_PARAMS["temperature"],
            "top_p": DEFAULT_QUERY_PARAMS["top_p"],
            "presence_penalty": DEFAULT_QUERY_PARAMS["presence_penalty"],
            "frequency_penalty": DEFAULT_QUERY_PARAMS["frequency_penalty"],
        }
        payload.update(self.config.extra_body)
        payload.update(probe.metadata.get("_query_extra_body", {}))

        error_attempt = 0
        request_attempt_count = 0
        empty_response_summaries: list[dict[str, Any]] = []
        max_error_attempts = max(1, self.config.max_retries)

        while True:
            try:
                request_attempt_count += 1
                t0 = time.monotonic()
                resp = await client.post(
                    url,
                    json=payload,
                    headers=headers,
                    timeout=self.config.timeout,
                )
                latency = (time.monotonic() - t0) * 1000

                if resp.status_code == 429:
                    if error_attempt >= max_error_attempts - 1:
                        error = self._format_http_error(resp)
                        logger.error(f"Query failed for probe {probe.id}: {error}")
                        return self._build_result(probe, "", 0.0, {"error": error})
                    wait = self._retry_wait(resp, error_attempt)
                    error_attempt += 1
                    logger.warning(f"Rate limited, waiting {wait}s...")
                    await asyncio.sleep(wait)
                    continue
                if resp.status_code in {500, 502, 503, 504}:
                    if error_attempt >= max_error_attempts - 1:
                        error = self._format_http_error(resp)
                        logger.error(f"Query failed for probe {probe.id}: {error}")
                        return self._build_result(probe, "", 0.0, {"error": error})
                    wait = self.config.retry_delay * (2 ** error_attempt)
                    error_attempt += 1
                    logger.warning(
                        f"Server error HTTP {resp.status_code}, waiting {wait}s..."
                    )
                    await asyncio.sleep(wait)
                    continue
                if 400 <= resp.status_code < 500:
                    error = self._format_http_error(resp)
                    logger.error(f"Query failed for probe {probe.id}: {error}")
                    return self._build_result(probe, "", 0.0, {"error": error})

                resp.raise_for_status()
                data = resp.json()

                output_text = ""
                choices = data.get("choices", [])
                if choices:
                    output_text = choices[0].get("text", "") or ""

                if (
                    output_text == ""
                    and len(empty_response_summaries)
                    < self.config.empty_output_retries
                ):
                    empty_response_summaries.append(
                        self._summarize_empty_response(data)
                    )
                    logger.warning(
                        "Empty output for probe %s from %s; retrying (%s/%s)",
                        probe.id,
                        self.model_name,
                        len(empty_response_summaries),
                        self.config.empty_output_retries,
                    )
                    await self._sleep_before_empty_retry(
                        len(empty_response_summaries)
                    )
                    continue

                raw = self._with_empty_retry_metadata(
                    data,
                    request_attempt_count=request_attempt_count,
                    empty_response_summaries=empty_response_summaries,
                    final_output_text=output_text,
                )
                return self._build_result(probe, output_text, latency, raw)

            except Exception as e:
                if error_attempt >= max_error_attempts - 1:
                    error = self._format_error(e)
                    logger.error(f"Query failed for probe {probe.id}: {error}")
                    return self._build_result(probe, "", 0.0, {"error": error})
                wait = self.config.retry_delay * (2 ** error_attempt)
                error_attempt += 1
                await asyncio.sleep(wait)

        return self._build_result(probe, "", 0.0, {"error": "max_retries_exceeded"})

    def _build_openai_messages(self, probe: Probe) -> list[dict[str, Any]]:
        if self.config.message_mode in DEEPSEEK_CHAT_PREFIX_MESSAGE_MODES:
            return [
                {"role": "system", "content": self.config.system_prompt},
                {
                    "role": "user",
                    "content": (
                        "Continue the assistant message exactly. Output only the "
                        "immediate continuation, with no explanation, no preface, "
                        "and no newline."
                    ),
                },
                {"role": "assistant", "content": probe.text, "prefix": True},
            ]
        if self.config.message_mode == "assistant_prefill":
            return [
                {"role": "system", "content": self.config.system_prompt},
                {
                    "role": "user",
                    "content": (
                        "继续上一条 assistant 消息的文本，只输出紧接着的后续内容。"
                        "不要解释，不要另起段落，不要输出换行。"
                    ),
                },
                {"role": "assistant", "content": probe.text},
            ]
        return [
            {"role": "system", "content": self.config.system_prompt},
            {"role": "user", "content": probe.text},
        ]

    async def _query_anthropic(
        self,
        probe: Probe,
        client: httpx.AsyncClient,
    ) -> SingleTokenResult:
        """Anthropic API 查询"""
        url = "https://api.anthropic.com/v1/messages"
        headers = {
            "x-api-key": self.config.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.config.model,
            "system": self.config.system_prompt,
            "messages": [
                {"role": "user", "content": probe.text},
            ],
            "max_tokens": DEFAULT_QUERY_PARAMS["max_tokens"],
            "temperature": DEFAULT_QUERY_PARAMS["temperature"],
            "top_p": DEFAULT_QUERY_PARAMS["top_p"],
        }
        payload.update(self.config.extra_body)
        payload.update(probe.metadata.get("_query_extra_body", {}))

        error_attempt = 0
        request_attempt_count = 0
        empty_response_summaries: list[dict[str, Any]] = []
        max_error_attempts = max(1, self.config.max_retries)

        while True:
            try:
                request_attempt_count += 1
                t0 = time.monotonic()
                resp = await client.post(
                    url,
                    json=payload,
                    headers=headers,
                    timeout=self.config.timeout,
                )
                latency = (time.monotonic() - t0) * 1000

                if resp.status_code == 429:
                    if error_attempt >= max_error_attempts - 1:
                        error = self._format_http_error(resp)
                        logger.error(f"Query failed for probe {probe.id}: {error}")
                        return self._build_result(probe, "", 0.0, {"error": error})
                    wait = self._retry_wait(resp, error_attempt)
                    error_attempt += 1
                    logger.warning(f"Rate limited, waiting {wait}s...")
                    await asyncio.sleep(wait)
                    continue
                if resp.status_code in {500, 502, 503, 504}:
                    if error_attempt >= max_error_attempts - 1:
                        error = self._format_http_error(resp)
                        logger.error(f"Query failed for probe {probe.id}: {error}")
                        return self._build_result(probe, "", 0.0, {"error": error})
                    wait = self.config.retry_delay * (2 ** error_attempt)
                    error_attempt += 1
                    logger.warning(
                        f"Server error HTTP {resp.status_code}, waiting {wait}s..."
                    )
                    await asyncio.sleep(wait)
                    continue
                if 400 <= resp.status_code < 500:
                    error = self._format_http_error(resp)
                    logger.error(f"Query failed for probe {probe.id}: {error}")
                    return self._build_result(probe, "", 0.0, {"error": error})

                resp.raise_for_status()
                data = resp.json()

                output_text = ""
                content = data.get("content", [])
                if content:
                    output_text = content[0].get("text", "") or ""

                if (
                    output_text == ""
                    and len(empty_response_summaries)
                    < self.config.empty_output_retries
                ):
                    empty_response_summaries.append(
                        self._summarize_empty_response(data)
                    )
                    logger.warning(
                        "Empty output for probe %s from %s; retrying (%s/%s)",
                        probe.id,
                        self.model_name,
                        len(empty_response_summaries),
                        self.config.empty_output_retries,
                    )
                    await self._sleep_before_empty_retry(
                        len(empty_response_summaries)
                    )
                    continue

                raw = self._with_empty_retry_metadata(
                    data,
                    request_attempt_count=request_attempt_count,
                    empty_response_summaries=empty_response_summaries,
                    final_output_text=output_text,
                )
                return self._build_result(probe, output_text, latency, raw)

            except Exception as e:
                if error_attempt >= max_error_attempts - 1:
                    error = self._format_error(e)
                    logger.error(f"Query failed for probe {probe.id}: {error}")
                    return self._build_result(probe, "", 0.0, {"error": error})
                wait = self.config.retry_delay * (2 ** error_attempt)
                error_attempt += 1
                await asyncio.sleep(wait)

        return self._build_result(probe, "", 0.0, {"error": "max_retries_exceeded"})

    async def _sleep_before_empty_retry(self, retry_count: int):
        if self.config.empty_retry_delay <= 0:
            return
        await asyncio.sleep(self.config.empty_retry_delay)

    def _with_empty_retry_metadata(
        self,
        raw: dict,
        request_attempt_count: int,
        empty_response_summaries: list[dict[str, Any]],
        final_output_text: str,
    ) -> dict:
        if not empty_response_summaries:
            return raw
        raw = dict(raw)
        retry_meta = {
            "configured_retries": self.config.empty_output_retries,
            "attempt_count": request_attempt_count,
            "empty_retries_used": len(empty_response_summaries),
            "empty_response_count": len(empty_response_summaries)
            + (1 if final_output_text == "" else 0),
            "recovered_after_empty": final_output_text != "",
            "final_is_empty": final_output_text == "",
            "empty_responses": empty_response_summaries,
        }
        if final_output_text == "":
            retry_meta["final_empty_response"] = self._summarize_empty_response(raw)
        raw["_empty_output_retry"] = retry_meta
        return raw

    @staticmethod
    def _summarize_empty_response(raw: dict) -> dict[str, Any]:
        usage = raw.get("usage", {}) if isinstance(raw, dict) else {}
        completion_details = usage.get("completion_tokens_details", {})
        summary: dict[str, Any] = {
            "completion_tokens": usage.get("completion_tokens"),
            "reasoning_tokens": (
                completion_details.get("reasoning_tokens")
                if isinstance(completion_details, dict)
                else None
            ),
        }
        choices = raw.get("choices", []) if isinstance(raw, dict) else []
        if choices:
            choice = choices[0]
            summary["finish_reason"] = choice.get("finish_reason")
            message = choice.get("message")
            if isinstance(message, dict):
                summary["message_keys"] = sorted(message.keys())
            else:
                summary["choice_keys"] = sorted(choice.keys())
        elif isinstance(raw.get("content"), list):
            summary["content_blocks"] = len(raw.get("content", []))
            summary["stop_reason"] = raw.get("stop_reason")
        return summary

    def _build_result(
        self,
        probe: Probe,
        output_text: str,
        latency_ms: float,
        raw: dict,
    ) -> SingleTokenResult:
        raw_output_text = output_text
        output_text, normalization_meta = self._normalize_output(output_text)
        if normalization_meta:
            raw = dict(raw)
            raw["_output_normalization"] = normalization_meta
            if normalization_meta["changed"]:
                raw["_raw_output_text"] = raw_output_text

        return SingleTokenResult(
            probe_id=probe.id,
            model_name=self.model_name,
            output_text=output_text,
            char_length=len(output_text),
            byte_length=len(output_text.encode("utf-8")),
            has_leading_space=output_text.startswith(" ") if output_text else False,
            has_leading_newline=output_text.startswith("\n") if output_text else False,
            is_empty=(output_text == ""),
            latency_ms=latency_ms,
            raw_response=raw,
        )

    def _normalize_output(self, output_text: str) -> tuple[str, dict]:
        normalization = self.config.output_normalization or {}
        if not normalization:
            return output_text, {}

        normalized = output_text
        applied = []

        if normalization.get("strip_leading_newlines"):
            stripped = normalized.lstrip("\r\n")
            if stripped != normalized:
                normalized = stripped
                applied.append("strip_leading_newlines")

        return normalized, {
            "config": normalization,
            "applied": applied,
            "changed": normalized != output_text,
        }

    def _should_force_completions(self) -> bool:
        base_url = f"{self.config.base_url.rstrip('/')}/"
        return (
            self.config.model in COMPLETIONS_ENDPOINT_MODELS
            or any(port in base_url for port in COMPLETIONS_ENDPOINT_PORTS)
        )

    @staticmethod
    def _format_error(e: Exception) -> str:
        if isinstance(e, httpx.HTTPStatusError):
            return f"{type(e).__name__}: {QueryEngine._format_http_error(e.response)}"
        return f"{type(e).__name__}: {e}"

    @staticmethod
    def _format_http_error(response: httpx.Response) -> str:
        response_text = response.text[:500]
        request_url = ""
        try:
            request = response.request
        except RuntimeError:
            request = None
        if request is not None:
            request_url = f"; url={request.url}"
        return (
            f"HTTP {response.status_code} {response.reason_phrase}{request_url}; "
            f"response={response_text!r}"
        )

    @staticmethod
    def _is_missing_chat_template_error(response: httpx.Response) -> bool:
        if response.status_code != 400:
            return False
        return "chat template" in response.text.lower()

    def _retry_wait(self, response: httpx.Response, attempt: int) -> float:
        retry_after = response.headers.get("retry-after")
        if retry_after:
            try:
                return max(float(retry_after), self.config.retry_delay)
            except ValueError:
                pass
        return self.config.retry_delay * (2 ** attempt)


async def query_model(
    probes: list[Probe],
    config: APIConfig,
    model_name: str = "",
    concurrency: int = 5,
    stability_repeat_ids: Optional[set[str]] = None,
    stability_repeat_count: int = 3,
    progress_callback=None,
    raw_results_path: Optional[Path] = None,
) -> list[SingleTokenResult]:
    """
    对一个模型执行全部 probe 查询。

    Args:
        probes: probe 列表
        config: API 配置
        model_name: 模型名称标识
        concurrency: 并发数
        stability_repeat_ids: 需要重复采样的 probe id 集合
        stability_repeat_count: 重复次数
        progress_callback: 进度回调函数(completed, total)
        raw_results_path: 逐 batch 追加保存原始 token 结果的 JSONL 路径
    Returns:
        SingleTokenResult 列表（含重复采样的结果）
    """
    engine = QueryEngine(config, model_name, concurrency)
    if raw_results_path:
        raw_results_path = Path(raw_results_path)
        raw_results_path.parent.mkdir(parents=True, exist_ok=True)
        raw_results_path.write_text("", encoding="utf-8")

    # 构建任务列表（包含重复采样）
    tasks_probes: list[Probe] = []
    for p in probes:
        tasks_probes.append(p)
        if stability_repeat_ids and p.id in stability_repeat_ids:
            for _ in range(stability_repeat_count - 1):
                tasks_probes.append(p)

    results: list[SingleTokenResult] = []
    total = len(tasks_probes)

    async with httpx.AsyncClient() as client:
        # 分批执行避免过多并发
        batch_size = concurrency * 2
        for i in range(0, total, batch_size):
            batch = tasks_probes[i : i + batch_size]
            batch_tasks = [
                engine.query_single(p, client) for p in batch
            ]
            batch_results = await asyncio.gather(*batch_tasks)
            fatal_errors = [
                r.raw_response.get("error", "")
                for r in batch_results
                if _is_fatal_config_error(r.raw_response.get("error", ""))
            ]
            if fatal_errors and len(fatal_errors) == len(batch_results):
                raise ValueError(
                    f"Fatal API configuration error for {model_name}: "
                    f"{fatal_errors[0]}"
                )
            if _is_fatal_thinking_marker_batch(batch_results):
                marker_counts = {}
                for r in batch_results:
                    marker_counts[r.output_text] = marker_counts.get(r.output_text, 0) + 1
                raise ValueError(
                    f"Fatal thinking-mode output for {model_name}: "
                    f"batch returned only thinking markers {marker_counts}. "
                    "Disable model thinking and rerun; existing raw token files "
                    "from this run are contaminated."
                )

            results.extend(batch_results)
            if raw_results_path:
                _append_raw_results(raw_results_path, batch_results)

            if progress_callback:
                progress_callback(min(i + batch_size, total), total)

    return results


def query_model_sync(
    probes: list[Probe],
    config: APIConfig,
    model_name: str = "",
    concurrency: int = 5,
    stability_repeat_ids: Optional[set[str]] = None,
    stability_repeat_count: int = 3,
) -> list[SingleTokenResult]:
    """同步封装"""
    return asyncio.run(
        query_model(
            probes=probes,
            config=config,
            model_name=model_name,
            concurrency=concurrency,
            stability_repeat_ids=stability_repeat_ids,
            stability_repeat_count=stability_repeat_count,
        )
    )


def _append_raw_results(path: Path, batch_results: list[SingleTokenResult]):
    with path.open("a", encoding="utf-8") as f:
        for result in batch_results:
            row = asdict(result)
            raw_response = row.get("raw_response", {})
            compact_raw_response = _compact_raw_response(raw_response)
            if compact_raw_response:
                row["raw_response"] = compact_raw_response
            else:
                row.pop("raw_response", None)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _compact_raw_response(raw_response: dict) -> dict:
    compact = {}
    if "error" in raw_response:
        compact["error"] = raw_response["error"]
    for key in ("_output_normalization", "_raw_output_text", "_empty_output_retry"):
        if key in raw_response:
            compact[key] = raw_response[key]
    return compact


def _is_fatal_config_error(error: str) -> bool:
    fatal_markers = (
        "HTTP 400",
        "HTTP 401",
        "HTTP 403",
        "HTTP 404",
        "model_not_found",
        "model does not exist",
        "invalid_request_error",
        "unauthorized",
        "forbidden",
    )
    lower_error = error.lower()
    return any(marker.lower() in lower_error for marker in fatal_markers)


def _is_fatal_thinking_marker_batch(batch_results: list[SingleTokenResult]) -> bool:
    if not batch_results:
        return False
    outputs = [r.output_text for r in batch_results if not r.raw_response.get("error")]
    if len(outputs) != len(batch_results):
        return False
    return all(output in THINKING_MARKER_OUTPUTS for output in outputs)
