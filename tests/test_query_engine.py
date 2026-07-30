import asyncio
import json
from pathlib import Path
import sys

import httpx

sys.path.insert(0, str(Path(__file__).parent.parent))

from tokenizer_fingerprint.query_engine import APIConfig, QueryEngine
from tokenizer_fingerprint.schema import Probe


def test_api_config_uses_query_protocol_defaults_with_model_override():
    config = APIConfig.from_dict(
        {
            "model": "test-model",
            "api_key": "test-key",
        },
        defaults={"system_prompt": "global prompt"},
    )
    overridden = APIConfig.from_dict(
        {
            "model": "test-model",
            "api_key": "test-key",
            "system_prompt": "model prompt",
        },
        defaults={"system_prompt": "global prompt"},
    )

    assert config.system_prompt == "global prompt"
    assert overridden.system_prompt == "model prompt"


def test_query_engine_remembers_completions_after_missing_chat_template():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        if request.url.path.endswith("/chat/completions"):
            return httpx.Response(
                400,
                json={
                    "error": {
                        "message": (
                            "As of transformers v4.44, default chat template is no "
                            "longer allowed, so you must provide a chat template if "
                            "the tokenizer does not define one."
                        )
                    }
                },
            )
        return httpx.Response(200, json={"choices": [{"text": " next"}]})

    async def run_query():
        client = httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
            base_url="http://testserver",
        )
        try:
            config = APIConfig(
                provider="openai",
                model="base-model",
                api_key="test-key",
                base_url="http://testserver/v1",
            )
            engine = QueryEngine(config, model_name="base-model")
            first_result = await engine.query_single(
                Probe(id="p1", text="The quick", category="english_natural"),
                client,
            )
            second_result = await engine.query_single(
                Probe(id="p2", text="A second", category="english_natural"),
                client,
            )
            return first_result, second_result
        finally:
            await client.aclose()

    first_result, second_result = asyncio.run(run_query())

    assert calls == ["/v1/chat/completions", "/v1/completions", "/v1/completions"]
    assert first_result.output_text == " next"
    assert second_result.output_text == " next"


def test_query_engine_can_use_completions_endpoint_directly():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        return httpx.Response(200, json={"choices": [{"text": "!"}]})

    async def run_query():
        client = httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
            base_url="http://testserver",
        )
        try:
            config = APIConfig(
                provider="openai",
                model="base-model",
                api_key="test-key",
                base_url="http://testserver/v1",
                endpoint="completions",
            )
            engine = QueryEngine(config, model_name="base-model")
            return await engine.query_single(
                Probe(id="p1", text="Hello", category="english_natural"),
                client,
            )
        finally:
            await client.aclose()

    result = asyncio.run(run_query())

    assert calls == ["/v1/completions"]
    assert result.output_text == "!"


def test_query_engine_can_use_deepseek_chat_prefix_completion():
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        requests.append((request.url.path, body))
        return httpx.Response(200, json={"choices": [{"message": {"content": "lel"}}]})

    async def run_query():
        client = httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
            base_url="http://testserver",
        )
        try:
            config = APIConfig(
                provider="openai",
                model="deepseek-chat",
                api_key="test-key",
                base_url="http://testserver/beta",
                system_prompt="custom continuation prompt",
                message_mode="deepseek_chat_prefix",
            )
            engine = QueryEngine(config, model_name="DeepSeek-V3")
            return await engine.query_single(
                Probe(id="p1", text="He began a paral", category="english_partial"),
                client,
            )
        finally:
            await client.aclose()

    result = asyncio.run(run_query())

    assert result.output_text == "lel"
    assert len(requests) == 1
    path, body = requests[0]
    assert path == "/beta/chat/completions"
    assert body["model"] == "deepseek-chat"
    assert body["max_tokens"] == 1
    assert body["temperature"] == 0
    assert body["messages"][0] == {
        "role": "system",
        "content": "custom continuation prompt",
    }
    assert body["messages"][-1] == {
        "role": "assistant",
        "content": "He began a paral",
        "prefix": True,
    }


def test_query_engine_retries_empty_chat_outputs_until_nonempty():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(json.loads(request.content.decode("utf-8")))
        if len(calls) == 1:
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {"content": "", "reasoning_content": None},
                        }
                    ],
                    "usage": {
                        "completion_tokens": 0,
                        "completion_tokens_details": {"reasoning_tokens": 0},
                    },
                },
            )
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "finish_reason": "length",
                        "message": {"content": "ca", "reasoning_content": None},
                    }
                ],
                "usage": {
                    "completion_tokens": 1,
                    "completion_tokens_details": {"reasoning_tokens": 0},
                },
            },
        )

    async def run_query():
        client = httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
            base_url="http://testserver",
        )
        try:
            config = APIConfig(
                provider="openai",
                model="deepseek-v4-pro",
                api_key="test-key",
                base_url="http://testserver/beta",
                message_mode="deepseek_chat_prefix",
                empty_output_retries=2,
            )
            engine = QueryEngine(config, model_name="DeepSeek-V4-Pro")
            return await engine.query_single(
                Probe(id="p1", text="militaire fran", category="french_partial"),
                client,
            )
        finally:
            await client.aclose()

    result = asyncio.run(run_query())

    assert len(calls) == 2
    assert result.output_text == "ca"
    retry_meta = result.raw_response["_empty_output_retry"]
    assert retry_meta["attempt_count"] == 2
    assert retry_meta["empty_retries_used"] == 1
    assert retry_meta["empty_response_count"] == 1
    assert retry_meta["recovered_after_empty"] is True
    assert retry_meta["final_is_empty"] is False
    assert retry_meta["empty_responses"][0]["finish_reason"] == "stop"
    assert retry_meta["empty_responses"][0]["reasoning_tokens"] == 0


def test_query_engine_keeps_empty_after_empty_retry_budget_is_exhausted():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(json.loads(request.content.decode("utf-8")))
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": "", "reasoning_content": None},
                    }
                ],
                "usage": {
                    "completion_tokens": 0,
                    "completion_tokens_details": {"reasoning_tokens": 0},
                },
            },
        )

    async def run_query():
        client = httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
            base_url="http://testserver",
        )
        try:
            config = APIConfig(
                provider="openai",
                model="deepseek-v4-pro",
                api_key="test-key",
                base_url="http://testserver/beta",
                message_mode="deepseek_chat_prefix",
                empty_output_retries=2,
            )
            engine = QueryEngine(config, model_name="DeepSeek-V4-Pro")
            return await engine.query_single(
                Probe(id="p1", text="Sekundärbatt", category="german_partial"),
                client,
            )
        finally:
            await client.aclose()

    result = asyncio.run(run_query())

    assert len(calls) == 3
    assert result.output_text == ""
    assert result.is_empty is True
    retry_meta = result.raw_response["_empty_output_retry"]
    assert retry_meta["attempt_count"] == 3
    assert retry_meta["empty_retries_used"] == 2
    assert retry_meta["empty_response_count"] == 3
    assert retry_meta["recovered_after_empty"] is False
    assert retry_meta["final_is_empty"] is True
    assert retry_meta["final_empty_response"]["finish_reason"] == "stop"
