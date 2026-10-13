"""The only LLM touchpoint in the project.

Design decision embodied here: **the transport is the standard library.** An OpenAI-compatible
`POST /chat/completions` over `urllib.request` means no vendor SDK is a dependency and any
compatible endpoint (OpenAI, OpenRouter, vLLM, Ollama, LM Studio, a gateway) works unchanged.
The cost is a little more code here; the gain is that nothing in the repository is coupled to
one vendor's client library.

Second decision: **a response cache keyed by the prompt hash.** A repeated run is byte-identical
and costs nothing, and that is what makes the "same inputs, same result" contract hold even with
an LLM in the loop. Config, model, schema and mode are all part of the key, so a change in any of
them is a cache miss rather than a stale answer.

Third decision: **the API key never leaves this module.** It goes into the request header and
nowhere else — not into a log line, not into the cache file, not into the manifest.
"""

from __future__ import annotations

import hashlib
import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..config import Settings

# HTTP statuses that mean "this endpoint does not accept that request shape", used to fall back
# from native tool calling to the JSON-in-prompt path. 500s are NOT here: a server error is
# transient and retrying in another mode would hide a real outage.
_FALLBACK_STATUSES = {400, 404, 405, 415, 422, 501}


class LLMError(RuntimeError):
    """Any failure from the endpoint. Callers degrade rather than crash (SPEC.md 4.3)."""


@dataclass
class LLMConfig:
    """Resolved LLM wiring: secrets from `.env`, everything else from `settings.yaml`."""

    enabled: bool = False
    base_url: str = ""
    api_key: str = ""
    model: str = ""
    timeout_s: float = 120.0
    temperature: float = 0.0
    max_tokens: int = 2000
    tool_calling: str = "auto"
    cache_enabled: bool = True
    cache_dir: Path = Path("artifacts/llm_cache")

    @classmethod
    def from_settings(cls, settings: Settings) -> "LLMConfig":
        """Build the config from the `llm:` section of settings.yaml plus the named env vars."""
        llm = settings.raw.get("llm", {}) or {}

        def env(key: str, default: str) -> str:
            return str(settings.env(llm.get(key, ""), default) or default)

        cache_dir = Path(str(llm.get("cache_dir", "artifacts/llm_cache")))
        return cls(
            enabled=env("enabled_env", "false").lower() == "true",
            base_url=env("base_url_env", "").rstrip("/"),
            api_key=env("api_key_env", ""),
            model=env("model_env", ""),
            timeout_s=float(env("timeout_env", "120")),
            temperature=float(llm.get("temperature", 0)),
            max_tokens=int(llm.get("max_tokens", 2000)),
            tool_calling=str(llm.get("tool_calling", "auto")),
            cache_enabled=bool(llm.get("cache", True)),
            cache_dir=cache_dir if cache_dir.is_absolute() else settings.repo_root / cache_dir,
        )

    @property
    def host(self) -> str:
        """Base-URL host only, which is what the manifest records (never the full URL, never a key)."""
        without_scheme = self.base_url.split("//", 1)[-1]
        return without_scheme.split("/", 1)[0]


@dataclass
class LLMResponse:
    """One model answer, with the metadata a reviewer or the manifest needs."""

    text: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    mode: str = "json_prompt"
    cached: bool = False
    prompt_hash: str = ""
    usage: dict[str, Any] = field(default_factory=dict)
    # True when the JSON came back truncated and the complete leading part was salvaged. Recorded
    # rather than hidden: a salvaged answer is a partial answer.
    repaired: bool = False


def extract_json(text: str, repair: bool = True) -> Any:
    """Parse the JSON object or array in a model answer.

    Models wrap JSON in ```json fences or a sentence of preamble even when told not to, so the
    outermost braces are located first. If the answer is *truncated* — the common failure, because
    a long proposal can hit the output-token cap — the longest complete prefix is salvaged and every
    open bracket is closed. That is safe here because the caller validates the result against a
    schema and, in the induction gate's case, tests every element it contains; a partial answer is
    strictly better than losing the whole call to one cut-off character.
    """
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```")[1]
        cleaned = cleaned[4:] if cleaned.lower().startswith("json") else cleaned
    start = min((i for i in (cleaned.find("{"), cleaned.find("[")) if i >= 0), default=-1)
    if start < 0:
        raise LLMError(f"no JSON found in model answer: {text[:200]!r}")
    candidate = cleaned[start:]
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass
    end = max(candidate.rfind("}"), candidate.rfind("]"))
    if end > 0:
        try:
            return json.loads(candidate[: end + 1])
        except json.JSONDecodeError:
            pass
    if not repair:
        raise LLMError(f"could not parse JSON from model answer: {text[:200]!r}")
    return _salvage_truncated(candidate)


def _salvage_truncated(candidate: str) -> Any:
    """Return the longest parseable prefix of a truncated JSON document, brackets closed.

    The cut point is always a *completed* value — a closing brace or bracket first, then a closing
    string quote — tried from the end backwards, so a half-written number or word is never completed
    into a value. That matters: inventing `2` from a truncated `[1,2000...` would be a fabricated
    number, while dropping the whole incomplete element is merely a partial answer, which the caller
    is told about. Composite values are preferred because a truncated element inside an array is
    almost always a partially written object, and cutting at the previous object's brace yields
    clean elements instead of a broken one.
    """
    composite_ends: list[int] = []
    string_ends: list[int] = []
    in_string = False
    escaped = False
    for index, char in enumerate(candidate):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
                string_ends.append(index)
            continue
        if char == '"':
            in_string = True
        elif char in "}]":
            composite_ends.append(index)

    for group in (composite_ends, string_ends):
        for end in reversed(group[-200:]):
            prefix = candidate[: end + 1]
            open_brackets: list[str] = []
            inside = False
            escape = False
            for char in prefix:
                if inside:
                    if escape:
                        escape = False
                    elif char == "\\":
                        escape = True
                    elif char == '"':
                        inside = False
                    continue
                if char == '"':
                    inside = True
                elif char in "{[":
                    open_brackets.append(char)
                elif char in "}]" and open_brackets:
                    open_brackets.pop()
            closed = prefix + "".join("}" if bracket == "{" else "]" for bracket in reversed(open_brackets))
            try:
                return json.loads(closed)
            except json.JSONDecodeError:
                continue
    raise LLMError(f"model answer is not valid JSON and could not be salvaged: {candidate[:160]!r}")


def _looks_truncated(text: str) -> bool:
    """True when an answer's brackets do not balance, i.e. it was cut off mid-document.

    Used only to flag truthfully that a salvaged answer was partial; the salvage itself is done by
    `extract_json`, and nothing downstream treats a repaired answer as complete.
    """
    depth = 0
    in_string = False
    escaped = False
    for char in text:
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char in "{[":
            depth += 1
        elif char in "}]":
            depth -= 1
    return depth > 0 or in_string


class LLMClient:
    """OpenAI-compatible chat client with caching, tool calling and a graceful fallback."""
    def __init__(self, settings: Settings, config: LLMConfig | None = None) -> None:
        self.settings = settings
        self.config = config or LLMConfig.from_settings(settings)
        # `mode` is resolved once per client and reported in the manifest, because "we used
        # native tool calling" is a property of the run, not of a single call.
        self.mode = "native" if self.config.tool_calling == "native" else (
            "json_prompt" if self.config.tool_calling == "json_prompt" else "native"
        )
        self.calls: list[dict[str, Any]] = []

    # --- capability ---------------------------------------------------------

    @property
    def available(self) -> bool:
        """True when the run may talk to an endpoint at all: enabled, URL and model set."""
        return bool(self.config.enabled and self.config.base_url and self.config.model)

    def describe(self) -> dict[str, Any]:
        """Return the non-secret LLM wiring for the manifest and the startup line."""
        return {
            "enabled": self.config.enabled,
            "available": self.available,
            "host": self.config.host,
            "model": self.config.model,
            "tool_calling": self.config.tool_calling,
            "mode_in_use": self.mode,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "cache": self.config.cache_enabled,
        }

    # --- transport ----------------------------------------------------------

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        """POST to /chat/completions and return the parsed body. Raises LLMError on any failure."""
        url = f"{self.config.base_url}/chat/completions"
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(url, data=body, method="POST")
        request.add_header("Content-Type", "application/json")
        if self.config.api_key:
            # The key exists only in this header. It is never logged and never cached.
            request.add_header("Authorization", f"Bearer {self.config.api_key}")
        try:
            with urllib.request.urlopen(request, timeout=self.config.timeout_s) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")[:400]
            raise LLMError(f"HTTP {error.code} from endpoint: {detail}") from error
        except urllib.error.URLError as error:
            raise LLMError(f"endpoint unreachable: {error.reason}") from error
        except json.JSONDecodeError as error:
            raise LLMError(f"endpoint returned non-JSON: {error}") from error

    def _cache_path(self, key: str) -> Path:
        return self.config.cache_dir / f"{key}.json"

    def _cache_key(self, payload: dict[str, Any]) -> str:
        return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:32]

    # --- calls --------------------------------------------------------------

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        response_schema: dict[str, Any] | None = None,
        purpose: str = "",
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Run one chat completion and return an `LLMResponse`.

        `tools` requests native function calling; `response_schema` is appended to the prompt as
        an instruction and validated by the caller. In `auto` mode a native request that the
        endpoint rejects falls back to the JSON-in-prompt path for the rest of the run, and that
        choice is recorded. Raises `LLMError` when the endpoint cannot answer at all — the caller
        decides what to do, because the induction stage must never fail a run.
        """
        if not self.available:
            raise LLMError("LLM is not configured (EDA_LLM_ENABLED / base URL / model)")

        use_native = bool(tools) and self.mode == "native"
        payload = {
            "model": self.config.model,
            "messages": list(messages),
            "temperature": self.config.temperature,
            "max_tokens": max_tokens or self.config.max_tokens,
        }
        if use_native:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        if response_schema is not None:
            payload["messages"] = payload["messages"] + [
                {
                    "role": "user",
                    "content": (
                        "Reply with ONE JSON object and nothing else — no prose, no code fences. "
                        "It must match this JSON schema exactly:\n"
                        + json.dumps(response_schema, separators=(",", ":"))
                    ),
                }
            ]

        key = self._cache_key({**payload, "mode": self.mode})
        cache_file = self._cache_path(key)
        if self.config.cache_enabled and cache_file.exists():
            cached = json.loads(cache_file.read_text(encoding="utf-8"))
            self._record(purpose, key, mode=cached.get("mode", self.mode), cached=True)
            return LLMResponse(
                text=cached.get("text", ""),
                tool_calls=cached.get("tool_calls", []),
                mode=cached.get("mode", self.mode),
                cached=True,
                prompt_hash=key,
                usage=cached.get("usage", {}) or {},
            )

        try:
            body = self._post(payload)
        except LLMError as error:
            # Only a shape rejection triggers the fallback; anything else is a real failure.
            rejected = any(f"HTTP {status}" in str(error) for status in _FALLBACK_STATUSES)
            if use_native and rejected and self.config.tool_calling == "auto":
                self.mode = "json_prompt"
                payload.pop("tools", None)
                payload.pop("tool_choice", None)
                key = self._cache_key({**payload, "mode": self.mode})
                cache_file = self._cache_path(key)
                self._record(purpose, key, mode="json_prompt", cached=False, note="fell back from native")
                body = self._post(payload)
            else:
                self._record(purpose, key, mode=self.mode, cached=False, error=str(error))
                raise

        response = self._parse(body, native=use_native and self.mode == "native")
        response.prompt_hash = key
        if self.config.cache_enabled:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(
                json.dumps(
                    {"text": response.text, "tool_calls": response.tool_calls, "mode": response.mode,
                     "usage": response.usage, "purpose": purpose},
                    indent=2,
                ),
                encoding="utf-8",
            )
        self._record(purpose, key, mode=response.mode, cached=False, usage=response.usage)
        return response

    def complete_json(
        self,
        messages: list[dict[str, Any]],
        response_schema: dict[str, Any],
        purpose: str = "",
        max_tokens: int | None = None,
    ) -> tuple[Any, LLMResponse]:
        """Run a completion and parse the answer as JSON. Raises LLMError when there is none.

        A truncated answer is salvaged to its longest complete prefix and flagged on the response,
        so the caller can record that it acted on a partial answer rather than a whole one.
        """
        response = self.complete(messages, response_schema=response_schema, purpose=purpose, max_tokens=max_tokens)
        try:
            payload = extract_json(response.text)
        except (LLMError, json.JSONDecodeError) as error:
            raise LLMError(f"model did not return valid JSON: {error}") from error
        response.repaired = _looks_truncated(response.text)
        return payload, response

    def _parse(self, body: dict[str, Any], native: bool) -> LLMResponse:
        """Turn an endpoint body into an `LLMResponse`, tolerating the two common shapes."""
        choices = body.get("choices") or []
        if not choices:
            raise LLMError(f"endpoint returned no choices: {json.dumps(body)[:300]}")
        message = choices[0].get("message", {}) or {}
        tool_calls = []
        for call in message.get("tool_calls") or []:
            function = call.get("function", {}) or {}
            arguments = function.get("arguments", "{}")
            try:
                parsed = json.loads(arguments) if isinstance(arguments, str) else arguments
            except json.JSONDecodeError:
                parsed = {}
            tool_calls.append({"name": function.get("name", ""), "arguments": parsed})
        return LLMResponse(
            text=message.get("content") or "",
            tool_calls=tool_calls,
            # A native request stays "native" even when the model answered with prose instead of
            # a tool call; the mode records what was asked for, which is what the manifest needs.
            mode="native" if native else "json_prompt",
            usage=body.get("usage", {}) or {},
        )

    def _record(self, purpose: str, key: str, mode: str, cached: bool, **extra: Any) -> None:
        """Append a non-secret record of the call for the manifest."""
        self.calls.append(
            {"purpose": purpose, "prompt_hash": key[:12], "mode": mode, "cached": cached,
             "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **extra}
        )

    # --- doctor -------------------------------------------------------------

    def ping(self, max_tokens: int = 16) -> dict[str, Any]:
        """Send a deliberately tiny request to prove reachability. Never logs the key.

        Sixteen tokens, not one: some routes enforce a minimum output budget and reject anything
        smaller with a 400, which would report a healthy endpoint as broken. There is still no
        prompt content beyond a single character, so the check cannot resemble a real call.
        Returns a dict so `doctor` can print a verdict instead of a traceback.
        """
        if not self.config.enabled:
            return {"attempted": False, "ok": False, "reason": "EDA_LLM_ENABLED is not true"}
        if not self.config.base_url or not self.config.model:
            return {"attempted": False, "ok": False, "reason": "base URL or model is unset"}
        started = time.time()
        try:
            payload = {
                "model": self.config.model,
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": max_tokens,
                "temperature": 0,
            }
            body = self._post(payload)
            elapsed = round(time.time() - started, 2)
            return {
                "attempted": True,
                "ok": True,
                "model": self.config.model,
                "host": self.config.host,
                "seconds": elapsed,
                "reply_preview": (body.get("choices", [{}])[0].get("message", {}) or {}).get("content", "")[:40],
            }
        except LLMError as error:
            return {"attempted": True, "ok": False, "model": self.config.model,
                    "host": self.config.host, "reason": str(error)}
