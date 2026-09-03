"""Module 13 — AI data pipeline: the LLM client.

Loads prompt text verbatim from `ufe/ai/prompts/*.md`, loads model/sampling settings from
`config/params/ai.yaml` (CONTRACT rule 1: no numeric literal in this file), and drives an
injectable transport so the whole pipeline is testable without a live Anthropic API key and
without network access (spec Section 17.2, task requirement 5).

Nothing here performs I/O at import time: no API key is read from the environment until a
transport's `complete()` is actually called.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Generic, Protocol, TypeVar

import yaml
from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
DEFAULT_SETTINGS_PATH = Path(__file__).resolve().parents[2] / "config" / "params" / "ai.yaml"

_PROMPT_FILE_RE = re.compile(r"^(?P<name>[A-Za-z]_[a-z0-9_]+)\.(?P<version>v\d+)\.md$")

_PLACEHOLDER_RE = re.compile(r"\{([a-zA-Z0-9_]+)\}")


# --------------------------------------------------------------------------------------
# Settings (Section 17.2) — loaded from YAML, never hardcoded.
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class AISettings:
    """Resolved AI pipeline settings, read from `config/params/ai.yaml`."""

    model_id: str
    temperature: float
    max_tokens: int
    seed: int | None
    max_retries: int
    confidence_auto_apply_threshold: float
    narrative_rel_tol: float
    narrative_abs_tol: float
    allowed_archetypes: tuple[str, ...]
    raw: dict[str, Any] = field(repr=False)

    @property
    def hash(self) -> str:
        """sha256 of the resolved settings tree, for provenance (mirrors ufe.params.Params.hash)."""
        blob = json.dumps(self.raw, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()


def load_settings(path: Path | None = None) -> AISettings:
    """Read Section 17.2 model/settings from YAML. Raises if the file is missing or malformed."""
    settings_path = Path(path) if path is not None else DEFAULT_SETTINGS_PATH
    if not settings_path.exists():
        raise FileNotFoundError(f"AI settings file not found: {settings_path}")
    raw = yaml.safe_load(settings_path.read_text(encoding="utf-8")) or {}

    model = raw.get("model", {})
    retry = raw.get("retry", {})
    review_queue = raw.get("review_queue", {})
    narrative = raw.get("narrative_verification", {})
    archetypes = raw.get("archetypes", {}).get("allowed", [])

    return AISettings(
        model_id=model["id"],
        temperature=float(model["temperature"]),
        max_tokens=int(model["max_tokens"]),
        seed=model.get("seed"),
        max_retries=int(retry.get("max_retries", 1)),
        confidence_auto_apply_threshold=float(review_queue["confidence_auto_apply_threshold"]),
        narrative_rel_tol=float(narrative.get("rel_tol", 0.0)),
        narrative_abs_tol=float(narrative.get("abs_tol", 0.0)),
        allowed_archetypes=tuple(archetypes),
        raw=raw,
    )


# --------------------------------------------------------------------------------------
# Prompts — loaded verbatim from disk, one versioned .md file per prompt (A-G).
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class PromptTemplate:
    """A single versioned prompt, loaded verbatim from `ufe/ai/prompts/`.

    `system` and `user_template` are the exact SYSTEM:/USER: sections of the source .md
    file, byte-for-byte from the spec. Variable substitution is done with plain
    string replacement of `{name}` tokens (never `str.format`, since the prompt text
    itself contains unrelated JSON braces that `.format()` would choke on).
    """

    name: str
    version: str
    path: Path
    system: str
    user_template: str

    def render_user(self, **variables: Any) -> str:
        text = self.user_template
        for key, value in variables.items():
            token = "{" + key + "}"
            text = text.replace(token, "" if value is None else str(value))
        return text

    @property
    def extracted_by(self) -> str:
        """Section 17.2: 'the version string is written into extracted_by on every record.'"""
        return f"ai:{self.path.name}"


def _split_system_user(text: str) -> tuple[str, str]:
    """Split a prompt .md file's body into its SYSTEM: and USER: sections."""
    sys_idx = text.find("SYSTEM:")
    usr_idx = text.find("USER:")
    if sys_idx == -1 or usr_idx == -1 or usr_idx <= sys_idx:
        raise ValueError("Prompt file must contain a SYSTEM: section followed by a USER: section")
    system = text[sys_idx + len("SYSTEM:") : usr_idx].strip("\n")
    user = text[usr_idx + len("USER:") :].strip("\n")
    return system.strip() + "\n", user + "\n"


def load_prompt(name: str, version: str, prompts_dir: Path | None = None) -> PromptTemplate:
    """Load a prompt verbatim from disk by name + version.

    `name` is the prompt letter+slug, e.g. "A_project_extraction". `version` is e.g. "v1".
    """
    directory = Path(prompts_dir) if prompts_dir is not None else PROMPTS_DIR
    path = directory / f"{name}.{version}.md"
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    text = path.read_text(encoding="utf-8")
    system, user_template = _split_system_user(text)
    return PromptTemplate(name=name, version=version, path=path, system=system, user_template=user_template)


def list_prompt_files(prompts_dir: Path | None = None) -> list[Path]:
    directory = Path(prompts_dir) if prompts_dir is not None else PROMPTS_DIR
    return sorted(p for p in directory.glob("*.md") if _PROMPT_FILE_RE.match(p.name))


# --------------------------------------------------------------------------------------
# Transport — injectable, so tests never touch the network or a real API key.
# --------------------------------------------------------------------------------------


class LLMTransport(Protocol):
    """Anything that can turn (system, user) text into a raw completion string."""

    def complete(
        self,
        *,
        system: str,
        user: str,
        model: str,
        temperature: float,
        max_tokens: int,
        seed: int | None = None,
    ) -> str: ...


class AnthropicTransport:
    """Real transport, backed by the `anthropic` SDK.

    The API key is resolved lazily, on the first call to `complete()` — never at import
    time and never at construction time — so importing this module (or constructing this
    class) never requires network access or a live key.
    """

    def __init__(self, api_key: str | None = None) -> None:
        self._api_key = api_key
        self._client: Any | None = None

    def _ensure_client(self) -> Any:
        if self._client is None:
            import os

            import anthropic

            key = self._api_key or os.environ.get("ANTHROPIC_API_KEY")
            if not key:
                raise RuntimeError("ANTHROPIC_API_KEY is not set and no api_key was provided")
            self._client = anthropic.Anthropic(api_key=key)
        return self._client

    def complete(
        self,
        *,
        system: str,
        user: str,
        model: str,
        temperature: float,
        max_tokens: int,
        seed: int | None = None,
    ) -> str:
        client = self._ensure_client()
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(block.text for block in response.content if getattr(block, "type", None) == "text")


class RecordedCall(BaseModel):
    """Provenance-friendly record of a single transport call, for tests and audit."""

    system: str
    user: str
    model: str
    temperature: float
    max_tokens: int
    seed: int | None
    response: str


class RecordReplayTransport:
    """Deterministic fake transport used by every test. Never touches the network.

    Construct with a list of canned response strings (or a dict keyed by an arbitrary
    label if the caller wants to pick responses out of order via `queue_response`).
    Each call to `complete()` pops the next canned response and records the call for
    later inspection.
    """

    def __init__(self, responses: list[str] | None = None) -> None:
        self._responses: list[str] = list(responses or [])
        self.calls: list[RecordedCall] = []

    def queue_response(self, response: str) -> None:
        self._responses.append(response)

    def complete(
        self,
        *,
        system: str,
        user: str,
        model: str,
        temperature: float,
        max_tokens: int,
        seed: int | None = None,
    ) -> str:
        if not self._responses:
            raise AssertionError("RecordReplayTransport exhausted: no canned response left to replay")
        response = self._responses.pop(0)
        self.calls.append(
            RecordedCall(
                system=system,
                user=user,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                seed=seed,
                response=response,
            )
        )
        return response


# --------------------------------------------------------------------------------------
# Parse-and-retry — shared by every prompt (Section 17.2).
# --------------------------------------------------------------------------------------

ModelT = TypeVar("ModelT", bound=BaseModel)


class ParseFailure(Exception):
    """Raised when an LLM response could not be parsed into the expected model
    even after the single retry Section 17.2 allows."""

    def __init__(self, message: str, raw_responses: list[str]) -> None:
        super().__init__(message)
        self.raw_responses = raw_responses


@dataclass
class ExtractionOutcome(Generic[ModelT]):
    """Result of running a prompt through `AIClient.extract`."""

    parsed: ModelT | None
    raw_responses: list[str]
    attempts: int
    prompt_name: str
    prompt_version: str
    model_id: str
    settings_hash: str
    ok: bool
    error: str | None = None

    @property
    def extracted_by(self) -> str:
        return f"ai:{self.prompt_name}.{self.prompt_version}"


def _strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"```$", "", text.strip())
    return text.strip()


@dataclass
class AIClient:
    """Drives a `LLMTransport` through a `PromptTemplate`, with Section 17.2's
    parse-once-retry-once-then-fail semantics."""

    transport: LLMTransport
    settings: AISettings

    def extract(
        self,
        prompt: PromptTemplate,
        variables: dict[str, Any],
        response_model: type[ModelT],
    ) -> ExtractionOutcome[ModelT]:
        user = prompt.render_user(**variables)
        raw_responses: list[str] = []
        last_error: str | None = None

        for attempt in range(self.settings.max_retries + 1):
            if attempt > 0 and last_error is not None:
                user_for_call = (
                    user
                    + "\n\nYour previous response failed validation with this error. "
                    + "Return ONLY corrected JSON matching the schema.\n"
                    + f"Validation error:\n{last_error}"
                )
            else:
                user_for_call = user

            raw = self.transport.complete(
                system=prompt.system,
                user=user_for_call,
                model=self.settings.model_id,
                temperature=self.settings.temperature,
                max_tokens=self.settings.max_tokens,
                seed=self.settings.seed,
            )
            raw_responses.append(raw)

            try:
                payload = json.loads(_strip_code_fences(raw))
                parsed = response_model.model_validate(payload)
            except (json.JSONDecodeError, ValidationError) as exc:
                last_error = str(exc)
                logger.warning(
                    "AI extraction parse failure (attempt %d) for prompt %s.%s: %s",
                    attempt + 1,
                    prompt.name,
                    prompt.version,
                    last_error,
                )
                continue

            return ExtractionOutcome(
                parsed=parsed,
                raw_responses=raw_responses,
                attempts=attempt + 1,
                prompt_name=prompt.name,
                prompt_version=prompt.version,
                model_id=self.settings.model_id,
                settings_hash=self.settings.hash,
                ok=True,
            )

        return ExtractionOutcome(
            parsed=None,
            raw_responses=raw_responses,
            attempts=self.settings.max_retries + 1,
            prompt_name=prompt.name,
            prompt_version=prompt.version,
            model_id=self.settings.model_id,
            settings_hash=self.settings.hash,
            ok=False,
            error=last_error,
        )

    def complete_raw(self, prompt: PromptTemplate, variables: dict[str, Any]) -> str:
        """For prompts whose output is plain prose, not JSON (Prompt G)."""
        user = prompt.render_user(**variables)
        return self.transport.complete(
            system=prompt.system,
            user=user,
            model=self.settings.model_id,
            temperature=self.settings.temperature,
            max_tokens=self.settings.max_tokens,
            seed=self.settings.seed,
        )
