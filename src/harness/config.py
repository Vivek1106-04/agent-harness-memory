"""Config loading, validation and hashing.

One YAML file fully describes a run (SPEC.md section 7). Unknown keys are an error, not a
warning: a typo'd component name must fail loudly rather than silently run a different
experiment than the one written down.

The hash of the validated config is stamped into every trace header. `llm.base_url` is
excluded from it because the endpoint a run was dialled at is a property of the machine,
not of the experiment - including it would give the same experiment different identities on
two laptops. Model identity is carried by `llm.model` and `llm.revision`, which are hashed.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from harness.redaction import contains_secret

ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")

#: Excluded from the config hash. Machine-local, not part of experiment identity.
HASH_EXCLUDED: tuple[tuple[str, str], ...] = (("llm", "base_url"), ("llm", "auth"))


class ConfigError(Exception):
    pass


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ComponentConfig(BaseModel):
    """A component block. `type` names a registered implementation.

    Extra keys are allowed here and only here: they are constructor arguments for the named
    component, which is where a typo surfaces - the component refuses the argument.
    """

    model_config = ConfigDict(extra="allow", frozen=True)

    type: str


class RunConfig(_Strict):
    seed: int
    max_steps: int = Field(gt=0)
    run_id_prefix: str
    #: How many items the loop asks the memory layer for each step. It belongs to the run,
    #: not to a provider: every condition must be measured under the same retrieval budget
    #: or the comparison is between budgets rather than between memories.
    retrieval_k: int = Field(default=5, gt=0)


class EmbeddingConfig(_Strict):
    model: str
    revision: str


class LLMConfig(_Strict):
    """`api_key_env` names the environment variable holding the key. It is never the key.

    A hosted endpoint names one and the client refuses to start if it is unset; a local
    vLLM server leaves it null and sends no Authorization header. The name is not a secret,
    so it is hashed into the config identity along with everything else.

    `auth: google_adc` sends the machine's Google login instead of a key, for Vertex AI. How
    a machine authenticates is not part of the experiment, so it is not hashed.
    """

    provider: Literal["local", "openai_compatible"]
    base_url: str
    model: str
    revision: str
    max_output_tokens: int = Field(gt=0)
    temperature: float = Field(ge=0.0)
    api_key_env: str | None = None
    auth: Literal["api_key", "google_adc"] = "api_key"
    timeout_s: float = Field(default=120.0, gt=0.0)
    max_attempts: int = Field(default=4, ge=1)
    #: Sent as-is when set. Gemini 3 counts thinking against `max_output_tokens`, so an
    #: unpinned thinking level can eat the whole cap and return an empty reply.
    reasoning_effort: str | None = None
    #: Paid-tier list prices. The spend limit is computed from these, so they are set even
    #: on a free tier: the meter then overstates spend, which errs on the safe side.
    price_input_usd_per_m: float = Field(default=0.0, ge=0.0)
    price_output_usd_per_m: float = Field(default=0.0, ge=0.0)

    @model_validator(mode="after")
    def _one_credential(self) -> LLMConfig:
        if self.auth == "google_adc" and self.api_key_env is not None:
            raise ValueError("auth: google_adc sends a Google login; set api_key_env: null")
        return self


class LimitsConfig(_Strict):
    max_tokens_per_run: int = Field(gt=0)
    max_spend_usd: float = Field(ge=0.0)


class SuiteConfig(_Strict):
    """The task set, described rather than listed.

    It lives in the config, and therefore in the config hash, so that "which tasks was this
    row scored against" is answerable from the results row alone. Generation is pure, so
    the description is enough to re-derive the set exactly.
    """

    count: int = Field(gt=0)
    chain_length: int = Field(ge=2)
    recall_distance: int = Field(ge=1)
    distractor_density: float = Field(ge=0.0, lt=1.0)
    horizon: int = Field(ge=2)
    seed: int
    #: Pad every document to about this many tokens, so the budget binds on documents
    #: rather than on sentences. 0 keeps one sentence per document.
    document_tokens: int = Field(default=0, ge=0)


class HarnessConfig(_Strict):
    run: RunConfig
    suite: SuiteConfig
    planner: ComponentConfig
    memory: ComponentConfig
    context: ComponentConfig
    tools: ComponentConfig
    evaluator: ComponentConfig
    embedding: EmbeddingConfig
    llm: LLMConfig
    limits: LimitsConfig


def load_config(path: str | Path, env: dict[str, str] | None = None) -> HarnessConfig:
    """Read, expand environment references, validate. Any failure raises `ConfigError`."""
    raw = yaml.safe_load(Path(path).read_text())
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: top level must be a mapping")
    expanded = _expand(raw, os.environ if env is None else env)
    _reject_secrets(expanded)
    try:
        return HarnessConfig.model_validate(expanded)
    except Exception as error:
        raise ConfigError(f"{path}: {error}") from error


def config_hash(config: HarnessConfig) -> str:
    """Stable sha256 over the validated config, excluding machine-local fields."""
    data = config.model_dump(mode="json")
    for section, key in HASH_EXCLUDED:
        data.get(section, {}).pop(key, None)
    canonical = json.dumps(data, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def _expand(value: Any, env: dict[str, str] | os._Environ[str]) -> Any:
    if isinstance(value, dict):
        return {key: _expand(inner, env) for key, inner in value.items()}
    if isinstance(value, list):
        return [_expand(inner, env) for inner in value]
    if isinstance(value, str):
        return ENV_PATTERN.sub(lambda m: _lookup(m.group(1), env), value)
    return value


def _lookup(name: str, env: dict[str, str] | os._Environ[str]) -> str:
    try:
        return env[name]
    except KeyError:
        raise ConfigError(
            f"environment variable {name} is referenced by config but not set"
        ) from None


def _reject_secrets(value: Any, path: str = "") -> None:
    """Credentials come from the environment, never from a config file. SECURITY.md section 1."""
    if isinstance(value, dict):
        for key, inner in value.items():
            _reject_secrets(inner, f"{path}.{key}" if path else str(key))
        return
    if isinstance(value, list):
        for index, inner in enumerate(value):
            _reject_secrets(inner, f"{path}[{index}]")
        return
    if isinstance(value, str) and contains_secret(value):
        raise ConfigError(
            f"{path or 'config'} looks like a credential. Keys come from environment "
            "variables only - see SECURITY.md section 1."
        )
