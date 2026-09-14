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
from pydantic import BaseModel, ConfigDict, Field

from harness.redaction import contains_secret

ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")

#: Excluded from the config hash. Machine-local, not part of experiment identity.
HASH_EXCLUDED: tuple[tuple[str, str], ...] = (("llm", "base_url"),)


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


class EmbeddingConfig(_Strict):
    model: str
    revision: str


class LLMConfig(_Strict):
    provider: Literal["local", "openai_compatible"]
    base_url: str
    model: str
    revision: str
    max_output_tokens: int = Field(gt=0)
    temperature: float = Field(ge=0.0)


class LimitsConfig(_Strict):
    max_tokens_per_run: int = Field(gt=0)
    max_spend_usd: float = Field(ge=0.0)


class HarnessConfig(_Strict):
    run: RunConfig
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
