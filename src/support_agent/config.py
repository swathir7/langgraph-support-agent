from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    openai_api_key: str | None = None
    classifier_model: str = "gpt-4.1-mini"
    """Only used when `openai_api_key` is set; otherwise the rule-based classifier
    is used (see graph/router.py) — the agent works fully offline by default."""

    escalation_confidence_threshold: float = 0.4
    """Router decisions below this confidence route to human escalation instead
    of an agent, regardless of which route the classifier picked."""

    order_retry_max_attempts: int = 3
    order_retry_base_delay_seconds: float = 0.2
    order_call_timeout_seconds: float = 5.0
    order_circuit_failure_threshold: int = 3
    order_circuit_reset_timeout_seconds: float = 30.0

    checkpoint_db_path: str = "support_agent_state.sqlite"


@lru_cache
def get_settings() -> Settings:
    return Settings()
