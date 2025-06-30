from typing import Any, Optional

from ai21 import AI21Client, AsyncAI21Client
from langchain_core.utils import from_env, secret_from_env
from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator
from typing_extensions import Self

_DEFAULT_TIMEOUT_SEC = 300


class AI21Base(BaseModel):
    """Base class for AI21 models."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    client: Any = Field(default=None, exclude=True)  #: :meta private:

    async_client: Any = Field(default=None, exclude=True)
    """Asynchronous client for API calls."""

    api_key: SecretStr = Field(
        default_factory=secret_from_env("AI21_API_KEY", default="")
    )
    """API key for AI21 API."""
    api_host: str = Field(
        default_factory=from_env(
            "AI21_API_URL", default="https://api.ai21.com/studio/v1"
        )
    )
    """Host URL"""
    timeout_sec: float = Field(
        default_factory=lambda: float(
            from_env("AI21_TIMEOUT_SEC", default=str(_DEFAULT_TIMEOUT_SEC))()
        )
    )
    """Timeout in seconds.
    
    If not set, it will default to the value of the environment 
    variable `AI21_TIMEOUT_SEC` or 300 seconds.
    """
    num_retries: Optional[int] = None
    """Maximum number of retries for API requests before giving up."""

    @model_validator(mode="after")
    def post_init(self) -> Self:
        api_key = self.api_key
        api_host = self.api_host
        timeout_sec = self.timeout_sec
        if (self.client or None) is None:
            self.client = AI21Client(
                api_key=api_key.get_secret_value(),
                api_host=api_host,
                timeout_sec=None if timeout_sec is None else float(timeout_sec),
                via="langchain",
            )

        return self

    @model_validator(mode="after")
    def init_async_client(self) -> Self:
        api_key = self.api_key
        api_host = self.api_host
        timeout_sec = self.timeout_sec
        if (self.async_client or None) is None:
            self.async_client = AsyncAI21Client(
                api_key=api_key.get_secret_value(),
                api_host=api_host,
                timeout_sec=None if timeout_sec is None else float(timeout_sec),
                via="langchain",
            )

        return self
