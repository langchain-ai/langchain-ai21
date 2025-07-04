"""Standard LangChain interface tests"""

import time
from typing import Optional, Type

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_standard_tests.integration_tests import ChatModelIntegrationTests

from langchain_ai21 import ChatAI21

rate_limiter = InMemoryRateLimiter(requests_per_second=0.5)


class BaseTestAI21(ChatModelIntegrationTests):
    def teardown(self) -> None:
        # avoid getting rate limited
        time.sleep(1)

    @property
    def chat_model_class(self) -> Type[BaseChatModel]:
        return ChatAI21

    @pytest.mark.xfail(reason="Not implemented.")
    def test_usage_metadata(self, model: BaseChatModel) -> None:
        super().test_usage_metadata(model)


class TestAI21JambaMini(BaseTestAI21):
    @property
    def tool_choice_value(self) -> Optional[str]:
        """Value to use for tool choice when used in tests."""
        return "any"

    @property
    def chat_model_params(self) -> dict:
        return {
            "model": "jamba-mini",
            "temperature": 0.0,
        }

    @pytest.mark.xfail(reason="Prompt doesn't generate tool calls for Jamba 1.5.")
    def test_tool_calling(self, model: BaseChatModel) -> None:
        super().test_tool_calling(model)

    @pytest.mark.xfail(reason="Prompt doesn't generate tool calls for Jamba 1.5.")
    async def test_tool_calling_async(self, model: BaseChatModel) -> None:
        await super().test_tool_calling_async(model)

    @pytest.mark.xfail(reason="Prompt doesn't generate tool calls for Jamba 1.5.")
    def test_tool_calling_with_no_arguments(self, model: BaseChatModel) -> None:
        super().test_tool_calling_with_no_arguments(model)

    @pytest.mark.xfail(reason="Requires tool calling & stream - still WIP")
    def test_structured_output(self, model: BaseChatModel) -> None:
        super().test_structured_output(model)

    @pytest.mark.xfail(reason="Requires tool calling & stream - still WIP")
    async def test_structured_output_async(self, model: BaseChatModel) -> None:
        await super().test_structured_output_async(model)

    @pytest.mark.xfail(reason="Requires tool calling & stream - still WIP")
    def test_structured_output_pydantic_2_v1(self, model: BaseChatModel) -> None:
        super().test_structured_output_pydantic_2_v1(model)

    @pytest.mark.xfail(reason="Requires tool calling & stream - still WIP")
    def test_structured_output_optional_param(self, model: BaseChatModel) -> None:
        super().test_structured_output_optional_param(model)
