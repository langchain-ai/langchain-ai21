import os
from contextlib import contextmanager
from typing import Generator
from unittest.mock import Mock

import pytest
from ai21 import AI21Client
from ai21.models.chat import (
    AssistantMessage,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
)
from ai21.models.usage_info import UsageInfo
from pytest_mock import MockerFixture

JAMBA_CHAT_MODEL_NAME = "jamba-instruct-preview"
JAMBA_1_5_MINI_CHAT_MODEL_NAME = "jamba-1.5-mini"
JAMBA_1_5_LARGE_CHAT_MODEL_NAME = "jamba-1.5-large"
DUMMY_API_KEY = "test_api_key"


BASIC_EXAMPLE_CHAT_PARAMETERS = {
    "max_tokens": 20,
    "temperature": 0.5,
    "top_p": 0.5,
    "n": 3,
}


BASIC_EXAMPLE_CHAT_PARAMETERS_AS_DICT = {
    "max_tokens": 20,
    "temperature": 0.5,
    "top_p": 0.5,
    "n": 3,
}


@pytest.fixture
def mock_client_with_chat(mocker: MockerFixture) -> Mock:
    mock_client = mocker.MagicMock(spec=AI21Client)
    mock_client.chat = mocker.MagicMock()
    output = AssistantMessage(  # type: ignore[call-arg]
        content="Hello Pickle Rick!",
    )
    response = ChatCompletionResponse(
        id="test_id",
        choices=[ChatCompletionResponseChoice(index=0, message=output)],
        usage=UsageInfo(prompt_tokens=5, completion_tokens=5, total_tokens=5),
    )

    mock_client.chat.completions.create.return_value = response

    return mock_client


@contextmanager
def temporarily_unset_api_key() -> Generator:
    """
    Unset and set environment key for testing purpose for when an API KEY is not set
    """
    api_key = os.environ.pop("AI21_API_KEY", None)
    yield

    if api_key is not None:
        os.environ["AI21_API_KEY"] = api_key
