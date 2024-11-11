import os
from contextlib import contextmanager
from typing import Generator
from unittest.mock import Mock

import pytest
from ai21 import AI21Client
from ai21.models import (
    ChatOutput,
    ChatResponse,
    FinishReason,
    Penalty,
    RoleType,
)
from pytest_mock import MockerFixture

JAMBA_CHAT_MODEL_NAME = "jamba-instruct-preview"
JAMBA_1_5_MINI_CHAT_MODEL_NAME = "jamba-1.5-mini"
JAMBA_1_5_LARGE_CHAT_MODEL_NAME = "jamba-1.5-large"
DUMMY_API_KEY = "test_api_key"

JAMBA_FAMILY_MODEL_NAMES = [
    JAMBA_CHAT_MODEL_NAME,
    JAMBA_1_5_MINI_CHAT_MODEL_NAME,
    JAMBA_1_5_LARGE_CHAT_MODEL_NAME,
]


BASIC_EXAMPLE_CHAT_PARAMETERS = {
    "num_results": 3,
    "max_tokens": 20,
    "min_tokens": 10,
    "temperature": 0.5,
    "top_p": 0.5,
    "top_k_return": 0,
    "frequency_penalty": Penalty(scale=0.2, apply_to_numbers=True),  # type: ignore[call-arg]
    "presence_penalty": Penalty(scale=0.2, apply_to_stopwords=True),  # type: ignore[call-arg]
    "count_penalty": Penalty(  # type: ignore[call-arg]
        scale=0.2,
        apply_to_punctuation=True,
        apply_to_emojis=True,
    ),
    "n": 3,
}


BASIC_EXAMPLE_CHAT_PARAMETERS_AS_DICT = {
    "num_results": 3,
    "max_tokens": 20,
    "min_tokens": 10,
    "temperature": 0.5,
    "top_p": 0.5,
    "top_k_return": 0,
    "frequency_penalty": Penalty(scale=0.2, apply_to_numbers=True).to_dict(),  # type: ignore[call-arg]
    "presence_penalty": Penalty(scale=0.2, apply_to_stopwords=True).to_dict(),  # type: ignore[call-arg]
    "count_penalty": Penalty(  # type: ignore[call-arg]
        scale=0.2,
        apply_to_punctuation=True,
        apply_to_emojis=True,
    ).to_dict(),
    "n": 3,
}


@pytest.fixture
def mock_client_with_chat(mocker: MockerFixture) -> Mock:
    mock_client = mocker.MagicMock(spec=AI21Client)
    mock_client.chat = mocker.MagicMock()

    output = ChatOutput(  # type: ignore[call-arg]
        text="Hello Pickle Rick!",
        role=RoleType.ASSISTANT,
        finish_reason=FinishReason(reason="testing"),
    )
    mock_client.chat.create.return_value = ChatResponse(outputs=[output])

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

