"""Test chat model integration."""

from typing import cast
from unittest.mock import Mock, call

import pytest
from ai21 import MissingApiKeyError
from ai21.models.chat import AssistantMessage, UserMessage
from ai21.models.chat import SystemMessage as AI21SystemMessage
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import SecretStr
from pytest import CaptureFixture, MonkeyPatch

from langchain_ai21.chat_models import ChatAI21
from tests.unit_tests.conftest import (
    BASIC_EXAMPLE_CHAT_PARAMETERS,
    BASIC_EXAMPLE_CHAT_PARAMETERS_AS_DICT,
    DUMMY_API_KEY,
    JAMBA_MINI_CHAT_MODEL_NAME,
    temporarily_unset_api_key,
)


def test_initialization__when_no_api_key__should_raise_exception() -> None:
    """Test integration initialization."""
    with temporarily_unset_api_key():
        with pytest.raises(MissingApiKeyError):
            ChatAI21(model=JAMBA_MINI_CHAT_MODEL_NAME)  # type: ignore[call-arg]


def test_initialization__when_default_parameters_in_init() -> None:
    """Test chat model initialization."""
    ChatAI21(api_key=DUMMY_API_KEY, model=JAMBA_MINI_CHAT_MODEL_NAME)  # type: ignore[call-arg, arg-type]


def test_initialization__when_custom_parameters_in_init() -> None:
    model = JAMBA_MINI_CHAT_MODEL_NAME
    n = 1
    max_tokens = 10
    temperature = 0.1
    top_p = 0.1

    llm = ChatAI21(  # type: ignore[call-arg]
        api_key=DUMMY_API_KEY,  # type: ignore[arg-type]
        model=model,
        n=n,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
    )
    assert llm.model == model
    assert llm.n == n
    assert llm.max_tokens == max_tokens
    assert llm.temperature == temperature
    assert llm.top_p == top_p


def test_invoke(mock_client_with_chat: Mock) -> None:
    chat_input = "I'm Pickle Rick"

    llm = ChatAI21(
        model=JAMBA_MINI_CHAT_MODEL_NAME,
        api_key=DUMMY_API_KEY,  # type: ignore[arg-type]
        client=mock_client_with_chat,
        **BASIC_EXAMPLE_CHAT_PARAMETERS,  # type: ignore[arg-type, arg-type, arg-type, arg-type, arg-type, arg-type, arg-type, arg-type, arg-type, arg-type, arg-type, arg-type, arg-type]
    )
    llm.invoke(input=chat_input, config=dict(tags=["foo"]))

    mock_client_with_chat.chat.completions.create.assert_called_once_with(
        model=JAMBA_MINI_CHAT_MODEL_NAME,
        messages=[UserMessage(role="user", content="I'm Pickle Rick")],
        stream=False,
        **BASIC_EXAMPLE_CHAT_PARAMETERS_AS_DICT,
    )


def test_generate(mock_client_with_chat: Mock) -> None:
    messages0 = [
        HumanMessage(content="I'm Pickle Rick"),
        AIMessage(content="Hello Pickle Rick! I am your AI Assistant"),
        HumanMessage(content="Nice to meet you."),
    ]
    messages1 = [
        SystemMessage(content="system message"),
        HumanMessage(content="What is 1 + 1"),
    ]
    llm = ChatAI21(
        model=JAMBA_MINI_CHAT_MODEL_NAME,
        client=mock_client_with_chat,
        **BASIC_EXAMPLE_CHAT_PARAMETERS,  # type: ignore[arg-type, arg-type, arg-type, arg-type, arg-type, arg-type, arg-type, arg-type, arg-type, arg-type, arg-type, arg-type, arg-type, arg-type]
    )

    llm.generate(messages=[messages0, messages1])

    mock_client_with_chat.chat.completions.create.assert_has_calls(
        [
            call(
                stream=False,
                messages=[
                    UserMessage(role="user", content="I'm Pickle Rick"),
                    AssistantMessage(
                        role="assistant",
                        content="Hello Pickle Rick! I am your AI Assistant",
                        tool_calls=[],
                    ),
                    UserMessage(role="user", content="Nice to meet you."),
                ],
                model=JAMBA_MINI_CHAT_MODEL_NAME,
                **BASIC_EXAMPLE_CHAT_PARAMETERS,
            ),
            call(
                stream=False,
                messages=[
                    AI21SystemMessage(role="system", content="system message"),
                    UserMessage(role="user", content="What is 1 + 1"),
                ],
                model=JAMBA_MINI_CHAT_MODEL_NAME,
                **BASIC_EXAMPLE_CHAT_PARAMETERS,
            ),
        ]
    )


def test_api_key_is_secret_string() -> None:
    llm = ChatAI21(model=JAMBA_MINI_CHAT_MODEL_NAME, api_key="secret-api-key")  # type: ignore[call-arg, arg-type]
    assert isinstance(llm.api_key, SecretStr)


def test_api_key_masked_when_passed_from_env(
    monkeypatch: MonkeyPatch, capsys: CaptureFixture
) -> None:
    """Test initialization with an API key provided via an env variable"""
    monkeypatch.setenv("AI21_API_KEY", "secret-api-key")
    llm = ChatAI21(model=JAMBA_MINI_CHAT_MODEL_NAME)  # type: ignore[call-arg]
    print(llm.api_key, end="")
    captured = capsys.readouterr()

    assert captured.out == "**********"


def test_api_key_masked_when_passed_via_constructor(
    capsys: CaptureFixture,
) -> None:
    """Test initialization with an API key provided via the initializer"""
    llm = ChatAI21(model=JAMBA_MINI_CHAT_MODEL_NAME, api_key="secret-api-key")  # type: ignore[call-arg, arg-type]
    print(llm.api_key, end="")
    captured = capsys.readouterr()

    assert captured.out == "**********"


def test_uses_actual_secret_value_from_secretstr() -> None:
    """Test that actual secret is retrieved using `.get_secret_value()`."""
    llm = ChatAI21(model=JAMBA_MINI_CHAT_MODEL_NAME, api_key="secret-api-key")  # type: ignore[call-arg, arg-type]
    assert cast(SecretStr, llm.api_key).get_secret_value() == "secret-api-key"
