from typing import List

import pytest
from ai21.models.chat import AssistantMessage, ChatMessage
from ai21.models.chat import SystemMessage as AI21SystemMessage
from ai21.models.chat import ToolMessage as AI21ToolMessage
from ai21.models.chat import UserMessage
from langchain_ai21.chat.chat_adapter import ChatAdapter
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

_JAMBA_MODEL_NAME = "jamba-instruct-preview"
_JAMBA_MINI_MODEL_NAME = "jamba-mini"
_JAMBA_LARGE_MODEL_NAME = "jamba-large"


@pytest.mark.parametrize(
    ids=[
        "when_human_message_jamba_model",
        "when_ai_message_jamba_model",
    ],
    argnames=["model", "message", "expected_ai21_message"],
    argvalues=[
        (
            _JAMBA_MODEL_NAME,
            HumanMessage(content="Human Message Content"),
            UserMessage(role="user", content="Human Message Content"),
        ),
        (
            _JAMBA_MODEL_NAME,
            AIMessage(content="AI Message Content"),
            AssistantMessage(
                role="assistant", content="AI Message Content", tool_calls=[]
            ),
        ),
    ],
)
def test_convert_message_to_ai21_message(
    message: BaseMessage,
    expected_ai21_message: ChatMessage,
    chat_adapter: ChatAdapter,
) -> None:
    ai21_message = chat_adapter._convert_message_to_ai21_message(message)
    assert ai21_message == expected_ai21_message


@pytest.mark.parametrize(
    ids=[
        "when_all_messages_are_human_messages__should_return_system_none_jamba_model",
        "when_first_message_is_system__should_return_system_jamba_model",
        "when_tool_calling_message__should_return_tool_jamba_mini_model",
        "when_tool_calling_message__should_return_tool_jamba_large_model",
    ],
    argnames=["model", "messages", "expected_messages"],
    argvalues=[
        (
            _JAMBA_MODEL_NAME,
            [
                HumanMessage(content="Human Message Content 1"),
                HumanMessage(content="Human Message Content 2"),
            ],
            {
                "messages": [
                    UserMessage(
                        role="user",
                        content="Human Message Content 1",
                    ),
                    UserMessage(
                        role="user",
                        content="Human Message Content 2",
                    ),
                ]
            },
        ),
        (
            _JAMBA_MODEL_NAME,
            [
                SystemMessage(content="System Message Content 1"),
                HumanMessage(content="Human Message Content 1"),
            ],
            {
                "messages": [
                    AI21SystemMessage(
                        role="system", content="System Message Content 1"
                    ),
                    UserMessage(role="user", content="Human Message Content 1"),
                ],
            },
        ),
        (
            _JAMBA_MINI_MODEL_NAME,
            [
                ToolMessage(
                    content="42",
                    tool_call_id="call_Jja7J89XsjrOLA5r!MEOW!SL",
                )
            ],
            {
                "messages": [
                    AI21ToolMessage(
                        role="tool",
                        tool_call_id="call_Jja7J89XsjrOLA5r!MEOW!SL",
                        content="42",
                    ),
                ],
            },
        ),
        (
            _JAMBA_LARGE_MODEL_NAME,
            [
                ToolMessage(
                    content="42",
                    tool_call_id="call_Jja7J89XsjrOLA5r!MEOW!SL",
                )
            ],
            {
                "messages": [
                    AI21ToolMessage(
                        role="tool",
                        tool_call_id="call_Jja7J89XsjrOLA5r!MEOW!SL",
                        content="42",
                    ),
                ],
            },
        ),
    ],
)
def test_convert_messages(
    chat_adapter: ChatAdapter,
    messages: List[BaseMessage],
    expected_messages: List[ChatMessage],
) -> None:
    converted_messages = chat_adapter.convert_messages(messages)
    assert converted_messages == expected_messages
