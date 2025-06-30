from typing import Type

import pytest

from langchain_ai21.chat.chat_adapter import ChatAdapter, JambaChatCompletionsAdapter
from langchain_ai21.chat.chat_factory import create_chat_adapter
from tests.unit_tests.conftest import JAMBA_MINI_CHAT_MODEL_NAME


@pytest.mark.parametrize(
    ids=[
        "when_jamba_model",
    ],
    argnames=["model", "expected_chat_type"],
    argvalues=[
        (JAMBA_MINI_CHAT_MODEL_NAME, JambaChatCompletionsAdapter),
    ],
)
def test_create_chat_adapter_with_supported_models(
    model: str, expected_chat_type: Type[ChatAdapter]
) -> None:
    adapter = create_chat_adapter(model)
    assert isinstance(adapter, expected_chat_type)


def test_create_chat_adapter__when_model_not_supported() -> None:
    with pytest.raises(ValueError):
        create_chat_adapter("unsupported-model")
