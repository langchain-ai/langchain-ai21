from typing import Any, Dict, List, Literal, Optional, Type

from ai21.models.maestro.run import Budget, RunResponse, ToolType
from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from typing_extensions import TypedDict

from langchain_ai21.ai21_base import AI21Base


class FileSearchToolResource(TypedDict, total=False):
    retrieval_similarity_threshold: Optional[float]
    labels: Optional[List[str]]
    labels_filter_mode: Optional[Literal["AND", "OR"]]
    labels_filter: Optional[dict]
    file_ids: Optional[List[str]]
    retrieval_strategy: Optional[str]
    max_neighbors: Optional[int]


class WebSearchToolResource(TypedDict, total=False):
    urls: Optional[List[str]]


class ToolResources(TypedDict, total=False):
    file_search: Optional[FileSearchToolResource]
    web_search: Optional[WebSearchToolResource]


class ChatMaestro(BaseChatModel, AI21Base):
    """Chat model using Maestro LLM."""

    output_type: Optional[Dict[str, Any]] = None
    """Optional dictionary specifying the output type."""

    models: Optional[List[str]] = None
    """Optional list of model names to use.
        Available models https://github.com/AI21Labs/ai21-python?tab=readme-ov-file#model-types"""

    tools: Optional[List[Dict[str, ToolType]]] = None
    """Optional list of tools."""

    tool_resources: Optional[ToolResources] = None
    """Optional resources for the tools."""

    context: Optional[Dict[str, Any]] = None
    """Optional dictionary providing context for the chat."""

    budget: Optional[Budget] = None
    """Optional budget constraints for the chat."""

    poll_interval_sec: Optional[float] = 1
    """Interval in seconds for polling the run status."""

    poll_timeout_sec: Optional[float] = 120
    """Timeout in seconds for polling the run status."""

    @property
    def _llm_type(self) -> str:
        """Return the type of LLM."""
        return "chat-maestro"

    def _call(self, messages: List[BaseMessage], **kwargs: Any) -> RunResponse:
        """API call to Maestro."""
        payload = self._prepare_payload(messages, **kwargs)
        result = self.client.beta.maestro.runs.create_and_poll(**payload)
        if result.status != "completed":
            raise RuntimeError(f"Maestro run failed with status: {result.status}")

        return result

    async def _acall(self, messages: List[BaseMessage], **kwargs: Any) -> RunResponse:
        """Asynchronous API call to Maestro."""
        payload = self._prepare_payload(messages, **kwargs)
        result = await self.async_client.beta.maestro.runs.create_and_poll(**payload)
        if result.status != "completed":
            raise RuntimeError(f"Maestro run failed with status: {result.status}")

        return result

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generates a response using Maestro LLM."""
        response_data = self._call(messages, **kwargs)
        return self._handle_chat_result(response_data)

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Asynchronous agent call to Maestro."""
        response_data = await self._acall(messages, **kwargs)
        return self._handle_chat_result(response_data)

    @staticmethod
    def _prepare_payload(messages: List[BaseMessage], **kwargs: Any) -> Dict[str, Any]:
        """Prepare the payload for the API call with validation."""
        formatted_messages = [
            {"role": "user", "content": message.content} for message in messages
        ]
        payload = {"input": formatted_messages, **kwargs}

        requirements = payload.pop("requirements", [])
        if requirements:
            ChatMaestro.validate_list(requirements, "requirements")
            payload["requirements"] = [
                {"name": req, "description": req} for req in requirements
            ]

        variables = payload.pop("variables", [])
        if variables:
            ChatMaestro.validate_list(variables, "variables")
            variables_str = " ".join(variables)
            payload["requirements"] = payload.get("requirements", []) + [
                {
                    "name": "output should contain only these variables:"
                    f" {variables_str}",
                    "description": variables_str,
                }
            ]

        return payload

    @staticmethod
    def validate_list(obj: List[str], obj_name: str, expected_type: Type = str) -> None:
        """Validate that obj is a list of the expected type."""
        if obj is not None and (
            not isinstance(obj, list)
            or any(not isinstance(var, expected_type) for var in obj)
        ):
            raise ValueError(f"{obj_name} must be a list of {expected_type.__name__}")

    @staticmethod
    def _handle_chat_result(response_data: RunResponse) -> ChatResult:
        """Handle the response data from the Maestro run."""
        ai_message = AIMessage(content=response_data.result)
        generation = ChatGeneration(message=ai_message)
        return ChatResult(generations=[generation])
