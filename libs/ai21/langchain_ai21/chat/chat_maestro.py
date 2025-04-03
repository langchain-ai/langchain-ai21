from typing import Any, List, Optional, Dict, Literal
from typing_extensions import TypedDict

from ai21.models.maestro.run import RunResponse, ToolType, Budget
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
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

    output_type: Optional[dict[str, Any]] = None
    models: Optional[List[str]] = None
    tools: Optional[List[Dict[str, ToolType]]] = None
    tool_resources: Optional[ToolResources] = None
    context: Optional[Dict[str, Any]] = None
    budget: Optional[Budget] = None

    @property
    def _llm_type(self) -> str:
        """Return the type of LLM."""
        return "chat-maestro"

    def _call(self, messages: List[BaseMessage], **kwargs: Any) -> RunResponse:
        """ API call to Maestro."""
        formatted_messages = [{"role": "user", "content": message.content} for message in messages]
        payload = {"input": formatted_messages}

        requirements = kwargs.pop("requirements")
        if requirements:
            requirements = [{"name": requirement, "description": requirement} for requirement in  requirements]
            payload["requirements"] = requirements

        result = self.client.beta.maestro.runs.create_and_poll(**payload, **kwargs)
        if result.status != "completed":
            raise RuntimeError(f"Maestro run failed with status: {result.status}")

        return result

    def _generate(self, messages: List[BaseMessage], stop: Optional[List[str]] = None, **kwargs: Any) -> ChatResult:
        """Generates a response using Maestro LLM."""
        response_data = self._call(messages, **kwargs)
        ai_message = AIMessage(content=response_data.result)
        generation = ChatGeneration(message=ai_message)

        return ChatResult(generations=[generation])
