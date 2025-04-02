from typing import Any, List, Optional

from ai21.models.maestro.run import RunResponse
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from langchain_ai21.ai21_base import AI21Base


class ChatMaestro(BaseChatModel, AI21Base):
    """Chat model using Maestro LLM."""

    @property
    def _llm_type(self) -> str:
        """Return the type of LLM."""
        return "chat-maestro"

    def _call(self, messages: List[BaseMessage], **kwargs: Any) -> RunResponse:
        """ API call to Maestro."""
        formatted_messages = [{"role": "user", "content": message.content} for message in messages]
        payload = {"input": formatted_messages}

        requirements = kwargs.get("requirements", [])
        if requirements:
            requirements = [{"name": requirement, "description": requirement} for requirement in  requirements]
            payload["requirements"] = requirements

        result = self.client.beta.maestro.runs.create_and_poll(payload)
        return result

    def _generate(self, messages: List[BaseMessage], stop: Optional[List[str]] = None, **kwargs: Any) -> ChatResult:
        """Generates a response using Maestro LLM."""
        response_data = self._call(messages, **kwargs)
        ai_message = AIMessage(content=response_data.result)
        generation = ChatGeneration(message=ai_message)

        return ChatResult(generations=[generation])
