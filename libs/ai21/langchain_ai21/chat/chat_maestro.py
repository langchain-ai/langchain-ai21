import asyncio
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

    async def _acall(self, messages: List[BaseMessage], **kwargs: Any) -> RunResponse:
        """Asynchronous API call to Maestro."""
        formatted_messages = [{"role": "user", "content": message.content} for message in messages]
        requirements = kwargs.get("requirements", [])
        requirements = [{"name": requirement, "description": requirement} for requirement in requirements]
        result = self.client.beta.maestro.runs.create_and_poll(input=formatted_messages, requirements=requirements)

        return result

    async def _generate(self, messages: List[BaseMessage], stop: Optional[List[str]] = None, **kwargs: Any) -> ChatResult:
        """Generates a response using Maestro LLM."""
        response_data = await self._acall(messages, **kwargs)
        ai_message = AIMessage(content=response_data.result)
        generation = ChatGeneration(message=ai_message)

        return ChatResult(generations=[generation])

    def invoke(self, messages: List[BaseMessage], **kwargs: Any) -> AIMessage:
        """Wrapper for _generate to return just the AIMessage (blocking)."""
        return asyncio.run(self._generate(messages, **kwargs)).generations[0].message
