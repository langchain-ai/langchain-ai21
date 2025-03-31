import json
from typing import Any, List, Optional, Sequence, Type, Union
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import BaseTool
from ai21.models.chat import ToolFunction as AI21ToolFunction

from langchain_core.runnables import Runnable
from pydantic import BaseModel
import httpx
import asyncio


class ChatMaestro(BaseChatModel):
    """Chat model using Maestro LLM."""

    api_key: str
    base_url: str
    tools: Optional[List[dict]] = None
    timeout: int = 60

    @property
    def _llm_type(self) -> str:
        """Return the type of LLM."""
        return "chat-maestro"

    def get_headers(self):
        return {"Authorization": f"Bearer {self.api_key}"}

    async def _acall(self, messages: List[BaseMessage], **kwargs: Any) -> str | Any:
        """Asynchronous API call to Maestro."""
        """Calls the Maestro API with formatted messages."""
        # formatted_messages = [{"role": "user" if m.type == "human" else "assistant", "content": m.content} for m in messages]
        formatted_messages = [{"role": "user", "content": message.content} for message in messages]
        requirements = kwargs.get("requirements", [])
        requirements = [{"name": requirement, "description": requirement} for requirement in  requirements]

        payload = {"input": formatted_messages
            # , "constraints": kwargs.get("constraints", None)
            , "requirements": requirements}

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(f"{self.base_url}/maestro/runs", json=payload, headers=self.get_headers())
            response.raise_for_status()

        if response.status_code != 200:
            raise Exception(f"API error: {response.text}")

        result = response.json()
        if result["status"] == "in_progress":
            # Handle polling or async streaming logic here
            await asyncio.sleep(1)  # Example delay
            return await self._poll_run(result["id"])

        return result["result"]

    async def _generate(self, messages: List[BaseMessage], stop: Optional[List[str]] = None, **kwargs: Any) -> ChatResult:
        """Generates a response using Maestro LLM."""
        response_data = await self._acall(messages, **kwargs)
        ai_message = AIMessage(content=response_data)

        if isinstance(response_data, dict) and "tool_calls" in response_data:
            ai_message.tool_calls = response_data["tool_calls"]

        generation = ChatGeneration(message=ai_message)
        return ChatResult(generations=[generation])

    def invoke(self, messages: List[BaseMessage], **kwargs: Any) -> AIMessage:
        """Wrapper for _generate to return just the AIMessage (blocking)."""
        return asyncio.run(self._generate(messages, **kwargs)).generations[0].message

    def bind_tools(self, tools: Sequence[Union[dict, Type[BaseModel], BaseTool]], **kwargs: Any) -> Runnable:
        """Attaches tools to the model."""
        self.tools = self.convert_lc_tool_calls_to_ai21_tool_calls(tools)
        return self

    def convert_lc_tool_calls_to_ai21_tool_calls(self, tool_calls) -> list[dict[str, Any]]:
        """
        Convert Langchain ToolCalls to AI21 ToolCalls.
        """
        if not isinstance(tool_calls, list):
            raise ValueError("Expected a list of ToolCall objects.")

        ai21_tool_calls = [self.to_ai21_tool_call(lc_tool_call) for lc_tool_call in tool_calls]
        return ai21_tool_calls

    async def _poll_run(self, run_id: str) -> str:
        """Polls the API until the run is completed or fails"""
        async with httpx.AsyncClient() as client:
            while True:
                response = await client.get(
                    f"{self.base_url}/maestro/runs/{run_id}", headers=self.get_headers()
                )
                if response.status_code != 200:
                    raise Exception(f"Polling error: {response.text}")

                result = response.json()
                if result["status"] in ["completed", "failed"]:
                    return result["result"]

                await asyncio.sleep(1)  # Polling interval

    @staticmethod
    def to_ai21_tool_call(tool) -> dict[str, Any]:
        """
        Converts the PromptInstructions instance to a Maestro ToolCall format.
        """
        # Example: use a predefined ID, or generate one dynamically
        tool_call_id = "prompt-instructions-tool"  # Unique ID for this tool call


        # Convert the PromptInstructions into the appropriate ToolFunction
        function_arguments = list(tool.model_fields.keys())

        tool_function = AI21ToolFunction(name="prompt_instructions_function",  # This could be a specific name based on your context
            arguments=json.dumps(function_arguments), )

        tool_call_dict = {"type": "web_search"}

         # tool_call_dict = {"type": "function",
         #    "function": {"name": tool_function.name, "arguments": json.loads(tool_function.arguments)  # Convert back to dict from JSON string
         #    }}

        return tool_call_dict
