import asyncio
from functools import partial
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Type,
    Union,
)

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import LanguageModelInput
from langchain_core.language_models.chat_models import (
    BaseChatModel,
    LangSmithParams,
    generate_from_stream,
)
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import ConfigDict, model_validator
from typing_extensions import Self

from langchain_ai21.ai21_base import AI21Base
from langchain_ai21.chat.chat_adapter import ChatAdapter
from langchain_ai21.chat.chat_factory import create_chat_adapter


class ChatAI21(BaseChatModel, AI21Base):
    """ChatAI21 chat model. Different model types support different parameters and
    different parameter values. Please read the [AI21 reference documentation]
    (https://docs.ai21.com/reference) for your model to understand which parameters
    are available.

    Example:
        .. code-block:: python

            from langchain_ai21 import ChatAI21


            model = ChatAI21(
                # defaults to os.environ.get("AI21_API_KEY")
                api_key="my_api_key"
            )
    """

    model: str
    """Model type you wish to interact with. 
        You can view the options at https://github.com/AI21Labs/ai21-python?tab=readme-ov-file#model-types"""

    stop: Optional[List[str]] = None
    """Default stop sequences."""

    max_tokens: int = 512
    """The maximum number of tokens to generate for each response."""

    temperature: float = 0.4
    """A value controlling the "creativity" of the model's responses."""

    top_p: float = 1
    """A value controlling the diversity of the model's responses."""

    n: int = 1
    """Number of chat completions to generate for each prompt."""
    streaming: bool = False

    _chat_adapter: ChatAdapter

    @model_validator(mode="after")
    def validate_environment(self) -> Self:
        """Validate the environment."""
        model = self.model
        self._chat_adapter = create_chat_adapter(model)
        return self

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return "chat-ai21"

    @property
    def _default_params(self) -> Mapping[str, Any]:
        base_params = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "n": self.n,
        }
        if self.stop:
            base_params["stop_sequences"] = self.stop

        return base_params

    def _get_ls_params(
        self, stop: Optional[List[str]] = None, **kwargs: Any
    ) -> LangSmithParams:
        """Get standard params for tracing."""
        params = self._get_invocation_params(stop=stop, **kwargs)
        ls_params = LangSmithParams(
            ls_provider="ai21",
            ls_model_name=self.model,
            ls_model_type="chat",
            ls_temperature=params.get("temperature", self.temperature),
        )
        if ls_max_tokens := params.get("max_tokens", self.max_tokens):
            ls_params["ls_max_tokens"] = ls_max_tokens
        if ls_stop := stop or params.get("stop", None) or self.stop:
            ls_params["ls_stop"] = ls_stop
        return ls_params

    def _build_params_for_request(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Mapping[str, Any]:
        params = {}
        converted_messages = self._chat_adapter.convert_messages(messages)

        if stop is not None:
            if "stop" in kwargs:
                raise ValueError("stop is defined in both stop and kwargs")
            params["stop_sequences"] = stop

        return {
            **converted_messages,
            **self._default_params,
            **params,
            **kwargs,
        }

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        stream: Optional[bool] = None,
        **kwargs: Any,
    ) -> ChatResult:
        should_stream = stream or self.streaming

        if should_stream:
            return self._handle_stream_from_generate(
                messages=messages,
                stop=stop,
                run_manager=run_manager,
                **kwargs,
            )

        params = self._build_params_for_request(
            messages=messages,
            stop=stop,
            stream=should_stream,
            **kwargs,
        )

        messages = self._chat_adapter.call(self.client, **params)
        generations = [ChatGeneration(message=message) for message in messages]

        return ChatResult(generations=generations)

    def _handle_stream_from_generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        stream_iter = self._stream(
            messages=messages,
            stop=stop,
            run_manager=run_manager,
            **kwargs,
        )
        return generate_from_stream(stream_iter)

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        params = self._build_params_for_request(
            messages=messages,
            stop=stop,
            stream=True,
            **kwargs,
        )

        for chunk in self._chat_adapter.call(self.client, **params):
            if run_manager and isinstance(chunk.message.content, str):
                run_manager.on_llm_new_token(token=chunk.message.content, chunk=chunk)
            yield chunk

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        return await asyncio.get_running_loop().run_in_executor(
            None, partial(self._generate, **kwargs), messages, stop, run_manager
        )

    def _get_system_message_from_message(self, message: BaseMessage) -> str:
        if not isinstance(message.content, str):
            raise ValueError(
                f"System Message must be of type str. Got {type(message.content)}"
            )

        return message.content

    def bind_tools(
        self,
        tools: Sequence[Union[Dict[str, Any], Type, Callable, BaseTool]],
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, BaseMessage]:
        formatted_tools = [convert_to_openai_tool(tool) for tool in tools]
        return super().bind(tools=formatted_tools, **kwargs)
