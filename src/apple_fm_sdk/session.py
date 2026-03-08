# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.

import asyncio
import json
import logging
from apple_fm_sdk.transcript import Transcript
from .c_helpers import (
    _ManagedObject,
    _register_handle,
    _session_callback,
    _session_structured_callback,
    _unregister_handle,
    StreamingCallback,
)
from .core import SystemLanguageModel
from .tool import Tool
from .generable import Generable, GeneratedContent
from .generation_schema import GenerationSchema
import threading
import queue
from typing import Any, Optional, AsyncIterator, Type, Union, overload
from .errors import FoundationModelsError

import ctypes

logger = logging.getLogger(__name__)

try:
    from . import _ctypes_bindings as lib
except ImportError:
    raise ImportError(
        "Foundation Models C bindings not found. Please ensure _foundationmodels_ctypes.py is available."
    )

Prompt = str  # Alias for prompt type


class LanguageModelSession(_ManagedObject):
    """Represents a language model session for foundation model interactions.

    A ``LanguageModelSession`` manages the lifecycle of a session with a foundation model,
    maintaining session history (transcript), handling tool calls, and providing both
    synchronous and streaming response capabilities.

    The session is thread-safe for sequential requests but does not support concurrent
    requests. If a request is in progress, attempting another request will wait for the
    first to complete.

    **Session Lifecycle:**

    1. **Creation**: Initialize with optional instructions, model configuration, and tools
    2. **Active Use**: Make requests via ``respond()`` or ``stream_response()``
    3. **Cleanup**: Automatically handled via context manager or explicit cleanup

    **Concurrent Request Handling:**

    Sessions use an internal lock to prevent concurrent requests. If you need to handle
    multiple requests simultaneously, create multiple session instances.

    Examples:
        Basic session creation and usage::

            import apple_fm_sdk as fm

            # Create a simple session
            session = fm.LanguageModelSession()
            response = await session.respond("Hello, how are you?")
            print(response)

        Session with instructions::

            import apple_fm_sdk as fm

            # Guide the model's behavior with instructions
            session = fm.LanguageModelSession(
                instructions="You are a helpful bird expert. Provide concise, "
                            "accurate information about birds."
            )
            response = await session.respond("What is a Swift?")

        Session with custom model and tools::

            import apple_fm_sdk as fm
            from my_tools import CalculatorTool, WeatherTool

            model = fm.SystemLanguageModel(
                temperature=0.7,
                top_p=0.9
            )

            session = fm.LanguageModelSession(
                instructions="You are a helpful assistant with access to tools.",
                model=model,
                tools=[CalculatorTool(), WeatherTool()]
            )

            response = await session.respond("What's the weather like in Cupertino?")

    See Also:
        - :class:`~apple_fm_sdk.core.SystemLanguageModel`: For model configuration
        - :class:`~apple_fm_sdk.tool.Tool`: For creating custom tools
        - :class:`~apple_fm_sdk.transcript.Transcript`: For accessing session history
    """

    def __init__(
        self,
        instructions: Optional[str] = None,
        model: Optional[SystemLanguageModel] = None,
        tools: Optional[list[Tool]] = None,
        _ptr=None,
    ):
        """Create a language model session.

        :param instructions: Optional system instructions to guide the model's behavior throughout
            the session. These instructions persist across all requests in the session.
            Example: "You are a helpful coding assistant."
        :type instructions: Optional[str]
        :param model: Optional specialized system model configuration. If not provided, uses default
            SystemLanguageModel() with standard settings. Use this to customize temperature,
            top_p, and other generation parameters.
        :type model: Optional[SystemLanguageModel]
        :param tools: Optional list of Tool instances that the model can invoke during generation.
            Tools enable the model to perform actions like calculations, API calls, or
            database queries. The model will automatically decide when to use tools based
            on the session context.
        :type tools: Optional[list[Tool]]
        :raises FoundationModelsError: If session creation fails

        Note:
            The session maintains a transcript of all interactions, which can be accessed
            via the ``transcript`` property. This transcript is automatically updated after
            each request.
        """
        # Initialize request lock for preventing concurrent requests
        self._request_lock = asyncio.Lock()
        self._active_task = None

        if _ptr is not None:
            # Internal constructor for specific ptr
            super().__init__(_ptr)
        else:
            # Create model pointer
            model_ptr = model._ptr if model else None

            # Encode instructions if provided
            instructions_cstr = None
            if instructions:
                instructions_cstr = instructions.encode("utf-8")

            # Create array of tool pointers
            tool_count = len(tools) if tools else 0
            tool_refs = (ctypes.c_void_p * tool_count)()
            if tools:
                for i, tool in enumerate(tools):
                    tool_refs[i] = tool._ptr

            # Create the session via C binding
            ptr = lib.FMLanguageModelSessionCreateFromSystemLanguageModel(
                model_ptr, instructions_cstr, tool_refs, tool_count
            )

            # Create transcript
            self.transcript = Transcript(ptr)

            # model object will be retained by LanguageModelSession in Swift
            # so here we don't need to retain model
            super().__init__(ptr)
        # This opaque pointer already has 1 ref count by `passRetained`

    @classmethod
    def from_transcript(
        cls,
        transcript: Transcript,
        model: Optional[SystemLanguageModel] = None,
        tools: Optional[list[Tool]] = None,
    ) -> "LanguageModelSession":
        """Create a new session from an existing transcript.

        This method creates a new LanguageModelSession initialized from an existing transcript.
        The new session will contain all the conversation history from the transcript, including
        user messages, model responses, and tool interactions.

        :param transcript: The Transcript instance to initialize the session from.
            This transcript contains the complete conversation history.
        :type transcript: Transcript
        :param model: Optional SystemLanguageModel to use for the new session. If not
            provided, uses the default model. This allows you to continue a conversation
            with a different model configuration than the original.
        :type model: Optional[SystemLanguageModel]
        :param tools: **IMPORTANT**: Tool mentions loaded from a Transcript are historical only.
            You must **also** pass tool instances here if you want to allow the model to make new
            tool calls in this session.
        :type tools: Optional[list[Tool]]
        :return: A new LanguageModelSession initialized with the transcript's history
        :rtype: LanguageModelSession
        :raises FoundationModelsError: If session creation fails

        Examples:
            Resume a conversation from a saved transcript::

                import apple_fm_sdk as fm
                import json

                # Load a saved transcript
                with open("transcript.json", "r") as f:
                    transcript_dict = json.load(f)

                transcript = await fm.Transcript.from_dict(transcript_dict)

                # Create a new session from the transcript
                session = fm.LanguageModelSession.from_transcript(transcript)

                # Continue the session
                response = await session.respond("Summarize the session so far.")

            Resume with tools::

                import apple_fm_sdk as fm
                from my_tools import CalculatorTool, WeatherTool

                # Load transcript that had tool calls
                transcript = await fm.Transcript.from_dict(transcript_dict)

                # IMPORTANT: You must pass the tool instances explicitly.
                # The transcript contains the history of tool calls, but not
                # the ability to make new tool calls unless you provide them.
                session = fm.LanguageModelSession.from_transcript(
                    transcript,
                    tools=[CalculatorTool(), WeatherTool()]
                )

                # Now the model can make new tool calls
                response = await session.respond("Calculate 15 * 24")

        Note:
            - The transcript contains the session instructions, so you don't need to
              pass instructions separately
            - The new session shares the transcript object with the original
            - Any new interactions will update the transcript
            - Tool mentions from the transcript are historical only. To allow the model to
              make new tool calls, you must explicitly pass the tool instances in the ``tools`` parameter.
        See Also:
            - :class:`~apple_fm_sdk.transcript.Transcript`: For working with transcripts
            - :meth:`~apple_fm_sdk.transcript.Transcript.from_dict`: For loading transcripts from JSON
            - :class:`~apple_fm_sdk.tool.Tool`: For creating custom tools
        """
        # Create model pointer
        model_ptr = model._ptr if model else None

        # Create array of tool pointers
        tool_count = len(tools) if tools else 0
        tool_refs = (ctypes.c_void_p * tool_count)()
        if tools:
            for i, tool in enumerate(tools):
                tool_refs[i] = tool._ptr

        # Create the session via C binding - use the new function that takes a transcript session
        ptr = lib.FMLanguageModelSessionCreateFromTranscript(
            transcript.session_ptr, model_ptr, tool_refs, tool_count
        )

        # Update transcript to use the new session pointer
        transcript._update_session_ptr(ptr)

        # Create session instance
        session = cls(_ptr=ptr)
        session.transcript = transcript  # Use the provided transcript
        return session

    @property
    def is_responding(self) -> bool:
        """Check if the session is currently responding to a request.

        Returns:
            bool: True if the session is currently processing a request, False otherwise
        """
        return lib.FMLanguageModelSessionIsResponding(self._ptr)

    def _reset_task_state(self):
        """Reset the task memory management state after a cancelled or failed request.

        This is an internal function that cleans up task-related state to ensure the
        session is ready to accept new requests. It does NOT create a new session or
        clear the session transcript - it only resets the internal task handling
        state.

        This function is automatically called after cancellations and errors, so client
        code should not need to call it directly.
        """
        lib.FMLanguageModelSessionReset(self._ptr)

    @overload  # This overload helps the type checker understand the return type
    async def respond(self, prompt: Prompt) -> str: ...

    @overload  # This overload helps the type checker understand the return type
    async def respond(
        self, prompt: Prompt, *, generating: type[Generable]
    ) -> Type[Any]: ...

    @overload  # This overload helps the type checker understand the return type
    async def respond(self, prompt: Prompt, *, generating: Generable) -> Type[Any]: ...

    @overload  # This overload helps the type checker understand the return type
    async def respond(
        self, prompt: Prompt, *, schema: GenerationSchema
    ) -> GeneratedContent: ...

    @overload  # This overload helps the type checker understand the return type
    async def respond(self, prompt: str, *, json_schema: dict) -> GeneratedContent: ...

    async def respond(
        self,
        prompt: str,
        generating: Optional[Union[Type[Generable], Generable]] = None,
        *,
        schema: Optional[GenerationSchema] = None,
        json_schema: Optional[dict] = None,
    ) -> Union[str, Any, GeneratedContent]:
        """Get a response to a prompt with optional guided generation.

        This function supports multiple response modes:

        1. **Basic text response**: Returns a plain string
        2. **Guided generation with Generable**: Returns a typed Python object
        3. **Guided generation with schema**: Returns structured GeneratedContent
        4. **Guided generation with JSON schema**: Returns structured GeneratedContent

        The session automatically updates its transcript after each response, maintaining
        the full session history.

        :param prompt: The input prompt string to send to the model
        :type prompt: str
        :param generating: Optional Generable type or instance for type-safe guided generation.
            When provided, the response will be constrained to match the structure of
            the Generable type and automatically converted to an instance of that type.
        :type generating: Optional[Union[Type[Generable], Generable]]
        :param schema: Optional GenerationSchema for explicit schema-based guided generation.
            Use this for custom schemas that don't map to a Generable type.
        :type schema: Optional[GenerationSchema]
        :param json_schema: Optional JSON schema dictionary for guided generation. The schema
            should follow JSON Schema specification.
        :type json_schema: Optional[dict]
        :return: Plain text response if no generation constraints are specified, or
            instance of generating type if ``generating`` parameter is provided, or
            structured content if ``schema`` or ``json_schema`` is provided
        :rtype: Union[str, Any, GeneratedContent]
        :raises FoundationModelsError: If the response fails or times out
        :raises ValueError: If both ``generating`` and ``schema`` are provided, or if the
            generating type is not a valid Generable
        :raises asyncio.CancelledError: If the request is cancelled

        Examples:
            Basic text response::
                import apple_fm_sdk as fm
                session = fm.LanguageModelSession()
                response = await session.respond("What is the capital of France?")
                print(response)  # Plain string response

            Guided generation with Generable type::

                import apple_fm_sdk as fm

                @fm.generable()
                class Cat:
                    name: str
                    age: int
                    profile: str

                session = fm.LanguageModelSession()
                cat = await session.respond(
                    "Generate a cat named Maomao",
                    generating=Cat
                )
                print(f"{cat.name} is {cat.age} years old")

            Multi-turn session::

                import apple_fm_sdk as fm

                session = fm.LanguageModelSession(
                    instructions="You are a helpful expert on architecture."
                )

                # First turn
                response1 = await session.respond("What is the tallest building in the world?")
                print(response1)

                # Second turn - context is maintained
                response2 = await session.respond(
                    "What's the architectural style of that building?"
                )
                print(response2)

        Note:
            - Only one of ``generating``, ``schema``, or ``json_schema`` can be specified
            - The session maintains session context across multiple ``respond()`` calls
            - Concurrent calls to ``respond()`` on the same session will be serialized
            - For streaming responses, use :meth:`stream_response` instead

        See Also:
            - :meth:`stream_response`: For streaming text responses
            - :class:`~apple_fm_sdk.generable.Generable`: For creating typed response structures
            - :class:`~apple_fm_sdk.generation_schema.GenerationSchema`: For custom schemas
        """
        # Validate arguments
        if generating is not None and schema is not None:
            raise ValueError("Cannot specify both 'generating' and 'schema' arguments")

        # Handle guided generation with generable type
        if generating is not None:
            if not isinstance(generating, Generable):
                raise ValueError(
                    f"{generating.__name__} is not a Generable type. Use @generable decorator."
                )

            # Get the generation schema for the type
            gen_schema = generating.generation_schema()

            # Use the schema-based respond method
            generated_content = await self._respond_with_schema(prompt, gen_schema)

            # Convert GeneratedContent to the target type
            return generating._from_generated_content(generated_content)

        # Handle guided generation with explicit schema
        if schema is not None:
            return await self._respond_with_schema(prompt, schema)

        # Handle guided generation from raw JSON schema string
        if json_schema is not None:
            return await self._respond_with_schema_from_json(prompt, json_schema)

        # Handle basic text response
        return await self._respond_basic(prompt)

    async def _respond_basic(self, prompt: str) -> str:
        """Get a complete basic text response to a prompt.

        Args:
            prompt: The input prompt

        Returns:
            The complete response text

        Raises:
            GenerationError: If the response fails with specific error type
            FoundationModelsError: For other errors like timeout
            ConcurrentRequestsError: If another request is already in progress
        """
        # Acquire lock to prevent concurrent requests
        async with self._request_lock:
            loop = asyncio.get_running_loop()
            future = loop.create_future()

            prompt_bytes = prompt.encode("utf-8")

            future_handle = _register_handle(future)

            task = lib.FMLanguageModelSessionRespond(
                self._ptr, prompt_bytes, future_handle, _session_callback
            )

            # Store active task reference
            self._active_task = task

            try:
                await future
            except asyncio.CancelledError as e:
                # Cancel the native task
                lib.FMTaskCancel(task)
                future.cancel()

                # Wait for the task to actually complete cancellation
                # Poll is_responding until it becomes False, with timeout
                max_wait_time = 1.0  # Maximum 1 second wait
                poll_interval = 0.01  # Poll every 10ms
                elapsed = 0.0

                while self.is_responding and elapsed < max_wait_time:
                    await asyncio.sleep(poll_interval)
                    elapsed += poll_interval

                # Reset task state to ensure the session is ready for new requests
                self._reset_task_state()

                raise e
            finally:
                # Clean up handle to prevent memory leaks
                _unregister_handle(future_handle)
                lib.FMRelease(task)
                self._active_task = None

            return future.result()

    async def _respond_with_schema(
        self, prompt: str, schema: GenerationSchema
    ) -> GeneratedContent:
        """Internal method for guided generation using a GenerationSchema."""
        # Acquire lock to prevent concurrent requests
        async with self._request_lock:
            loop = asyncio.get_running_loop()
            future = loop.create_future()

            prompt_bytes = prompt.encode("utf-8")

            future_handle = _register_handle(future)

            # Always use the proper C binding for guided generation
            task = lib.FMLanguageModelSessionRespondWithSchema(
                self._ptr,
                prompt_bytes,
                schema._ptr,
                future_handle,
                _session_structured_callback,
            )

            # Store active task reference
            self._active_task = task

            try:
                await future
            except asyncio.CancelledError as e:
                # Cancel the native task
                lib.FMTaskCancel(task)
                future.cancel()

                # Wait for the task to actually complete cancellation
                # Poll is_responding until it becomes False, with timeout
                max_wait_time = 1.0  # Maximum 1 second wait
                poll_interval = 0.01  # Poll every 10ms
                elapsed = 0.0

                while self.is_responding and elapsed < max_wait_time:
                    await asyncio.sleep(poll_interval)
                    elapsed += poll_interval

                # Reset task state to ensure the session is ready for new requests
                self._reset_task_state()

                raise e
            except Exception as e:
                # On any error, reset task state to ensure clean state
                self._reset_task_state()
                raise e
            finally:
                # Clean up handle to prevent memory leaks
                _unregister_handle(future_handle)
                lib.FMRelease(task)
                self._active_task = None

            return future.result()

    async def _respond_with_schema_from_json(
        self, prompt: str, json_schema: dict
    ) -> GeneratedContent:
        """Internal method for guided generation using a JSON schema string."""
        # Acquire lock to prevent concurrent requests
        async with self._request_lock:
            loop = asyncio.get_running_loop()
            future = loop.create_future()

            prompt_bytes = prompt.encode("utf-8")
            json_schema_bytes = json.dumps(json_schema).encode("utf-8")

            future_handle = _register_handle(future)

            # Use the C binding for guided generation with JSON schema
            task = lib.FMLanguageModelSessionRespondWithSchemaFromJSON(
                self._ptr,
                prompt_bytes,
                json_schema_bytes,
                future_handle,
                _session_structured_callback,
            )

            # Store active task reference
            self._active_task = task

            try:
                await future
            except asyncio.CancelledError as e:
                # Cancel the native task
                lib.FMTaskCancel(task)
                future.cancel()

                # Wait for the task to actually complete cancellation
                # Poll is_responding until it becomes False, with timeout
                max_wait_time = 1.0  # Maximum 1 second wait
                poll_interval = 0.01  # Poll every 10ms
                elapsed = 0.0

                while self.is_responding and elapsed < max_wait_time:
                    await asyncio.sleep(poll_interval)
                    elapsed += poll_interval

                # Reset task state to ensure the session is ready for new requests
                self._reset_task_state()

                raise e
            except Exception as e:
                # On any error, reset task state to ensure clean state
                self._reset_task_state()
                raise e
            finally:
                # Clean up handle to prevent memory leaks
                _unregister_handle(future_handle)
                lib.FMRelease(task)
                self._active_task = None

            return future.result()

    async def stream_response(self, prompt: Prompt) -> AsyncIterator:
        """Stream response chunks for a prompt (text only).

        This function provides real-time streaming of the model's response, yielding text
        snapshots as they become available. Each yielded value represents the complete
        response text generated so far, rather than the delta from the previous chunk.

        **Streaming Behavior:**

        - Yields complete text snapshots (not deltas) as generation progresses
        - The final yield contains the complete response
        - Automatically updates the session transcript after completion
        - Does not support guided generation (text responses only)
        - Can be cancelled mid-stream using asyncio cancellation

        :param prompt: The input prompt string to send to the model
        :type prompt: str
        :yields: Progressive snapshots of the response text. Each snapshot contains
            the full text generated so far, rather than only the new tokens.
        :ytype: str
        :raises FoundationModelsError: If streaming fails or encounters an error
        :raises asyncio.CancelledError: If the stream is cancelled

        Examples:
            Basic streaming::

                import apple_fm_sdk as fm

                session = fm.LanguageModelSession()

                async for chunk in session.stream_response("Tell me a story"):
                    print(chunk, end="", flush=True)

            Cancelling a stream::

                import asyncio
                import apple_fm_sdk as fm

                session = fm.LanguageModelSession()

                async def stream_with_timeout():
                    try:
                        async for chunk in session.stream_response("Write a long essay"):
                            print(chunk)
                            # Simulate some processing
                            await asyncio.sleep(0.1)
                    except asyncio.CancelledError:
                        print("Stream cancelled")
                        raise

                # Cancel after 5 seconds
                task = asyncio.create_task(stream_with_timeout())
                await asyncio.sleep(5)
                task.cancel()

            Streaming with error handling::
                import apple_fm_sdk as fm
                session = fm.LanguageModelSession()

                try:
                    async for chunk in session.stream_response("Hello"):
                        print(chunk)
                except fm.FoundationModelsError as e:
                    print(f"Streaming error: {e}")

        Note:
            - Streaming currently only supports basic text responses
            - For guided generation, use :meth:`respond` instead
            - Each snapshot contains the full text, rather than only new tokens
            - The session transcript is updated only after streaming completes
            - Breaking out of the async for loop early will properly clean up resources

        See Also:
            - :meth:`respond`: For non-streaming responses with guided generation support
        """
        # Handle basic text streaming only
        async for chunk in self._stream_response_basic(prompt):
            yield chunk

    async def _stream_response_basic(self, prompt: Prompt) -> AsyncIterator[str]:
        """Stream basic text response chunks for a prompt.

        Args:
            prompt: The input prompt

        Yields:
            Response text snapshots as they become available
        """
        callback = StreamingCallback()
        stream_thread = None
        stream_ptr_holder = [None]  # Use list to allow modification in nested function

        def _start_stream():
            prompt_bytes = prompt.encode("utf-8")
            stream_ptr = lib.FMLanguageModelSessionStreamResponse(
                self._ptr, prompt_bytes
            )
            stream_ptr_holder[0] = stream_ptr  # Store for cleanup

            if not stream_ptr:
                callback.error = FoundationModelsError(
                    "Failed to create response stream"
                )
                callback.queue.put(None)
                callback.completed.set()
                return

            try:
                lib.FMLanguageModelSessionResponseStreamIterate(
                    stream_ptr, None, callback._callback
                )
            except Exception as e:
                callback.error = FoundationModelsError(f"Stream iteration error: {e}")
                callback.queue.put(None)
                callback.completed.set()

        try:
            # Start streaming in a separate thread
            stream_thread = threading.Thread(target=_start_stream)
            stream_thread.daemon = True
            stream_thread.start()

            # Yield snapshots as they become available
            while True:
                try:
                    # Use a timeout to allow checking for completion
                    snapshot = callback.queue.get(timeout=0.1)
                    if snapshot is None:  # End signal
                        break
                    yield snapshot
                except queue.Empty:
                    # Check if we're done or have an error
                    if callback.completed.is_set():
                        # Check for any remaining items in queue
                        try:
                            while True:
                                snapshot = callback.queue.get_nowait()
                                if snapshot is None:
                                    break
                                yield snapshot
                        except queue.Empty:
                            pass
                        break
                    # Continue waiting for more data
                    continue

            # Check for errors after completion
            if callback.error:
                raise callback.error
        finally:
            # Ensure the stream thread completes before we exit
            # This prevents segfaults when breaking early from the stream
            if stream_thread and stream_thread.is_alive():
                # Wait for thread to finish - the native iteration must complete
                # before we can safely release the stream pointer
                stream_thread.join(timeout=2.0)

            # Now it's safe to release the stream pointer
            # This must happen after the thread completes to prevent segfaults
            if stream_ptr_holder[0]:
                lib.FMRelease(stream_ptr_holder[0])
