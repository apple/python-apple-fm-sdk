# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.

"""
This module provides the core Python bindings for Foundation Models, including
system foundation model access and configuration.

The main classes provided are:

* :class:`SystemLanguageModel` - Interface to the on-device foundation model used by Apple Intelligence
* :class:`SystemLanguageModelUseCase` - Enumeration of model use cases
* :class:`SystemLanguageModelGuardrails` - Enumeration of guardrail settings
* :class:`SystemLanguageModelUnavailableReason` - Enumeration of unavailability reasons

Example:
    Basic usage of SystemLanguageModel::

        import apple_fm_sdk as fm

        model = fm.SystemLanguageModel()
        is_available, reason = model.is_available()
        if is_available:
            # Use the model
            pass
"""

from .c_helpers import (
    _ManagedObject,
    _register_handle,
    _token_count_callback,
    _unregister_handle,
)
from enum import IntEnum
from typing import Optional, TYPE_CHECKING, Union
from .errors import FoundationModelsError

import asyncio
import ctypes
from ctypes import c_int

if TYPE_CHECKING:
    from .prompt import Prompt
    from .generation_schema import GenerationSchema
    from .transcript import Transcript
    from .tool import Tool

try:
    from . import _ctypes_bindings as lib
except ImportError:
    raise ImportError(
        "Foundation Models C bindings not found. Please ensure _foundationmodels_ctypes.py is available."
    )


class SystemLanguageModelUnavailableReason(IntEnum):
    """Reasons why a system foundation model might be unavailable.

    This enumeration defines the possible reasons why a system foundation model
    cannot be used on the current device or in the current context.

    Attributes:
        APPLE_INTELLIGENCE_NOT_ENABLED: Apple Intelligence features are not enabled
            on this device or for this user.
        DEVICE_NOT_ELIGIBLE: The device does not meet the requirements for running
            the system language model.
        MODEL_NOT_READY: The model is still being downloaded or prepared and is not
            yet ready for use.
        UNKNOWN: The reason for unavailability is unknown or not specified.
    """

    APPLE_INTELLIGENCE_NOT_ENABLED = 0
    DEVICE_NOT_ELIGIBLE = 1
    MODEL_NOT_READY = 2
    UNKNOWN = 0xFF


class SystemLanguageModelUseCase(IntEnum):
    """Use cases for system foundation models.

    This enumeration defines the different use cases that can be specified when
    creating a system foundation model. The use case helps optimize the model's
    behavior for specific tasks.

    Attributes:
        GENERAL: General-purpose foundation model use case suitable for a wide range
            of natural language processing tasks.
        CONTENT_TAGGING: Specialized use case optimized for content classification
            and tagging tasks.
    """

    GENERAL = 0
    CONTENT_TAGGING = 1


class SystemLanguageModelGuardrails(IntEnum):
    """Guardrail settings for system foundation models.

    This enumeration defines the different levels of content filtering and safety
    guardrails that can be applied to system foundation models. Guardrails help
    ensure appropriate and safe model behavior.

    Attributes:
        DEFAULT: Standard guardrails with balanced content filtering appropriate
            for general use cases.
        PERMISSIVE_CONTENT_TRANSFORMATIONS: More permissive guardrails that allow
            greater flexibility in content transformation tasks while maintaining
            basic safety measures.
    """

    DEFAULT = 0
    PERMISSIVE_CONTENT_TRANSFORMATIONS = 1


class SystemLanguageModel(_ManagedObject):
    """Represents the on-device foundation model used by Apple Intelligence.

    This class provides the main interface for interacting with the system foundation
    model. It manages the lifecycle of the underlying C model object and provides
    methods to check availability and configure model behavior.

    The model can be configured with different use cases and guardrail settings
    to optimize behavior for specific tasks and ensure appropriate content filtering.

    Example:
        Creating and checking availability of a model::

            import apple_fm_sdk as fm

            # Create a model with default settings
            model = fm.SystemLanguageModel()

            # Create a model with specific settings
            model = fm.SystemLanguageModel(
                use_case=fm.SystemLanguageModelUseCase.CONTENT_TAGGING,
                guardrails=fm.SystemLanguageModelGuardrails.PERMISSIVE_CONTENT_TRANSFORMATIONS
            )

            # Check if the model is available
            is_available, reason = model.is_available()
            if not is_available:
                print(f"Model unavailable: {reason}")
    """

    def __init__(
        self,
        use_case: SystemLanguageModelUseCase = SystemLanguageModelUseCase.GENERAL,
        guardrails: SystemLanguageModelGuardrails = SystemLanguageModelGuardrails.DEFAULT,
        _ptr=None,
    ):
        """Create a system language model.

        Initializes a new system language model with the specified use case and
        guardrail settings. The model is backed by a C implementation and this
        constructor manages the creation and lifecycle of the underlying C object.

        :param use_case: The use case for the model, which optimizes its behavior
            for specific tasks. Defaults to GENERAL for general-purpose use.
        :type use_case: SystemLanguageModelUseCase
        :param guardrails: The guardrail settings that control content filtering
            and safety measures. Defaults to DEFAULT for standard filtering.
        :type guardrails: SystemLanguageModelGuardrails

        :raises ImportError: If the Foundation Models C bindings are not available.

        Example:
            Creating a custom-configured model::

                import apple_fm_sdk as fm

                # Create with default settings
                model = fm.SystemLanguageModel()

                # Create with custom settings
                model = fm.SystemLanguageModel(
                    use_case=fm.SystemLanguageModelUseCase.CONTENT_TAGGING,
                    guardrails=fm.SystemLanguageModelGuardrails.PERMISSIVE_CONTENT_TRANSFORMATIONS
                )
        """
        if _ptr is not None:
            # Internal constructor for specific ptr
            super().__init__(_ptr)
        else:
            # Public constructor - create new model
            ptr = lib.FMSystemLanguageModelCreate(use_case.value, guardrails.value)
            super().__init__(ptr)
        # This opaque pointer already has 1 ref count by `passRetained`

    def is_available(
        self,
    ) -> tuple[bool, Optional[SystemLanguageModelUnavailableReason]]:
        """Check if the model is available for use.

        This method queries the system to determine whether the language model
        can be used on the current device and in the current context. It returns
        both an availability status and, if unavailable, a reason code explaining
        why the model cannot be used.

        Common reasons for unavailability include Apple Intelligence not being
        enabled, the device not meeting requirements, or the model still being
        downloaded.

        :return: A tuple containing:

            - bool: True if the model is available, False otherwise
            - Optional[SystemLanguageModelUnavailableReason]: The reason for
              unavailability if the model is not available, or None if it is
              available
        :rtype: tuple[bool, Optional[SystemLanguageModelUnavailableReason]]

        Example:
            Checking model availability::

                import apple_fm_sdk as fm

                test_model = fm.SystemLanguageModel()
                is_available, reason = test_model.is_available()

                if is_available:
                    print("Model is ready to use")
                else:
                    if reason:
                        print(f"Model unavailable: {reason.name}")

                        if reason == fm.SystemLanguageModelUnavailableReason.MODEL_NOT_READY:
                            print("Please wait for the model to finish downloading")
                        elif reason == fm.SystemLanguageModelUnavailableReason.DEVICE_NOT_ELIGIBLE:
                            print("This device does not support the model")

        Note:
            This method should be called before attempting to use the model to
            ensure it is ready for inference.
        """
        reason = c_int()
        is_available = lib.FMSystemLanguageModelIsAvailable(
            self._ptr, ctypes.byref(reason)
        )

        if is_available:
            return True, None
        else:
            return False, SystemLanguageModelUnavailableReason(reason.value)

    @property
    def context_size(self) -> int:
        """The model's maximum context window size, measured in tokens.

        The context size is the total number of tokens (prompt, instructions, tools,
        and response combined) that the model can process in a single session. Use
        :meth:`token_count` to measure how many tokens a given input consumes and
        compare it against this value.

        :return: The maximum context window size in tokens.
        :rtype: int

        Example:
            Checking how much of the context window a prompt uses::

                import apple_fm_sdk as fm

                model = fm.SystemLanguageModel()
                budget = model.context_size
                used = await model.token_count("Tell me about the history of Swift.")
                print(f"Using {used} of {budget} tokens")
        """
        return int(lib.FMSystemLanguageModelGetContextSize(self._ptr))

    async def _token_count(self, start_task) -> int:
        """Run a token-counting C call and await its result.

        :param start_task: A callable that takes ``(future_handle, callback)`` and
            returns the native ``FMTaskRef`` for the dispatched work.
        :type start_task: Callable
        :return: The token count returned by the model.
        :rtype: int
        """
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        future_handle = _register_handle(future)

        task = start_task(future_handle, _token_count_callback)
        try:
            await future
        except asyncio.CancelledError:
            lib.FMTaskCancel(task)
            future.cancel()
            raise
        finally:
            _unregister_handle(future_handle)
            lib.FMRelease(task)

        return future.result()

    async def token_count(
        self,
        value: "Optional[Union[Prompt, GenerationSchema, Transcript, list[Tool]]]" = None,
        *,
        instructions: Optional[str] = None,
    ) -> int:
        """Count the number of tokens a given input would consume.

        This method measures how many tokens the model would use to represent the
        provided input, without running generation. It supports several input kinds:

        1. **Prompt** (a string or list of prompt components): pass the prompt as ``value``
        2. **Instructions** (system instructions text): pass via the ``instructions`` keyword
        3. **Tools** (a list of :class:`~apple_fm_sdk.tool.Tool`): pass the list as ``value``
        4. **GenerationSchema**: pass the schema as ``value``
        5. **Transcript**: pass the transcript as ``value``

        Compare the result against :attr:`context_size` to gauge how much of the
        model's context window an input occupies.

        :param value: The input to count tokens for. One of a prompt (``str`` or list
            of prompt components), a list of :class:`~apple_fm_sdk.tool.Tool`, a
            :class:`~apple_fm_sdk.generation_schema.GenerationSchema`, or a
            :class:`~apple_fm_sdk.transcript.Transcript`. Omit when counting tokens
            for ``instructions``.
        :type value: Union[Prompt, GenerationSchema, Transcript, list[Tool]]
        :param instructions: System instructions text to count tokens for. Mutually
            exclusive with ``value``.
        :type instructions: Optional[str]
        :return: The number of tokens the input consumes.
        :rtype: int
        :raises ValueError: If both ``value`` and ``instructions`` are provided, or
            if neither is provided.
        :raises FoundationModelsError: If token counting fails.

        Examples:
            Counting tokens in a prompt::

                import apple_fm_sdk as fm

                model = fm.SystemLanguageModel()
                count = await model.token_count("Hello, world!")

            Counting tokens in instructions::

                count = await model.token_count(instructions="You are a helpful assistant.")

            Counting tokens for tools, a schema, or a transcript::

                count = await model.token_count([CalculatorTool(), WeatherTool()])
                count = await model.token_count(Cat.generation_schema())
                count = await model.token_count(session.transcript)
        """
        # Imported lazily to avoid circular imports at module load time.
        from .generation_schema import GenerationSchema
        from .transcript import Transcript
        from .tool import Tool
        from .prompt import _composed_prompt_from_prompt

        if instructions is not None:
            if value is not None:
                raise ValueError(
                    "Provide either a value or instructions to token_count(), not both"
                )
            instructions_cstr = instructions.encode("utf-8")
            return await self._token_count(
                lambda handle, cb: lib.FMSystemLanguageModelTokenCountForInstructions(
                    self._ptr, instructions_cstr, handle, cb
                )
            )

        if value is None:
            raise ValueError("token_count() requires either a value or instructions")

        if isinstance(value, GenerationSchema):
            return await self._token_count(
                lambda handle, cb: lib.FMSystemLanguageModelTokenCountForSchema(
                    self._ptr, value._ptr, handle, cb
                )
            )

        if isinstance(value, Transcript):
            return await self._token_count(
                lambda handle, cb: lib.FMSystemLanguageModelTokenCountForTranscript(
                    self._ptr, value.session_ptr, handle, cb
                )
            )

        # A list of Tool instances.
        if isinstance(value, list) and all(isinstance(t, Tool) for t in value):
            tool_count = len(value)
            tool_refs = (ctypes.c_void_p * tool_count)()
            for i, tool in enumerate(value):
                tool_refs[i] = tool._ptr
            return await self._token_count(
                lambda handle, cb: lib.FMSystemLanguageModelTokenCountForTools(
                    self._ptr, tool_refs, tool_count, handle, cb
                )
            )

        # Otherwise treat the value as a prompt (str or list of prompt components).
        composed_prompt = _composed_prompt_from_prompt(value)
        return await self._token_count(
            lambda handle, cb: lib.FMSystemLanguageModelTokenCountForPrompt(
                self._ptr, composed_prompt, handle, cb
            )
        )
