# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.

import json
import ctypes

from apple_fm_sdk.errors import _status_code_to_exception
from apple_fm_sdk.c_helpers import _get_error_string


try:
    from . import _ctypes_bindings as lib
except ImportError:
    raise ImportError(
        "Foundation Models C bindings not found. Please ensure _foundationmodels_ctypes.py is available."
    )


class Transcript:
    """Represents a foundation model session's transcript.

    A ``Transcript`` provides access to the complete session history of a
    LanguageModelSession, including all user prompts, model responses, and tool
    invocations. The transcript is automatically updated after each interaction
    with the model.

    **Transcript Format:**

    The transcript follows the Foundation Models Swift framework structure and is
    organized as a dictionary containing:

    - **version**: Integer version number (currently 1)
    - **type**: String identifier "FoundationModels.Transcript"
    - **transcript**: Object containing:
        - **entries**: List of session entries, each with:
            - ``id``: Unique UUID for the entry
            - ``role``: One of "instructions", "user", "response", or "tool"
            - ``contents``: Array of content objects with ``type``, ``id``, and type-specific fields

    **Entry Types by Role:**

    - **instructions**: System instructions and tool definitions
        - ``tools``: Array of available tool definitions
        - ``contents``: Array with text instructions
    - **user**: User messages and requests
        - ``contents``: Array of content objects (text and more)
        - ``options``: Optional configuration for the request
        - ``responseFormat``: Optional structured output schema
    - **response**: Model-generated responses
        - ``toolCalls``: Array of tool invocations (if tools were called)
        - ``contents``: Array of response content (text or structured data)
        - ``assets``: Array of model asset identifiers used
    - **tool**: Tool execution results
        - ``toolName``: Name of the executed tool
        - ``toolCallID``: UUID linking to the tool call
        - ``contents``: Array with tool execution results

    **When Transcripts Are Updated:**

    - After each ``respond()`` call completes successfully
    - After each ``stream_response()`` completes
    - After tool invocations are processed
    - NOT during streaming (only after completion)
    - NOT if a request fails or is cancelled

    Examples:
        Accessing transcript after session::

            import apple_fm_sdk as fm

            session = fm.LanguageModelSession()

            await session.respond("Hello!")
            await session.respond("What is Python?")

            # Get the full session history
            transcript = await session.transcript.to_dict()

            # Access the entries
            entries = transcript["transcript"]["entries"]
            for entry in entries:
                role = entry["role"]
                print(f"Entry role: {role}")
                if "contents" in entry:
                    for content in entry["contents"]:
                        if content["type"] == "text":
                            print(f"  Text: {content['text']}")

        Monitoring session length::
            import apple_fm_sdk as fm

            session = fm.LanguageModelSession()

            for i in range(5):
                await session.respond(f"Question {i}")

                transcript = await session.transcript.to_dict()
                entry_count = len(transcript["transcript"]["entries"])
                print(f"Session has {entry_count} entries")

        Examining tool calls in transcript::

            import apple_fm_sdk as fm
            from my_tools import CalculatorTool

            session = fm.LanguageModelSession(
                tools=[CalculatorTool()]
            )

            await session.respond("What is 15 * 24?")

            transcript = await session.transcript.to_dict()

            # Find entries with tool calls
            for entry in transcript["transcript"]["entries"]:
                if entry["role"] == "response" and "toolCalls" in entry:
                    print(f"Tool calls: {entry['toolCalls']}")
                elif entry["role"] == "tool":
                    print(f"Tool result for {entry['toolName']}")

        Saving session history::

            import apple_fm_sdk as fm
            import json

            session = fm.LanguageModelSession()

            # Have a session
            await session.respond("Hello")
            await session.respond("Tell me about Python")

            # Save transcript to file
            transcript = await session.transcript.to_dict()
            with open("session.json", "w") as f:
                json.dump(transcript, f, indent=2)

    Note:
        - The transcript object shares the session's internal pointer
        - Transcripts are read-only; you cannot modify session history
        - Large sessions may result in large transcript dictionaries
        - The transcript format follows the Foundation Models Swift framework structure
        - Accessing the transcript does not affect the session state

    See Also:
        - :class:`~apple_fm_sdk.session.LanguageModelSession`: For creating sessions
        - :meth:`~apple_fm_sdk.session.LanguageModelSession.respond`: For making requests
    """

    def __init__(
        self,
        _ptr,
    ):
        """Initialize a Transcript instance.

        Note:
            Transcript instances are automatically created by LanguageModelSession.
            Do not create Transcript instances directly.
        """
        # A transcript doesn't get it's own pointer, it uses the session's pointer
        self.session_ptr = _ptr

    async def to_dict(self) -> dict:
        """Get the current transcript of the session as a dictionary.

        This function retrieves the complete session history, including all
        user messages, model responses, and tool interactions. The transcript
        is returned as a structured dictionary following the Foundation Models
        Swift framework format.

        :return: The transcript data as a dictionary with the following structure:

            - ``version`` (int): Version number (currently 1)
            - ``type`` (str): Type identifier "FoundationModels.Transcript"
            - ``transcript`` (dict): Object containing:
                - ``entries`` (list): List of session entries, each with:
                    - ``id`` (str): Unique UUID for the entry
                    - ``role`` (str): One of "instructions", "user", "response", or "tool"
                    - ``contents`` (list): Array of content objects
                    - Additional role-specific fields (tools, toolCalls, options, and more)
        :rtype: dict
        :raises GenerationError: If fetching the transcript fails due to an internal error

        Example:
            ::

                import apple_fm_sdk as fm

                session = fm.LanguageModelSession()
                await session.respond("Hello!")

                transcript = await session.transcript.to_dict()

                # Access entries
                entries = transcript["transcript"]["entries"]
                for entry in entries:
                    print(f"{entry['role']}: {entry.get('id')}")

        Note:
            - This is an async function and must be awaited
            - The returned dictionary is a snapshot; it won't update automatically
            - Call this function again to get an updated transcript after new interactions
        """
        error_code = ctypes.c_int32()  # C error status code
        error_description = ctypes.POINTER(
            ctypes.c_char
        )()  # C error description pointer
        jsn_string = lib.FMLanguageModelSessionGetTranscriptJSONString(
            self.session_ptr, ctypes.byref(error_code), ctypes.byref(error_description)
        )

        # Check if we got a valid result or an error
        if jsn_string is None or (
            hasattr(jsn_string, "data") and jsn_string.data is None
        ):
            # An error occurred, raise appropriate exception
            err_code, err_desc = _get_error_string(error_code, error_description)
            error_msg = "Failed to fetch session transcript"
            if err_desc:
                error_msg = error_msg + ": " + err_desc
            raise _status_code_to_exception(err_code or error_code.value, error_msg)

        # Successfully got the JSON string, parse it and free the C string
        # The return value is wrapped in a String object by ctypes
        # The String wrapper handles memory, so we don't need to manually free
        json_str = str(jsn_string)
        result = json.loads(json_str)
        return result

    @classmethod
    async def from_dict(cls, dict: dict) -> "Transcript":
        """Create a Transcript from a dictionary representation.

        This method deserializes a transcript dictionary (typically loaded from JSON)
        and creates a Transcript instance. This is useful for loading saved session
        transcripts and resuming sessions with the full history intact.

        :param dict: Dictionary representation of a transcript, typically loaded from
            a JSON file. Must match the Foundation Models transcript format.
        :type dict: dict
        :return: A new Transcript instance initialized with the provided data
        :rtype: Transcript
        :raises GenerationError: If the dictionary format is invalid or cannot be parsed

        .. warning::
            **Tools in a transcript:**

            The transcript preserves the *history* of tool calls (what was called and
            what the results were), but not the *capability* to make new tool calls.
            Tool definitions stored in a transcript JSON will appear in the transcript's
            content history, but they will **not** be automatically available for the model
            to call. That's because the transcript doesn't actually contain the tool
            implementations. To allow the model to invoke tools mentioned in the transcript,
            implement each tool in Python and then create a session with both the transcript
            and tools using :meth:`~apple_fm_sdk.session.LanguageModelSession.from_transcript`

        Examples:
            Load a transcript from a JSON file::

                import apple_fm_sdk as fm
                import json

                # Load transcript from file
                with open("transcript.json", "r") as f:
                    transcript_dict = json.load(f)

                # Create Transcript instance
                transcript = await fm.Transcript.from_dict(transcript_dict)

                # Now you can create a session starting from this transcript
                session = fm.LanguageModelSession.from_transcript(transcript)

            Load and resume with tools::

                import apple_fm_sdk as fm
                import json
                from my_tools import CalculatorTool, WeatherTool

                # Load transcript that had tool calls
                with open("transcript_with_tools.json", "r") as f:
                    transcript_dict = json.load(f)

                transcript = await fm.Transcript.from_dict(transcript_dict)

                # IMPORTANT: Tools in the transcript are historical mentions only.
                # To allow the model to call a tool, you must explicitly instantiate each
                # tool in Python and then pass them to the session initializer.
                session = fm.LanguageModelSession.from_transcript(
                    transcript,
                    tools=[CalculatorTool(), WeatherTool()]
                )

        Note:
            - The dictionary must follow the Foundation Models transcript format
            - Tool definitions in the transcript are for historical reference only
            - To use tools with the transcript, pass them to
              :meth:`~apple_fm_sdk.session.LanguageModelSession.from_transcript`

        See Also:
            - :meth:`to_dict`: For converting a Transcript to a dictionary
            - :meth:`~apple_fm_sdk.session.LanguageModelSession.from_transcript`: For creating sessions from transcripts
            - :class:`~apple_fm_sdk.tool.Tool`: For creating custom tools
        """
        error_code = ctypes.c_int32()  # C error status code
        error_description = ctypes.POINTER(
            ctypes.c_char
        )()  # C error description pointer

        # Create a session pointer initialized with the transcript data from the dictionary
        # We can't create transcript pointer directly, so we create a new session pointer that
        # holds the Transcript.
        session_ptr = lib.FMTranscriptCreateFromJSONString(
            json.dumps(dict), ctypes.byref(error_code), ctypes.byref(error_description)
        )

        # Check if we got a valid result or an error
        if session_ptr is None:
            # An error occurred, raise appropriate exception
            err_code, err_desc = _get_error_string(error_code, error_description)
            error_msg = "Failed to create transcript from dictionary"
            if err_desc:
                error_msg = error_msg + ": " + err_desc
            raise _status_code_to_exception(err_code or error_code.value, error_msg)

        return cls(_ptr=session_ptr)

    def _update_session_ptr(self, new_ptr):
        """Update the session pointer associated with this transcript.

        This is used internally to keep the transcript's session pointer in sync
        with the LanguageModelSession's pointer after interactions that may change it.

        Note:
            - This method is for internal use only and should not be called directly.
            - The transcript shares the session's pointer, so updating it ensures the
              transcript reflects the current session state.
        """
        self.session_ptr = new_ptr
