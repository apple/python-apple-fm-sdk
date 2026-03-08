# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.

"""
Test all code snippets from Python library code source documentation

RULES:
Use this consistent testing format:
- Each test function should correspond to a specific code snippet or section in the documentation.
- Include comments indicating the source documentation file and section for clarity.
- No extra tests beyond those needed to validate the snippets.

Copy the snippet from the source **exactly** as it appears in the documentation.
Surround the original source with:
##############################################################################
# From: src/apple_fm_sdk/<source_file>.py
# class, function, or other entity name: <source_section_name>
<actual code here uncommented>
##############################################################################

The test passes if the snippet runs without errors. No additional assertions are necessary
beyond ensuring the snippet executes successfully.
"""

import pytest
from tester_tools.tester_tools import SimpleCalculatorTool


# =============================================================================
# SESSION TESTS (from src/apple_fm_sdk/session.py)
# =============================================================================


@pytest.mark.asyncio
async def test_session_basic_creation(model):
    """Test from: src/apple_fm_sdk/session.py - LanguageModelSession class docstring"""
    print("\n=== Testing Session Basic Creation ===")

    ##############################################################################
    # From: src/apple_fm_sdk/session.py
    # class, function, or other entity name: LanguageModelSession - Basic session creation
    import apple_fm_sdk as fm

    # Create a simple session
    session = fm.LanguageModelSession()
    response = await session.respond("Hello, how are you?")
    print(response)
    ##############################################################################

    print("✅ Session basic creation - PASSED")


@pytest.mark.asyncio
async def test_session_with_instructions(model):
    """Test from: src/apple_fm_sdk/session.py - LanguageModelSession with instructions"""
    print("\n=== Testing Session with Instructions ===")

    ##############################################################################
    # From: src/apple_fm_sdk/session.py
    # class, function, or other entity name: LanguageModelSession - Session with instructions
    # Guide the model's behavior with instructions
    import apple_fm_sdk as fm

    # Guide the model's behavior with instructions
    session = fm.LanguageModelSession(
        instructions="You are a helpful bird expert. Provide concise, "
        "accurate information about birds."
    )
    response = await session.respond("What is a Swift?")
    ##############################################################################

    assert response is not None
    print("✅ Session with instructions - PASSED")


@pytest.mark.asyncio
async def test_respond_basic_text(model):
    """Test from: src/apple_fm_sdk/session.py - respond() basic text response"""
    print("\n=== Testing Basic Text Response ===")

    ##############################################################################
    # From: src/apple_fm_sdk/session.py
    # class, function, or other entity name: respond - Basic text response
    import apple_fm_sdk as fm

    session = fm.LanguageModelSession()
    response = await session.respond("What is the capital of France?")
    print(response)  # Plain string response
    ##############################################################################

    print("✅ Basic text response - PASSED")


@pytest.mark.asyncio
async def test_respond_guided_generation(model):
    """Test from: src/apple_fm_sdk/session.py - respond() with Generable"""
    print("\n=== Testing Guided Generation ===")

    ##############################################################################
    # From: src/apple_fm_sdk/session.py
    # class, function, or other entity name: respond - Guided generation with Generable type
    import apple_fm_sdk as fm

    @fm.generable()
    class Cat:
        name: str
        age: int
        profile: str

    session = fm.LanguageModelSession()
    cat = await session.respond("Generate a cat named Maomao", generating=Cat)
    print(f"{cat.name} is {cat.age} years old")
    ##############################################################################
    assert cat.name is not None
    print("✅ Guided generation - PASSED")


@pytest.mark.asyncio
async def test_respond_multi_turn(model):
    """Test from: src/apple_fm_sdk/session.py - respond() multi-turn conversation"""
    print("\n=== Testing Multi-turn Conversation ===")

    ##############################################################################
    # From: src/apple_fm_sdk/session.py
    # class, function, or other entity name: respond - Multi-turn conversation
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
    ##############################################################################

    print("✅ Multi-turn conversation - PASSED")


@pytest.mark.asyncio
async def test_session_with_model_and_tools(model):
    """Test from: src/apple_fm_sdk/session.py - LanguageModelSession with custom model and tools"""
    print("\n=== Testing Session with Custom Model and Tools ===")

    ##############################################################################
    # From: src/apple_fm_sdk/session.py
    # class, function, or other entity name: LanguageModelSession - Session with custom model and tools
    import apple_fm_sdk as fm

    session = fm.LanguageModelSession(
        instructions="You are a helpful assistant with access to math tools.",
        tools=[SimpleCalculatorTool()],
    )

    response = await session.respond("What is 5 + 3?")
    ##############################################################################

    assert response is not None
    print("✅ Session with custom model and tools - PASSED")


@pytest.mark.asyncio
async def test_stream_response_basic(model):
    """Test from: src/apple_fm_sdk/session.py - stream_response() basic"""
    print("\n=== Testing Basic Streaming ===")

    ##############################################################################
    # From: src/apple_fm_sdk/session.py
    # class, function, or other entity name: stream_response - Basic streaming
    import apple_fm_sdk as fm

    session = fm.LanguageModelSession()

    async for chunk in session.stream_response("Tell me a story"):
        print(chunk, end="", flush=True)
    ##############################################################################

    print("✅ Basic streaming - PASSED")


@pytest.mark.asyncio
async def test_stream_response_cancellation():
    """Test from: src/apple_fm_sdk/session.py - stream_response() cancelling a stream"""
    print("\n=== Testing Stream Cancellation ===")

    ##############################################################################
    # From: src/apple_fm_sdk/session.py
    # class, function, or other entity name: stream_response - Cancelling a stream
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
    ##############################################################################

    # Expect the task to be cancelled
    try:
        await task
    except asyncio.CancelledError:
        print("✅ Stream cancellation - PASSED")
    else:
        raise AssertionError("Expected CancelledError was not raised")


@pytest.mark.asyncio
async def test_stream_response_error_handling(model):
    """Test from: src/apple_fm_sdk/session.py - stream_response() with error handling"""
    print("\n=== Testing Stream Error Handling ===")

    ##############################################################################
    # From: src/apple_fm_sdk/session.py
    # class, function, or other entity name: stream_response - Streaming with error handling
    import apple_fm_sdk as fm

    session = fm.LanguageModelSession()

    try:
        async for chunk in session.stream_response("Hello"):
            print(chunk)
    except fm.FoundationModelsError as e:
        print(f"Streaming error: {e}")
    ##############################################################################

    print("✅ Stream error handling - PASSED")


@pytest.mark.asyncio
async def test_from_transcript_resume_conversation(model):
    """Test from: src/apple_fm_sdk/session.py - from_transcript() - Resume a conversation from a saved transcript"""
    print("\n=== Testing from_transcript Resume Conversation ===")

    # First create and save a transcript
    import apple_fm_sdk as fm
    import json

    session = fm.LanguageModelSession()
    await session.respond("Hello")
    await session.respond("Tell me about Python")
    transcript_dict = await session.transcript.to_dict()
    with open("transcript.json", "w") as f:
        json.dump(transcript_dict, f, indent=2)

    ##############################################################################
    # From: src/apple_fm_sdk/session.py
    # class, function, or other entity name: from_transcript - Resume a conversation from a saved transcript
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
    ##############################################################################

    assert response is not None

    # Clean up the file
    import os

    if os.path.exists("transcript.json"):
        os.remove("transcript.json")
    print("✅ from_transcript resume conversation - PASSED")


@pytest.mark.asyncio
async def test_from_transcript_resume_with_tools(model):
    """Test from: src/apple_fm_sdk/session.py - from_transcript() - Resume with tools"""
    print("\n=== Testing from_transcript Resume with Tools ===")

    # First create and save a transcript with tool calls
    import apple_fm_sdk as fm
    import json
    from tester_tools.tester_tools import SimpleCalculatorTool

    session = fm.LanguageModelSession(tools=[SimpleCalculatorTool()])
    await session.respond("What is 5 + 3?")
    transcript_dict = await session.transcript.to_dict()
    with open("transcript_with_tools.json", "w") as f:
        json.dump(transcript_dict, f, indent=2)

    ##############################################################################
    # From: src/apple_fm_sdk/session.py
    # class, function, or other entity name: from_transcript - Resume with tools
    import apple_fm_sdk as fm
    from tester_tools.tester_tools import SimpleCalculatorTool

    # Load transcript that had tool calls
    transcript = await fm.Transcript.from_dict(transcript_dict)

    # IMPORTANT: You must pass the tool instances explicitly.
    # The transcript contains the history of tool calls, but not
    # the ability to make new tool calls unless you provide them.
    session = fm.LanguageModelSession.from_transcript(
        transcript, tools=[SimpleCalculatorTool()]
    )

    # Now the model can make new tool calls
    response = await session.respond("Calculate 15 * 24")
    ##############################################################################

    assert response is not None

    # Clean up the file
    import os

    if os.path.exists("transcript_with_tools.json"):
        os.remove("transcript_with_tools.json")
    print("✅ from_transcript resume with tools - PASSED")
