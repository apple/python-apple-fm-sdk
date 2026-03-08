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


# =============================================================================
# TRANSCRIPT TESTS (from src/apple_fm_sdk/transcript.py)
# =============================================================================


@pytest.mark.asyncio
async def test_transcript_accessing_after_session(model):
    """Test from: src/apple_fm_sdk/transcript.py - Transcript class docstring - Accessing transcript after session"""
    print("\n=== Testing Accessing Transcript After Session ===")

    ##############################################################################
    # From: src/apple_fm_sdk/transcript.py
    # class, function, or other entity name: Transcript - Accessing transcript after session
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
    ##############################################################################

    assert transcript is not None
    assert "transcript" in transcript
    print("✅ Accessing transcript after session - PASSED")


@pytest.mark.asyncio
async def test_transcript_monitoring_session_length(model):
    """Test from: src/apple_fm_sdk/transcript.py - Transcript class docstring - Monitoring session length"""
    print("\n=== Testing Monitoring Session Length ===")
    transcript = None

    ##############################################################################
    # From: src/apple_fm_sdk/transcript.py
    # class, function, or other entity name: Transcript - Monitoring session length
    import apple_fm_sdk as fm

    session = fm.LanguageModelSession()

    for i in range(5):
        await session.respond(f"Question {i}")

        transcript = await session.transcript.to_dict()
        entry_count = len(transcript["transcript"]["entries"])
        print(f"Session has {entry_count} entries")
    ##############################################################################

    assert transcript is not None
    print("✅ Monitoring session length - PASSED")


@pytest.mark.asyncio
async def test_transcript_examining_tool_calls(model):
    """Test from: src/apple_fm_sdk/transcript.py - Transcript class docstring - Examining tool calls in transcript"""
    print("\n=== Testing Examining Tool Calls ===")

    ##############################################################################
    # From: src/apple_fm_sdk/transcript.py
    # class, function, or other entity name: Transcript - Examining tool calls in transcript
    import apple_fm_sdk as fm
    from tester_tools.tester_tools import SimpleCalculatorTool

    session = fm.LanguageModelSession(tools=[SimpleCalculatorTool()])

    await session.respond("What is 15 * 24?")

    transcript = await session.transcript.to_dict()

    # Find entries with tool calls
    for entry in transcript["transcript"]["entries"]:
        if entry["role"] == "response" and "toolCalls" in entry:
            print(f"Tool calls: {entry['toolCalls']}")
        elif entry["role"] == "tool":
            print(f"Tool result for {entry['toolName']}")
    ##############################################################################

    assert transcript is not None
    print("✅ Examining tool calls - PASSED")


@pytest.mark.asyncio
async def test_transcript_saving_session_history(model):
    """Test from: src/apple_fm_sdk/transcript.py - Transcript class docstring - Saving session history"""
    print("\n=== Testing Saving Session History ===")

    ##############################################################################
    # From: src/apple_fm_sdk/transcript.py
    # class, function, or other entity name: Transcript - Saving session history
    import apple_fm_sdk as fm
    import json

    session = fm.LanguageModelSession()

    # Have a conversation
    await session.respond("Hello")
    await session.respond("Tell me about Python")

    # Save transcript to file
    transcript = await session.transcript.to_dict()
    with open("conversation.json", "w") as f:
        json.dump(transcript, f, indent=2)
    ##############################################################################

    assert transcript is not None
    # Clean up the file
    import os

    if os.path.exists("conversation.json"):
        os.remove("conversation.json")
    print("✅ Saving session history - PASSED")


@pytest.mark.asyncio
async def test_transcript_to_dict_method(model):
    """Test from: src/apple_fm_sdk/transcript.py - to_dict method docstring"""
    print("\n=== Testing to_dict Method ===")

    ##############################################################################
    # From: src/apple_fm_sdk/transcript.py
    # class, function, or other entity name: to_dict - Method example
    import apple_fm_sdk as fm

    session = fm.LanguageModelSession()
    await session.respond("Hello!")

    transcript = await session.transcript.to_dict()

    # Access entries
    entries = transcript["transcript"]["entries"]
    for entry in entries:
        print(f"{entry['role']}: {entry.get('id')}")
    ##############################################################################

    assert transcript is not None
    print("✅ to_dict method - PASSED")


@pytest.mark.asyncio
async def test_transcript_from_dict_load_from_file(model):
    """Test from: src/apple_fm_sdk/transcript.py - from_dict method - Load a transcript from a JSON file"""
    print("\n=== Testing from_dict Load from File ===")

    # First create and save a transcript
    import apple_fm_sdk as fm
    import json

    session = fm.LanguageModelSession()
    await session.respond("Hello")
    transcript_dict = await session.transcript.to_dict()
    with open("transcript.json", "w") as f:
        json.dump(transcript_dict, f, indent=2)

    ##############################################################################
    # From: src/apple_fm_sdk/transcript.py
    # class, function, or other entity name: from_dict - Load a transcript from a JSON file
    import apple_fm_sdk as fm
    import json

    # Load transcript from file
    with open("transcript.json", "r") as f:
        transcript_dict = json.load(f)

    # Create Transcript instance
    transcript = await fm.Transcript.from_dict(transcript_dict)

    # Now you can create a session starting from this transcript
    session = fm.LanguageModelSession.from_transcript(transcript)
    ##############################################################################

    assert transcript is not None
    assert session is not None

    # Clean up the file
    import os

    if os.path.exists("transcript.json"):
        os.remove("transcript.json")
    print("✅ from_dict load from file - PASSED")


@pytest.mark.asyncio
async def test_transcript_from_dict_load_and_resume_with_tools(model):
    """Test from: src/apple_fm_sdk/transcript.py - from_dict method - Load and resume with tools"""
    print("\n=== Testing from_dict Load and Resume with Tools ===")

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
    # From: src/apple_fm_sdk/transcript.py
    # class, function, or other entity name: from_dict - Load and resume with tools
    import apple_fm_sdk as fm
    import json
    from tester_tools.tester_tools import SimpleCalculatorTool

    # Load transcript that had tool calls
    with open("transcript_with_tools.json", "r") as f:
        transcript_dict = json.load(f)

    transcript = await fm.Transcript.from_dict(transcript_dict)

    # IMPORTANT: Tools in the transcript are historical mentions only.
    # To allow the model to call a tool, you must explicitly instantiate each
    # tool in Python and then pass them to the session initializer.
    session = fm.LanguageModelSession.from_transcript(
        transcript, tools=[SimpleCalculatorTool()]
    )
    ##############################################################################

    assert transcript is not None
    assert session is not None

    # Clean up the file
    import os

    if os.path.exists("transcript_with_tools.json"):
        os.remove("transcript_with_tools.json")
    print("✅ from_dict load and resume with tools - PASSED")
