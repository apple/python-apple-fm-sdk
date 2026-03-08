# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.

"""
Tests for LanguageModelSession functionality.
"""

import pytest
import json
import asyncio
import apple_fm_sdk as fm
from tester_tools.tester_tools import SearchBreadDatabaseTool


def test_import_session():
    """Test that we can import LanguageModelSession and related classes."""
    print("\n=== Testing LanguageModelSession imports ===")

    import apple_fm_sdk  # noqa: F401 expected unused import

    print("✓ Successfully imported apple_fm_sdk")

    from apple_fm_sdk import (
        LanguageModelSession,  # noqa: F401 expected unused import
        FoundationModelsError,  # noqa: F401 expected unused import
        GenerationError,  # noqa: F401 expected unused import
        ExceededContextWindowSizeError,  # noqa: F401 expected unused import
        GuardrailViolationError,  # noqa: F401 expected unused import
    )

    print("✓ Successfully imported LanguageModelSession and error classes")


def test_is_responding(model):
    """Test the is_responding property."""
    print("\n=== Testing is_responding property ===")

    import apple_fm_sdk as fm

    # Create a session
    session = fm.LanguageModelSession("You are a helpful assistant.", model=model)

    # Initially should not be responding
    initial_responding = session.is_responding
    print(f"✓ Initial is_responding: {initial_responding}")

    # Property should be accessible (boolean type)
    assert isinstance(initial_responding, bool), (
        f"is_responding should be bool, got {type(initial_responding)}"
    )
    print("✓ is_responding property returns boolean")


def test_session_initialization_options(model):
    """Test initializing LanguageModelSession with different options."""
    print("\n=== Testing LanguageModelSession initialization options ===")

    import apple_fm_sdk as fm

    # Test 1: Session with no parameters (all defaults)
    print("\n1. Testing session with no parameters...")
    session1 = fm.LanguageModelSession()
    assert session1 is not None
    print("✓ Created session with no parameters")

    # Test 2: Session with instructions only
    print("\n2. Testing session with instructions only...")
    instructions = "You are a helpful assistant that provides concise answers."
    session2 = fm.LanguageModelSession(instructions=instructions)
    assert session2 is not None
    print("✓ Created session with instructions")

    # Test 3: Session with custom model only
    print("\n3. Testing session with custom model...")
    session3 = fm.LanguageModelSession(model=model)
    assert session3 is not None
    print("✓ Created session with custom model")

    # Test 4: Session with instructions and model
    print("\n4. Testing session with instructions and model...")
    session4 = fm.LanguageModelSession(
        instructions="You are a creative writer.", model=model
    )
    assert session4 is not None
    print("✓ Created session with instructions and model")

    # Test 5: Session with different model use cases
    print("\n5. Testing session with different model use cases...")

    # General use case
    model_general = fm.SystemLanguageModel(
        use_case=fm.SystemLanguageModelUseCase.GENERAL
    )
    session5a = fm.LanguageModelSession(model=model_general)
    assert session5a is not None
    print("✓ Created session with GENERAL use case model")

    # Content tagging use case
    model_tagging = fm.SystemLanguageModel(
        use_case=fm.SystemLanguageModelUseCase.CONTENT_TAGGING
    )
    session5b = fm.LanguageModelSession(model=model_tagging)
    assert session5b is not None
    print("✓ Created session with CONTENT_TAGGING use case model")

    # Test 6: Session with different guardrails
    print("\n6. Testing session with different guardrails...")

    # Default guardrails
    model_default = fm.SystemLanguageModel(
        guardrails=fm.SystemLanguageModelGuardrails.DEFAULT
    )
    session6a = fm.LanguageModelSession(model=model_default)
    assert session6a is not None
    print("✓ Created session with DEFAULT guardrails")

    # Permissive content transformations
    model_permissive_ct = fm.SystemLanguageModel(
        guardrails=fm.SystemLanguageModelGuardrails.PERMISSIVE_CONTENT_TRANSFORMATIONS
    )
    session6b = fm.LanguageModelSession(model=model_permissive_ct)
    assert session6b is not None
    print("✓ Created session with PERMISSIVE_CONTENT_TRANSFORMATIONS guardrails")

    # Test 8: Session with empty instructions
    print("\n8. Testing session with empty instructions...")
    session8 = fm.LanguageModelSession(instructions="", model=model)
    assert session8 is not None
    print("✓ Created session with empty instructions")

    # Test 9: Session with None instructions (explicit)
    print("\n9. Testing session with None instructions...")
    session9 = fm.LanguageModelSession(instructions=None, model=model)
    assert session9 is not None
    print("✓ Created session with None instructions")

    # Test 10: Session with tools (empty list)
    print("\n10. Testing session with empty tools list...")
    session10 = fm.LanguageModelSession(
        instructions="You are a helpful assistant.", model=model, tools=[]
    )
    assert session10 is not None
    print("✓ Created session with empty tools list")

    # Test 11: Session with None tools (explicit)
    print("\n11. Testing session with None tools...")
    session11 = fm.LanguageModelSession(
        instructions="You are a helpful assistant.", model=model, tools=None
    )
    assert session11 is not None
    print("✓ Created session with None tools")

    print("\n✓ All session initialization tests passed!")


@pytest.mark.asyncio
async def test_generation_options_with_respond(model):
    """Test using GenerationOptions with respond() method."""
    print("\n=== Testing GenerationOptions with respond() ===")

    import apple_fm_sdk as fm

    # Create a session
    session = fm.LanguageModelSession(
        instructions="You are a helpful assistant.", model=model
    )

    # Test 1: Basic response without options
    print("\n1. Testing respond without options...")
    response1 = await session.respond("Say hello")
    assert response1 is not None
    assert isinstance(response1, str)
    print(f"✓ Response without options: {response1[:50]}...")

    # Test 2: Response with temperature only
    print("\n2. Testing respond with temperature...")
    options2 = fm.GenerationOptions(temperature=0.7)
    response2 = await session.respond("Say hello", options=options2)
    assert response2 is not None
    assert isinstance(response2, str)
    print(f"✓ Response with temperature: {response2[:50]}...")

    # Test 3: Response with greedy sampling
    print("\n3. Testing respond with greedy sampling...")
    options3 = fm.GenerationOptions(sampling=fm.SamplingMode.greedy())
    response3 = await session.respond("Say hello", options=options3)
    assert response3 is not None
    assert isinstance(response3, str)
    print(f"✓ Response with greedy sampling: {response3[:50]}...")

    # Test 4: Response with random sampling (top-k)
    print("\n4. Testing respond with random sampling (top-k)...")
    options4 = fm.GenerationOptions(sampling=fm.SamplingMode.random(top=50, seed=42))
    response4 = await session.respond("Say hello", options=options4)
    assert response4 is not None
    assert isinstance(response4, str)
    print(f"✓ Response with random sampling (top-k): {response4[:50]}...")

    # Test 5: Response with random sampling (probability threshold)
    print("\n5. Testing respond with random sampling (probability threshold)...")
    options5 = fm.GenerationOptions(
        sampling=fm.SamplingMode.random(probability_threshold=0.9, seed=42)
    )
    response5 = await session.respond("Say hello", options=options5)
    assert response5 is not None
    assert isinstance(response5, str)
    print(
        f"✓ Response with random sampling (probability threshold): {response5[:50]}..."
    )

    # Test 6: Response with maximum_response_tokens
    print("\n6. Testing respond with maximum_response_tokens...")
    options6 = fm.GenerationOptions(maximum_response_tokens=2)
    response6 = await session.respond("Tell me a story", options=options6)
    assert response6 is not None
    assert (
        len(response6.split())
        <= 4  # 4 words max. This is a simple check, actual tokenization may differ
    )
    assert isinstance(response6, str)
    print(f"✓ Response with max tokens: {response6[:50]}")

    # Test 7: Response with all options combined
    print("\n7. Testing respond with all options combined...")
    options7 = fm.GenerationOptions(
        temperature=0.8,
        sampling=fm.SamplingMode.random(top=50, seed=42),
        maximum_response_tokens=100,
    )
    response7 = await session.respond("Write a short poem", options=options7)
    assert response7 is not None
    assert isinstance(response7, str)
    print(f"✓ Response with all options: {response7[:50]}...")

    print("\n✓ All respond() with options tests passed!")


def test_session_from_transcript_basic(model):
    """Test creating a session from a basic transcript."""
    print("\n=== Testing LanguageModelSession.from_transcript (basic) ===")

    # Load the basic test transcript
    with open("tests/tester_schemas/test_transcript.json", "r") as f:
        transcript_dict = json.load(f)

    print("\n1. Loading transcript from dictionary...")
    transcript = asyncio.run(fm.Transcript.from_dict(transcript_dict))
    assert transcript is not None
    print("✓ Created Transcript from dictionary")

    print("\n2. Creating session from transcript...")
    session = fm.LanguageModelSession.from_transcript(transcript, model=model)
    assert session is not None
    print("✓ Created LanguageModelSession from transcript")

    print("\n3. Verifying session has the transcript...")
    assert session.transcript is not None
    assert session.transcript is transcript
    print("✓ Session has the correct transcript")

    print("\n4. Verifying transcript contents...")
    transcript_data = asyncio.run(session.transcript.to_dict())
    assert "transcript" in transcript_data
    assert "entries" in transcript_data["transcript"]
    entries = transcript_data["transcript"]["entries"]
    assert len(entries) > 0
    print(f"✓ Transcript has {len(entries)} entries")

    # Verify the transcript contains expected content
    has_instructions = any(e["role"] == "instructions" for e in entries)
    has_user = any(e["role"] == "user" for e in entries)
    has_response = any(e["role"] == "response" for e in entries)
    assert has_instructions, "Transcript should have instructions entry"
    assert has_user, "Transcript should have user entries"
    assert has_response, "Transcript should have response entries"
    print("✓ Transcript contains expected entry types")

    print("\n✓ Basic from_transcript test passed!")


def test_session_from_transcript_full(model):
    """Test creating a session from a full transcript with tools and structured output."""
    print("\n=== Testing LanguageModelSession.from_transcript (full) ===")

    # Load the full test transcript
    with open("tests/tester_schemas/test_transcript_full.json", "r") as f:
        transcript_dict = json.load(f)

    print("\n1. Loading transcript from dictionary...")
    transcript = asyncio.run(fm.Transcript.from_dict(transcript_dict))
    assert transcript is not None
    print("✓ Created Transcript from dictionary")

    print("\n2. Creating session from transcript...")
    session = fm.LanguageModelSession.from_transcript(transcript, model=model)
    assert session is not None
    print("✓ Created LanguageModelSession from transcript")

    print("\n3. Verifying session has the transcript...")
    assert session.transcript is not None
    assert session.transcript is transcript
    print("✓ Session has the correct transcript")

    print("\n4. Verifying transcript contents...")
    transcript_data = asyncio.run(session.transcript.to_dict())
    assert "transcript" in transcript_data
    assert "entries" in transcript_data["transcript"]
    entries = transcript_data["transcript"]["entries"]
    assert len(entries) > 0
    print(f"✓ Transcript has {len(entries)} entries")

    # Verify the transcript contains expected content for full transcript
    has_instructions = any(e["role"] == "instructions" for e in entries)
    has_user = any(e["role"] == "user" for e in entries)
    has_response = any(e["role"] == "response" for e in entries)
    has_tool = any(e["role"] == "tool" for e in entries)
    assert has_instructions, "Transcript should have instructions entry"
    assert has_user, "Transcript should have user entries"
    assert has_response, "Transcript should have response entries"
    assert has_tool, "Full transcript should have tool entries"
    print("✓ Transcript contains expected entry types including tools")

    # Verify tool calls are present
    has_tool_calls = any("toolCalls" in e for e in entries if e["role"] == "response")
    assert has_tool_calls, "Transcript should have tool calls"
    print("✓ Transcript contains tool calls")

    # Verify structured output is present
    has_structured = any(
        "contents" in e and any(c.get("type") == "structure" for c in e["contents"])
        for e in entries
        if e["role"] == "response"
    )
    assert has_structured, "Transcript should have structured output"
    print("✓ Transcript contains structured output")

    print("\n✓ Full from_transcript test passed!")


def test_session_from_transcript_with_tools(model):
    """Test creating a session from transcript with custom tools."""
    print("\n=== Testing LanguageModelSession.from_transcript with tools ===")

    # Load the full test transcript (which has tools)
    with open("tests/tester_schemas/test_transcript_full.json", "r") as f:
        transcript_dict = json.load(f)

    print("\n1. Loading transcript from dictionary...")
    transcript = asyncio.run(fm.Transcript.from_dict(transcript_dict))
    assert transcript is not None
    print("✓ Created Transcript from dictionary")

    print("\n2. Creating session from transcript with custom tools...")
    tools: list[fm.Tool] = [SearchBreadDatabaseTool()]
    session = fm.LanguageModelSession.from_transcript(
        transcript, model=model, tools=tools
    )
    assert session is not None
    print("✓ Created LanguageModelSession from transcript with tools")

    print("\n3. Verifying session has the transcript...")
    assert session.transcript is not None
    print("✓ Session has the transcript")

    print("\n✓ from_transcript with tools test passed!")


def test_session_from_transcript_no_model(model):
    """Test creating a session from transcript without specifying a model."""
    print("\n=== Testing LanguageModelSession.from_transcript without model ===")

    # Load the basic test transcript
    with open("tests/tester_schemas/test_transcript.json", "r") as f:
        transcript_dict = json.load(f)

    print("\n1. Loading transcript from dictionary...")
    transcript = asyncio.run(fm.Transcript.from_dict(transcript_dict))
    assert transcript is not None
    print("✓ Created Transcript from dictionary")

    print("\n2. Creating session from transcript without model...")
    session = fm.LanguageModelSession.from_transcript(transcript)
    assert session is not None
    print("✓ Created LanguageModelSession from transcript (using default model)")

    print("\n3. Verifying session has the transcript...")
    assert session.transcript is not None
    print("✓ Session has the transcript")

    print("\n✓ from_transcript without model test passed!")


def test_session_from_transcript_continue_conversation(model):
    """Test that a session created from transcript can continue the conversation."""
    print(
        "\n=== Testing LanguageModelSession.from_transcript can continue conversation ==="
    )

    # Load the basic test transcript
    with open("tests/tester_schemas/test_transcript.json", "r") as f:
        transcript_dict = json.load(f)

    print("\n1. Loading transcript from dictionary...")
    transcript = asyncio.run(fm.Transcript.from_dict(transcript_dict))
    assert transcript is not None
    print("✓ Created Transcript from dictionary")

    print("\n2. Creating session from transcript...")
    session = fm.LanguageModelSession.from_transcript(transcript, model=model)
    assert session is not None
    print("✓ Created LanguageModelSession from transcript")

    print("\n3. Getting initial transcript entry count...")
    initial_transcript = asyncio.run(session.transcript.to_dict())
    initial_entry_count = len(initial_transcript["transcript"]["entries"])
    print(f"✓ Initial transcript has {initial_entry_count} entries")

    print("\n4. Making a new request to continue the conversation...")
    response = asyncio.run(session.respond("What is 7+8?"))
    assert response is not None
    assert isinstance(response, str)
    assert len(response) > 0
    print(f"✓ Got response: {response[:50]}...")

    print("\n5. Verifying transcript was updated...")
    updated_transcript = asyncio.run(session.transcript.to_dict())
    updated_entry_count = len(updated_transcript["transcript"]["entries"])
    assert updated_entry_count > initial_entry_count
    print(
        f"✓ Transcript updated from {initial_entry_count} to {updated_entry_count} entries"
    )

    print("\n✓ Continue conversation test passed!")


def test_session_from_transcript_empty_tools(model):
    """Test creating a session from transcript with empty tools list."""
    print("\n=== Testing LanguageModelSession.from_transcript with empty tools ===")

    # Load the basic test transcript
    with open("tests/tester_schemas/test_transcript.json", "r") as f:
        transcript_dict = json.load(f)

    print("\n1. Loading transcript from dictionary...")
    transcript = asyncio.run(fm.Transcript.from_dict(transcript_dict))
    assert transcript is not None
    print("✓ Created Transcript from dictionary")

    print("\n2. Creating session from transcript with empty tools list...")
    session = fm.LanguageModelSession.from_transcript(transcript, model=model, tools=[])
    assert session is not None
    print("✓ Created LanguageModelSession from transcript with empty tools")

    print("\n3. Verifying session has the transcript...")
    assert session.transcript is not None
    print("✓ Session has the transcript")

    print("\n✓ from_transcript with empty tools test passed!")
