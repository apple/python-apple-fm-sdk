# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.

"""
Tests for streaming response functionality.
"""

import apple_fm_sdk as fm
import pytest


@pytest.mark.asyncio
async def test_streaming(model):
    """Test streaming response functionality."""
    print("\n=== Testing Streaming Response ===")

    # Get the default model
    session = fm.LanguageModelSession("You are a helpful assistant.", model=model)

    print("Starting streaming response...")
    chunks = []
    chunk_count = 0

    # Stream response chunks
    async for chunk in session.stream_response(
        "Tell me a very short story about a cat"
    ):
        chunks.append(chunk)
        chunk_count += 1
        if chunk_count <= 3:  # Show first 3 chunks
            print(f"✓ Received chunk {chunk_count}: {chunk[:50]}...")

    full_response = chunks[-1] if chunks else ""
    print(f"✓ Streaming completed with {chunk_count} chunks")
    print(f"✓ Final response length: {len(full_response)} characters")

    # Validate we got a reasonable response
    assert len(full_response) > 10, "Response too short"
    assert isinstance(full_response, str), (
        f"Expected string response, got {type(full_response)}"
    )
    print("Full response:", full_response)


@pytest.mark.asyncio
async def test_generation_options_with_stream(model):
    """Test using GenerationOptions with stream_response() method."""
    print("\n=== Testing GenerationOptions with stream_response() ===")

    # Create a session
    session = fm.LanguageModelSession(
        instructions="You are a helpful assistant.", model=model
    )

    # Test 1: Stream without options
    print("\n1. Testing stream without options...")
    chunks1 = []
    async for chunk in session.stream_response("Count to 3"):
        chunks1.append(chunk)
    assert len(chunks1) > 0
    final_response1 = chunks1[-1]
    print(f"✓ Streamed {len(chunks1)} chunks without options")
    print(f"  Final response: {final_response1[:50]}...")

    # Test 2: Stream with temperature
    print("\n2. Testing stream with temperature...")
    options2 = fm.GenerationOptions(temperature=0.5)
    chunks2 = []
    async for chunk in session.stream_response("Count to 3", options=options2):
        chunks2.append(chunk)
    assert len(chunks2) > 0
    final_response2 = chunks2[-1]
    print(f"✓ Streamed {len(chunks2)} chunks with temperature")
    print(f"  Final response: {final_response2[:50]}...")

    # Test 3: Stream with greedy sampling
    print("\n3. Testing stream with greedy sampling...")
    options3 = fm.GenerationOptions(sampling=fm.SamplingMode.greedy())
    chunks3 = []
    async for chunk in session.stream_response("Count to 3", options=options3):
        chunks3.append(chunk)
    assert len(chunks3) > 0
    final_response3 = chunks3[-1]
    print(f"✓ Streamed {len(chunks3)} chunks with greedy sampling")
    print(f"  Final response: {final_response3[:50]}...")

    # Test 4: Stream with random sampling (top-k)
    print("\n4. Testing stream with random sampling (top-k)...")
    options4 = fm.GenerationOptions(sampling=fm.SamplingMode.random(top=50, seed=42))
    chunks4 = []
    async for chunk in session.stream_response("Count to 3", options=options4):
        chunks4.append(chunk)
    assert len(chunks4) > 0
    final_response4 = chunks4[-1]
    print(f"✓ Streamed {len(chunks4)} chunks with random sampling (top-k)")
    print(f"  Final response: {final_response4[:50]}...")

    # Test 5: Stream with maximum_response_tokens
    print("\n5. Testing stream with maximum_response_tokens...")
    options5 = fm.GenerationOptions(maximum_response_tokens=5)
    chunks5 = []
    async for chunk in session.stream_response("Tell me a story", options=options5):
        chunks5.append(chunk)
    assert len(chunks5) > 0
    final_response5 = chunks5[-1]
    print(f"✓ Streamed {len(chunks5)} chunks with max tokens")
    print(
        f"  Final response: {final_response5[:50]} ... length: {len(final_response5)} characters, {len(final_response5.split())} words"
    )

    # Test 6: Stream with all options combined
    print("\n6. Testing stream with all options combined...")
    options6 = fm.GenerationOptions(
        temperature=0.7,
        sampling=fm.SamplingMode.random(top=40, seed=123),
        maximum_response_tokens=50,
    )
    chunks6 = []
    async for chunk in session.stream_response("Count to 5", options=options6):
        chunks6.append(chunk)
    assert len(chunks6) > 0
    final_response6 = chunks6[-1]
    print(f"✓ Streamed {len(chunks6)} chunks with all options")
    print(f"  Final response: {final_response6[:50]}...")

    print("\n✓ All stream_response() with options tests passed!")
