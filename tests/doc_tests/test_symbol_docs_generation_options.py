# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.

"""
Test all code snippets from generation_options.py source documentation

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
# SAMPLING MODE TESTS (from src/apple_fm_sdk/generation_options.py)
# =============================================================================


@pytest.mark.asyncio
async def test_sampling_mode_greedy(model):
    """Test from: src/apple_fm_sdk/generation_options.py - SamplingMode.greedy"""
    print("\n=== Testing SamplingMode.greedy ===")

    ##############################################################################
    # From: src/apple_fm_sdk/generation_options.py
    # class, function, or other entity name: SamplingMode.greedy
    import apple_fm_sdk as fm

    sampling = fm.SamplingMode.greedy()
    options = fm.GenerationOptions(sampling=sampling)
    ##############################################################################

    assert sampling is not None
    assert options is not None
    print("✅ SamplingMode.greedy - PASSED")


@pytest.mark.asyncio
async def test_sampling_mode_random_top_k(model):
    """Test from: src/apple_fm_sdk/generation_options.py - SamplingMode.random - top-k"""
    print("\n=== Testing SamplingMode.random with top-k ===")

    ##############################################################################
    # From: src/apple_fm_sdk/generation_options.py
    # class, function, or other entity name: SamplingMode.random - Random sampling with top-k
    import apple_fm_sdk as fm

    # Consider only top 50 most likely tokens
    sampling = fm.SamplingMode.random(top=50, seed=42)
    options = fm.GenerationOptions(sampling=sampling)
    ##############################################################################

    assert sampling is not None
    assert options is not None
    print("✅ SamplingMode.random with top-k - PASSED")


@pytest.mark.asyncio
async def test_sampling_mode_random_probability_threshold(model):
    """Test from: src/apple_fm_sdk/generation_options.py - SamplingMode.random - probability threshold"""
    print("\n=== Testing SamplingMode.random with probability threshold ===")

    ##############################################################################
    # From: src/apple_fm_sdk/generation_options.py
    # class, function, or other entity name: SamplingMode.random - Random sampling with probability threshold
    import apple_fm_sdk as fm

    # Consider tokens until 90% cumulative probability
    sampling = fm.SamplingMode.random(probability_threshold=0.9, seed=42)
    options = fm.GenerationOptions(sampling=sampling)
    ##############################################################################

    assert sampling is not None
    assert options is not None
    print("✅ SamplingMode.random with probability threshold - PASSED")


@pytest.mark.asyncio
async def test_sampling_mode_random_seed_only(model):
    """Test from: src/apple_fm_sdk/generation_options.py - SamplingMode.random - seed only"""
    print("\n=== Testing SamplingMode.random with seed only ===")

    ##############################################################################
    # From: src/apple_fm_sdk/generation_options.py
    # class, function, or other entity name: SamplingMode.random - Random sampling with seed only
    import apple_fm_sdk as fm

    # Reproducible random sampling without constraints
    sampling = fm.SamplingMode.random(seed=42)
    options = fm.GenerationOptions(sampling=sampling)
    ##############################################################################

    assert sampling is not None
    assert options is not None
    print("✅ SamplingMode.random with seed only - PASSED")


# =============================================================================
# GENERATION OPTIONS TESTS (from src/apple_fm_sdk/generation_options.py)
# =============================================================================


@pytest.mark.asyncio
async def test_generation_options_default(model):
    """Test from: src/apple_fm_sdk/generation_options.py - GenerationOptions - Default options"""
    print("\n=== Testing GenerationOptions default ===")

    ##############################################################################
    # From: src/apple_fm_sdk/generation_options.py
    # class, function, or other entity name: GenerationOptions - Default options
    import apple_fm_sdk as fm

    options = fm.GenerationOptions()
    ##############################################################################

    assert options is not None
    print("✅ GenerationOptions default - PASSED")


@pytest.mark.asyncio
async def test_generation_options_temperature_and_tokens(model):
    """Test from: src/apple_fm_sdk/generation_options.py - GenerationOptions - Custom temperature and token limit"""
    print("\n=== Testing GenerationOptions with temperature and token limit ===")

    ##############################################################################
    # From: src/apple_fm_sdk/generation_options.py
    # class, function, or other entity name: GenerationOptions - Custom temperature and token limit
    import apple_fm_sdk as fm

    options = fm.GenerationOptions(temperature=0.7, maximum_response_tokens=500)
    ##############################################################################

    assert options is not None
    assert options.temperature == 0.7
    assert options.maximum_response_tokens == 500
    print("✅ GenerationOptions with temperature and token limit - PASSED")


@pytest.mark.asyncio
async def test_generation_options_greedy_with_temperature(model):
    """Test from: src/apple_fm_sdk/generation_options.py - GenerationOptions - Greedy sampling with temperature"""
    print("\n=== Testing GenerationOptions with greedy sampling and temperature ===")

    ##############################################################################
    # From: src/apple_fm_sdk/generation_options.py
    # class, function, or other entity name: GenerationOptions - Greedy sampling with temperature
    import apple_fm_sdk as fm

    options = fm.GenerationOptions(sampling=fm.SamplingMode.greedy(), temperature=0.3)
    ##############################################################################

    assert options is not None
    assert options.temperature == 0.3
    print("✅ GenerationOptions with greedy sampling and temperature - PASSED")


@pytest.mark.asyncio
async def test_generation_options_random_with_constraints(model):
    """Test from: src/apple_fm_sdk/generation_options.py - GenerationOptions - Random sampling with constraints"""
    print("\n=== Testing GenerationOptions with random sampling and constraints ===")

    ##############################################################################
    # From: src/apple_fm_sdk/generation_options.py
    # class, function, or other entity name: GenerationOptions - Random sampling with constraints
    import apple_fm_sdk as fm

    options = fm.GenerationOptions(
        sampling=fm.SamplingMode.random(top=50, seed=42),
        temperature=0.8,
        maximum_response_tokens=1000,
    )
    ##############################################################################

    assert options is not None
    assert options.temperature == 0.8
    assert options.maximum_response_tokens == 1000
    print("✅ GenerationOptions with random sampling and constraints - PASSED")


@pytest.mark.asyncio
async def test_generation_options_to_dict(model):
    """Test from: src/apple_fm_sdk/generation_options.py - GenerationOptions.to_dict"""
    print("\n=== Testing GenerationOptions.to_dict ===")

    ##############################################################################
    # From: src/apple_fm_sdk/generation_options.py
    # class, function, or other entity name: GenerationOptions.to_dict
    import apple_fm_sdk as fm

    options = fm.GenerationOptions(
        temperature=0.7,
        sampling=fm.SamplingMode.random(top=50),
        maximum_response_tokens=500,
    )
    options_dict = options.to_dict()
    # {'temperature': 0.7, 'sampling': {'mode': 'random', 'top_k': 50}, 'maximum_response_tokens': 500}
    ##############################################################################

    assert options_dict is not None
    assert isinstance(options_dict, dict)
    assert "temperature" in options_dict
    assert options_dict["temperature"] == 0.7
    assert "sampling" in options_dict
    assert "maximum_response_tokens" in options_dict
    print("✅ GenerationOptions.to_dict - PASSED")


# =============================================================================
# INTEGRATION TESTS WITH SESSION (from src/apple_fm_sdk/generation_options.py)
# =============================================================================


@pytest.mark.asyncio
async def test_generation_options_with_respond(model):
    """Test from: src/apple_fm_sdk/generation_options.py - Using options with respond"""
    print("\n=== Testing GenerationOptions with respond ===")

    ##############################################################################
    # From: src/apple_fm_sdk/generation_options.py
    # class, function, or other entity name: Usage with respond
    import apple_fm_sdk as fm

    session = fm.LanguageModelSession(model=model)

    # Control generation with custom options
    options = fm.GenerationOptions(
        temperature=0.7,
        sampling=fm.SamplingMode.random(top=50, seed=42),
        maximum_response_tokens=500,
    )

    response = await session.respond("Write a creative story", options=options)
    ##############################################################################

    assert response is not None
    assert isinstance(response, str)
    print("✅ GenerationOptions with respond - PASSED")


@pytest.mark.asyncio
async def test_generation_options_with_stream(model):
    """Test from: src/apple_fm_sdk/generation_options.py - Using options with stream_response"""
    print("\n=== Testing GenerationOptions with stream_response ===")

    ##############################################################################
    # From: src/apple_fm_sdk/generation_options.py
    # class, function, or other entity name: Usage with stream_response
    import apple_fm_sdk as fm

    session = fm.LanguageModelSession(model=model)

    options = fm.GenerationOptions(
        temperature=0.8,
        sampling=fm.SamplingMode.random(top=50),
        maximum_response_tokens=1000,
    )

    chunks = []
    async for chunk in session.stream_response("Tell me a story", options=options):
        chunks.append(chunk)
    ##############################################################################

    assert len(chunks) > 0
    print("✅ GenerationOptions with stream_response - PASSED")
