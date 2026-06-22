# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.

"""
Tests for SystemLanguageModel token counting and context size.
"""

import apple_fm_sdk as fm
import pytest

from tester_tools.tester_tools import SimpleCalculatorTool, GetUserInfoTool
import tester_schemas.schemas as tester_schemas


# MARK: - Context size


def test_context_size(model):
    """The model exposes a positive context window size."""
    context_size = model.context_size
    assert isinstance(context_size, int)
    assert context_size >= 1, "Context size should be positive"


# MARK: - Basic prompt token counting


@pytest.mark.asyncio
async def test_token_count_simple_prompt(model):
    """A short prompt produces a small, positive token count."""
    short = await model.token_count("Hello")
    longer = await model.token_count(
        "Hello, this is a considerably longer prompt with many more words in it."
    )
    assert short > 0
    assert longer > short, "A longer prompt should count more tokens"


@pytest.mark.asyncio
async def test_token_count_increases_with_length(model):
    """Longer prompts yield more tokens than shorter ones."""
    short = await model.token_count("Hi")
    long = await model.token_count(
        "This is a much longer prompt with many more words that should result "
        "in a higher token count"
    )
    assert long > short


@pytest.mark.asyncio
async def test_token_count_is_deterministic(model):
    """The same prompt always produces the same token count."""
    prompt = "This is a test prompt"
    first = await model.token_count(prompt)
    second = await model.token_count(prompt)
    assert first == second


@pytest.mark.asyncio
async def test_token_count_list_prompt(model):
    """A prompt expressed as a list of text components is supported."""
    count = await model.token_count(["First line of text", "Second line of text"])
    assert count > 0


@pytest.mark.asyncio
async def test_token_count_unicode(model):
    """Non-English text and emoji produce tokens."""
    assert await model.token_count("こんにちは世界") > 0
    assert await model.token_count("Hello 👋 world 🌍") > 0


# MARK: - Instructions


@pytest.mark.asyncio
async def test_token_count_instructions(model):
    """Instructions text produces a positive token count."""
    count = await model.token_count(instructions="You are a helpful assistant")
    assert count > 0


# MARK: - Tools


@pytest.mark.asyncio
async def test_token_count_single_tool(model):
    """A single tool contributes tokens."""
    count = await model.token_count([SimpleCalculatorTool()])
    assert count > 0


@pytest.mark.asyncio
async def test_token_count_more_tools_more_tokens(model):
    """More tools increase the token count."""
    single = await model.token_count([SimpleCalculatorTool()])
    multiple = await model.token_count([SimpleCalculatorTool(), GetUserInfoTool()])
    assert multiple > single


# MARK: - Schema


@pytest.mark.asyncio
async def test_token_count_schema(model):
    """A generation schema produces a positive token count."""
    count = await model.token_count(tester_schemas.Cat.generation_schema())
    print(count)
    assert count > 0


# MARK: - Transcript


@pytest.mark.asyncio
async def test_token_count_transcript(model):
    """A populated session transcript produces a positive token count."""
    session = fm.LanguageModelSession(model=model)
    await session.respond("Say hello briefly.")
    count = await model.token_count(session.transcript)
    assert count > 0


# MARK: - Argument validation


@pytest.mark.asyncio
async def test_token_count_requires_an_argument(model):
    """Calling without a value or instructions raises ValueError."""
    with pytest.raises(ValueError):
        await model.token_count()


@pytest.mark.asyncio
async def test_token_count_rejects_value_and_instructions(model):
    """Passing both a value and instructions raises ValueError."""
    with pytest.raises(ValueError):
        await model.token_count("Hello", instructions="You are an assistant")
