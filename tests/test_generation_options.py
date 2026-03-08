# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.

"""
Test Foundation Models GenerationOptions and SamplingMode classes
"""

import apple_fm_sdk as fm
import pytest


def test_generation_options_default():
    """Test creating GenerationOptions with default values."""
    options = fm.GenerationOptions()

    assert options.sampling is None
    assert options.temperature is None
    assert options.maximum_response_tokens is None


def test_generation_options_with_temperature():
    """Test creating GenerationOptions with temperature."""
    options = fm.GenerationOptions(temperature=0.7)

    assert options.temperature == 0.7
    assert options.sampling is None
    assert options.maximum_response_tokens is None


def test_generation_options_with_max_tokens():
    """Test creating GenerationOptions with maximum_response_tokens."""
    options = fm.GenerationOptions(maximum_response_tokens=500)

    assert options.maximum_response_tokens == 500
    assert options.temperature is None
    assert options.sampling is None


def test_generation_options_full_config():
    """Test creating GenerationOptions with all parameters."""
    sampling = fm.SamplingMode.greedy()
    options = fm.GenerationOptions(
        sampling=sampling, temperature=0.8, maximum_response_tokens=1000
    )

    assert options.sampling == sampling
    assert options.temperature == 0.8
    assert options.maximum_response_tokens == 1000


def test_generation_options_negative_temperature():
    """Test that negative temperature raises ValueError."""
    with pytest.raises(ValueError, match="temperature.*non-negative"):
        fm.GenerationOptions(temperature=-0.5)


def test_generation_options_zero_max_tokens():
    """Test that zero maximum_response_tokens raises ValueError."""
    with pytest.raises(ValueError, match="maximum_response_tokens.*positive"):
        fm.GenerationOptions(maximum_response_tokens=0)


def test_generation_options_negative_max_tokens():
    """Test that negative maximum_response_tokens raises ValueError."""
    with pytest.raises(ValueError, match="maximum_response_tokens.*positive"):
        fm.GenerationOptions(maximum_response_tokens=-100)


def test_generation_options_invalid_temperature_type():
    """Test that non-numeric temperature raises ValueError."""
    with pytest.raises(ValueError, match="temperature.*number"):
        fm.GenerationOptions(temperature="high")  # type: ignore purposefully passing wrong type


def test_generation_options_invalid_max_tokens_type():
    """Test that non-integer maximum_response_tokens raises ValueError."""
    with pytest.raises(ValueError, match="maximum_response_tokens.*integer"):
        fm.GenerationOptions(maximum_response_tokens=500.5)  # type: ignore purposefully passing wrong type


def test_generation_options_invalid_sampling_type():
    """Test that invalid sampling type raises ValueError."""
    with pytest.raises(ValueError, match="sampling.*SamplingMode"):
        fm.GenerationOptions(sampling="greedy")  # type: ignore purposefully passing wrong type


def test_sampling_mode_greedy():
    """Test creating greedy SamplingMode."""
    sampling = fm.SamplingMode.greedy()

    assert sampling.mode_type == "greedy"
    assert sampling.top is None
    assert sampling.probability_threshold is None
    assert sampling.seed is None


def test_sampling_mode_random_no_constraints():
    """Test creating random SamplingMode without constraints."""
    sampling = fm.SamplingMode.random(seed=42)

    assert sampling.mode_type == "random"
    assert sampling.top is None
    assert sampling.probability_threshold is None
    assert sampling.seed == 42


def test_sampling_mode_random_with_top():
    """Test creating random SamplingMode with top-k constraint."""
    sampling = fm.SamplingMode.random(top=50, seed=42)

    assert sampling.mode_type == "random"
    assert sampling.top == 50
    assert sampling.probability_threshold is None
    assert sampling.seed == 42


def test_sampling_mode_random_with_probability_threshold():
    """Test creating random SamplingMode with probability threshold."""
    sampling = fm.SamplingMode.random(probability_threshold=0.9, seed=42)

    assert sampling.mode_type == "random"
    assert sampling.top is None
    assert sampling.probability_threshold == 0.9
    assert sampling.seed == 42


def test_sampling_mode_random_both_constraints_error():
    """Test that specifying both top and probability_threshold raises ValueError."""
    with pytest.raises(ValueError, match="Cannot specify both"):
        fm.SamplingMode.random(top=50, probability_threshold=0.9)


def test_sampling_mode_random_invalid_top():
    """Test that invalid top value raises ValueError."""
    with pytest.raises(ValueError, match="top.*positive integer"):
        fm.SamplingMode.random(top=0)

    with pytest.raises(ValueError, match="top.*positive integer"):
        fm.SamplingMode.random(top=-5)

    with pytest.raises(ValueError, match="top.*positive integer"):
        fm.SamplingMode.random(top=50.5)  # type: ignore purposefully passing wrong type


def test_sampling_mode_random_invalid_probability_threshold():
    """Test that invalid probability_threshold raises ValueError."""
    with pytest.raises(ValueError, match="probability_threshold.*between 0.0 and 1.0"):
        fm.SamplingMode.random(probability_threshold=-0.1)

    with pytest.raises(ValueError, match="probability_threshold.*between 0.0 and 1.0"):
        fm.SamplingMode.random(probability_threshold=1.5)


def test_sampling_mode_random_invalid_seed_type():
    """Test that invalid seed type raises ValueError."""
    with pytest.raises(ValueError, match="seed.*integer"):
        fm.SamplingMode.random(seed="random")  # type: ignore purposefully passing wrong type


def test_sampling_mode_random_edge_cases():
    """Test edge cases for random sampling parameters."""
    # Probability threshold at boundaries
    sampling1 = fm.SamplingMode.random(probability_threshold=0.0)
    assert sampling1.probability_threshold == 0.0

    sampling2 = fm.SamplingMode.random(probability_threshold=1.0)
    assert sampling2.probability_threshold == 1.0

    # Top with minimum valid value
    sampling3 = fm.SamplingMode.random(top=1)
    assert sampling3.top == 1


def test_generation_options_with_greedy_sampling():
    """Test GenerationOptions with greedy sampling."""
    sampling = fm.SamplingMode.greedy()
    options = fm.GenerationOptions(sampling=sampling, temperature=0.3)

    assert options.sampling is not None
    assert options.sampling.mode_type == "greedy"
    assert options.temperature == 0.3


def test_generation_options_with_random_sampling():
    """Test GenerationOptions with random sampling."""
    sampling = fm.SamplingMode.random(top=100, seed=123)
    options = fm.GenerationOptions(
        sampling=sampling, temperature=0.75, maximum_response_tokens=1000
    )

    assert options.sampling is not None
    assert options.sampling.mode_type == "random"
    assert options.sampling.top == 100
    assert options.sampling.seed == 123
    assert options.temperature == 0.75
    assert options.maximum_response_tokens == 1000


def test_generation_options_dataclass_behavior():
    """Test that GenerationOptions behaves as a dataclass."""
    options1 = fm.GenerationOptions(temperature=0.7)
    options2 = fm.GenerationOptions(temperature=0.7)

    # Dataclasses with same values should be equal
    assert options1 == options2

    # Different values should not be equal
    options3 = fm.GenerationOptions(temperature=0.8)
    assert options1 != options3


def test_sampling_mode_dataclass_behavior():
    """Test that SamplingMode behaves as a dataclass."""
    sampling1 = fm.SamplingMode.greedy()
    sampling2 = fm.SamplingMode.greedy()

    # Dataclasses with same values should be equal
    assert sampling1 == sampling2

    # Different sampling modes should not be equal
    sampling3 = fm.SamplingMode.random(seed=42)
    assert sampling1 != sampling3


def test_generation_options_repr():
    """Test that GenerationOptions has a useful string representation."""
    options = fm.GenerationOptions(temperature=0.7, maximum_response_tokens=500)

    repr_str = repr(options)
    assert "GenerationOptions" in repr_str
    assert "0.7" in repr_str
    assert "500" in repr_str


def test_sampling_mode_repr():
    """Test that SamplingMode has a useful string representation."""
    sampling = fm.SamplingMode.random(top=50, seed=42)

    repr_str = repr(sampling)
    assert "SamplingMode" in repr_str
    assert "random" in repr_str
    assert "50" in repr_str
    assert "42" in repr_str
