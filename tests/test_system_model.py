# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.

"""
Tests for SystemLanguageModel functionality.
"""

import pytest
import apple_fm_sdk as fm


def test_import_systemlanguagemodel():
    """Test that we can import SystemLanguageModel and related classes."""
    print("\n=== Testing SystemLanguageModel imports ===")

    import apple_fm_sdk  # noqa: F401 expected unused import

    print("✓ Successfully imported apple_fm_sdk")

    from apple_fm_sdk import (
        SystemLanguageModel,  # noqa: F401 expected unused import
        SystemLanguageModelUseCase,  # noqa: F401 expected unused import
        SystemLanguageModelGuardrails,  # noqa: F401 expected unused import
        SystemLanguageModelUnavailableReason,  # noqa: F401 expected unused import
        SystemLanguageModelVariant,  # noqa: F401 expected unused import
    )

    print("✓ Successfully imported SystemLanguageModel classes and enums")


def test_model_availability(model):
    """Test model availability and initializers."""
    print("\n=== Testing model availability and initializers ===")

    print("✓ Model fixture provided and available")

    print("Creating language model session...")
    session = fm.LanguageModelSession(model=model)  # noqa: F841 expected unused session
    print("✓ Successfully created LanguageModelSession")


def test_model_option_enums():
    """Test that enums for model options work correctly."""
    print("\n=== Testing enums for model options ===")

    # Test enum values
    assert fm.SystemLanguageModelUseCase.GENERAL == 0
    assert fm.SystemLanguageModelUseCase.CONTENT_TAGGING == 1
    print("✓ SystemLanguageModelUseCase enum works")

    assert fm.SystemLanguageModelGuardrails.DEFAULT == 0
    assert fm.SystemLanguageModelGuardrails.PERMISSIVE_CONTENT_TRANSFORMATIONS == 1
    print("✓ SystemLanguageModelGuardrails enum works")

    assert fm.SystemLanguageModelUnavailableReason.APPLE_INTELLIGENCE_NOT_ENABLED == 0
    assert fm.SystemLanguageModelUnavailableReason.DEVICE_NOT_ELIGIBLE == 1
    assert fm.SystemLanguageModelUnavailableReason.MODEL_NOT_READY == 2
    assert fm.SystemLanguageModelUnavailableReason.UNKNOWN == 0xFF
    print("✓ SystemLanguageModelUnavailableReason enum works")


@pytest.mark.asyncio
async def test_custom_model():
    """Test custom model creation."""
    print("\n=== Testing Custom Model Usage ===")

    # Create a custom model for content tagging
    custom_model = fm.SystemLanguageModel(
        use_case=fm.SystemLanguageModelUseCase.CONTENT_TAGGING,
        guardrails=fm.SystemLanguageModelGuardrails.PERMISSIVE_CONTENT_TRANSFORMATIONS,
    )

    # Check if the custom model is available
    is_available, reason = custom_model.is_available()
    if not is_available:
        print(f"Custom model not available: {reason}")
        pytest.skip(f"Custom model not available: {reason}")

    print("✓ Successfully created custom model")

    # Create a session with the custom model
    session = fm.LanguageModelSession(
        instructions="You are a content tagging system.", model=custom_model
    )

    print("✓ Successfully created session with custom model")

    # Test a simple response
    response = await session.respond(
        "Tag this content: A beautiful sunset over the mountains"
    )
    print(f"✓ Custom model response: {response[:100]}...")

    assert len(response) > 0, "Response should not be empty"


@pytest.mark.asyncio
async def test_invalid_use_case():
    """Test that invalid use case raises an error."""
    print("\n=== Testing Invalid Use Case Handling ===")

    class InvalidUseCase:
        value = 999  # Invalid value

    # Should produce a warning but default to GENERAL use case
    custom_model = fm.SystemLanguageModel(
        use_case=InvalidUseCase,  # Invalid use case # type: ignore
        guardrails=fm.SystemLanguageModelGuardrails.DEFAULT,
    )

    assert custom_model.is_available()[0], (
        "Model should be available despite invalid use case"
    )


def test_variant():
    """Test that the model reports its variant display name."""
    variant = fm.SystemLanguageModel().variant
    if variant is None:
        pytest.skip("Model variant requires macOS 27 and a macOS 27 SDK build")
    print(f"Model variant: {variant}")
    assert isinstance(variant, fm.SystemLanguageModelVariant)
    assert isinstance(variant.display_name, str)
