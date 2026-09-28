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
# CORE TESTS (from src/apple_fm_sdk/core.py)
# =============================================================================


@pytest.mark.asyncio
async def test_core_module_basic_usage(model):
    """Test from: src/apple_fm_sdk/core.py - Module docstring - Basic usage of SystemLanguageModel"""
    print("\n=== Testing Core Module Basic Usage ===")

    ##############################################################################
    # From: src/apple_fm_sdk/core.py
    # class, function, or other entity name: Module - Basic usage of SystemLanguageModel
    import apple_fm_sdk as fm

    model = fm.SystemLanguageModel()
    is_available, reason = model.is_available()
    if is_available:
        # Use the model
        pass
    ##############################################################################

    assert model is not None
    print("✅ Core module basic usage - PASSED")


@pytest.mark.asyncio
async def test_system_language_model_creating_and_checking(model):
    """Test from: src/apple_fm_sdk/core.py - SystemLanguageModel class docstring - Creating and checking availability of a model"""
    print("\n=== Testing SystemLanguageModel Creating and Checking ===")

    ##############################################################################
    # From: src/apple_fm_sdk/core.py
    # class, function, or other entity name: SystemLanguageModel - Creating and checking availability of a model
    import apple_fm_sdk as fm

    # Create a model with default settings
    model = fm.SystemLanguageModel()

    # Create a model with specific settings
    model = fm.SystemLanguageModel(
        use_case=fm.SystemLanguageModelUseCase.CONTENT_TAGGING,
        guardrails=fm.SystemLanguageModelGuardrails.PERMISSIVE_CONTENT_TRANSFORMATIONS,
    )

    # Check if the model is available
    is_available, reason = model.is_available()
    if not is_available:
        print(f"Model unavailable: {reason}")
    ##############################################################################

    assert model is not None
    print("✅ SystemLanguageModel creating and checking - PASSED")


@pytest.mark.asyncio
async def test_system_language_model_init_default(model):
    """Test from: src/apple_fm_sdk/core.py - SystemLanguageModel.__init__ - Create with default settings"""
    print("\n=== Testing SystemLanguageModel Init Default ===")

    ##############################################################################
    # From: src/apple_fm_sdk/core.py
    # class, function, or other entity name: __init__ - Create with default settings
    import apple_fm_sdk as fm

    # Create with default settings
    model = fm.SystemLanguageModel()

    # Create with custom settings
    model = fm.SystemLanguageModel(
        use_case=fm.SystemLanguageModelUseCase.CONTENT_TAGGING,
        guardrails=fm.SystemLanguageModelGuardrails.PERMISSIVE_CONTENT_TRANSFORMATIONS,
    )
    ##############################################################################

    assert model is not None
    print("✅ SystemLanguageModel init default - PASSED")


@pytest.mark.asyncio
async def test_system_language_model_is_available(model):
    """Test from: src/apple_fm_sdk/core.py - SystemLanguageModel.is_available"""
    print("\n=== Testing SystemLanguageModel is_available ===")

    ##############################################################################
    # From: src/apple_fm_sdk/core.py
    # class, function, or other entity name: is_available
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
    ##############################################################################

    assert isinstance(is_available, bool)
    print("✅ SystemLanguageModel is_available - PASSED")


def test_system_language_model_variant():
    """Test from: src/apple_fm_sdk/core.py - SystemLanguageModel.variant"""
    print("\n=== Testing SystemLanguageModel variant ===")

    ##############################################################################
    # From: src/apple_fm_sdk/core.py
    # class, function, or other entity name: variant
    import apple_fm_sdk as fm

    model = fm.SystemLanguageModel()
    if model.variant:
        print(f"Model variant: {model.variant.display_name}")
    ##############################################################################

    print("✅ SystemLanguageModel variant - PASSED")
