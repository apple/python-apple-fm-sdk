# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.

"""
Test all code snippets from the README.md

RULES:
Use this consistent testing format:
- Each test function should correspond to a specific code snippet or section in the documentation.
- Include comments indicating the source documentation file and section for clarity.
- No extra tests beyond those needed to validate the snippets.

Copy the snippet from the source **exactly** as it appears in the documentation.
Surround the original source with:
##############################################################################
# README Section: <section_name>
<actual code here uncommented>
##############################################################################

The test passes if the snippet runs without errors. No additional assertions are necessary
beyond ensuring the snippet executes successfully.
"""

import pytest


@pytest.mark.asyncio
async def test_installation_snippet(model):
    """Test the installation verification snippet"""
    print("\n=== Testing Installation Snippet ===")

    ##############################################################################
    # README Section: Quick Start - Basic Usage
    import apple_fm_sdk as fm
    import asyncio

    async def main():
        # Get the default system foundation model
        model = fm.SystemLanguageModel()

        # Check if the model is available
        is_available, reason = model.is_available()
        if is_available:
            # Create a session
            session = fm.LanguageModelSession()

            # Generate a response
            response = await session.respond("Hello, how are you?")
            print(f"Model response: {response}")
        else:
            print(f"Foundation Models not available: {reason}")

    await main()
    ##############################################################################
    print("✅ Installation snippet - PASSED")
