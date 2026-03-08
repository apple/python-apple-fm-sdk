# Foundation Models SDK for Python

Python bindings for Apple's [Foundation Models framework](https://developer.apple.com/documentation/foundationmodels), providing access to the on-device foundation model
at the core of Apple Intelligence on macOS.

## Overview

The Foundation Models SDK for Python provides a Pythonic interface to Apple's Foundation Models framework.

You can:

- **Evaluate Swift Foundation Models app features** by running batch inference and analyzing results from Python
- Perform **on-device inference** with the system foundation model
- Stream **real-time text generation** responses
- Use **guided generation** with structured output schemas and constraints
- Get **type-safe responses** using Python decorators for guided generation
- Configure **custom model settings** for different model options
- Process **transcripts exported from Swift apps** for quality analysis

Keep in mind that it's your responsibility to design AI experiences with care.
To learn about practical strategies you can implement in code, **check out:**
[Improving the safety of generative model output](https://developer.apple.com/documentation/foundationmodels/improving-the-safety-of-generative-model-output)
and Apple's [Human Interface Guidelines on Generative AI](https://developer.apple.com/design/human-interface-guidelines/generative-ai).

## Requirements

- macOS 26.0+
- Download [Xcode 26.0+](https://developer.apple.com/xcode/) and agree to the [Xcode and Apple SDKs agreement](https://www.apple.com/legal/sla/docs/xcode.pdf) in the Xcode app.
- Python 3.10+
- Apple Intelligence turned on for [a compatible Mac](https://support.apple.com/en-us/121115)

## Contributing

This project is not yet taking contributions. Stay tuned!

## Installation

```bash
pip install apple-fm-sdk
```

Alternatively, you can use the development install instructions below.

## Documentation

- [Documentation page](https://apple.github.io/python-apple-fm-sdk/)
- [Code examples](https://github.com/apple/python-apple-fm-sdk/tree/main/examples)

## Basic usage

```python
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

# Run async function
asyncio.run(main())
```

### Development Installation

If you need to modify the SDK or install from source:

1. Get the code

```bash
git clone https://github.com/apple/python-apple-fm-sdk
cd python-apple-fm-sdk
```

2. (Optional but recommended) Make a virtual environment. Install [uv](https://docs.astral.sh/uv/getting-started/installation/) (or your package manager of choice) then:

```bash
uv venv
source .venv/bin/activate
```

3. Install the package locally in editable mode:

```bash
uv sync
```

4. After making any change, be sure to build the project again and test:

```bash
uv pip install -e .
pytest
```

---

For licensing see accompanying LICENSE file.
Copyright (C) 2026 Apple Inc. All Rights Reserved.
