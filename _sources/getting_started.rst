..
    For licensing see accompanying LICENSE file.
    Copyright (C) 2026 Apple Inc. All Rights Reserved.

Getting Started
===============

.. note::
   **Swift Equivalent:** This guide covers installation and setup for the Python SDK. For Swift development, see the `Foundation Models framework <https://developer.apple.com/documentation/foundationmodels>`_ documentation.

Get started with the Foundation Models SDK for Python.

Prerequisites
-------------

Before you begin, ensure you have:

1. **Compatible Hardware**: macOS 26.0+ on compatible Mac that supports Apple Intelligence

2. **Xcode Installed**: Xcode 26.0+ or later with command line tools installed. **Tip**: Make sure your Xcode version matches your macOS version to avoid model compatibility issues.

3. **Python Environment**: Python 3.10 or later installed

4. **Apple Intelligence**: Turned on for your device (see `Apple's support page <https://support.apple.com/en-us/121115>`_)

Installation
------------

Step 1: Install 
~~~~~~~~~~~~~~~

Foundation Models SDK for Python is currently in Beta. Install using the development installation
instruction below.

Step 2: Verify Installation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Verify the installation:

.. code-block:: python

   import apple_fm_sdk as fm

   # Check if the model is available
   model = fm.SystemLanguageModel()
   is_available, reason = model.is_available()
   
   if is_available:
       print("Foundation Models SDK is ready!")
   else:
       print(f"Foundation Models not available: {reason}")

Development Installation
------------------------

If you need to modify the SDK, install from source:

Step 1: Clone the Repository
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   git clone https://github.com/apple/python-apple-fm-sdk
   cd python-apple-fm-sdk

Step 2: Create a Virtual Environment (Recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Using `uv <https://docs.astral.sh/uv/>`_ (recommended):

.. code-block:: bash

   uv venv
   source .venv/bin/activate

Step 3: Install in Editable Mode
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   uv sync
   uv pip install -e .

This will install the package in editable mode along with all dependencies.

Step 4: Verify Changes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
After making any change, be sure to build the project again and test:

.. code-block:: bash

   uv pip install -e .
   pytest

All tests can be found in the `tests/ <https://github.com/apple/python-apple-fm-sdk/tree/main/tests>`_ directory.


Common Installation Issues
--------------------------

Model Not Available
~~~~~~~~~~~~~~~~~~~

If you see "Foundation Models not available", check:

1. Your device supports Apple Intelligence
2. Apple Intelligence is turned on in System Settings
3. You're running a compatible OS version (macOS 26.0+)
4. Verify that your Xcode version matches your macOS version exactly.


Next Steps
----------

See examples on the git repository under `examples/ <https://github.com/apple/python-apple-fm-sdk/tree/main/examples>`_ for more usage patterns.
