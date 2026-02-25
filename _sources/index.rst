..
    For licensing see accompanying LICENSE file.
    Copyright (C) 2026 Apple Inc. All Rights Reserved.

Foundation Models SDK for Python Documentation
==============================================

**Platform Support:** macOS

**Python Version:** 3.10+

Python bindings for Apple's `Foundation Models framework <https://developer.apple.com/documentation/foundationmodels>`_ 
give you access to the on-device foundation model at the core of Apple Intelligence on macOS. You can use this Python SDK to evaluate 
your Swift app's Foundation Models features, using Python's rich ecosystem of tools and libraries for data and machine learning. 
These Python bindings run the Apple's Foundation Models framework in Swift under the hood, so you can be confident that your evaluations reflect 
real on-device performance and behavior.

Keep in mind that it's your responsibility to design AI experiences with care.
To learn about practical strategies you can implement in code, **check out:** 
`Improving the safety of generative model output <https://developer.apple.com/documentation/foundationmodels/improving-the-safety-of-generative-model-output>`_
and Apple's `Human Interface Guidelines on Generative AI <https://developer.apple.com/design/human-interface-guidelines/generative-ai>`_.

Requirements
------------
Since this SDK calls the on-device foundation model at the core of Apple Intelligence it requires:

* macOS 26.0+
* Download `Xcode 26.0+ <https://developer.apple.com/xcode/>`_ and agree to the `Xcode and Apple SDKs agreement <https://www.apple.com/legal/sla/docs/xcode.pdf>`_ in the Xcode app.
* Python 3.10+
* Apple Intelligence turned on for a `compatible device <https://support.apple.com/en-us/121115>`_

Quick Start
-----------

Installation
~~~~~~~~~~~~

Foundation Models SDK for Python is currently in Beta. Use the development installation:

1. Ensure you have met the requirements above.

2. Clone the repository and navigate to the project directory:

.. code-block:: bash

      git clone https://github.com/apple/python-apple-fm-sdk
      cd python-apple-fm-sdk

3. Create a virtual environment (`uv <https://docs.astral.sh/uv/>`_ recommended) and activate it:

.. code-block:: bash

      uv venv
      source .venv/bin/activate
      
4. Install the package locally in editable mode:

.. code-block:: bash

      uv pip install -e .

Basic Usage
~~~~~~~~~~~

.. code-block:: python

   import apple_fm_sdk as fm

   # Get the default system foundation model
   model = fm.SystemLanguageModel()

   # Check if the model is available
   is_available, reason = model.is_available()
   if is_available:
       # Create a session
       session = fm.LanguageModelSession(model=model)

       # Generate a response
       response = await session.respond(prompt="Hello, how are you?")
       print(f"Model response: {response}")
   else:
       print(f"Foundation Models not available: {reason}")

Documentation Contents
----------------------

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   getting_started
   basic_usage
   streaming
   guided_generation
   tools
   evaluation

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/systemmodel
   api/session
   api/generable
   api/tools
   api/transcript
   api/errors

.. toctree::
   :maxdepth: 1
   :caption: Development
   

Indices and tables
==================

* :ref:`genindex`
* :ref:`search`
