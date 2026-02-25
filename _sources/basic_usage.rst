..
    For licensing see accompanying LICENSE file.
    Copyright (C) 2026 Apple Inc. All Rights Reserved.

Basic Usage
===========

.. note::
   **Swift Equivalent:** This guide covers concepts that correspond to the `Foundation Models framework <https://developer.apple.com/documentation/foundationmodels>`_ in Swift.

This guide covers the fundamental concepts and usage patterns of the Foundation Models SDK for Python.

Core Concepts
-------------

The SDK is built around three main components:

1. **SystemLanguageModel**: Represents the on-device Apple Intelligence model
2. **LanguageModelSession**: Manages sessions and context
3. **Responses**: Generated text or structured data

Creating a Model
----------------

Get started with the default system model:

.. code-block:: python

   import apple_fm_sdk as fm

   model = fm.SystemLanguageModel()

Checking Model Availability
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Always check if the model is available before use:

.. code-block:: python

   is_available, reason = model.is_available()
   
   if not is_available:
       print(f"Model not available: {reason}")
       # Handle unavailability (for example, display an error message)
   else:
       # Proceed with model usage
       pass

Custom Model Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~

You can customize the model with specific use cases and guardrails:

.. code-block:: python

   import apple_fm_sdk as fm

   model = fm.SystemLanguageModel(
       use_case=fm.SystemLanguageModelUseCase.CONTENT_TAGGING,
       guardrails=fm.SystemLanguageModelGuardrails.PERMISSIVE_CONTENT_TRANSFORMATIONS
   )

Current available use cases:

* ``GENERAL``: Default for general text generation
* ``CONTENT_TAGGING``: For categorizing and labeling content
* (See :doc:`api/systemmodel` for complete list)

Creating a Session
------------------

A session manages the session context and handles requests:

Basic Session
~~~~~~~~~~~~~

.. code-block:: python

   import apple_fm_sdk as fm

   # Create a session with default model
   session = fm.LanguageModelSession()

Session with Custom Model with Custom Instructions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import apple_fm_sdk as fm

   custom_model = fm.SystemLanguageModel(
       use_case=fm.SystemLanguageModelUseCase.CONTENT_TAGGING,
   )
   
   session = fm.LanguageModelSession(
       instructions="You are a tagging assistant who can help tag journal entries.",
       model=custom_model
   )

Generating Responses
--------------------

Simple Text Generation
~~~~~~~~~~~~~~~~~~~~~~

The most basic usage is generating a text response:

.. code-block:: python

   import apple_fm_sdk as fm

   async def generate_response():
       model = fm.SystemLanguageModel()
       is_available, _ = model.is_available()
       
       if is_available:
           session = fm.LanguageModelSession()
           
           response = await session.respond("What is the Swift bird species?")
           print(response)

   # Run the async function
   await generate_response()

Multi-turn Sessions
~~~~~~~~~~~~~~~~~~~

Sessions maintain context across multiple interactions:

.. code-block:: python

    import apple_fm_sdk as fm

    async def multi_turn_session():
        session = fm.LanguageModelSession()

        # First question
        response1 = await session.respond("What is the capital of France?")
        print(f"Assistant: {response1}")

        # Follow-up question (context is maintained)
        response2 = await session.respond("What are some famous landmarks there?")
        print(f"Assistant: {response2}")

    await multi_turn_session()

Checking Session State
~~~~~~~~~~~~~~~~~~~~~~

You can check if a session is currently processing:

.. code-block:: python

   if session.is_responding:
       print("Session is currently generating a response")
   else:
       print("Session is idle")

Error Handling
--------------

Always handle potential errors when working with the model:

.. code-block:: python

    import apple_fm_sdk as fm


    async def safe_generation():
        session = fm.LanguageModelSession()
        
        try:
            response = await session.respond("Your prompt here")
            print(response)
        except fm.ExceededContextWindowSizeError:
            print("Prompt is too long")
        except fm.GuardrailViolationError as e:
            print(f"Caught GuardrailViolationError: {e}")
        except fm.GenerationError as e:
            print(f"Generation error: {e}")

Common error types:

* ``ExceededContextWindowSizeError``: Input is too long
* ``GuardrailViolationError``: Content violates safety guidelines
* ``AssetsUnavailableError``: Model assets not available

See :doc:`api/errors` for complete error documentation.
