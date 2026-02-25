..
    For licensing see accompanying LICENSE file.
    Copyright (C) 2026 Apple Inc. All Rights Reserved.

Errors
======

This page documents the exception classes for error handling.

.. note::
   **Swift Equivalent:** ``GenerationError`` cooresponds to the the `LanguageModelSession.GenerationError 
   <https://developer.apple.com/documentation/foundationmodels/languagemodelsession/generationerror>`_ 
   in the Swift Foundation Models Framework. ``ToolCallError`` corresponds to the 
   `LanguageModelSession.ToolCallError <https://developer.apple.com/documentation/foundationmodels/languagemodelsession/toolcallerror>`_ in 
   the Swift Foundation Models Framework. ``InvalidGenerationSchemaError`` is unique to the Python SDK 
   and does not have a direct Swift equivalent since it means a schema failed to compile in the underlying Swift.

Base Exceptions
---------------

FoundationModelsError
~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: apple_fm_sdk.FoundationModelsError
   :members:
   :undoc-members:
   :show-inheritance:

   Base exception for all Foundation Models SDK errors.

Generation Errors
-----------------

GenerationError
~~~~~~~~~~~~~~~

.. autoclass:: apple_fm_sdk.GenerationError
   :members:
   :undoc-members:
   :show-inheritance:

   Base class for errors that occur during text generation.

ExceededContextWindowSizeError
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: apple_fm_sdk.ExceededContextWindowSizeError
   :members:
   :undoc-members:
   :show-inheritance:

   Raised when the input exceeds the model's context window size.

AssetsUnavailableError
~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: apple_fm_sdk.AssetsUnavailableError
   :members:
   :undoc-members:
   :show-inheritance:

   Raised when required model assets are not available.

GuardrailViolationError
~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: apple_fm_sdk.GuardrailViolationError
   :members:
   :undoc-members:
   :show-inheritance:

   Raised when content violates safety guardrails. For more on the guardrails, see the Swift 
   reference article on ``SystemLanguageModel`` safety meansures: `Improving the safety of generative model output <https://developer.apple.com/documentation/foundationmodels/improving-the-safety-of-generative-model-output>`_. 

UnsupportedGuideError
~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: apple_fm_sdk.UnsupportedGuideError
   :members:
   :undoc-members:
   :show-inheritance:

   Raised when an unsupported guide constraint is used.

UnsupportedLanguageOrLocaleError
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: apple_fm_sdk.UnsupportedLanguageOrLocaleError
   :members:
   :undoc-members:
   :show-inheritance:

   Raised when the requested language or locale is not supported.

DecodingFailureError
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: apple_fm_sdk.DecodingFailureError
   :members:
   :undoc-members:
   :show-inheritance:

   Raised when decoding the model's output fails.

RateLimitedError
~~~~~~~~~~~~~~~~

.. autoclass:: apple_fm_sdk.RateLimitedError
   :members:
   :undoc-members:
   :show-inheritance:

   Raised when rate limits are exceeded. Rate limits do not apply to the 
   on-device ``SystemLanguageModel`` on macOS so you should not encounter this error.

ConcurrentRequestsError
~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: apple_fm_sdk.ConcurrentRequestsError
   :members:
   :undoc-members:
   :show-inheritance:

   Raised when too many concurrent requests are made. The python SDK does 
   not enforce concurrency limits so you should not encounter this error.

RefusalError
~~~~~~~~~~~~

.. autoclass:: apple_fm_sdk.RefusalError
   :members:
   :undoc-members:
   :show-inheritance:

   Raised when the model refuses to generate a response _specifically_ for 
   safety reasons on a generable output. For more on refusals, see the Swift 
   reference article on ``SystemLanguageModel`` safety meansures: `Improving the safety of generative model output <https://developer.apple.com/documentation/foundationmodels/improving-the-safety-of-generative-model-output>`_. 


   **Attributes:**

   * ``reason`` (str): The reason for refusal

Schema Errors
-------------

InvalidGenerationSchemaError
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: apple_fm_sdk.InvalidGenerationSchemaError
   :members:
   :undoc-members:
   :show-inheritance:

   Raised when a generation schema is invalid. This means the schema failed 
   to compile in the underlying Swift code and most commonly occurs if you 
   give an invalid json schema to the ``json_schema`` parameter of the ``respond()`` method.
   This error may also occur if you use the ``@generable`` decorator on a class with 
   properties that cannot be serialized to a json schema.

   To validate if your json schema is valid, run the following code snippet in **Swift**:

   .. code-block:: swift

      import FoundationModels

      jsonSchemaString = "..."  // Your JSON schema as a string

      let schema = try JSONDecoder().decode(
        GenerationSchema.self,
        from: Data(jsonSchemaString.utf8)
      )

   The code above is the same validation step used in the Python SDK, so if it runs 
   without throwing an error, your schema is valid.

   For examples of what valid json schemas look like, see the tester schemas in the Github repo, 
   such as ``tests/tester_schemas/hedgehog.json``.

Tool Errors
-----------

ToolCallError
~~~~~~~~~~~~~

.. autoclass:: apple_fm_sdk.ToolCallError
   :members:
   :undoc-members:
   :show-inheritance:

   Raised when a ``Tool`` call fails.

   **Attributes:**

   * ``tool_name`` (str): Name of the tool that failed
   * ``underlying_error`` (Exception): The original exception
