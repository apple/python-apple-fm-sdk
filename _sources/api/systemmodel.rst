..
    For licensing see accompanying LICENSE file.
    Copyright (C) 2026 Apple Inc. All Rights Reserved.

SystemLanguageModel
===================

This page documents the core classes and functions of the Foundation Models SDK.

.. note::
   **Swift Equivalent:** This Python API corresponds to the `SystemLanguageModel <https://developer.apple.com/documentation/foundationmodels/systemlanguagemodel>`_ class in the Swift Foundation Models Framework.

SystemLanguageModel
-------------------

.. autoclass:: apple_fm_sdk.SystemLanguageModel
   :members:
   :undoc-members:

SystemLanguageModelUseCase
---------------------------

.. autoclass:: apple_fm_sdk.SystemLanguageModelUseCase
   :members:
   :undoc-members:
   :exclude-members: GENERAL, CONTENT_TAGGING

SystemLanguageModelGuardrails
------------------------------

.. autoclass:: apple_fm_sdk.SystemLanguageModelGuardrails
   :members:
   :undoc-members:
   :exclude-members: DEFAULT, PERMISSIVE_CONTENT_TRANSFORMATIONS

SystemLanguageModelUnavailableReason
-------------------------------------

.. autoclass:: apple_fm_sdk.SystemLanguageModelUnavailableReason
   :members:
   :undoc-members:
   :exclude-members: APPLE_INTELLIGENCE_NOT_ENABLED, DEVICE_NOT_ELIGIBLE, MODEL_NOT_READY, UNKNOWN
