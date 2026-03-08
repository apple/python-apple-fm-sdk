..
    For licensing see accompanying LICENSE file.
    Copyright (C) 2026 Apple Inc. All Rights Reserved.

Generation Options
==================

This page documents the classes for controlling how the model generates responses.

.. note::
   **Swift Equivalent:** This Python API corresponds to the `GenerationOptions <https://developer.apple.com/documentation/foundationmodels/generationoptions>`_ structure in the Swift Foundation Models Framework.

GenerationOptions
-----------------

.. autoclass:: apple_fm_sdk.GenerationOptions
   :members:
   :undoc-members:
   :exclude-members: to_dict, __post_init__

SamplingMode
------------

.. autoclass:: apple_fm_sdk.SamplingModeType
   :exclude-members: RANDOM, GREEDY

.. autoclass:: apple_fm_sdk.SamplingMode
   :members:
   :undoc-members: