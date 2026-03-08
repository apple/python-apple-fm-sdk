# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.

"""
Foundation Models SDK for Python package initialization.
"""

from .core import (
    SystemLanguageModel,
    SystemLanguageModelUseCase,
    SystemLanguageModelGuardrails,
    SystemLanguageModelUnavailableReason,
)

from .session import LanguageModelSession

from .transcript import Transcript

from .errors import (
    FoundationModelsError,
    GenerationError,
    ExceededContextWindowSizeError,
    AssetsUnavailableError,
    GuardrailViolationError,
    UnsupportedGuideError,
    UnsupportedLanguageOrLocaleError,
    DecodingFailureError,
    RateLimitedError,
    ConcurrentRequestsError,
    RefusalError,
    ToolCallError,
    GenerationErrorCode,
    InvalidGenerationSchemaError,
)

from .generable import (
    GeneratedContent,
    GenerationID,
    ConvertibleFromGeneratedContent,
    ConvertibleToGeneratedContent,
    Generable,
)

from .generation_schema import GenerationSchema

from .generable_utils import generable

from .generation_guide import GenerationGuide, GuideType, guide

from .generation_options import GenerationOptions, SamplingMode, SamplingModeType

from .tool import Tool

__version__ = "0.1.0"
__all__ = [
    "SystemLanguageModel",
    "LanguageModelSession",
    "Transcript",
    "SystemLanguageModelUseCase",
    "SystemLanguageModelGuardrails",
    "SystemLanguageModelUnavailableReason",
    "Tool",
    "FoundationModelsError",
    "GenerationError",
    "ExceededContextWindowSizeError",
    "AssetsUnavailableError",
    "GuardrailViolationError",
    "UnsupportedGuideError",
    "UnsupportedLanguageOrLocaleError",
    "InvalidGenerationSchemaError",
    "DecodingFailureError",
    "RateLimitedError",
    "ConcurrentRequestsError",
    "RefusalError",
    "ToolCallError",
    "GenerationErrorCode",
    "generable",
    "guide",
    "GenerationSchema",
    "GeneratedContent",
    "GenerationGuide",
    "GuideType",
    "GenerationOptions",
    "SamplingMode",
    "SamplingModeType",
    "GenerationID",
    "ConvertibleFromGeneratedContent",
    "ConvertibleToGeneratedContent",
    "Generable",
]
