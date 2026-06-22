# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.

from abc import ABC, abstractmethod
import logging
from typing import Optional, Union
from pathlib import Path

import ctypes

logger = logging.getLogger(__name__)

try:
    from . import _ctypes_bindings as lib
    from ._ctypes_bindings import (
        FMComposedPromptAddImageErrorUnsupportedOS,
        FMComposedPromptAddImageErrorUnsupportedSDK,
    )
except ImportError:
    raise ImportError(
        "Foundation Models C bindings not found. Please ensure _foundationmodels_ctypes.py is available."
    )


class Attachment(ABC):
    """Represents an attachment passed to a prompt with an optional label.

    Attachments allow you to include non-text content (such as images) in your
    prompts. The model can process these attachments as part of the input context.

    Use the optional ``label`` parameter to allow the model to reference this
    attachment by name, which is particularly useful when the model needs to call
    tools that reference specific attachments. When omitted, the attachment is
    added without a label.

    Note:
        This is an abstract base class. Use concrete implementations like
        :class:`ImageAttachment` to create actual attachments.

    See Also:
        - :class:`ImageAttachment`: For attaching images to prompts
    """

    @abstractmethod
    def add_to_composed_prompt(self, composed_prompt):
        """Add this attachment to a composed prompt.

        This internal method is called by the framework to add the attachment
        to the native prompt representation.

        :param composed_prompt: The native composed prompt object to add this attachment to
        :type composed_prompt: ctypes pointer
        :raises ImagePromptError: If the attachment cannot be added
        """
        pass


class ImageAttachment(Attachment):
    """Represents an image attachment passed to a prompt with an optional label.

    Image attachments enable multimodal prompts by allowing you to include images
    alongside text in your prompts. The model can analyze and reference these images
    when generating responses.

    When you provide a ``label``, the model can reference the image by name, which is
    particularly useful for:

    - Distinguishing between multiple images in a single prompt
    - Allowing tools to reference specific images
    - Providing context about what each image represents

    :ivar _path: Path to the image file on disk
    :vartype _path: Path
    :ivar _label: Optional label for the image
    :vartype _label: Optional[str]

    Examples:
        Basic image attachment::

            import apple_fm_sdk as fm
            from pathlib import Path

            session = fm.LanguageModelSession()
            image = fm.ImageAttachment(Path("photo.jpg"))
            response = await session.respond(["What's in this image?", image])

        Labeled image attachment::

            import apple_fm_sdk as fm
            from pathlib import Path

            session = fm.LanguageModelSession()
            diagram = fm.ImageAttachment(Path("diagram.png"), label="architecture")
            response = await session.respond([
                "Explain the architecture shown in the diagram",
                diagram
            ])

        Multiple images with labels::

            import apple_fm_sdk as fm
            from pathlib import Path

            session = fm.LanguageModelSession()
            before = fm.ImageAttachment(Path("before.jpg"), label="before")
            after = fm.ImageAttachment(Path("after.jpg"), label="after")
            response = await session.respond([
                "Compare the before and after images",
                before,
                after
            ])

    See Also:
        - :class:`Attachment`: Base class for all attachment types
        - :class:`~apple_fm_sdk.session.LanguageModelSession`: For using attachments in sessions

    Note:
        The image file must exist at the specified path when the attachment is created.
        Supported image formats depend on the underlying model capabilities.
    """

    def __init__(self, path: Path, label: Optional[str] = None):
        """Create an image attachment for use in prompts.

        :param path: Path on disk to the image file to attach. The file must exist
            at this location when the attachment is created.
        :type path: Path
        :param label: Optional label for the attachment. Use this to allow the model
            to reference the image by name, particularly useful when working with
            multiple images or when tools need to reference specific images.
        :type label: Optional[str]
        :raises ImagePromptError: If the file does not exist at the specified path

        Example::

            import apple_fm_sdk as fm
            from pathlib import Path

            # Create an unlabeled image attachment
            image = fm.ImageAttachment(Path("photo.jpg"))

            # Create a labeled image attachment
            diagram = fm.ImageAttachment(Path("diagram.png"), label="system_diagram")
        """
        if not path.is_file():
            raise ImagePromptError(
                f"Failed to add attachment to prompt: file does not exist at {path}"
            )
        self._path = path
        self._label = label

    def add_to_composed_prompt(self, composed_prompt):
        """Add this image attachment to a composed prompt.

        This internal method is called by the framework to add the image attachment
        to the native prompt representation. It handles encoding the file path and
        optional label, then calls the C binding to attach the image.

        :param composed_prompt: The native composed prompt object to add this attachment to
        :type composed_prompt: ctypes pointer
        :raises ImagePromptError: If the attachment cannot be added, either because
            the runtime OS or the build-time SDK doesn't support attachments, or
            another error occurs.
        """
        label_bytes = self._label.encode("utf-8") if self._label else None
        error_reason = ctypes.c_int()
        if not lib.FMComposedPromptAddAttachment(
            composed_prompt,
            str(self._path).encode("utf-8"),
            label_bytes,
            ctypes.byref(error_reason),
        ):
            if error_reason.value == FMComposedPromptAddImageErrorUnsupportedOS:
                detail = "the current OS does not support attachment prompts"
            elif error_reason.value == FMComposedPromptAddImageErrorUnsupportedSDK:
                detail = "the Xcode version used to build this package doesn't include macOS 27 SDKs"
            else:
                detail = "an unknown error occurred while adding the attachment"
            raise ImagePromptError(f"Failed to add attachment to prompt: {detail}")


PromptComponent = Union[str, Attachment]
"""Type alias for a single component of a prompt.

A prompt component can be either a text string or an :class:`Attachment` (such as
an :class:`ImageAttachment`). Components are the building blocks of prompts.

See Also:
    - :data:`Prompt`: For the complete prompt type definition
    - :class:`Attachment`: Base class for non-text prompt components
"""

Prompt = Union[PromptComponent, list[PromptComponent]]
"""Type alias for a complete prompt that can be sent to a model.

A prompt can be:

- A single string: ``"What is the capital of France?"``
- A single attachment: ``ImageAttachment(Path("photo.jpg"))``
- A list of components: ``["Describe this image:", ImageAttachment(Path("photo.jpg"))]``

This flexible type allows you to construct simple text prompts or complex multimodal
prompts that combine text and attachments.

Examples:
    Simple text prompt::

        prompt = "Hello, how are you?"

    Image-only prompt::

        from pathlib import Path
        prompt = ImageAttachment(Path("diagram.png"))

    Mixed text and image prompt::

        from pathlib import Path
        prompt = [
            "Analyze this diagram and explain the architecture:",
            ImageAttachment(Path("diagram.png"), label="system_architecture")
        ]

    Multiple images with text::

        from pathlib import Path
        prompt = [
            "Compare these two images:",
            ImageAttachment(Path("before.jpg"), label="before"),
            ImageAttachment(Path("after.jpg"), label="after"),
            "What are the main differences?"
        ]

See Also:
    - :class:`~apple_fm_sdk.session.LanguageModelSession`: For using prompts in sessions
    - :class:`ImageAttachment`: For adding images to prompts
"""


class PromptError(Exception):
    """Base exception for prompt-related errors.

    This exception is raised when there are issues with prompt construction or
    processing, such as invalid prompt components or failures in adding attachments.

    See Also:
        - :class:`ImagePromptError`: Specific error type for image attachment issues
    """

    pass


class ImagePromptError(PromptError):
    """Exception raised for errors specific to image prompts.

    This exception is raised when there are issues with image attachments, such as:

    - Image file not found at the specified path
    - The current OS does not support attachment prompts
    - The Xcode version used to build this package doesn't include the right SDKs for
      image input support

    Examples:
        Handling image prompt errors::

            import apple_fm_sdk as fm
            from pathlib import Path

            try:
                image = fm.ImageAttachment(Path("nonexistent.jpg"))
            except fm.ImagePromptError as e:
                print(f"Failed to create image attachment: {e}")

            try:
                session = fm.LanguageModelSession()
                response = await session.respond([
                    "Describe this image:",
                    fm.ImageAttachment(Path("photo.jpg"))
                ])
            except fm.ImagePromptError as e:
                print(f"Failed to process image prompt: {e}")

    See Also:
        - :class:`PromptError`: Base class for all prompt errors
        - :class:`ImageAttachment`: For creating image attachments
    """

    pass


def _composed_prompt_from_prompt(prompt: "Prompt"):
    """Build an ``FMComposedPrompt`` (the native prompt type) from a :data:`Prompt`.

    Accepts a single text string, a single :class:`Attachment`, or a list mixing the
    two, and appends each component to a freshly initialized composed prompt.

    :param prompt: The prompt to convert.
    :return: A native ``FMComposedPrompt`` pointer.
    :raises PromptError: If a component is not a ``str`` or :class:`Attachment`.
    """
    composed_prompt = lib.FMComposedPromptInitialize()

    def add_component(component):
        if isinstance(component, str):
            lib.FMComposedPromptAddText(composed_prompt, component.encode("utf-8"))
        elif isinstance(component, Attachment):
            component.add_to_composed_prompt(composed_prompt=composed_prompt)
        else:
            raise PromptError(
                f"Unsupported prompt component type {type(component)}, only str, Image, IdentifiedImage, and Attachment are supported"
            )

    from collections.abc import Iterable

    if isinstance(prompt, Iterable) and not isinstance(prompt, str):
        for element in prompt:
            add_component(element)
    else:
        add_component(prompt)
    return composed_prompt
