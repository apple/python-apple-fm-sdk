# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.

"""
Internal C/Python interoperability helpers for Foundation Models.

This module provides low-level utilities for interfacing between Python and the
Foundation Models C/Swift implementation. It handles memory management, callback
registration, error handling, and type conversions across the language boundary.

.. warning::
    This module is for internal use only. The APIs in this module are not part
    of the public interface and may change without notice. Users should not
    import or use these utilities directly.

Key Components:
    - Memory management for C objects via :class:`_ManagedObject`
    - Callback registration and lifecycle management
    - Error extraction and conversion from C error codes
    - Streaming callback infrastructure

Memory Management:
    The module implements a reference counting system to ensure proper cleanup
    of C objects. Python objects that wrap C pointers inherit from
    :class:`_ManagedObject` which handles retain/release semantics compatible
    with Swift's ARC (Automatic Reference Counting).

Callback Safety:
    Python callback objects are registered in a global registry to prevent
    garbage collection while they're in use by C code. The registry uses
    thread-safe operations to handle concurrent access.

.. note::
    All C pointers passed from Swift to Python are assumed to be retained
    (ownership transferred). Python is responsible for releasing them exactly
    once when the object is deallocated.
"""

import asyncio
import threading
import queue
import logging
from typing import Optional

from .errors import (
    FoundationModelsError,
    GenerationErrorCode,
    _status_code_to_exception,
)

import ctypes
from ctypes import c_void_p

try:
    from . import _ctypes_bindings as lib
except ImportError:
    raise ImportError(
        "Foundation Models C bindings not found. Please ensure _foundationmodels_ctypes.py is available."
    )

# Global registry to keep Python objects alive while used as ctypes callbacks
_active_handles = {}
_handle_lock = threading.Lock()

# Logger for error reporting
logger = logging.getLogger(__name__)


def _register_handle(obj):
    """
    Register a Python object to prevent garbage collection during C callbacks.

    This function adds a Python object to a global registry, keeping it alive
    while it's being used by C code. The object's memory address is used as
    a handle that can be passed to C and later used to retrieve the object.

    :param obj: The Python object to register (typically an asyncio.Future or
        callback object)
    :type obj: Any
    :return: A ctypes void pointer containing the object's memory address
    :rtype: ctypes.c_void_p

    .. warning::
        Every call to _register_handle must be paired with a call to
        _unregister_handle to prevent memory leaks.

    .. note::
        This function is thread-safe and can be called from multiple threads.
    """
    handle_addr = id(obj)
    handle_ptr = ctypes.c_void_p(handle_addr)
    with _handle_lock:
        _active_handles[handle_addr] = obj
    return handle_ptr


def _unregister_handle(handle_ptr):
    """
    Unregister a previously registered Python object handle.

    Removes the object from the global registry, allowing it to be garbage
    collected if there are no other references to it.

    :param handle_ptr: The handle pointer returned by _register_handle, or None
    :type handle_ptr: Optional[ctypes.c_void_p]

    .. note::
        This function is thread-safe and can be called from multiple threads.
        It's safe to call with None or an already-unregistered handle.
    """
    if handle_ptr:
        handle_addr = (
            handle_ptr.value if isinstance(handle_ptr, c_void_p) else handle_ptr
        )
        with _handle_lock:
            _active_handles.pop(handle_addr, None)


def _safe_from_handle(handle_ptr):
    """
    Safely retrieve a Python object from its handle pointer.

    Looks up the object in the global registry using the handle pointer.
    Returns None if the handle is invalid or the object has been unregistered.

    :param handle_ptr: The handle pointer to look up
    :type handle_ptr: Optional[ctypes.c_void_p]
    :return: The registered Python object, or None if not found
    :rtype: Optional[Any]

    .. note::
        This function is thread-safe and can be called from multiple threads.
    """
    if not handle_ptr:
        return None

    handle_addr = handle_ptr.value if isinstance(handle_ptr, c_void_p) else handle_ptr
    with _handle_lock:
        return _active_handles.get(handle_addr, None)


def _get_error_string(error_code, error_desc):
    """
    Extract error information from C error output parameters.

    :param error_code: c_int object containing the error code
    :type error_code: ctypes.c_int
    :param error_desc: POINTER(c_char) object containing the error description
    :type error_desc: ctypes.POINTER(ctypes.c_char)
    :return: (error_code, error_description) or (None, None) if no error
    :rtype: tuple
    """
    if error_code is None or error_desc is None:
        return None, None

    # Get the error code value directly from the c_int object
    err_code = error_code.value if hasattr(error_code, "value") else None
    err_desc = None

    # Get the error description from the POINTER(c_char) object
    if error_desc:
        try:
            err_desc = ctypes.string_at(error_desc).decode("utf-8")
        except Exception:
            err_desc = "Failed to decode error description"
        finally:
            # Free the error string allocated by C
            lib.FMFreeString(error_desc)

    return err_code, err_desc


def _get_error_string(error_code, error_desc):
    """
    Extract error information from C error output parameters.

    :param error_code: c_int object containing the error code
    :type error_code: ctypes.c_int
    :param error_desc: POINTER(c_char) object containing the error description
    :type error_desc: ctypes.POINTER(ctypes.c_char)
    :return: (error_code, error_description) or (None, None) if no error
    :rtype: tuple
    """
    if error_code is None or error_desc is None:
        return None, None

    # Get the error code value directly from the c_int object
    err_code = error_code.value if hasattr(error_code, "value") else None
    err_desc = None

    # Get the error description from the POINTER(c_char) object
    if error_desc:
        try:
            err_desc = ctypes.string_at(error_desc).decode("utf-8")
        except Exception:
            err_desc = "Failed to decode error description"
        finally:
            # Free the error string allocated by C
            lib.FMFreeString(error_desc)

    return err_code, err_desc


class _ManagedObject:
    """
    Base class for Python objects that wrap C/Swift pointers requiring memory management.

    This class implements reference counting semantics compatible with Swift's ARC
    (Automatic Reference Counting). Objects that wrap C pointers should inherit from
    this class to ensure proper memory management across the Python/C boundary.

    The class follows these ownership rules:
    - When Swift passes a pointer via ``passRetained``, ownership is transferred to
      Python with a +1 reference count
    - Python must release the pointer exactly once when the object is deallocated
    - Subclasses should NOT call ``_retain()`` in their ``__init__`` methods

    :ivar _ptr: The C pointer being managed
    :vartype _ptr: ctypes.c_void_p

    Example:
        Subclassing for a custom managed type::

            class MyManagedType(_ManagedObject):
                def __init__(self, ptr):
                    super().__init__(ptr)
                    # Do NOT call self._retain() here

                def some_method(self):
                    # Use self._ptr to call C functions
                    lib.FMSomeFunction(self._ptr)

    .. warning::
        Improper use of ``_retain()`` can cause memory leaks. The pointer is
        already retained when passed from Swift to Python.

    .. note::
        The ``__del__`` method automatically releases the pointer when the
        Python object is garbage collected.
    """

    def __init__(self, ptr):
        """
        Initialize a managed object with a C pointer.

        :param ptr: The C pointer to manage. Must not be None.
        :type ptr: ctypes.c_void_p
        :raises FoundationModelsError: If ptr is None or invalid
        """
        if not ptr:
            raise FoundationModelsError("Failed to create object")
        self._ptr = ptr

    def _retain(self):
        """
        Manually increment the reference count of the managed pointer.

        .. warning::
            IMPORTANT OWNERSHIP RULE: When Swift passes a pointer via
            ``passRetained``, it transfers ownership to Python with +1 reference
            count. Python must release it exactly once in ``__del__``.

            Subclasses should NOT call ``_retain()`` in their ``__init__`` methods,
            as this would create +2 references but only -1 release, causing memory
            leaks.

            The pointer is already retained by Swift when passed to Python.

        .. note::
            This method is rarely needed in normal usage. It's provided for
            advanced scenarios where manual reference counting is required.
        """
        lib.FMRetain(self._ptr)

    def _release(self):
        """
        Decrement the reference count of the managed pointer.

        This method is called automatically by ``__del__`` when the Python object
        is garbage collected. It should not normally be called directly.

        .. note::
            It's safe to call this method multiple times or on an already-released
            pointer due to the internal null check.
        """
        if hasattr(self, "_ptr") and self._ptr:
            lib.FMRelease(self._ptr)

    def __del__(self):
        """
        Destructor that releases the managed pointer.

        This method is called automatically by Python's garbage collector when
        the object is being destroyed. It ensures the C pointer is properly
        released to prevent memory leaks.
        """
        self._release()


# Use the callback type from ctypes bindings instead of redefining it
@lib.FMLanguageModelSessionResponseCallback
def _session_callback(status, content, length, future_handle):
    """ctypes callback function."""

    # Define helper functions outside try block to avoid "possibly unbound" errors
    async def _set_future_result(future: asyncio.Future, result):
        if not future.cancelled():
            future.set_result(result)

    async def _set_future_exception(future: asyncio.Future, e):
        if not future.cancelled():
            future.set_exception(e)

    try:
        future = _safe_from_handle(future_handle)
        if future is None or future.cancelled():
            # Future is invalid or cancelled - nothing to clean up for this callback
            # (content is a byte array, not a managed pointer)
            return

        if content and length > 0:
            # Get the actual bytes and decode to string
            content_bytes = bytes(content[:length].data)
            content_str = content_bytes.decode("utf-8")
        else:
            content_str = None

        if status == GenerationErrorCode.SUCCESS:
            asyncio.run_coroutine_threadsafe(
                _set_future_result(future, content_str), future.get_loop()
            )
        else:
            # Convert status code to specific error
            error = _status_code_to_exception(status)
            asyncio.run_coroutine_threadsafe(
                _set_future_exception(future, error), future.get_loop()
            )

    except Exception as e:
        try:
            future = _safe_from_handle(future_handle)
            if future and not future.cancelled():
                asyncio.run_coroutine_threadsafe(
                    _set_future_exception(future, e), future.get_loop()
                )
        except Exception as error:
            logger.error(f"Unhandled Exception in session callback cleanup: {error}")


# Use the callback type from the bindings
@lib.FMLanguageModelSessionStructuredResponseCallback
def _session_structured_callback(status, content_ptr, future_handle):
    """ctypes callback function."""
    from .generable import GeneratedContent  # Import here to avoid circular import

    # Track whether we've transferred ownership of content_ptr to a GeneratedContent object
    content_ptr_owned = False

    # Define helper functions outside try block to avoid "possibly unbound" errors
    async def _set_future_result(future: asyncio.Future, result):
        if not future.cancelled():
            future.set_result(result)

    async def _set_future_exception(future: asyncio.Future, e):
        if not future.cancelled():
            future.set_exception(e)

    try:
        future = _safe_from_handle(future_handle)
        if future is None or future.cancelled():
            # Future is invalid or cancelled - clean up content_ptr and return
            if content_ptr:
                lib.FMRelease(content_ptr)
            return

        if status != GenerationErrorCode.SUCCESS:
            debug_info = None
            if content_ptr:
                # Release the content pointer if error occurred
                generated_content = GeneratedContent(_ptr=content_ptr)
                content_ptr_owned = True  # GeneratedContent now owns it
                debug_info = str(generated_content._content_dict)
                del generated_content  # Ensure release

            # Convert status code to specific error
            error = _status_code_to_exception(status, debug_description=debug_info)
            asyncio.run_coroutine_threadsafe(
                _set_future_exception(future, error), future.get_loop()
            )
            return

        if content_ptr:
            # Create GeneratedContent from the C pointer
            generated_content = GeneratedContent(_ptr=content_ptr)
            content_ptr_owned = True  # GeneratedContent now owns it
            asyncio.run_coroutine_threadsafe(
                _set_future_result(future, generated_content), future.get_loop()
            )
        else:
            # Should not happen, but handle gracefully
            error = FoundationModelsError("No content returned from guided generation")
            asyncio.run_coroutine_threadsafe(
                _set_future_exception(future, error), future.get_loop()
            )

    except Exception as e:
        # On exception, clean up content_ptr if we haven't transferred ownership
        if content_ptr and not content_ptr_owned:
            try:
                lib.FMRelease(content_ptr)
            except Exception as cleanup_error:
                logger.error(
                    f"Error releasing C pointer during session cleanup: {cleanup_error}"
                )

        try:
            future = _safe_from_handle(future_handle)
            if future and not future.cancelled():
                asyncio.run_coroutine_threadsafe(
                    _set_future_exception(future, e), future.get_loop()
                )
        except Exception as error:
            logger.error(
                f"Unhandled Exception in structured session callback cleanup: {error}"
            )


@lib.FMSystemLanguageModelTokenCountCallback
def _token_count_callback(status, token_count, error_desc, future_handle):
    """ctypes callback for token counting.

    Resolves the registered future with the integer token count on success, or with
    an appropriate exception (built from the status code and optional error description)
    on failure.
    """

    async def _set_future_result(future: asyncio.Future, result):
        if not future.cancelled():
            future.set_result(result)

    async def _set_future_exception(future: asyncio.Future, e):
        if not future.cancelled():
            future.set_exception(e)

    try:
        future = _safe_from_handle(future_handle)
        if future is None or future.cancelled():
            return

        if status == GenerationErrorCode.SUCCESS:
            asyncio.run_coroutine_threadsafe(
                _set_future_result(future, int(token_count)), future.get_loop()
            )
        else:
            debug_description = None
            if error_desc:
                try:
                    debug_description = ctypes.string_at(error_desc).decode("utf-8")
                except Exception:
                    debug_description = None
            error = _status_code_to_exception(
                status, debug_description=debug_description
            )
            asyncio.run_coroutine_threadsafe(
                _set_future_exception(future, error), future.get_loop()
            )

    except Exception as e:
        try:
            future = _safe_from_handle(future_handle)
            if future and not future.cancelled():
                asyncio.run_coroutine_threadsafe(
                    _set_future_exception(future, e), future.get_loop()
                )
        except Exception as error:
            logger.error(f"Unhandled Exception in token count callback cleanup: {error}")


class StreamingCallback:
    """
    Callback handler for streaming generation responses.

    This class manages the callback mechanism for streaming text generation,
    collecting generated content chunks as they arrive from the C layer and
    making them available through a thread-safe queue. It handles both successful
    content delivery and error conditions.

    The callback uses a queue-based approach to bridge between the C callback
    thread and Python consumer threads, allowing for real-time processing of
    generated content as it becomes available.

    :ivar error: Stores any error that occurred during streaming, or None if
        no error occurred
    :vartype error: Optional[FoundationModelsError]
    :ivar queue: Thread-safe queue containing content chunks. None is used as
        an end-of-stream sentinel value.
    :vartype queue: queue.Queue
    :ivar completed: Event that is set when streaming completes (either
        successfully or with an error)
    :vartype completed: threading.Event

    Example:
        Using StreamingCallback (typically done internally by Session)::

            callback = StreamingCallback()
            # Pass callback._callback to C layer
            # ...
            # Consume from the queue
            while True:
                content = callback.queue.get()
                if content is None:
                    break  # End of stream
                print(content, end='', flush=True)

            if callback.error:
                raise callback.error

    .. note::
        This class is used internally by the Session class for streaming
        operations. Users typically don't need to instantiate it directly.

    .. warning::
        The callback must remain alive (not garbage collected) while the C
        layer is using it. The Session class handles this automatically.
    """

    error: Optional[FoundationModelsError]
    queue: "queue.Queue"
    completed: threading.Event

    def __init__(self):
        """
        Initialize a new StreamingCallback instance.

        Creates the internal queue, error tracking, and completion event,
        and sets up the C callback function that will be invoked by the
        Foundation Models runtime.
        """
        self.queue = queue.Queue()
        self.error = None
        self.completed = threading.Event()

        # Use the callback type from ctypes bindings
        @lib.FMLanguageModelSessionResponseCallback
        def _callback_impl(status, content, length, user_info):
            try:
                if status != GenerationErrorCode.SUCCESS:
                    # Convert status code to specific error
                    self.error = _status_code_to_exception(status)
                    self.queue.put(None)  # Signal end
                    self.completed.set()
                    return

                if content and length > 0:
                    # Get the actual bytes and decode to string
                    content_bytes = bytes(content[:length].data)
                    current_content = content_bytes.decode("utf-8", errors="replace")
                    self.queue.put(current_content)
                else:
                    # End of stream
                    self.queue.put(None)  # Signal end
                    self.completed.set()

            except Exception as e:
                self.error = FoundationModelsError(f"Callback error: {e}")
                self.queue.put(None)  # Signal end
                self.completed.set()

        self._callback = _callback_impl
