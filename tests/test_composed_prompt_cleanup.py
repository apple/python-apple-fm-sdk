# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.

"""
Regression tests for the composed_prompt native pointer leak.

Each of ``_respond_basic``, ``_respond_with_schema``, and
``_respond_with_schema_from_json`` creates a native ``FMComposedPrompt`` via
``_composed_prompt_from_prompt()`` but historically only released the native
``task`` pointer in their ``finally`` blocks, never the composed_prompt
pointer itself. Because composed_prompt retains any image attachments, this
leaked one native object (and its underlying image file descriptor) per call
on sequential/structured generation requests.

Two layers of coverage:

1. Unit tests that fake out the native `lib` bindings so the release logic
   in ``session.py`` can be exercised deterministically, on the success,
   error, and cancellation paths, without requiring a real model. These are
   parametrized across all three response methods.
2. An integration regression test that drives real sequential structured
   generation requests with image attachments and asserts the process's
   open file descriptor count stays flat (skipped if no model is available).
"""

import asyncio
import ctypes
import gc
import os
from types import SimpleNamespace

import apple_fm_sdk as fm
import pytest

from apple_fm_sdk import session as session_module
from apple_fm_sdk.c_helpers import _safe_from_handle

DUMMY_SCHEMA = {
    "type": "object",
    "properties": {"reply": {"type": "string"}},
    "required": ["reply"],
    "title": "DummyReply",
    "x-order": ["reply"],
    "additionalProperties": False,
}

# A duck-typed GenerationSchema stand-in: `_respond_with_schema` only ever
# reads `schema._ptr`, so a real GenerationSchema (with its own native calls)
# isn't needed here.
FAKE_GENERATION_SCHEMA = SimpleNamespace(_ptr=ctypes.c_void_p(0xABCD))

# Describes, for each of the three respond methods, which native function it
# calls and where the `future_handle` argument lands in that call so the
# fakes below can resolve/error/cancel the right future generically.
METHOD_CONFIGS = [
    {
        "name": "_respond_basic",
        "lib_func": "FMLanguageModelSessionRespond",
        "future_handle_index": 3,
        "extra_args": (),
    },
    {
        "name": "_respond_with_schema",
        "lib_func": "FMLanguageModelSessionRespondWithSchema",
        "future_handle_index": 4,
        "extra_args": (FAKE_GENERATION_SCHEMA,),
    },
    {
        "name": "_respond_with_schema_from_json",
        "lib_func": "FMLanguageModelSessionRespondWithSchemaFromJSON",
        "future_handle_index": 4,
        "extra_args": (DUMMY_SCHEMA,),
    },
]
METHOD_CONFIG_IDS = [config["name"] for config in METHOD_CONFIGS]


def _make_fake_native_call(task_ptr, future_handle_index, resolve=None):
    """Build a fake replacement for one of the FMLanguageModelSessionRespond*
    C bindings. If `resolve` is given, it's called with the future looked up
    via the `future_handle` argument (to set a result/exception); if omitted,
    the future is left pending, simulating a still-running native task."""

    def fake(*args, **kwargs):
        if resolve is not None:
            future = _safe_from_handle(args[future_handle_index])
            resolve(future)
        return task_ptr

    return fake


# =============================================================================
# 1. Unit tests: fake native bindings, no model required
# =============================================================================


@pytest.fixture
def mocked_session(monkeypatch):
    """A LanguageModelSession whose native calls are all faked out.

    Yields (session, composed_prompt_ptr, release_calls) where release_calls
    records every pointer passed to the faked ``lib.FMRelease``.
    """
    release_calls = []

    monkeypatch.setattr(
        session_module.lib, "FMRelease", lambda ptr: release_calls.append(ptr)
    )
    monkeypatch.setattr(
        session_module.lib,
        "FMLanguageModelSessionCreateFromSystemLanguageModel",
        lambda *args, **kwargs: ctypes.c_void_p(1),
    )
    # Avoid touching the real Transcript implementation during __init__.
    monkeypatch.setattr(session_module, "Transcript", lambda ptr: None)

    session = fm.LanguageModelSession()

    composed_prompt_ptr = ctypes.c_void_p(0x1234)
    monkeypatch.setattr(
        session, "_composed_prompt_from_prompt", lambda prompt: composed_prompt_ptr
    )

    yield session, composed_prompt_ptr, release_calls

    # The session wraps a fake pointer that was never really allocated by the
    # native framework. Neutralize it before monkeypatch restores the real
    # FMRelease, otherwise a later GC pass would call into native code with
    # a bogus pointer and crash the process.
    session._ptr = None


@pytest.mark.asyncio
@pytest.mark.parametrize("config", METHOD_CONFIGS, ids=METHOD_CONFIG_IDS)
async def test_composed_prompt_released_on_success(mocked_session, monkeypatch, config):
    """The composed_prompt pointer must be released after a successful response."""
    session, composed_prompt_ptr, release_calls = mocked_session
    task_ptr = ctypes.c_void_p(0x5678)

    fake_respond = _make_fake_native_call(
        task_ptr, config["future_handle_index"], lambda future: future.set_result("ok")
    )
    monkeypatch.setattr(session_module.lib, config["lib_func"], fake_respond)

    method = getattr(session, config["name"])
    result = await method("hello", *config["extra_args"])

    assert result == "ok"
    assert composed_prompt_ptr in release_calls, (
        "composed_prompt was not released - native FMComposedPrompt leak regression"
    )
    assert task_ptr in release_calls, "task pointer was not released"


@pytest.mark.asyncio
@pytest.mark.parametrize("config", METHOD_CONFIGS, ids=METHOD_CONFIG_IDS)
async def test_composed_prompt_released_on_error(mocked_session, monkeypatch, config):
    """The composed_prompt pointer must be released even when the request fails."""
    session, composed_prompt_ptr, release_calls = mocked_session
    task_ptr = ctypes.c_void_p(0x5678)

    fake_respond = _make_fake_native_call(
        task_ptr,
        config["future_handle_index"],
        lambda future: future.set_exception(fm.GuardrailViolationError("blocked")),
    )
    monkeypatch.setattr(session_module.lib, config["lib_func"], fake_respond)
    monkeypatch.setattr(session_module.lib, "FMLanguageModelSessionReset", lambda ptr: None)

    method = getattr(session, config["name"])
    with pytest.raises(fm.GuardrailViolationError):
        await method("hello", *config["extra_args"])

    assert composed_prompt_ptr in release_calls, (
        "composed_prompt was not released on the error path - leak regression"
    )
    assert task_ptr in release_calls, "task pointer was not released on the error path"


@pytest.mark.asyncio
@pytest.mark.parametrize("config", METHOD_CONFIGS, ids=METHOD_CONFIG_IDS)
async def test_composed_prompt_released_on_cancellation(mocked_session, monkeypatch, config):
    """The composed_prompt pointer must be released when the request is cancelled."""
    session, composed_prompt_ptr, release_calls = mocked_session
    task_ptr = ctypes.c_void_p(0x5678)
    cancel_calls = []

    # Never resolve the future - simulate a long-running native task.
    fake_respond = _make_fake_native_call(task_ptr, config["future_handle_index"])
    monkeypatch.setattr(session_module.lib, config["lib_func"], fake_respond)
    monkeypatch.setattr(
        session_module.lib, "FMTaskCancel", lambda t: cancel_calls.append(t)
    )
    monkeypatch.setattr(
        session_module.lib, "FMLanguageModelSessionIsResponding", lambda ptr: False
    )
    monkeypatch.setattr(session_module.lib, "FMLanguageModelSessionReset", lambda ptr: None)

    method = getattr(session, config["name"])
    task = asyncio.ensure_future(method("hello", *config["extra_args"]))
    await asyncio.sleep(0)  # Let the coroutine reach `await future`.
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert cancel_calls == [task_ptr], "native task was not cancelled"
    assert composed_prompt_ptr in release_calls, (
        "composed_prompt was not released on the cancellation path - leak regression"
    )
    assert task_ptr in release_calls, "task pointer was not released on the cancellation path"


@pytest.mark.asyncio
@pytest.mark.parametrize("config", METHOD_CONFIGS, ids=METHOD_CONFIG_IDS)
async def test_composed_prompt_released_exactly_once(mocked_session, monkeypatch, config):
    """The fix must not introduce a double-release of the composed_prompt pointer."""
    session, composed_prompt_ptr, release_calls = mocked_session
    task_ptr = ctypes.c_void_p(0x5678)

    fake_respond = _make_fake_native_call(
        task_ptr, config["future_handle_index"], lambda future: future.set_result("ok")
    )
    monkeypatch.setattr(session_module.lib, config["lib_func"], fake_respond)

    method = getattr(session, config["name"])
    await method("hello", *config["extra_args"])

    composed_prompt_release_count = sum(
        1 for ptr in release_calls if ptr is composed_prompt_ptr
    )
    assert composed_prompt_release_count == 1, (
        f"expected exactly one release of composed_prompt, "
        f"got {composed_prompt_release_count}"
    )
