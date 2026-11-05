# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.

"""
Regression test: ``respond(generating=..., options=...)`` must forward ``options``.

The ``generating`` branch of ``LanguageModelSession.respond`` built the type's schema and
called ``_respond_with_schema`` without ``options``, so sampling mode, temperature and
``maximum_response_tokens`` were silently ignored for guided generation with a Generable
type, while ``schema=`` and ``json_schema=`` honored them. Fakes the native session so no
model is required.
"""

import ctypes

import pytest

import apple_fm_sdk as fm
from apple_fm_sdk import session as session_module


@fm.generable("A reply")
class Reply:
    text: str = fm.guide("The reply text")


class _Captured(Exception):
    pass


@pytest.fixture
def session(monkeypatch):
    monkeypatch.setattr(session_module.lib, "FMRelease", lambda ptr: None)
    monkeypatch.setattr(
        session_module.lib,
        "FMLanguageModelSessionCreateFromSystemLanguageModel",
        lambda *args, **kwargs: ctypes.c_void_p(1),
    )
    monkeypatch.setattr(session_module, "Transcript", lambda ptr: None)
    session = fm.LanguageModelSession()
    yield session
    # Fake pointer; keep a later GC pass from releasing it through the real FMRelease.
    session._ptr = None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kwargs",
    [
        {"generating": Reply},
        {"schema": Reply.generation_schema()},
        {"json_schema": Reply.generation_schema().to_dict()},
    ],
    ids=["generating", "schema", "json_schema"],
)
async def test_respond_forwards_options(session, monkeypatch, kwargs):
    seen = []

    async def capture(prompt, schema, options=None):
        seen.append(options)
        raise _Captured

    monkeypatch.setattr(session, "_respond_with_schema", capture)
    monkeypatch.setattr(session, "_respond_with_schema_from_json", capture)

    options = fm.GenerationOptions(
        sampling=fm.SamplingMode.greedy(), temperature=None, maximum_response_tokens=64
    )
    with pytest.raises(_Captured):
        await session.respond("Say hi", options=options, **kwargs)
    assert seen == [options]
