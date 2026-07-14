# PydanticAI `RunUsage` compatibility breaks successful structured runs

## Summary

Lodestar's in-process OpenAI-compatible integration proof found a runtime
compatibility defect in the pinned `structured-agents-v2` implementation. A
valid JSON-schema response is parsed successfully, but `StructuredAgent.run()`
then raises while constructing `AgentResult`. The failure occurs before the
caller receives the normalized output.

The affected code is `src/structured_agents_v2/agent.py`:

```python
return AgentResult(output=raw.output, usage=raw.usage(), request_body=request_body, raw=raw)
```

With the PydanticAI version resolved in Lodestar's reviewed environment,
`raw.usage` is a `RunUsage` value, not a callable method. Calling it raises:

```text
TypeError: 'RunUsage' object is not callable
```

This makes otherwise successful model calls fail after their output has already
been validated.

## Reproduction

Observed on 2026-07-14 with the reviewed `structured-agents-v2` commit
`5ca614249910d76266066df81f03c9d68df1a413`.

- Lodestar constructs exactly `Backend(capture=False)`, builds one fixed
  Pydantic JSON-schema profile, then awaits `agent.run(prompt)` without kwargs.
- Transport is an in-process `httpx.ASGITransport`; no network provider is
  involved.
- The server returns a standard OpenAI-compatible chat completion with valid
  assistant content: `{"kind":"no_action"}`.

```text
StructuredAgent.run(prompt)
  -> PydanticAI validates ProjectActionOutput(kind="no_action")
  -> StructuredAgent._result(raw)
  -> raw.usage()
  -> TypeError: 'RunUsage' object is not callable
```

The response reaches the transport and uses JSON-schema response format; this is
not a connection, schema, capture, prompt, or model failure.

## Impact

- Successful `StructuredAgent.run()` calls fail in this environment.
- Safe applications convert the exception to a fixed provider failure, so no
  action is admitted.
- `capture=False` is not a workaround: `_result()` calls `raw.usage()`
  regardless of capture status.
- Privacy-sensitive callers cannot work around this by reading `raw` themselves
  when their boundary intentionally discards native provider data.

## Recommended fix

1. Make `_result()` compatible with both usage API shapes:

   ```python
   usage = raw.usage
   if callable(usage):
       usage = usage()
   return AgentResult(output=raw.output, usage=usage, request_body=request_body, raw=raw)
   ```

   Prefer the documented/current API directly once the supported PydanticAI
   version range is explicit.

2. Add a regression test using the resolved PydanticAI version and an in-process
   OpenAI-compatible transport. Assert valid JSON-schema output produces
   `AgentResult.output` without raising.

3. Test both `capture=False` and `capture=True`; retain request-body assertions
   for capture, but do not make result construction depend on it.

4. Pin or constrain the compatible PydanticAI version in package metadata and
   lockfiles, and document the supported range.

5. Consider whether native `raw` and `usage` should remain default result fields.
   At minimum, document that privacy-sensitive callers should discard them
   immediately and never need them to obtain validated output.

## Acceptance criteria

- Valid object-schema JSON completion completes without exception.
- `AgentResult.output` is the validated Pydantic model.
- `capture=False` has no captured request body.
- `capture=True` retains existing request capture behavior.
- The regression test runs fully in process.
- The lock resolves a PydanticAI version covered by that test.

## Lodestar mitigation

Lodestar catches this provider-origin exception and returns only the fixed
classification `provider_unavailable`; it does not persist or log exception text
or native result data. This preserves privacy and authority boundaries, but
structured tasks cannot complete successfully until the library defect is fixed.
