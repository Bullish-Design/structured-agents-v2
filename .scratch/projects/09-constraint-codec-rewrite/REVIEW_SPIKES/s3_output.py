"""S3b: how is AgentRunResult.output defined?"""

from __future__ import annotations

import inspect

from pydantic_ai.agent import AgentRunResult

print("is dataclass:", hasattr(AgentRunResult, "__dataclass_fields__"))
if hasattr(AgentRunResult, "__dataclass_fields__"):
    for name, fld in AgentRunResult.__dataclass_fields__.items():
        print(f"  field {name!r}: type={fld.type!r}")

print("class __annotations__:", getattr(AgentRunResult, "__annotations__", {}))
print("MRO:", [c.__name__ for c in AgentRunResult.__mro__])

# Is 'output' a property somewhere in MRO?
for klass in AgentRunResult.__mro__:
    if "output" in vars(klass):
        out = vars(klass)["output"]
        print(f"'output' found in {klass.__name__}: type={type(out).__name__}")
    if "usage" in vars(klass):
        print(f"'usage' found in {klass.__name__}: type={type(vars(klass)['usage']).__name__}")

# Is output generic? get_type_hints
try:
    hints = inspect.get_annotations(AgentRunResult, eval_str=False)
    print("get_annotations:", hints)
except Exception as e:  # noqa: BLE001
    print("get_annotations FAIL:", repr(e))
