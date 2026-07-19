"""S3: empirical introspection of pydantic-ai 2.11.0 surface. No network calls."""

from __future__ import annotations

import inspect

import pydantic_ai
from pydantic import BaseModel
from pydantic_ai import Agent, NativeOutput
from pydantic_ai.agent import AgentRunResult
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

print("pydantic_ai version:", pydantic_ai.__version__)

# 1) NativeOutput import
print("NativeOutput import: OK ->", NativeOutput)
print("pydantic_ai has NativeOutput attr:", hasattr(pydantic_ai, "NativeOutput"))

# 2) AgentRunResult.usage is a property
usage_attr = inspect.getattr_static(AgentRunResult, "usage")
print("AgentRunResult.usage isinstance property:", isinstance(usage_attr, property))
print("AgentRunResult.usage attr repr:", repr(usage_attr))
print("usage attr type:", type(usage_attr).__name__)
if isinstance(usage_attr, property) and usage_attr.fget is not None:
    print("usage.fget signature:", inspect.signature(usage_attr.fget))
    print("usage.fget return annotation:", inspect.signature(usage_attr.fget).return_annotation)

# probe: is usage ALSO callable in this version (back-compat)?  -> a property is not callable
print("usage_attr callable:", callable(usage_attr))

# 3) output attribute type — it is a generic dataclass FIELD, not a property
print("AgentRunResult is dataclass:", hasattr(AgentRunResult, "__dataclass_fields__"))
fields = getattr(AgentRunResult, "__dataclass_fields__", {})
print("AgentRunResult 'output' field type:", fields["output"].type if "output" in fields else "<absent>")
print("AgentRunResult MRO:", [c.__name__ for c in AgentRunResult.__mro__])


# 4) Agent construction with output_type=str and output_type=NativeOutput(model)
class SomeModel(BaseModel):
    x: int


provider = OpenAIProvider(base_url="http://localhost:1/v1", api_key="dummy")
model = OpenAIChatModel("dummy-model", provider=provider)

try:
    a_str = Agent(model, output_type=str)
    print("Agent(output_type=str): OK ->", type(a_str).__name__)
except Exception as e:  # noqa: BLE001
    print("Agent(output_type=str): FAIL ->", repr(e))

try:
    a_native = Agent(model, output_type=NativeOutput(SomeModel))
    print("Agent(output_type=NativeOutput(SomeModel)): OK ->", type(a_native).__name__)
except Exception as e:  # noqa: BLE001
    print("Agent(output_type=NativeOutput(SomeModel)): FAIL ->", repr(e))
