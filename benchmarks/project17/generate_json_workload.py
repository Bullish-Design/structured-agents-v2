"""Generate the deterministic Project 17 structured-output workload corpus."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

CORPUS_VERSION = "project17-json-workload-v1"
GENERATION_SEED = 17001
ROOT = Path(__file__).parent
REGISTRY = "schema_registry_v1.json"

TASKS = (
    ("document_extraction", "document_fields", "Extract the key facts from this short document"),
    ("ticket_triage", "ticket_triage", "Triage this customer-support ticket"),
    ("tool_selection", "tool_selection", "Select the single best tool and arguments"),
    ("routing", "routing", "Route this request to the appropriate queue"),
    ("calendar_normalization", "calendar_event", "Normalize this calendar request"),
    ("contact_normalization", "contact_address", "Normalize this contact and mailing address"),
    ("invoice_extraction", "invoice_order", "Extract invoice or order fields"),
    ("support_action_plan", "support_action", "Create a concise support action plan"),
    ("content_moderation", "moderation", "Classify this content for moderation"),
    ("ecommerce_attributes", "product_attributes", "Extract ecommerce product attributes"),
    ("lead_qualification", "lead_qualification", "Qualify this sales lead"),
    ("agent_handoff", "agent_handoff", "Prepare a concise handoff payload for the next agent"),
)

DETAILS = (
    "Reference: ACME-{n:04d}. Customer reports a delayed shipment and asks for an update.",
    "Record {n:04d}: Jordan Lee needs a response before Friday at 15:00 America/New_York.",
    "Order P17-{n:05d} contains two items; the requester wants the next safe action.",
    "Message {n:04d}: keep the answer factual, concise, and limited to the supplied text.",
)


def entry(index: int) -> dict[str, object]:
    category, schema_id, instruction = TASKS[index % len(TASKS)]
    detail = DETAILS[(index // len(TASKS)) % len(DETAILS)].format(n=index + GENERATION_SEED)
    return {
        "id": f"p17-{index + 1:04d}",
        "category": category,
        "prompt": f"{instruction}. Return only JSON matching schema '{schema_id}'.\n\n{detail}",
        "schema_id": schema_id,
        "schema_ref": f"{REGISTRY}#{schema_id}",
        "max_tokens": 64,
        "input_metadata": {
            "utf8_bytes": None,
            "utf8_chars": None,
            "prompt_tokens": None,
            "tokenizer": "runtime-filled-from-exact-gguf",
        },
    }


def write_jsonl(path: Path, entries: list[dict[str, object]]) -> str:
    payload = "".join(
        json.dumps(item, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n" for item in entries
    )
    path.write_text(payload, encoding="utf-8")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def main() -> None:
    entries = [entry(index) for index in range(1000)]
    digest_100 = write_jsonl(ROOT / "json_workload_100.jsonl", entries[:100])
    digest_1000 = write_jsonl(ROOT / "json_workload_1000.jsonl", entries)
    manifest = {
        "corpus_version": CORPUS_VERSION,
        "generation_seed": GENERATION_SEED,
        "schema_registry": REGISTRY,
        "files": {
            "json_workload_100.jsonl": {"entries": 100, "sha256": digest_100},
            "json_workload_1000.jsonl": {"entries": 1000, "sha256": digest_1000},
        },
        "prefix_contract": (
            "The first 100 UTF-8 lines of json_workload_1000.jsonl are byte-for-byte identical to "
            "json_workload_100.jsonl."
        ),
        "categories": [task[0] for task in TASKS],
    }
    (ROOT / "json_workload_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
