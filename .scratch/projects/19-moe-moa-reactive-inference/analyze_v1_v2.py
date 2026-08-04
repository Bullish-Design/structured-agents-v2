#!/usr/bin/env python
"""Corrected per-level analysis v1 vs v2 (uses scd.score_line, format-agnostic)."""
import json

import spike_context_distill as scd

RES = {
    "v1 (old CONTEXT)": "spike_distill_template_result.json",
    "v2 (teacher-fixed)": "spike_distill_template_v2_result.json",
    "v3 (TRACE x3)": "spike_distill_template_v3_result.json",
}

for label, path in RES.items():
    d = json.load(open(path))
    print(f"===== {label} =====")
    for name, r in d["report"].items():
        tot = round(r["task"]["level_correct"] * 24)
        by_lvl = {lvl: [0, 0] for lvl in ("TRACE", "WARN", "FATAL")}
        for s in r["samples"]:
            sc = scd.score_line(s["out_stripped"], s["expected"])
            by_lvl[s["expected"]][1] += 1
            by_lvl[s["expected"]][0] += int(sc["level_correct"])
        tr, wf = (by_lvl["TRACE"][0], by_lvl["WARN"][0])
        wf_tot = by_lvl["WARN"][1] + by_lvl["FATAL"][1]
        print(f"  {name}: total={tot}/24 TRACE={tr}/8 "
              f"WARN+FATAL={wf}/{wf_tot} format={r['task']['format_ok']} "
              f"kl={r['mean_token_kl_vs_ceiling']}")
        if name == "student_lora_noctx":
            for s in r["samples"]:
                sc = scd.score_line(s["out_stripped"], s["expected"])
                print(f"     exp={s['expected']:5s} "
                      f"correct={sc['level_correct']}  {s['out_stripped'][:72]!r}")
    print(f"  verdict: gap={d['verdict']['level_correct_gap_recovered']}")
