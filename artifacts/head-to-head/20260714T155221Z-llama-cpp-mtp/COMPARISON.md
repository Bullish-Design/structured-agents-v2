# Five-paragraph essay comparison

Every run used the same exact-1,000-word essay prompt and disabled Gemma thinking
for a visible-output comparison. The models undershot the requested word count; the
responses are preserved exactly as generated.

| Engine/run | Visible words | Paragraphs | Completion tokens | Wall time | End-to-end tok/s | Server generation tok/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| llama.cpp, no MTP | 821 | 5 | 957 | 28.173 s | 33.969 | 34.436 |
| llama.cpp, MTP | 693 | 5 | 828 | 20.622 s | 40.152 | 41.003 |
| vLLM, no MTP | 912 | 5 | 1,050 | 116.510 s | 9.012 | unavailable |

The MTP run generated 1,376 draft tokens and accepted 485 (35.25%). Relative to
the non-MTP llama.cpp run, its server generation rate improved by 19.1% and its
end-to-end completion rate improved by 18.2%. The differing completion lengths mean
wall times alone are not directly comparable; token rates are the useful metric.

Raw artifacts for the MTP run are in this directory. The two baseline records are:

- `../20260714T135228Z/performance.json` for llama.cpp without MTP and vLLM.
- `performance.json` for the MTP run, including llama.cpp draft acceptance counters.
