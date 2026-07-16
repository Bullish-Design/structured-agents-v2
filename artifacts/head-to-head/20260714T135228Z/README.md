# Final head-to-head essay run

Both engines received the same request for an exactly 1,000-word, five-paragraph essay.
They each returned five visible paragraphs but undershot the requested word count; the
measured counts are retained rather than padded or altered after generation.

| Engine | Visible words | Paragraphs | Completion tokens | Wall time | Completion tokens/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| llama.cpp | 821 | 5 | 957 | 28.173 s | 33.969 |
| vLLM | 912 | 5 | 1,050 | 116.510 s | 9.012 |

`*-essay.txt` contains each visible response, `*-thinking.txt` contains any extracted
reasoning content, and `*-response.json` preserves each raw API response. No final-run
thinking tokens were generated: llama.cpp used `enable_thinking: false`; vLLM returned
no reasoning field. `performance.json` contains the full request, token counts, timings,
and the llama.cpp server timing breakdown.
