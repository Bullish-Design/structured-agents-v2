# llama.cpp MTP concurrency sweep

One round was run at each client concurrency level against the live GPU-0 MTP
endpoint. Each request used the same profiler prompts, disabled Gemma thinking,
and generated exactly 128 completion tokens. The server is configured with one
generation slot, so higher client counts queue work rather than add parallel
decode slots.

| Clients | Requests | Completion tokens | Wall time | Aggregate tok/s | Mean latency | P95 latency |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1 | 128 | 3.053 s | 41.924 | 3.052 s | 3.052 s |
| 2 | 2 | 256 | 5.169 s | 49.527 | 3.646 s | 5.168 s |
| 3 | 3 | 384 | 7.944 s | 48.339 | 5.058 s | 7.942 s |
| 4 | 4 | 512 | 11.072 s | 46.244 | 6.829 s | 11.069 s |
| 5 | 5 | 640 | 13.816 s | 46.324 | 8.044 s | 13.811 s |

The raw measurements are in `concurrency.csv`. During the final queued requests,
llama.cpp reported MTP acceptance rates from 36.9% to 44.5% and generation rates
from 43.2 to 48.4 tokens/s per active decode.
