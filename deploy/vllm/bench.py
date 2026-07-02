#!/usr/bin/env python3
"""Batched-inference throughput profile for a running structured-agents vLLM endpoint.

Two workloads, both with DIVERSE prompts (no shared prefix -> no prefix-cache
inflation) and natural output lengths:

  json : response_format=json_schema command objects — the real constrained-agent
         path. Reports schema-validity per wave.
  text : free-form generation (longer outputs).

For each concurrency level it fires C distinct requests simultaneously and reports
aggregate output tok/s, the scaling factor vs single-stream, and latency spread.
This is the live reproduction of the "batched against one server" design premise
(cf. AgentSet.run_batch). Stdlib only — no deps, GPU-free on the client side.

Config comes from the environment (defaults mirror deploy/vllm/.env and verify.sh):
  LLM_BASE_URL   default http://localhost:8000/v1   (e.g. http://tower:8000/v1)
  LLM_API_KEY    default $VLLM_API_KEY               (sent as Bearer when non-empty)
  LLM_MODEL      default base                        (SERVED_MODEL_NAME)
  BENCH_LEVELS   default 1,4,8,16,32                 (comma-separated concurrencies)
  BENCH_TEXT_TOKENS  default 256                     (max_tokens for the text workload)

Usage:
  LLM_BASE_URL=http://tower:8000/v1 LLM_API_KEY=$KEY deploy/vllm/bench.py
"""
import json, os, time, urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor

BASE = os.environ.get("LLM_BASE_URL", "http://localhost:8000/v1").rstrip("/") + "/chat/completions"
KEY = os.environ.get("LLM_API_KEY", os.environ.get("VLLM_API_KEY", ""))
MODEL = os.environ.get("LLM_MODEL", "base")
LEVELS = [int(x) for x in os.environ.get("BENCH_LEVELS", "1,4,8,16,32").split(",") if x.strip()]
TEXT_TOKENS = int(os.environ.get("BENCH_TEXT_TOKENS", "256"))

HEADERS = {"Content-Type": "application/json"}
if KEY:
    HEADERS["Authorization"] = f"Bearer {KEY}"

CMD_TASKS = [
    "create a file called report.txt", "delete the folder named temp",
    "rename config.yaml to config.old.yaml", "move image.png into assets",
    "copy notes.md to backup/notes.md", "search the codebase for parse_args",
    "commit all changes with message fix login bug", "install the package requests",
    "run the test suite in the tests directory", "open the file src/main.py",
    "list all python files under src", "remove the file debug.log",
    "create a directory called build", "checkout the branch feature/api",
    "push the current branch to origin", "pull the latest changes from main",
    "add the file README.md to staging", "grep for TODO in the lib folder",
    "download the dataset from the data bucket", "compress the logs folder to logs.zip",
    "extract archive release.tar.gz", "change permissions of deploy.sh to executable",
    "start the web server on port 8080", "stop the running database container",
    "restart the nginx service", "tail the last 50 lines of app.log",
    "count the lines in main.c", "format the file utils.py",
    "lint the src directory", "build the docker image tagged app:latest",
    "tag the current commit as v1.2.0", "revert the last commit",
    "create a symlink from bin/app to build/app", "sync the folder photos to remote backup",
]
TEXT_PROMPTS = [
    "Explain how TCP congestion control works.",
    "Describe the water cycle and its main stages.",
    "Write a short story about a lighthouse keeper who finds a message in a bottle.",
    "Summarize the main causes of the French Revolution.",
    "How does a lithium-ion battery store and release energy?",
    "Explain the difference between supervised and unsupervised learning.",
    "Describe how vaccines train the immune system.",
    "What makes sourdough bread rise? Explain the chemistry.",
    "Explain how GPS determines your location.",
    "Describe the life cycle of a star like the Sun.",
    "Write a haiku about autumn, then explain its imagery.",
    "How does a blockchain achieve consensus without a central authority?",
    "Explain the greenhouse effect in simple terms.",
    "Describe how the human eye focuses light.",
    "What is the CAP theorem and why does it matter for databases?",
    "Explain how noise-cancelling headphones work.",
    "Summarize the plot structure of a classic three-act story.",
    "How do plants convert sunlight into chemical energy?",
    "Explain the concept of compound interest with an example.",
    "Describe how a jet engine produces thrust.",
    "What causes the northern lights?",
    "Explain how DNS resolves a domain name to an IP address.",
    "Describe the process of natural selection.",
    "How does a refrigerator move heat out of its interior?",
    "Explain what a hash function is and where it is used.",
    "Describe how tides are caused by the moon and sun.",
    "What is the difference between latency and bandwidth?",
    "Explain how a suspension bridge carries its load.",
    "Describe how mRNA vaccines differ from traditional ones.",
    "How does a transistor act as a switch?",
    "Explain the traveling salesman problem and why it is hard.",
    "Describe how coffee extraction works during brewing.",
    "What is entropy in the context of thermodynamics?",
    "Explain how a search engine ranks web pages.",
]
CMD_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string"},
        "target": {"type": "string"},
        "args": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["action", "target"],
    "additionalProperties": False,
}


def req_json(task):
    return {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "You translate a user request into a single command object."},
            {"role": "user", "content": task},
        ],
        "response_format": {"type": "json_schema", "json_schema": {"name": "Command", "strict": True, "schema": CMD_SCHEMA}},
        "max_tokens": 128, "temperature": 0.0,
    }


def req_text(prompt):
    return {"model": MODEL, "messages": [{"role": "user", "content": prompt}], "max_tokens": TEXT_TOKENS, "temperature": 0.7}


def call(payload):
    body = json.dumps(payload).encode()
    r = urllib.request.Request(BASE, data=body, headers=HEADERS, method="POST")
    t0 = time.time()
    with urllib.request.urlopen(r, timeout=300) as resp:
        d = json.loads(resp.read())
    dt = time.time() - t0
    return d["usage"]["completion_tokens"], dt, d["choices"][0]["message"]["content"]


def valid_json(s):
    try:
        json.loads(s); return True
    except Exception:
        return False


def run_workload(name, build, pool):
    print(f"\n=== {name} (diverse prompts, natural EOS) ===")
    print(f"{'conc':>5} {'wall_s':>8} {'reqs':>5} {'tot_tok':>8} {'mean_tok':>9} {'tok/s':>9} {'x1':>6} {'avg_lat':>8} {'p95_lat':>8} {'ok':>6}")
    base_tps = None
    offset = 0
    for c in LEVELS:
        prompts = [pool[(offset + i) % len(pool)] for i in range(c)]
        offset += c
        t0 = time.time()
        try:
            with ThreadPoolExecutor(max_workers=c) as ex:
                res = list(ex.map(lambda p: call(build(p)), prompts))
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
            print(f"{c:>5}  ERROR: {e}")
            continue
        wall = time.time() - t0
        toks = sum(r[0] for r in res)
        lats = sorted(r[1] for r in res)
        p95 = lats[min(len(lats) - 1, int(0.95 * len(lats)))]
        ok = sum(1 for r in res if valid_json(r[2])) if name == "json" else c
        tps = toks / wall
        base_tps = base_tps or tps
        print(f"{c:>5} {wall:>8.2f} {c:>5} {toks:>8} {toks/c:>9.1f} {tps:>9.1f} {tps/base_tps:>5.2f}x {sum(lats)/len(lats):>8.2f} {p95:>8.2f} {ok:>4}/{c}")


if __name__ == "__main__":
    print(f"endpoint: {BASE}   model: {MODEL}   auth: {'yes' if KEY else 'no'}   levels: {LEVELS}")
    try:
        call(req_text("Say hello."))  # warmup: first request pays compile/cache costs
    except Exception as e:
        print("warmup failed:", e)
    run_workload("json", req_json, CMD_TASKS)
    run_workload("text", req_text, TEXT_PROMPTS)
