import argparse
import json
import logging
import os

from dotenv import load_dotenv
load_dotenv()  # read API key from .env
parser = argparse.ArgumentParser(description="Configure dataset and model parameters.")
parser.add_argument("--data", type=str, default="locomo", help="Dataset name, e.g., AR / LM / locomo")
parser.add_argument("--model", type=str, default="gemini", help="Model name, e.g., gemini / claude / gpt4o / qwen")
parser.add_argument("--file", type=str, default="0", help="Run/experiment tag appended to result filenames")
parser.add_argument("--sample", type=int, default=None, help="Sample id to run (e.g. 42). Omit to run all samples.")
parser.add_argument("--qu", type=int, default=0, help="Dataset name, e.g., AR / LM / locomo")
parser.add_argument("--re_model", type=str, default=None, help="Dataset name, e.g., AR / LM / locomo")
parser.add_argument("--ca", type=int, default=1, help="LM category index: 0=multi-session,1=single-session-user,2=temporal-reasoning,3=single-session-preference,4=knowledge-update,5=single-session-assistant")
parser.add_argument("--lm_batch", type=int, default=1, help="LM: sessions merged per rewrite call. 1=per-session (key=session_i, compatible with existing files/per-session readers); >1=merged (key=session_first-session_last)")
parser.add_argument("--maxq", type=int, default=None, help="Limit the number of questions per sample (for cheap pilot runs). Omit to run all.")

# parse_known_args (not parse_args) so importing this module under a foreign argv
# (pytest, notebooks, helper scripts) does not crash on unrecognized arguments.
args, _ = parser.parse_known_args()


OPENROUTER_URL = "https://openrouter.ai/api/v1"

# LLM endpoint. Defaults to OpenRouter; point LLM_BASE_URL / LLM_API_KEY at any
# OpenAI-compatible service (DeepSeek, a relay, ...) without touching the code.
LLM_BASE_URL = (os.getenv("LLM_BASE_URL") or OPENROUTER_URL).rstrip("/")
LLM_API_KEY = os.getenv("LLM_API_KEY") or os.getenv("OPENROUTER_API_KEY") or ""

# Short CLI name -> provider model id. Names not listed here pass through verbatim,
# so `--model deepseek-flash` works without editing this table.
MODEL_ALIASES = {
    "gpt4.1mini": "openai/gpt-4.1-mini",
    "gpt4omini": "openai/gpt-4o-mini-2024-07-18",
    "claude": "anthropic/claude-sonnet-4.5",
    "gpt4o": "openai/gpt-4o",
    "claude3.5": "anthropic/claude-3.5-haiku",
    "qwen": "qwen/qwen3-max",
    "gemini": "google/gemini-2.5-flash",
    "deepseek": os.getenv("MODEL_DEEPSEEK", "deepseek-flash"),
    "ds": os.getenv("MODEL_DEEPSEEK", "deepseek-flash"),
}


def resolve_model(name: str) -> str:
    """CLI short name -> provider model id. `MODEL_<NAME>` env var wins; unknown names pass through."""
    override = os.getenv("MODEL_" + name.upper().replace(".", "_").replace("-", "_"))
    return override or MODEL_ALIASES.get(name, name)


MODEL = resolve_model(args.model)
CHOOSE_MODEL = MODEL
MODEL_NAME = args.model  # short name (gemini/claude/...), used by the LM temporal method answer_question_with_time_lm
RE_MODEL = resolve_model(args.re_model) if args.re_model else MODEL
API_KEY = LLM_API_KEY  # backwards-compatible alias
MODEL_SORT = MODEL
K1=80                 # coarse retrieval breadth (embedding similarity)
K2=20                 # fine retrieval breadth (LLM re-ranking)
TAG_MAX=15            # select_key_tag: re-rank a key's tags only when it has more than this many
TAG_LIMIT=10         # select_key_tag: keep at most this many tags after re-ranking
TIME_EVENT_LIMIT=50  # answer_question_with_time: dense-time fast path threshold (locomo)
TOPIC_K=8            # select_topic: number of topic candidates
RERANK_LIMIT=20      # event_by_tag: re-rank events only when more than this many match
MAX_ROUNDS=8         # tool-calling loop: max assistant rounds
MAX_TOOL_CALLS=50    # tool-calling loop: safety cap on total tool calls
sample_id = args.sample
maxq = args.maxq  # optional cap on questions per sample (cheap pilot runs)
qu = args.qu
ca = args.ca
LM_REWRITE_BATCH = args.lm_batch  # sessions merged per LM rewrite call

# ------------------------------------------------------ provider quirks (env)
# Comma-separated chat-request keys to drop. DeepSeek does not accept
# `seed` / `parallel_tool_calls`; OpenRouter accepts both, hence the empty default.
LLM_DROP_PARAMS = {p.strip() for p in os.getenv("LLM_DROP_PARAMS", "").split(",") if p.strip()}
# JSON merged into every chat request body. DeepSeek runs thinking mode by default
# (its CoT is billed as output tokens), so set:
#   LLM_EXTRA_BODY={"thinking":{"type":"disabled"}}
try:
    LLM_EXTRA_BODY = json.loads(os.getenv("LLM_EXTRA_BODY", "") or "{}")
except json.JSONDecodeError as exc:
    raise SystemExit(f"LLM_EXTRA_BODY is not valid JSON: {exc}")
if not isinstance(LLM_EXTRA_BODY, dict):
    LLM_EXTRA_BODY = {}

dataset = args.data
DATASET = dataset
datapath = f"data/dataset_{dataset}.json"
ADDITIONAL_TK = f"_{args.model}"#"_gpt4o-mini"
ADDITIONAL_EM = f"_{args.model}"#
ADDITIONAL_RE = f"_{args.model}_{args.file}" #"_gpt4o-mini"
base_dir_t = f"data/{{dataset}}/rewrite{ADDITIONAL_TK}/"
base_dir_k = f"data/{{dataset}}/keyword{ADDITIONAL_TK}/"
base_dir_emb = f"data/{{dataset}}/embedding/gpt{ADDITIONAL_EM}/"
# auto-create all output dirs so a fresh checkout runs without manual mkdir
os.makedirs(base_dir_t.format(dataset=dataset), exist_ok=True)      # data/<ds>/rewrite_<model>/
os.makedirs(base_dir_k.format(dataset=dataset), exist_ok=True)      # data/<ds>/keyword_<model>/
os.makedirs(base_dir_emb.format(dataset=dataset), exist_ok=True)    # data/<ds>/embedding/gpt_<model>/
os.makedirs(f"result/{dataset}", exist_ok=True)                    # prediction outputs
os.makedirs(f"log/{dataset}", exist_ok=True)                       # logs (also creates log/ for the run-level handler)

rewrite_template = f"data/{{dataset}}/rewrite{ADDITIONAL_TK}/{{sample_id}}_rewrite.json"
keyword_template = f"data/{{dataset}}/keyword{ADDITIONAL_TK}/{{sample_id}}_keyword.json"
embedding_template = f"data/{{dataset}}/embedding/gpt{ADDITIONAL_EM}/{{sample_id}}_embedding.pkl"
result_template = f"result/{{dataset}}/{{sample_id}}_result{ADDITIONAL_RE}.jsonl"

# per-API-call token accounting, appended as JSONL so a run's cost can be audited
USAGE_LOG = os.getenv("USAGE_LOG", f"log/{dataset}/token_usage.jsonl")
