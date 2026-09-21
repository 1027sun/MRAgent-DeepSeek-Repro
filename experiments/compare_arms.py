"""Comparison arms for the MRAgent ablation.

ARM=passive     : identical memory + identical initial retrieval as MRAgent, but the
                  tool-calling loop is replaced by ONE answer call. This isolates the
                  contribution of multi-turn active reconstruction.
ARM=fullcontext : no retrieval at all — the whole conversation goes into the prompt.
                  This is the "no memory system" reference point.

Both reuse the memory cache built by the MRAgent run, so construction costs nothing.
Results are written in the same JSONL format, so eval/evaluate_reasoning.py can read them.

Env knobs: ARM, FILE_TAG, MAXQ, SAMPLE
"""
import json
import os
import sys

# run from anywhere: behave as if launched from the repo root
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO_ROOT)
sys.path.insert(0, REPO_ROOT)

ARM = os.environ.get("ARM", "passive").strip().lower()
SAMPLE = os.environ.get("SAMPLE", "26")
FILE_TAG = os.environ.get("FILE_TAG", ARM)
MAXQ = os.environ.get("MAXQ", "").strip()

assert ARM in ("passive", "fullcontext"), f"unknown ARM={ARM!r}"

sys.argv = ["run.py", "--data", "locomo", "--model", "deepseek",
            "--file", FILE_TAG, "--sample", SAMPLE]
if MAXQ:
    sys.argv += ["--maxq", MAXQ]

from common import config          # noqa: E402
from agent.agent import Agent      # noqa: E402
from data.get_data import get_data  # noqa: E402

# --------------------------------------------------------------------------- helpers
_SESSION_RE = None


def _build_full_context(sample_id: str) -> str:
    """Raw session texts for one conversation, as fed to the full-context arm."""
    conv_list, _, _, _ = get_data(config.dataset, config.datapath)
    sessions = conv_list[sample_id]
    parts = []
    for key in sorted(sessions, key=lambda k: int(k.split("_")[1])):
        parts.append(f"### Session {key}\n{sessions[key]}")
    return "\n\n".join(parts)


def _answer_json(llm, system: str, user: str):
    """One-shot answer call returning (answer, supports)."""
    out = llm.chat_text(
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
        model=config.RE_MODEL,
    )
    if not isinstance(out, dict):
        return "no information available", []
    return out.get("answer"), out.get("supports") or []


# ------------------------------------------------------------------- ARM: passive
def _passive_chat_with_tools(self, system_prompt, user_obj, category):
    """Stand-in for Agent._chat_with_tools: same context, no tools, one shot.

    The context handed in here (key_sentences / keys_candidates / key_tag_sentences /
    similar_topic, or the temporal fast path) is byte-for-byte what the MRAgent arm
    starts its tool loop with.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": json.dumps(user_obj, ensure_ascii=False)},
        {"role": "user",
         "content": "Answer now, using only the material above. You have no tools. "
                    'Reply with a single JSON object: {"answer": "...", "supports": []}.'},
    ]
    out = self.llm.chat_text(messages=messages, model=config.RE_MODEL)
    if not isinstance(out, dict):
        return "no information available", []
    return out.get("answer"), out.get("supports") or []


# --------------------------------------------------------------- ARM: fullcontext
_FULL_TEXT = None


def _fullcontext_answer_question(self, question, category=0, question_emb=None,
                                 override_question_time=None, lm_current_date=None):
    """Replace the whole retrieve-then-reason pipeline with one giant prompt."""
    global _FULL_TEXT
    if _FULL_TEXT is None:
        _FULL_TEXT = _build_full_context(f"conv-{SAMPLE}")
        print(f"[fullcontext] conversation text: {len(_FULL_TEXT)} chars "
              f"(~{len(_FULL_TEXT)//4} tokens)", flush=True)

    system = (
        "You are answering questions about a long multi-session conversation between two "
        "people. Use ONLY the conversation provided. Reply with one JSON object and "
        'nothing else: {"answer": "<short answer>", "supports": []}. '
        "Give the shortest answer that is still specific; no explanation. "
        "If the question asks when something happened, answer with an absolute date. "
        "If the conversation does not mention it, answer exactly 'Not mentioned'."
    )
    user = f"CONVERSATION:\n{_FULL_TEXT}\n\nQUESTION: {question}"
    return _answer_json(self.llm, system, user)


Agent._chat_with_tools = _passive_chat_with_tools
if ARM == "fullcontext":
    Agent.answer_question = _fullcontext_answer_question

import run  # noqa: E402  (imported after patching)

if __name__ == "__main__":
    print(f"ARM={ARM}  sample=conv-{SAMPLE}  file_tag={FILE_TAG}  maxq={MAXQ or 'all'}",
          flush=True)
    run.main()
    print("done", flush=True)
