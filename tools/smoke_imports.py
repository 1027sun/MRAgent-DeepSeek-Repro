"""Check that every module in the MRAgent repo imports on this machine."""
import sys
import time

MODS = [
    "common.config",
    "common.utils",
    "common.logging_utils",
    "prompts.prompts",
    "prompts.schema",
    "memory.system",
    "memory.controller",
    "llm.embeddings",
    "llm.rag_utils",
    "llm.controller",
    "agent.tools",
    "agent.agent",
    "data.get_data",
    "data.embed_rewrite",
    "eval.judge",
    "eval.evaluation",
    "eval.evaluate_reasoning",
    "run",
]

if __name__ == "__main__":
    t0 = time.time()
    fails = 0
    for m in MODS:
        try:
            __import__(m)
            print("OK   ", m)
        except Exception as e:  # noqa: BLE001
            fails += 1
            print("FAIL ", m, "->", type(e).__name__, e)

    import torch

    print(f"\ntorch {torch.__version__} | cuda available: {torch.cuda.is_available()}")
    print(f"python {sys.version.split()[0]}")
    print(f"{fails} failures, {time.time() - t0:.1f}s")
