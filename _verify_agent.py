"""Temporary: inspect the ReAct agent trace. Safe to delete."""
import re

from agent import agent_executor
from rag import query_rag

PHONE_PAT = re.compile(r"\d[\d\s\-().]{5,}\d")

QUESTIONS = [
    ("TARGET phone number", "What exact phone number should I call for the University IT Service Desk?"),
    ("in-corpus lockout", "How long should I wait before signing in again after my account is locked?"),
    ("in-corpus storage", "Where should large research datasets be stored?"),
]

for label, q in QUESTIONS:
    print("=" * 78)
    print(label, "|", q)
    try:
        res = agent_executor.invoke({"input": q})
    except Exception as exc:
        print("  agent raised:", type(exc).__name__, str(exc)[:200])
        continue
    steps = res.get("intermediate_steps") or []
    print("  agent tool calls:", len(steps))
    for action, observation in steps:
        print("    -> Action:", getattr(action, "tool", "?"))
        print("       Action Input:", repr(getattr(action, "tool_input", "?"))[:160])
        print("       Observation:", repr(str(observation))[:200])
    out = (res.get("output") or "").strip()
    print("  final output:", repr(out)[:300])
    print("  invents number:", bool(PHONE_PAT.search(out)))
    print("  query_rag would give:", repr(query_rag(q))[:200])
    print()
