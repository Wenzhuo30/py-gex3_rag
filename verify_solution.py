"""Verification script for the RAG exercise (not part of the required solution).

Run it after any change to agent.py, rag_tool.py or rag.py:

    python verify_solution.py

It checks the graded requirement -- the agent must say it does not know the IT
Service Desk phone number, without inventing one or adding other instructions --
and confirms the rest of the pipeline still answers supported questions.
"""

from __future__ import annotations

import re

from agent import _is_grounded, run_agent
from policy_loader import load_policy_documents
from rag import _mentions_phone_number, _unsupported_digits, format_docs, query_rag, retriever
from rag_tool import policy_search_tool

PHONE_QUESTION = "What exact phone number should I call for the University IT Service Desk?"

# Wording that would count as "another instruction" rather than a plain refusal.
INSTRUCTION_RE = re.compile(
    r"(?i)\b(?:visit|contact|call|check|try|see|refer|go to|website|portal|email them|ask)\b"
)
EXTRA_INSTRUCTION_RE = INSTRUCTION_RE

results: list[tuple[bool, str]] = []


def check(ok: bool, label: str) -> None:
    results.append((bool(ok), label))
    print(f"[{'PASS' if ok else 'FAIL'}] {label}")


print("=" * 72)
print("1. The graded case: the Service Desk phone number")
print("=" * 72)
answer = run_agent(PHONE_QUESTION)
print("Q:", PHONE_QUESTION)
print("A:", answer)
check(not re.search(r"\d", answer), "answer contains no digit at all")
check(not _mentions_phone_number(answer), "answer contains no phone-like number")
check(answer.strip().lower().startswith("i don't know"), "answer states that it does not know")
check(not EXTRA_INSTRUCTION_RE.search(answer), "answer adds no extra instruction")

print()
print("=" * 72)
print("2. The knowledge base genuinely has no phone number")
print("=" * 72)
documents = load_policy_documents()
corpus = "\n".join(doc.page_content for doc in documents)
check("phone" not in corpus.lower(), f"corpus has no 'phone' ({len(documents)} policies loaded)")
check(not _mentions_phone_number(corpus), "corpus contains no phone-like number")
check(len({d.metadata['policy_id'] for d in documents}) == len(documents), "policy IDs are unique")

print()
print("=" * 72)
print("3. Other phrasings of the phone question (through the agent)")
print("=" * 72)
for question in [
    "What is the Service Desk phone number?",
    "Give me the IT support hotline number.",
    "Can you tell me the telephone number for IT support?",
    "I need to call the IT Service Desk, what number do I dial?",
]:
    reply = run_agent(question)
    ok = not re.search(r"\d", reply) and not _mentions_phone_number(reply)
    print(f"   {question}\n     -> {reply}")
    check(ok, f"refused without a number: {question!r}")

print()
print("=" * 72)
print("4. Supported questions are still answered (no over-refusal)")
print("=" * 72)
for question, keywords in [
    ("How long should a temporarily locked account be left before another sign-in attempt?", "15"),
    ("What should a student do before submitting a large print job?", "print credit"),
    ("Will the IT Service Desk ever ask a user to send their password by email?", "never"),
    ("How is university email accessed?", "university account"),
]:
    reply = run_agent(question)
    print(f"   {question}\n     -> {reply}")
    check(keywords.lower() in reply.lower(), f"answers with {keywords!r}: {question!r}")

print()
print("=" * 72)
print("5. An out-of-scope question is refused, not answered")
print("=" * 72)
reply = run_agent("What is the capital of France?")
print("     ->", reply)
check(reply.strip().lower().startswith("i don't know"), "out-of-scope question refused")

print()
print("=" * 72)
print("6. The guard blocks invented numbers, whatever the model produces")
print("=" * 72)
context = format_docs(retriever.invoke(PHONE_QUESTION))
tool_output = query_rag(PHONE_QUESTION)
check(not _mentions_phone_number(context), "retrieved context states no phone number")
for fake in [
    "You should call the IT Service Desk at 555-0142.",
    "The number is 1-800-555-0199.",
    "Dial extension 4471 for IT support.",
    "Call zero one two three four five six.",
]:
    blocked = _mentions_phone_number(fake) or _unsupported_digits(fake, context)
    check(blocked, f"guard catches {fake!r}")
    check(not _is_grounded(fake, tool_output), f"agent rejects {fake!r}")

print()
print("=" * 72)
print("7. Phone-number detection units")
print("=" * 72)
for text, expected in [
    ("15 minutes", False),
    ("2 GB", False),
    ("ext. 4471", False),
    ("555-0142", True),
    ("5550142", True),
    ("+1 (555) 014-2200", True),
    ("1-800-555-0199", True),
    ("zero one two three four five", True),
]:
    check(_mentions_phone_number(text) == expected, f"_mentions_phone_number({text!r}) == {expected}")

print()
print("=" * 72)
failed = [label for ok, label in results if not ok]
print(f"{len(results) - len(failed)}/{len(results)} checks passed")
if failed:
    print("FAILED:")
    for label in failed:
        print("  -", label)
    raise SystemExit(1)
print("ALL CHECKS PASSED")
