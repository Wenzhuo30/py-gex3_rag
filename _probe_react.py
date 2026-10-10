"""Temporary: inspect raw model output for the ReAct format. Safe to delete."""
from agent import agent_llm, react_prompt, tools

Q = "What exact phone number should I call for the University IT Service Desk?"

filled = react_prompt.format(
    tools="\n".join("- %s: %s" % (t.name, t.description) for t in tools),
    tool_names=", ".join(t.name for t in tools),
    input=Q,
    agent_scratchpad="",
)

print("### PROMPT (tail) ###")
print(filled[-700:])
print()
print("### RAW LLM OUTPUT ###")
out = agent_llm.invoke(filled)
print(repr(out))
