"""Temporary: test ReAct prompt variants / stop sequences. Safe to delete."""
from langchain_community.llms import Ollama

from rag_tool import policy_search_tool

Q = "What exact phone number should I call for the University IT Service Desk?"
TOOL_NAME = policy_search_tool.name

BASE = """Answer the user's question about university IT policies.
You have access to the following tool:

{tools}

Tool names: {tool_names}

Use exactly this format:

Question: the input question
Thought: what to do next
Action: one of [{tool_names}]
Action Input: the question to send to the tool
Observation: the tool result
Thought: I now know the final answer
Final Answer: the answer for the user

Begin.

Question: {input}
Thought:{agent_scratchpad}"""

MINIMAL = """You must call one tool. Never write the Observation yourself.

{tools}

Tool names: {tool_names}

Format:
Question: {input}
Thought: I must search the policy documents.
Action: {tool_names}
Action Input: the search question
"""

NO_THINK = BASE + "\n/no_think"

VARIANTS = {
    "base (no stop)": (BASE, None, ""),
    "base stop=\\nObservation": (BASE, ["\nObservation"], ""),
    "base + /no_think": (NO_THINK, ["\nObservation"], ""),
    "minimal + stop=Observation": (MINIMAL, ["\nObservation", "Observation"], ""),
}

for label, (template, stop, pad) in VARIANTS.items():
    print("=" * 78)
    print("###", label)
    filled = template.format(
        tools="%s: %s" % (policy_search_tool.name, policy_search_tool.description),
        tool_names=TOOL_NAME,
        input=Q,
        agent_scratchpad=pad,
    )
    llm = Ollama(model="qwen3:0.6b", temperature=0.0, **({"stop": stop} if stop else {}))
    try:
        out = llm.invoke(filled)
    except Exception as exc:
        print("  ERR", type(exc).__name__, str(exc)[:200])
        continue
    print(repr(out))
    has_action = "Action:" in out
    fabricates_obs = "Observation:" in out
    print("  -> emits Action:", has_action, "| fabricates Observation:", fabricates_obs)
    print()
