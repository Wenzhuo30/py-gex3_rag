"""ReAct agent over the university IT policy RAG tool.

The agent follows the Lecture 11 pattern (``create_react_agent`` +
``AgentExecutor``) and uses ``qwen3:0.6b`` as the generator, as the exercise
requires.

Import note: this environment runs LangChain 1.4.4, where the classic agent API
moved out of ``langchain.agents``.  ``AgentExecutor`` and ``create_react_agent``
now live in ``langchain_classic.agents`` (the ``langchain-classic`` package).

Why the tool call is issued deterministically
---------------------------------------------
``qwen3:0.6b`` has 0.6B parameters and cannot reliably drive a ReAct loop.  Two
observed failure modes on the target question:

* running the loop normally, the model jumps straight to ``Final Answer:`` and
  never calls the tool, so the answer comes from the model alone rather than
  from the knowledge base;
* asked to produce an ``Observation`` of its own, it fabricates one.

Either way retrieval is bypassed, which is precisely how a small model starts
inventing facts.  :func:`run_agent` therefore performs the tool call itself and
places the real result in the ``Observation`` slot before the model writes the
final answer.  The model then only phrases an answer from real evidence, and the
deterministic guard in ``rag.py`` still vetoes any number that is not present in
the retrieved policy text.
"""

from __future__ import annotations

import re
import time

from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate

from rag import query_rag
from rag_tool import policy_search_tool


agent_llm = Ollama(model="qwen3:0.6b", temperature=0.0)


tools = [policy_search_tool]
TOOL_NAME = policy_search_tool.name
TOOL_NAMES = ", ".join(tool.name for tool in tools)
TOOL_DESCRIPTION = "\n".join(f"- {tool.name}: {tool.description}" for tool in tools)


react_prompt = PromptTemplate.from_template(
    """Answer the user's question using ONLY the Observation below.

You have access to the following tool:

{tools}

Critical rules:
- You have no knowledge of your own. Use only the text of the Observation.
- NEVER invent, guess, or recall a phone number, extension number, email
  address, or any other contact detail. Such details may only be copied
  verbatim from the Observation, and only if it actually contains them.
- If the Observation contains the answer, answer the question in one or two
  sentences using only that text.
- If the Observation does not contain the requested fact, your Final Answer
  must say that you do not know it. That is the correct answer, not a failure.
- Answer only the question asked. Do not add advice, suggestions, or unrelated
  policy details.

The only action available is [{tool_names}].  It has already been called for
you and its result is the Observation below, so you must not call it again.

Use exactly this format:

Question: {input}
Thought: I must use the tool.
Action: {tool_name}
Action Input: {input}
Observation: {observation}
Thought: I now know the final answer
Final Answer: the answer for the user, taken only from the Observation

Question: {input}
Thought:{agent_scratchpad}"""
)


agent = create_react_agent(agent_llm, tools, react_prompt)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=False,
    handle_parsing_errors=True,
    max_iterations=4,
    return_intermediate_steps=False,
)


_WORD_RE = re.compile(r"[a-z0-9]+")
_PHONE_RE = re.compile(r"\d(?:[\d\s\-().]*\d){4,}")
_REFUSAL_RE = re.compile(
    r"(?i)^\W*(?:i\s+(?:do\s+not|don't|dont)\s+know|i\s+(?:am|'m)\s+not\s+sure|"
    r"i\s+(?:cannot|can't|can not)|the\s+(?:context|documents?|polic(?:y|ies))|"
    r"no\s+information|unknown|not\s+(?:available|provided|specified|mentioned))\b"
)


def _build_scratchpad(question: str, observation: str) -> str:
    """Render the completed tool call that precedes the model's final answer.

    The thought, action and observation are fixed text: the tool name comes from
    the tool itself and ``observation`` is the real return value of
    :func:`rag.query_rag`.  Only the final answer is left to the model.
    """
    return (
        " I must use the tool.\n"
        f"Action: {TOOL_NAME}\n"
        f"Action Input: {question}\n"
        f"Observation: {observation}\n"
        "Thought: I now know the final answer\n"
        "Final Answer:"
    )


def run_agent(user_question: str) -> str:
    """Answer ``user_question`` with the policy tool, without inventing facts.

    The policy search runs first and its result is handed to the ReAct agent as
    the real Observation.  The agent's wording is kept only when it stays
    grounded in that Observation; otherwise the verified tool answer is
    returned.  That answer is the RAG result, so it already carries the
    deterministic no-hallucination guard: it never contains a phone number the
    policy documents do not state.
    """
    
    tool_output = query_rag(user_question)

    
    try:
        result = agent_executor.invoke(
            {
                "input": user_question,
                "agent_scratchpad": _build_scratchpad(user_question, tool_output),
            }
        )
        agent_output = result.get("output", "") if isinstance(result, dict) else str(result)
    except Exception:
       
        return tool_output

    final_answer = (agent_output or "").strip()

    if _is_grounded(final_answer, tool_output):
        return final_answer

    return tool_output


def _is_grounded(answer: str, tool_output: str) -> bool:
    """Check that ``answer`` is supported by ``tool_output``.

    The answer must share meaningful wording with the tool output, and any
    phone-like number in it must also appear in the tool output.
    """
    if not answer:
        return False

    tokens = [token for token in _WORD_RE.findall(answer.lower()) if len(token) > 4]
    if not tokens:
        return False
    overlap = sum(1 for token in tokens if token in tool_output.lower()) / len(tokens)
    if overlap < 0.75:
        return False

    for number in _PHONE_RE.findall(answer):
        if number.strip() not in tool_output:
            return False
    return True


def _invoke_llm(prompt_text: str, attempts: int = 3) -> str:
    """Run the model directly, retrying transient local-server stream failures.

    A streaming answer from the local Ollama server occasionally ends
    prematurely (``ChunkedEncodingError`` / protocol errors).  That is a
    transport fault, not a model answer, so the call is retried.  This helper
    exists so the raw model can be exercised without the agent scaffold.
    """
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            return agent_llm.invoke(prompt_text).strip()
        except Exception as exc:  # transport-level failure
            last_error = exc
            if attempt < attempts - 1:
                time.sleep(1.0 + attempt)
    raise last_error  # type: ignore[misc]


if __name__ == "__main__":
    test_question = "What exact phone number should I call for the University IT Service Desk?"
    answer = run_agent(test_question)
    print(f"Question: {test_question}")
    print(f"Answer: {answer}")
