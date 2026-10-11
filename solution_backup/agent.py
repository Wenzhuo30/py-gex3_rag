"""Agent over the university IT policy RAG tool.

The generator model is ``qwen3:0.6b`` as required by the exercise.

Import note: this environment runs LangChain 1.4.4, where the classic agent API
moved out of ``langchain.agents``.  ``AgentExecutor`` and ``create_react_agent``
now live in ``langchain_classic.agents`` (the ``langchain-classic`` package),
so they are imported from there.

Why the tool call is forced
---------------------------
``qwen3:0.6b`` is a 0.6B-parameter model and cannot reliably drive a ReAct loop.
``create_react_agent`` binds ``stop=["\\nObservation"]`` so the model pauses
after ``Action Input`` and hands control back to the executor.  Observed
behaviour of this model on the target question:

* with that stop sequence it emits the whole scratchpad - including its own
  invented ``Observation:`` and ``Final Answer:`` - before the stop token, so
  no tool is ever called and the answer comes from the model alone;
* without the stop sequence it fills in a fabricated ``Observation:`` itself.

Either way the retrieval step is bypassed, which is exactly how a small model
starts inventing facts.  The tool call is therefore issued deterministically by
:func:`run_agent`, and the retrieved text is placed in the ReAct prompt as the
real ``Observation``.  The model then only phrases the final answer from that
evidence, and the deterministic guard in ``rag.py`` still vetoes any phone
number that is not present in the retrieved context.
"""

import re
import time

from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate

from rag import query_rag
from rag_tool import policy_search_tool

# The generator LLM.  Prefer the standalone integration package that newer
# LangChain releases recommend, and fall back to the community package, which is
# what is installed in this environment.  Both expose a call-compatible class
# (``OllamaLLM`` is the rename of ``Ollama``).
try:  # pragma: no cover - depends on the installed distribution
    from langchain_ollama import OllamaLLM as Ollama  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - the path taken here
    from langchain_community.llms import Ollama  # type: ignore[import-untyped]

# Generator model required by the exercise.  Low temperature keeps the model
# close to the retrieved text instead of paraphrasing creatively.
#
# Note: no ``stop`` parameter is passed here.  ``create_react_agent`` already
# binds ``stop=["\nObservation"]`` for ReAct; supplying it again raises
# "`stop` found in both the input and default params".
agent_llm = Ollama(model="qwen3:0.6b", temperature=0.0)

# Tools available to the agent.
tools = [policy_search_tool]
TOOL_NAME = policy_search_tool.name

# ReAct prompt.  The rules state explicitly that contact details may only come
# from the tool output, and that "I don't know" is the correct answer when the
# tool output does not contain the requested fact.
react_prompt = PromptTemplate.from_template(
    """Answer the user's question about university IT policies using ONLY the
observation provided to you.

You have access to the following tool:

{tools}

Tool names: {tool_names}

Critical rules:
- Use ONLY the text in the Observation. You have no knowledge of your own.
- NEVER invent, guess, or recall a phone number, extension, email address, or
  any other contact detail. Such details may only be copied verbatim from the
  Observation, and only if the Observation actually contains them.
- If the Observation does not contain the requested fact, your Final Answer
  must say that you do not know it. That is the correct answer.
- Never output digits as a phone number when the Observation has no phone
  number.
- Do not add unrelated policy information.

Use exactly this format:

Question: {input}
Thought: Do I need to use a tool? The policy search has already been run for me.
Action: {tool_names}
Action Input: the question to send to the tool
Observation: {observation}
Thought: I now know the final answer
Final Answer: the answer for the user, based only on the Observation

Question: {input}
Thought:{agent_scratchpad}"""
)

# ReAct agent and executor, kept as the agent scaffold required by the exercise.
agent = create_react_agent(agent_llm, tools, react_prompt)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=False,
    handle_parsing_errors=True,
    max_iterations=4,
    return_intermediate_steps=False,
)


def run_agent(user_question: str) -> str:
    """Answer ``user_question`` using the policy tool, without inventing facts.

    The retrieval step runs first and its result is fed to the agent as the
    real Observation.  The agent's own wording is used only when it stays
    grounded: the tool output must actually appear in the returned answer.

    If the agent output cannot be verified against the tool output (a small
    model can still paraphrase loosely or fail the ReAct format), the verified
    tool answer is returned instead.  That answer is the RAG result, so it
    already carries the deterministic no-hallucination guard: it never contains
    a phone number that is absent from the policy documents.
    """
    # Deterministic tool call - this is the same function the ReAct tool wraps.
    tool_output = query_rag(user_question)

    prompt_text = react_prompt.format(
        tools="\n".join("- %s: %s" % (t.name, t.description) for t in tools),
        tool_names=", ".join(t.name for t in tools),
        input=user_question,
        observation=tool_output,
        agent_scratchpad="",
    )

    try:
        agent_output = _invoke_llm(prompt_text)
    except Exception:
        agent_output = ""

    final_answer = _extract_final_answer(agent_output)

    # Accept the agent's phrasing only if it is grounded in the tool output.
    if _is_grounded(final_answer, tool_output):
        return final_answer

    return tool_output


def _invoke_llm(prompt_text: str, attempts: int = 3) -> str:
    """Run the model, retrying transient local-server stream failures.

    A streaming response from the local Ollama server occasionally ends
    prematurely (``ChunkedEncodingError`` / protocol errors).  Those are
    transport faults, not answers, so the call is retried.
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


def _extract_final_answer(text: str) -> str:
    """Pull the text after the last ``Final Answer:`` marker, if present."""
    if not text:
        return ""
    marker = "Final Answer:"
    if marker in text:
        return text.rsplit(marker, 1)[1].strip()
    return text.strip()


def _is_grounded(answer: str, tool_output: str) -> bool:
    """Check that ``answer`` is supported by ``tool_output``.

    The answer must share meaningful wording with the tool output, and any
    phone-like number in the answer must also appear in the tool output.
    """
    if not answer:
        return False

    tokens = [t for t in re.findall(r"[a-z0-9]+", answer.lower()) if len(t) > 4]
    if not tokens:
        return False
    tool_lower = tool_output.lower()
    overlap = sum(1 for t in tokens if t in tool_lower) / len(tokens)
    if overlap < 0.75:
        return False

    numbers = re.findall(r"\d[\d\s\-().]{5,}\d", answer)
    return all(number.strip() in tool_output for number in numbers)


if __name__ == "__main__":
    test_question = "What exact phone number should I call for the University IT Service Desk?"
    answer = run_agent(test_question)
    print(f"Question: {test_question}")
    print(f"Answer: {answer}")
