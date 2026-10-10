"""ReAct agent over the university IT policy RAG tool.

The agent uses ``qwen3:0.6b`` as required by the exercise.

Import note: this environment runs LangChain 1.4.4, where the classic agent
API moved out of ``langchain.agents``.  ``AgentExecutor`` and
``create_react_agent`` now live in ``langchain_classic.agents`` (the
``langchain-classic`` package), so they are imported from there.
"""

from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate

from rag import query_rag
from rag_tool import policy_search_tool

# Generator model required by the exercise.  Low temperature keeps the model
# close to the retrieved text instead of paraphrasing creatively.
#
# Note: no ``stop`` parameter here.  ``create_react_agent`` already binds
# ``stop=["\nObservation"]`` for ReAct; passing it again raises
# "`stop` found in both the input and default params".
agent_llm = Ollama(model="qwen3:0.6b", temperature=0.0)

# Tools available to the agent.
tools = [policy_search_tool]

# ReAct prompt.  The rules are explicit that the agent must relay the tool
# result and must never supply contact details on its own.
react_prompt = PromptTemplate.from_template(
    """Answer the user's question about university IT policies.
You have access to the following tool:

{tools}

Tool names: {tool_names}

Critical rules:
- ALWAYS call the university_it_policy_search tool before answering.
- You have NO knowledge of your own. Use only what the tool returns.
- NEVER invent, guess, or recall a phone number, extension, email address, or
  any other contact detail. Contact details may only be copied verbatim from
  the tool output, and only if the tool output actually contains them.
- If the tool output does not contain the requested fact, reply that you do not
  know. Saying "I don't know" is the correct answer in that case.
- Do not answer with unrelated policy information.

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


def run_agent(user_question: str) -> str:
    """Run the agent and return the final answer.

    ``qwen3:0.6b`` is a very small model, so ReAct formatting can occasionally
    fail.  If the agent cannot produce an answer, the RAG chain is used
    directly; that keeps the answer grounded and preserves the no-hallucination
    guarantee instead of returning a parser error.
    """
    try:
        result = agent_executor.invoke({"input": user_question})
        output = (result.get("output") or "").strip()
    except Exception:
        output = ""

    if not output:
        return query_rag(user_question)
    return output


if __name__ == "__main__":
    test_question = "What exact phone number should I call for the University IT Service Desk?"
    answer = run_agent(test_question)
    print(f"Question: {test_question}")
    print(f"Answer: {answer}")
