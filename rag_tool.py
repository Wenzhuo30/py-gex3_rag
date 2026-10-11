"""LangChain tool wrapper that exposes the policy RAG chain to the agent.

### COMPLETE THE CODE  ###
"""

from langchain_core.tools import Tool

from policy_loader import load_policy_documents
from rag import query_rag

## TO LOAD THE DOCUMENT, USE THE FOLLOWING ONLY:
documents = load_policy_documents()

# The tool the ReAct agent calls.  ``query_rag`` already retrieves from the
# Chroma vector store, prompts qwen3:0.6b with the retrieved context, and
# applies the anti-hallucination guard, so its return value is the grounded
# answer the agent must pass through.
#
# The description is part of the grounding contract: it tells the agent that a
# "not available" result means the fact is unknown, and that the agent must
# never supply contact details of its own.
policy_search_tool = Tool(
    name="university_it_policy_search",
    description=(
        "Search the official university IT policy documents and return the "
        "relevant policy text. Use this for any question about university IT "
        "services: accounts and passwords, printing, WiFi, email, VPN, and "
        "storage. The input is a clear question about university IT policy. "
        "The tool returns grounded policy text, or a statement that the "
        "information is not in the documents. If the tool reports that the "
        "information is not available, the answer is unknown: report that you "
        "do not know it, and never supply contact details such as phone "
        "numbers yourself."
    ),
    func=query_rag,
)
