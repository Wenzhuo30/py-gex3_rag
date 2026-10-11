"""LangChain tool wrapper that exposes the policy RAG chain to the agent.

### COMPLETE THE CODE  ###
"""

from langchain_core.tools import Tool

from policy_loader import load_policy_documents
from rag import query_rag

## TO LOAD THE DOCUMENT, USE THE FOLLOWING ONLY:
documents = load_policy_documents()


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
