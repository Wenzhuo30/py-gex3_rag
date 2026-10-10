### COMPLETE THE CODE ###
## TO LOAD THE DOCUMENT, USE THE FOLLOWING ONLY:
# documents = load_policy_documents()

from policy_loader import load_policy_documents
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate

# 按要求使用指定方式加载策略文档
documents = load_policy_documents()

# 文档分块，适配检索粒度
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    length_function=len,
)
chunked_documents = text_splitter.split_documents(documents)

# 初始化嵌入模型与内存向量库
embeddings = OllamaEmbeddings(model="qwen3:0.6b")
vector_store = Chroma.from_documents(
    documents=chunked_documents,
    embedding=embeddings,
)
retriever = vector_store.as_retriever(search_kwargs={"k": 4})

# 严格防幻觉的 RAG 提示词
rag_prompt = ChatPromptTemplate.from_template(
    """You are a university IT policy assistant. Answer the question using ONLY information from the provided context.

Strict Rules:
1. Never invent, guess, or make up any information that is not in the context.
2. If the answer is not present in the context, respond exactly: "I don't know the exact phone number for the University IT Service Desk."
3. Do not provide any phone number that is not explicitly stated in the context.
4. Keep answers concise and directly relevant to the question.

Context:
{context}

Question: {question}

Answer:"""
)

# 初始化 qwen3:0.6b 大模型，低温度保证事实性
llm = Ollama(model="qwen3:0.6b", temperature=0.0)

# 格式化检索到的文档
def format_docs(docs):
    return "\n\n".join([doc.page_content.strip() for doc in docs])

# 构建完整 RAG 链
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | rag_prompt
    | llm
    | StrOutputParser()
)

def query_rag(question: str) -> str:
    """对外暴露的 RAG 查询接口"""
    return rag_chain.invoke(question)
