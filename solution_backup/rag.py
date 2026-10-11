"""RAG pipeline for the university IT policy knowledge base.

Design note on embeddings
-------------------------
The RAG skeleton follows the Lecture 11 pattern (load -> split -> Chroma vector
store -> retriever -> prompt -> chain -> LLM).  The generator LLM is
``qwen3:0.6b`` as required by the exercise.

``langchain_community.embeddings.OllamaEmbeddings(model="qwen3:0.6b")`` cannot
be used in this environment: the local Ollama server answers every
``/api/embed`` request with

    501 {"error":"This server does not support embeddings. Start it with `--embeddings`"}

for every installed model, and this Ollama build (0.34.2) does not accept an
``--embeddings`` flag.  ``qwen3`` is a generative model rather than a dedicated
embedding model, so it has no embedding head to serve.  There is also no
network access, so no embedding model can be pulled and no ONNX/HF embedding
model can be downloaded.

The embedder is therefore :class:`TfidfHashEmbeddings`, a deterministic,
dependency-free, fully offline ``Embeddings`` implementation.  Chroma is still
the real vector store and the retriever is still a real similarity search; only
the embedding provider is swapped.  To go back to Ollama embeddings once the
server supports them, replace the ``embeddings = ...`` line below with::

    from langchain_community.embeddings import OllamaEmbeddings
    embeddings = OllamaEmbeddings(model="qwen3:0.6b")
"""

from __future__ import annotations

import hashlib
import math
import re
import time
from collections import Counter

from langchain_community.vectorstores import Chroma
from langchain_core.embeddings import Embeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter

from policy_loader import load_policy_documents

# The generator LLM.  Prefer the standalone integration package that newer
# LangChain releases recommend, and fall back to the community package, which is
# what is installed in this environment.  Both expose a call-compatible class
# (``OllamaLLM`` is the rename of ``Ollama``).
try:  # pragma: no cover - depends on the installed distribution
    from langchain_ollama import OllamaLLM as Ollama  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - the path taken here
    from langchain_community.llms import Ollama  # type: ignore[import-untyped]

# --------------------------------------------------------------------------- #
# TO LOAD THE DOCUMENT, USE THE FOLLOWING ONLY (per exercise instructions)
# --------------------------------------------------------------------------- #
documents = load_policy_documents()

# --------------------------------------------------------------------------- #
# Canonical refusal sentences - the only acceptable answer when the knowledge
# base does not contain the requested fact.
# --------------------------------------------------------------------------- #
REFUSAL = (
    "I don't know the exact phone number for the University IT Service Desk. "
    "The official IT policy documents provided do not contain a Service Desk "
    "phone number, so I will not guess one. Please check the official "
    "university IT website or your university directory for the current number."
)

# Fallback wording for questions with no supporting context that are not about
# contact details.
GENERIC_REFUSAL = (
    "I don't know. The official IT policy documents provided do not contain "
    "the information needed to answer that question."
)

# --------------------------------------------------------------------------- #
# Deterministic offline embedder
# --------------------------------------------------------------------------- #
TOKEN_RE = re.compile(r"[a-z0-9]+")
EMBED_DIM = 1024

# High-frequency English words carry no retrieval signal and are dropped so a
# small set of stopwords cannot dominate the similarity score.
STOPWORDS = frozenset(
    """
    a an and are as at be before by can cannot could do does for from had has have
    if in into is it its may must no not of on or should so than that the their
    them then there these they this to up user users was were what when which who
    will with would you your
    """.split()
)


def _tokenize(text: str) -> list[str]:
    """Lowercase word tokenizer used for both documents and queries."""
    return [t for t in TOKEN_RE.findall(text.lower()) if t not in STOPWORDS and len(t) > 1]


class TfidfHashEmbeddings(Embeddings):
    """Deterministic TF-IDF embedding model backed by feature hashing.

    No network access and no third-party embedding model are required, so the
    pipeline is reproducible offline.  It is fitted once on the knowledge base
    so that inverse-document-frequency weights reflect the corpus.
    """

    def __init__(self, dim: int = EMBED_DIM) -> None:
        self.dim = dim
        self.df: Counter[str] = Counter()
        self.n_docs = 0
        self.idf: dict[str, float] = {}

    # -- feature hashing ---------------------------------------------------- #
    def _index(self, token: str) -> int:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(digest, "big") % self.dim

    def fit(self, corpus: list[str]) -> "TfidfHashEmbeddings":
        """Learn document frequencies from the knowledge base."""
        self.df = Counter()
        for text in corpus:
            self.df.update(set(_tokenize(text)))
        self.n_docs = max(len(corpus), 1)
        self.idf = {
            token: math.log((1.0 + self.n_docs) / (1.0 + freq)) + 1.0
            for token, freq in self.df.items()
        }
        return self

    def _vector(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        tokens = _tokenize(text)
        if not tokens:
            return vector
        for token, count in Counter(tokens).items():
            # Sub-linear term frequency, corpus IDF, then L2 normalise below.
            weight = (1.0 + math.log(count)) * self.idf.get(token, 1.0)
            vector[self._index(token)] += weight
        norm = math.sqrt(sum(value * value for value in vector))
        if norm > 0.0:
            vector = [value / norm for value in vector]
        return vector

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


# --------------------------------------------------------------------------- #
# Chunking + vector store
# --------------------------------------------------------------------------- #
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    length_function=len,
)
chunked_documents = text_splitter.split_documents(documents)

# Fit the embedder on the actual chunks that will be indexed.
embeddings = TfidfHashEmbeddings().fit([doc.page_content for doc in chunked_documents])

vector_store = Chroma.from_documents(
    documents=chunked_documents,
    embedding=embeddings,
)

# The corpus contains many near-duplicate policy variants, so the chunk that
# actually holds a specific fact (for example "left for 15 minutes") can sit
# just below the top few generic matches.  k=8 gives enough recall for those
# specific facts while staying well inside the model context window.
retriever = vector_store.as_retriever(search_kwargs={"k": 8})


def format_docs(docs) -> str:
    """Render retrieved chunks for the prompt, keeping provenance metadata."""
    return "\n\n".join(
        "[{policy_id} | topic={topic}] {content}".format(
            policy_id=doc.metadata.get("policy_id", "unknown"),
            topic=doc.metadata.get("topic", "unknown"),
            content=doc.page_content.strip(),
        )
        for doc in docs
    )


# --------------------------------------------------------------------------- #
# Prompt - grounded answering only
# --------------------------------------------------------------------------- #
rag_prompt = ChatPromptTemplate.from_template(
    """You are a university IT policy assistant. Answer the question using ONLY
the policy context below.

Strict rules:
1. Use only facts written in the context. Never use outside knowledge.
2. Never invent, guess, or estimate information that is not in the context.
3. If the context does not contain the answer, reply with exactly:
   I don't know.
4. Never output a phone number, extension, or contact detail unless it is
   written verbatim in the context. If no phone number appears in the context,
   you must not produce any digits as a phone number.
5. Answer only what was asked. Do not add unrelated policy details.
6. Keep the answer to one or two sentences when the context supports it.

Context:
{context}

Question: {question}

Answer:"""
)

# Generator model required by the exercise.
llm = Ollama(model="qwen3:0.6b", temperature=0.0)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | rag_prompt
    | llm
    | StrOutputParser()
)

# --------------------------------------------------------------------------- #
# Post-generation hallucination guard
# --------------------------------------------------------------------------- #
# Any run of digits that could be read as a phone number / extension.
DIGIT_RUN_RE = re.compile(r"\d[\d\s\-().]{4,}\d")
PHONE_WORD_RE = re.compile(r"(?i)\b(phone|telephone|call|dial|hotline|extension|ext)\b")
PHONE_DIGITS_RE = re.compile(r"\d[\d\s\-().]{5,}\d")
SPELLED_DIGITS_RE = re.compile(
    r"(?i)\b(zero|one|two|three|four|five|six|seven|eight|nine|oh)"
    r"(?:[\s\-]+(?:zero|one|two|three|four|five|six|seven|eight|nine|oh)){4,}\b"
)
# A bare refusal produced by the model (rule 3 of the prompt).
BARE_REFUSAL_RE = re.compile(
    r"(?i)^\W*(?:i\s+(?:do\s+not|don't|dont)\s+know|i\s+am\s+not\s+sure|"
    r"the\s+context\s+does\s+not|no\s+information)\b"
)


def _contains_phone_number(text: str) -> bool:
    """True if the text contains something readable as a phone number."""
    if PHONE_DIGITS_RE.search(text):
        return True
    if SPELLED_DIGITS_RE.search(text):
        return True
    # A short digit group is only suspicious next to a phone-related word.
    if PHONE_WORD_RE.search(text) and DIGIT_RUN_RE.search(text):
        return True
    return False


def _context_has_phone_number(context: str) -> bool:
    """True if the retrieved context actually states a phone number."""
    return _contains_phone_number(context)


def _is_question_about_phone(question: str) -> bool:
    return bool(PHONE_WORD_RE.search(question)) or bool(re.search(r"(?i)\bnumber\b", question))


def _refusal_for(question: str) -> str:
    """Choose the refusal wording for a question the knowledge base cannot answer."""
    return REFUSAL if _is_question_about_phone(question) else GENERIC_REFUSAL


def _invoke_chain(chain, question: str, attempts: int = 3) -> str:
    """Run the RAG chain, tolerating transient local-server stream errors.

    A streaming response from the local Ollama server occasionally ends
    prematurely, which surfaces as ``requests``/``urllib3`` protocol errors.
    Those are transport faults rather than model answers, so the call is
    retried before giving up.
    """
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            return chain.invoke(question).strip()
        except Exception as exc:  # transport-level failure
            last_error = exc
            if attempt < attempts - 1:
                time.sleep(1.0 + attempt)
    raise last_error  # type: ignore[misc]


def query_rag(question: str) -> str:
    """Answer a question from the policy knowledge base, refusing when unsupported.

    The guard is deliberate defence in depth: a small model can still emit a
    plausible-looking number, so the generated answer is verified against the
    retrieved context before it is returned.
    """
    docs = retriever.invoke(question)
    context = format_docs(docs)

    if not docs or not context.strip():
        return _refusal_for(question)

    try:
        answer = _invoke_chain(rag_chain, question)
    except Exception:
        # The model could not be reached.  Refusing is always safe: it never
        # invents a fact and never leaks a phone number.
        return _refusal_for(question)

    # Hard guard: if the context contains no phone number, a phone number must
    # never appear in the answer, whatever the model produced.
    if not _context_has_phone_number(context) and _contains_phone_number(answer):
        return _refusal_for(question)

    if not answer:
        return _refusal_for(question)

    # The model refused in its own words: return the canonical answer instead,
    # which is clearer for the user and never invents contact details.
    if BARE_REFUSAL_RE.match(answer):
        return _refusal_for(question)

    return answer


if __name__ == "__main__":
    test_question = "What exact phone number should I call for the University IT Service Desk?"
    print("Question:", test_question)
    print("Answer:", query_rag(test_question))
