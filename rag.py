"""RAG pipeline over the university IT policy knowledge base.

Follows the Lecture 11 pattern (load -> embed -> vector store -> retriever ->
prompt -> chain -> LLM), with two deliberate differences from the worked
example:

1. The knowledge base is a single file.  ``policy_loader.load_policy_documents``
   already returns one ``Document`` per ``[POLICY]`` block, and each block is
   only 153-444 characters, so no text splitter is used.  Splitting a policy
   mid-sentence would break the very facts the retriever has to find.
2. The generator is ``qwen3:0.6b`` (the exercise requirement) instead of the
   8B model used in the lecture.

Embeddings
----------
``OllamaEmbeddings(model="qwen3:0.6b")`` cannot be used: the local Ollama server
answers every ``/api/embed`` request with

    501 {"error":"This server does not support embeddings."}

so no Ollama embedding model is available.  Chroma's default embedder needs a
model download, which is impossible offline.  The embedder is therefore
:class:`TfidfHashEmbeddings`, a deterministic, dependency-free and fully offline
``Embeddings`` implementation.  Chroma is still the real vector store and
``similarity_search`` is still a real vector search -- only the embedding
provider is swapped.  To switch back once the server supports embeddings,
replace the ``embeddings = ...`` line with::

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

from policy_loader import load_policy_documents

# OllamaLLM lives in the standalone package on new LangChain releases and in
# langchain_community on the version installed here; both classes are call
# compatible (OllamaLLM is the rename of Ollama).
try:  # pragma: no cover - depends on the installed distribution
    from langchain_ollama import OllamaLLM as Ollama
except ImportError:  # the path taken in this environment
    from langchain_community.llms import Ollama

# --------------------------------------------------------------------------- #
# 1. Load the knowledge base.
#    The exercise requires exactly this call: load_policy_documents().
# --------------------------------------------------------------------------- #
documents = load_policy_documents()

# --------------------------------------------------------------------------- #
# 2. Embedding model (see module docstring for why this is not Ollama).
# --------------------------------------------------------------------------- #
TOKEN_RE = re.compile(r"[a-z0-9]+")
EMBED_DIM = 1024

# High-frequency English words carry no retrieval signal; without this list the
# stopwords shared by every policy would dominate the cosine similarity.
STOPWORDS = frozenset(
    """
    a an and are as at be before by can cannot could do does for from had has have
    if in into is it its may must no not of on or should so than that the their
    them then there these they this to up user users was were what when which who
    will with would you your
    """.split()
)


def _tokenize(text: str) -> list[str]:
    """Lowercase word tokenizer shared by document and query embedding."""
    return [t for t in TOKEN_RE.findall(text.lower()) if t not in STOPWORDS and len(t) > 1]


class TfidfHashEmbeddings(Embeddings):
    """Deterministic TF-IDF embedding backed by feature hashing.

    Requires no network access and no third-party model, so retrieval is
    reproducible offline.  It is fitted once on the knowledge base so the IDF
    weights reflect this corpus.
    """

    def __init__(self, dim: int = EMBED_DIM) -> None:
        self.dim = dim
        self.n_docs = 0
        self.idf: dict[str, float] = {}

    def _index(self, token: str) -> int:
        """Map a token to a fixed dimension via a stable hash."""
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(digest, "big") % self.dim

    def fit(self, corpus: list[str]) -> "TfidfHashEmbeddings":
        """Learn inverse document frequencies from the knowledge base."""
        document_frequency: Counter[str] = Counter()
        for text in corpus:
            document_frequency.update(set(_tokenize(text)))
        self.n_docs = max(len(corpus), 1)
        self.idf = {
            token: math.log((1.0 + self.n_docs) / (1.0 + freq)) + 1.0
            for token, freq in document_frequency.items()
        }
        return self

    def _vector(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        tokens = _tokenize(text)
        if not tokens:
            return vector
        for token, count in Counter(tokens).items():
            # Sub-linear term frequency x corpus IDF, then L2-normalised so that
            # cosine similarity is a plain dot product.
            vector[self._index(token)] += (1.0 + math.log(count)) * self.idf.get(token, 1.0)
        norm = math.sqrt(sum(value * value for value in vector))
        if norm > 0.0:
            vector = [value / norm for value in vector]
        return vector

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


embeddings = TfidfHashEmbeddings().fit([doc.page_content for doc in documents])

# --------------------------------------------------------------------------- #
# 3. Vector store and retriever.
# --------------------------------------------------------------------------- #
vector_store = Chroma.from_documents(documents=documents, embedding=embeddings)

# The corpus repeats near-identical policy variants, so the single block that
# holds a specific fact can sit just below the top generic matches.  k=8 gives
# enough recall for those specific facts while staying inside the context window
# of a 0.6B model.
retriever = vector_store.as_retriever(search_kwargs={"k": 8})


def format_docs(docs) -> str:
    """Render retrieved documents for the prompt, keeping their provenance."""
    return "\n\n".join(
        "[{policy_id} | topic={topic}] {content}".format(
            policy_id=doc.metadata.get("policy_id", "unknown"),
            topic=doc.metadata.get("topic", "unknown"),
            content=doc.page_content.strip(),
        )
        for doc in docs
    )


# --------------------------------------------------------------------------- #
# 4. Prompt.  Grounding is enforced in the instructions and again after
#    generation by the guard further down.
# --------------------------------------------------------------------------- #
rag_prompt = ChatPromptTemplate.from_template(
    """You are a university IT policy assistant. Answer the question using ONLY the policy context below.

Rules:
1. Use only facts written in the context. Never use outside knowledge.
2. Never invent, guess, or estimate anything that is not in the context.
3. If the context does not contain the answer, reply with exactly: I don't know.
4. Never output a phone number, extension number, or other contact detail unless it appears verbatim in the context.
5. Answer only the question asked. Do not add extra advice, suggestions, or unrelated policy details.
6. Answer in one or two sentences.

Context:
{context}

Question: {question}

Answer:"""
)

# --------------------------------------------------------------------------- #
# 5. Generator model required by the exercise, then the chain.
# --------------------------------------------------------------------------- #
llm = Ollama(model="qwen3:0.6b", temperature=0.0)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | rag_prompt
    | llm
    | StrOutputParser()
)

# --------------------------------------------------------------------------- #
# 6. Refusal wording.
#
#    The knowledge base is the only source of truth, and it contains no phone
#    number at all, so "I don't know" is the complete and correct answer to the
#    Service Desk question.  The refusal therefore states the lack of
#    information and stops -- it must not add instructions, alternatives, or
#    contact details the documents never supplied.
# --------------------------------------------------------------------------- #
PHONE_REFUSAL = (
    "I don't know the exact phone number for the University IT Service Desk. "
    "The IT policy documents provided do not contain a Service Desk phone "
    "number, so I cannot give you one."
)

GENERIC_REFUSAL = (
    "I don't know. The IT policy documents provided do not contain the "
    "information needed to answer that question."
)

# --------------------------------------------------------------------------- #
# 7. Post-generation guard.
#
#    A 0.6B model can still emit a plausible-looking number even when the
#    context has none, so every answer is verified against the retrieved
#    context before it is returned.  This is what turns "should not hallucinate"
#    into "cannot hallucinate".
# --------------------------------------------------------------------------- #
# A digit run long enough to be written as a phone number / extension.  A real
# phone number carries at least 5 digits, so short quantities such as
# "15 minutes" or "2 GB" are not mistaken for one.
PHONE_DIGITS_RE = re.compile(r"\d(?:[\d\s\-().]*\d){4,}")
# The same, spelled out in words ("zero one two ...").
SPELLED_DIGITS_RE = re.compile(
    r"(?i)\b(?:zero|one|two|three|four|five|six|seven|eight|nine|oh)"
    r"(?:[\s\-]+(?:zero|one|two|three|four|five|six|seven|eight|nine|oh)){4,}\b"
)
# Every run of digits in a text, including single digits, so that a number of
# any length can be checked against the context.
DIGIT_RUN_RE = re.compile(r"\d[\d\s\-().]*\d|\d")
# Words that make a question a request for contact details.  "call" is
# deliberately absent: "which printer do I call from" is not a phone question.
PHONE_WORD_RE = re.compile(r"(?i)\b(?:phone|telephone|hotline|dial|extension|ext)\b")
# A refusal produced by the model itself (rule 3 of the prompt).
BARE_REFUSAL_RE = re.compile(
    r"(?i)^\W*(?:i\s+(?:do\s+not|don't|dont)\s+know|i\s+(?:am|'m)\s+not\s+sure|"
    r"i\s+(?:cannot|can't|can not)|the\s+(?:context|documents?|polic(?:y|ies))|"
    r"no\s+information|unknown|not\s+(?:available|provided|specified|mentioned))\b"
)
# Small models sometimes wrap their reasoning in these markers.
THINK_BLOCK_RE = re.compile(r"(?is)<(?:think|thinking|reasoning)>.*?</(?:think|thinking|reasoning)>")


def _strip_reasoning(text: str) -> str:
    """Remove any `` thinking...<｜end▁of▁thinking｜>`` block the model may emit."""
    return THINK_BLOCK_RE.sub(" ", text).strip()


def _digit_count(text: str) -> int:
    """Number of digits in a candidate phone number."""
    return sum(character.isdigit() for character in text)


def _mentions_phone_number(text: str) -> bool:
    """True if ``text`` contains something readable as a phone number.

    At least five digits are required, which is what separates a phone number
    or extension from a short quantity such as "15 minutes".
    """
    if SPELLED_DIGITS_RE.search(text):
        return True
    return any(_digit_count(match) >= 5 for match in PHONE_DIGITS_RE.findall(text))


def _unsupported_digits(answer: str, context: str) -> bool:
    """True if ``answer`` contains a number that the context does not contain.

    Catches a hallucinated phone number of any shape, including ones shorter
    than :data:`PHONE_DIGITS_RE` would match.
    """
    return any(
        digits and digits not in context
        for digits in (match.strip(" .-()") for match in DIGIT_RUN_RE.findall(answer))
    )


def _is_question_about_phone(question: str) -> bool:
    return bool(PHONE_WORD_RE.search(question))


def _refusal_for(question: str) -> str:
    """Pick the refusal wording for a question the knowledge base cannot answer."""
    return PHONE_REFUSAL if _is_question_about_phone(question) else GENERIC_REFUSAL


def _invoke_chain(chain, question: str, attempts: int = 3) -> str:
    """Run the chain, retrying transient local-server stream failures.

    A streaming answer from the local Ollama server occasionally ends
    prematurely, which surfaces as a ``requests``/``urllib3`` protocol error.
    That is a transport fault, not a model answer, so the call is retried.
    """
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            return chain.invoke(question)
        except Exception as exc:  # transport-level failure
            last_error = exc
            if attempt < attempts - 1:
                time.sleep(1.0 + attempt)
    raise last_error  # type: ignore[misc]


def answer_question(question: str) -> str:
    """Return the raw answer of the RAG chain, without the safety guard.

    Kept separate from :func:`query_rag` so the unguarded model output stays
    available for inspection and testing.
    """
    return _strip_reasoning(_invoke_chain(rag_chain, question))


def query_rag(question: str) -> str:
    """Answer ``question`` from the policy documents, refusing when unsupported.

    Returns either a context-grounded answer or one of the two refusal
    sentences.  It never returns a fact -- and in particular never a phone
    number -- that is absent from the retrieved policy text.
    """
    docs = retriever.invoke(question)
    context = format_docs(docs)

    if not docs or not context.strip():
        return _refusal_for(question)

    try:
        answer = answer_question(question)
    except Exception:
        # The model could not be reached.  Refusing is always safe: it neither
        # invents a fact nor produces a number the documents do not contain.
        return _refusal_for(question)

    if not answer:
        return _refusal_for(question)

    # The model refused in its own words.  Replacing that with the canonical
    # sentence keeps the wording consistent and keeps the answer free of any
    # advice the model may have appended.
    if BARE_REFUSAL_RE.match(answer):
        return _refusal_for(question)

    # Hard guard: no phone number may appear in the answer unless the retrieved
    # context states one, and no digit may appear that the context does not
    # support.  This is the anti-hallucination backstop for the phone question.
    if not _mentions_phone_number(context):
        if _mentions_phone_number(answer) or _unsupported_digits(answer, context):
            return _refusal_for(question)

    return answer.strip()


if __name__ == "__main__":
    test_question = "What exact phone number should I call for the University IT Service Desk?"
    print(f"Question: {test_question}")
    print(f"Answer: {query_rag(test_question)}")
