import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

BASE = Path(__file__).resolve().parent
CSV_PATH = BASE / "dataset" / "dataset.csv"
INDEX_DIR = BASE / "faiss_index"

_embeddings = None
_llm = None


def get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
    return _embeddings


def get_llm():
    global _llm
    if _llm is None:
        _llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",  # use a current model name from Google AI Studio
            google_api_key=os.environ["GEMINI_API_KEY"],
            temperature=0.1,
        )
    return _llm


def create_vector_db():
    loader = CSVLoader(
        file_path=str(CSV_PATH),
        source_column="prompt",
        encoding="utf-8-sig",
    )
    data = loader.load()
    db = FAISS.from_documents(data, get_embeddings())
    db.save_local(str(INDEX_DIR))


def get_answer(question: str, min_score: float = 0.5) -> str:
    db = FAISS.load_local(
        str(INDEX_DIR),
        get_embeddings(),
        allow_dangerous_deserialization=True,
    )
    results = db.similarity_search_with_relevance_scores(question, k=3)
    results = [(doc, score) for doc, score in results if score >= min_score]
    if not results:
        return "I don't know."

    context = "\n\n".join(doc.page_content for doc, _ in results)
    prompt = f"""Given the following context and a question, answer using this context only.
Reuse as much text as possible from the "response" part of the context without
changing it much. If the answer is not in the context, say "I don't know."
Do not make up an answer.

CONTEXT:
{context}

QUESTION: {question}"""
    return get_llm().invoke(prompt).content


if __name__ == "__main__":
    create_vector_db()
    print(get_answer("Why should I trust Nullclass?"))