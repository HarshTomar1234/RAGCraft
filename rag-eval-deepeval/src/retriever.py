import os
import re
import glob

import anthropic
from dotenv import load_dotenv

# from langchain_openai import OpenAIEmbeddings  # swapped out: no OpenAI credits
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

load_dotenv()  # loads ANTHROPIC_API_KEY from .env

DATA_DIR = "data"
DB_DIR = "chroma_store"
MODEL = "claude-haiku-4-5-20251001"


# Anthropic has no embeddings endpoint, so embed locally instead (free, offline).
# bge-base > MiniLM for semantic recall; BGE wants a query instruction prefix
# on queries only (not on documents) per its model card.
class LocalEmbeddings(Embeddings):
    QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

    def __init__(self, name="BAAI/bge-base-en-v1.5"):
        self.model = SentenceTransformer(name)

    def embed_documents(self, texts):
        return self.model.encode(texts).tolist()

    def embed_query(self, text):
        return self.model.encode(self.QUERY_PREFIX + text).tolist()


# 1. LOAD ---- read each transcript, throw away the VTT timestamps
def load_transcripts():

    docs = []
    for path in glob.glob(f"{DATA_DIR}/*.vtt"):
        lines = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line == "WEBVTT" or "-->" in line:
                    continue
                lines.append(line)
        text = " ".join(lines)

        session = re.search(r"Session[ _]*(\d+)", path).group(1)

        docs.append(Document(page_content=text, metadata={"session": session}))

    return docs


# 2. BUILD ---- chunk, embed once, and keep it on disk so we don't re-embed
def load_store():
    # embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    embeddings = LocalEmbeddings()

    if os.path.exists(DB_DIR):
        return Chroma(persist_directory=DB_DIR, embedding_function=embeddings)

    docs = load_transcripts()

    chunks = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
    ).split_documents(docs)

    return Chroma.from_documents(chunks, embeddings, persist_directory=DB_DIR)


def build_retriever():
    return load_store().as_retriever(search_kwargs={"k": 5})


# 3. ANSWER ---- Haiku reads the retrieved chunks. Returns (answer, contexts)
#    so deepeval gets both actual_output and retrieval_context.
def answer(question, retriever=None):
    retriever = retriever or build_retriever()
    contexts = [d.page_content for d in retriever.invoke(question)]

    msg = anthropic.Anthropic().messages.create(
        model=MODEL,
        max_tokens=500,
        system="Answer only from the transcript excerpts. If they don't cover it, say so.",
        messages=[
            {
                "role": "user",
                "content": "\n\n".join(contexts) + f"\n\nQuestion: {question}",
            }
        ],
    )

    return msg.content[0].text, contexts


# 4. TRY IT ---- python src/retriever.py
if __name__ == "__main__":

    text, contexts = answer("what is regression testing?")

    print(text, "\n")
    print(f"--- grounded in {len(contexts)} chunks ---")