"""
One-time script to build the Chroma vector store from your guideline PDFs.

Run this ONCE (from the project root, same folder as app.py) whenever you
add or change the source PDFs:

    python ingest.py

It reads every PDF in ./guidelines, splits them into chunks, embeds them,
and writes the resulting vector database into ./chroma_db.
"""

from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

BASE_DIR = Path(__file__).resolve().parent
SOURCE_DIR = BASE_DIR / "guidelines"        # put your PDF files here
PERSIST_DIR = str(BASE_DIR / "chroma_db")   # Chroma will create/overwrite this

def main():
    pdf_files = sorted(SOURCE_DIR.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(
            f"No PDFs found in '{SOURCE_DIR}/'. Put your guideline PDF files there first."
        )

    print(f"Found {len(pdf_files)} PDF(s): {[p.name for p in pdf_files]}")

    all_docs = []
    for pdf_path in pdf_files:
        print(f"Loading {pdf_path.name} ...")
        loader = PyPDFLoader(str(pdf_path))
        all_docs.extend(loader.load())

    print(f"Loaded {len(all_docs)} page(s) total. Splitting into chunks ...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(all_docs)
    print(f"Created {len(chunks)} chunk(s).")

    print("Embedding and writing to Chroma (this can take a minute) ...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    Chroma.from_documents(chunks, embeddings, persist_directory=PERSIST_DIR)

    print(f"Done. Vector store saved to '{PERSIST_DIR}/'.")


if __name__ == "__main__":
    main()