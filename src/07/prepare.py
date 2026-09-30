from langchain_text_splitters import RecursiveCharacterTextSplitter

import os
import sys

PDF_PATH = "../../data/manual.pdf"

CHUNK_SIZE = 250
CHUNK_OVERLAP = 30

CURRENT_DIR = os.path.dirname(__file__)
INGEST_DIR = os.path.join(CURRENT_DIR, "..", "06")
sys.path.insert(0, INGEST_DIR)

from ingest import load_documents

def prepare_chunks(path):
    docs = load_documents(path, False)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )
    
    chunks = splitter.split_documents(docs)

    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = index

    print(f"chunk size: {splitter._chunk_size}")
    print(f"chunk overlap: {splitter._chunk_overlap}")
    print("-" * 45)
    print(f"문서 분할(chunking) 완료!: {len(chunks)}")
    return chunks

if __name__ == '__main__':
    chunks = prepare_chunks(PDF_PATH)

    for chunk in chunks[:3]:
        print("=" * 45)
        print(chunk.page_content)
        print("-" * 45)
        print(chunk.metadata["chunk_id"])