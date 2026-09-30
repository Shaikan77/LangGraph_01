from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

import os
import sys

PDF_PATH = "../../data/manual.pdf"

CURRENT_DIR = os.path.dirname(__file__)
INGEST_DIR = os.path.join(CURRENT_DIR, "..", "06")
sys.path.insert(0, INGEST_DIR)

from ingest import load_documents

docs = load_documents(PDF_PATH, False)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=350,
    chunk_overlap=20,
    separators=["\n\n", "\n", ". ", " ", ""],
    length_function=len,
)

full_text = ""
for d in docs:
    full_text += d.page_content + "\n\n"

merged_doc = Document(page_content=full_text)

chunks = splitter.split_documents([merged_doc])

def show_boundary_info(chunks, index=0):
    print(chunks[index].page_content[-80:])
    print("-" * 45)
    print(chunks[index+1].page_content[:80])

show_boundary_info(chunks, 0)