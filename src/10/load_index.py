# 3일차. Vector Store
# 2026/09/30(수) AM 11:24


from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

import os
import sys

CURRENT_DIR = os.path.dirname(__file__)
PREPARE_DIR = os.path.join(CURRENT_DIR, "..", "07")
sys.path.insert(0, PREPARE_DIR)

from prepare import prepare_chunks

DOC_PATH = "../../data/manual.pdf"
INDEX_PATH = "faiss_index"
EMBEDDING_MODEL = "text-embedding-3-small"

chunks = prepare_chunks(DOC_PATH)
print("-" * 45)

load_dotenv()

emb = OpenAIEmbeddings(model="text-embedding-3-small")
print(f"> Embedding model: {EMBEDDING_MODEL} 임베딩 중 ...")

store = FAISS.load_local(
    INDEX_PATH, 
    emb, 
    allow_dangerous_deserialization=True
)
print("> 인덱스 로드 완료!")
print("-" * 45)

query = "환불 규정"
found = store.similarity_search(query, k=3)
print(f"> 검색 결과 : {len(found)}개")
print("-" * 45)

for i, doc in enumerate(found):
    print(f"[{i+1}]\n")
    print(doc.page_content)
    print("-" * 45)