# 3일차. Vector Store
# 2026/09/30(수) AM 11:11

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

store = FAISS.from_documents(chunks, emb)
store.save_local(INDEX_PATH)
print(f"> Index saved to {INDEX_PATH}!") # faiss_index 폴더에 (index.faiss, index.pkl) 인덱스 파일 생성 완료!
print("-" * 45) # index 파일이 있으면 불러오기, 없으면 새로 만들고 저장

