# 3일차. Vector Store
# 2026/09/30(수) AM 11:34

import datetime
import json
import os, sys
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

INDEX_PATH = "faiss_index"
EMBEDDING_MODEL = "text-embedding-3-small"
DOC_PATH = "../../data/manual.pdf"

CHUNK_SIZE = 250
CHUNK_OVERLAP = 30

CURRENT_DIR = os.path.dirname(__file__)
PREPARE_DIR = os.path.join(CURRENT_DIR, "..", "07")
sys.path.insert(0, PREPARE_DIR)

from prepare import prepare_chunks

load_dotenv()

def save_with_meta(store, path, info):
    store.save_local(path)
    
    info["created_at"] = datetime.datetime.now().isoformat() # 인데스를 만든 시간 기록

    meta_path = os.path.join(path, "build_info.json") # 인덱스 폴더에 build_info.json 파일 생성
    with open(meta_path, "w", encoding="utf-8") as file:
        json.dump(
            info,
            file,
            ensure_ascii=False,
            indent=2
        )

def get_store(rebuild=False):
    emb = OpenAIEmbeddings(model="text-embedding-3-small")
    print(f"> Embedding model: {EMBEDDING_MODEL} 임베딩 중 ...")
    
    if os.path.exists(INDEX_PATH) and not rebuild:    
        store = FAISS.load_local(
            INDEX_PATH,
            emb,
            allow_dangerous_deserialization=True
        )
        print("> 인덱스 로드 완료!")
    else:
        chunks = prepare_chunks(DOC_PATH)
            
        store = FAISS.from_documents(chunks, emb)

        build_info = {
            "source": DOC_PATH,
            "embedding_model": EMBEDDING_MODEL,
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
            "num_chunks": len(chunks)
        }
        save_with_meta(store, INDEX_PATH, build_info)

        print(f"> Index saved to {INDEX_PATH}!") # faiss_index 폴더에 (index.faiss, index.pkl) 인덱스 파일 생성 완료!
        
    print("-" * 45) # index 파일이 있으면 불러오기, 없으면 새로 만들고 저장
    return store

if __name__ == "__main__":
    store = get_store()