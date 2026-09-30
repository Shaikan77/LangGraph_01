# 3일차. Vector Store
# 2026/09/30(수) AM 11:34

import datetime
import json
import os, sys
import warnings

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

INDEX_PATH = "faiss_index"
EMBEDDING_MODEL = "text-embedding-3-small"
DOC_PATH = "../../data/manual.pdf"

CHUNK_SIZE = 250
CHUNK_OVERLAP = 30

MIN_SCORE = 0.0
QUESTION = "환불과 교환은 어떻게 다른가요?"

warnings.simplefilter("ignore", DeprecationWarning)
warnings.simplefilter("ignore", UserWarning)

CURRENT_DIR = os.path.dirname(__file__)
INDEXER_DIR = os.path.join(CURRENT_DIR, "..", "10")

sys.path.insert(0, INDEXER_DIR)

from indexer import get_store

store = get_store()

def compare_k(q1, ks=(1, 3, 5)):

    print(f"> 질문: {q1}")
    print("=" * 58)

    for k in ks:
        pairs = store.similarity_search_with_relevance_scores(q1, k=k)

        scores = []
        low_count = 0

        for doc, score in pairs:
            scores.append(score)

            if score < MIN_SCORE:
                low_count = low_count + 1

        rounded_scores = []

        for score in scores:
            rounded_score = round(float(score), 3)
            rounded_scores.append(rounded_score)

        print()
        print(f"> k={k:<3}, 점수={rounded_scores}")
        print(f">     기준 미달({MIN_SCORE}) 조각: {low_count}개\n\n")

def show_result(name, retriever):
    docs = retriever.invoke(QUESTION)
    print(f"> {name}, {len(docs)}개의 검색 결과")

    for doc in docs:
        print("-" * 45)
        preview = doc.page_content[:55].replace("\n", " ")
        print(f"   p.{doc.metadata['page_no']} >>> {preview} ...")

    print("=" * 45)

# similarity_retriever = store.as_retriever(
#     search_type="similarity", 
#     search_kwargs={"k": 3}
# ) 
# show_result("similarity k=3", similarity_retriever)

manual_retriever = store.as_retriever(
    search_type="similarity", 
    search_kwargs={"k": 3, "filter": {"filename": "manual.pdf"}}
) 
show_result("Manual Filter k=3", manual_retriever)

# compare_k("환불은 며칠 이내인가요?")
# compare_k("환불과 교환은 어떻게 다른가요?")