"""
사업관리 가이드 (예: eGov-PM Advisor)
=====================================================
- 작성자: 김동욱 (shaikan.msn@gmail.com)
- 작성일: 2026-10-01(목) 10:00
- 최종 수정일: 2026-10-01(목)
- 저작권: Copyright (c) 2026 한국기술교육대학교. All rights reserved.
- 라이선스: MIT License (또는 비공개/사내 라이선스)
- Project Structure
-----------------------------------------------------
- PMGuide_RAG
- ┠── config.py     모든 설정값 저장
- ┠── indexer.py    문서(pdf) → 인덱스 (한번만, 준비단계)
- ┠── rag.py        검색 + 생성 (핵심 로직)
- └── app.py        UI
"""

# 
# ============================================================
# 전자정부지원사업 사업관리매뉴얼 v6(PDF) 문서를 읽고 정리한 뒤 
# FAISS 벡터 인덱스를 생성
# 
# 저장된 인덱스가 있으면 새로 만들지 않고 기존 인덱스를 로딩
# 인덱싱 단계 (한 번만) indexer.app
# [1] 문서 읽기(ingest) → [2] 조각 내기(chunking) → [3] 벡터로 저장(embedding)
# ============================================================
import os
import re

# 공공기관 문서의 특징을 고려하여 PyMuPDF 엔진 사용 - 한글 추출 정확도와 속도 면에서 월등
# from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


import config

# PDF에서 반복적으로 제거할 불필요한 문구
NOISE = ["행정안전부", "한국지능정보사회진흥원", "NIA"]

# ============================================================
# PDF 문서를 읽고 텍스트와 메타데이터를 정리
# ============================================================
def _load():
    # PDF 파일의 모든 페이지를 읽습니다.
    # loader = PyPDFLoader(str(config.DOC_PATH))
    loader = PyMuPDFLoader(str(config.DOC_PATH))

    docs = loader.load()

    # 각 페이지를 하나씩 정리합니다.
    for doc in docs:
        # 페이지의 텍스트를 가져옵니다.
        text = doc.page_content

        # 불필요한 문구를 제거합니다.
        for noise in NOISE:
            text = text.replace(noise, "")

        # 연속된 빈 줄을 최대 두 줄로 정리합니다.
        text = re.sub(r"\n{3,}", "\n\n", text)

        # 정리된 텍스트를 다시 저장합니다.
        doc.page_content = text.strip()

        # 원본 파일 이름을 metadata에 저장합니다.
        doc.metadata["filename"] = config.DOC_PATH.name

        # 사람이 보기 편하도록 페이지 번호를 1부터 시작합니다.
        page = doc.metadata.get("page", 0)
        doc.metadata["page_no"] = page + 1

    # 모든 페이지 처리가 끝난 뒤 반환합니다.
    return docs

# ============================================================
# PDF 페이지를 검색에 적합한 작은 조각(chunk)으로 나누기
# → CHUNK_SIZE와 CHUNK_OVERLA이 성능에 중요한 영향을 미침
#       chunk_size=config.CHUNK_SIZE
#       chunk_overlap=config.CHUNK_OVERLA
# ============================================================
def _split(docs):
    # 문서를 나누는 도구를 만듭니다.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    # 전체 문서를 작은 조각으로 나눕니다.
    chunks = splitter.split_documents(docs)

    # 각 조각에 고유 번호를 붙입니다.
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i

    return chunks

# ============================================================
# FAISS 벡터 저장소를 준비
# 기존 인덱스가 있으면 로드하고, 없으면 새로 생성
# ============================================================
def get_store(rebuild=False):

    # OpenAI 임베딩 모델을 준비합니다.
    embeddings = OpenAIEmbeddings(model=config.EMBED_MODEL)

    # 저장된 인덱스가 있고 강제 재생성이 아니면 불러옵니다.
    if os.path.exists(config.INDEX_PATH) and not rebuild:
        store = FAISS.load_local(
            config.INDEX_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )
        return store

    # PDF 문서를 읽습니다.
    docs = _load()

    # 문서를 작은 조각으로 나눕니다.
    chunks = _split(docs)

    # 각 조각을 임베딩하여 FAISS 인덱스를 만듭니다.
    store = FAISS.from_documents(chunks, embeddings)

    # 생성한 인덱스를 디스크에 저장합니다.
    store.save_local(config.INDEX_PATH)

    print(f" chunk size: {config.CHUNK_SIZE}")
    print(f" chunk overlap: {config.CHUNK_OVERLAP}")
    print("-" * 45)
    print(f"> 인덱스 생성 완료!: (조각: {len(chunks)}개)")
    print("=" * 50)

    return store