# LangGraph기반 Modular RAG
2026.09.28(MON) ~ 10.02(FRI)

## 전자정부지원사업 사업관리 도우미 RAG

### 사업관리 가이드 (예: eGov-PM Advisor)
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

#### 공공기관 문서의 특징을 고려하여 PyMuPDF 엔진 사용 - 한글 추출 정확도와 속도 면에서 월등
from langchain_community.document_loaders import PyPDFLoader 대신에
from langchain_community.document_loaders import PyMuPDFLoader 사용