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
from pathlib import Path
BASE = Path(__file__).resolve().parent


# ── 문서 처리 ──
DOC_PATH      = BASE.parent.parent / "data/pm_manual.pdf"
CHUNK_SIZE    = 500      # 500→300: +10%p, 토큰 33% 절감
CHUNK_OVERLAP = 50       # chunk_size의 10%

# ── 임베딩 · 저장소 ──
EMBED_MODEL   = "text-embedding-3-small"   # ⚠ 변경 시 인덱스 재생성 
INDEX_PATH    = "faiss_index"

# ── 검색 ──
TOP_K         = 5       # 3→5: +10%p (k=8은 노이즈로 하락) 
MIN_SCORE     = -0.01     # 9차시 실측값
SEARCH_TYPE   = "similarity"

# ── 생성 ──
LLM_MODEL     = "gpt-4o-mini"
TEMPERATURE   = 0
PROMPT_VER    = "v3"     # v2→v3: +10%p, 토큰 증가 없음

# ── 메시지 ──
MSG_NO_DOC    = "관련 자료를 찾지 못했습니다. ㅡ,ㅡ"
MSG_ERROR     = "일시적인 오류가 발생했습니다. 잠시 후 다시 시도해주세요."