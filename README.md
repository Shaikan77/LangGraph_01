# LangGraph기반 Modular RAG
2026.09.28(MON) ~ 10.02(FRI)

## LangChain 입문

### RAG 5단계
[1] 수집(ingestion) → [2] 분할(chunking)  → [3] 임베딩(embedding) → [4] 검색(retrieve) → [5] 생성(generation)

#### 1. 인덱싱 단계 (한 번만)
문서 읽기(ingest) → 조각 내기(chunk) → 벡터로 저장(embed)

#### 2. 질의 단계 (질문할 때 마다 반복)
관련 조각 검색 → 근거 주고 답변 받기

## LangGraph 입문
그래프(Graph) = 점(Node)과 선(Edge)으로 AI의 흐름을 그린다.
- Node: 함수(작업 단위…기능) - 무엇을 할지
- Edge: 분기(연결선) - 어디로 갈지
- state: 전역 변수(=데이터) - 노드(node)들이 함께 쓰는 공책 … 노드를 거쳐갈 수록 내용이 채워진다. 지금까지의 진행상황을 알 수 있다.

