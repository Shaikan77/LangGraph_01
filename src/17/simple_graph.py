from typing import TypedDict
from langgraph.graph import StateGraph, START, END

# state
class SimpleRAGState(TypedDict):
    query: str      # 사용자 질문
    results: list   # 검색 결과 목록
    answer: str     # 최종 답변

# node 정의
def search_node(state: SimpleRAGState):
    # 현재 State에서 질문을 읽습니다.
    print("[검색 노드] 질문:", state["query"])

    # 실제 검색 대신 고정된 문서 목록을 준비합니다.
    results = [
        "문서1: 제품 가격은 10,000원입니다.",
        "문서2: 배송비는 무료입니다.",
        "문서3: 현재 할인 행사가 진행 중입니다."
    ]

    # LangGraph가 results 항목을 갱신하도록 반환합니다.
    return {"results": results}

def answer_node(state: SimpleRAGState):
    # 앞의 검색 노드가 반환한 결과를 읽습니다.
    print("[답변 노드] 검색 결과:", state["results"])

    # 실제 LLM 대신 고정된 답변을 준비합니다.
    answer = "검색 결과를 바탕으로 만든 답변입니다."

    # LangGraph가 answer 항목을 갱신하도록 반환합니다.
    return {"answer": answer}

# graph 구조 생성
graph = StateGraph(SimpleRAGState)

# (1) node 추가
graph.add_node("search", search_node)
graph.add_node("answer", answer_node)

# (2) edge 추가
graph.add_edge(START, "search")     # 시작점: search
graph.add_edge("search", "answer")
graph.add_edge("answer", END)       # 종료점: answer

app = graph.compile()

# 실제 수행을 위한 ...
initial_state = {
    "query": "가격이 얼마야?",
    "results": [],
    "answer": ""
}

result = app.invoke(initial_state)