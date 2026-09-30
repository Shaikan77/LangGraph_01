# 2일차 PM (09/29) 실습
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
import os
import sys

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

PDF_PATH = "../../data/manual.pdf"

CURRENT_DIR = os.path.dirname(__file__)
PREPARE_DIR = os.path.join(CURRENT_DIR, "..", "07")
sys.path.insert(0, PREPARE_DIR)

from prepare import prepare_chunks

print("-" * 45)
print("Ingest & Chunking ...")
load_dotenv()
chunks = prepare_chunks(PDF_PATH)

print("-" * 45)
print("Embedding ... (시간이 조금 걸립니다.)")
emb = OpenAIEmbeddings(model="text-embedding-3-small")

store = FAISS.from_documents(chunks, embedding=emb)
print("-" * 45)
print("Indexing 완료!")

# 답변을 생성할 LLM 준비
llm = ChatOpenAI(
    model="gpt-4o-mini", 
    temperature=0.2, 
    max_tokens=500
)
print("LLM 준비 완료!")

def build_context(docs):
    parts = []
    for i, doc in enumerate(docs, 1):
        text = (
            f"[{i}] "
            f"({doc.metadata['filename']} "
            f"p.{doc.metadata['page_no']})\n"
            f"{doc.page_content}\n"
        )
        parts.append(text)

    context = "\n\n".join(parts)
    return context

def ask(question, k=3):
    found = store.similarity_search(question, k=k)
    if not found:
        print("관련 자료를 찾을 수 없습니다.")
        return

    context = build_context(found)
    
    prompt = (
        "아래 자료만 근거로 답변해줘.\n"
        "자료에 없는 내용은 '자료에서 확인할 수 없습니다.'라고 답변해줘.\n"
        "추측은 하지마!\n\n"
        f"[자료]\n{context}\n\n"
        f"[질문] {question}"
    )
    
    response = llm.invoke(prompt)
    answer = response.content.strip()

    print("> 질문 : ", question)
    print("> 답변 : ", answer)

if __name__ == '__main__':
    ask("제품의 모델이 어떻게 되지?")