# 3일차. Embedding 
# 2026/09/30(수) AM09:39

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

# .env 파일에 환경변수 읽어오기. OPENAI_API_KEY=sk-xxxxxx
load_dotenv()


prompt = "환불 규정이 궁금해"
emb = OpenAIEmbeddings(model="text-embedding-3-small")
store = FAISS.from_documents(chunks, embedding=emb)

vectors = emb.embed_query(prompt)

print(f"벡터 차원수: {len(vectors)}") # 1536

first_10 = []

for value in vectors[:10]:
    first_10.append(round(value, 4))

print(f"벡터 값(앞 10개): {first_10}")

# 답변을 생성할 LLM 준비
llm = ChatOpenAI(
    model="gpt-4o-mini", 
    temperature=0.2, 
    max_tokens=500
)
print("LLM 준비 완료!")

response = llm.invoke(prompt)
answer = response.content.strip()

print("> 질문 : ", prompt)
print("> 답변 : ", answer)