import os, sys

from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
RETRIEVER_DIR = os.path.join(CURRENT_DIR, "..", "11")

sys.path.insert(0, RETRIEVER_DIR)

from retriever import build_context, search
from prompts import PROMPTS

load_dotenv()

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

def compare(question):

    documents = search(question)
    context = build_context(documents)

    print("=" * 60)
    print("> Q:", question)
    print("> 근거:", len(documents), "개")
    print("-" * 60)

    for name, prompt in PROMPTS.items():
        chain = prompt | llm | StrOutputParser()
        answer = chain.invoke({
            "context": context,
            "question": question
        })

        print()
        print(f"[{name}]")
        print(answer)

if __name__ == "__main__":
    compare("환불 신청 방법과 수수료를 알려줘!")
