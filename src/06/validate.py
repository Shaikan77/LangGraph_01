from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader(
    "../../data/LangChain.pdf"
)

docs = loader.load()

def validate(docs):

    empty_pages = []

    for d in docs:
        text = d.page_content.strip()
        if len(text) < 10:
            empty_pages.append(d.metadata.get("page", "?"))
        print(f"Meta:\n {d.metadata}")

    if empty_pages:
            print(" 스캔 pdf일 수 있습니다.")
            

validate(docs)