from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader(
    "../../data/LangChain.pdf"
)

docs = loader.load()

def validate(docs):

    empty_pages = []

    print("-" * 45)
    print(f"Docs: {docs[0].metadata["source"]}")
    print("-" * 45)

    for d in docs:
        text = d.page_content.strip()
        if len(text) < 10:
            empty_pages.append(d.metadata.get("page", "?"))
            print(f"Page: {d.metadata["page"]+1} / {d.metadata["total_pages"]}")
            print(f"Meta:\n {d.metadata}")
            print("-" * 45)

    if empty_pages:
            print(" 스캔 pdf일 수 있습니다.")
            

validate(docs)