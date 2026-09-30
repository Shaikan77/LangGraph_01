from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader(
    "../../data/certi.pdf"
)

documents = loader.load()

print("---")
print(f"[1] Load: {len(documents)}")
print("---")
print(f"[2] Contents:\n {documents[0].page_content}")
print("---")
print(f"[3] Meta:\n {documents[0].metadata}")