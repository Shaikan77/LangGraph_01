from langchain_community.document_loaders import TextLoader

loader = TextLoader("../../data/notice.txt", encoding="utf-8")

documents = loader.load()

print("---")
print(f"Load: {len(documents)}")
print("---")
print(f"Contents: {documents[0].page_content}")
print("---")
print(f"Meta: {documents[0].metadata}")