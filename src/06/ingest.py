from langchain_community.document_loaders import PyPDFLoader
import re
import os

NOISE = ["대외비"]
PDF_PATH = "../../data/manual.pdf"

def load_documents(path, verbose: bool = False):
    loader = PyPDFLoader(path)
    docs = loader.load()
    
    filename = os.path.basename(path)

    for d in docs:
        text = d.page_content
        
        for n in NOISE:
            text = text.replace(n, "")

        text = re.sub(r"\n{3,}", "\n\n", text)

        d.page_content = text.strip()

        d.metadata["filename"] = filename
        d.metadata["page_no"] = d.metadata.get("page", 0) + 1

    empty_pages = []
    total_len = 0

    for d in docs:
        total_len += len(d.page_content)
        text = d.page_content.strip()

        if len(text) < 10:
            page = d.metadata.get("page", "?")
            empty_pages.append(page)

        if verbose:
            print("-" * 45)
            print(d.page_content)

    if empty_pages:
        print(" 스캔 pdf일 수 있습니다.")

    print("=" * 45)
    print(f"> {filename} 로딩 완료! (총{len(docs)}쪽 {total_len}자)")
    print("=" * 45)

    return docs

if __name__ == '__main__':
    docs = load_documents(PDF_PATH, False)

        