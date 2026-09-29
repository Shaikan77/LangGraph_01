# 1. 필요 모듈 임포트
# - TextLoader: 텍스트 파일(.txt)을 LangChain 표준 Document 객체로 로드하는 클래스
from langchain_community.document_loaders import TextLoader
# - RecursiveCharacterTextSplitter: 단락, 문장, 단어 단위의 구분자 순서로 재귀적으로 문맥을 보존하며 분할하는 스플리터
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 2. 문서 로더(TextLoader) 인스턴스화
# - "data/notice.txt": 로드할 원본 텍스트 파일 경로
# - encoding="utf-8": 한글 및 다국어 인코딩 에러 방지
loader = TextLoader("data/notice.txt", encoding="utf-8")

# 3. 문서 로드 실행
# - 반환값: [Document(page_content="...", metadata={'source': 'data/notice.txt'})] 형태의 리스트
documents = loader.load()

# 4. 텍스트 분할기(Text Splitter) 설정
# - RecursiveCharacterTextSplitter는 기본 구분자 우선순위(["\n\n", "\n", " ", ""])를 바탕으로 문맥 단절을 최소화함
splitter = RecursiveCharacterTextSplitter(
    chunk_size=125,     # 각 청크(Chunk)의 최대 문자(Character) 수
    chunk_overlap=30    # 청크 간 중복(Overlap) 구간의 문자 수 (문맥 연속성 유지 목적)
)

# 5. Document 객체 분할 수행
# - split_documents(): Document 객체 리스트를 받아 분할된 다수의 Document 청크 리스트를 생성함
# - 원본 Document의 metadata(source 등)는 생성된 각 청크 Document에 그대로 상속됨
chunks = splitter.split_documents(documents)

# 6. 분할 결과 요약 출력
print("----------------")
print(f"chunk size: {splitter._chunk_size}")
print(f"chunk overlap: {splitter._chunk_overlap}")
print("----------------")

# 전체 생성된 청크 객체의 개수 출력
print(f"chunk 개수: {len(chunks)}\n")

# 7. 각 청크별 글자 수 검증 및 순회 출력
# - enumerate(chunks, 1): 1번 인덱스부터 시작하여 각 청크의 길이(글자 수) 확인
print("=" * 15)
for i, chunk in enumerate(chunks, 1):
    print(f"Chunk {i}: {len(chunk.page_content)} 글자")
    print(f"Chunk {i}: \n{chunk.page_content.strip()}\n")
    

print("=" * 15)
# 8. 원본 Document 객체 상태 확인
# - len(documents): 분할 전 원본 문서의 개수 (단일 파일이므로 1)
print(f"Load: {len(documents)}")
print("-" * 15)
# - page_content: 분할되기 전 원본 전체 본문 텍스트
print(f"Contents: {documents[0].page_content}")
# - metadata: 원본 파일 경로 등의 메타데이터 확인
print("-" * 15)
print(f"Meta: {documents[0].metadata}")
