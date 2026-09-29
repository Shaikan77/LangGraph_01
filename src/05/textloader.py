# LangChain 커뮤니티 패키지에서 텍스트 파일(.txt)을 불러오는 전용 로더 모듈 임포트
from langchain_community.document_loaders import TextLoader

# 1. 문서 로더(TextLoader) 객체 초기화
# - 대상 파일: "data/notice.txt" (현재 작업 디렉터리 기준 상대 경로)
# - encoding="utf-8": 한글 깨짐 방지 및 다국어 지원을 위한 인코딩 명시
loader = TextLoader("data/notice.txt", encoding="utf-8")

# 2. 파일 읽기 및 LangChain Document 객체 리스트 생성
# - load() 메서드는 파일 내용을 읽어 Document 객체 리스트(List[Document]) 형태로 반환함
# - TextLoader는 단일 텍스트 파일을 통째로 읽으므로 리스트의 길이는 기본적으로 1임
documents = loader.load()

# 3. 로드 결과 및 Document 내부 속성 출력
# - len(documents): 로드된 LangChain Document 객체의 총 개수 확인 (기본값: 1)
print("---")
print(f"Load: {len(documents)}")

# - page_content: 문서의 실제 본문 텍스트 데이터 추출
print("---")
print(f"Contents: {documents[0].page_content}")

# - metadata: 문서의 메타데이터(딕셔너리 형태, 기본적으로 파일 출처인 'source' 경로 포함) 추출
print("---")
print(f"Meta: {documents[0].metadata}")