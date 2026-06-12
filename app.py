
import os
from flask import Flask, request, jsonify, render_template, session
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage

app = Flask(__name__)
app.secret_key = "landleguard-secret-key"

# ==========================================
# 1. LLM 및 API 설정 (Tensorix API 적용)
# ==========================================
tensorix_api_key = os.getenv("TENSORIX_API_KEY")

if not tensorix_api_key:
    raise ValueError("TENSORIX_API_KEY 환경변수가 설정되어 있지 않습니다.")

llm = ChatOpenAI(
    model_name="moonshotai/kimi-k2.5",
    openai_api_key=tensorix_api_key,  # 직접 적는 대신 변수 사용
    openai_api_base="https://api.tensorix.ai/v1",
    temperature=0
)

# ==========================================
# 2. RAG 파이프라인 구축 (서버 시작 시 1회 로드)
# ==========================================
def initialize_rag():
    print("PDF 문서를 로드하고 RAG 시스템을 구축합니다...")
    docs = []
    
    # PDF 파일 경로 (data 폴더 내에 위치해야 함)
    pdf_files = [
        "data/Hochwasservorsorge-in-BW.pdf",
        "data/BBK-Vorsorgen-fuer-Krisen-und-Katastrophen.pdf"
    ]
    
    for file in pdf_files:
        if os.path.exists(file):
            loader = PyPDFLoader(file)
            docs.extend(loader.load())
        else:
            print(f"경고: {file} 파일을 찾을 수 없습니다.")

    # 문서 분할
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    # 임베딩 및 FAISS 벡터 스토어 생성 (HuggingFace 무료 모델 사용)
    embeddings = HuggingFaceEmbeddings(model_name="jhgan/ko-sroberta-multitask") # 한국어 특화 임베딩
    if splits:
        vectorstore = FAISS.from_documents(splits, embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    else:
        retriever = None
        print("문서가 없어 RAG 검색기를 초기화하지 못했습니다.")
        
    return retriever

retriever = initialize_rag()

# ==========================================
# 3. 프롬프트 템플릿 설정 (FLIWAS 홍수 분석용)
# ==========================================
system_prompt = (
    "너는 바덴뷔르템베르크(Baden-Württemberg) 지역의 지리적 데이터와 수문학적 특성을 분석하는 재난 방재 시스템 'LÄNDLEGUARD'의 핵심 AI야. "
    "사용자가 입력한 주소(예: 하일브론, 네카르줄름 등)와 주어진 문서를 바탕으로 홍수 위험도를 분석하고 리포트를 작성해줘.\n\n"
    "다음 단계를 거쳐 출력해:\n"
    "1. 주소 확인 및 해당 지역의 수문학적 특성 요약\n"
    "2. 검색된 문서를 기반으로 한 홍수(침수) 위험 수준(안전/주의/위험)\n"
    "3. 건축물 또는 거주지에 필요한 예방 조치 (Hochwasservorsorge)\n"
    "4. 비상시 행동 요령 및 연락처 가이드라인 (BBK 가이드라인 기반)\n"
    "5. 최대한 간결하고 전문적으로 문서화할 것. PDF 보고서 형태로 바로 쓸 수 있도록 Markdown의 제목 태그와 글머리 기호를 활용해줘.\n\n"
    "Context: {context}"
)

prompt_template = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])

if retriever:
    question_answer_chain = create_stuff_documents_chain(llm, prompt_template)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
else:
    rag_chain = None

# ==========================================
# 4. 라우팅
# ==========================================
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json()
    address = data.get("address", "")

    if not address:
        return jsonify({"answer": "주소를 입력해주세요."})

    try:
        if rag_chain:
            # RAG를 통해 컨텍스트 검색 후 답변 생성
            response = rag_chain.invoke({"input": f"내 집 주소는 '{address}'입니다. 홍수 위험을 분석해주세요."})
            answer = response["answer"]
        else:
            # 문서가 없을 경우 일반 LLM 호출
            answer = llm.invoke(f"내 집 주소는 '{address}'입니다. 홍수 위험을 분석해주세요.").content
            
        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"answer": f"분석 중 오류가 발생했습니다: {str(e)}"}), 500

if __name__ == "__main__":
    # 로컬 테스트용
    app.run(debug=True, port=5000)
