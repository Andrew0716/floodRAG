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

app = Flask(__name__)
app.secret_key = "landleguard-secret-key"

# ==========================================
# 1. LLM 및 API 설정 
# ==========================================
# Render 환경변수에서 키를 가져옵니다. 
tensorix_api_key = os.getenv("TENSORIX_API_KEY")

# 만약 환경변수 설정이 안 되어 있다면 임시로 기존 키를 사용합니다.
if not tensorix_api_key:
    tensorix_api_key = "sk-rxscDJ1H-sxt4wRJ93zpIA"

llm = ChatOpenAI(
    model_name="moonshotai/kimi-k2.5",
    openai_api_key=tensorix_api_key,
    openai_api_base="https://api.tensorix.ai/v1",
    temperature=0
)

# ==========================================
# 2. RAG 파이프라인 구축
# ==========================================
def initialize_rag():
    print("PDF 문서를 로드하고 RAG 시스템을 구축합니다...")
    docs = []
    
    # PDF 파일 경로 (가장 상위 폴더의 data 폴더 안에 넣으세요)
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

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="jhgan/ko-sroberta-multitask")
    if splits:
        vectorstore = FAISS.from_documents(splits, embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    else:
        retriever = None
        
    return retriever

retriever = initialize_rag()

# ==========================================
# 3. 프롬프트 템플릿 설정 
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
            response = rag_chain.invoke({"input": f"내 집 주소는 '{address}'입니다. 홍수 위험을 분석해주세요."})
            answer = response["answer"]
        else:
            answer = llm.invoke(f"내 집 주소는 '{address}'입니다. 홍수 위험을 분석해주세요.").content
            
        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"answer": f"분석 중 오류가 발생했습니다: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)
