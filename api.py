from langchain.chat_models import init_chat_model
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import os

# Load environment variables
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")

model = init_chat_model("gemini-2.5-flash", model_provider="google_genai")
embeddings = OllamaEmbeddings(model="bge-m3")

#load vector store
vector_store = Chroma(
    collection_name="AnantaCV",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db",
)

template = """
system berperan sebagai Gibran Alfi Ananta yang sedang diwawancarai oleh pemberi question.
Berikan jawaban yang sesuai dengan context CV berikut.
Jika informasi tidak tersedia dalam context (CV), nyatakan dengan sopan bahwa Anda tidak dapat menjawab.
Jawab dengan singkat tetapi mudah dipahami.
question: {question}
contezt: {context}
answer:
."""

prompt = ChatPromptTemplate.from_template(template)

retriever = vector_store.as_retriever(
    search_type="mmr", search_kwargs={"k": 1, "fetch_k": 5}
)
# Fungsi untuk gabungkan dokumen
def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

# Buat RAG chain
chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

result = chain.invoke("Perkenalkan diri anda dengan ringkas seperti berbicara kepada manusia")

print(result)

while True:
    question = input("Berikan Pertanyaan ('q' untuk keluar): ")
    print("=" * 64)
    if question.lower() in ["q", "quit"]:
        break
    if not question:
        print("Pertanyaan tidak boleh kosong  ('q' untuk keluar). ")
        continue
    result = chain.invoke(question)
    print(f"Ananta: {result}")
    print("=" * 64)