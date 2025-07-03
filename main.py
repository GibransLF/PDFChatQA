from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

#ollama

template = """
Anda berperan sebagai Gibran Alfi Ananta yang sedang diwawancarai oleh pemberi question.
Berikan jawaban yang sesuai dengan context CV berikut.
Jika informasi tidak tersedia dalam context (CV), nyatakan dengan sopan bahwa Anda tidak dapat menjawab.
question: {question}
contezt: {context}
answer:
."""

prompt = ChatPromptTemplate.from_template(template)

# Initialize Ollama embeddings
llm = OllamaLLM(model="phi4-mini")
embeddings = OllamaEmbeddings(model="bge-m3")

#load vector store
vector_store = Chroma(
    collection_name="AnantaCV",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db",
)

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
    | llm
    | StrOutputParser()
)
result = chain.invoke("Perkenalkan diri anda dengan ringkas dalam 10 kalimat seperti berbicara kepada manusia")

print(result)
print("=" * 64)

while True:
    question = input("Berikan Pertanyaan ('q' untuk keluar): ")
    if question.lower() in ["q", "quit"]:
        break
    if not question:
        print("Pertanyaan tidak boleh kosong  ('q' untuk keluar). ")
        continue
    result = chain.invoke(question)
    print(result)
    print("=" * 64)