# PDFChatQA
Chat Bot menggunakan Ollama dengan RAG PDF

## Cara penggunaan dasar
install python, saya menggunakan Python 3.12.7

install kebutuhan jalankan perintah
```
pip install -r .\requirements.txt
```

install [Ollama](https://ollama.com/)

terbagi menjadi 3 pembuatan embending vektor, lokal, dan api

### vektor.py
vektor digunakan untuk mengubah file pdf menjadi sebuat vector hasil dari embending

saya menggunakan **bge-m3** sebagai contoh, bisa dengan model embending yang lain. contoh script
```
embeddings = OllamaEmbeddings(
    model="bge-m3",
)
```

jalankan ollama dan install terlebih dahulu [bge-m3](https://ollama.com/library/bge-m3) di terminal dengan perintah
```
ollama pull bge-m3
```

path penyimpanan pdf bisa dibuah pada script
```
loader = PyPDFLoader("./pdf/cv.pdf")
```
sebagai contoh adalah **PDFChatQA/pdf/cv.pdf**

lalu jalankan perintah ini untuk membuat vector

```
python vector.py
```
yang akan menghasilkan database lokal dari chromaDB dengan folder bernama **chroma_langchain_db**


selanjutnya ke bagian main.py (lokal) atau api.py

### Lokal (main.py)
saya menggunakan model llm **phi4-mini** seagai contoh, bisa gunakan model yang lain. contoh Script 
```
llm = OllamaLLM(model="phi4-mini")
embeddings = OllamaEmbeddings(model="bge-m3")
```

install model [phi4-mini](https://ollama.com/library/phi4-mini) dengan ollama dengan menjalankan perintah
```
ollama pull phi4-mini
```

dan jalankan model phi4-mini dengan perintah

```
ollama run phi4-mini
```

lalu jalankan main.py dengan perintah
```
python main.py
```

### API (api.py)
anda perlu memiliki API KEY untuk menggunakan ini

API sebagai contoh ini menggunakan Gemini dengan model **gemini-2.5-flash**
```
model = init_chat_model("gemini-2.5-flash", model_provider="google_genai")
embeddings = OllamaEmbeddings(model="bge-m3")
```

untuk mengisi API bisa tambahkan .env dari contoh .env.example dan isi API_KEY yang dimiliki

```
GEMINI_API_KEY="" # Your Gemini API Key

```

lalu jalankan api.py dengan perintah perintah
```
python api.py
```


## Tambahan
ubah template perintah yang berada di main.py dan api.py untuk menyesuaikan konteks
```
template = """
system berperan sebagai Gibran Alfi Ananta yang sedang diwawancarai oleh pemberi question.
Berikan jawaban yang sesuai dengan context CV berikut.
Jika informasi tidak tersedia dalam context (CV), nyatakan dengan sopan bahwa Anda tidak dapat menjawab.
Jawab dengan singkat tetapi mudah dipahami.
question: {question}
context: {context}
answer:
."""
```
