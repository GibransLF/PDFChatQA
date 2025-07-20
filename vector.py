from langchain_community.document_loaders import PyPDFLoader
import asyncio
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

# load pdf
async def main():
    loader = PyPDFLoader("./pdf/cv.pdf")
    docs = []
    async for page in loader.alazy_load():
        docs.append(page)
    return docs
docs = asyncio.run(main())

#print(docs)

print(f"Total characters: {len(docs[0].page_content)}")

#splitting documents
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # chunk size (characters)
    chunk_overlap=200,  # chunk overlap (characters)
    add_start_index=True,  # track index in original document
)
all_splits = text_splitter.split_documents(docs)

#print(all_splits)
print(f"Split blog post into {len(all_splits)} sub-documents.")

# Initialize Ollama embeddings
embeddings = OllamaEmbeddings(
    model="bge-m3",
)

#embedding text to vector
vector_store = Chroma(
    collection_name="AnantaCV",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db",
)

document_ids = vector_store.add_documents(documents=all_splits)
print(document_ids[:10])

print("done...")
