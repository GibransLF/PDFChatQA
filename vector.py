from langchain_community.document_loaders import PyPDFLoader
import asyncio
from langchain_ollama import OllamaEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore

# load pdf
async def main():
    loader = PyPDFLoader("./pdf/cv.pdf")
    pages = []
    async for page in loader.alazy_load():
        pages.append(page)
    return pages
pages = asyncio.run(main())


# Initialize Ollama embeddings
embeddings = OllamaEmbeddings(
    model="bge-m3",
)

text = [page.page_content for page in pages]
vectorstore = InMemoryVectorStore.from_texts(
    text,
    embedding=embeddings,
)

single_vector = embeddings.embed_query(text[0])

print(str(single_vector)[:100])