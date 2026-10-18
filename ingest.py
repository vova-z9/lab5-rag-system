import os
from dotenv import load_dotenv

from langchain_community.document_loaders import DirectoryLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

DOCS_PATH = "docs"
VECTORSTORE_PATH = "vectorstore"

loader = DirectoryLoader(
    DOCS_PATH,
    glob="**/*.docx",
    loader_cls=Docx2txtLoader,
    show_progress=True,
)

documents = loader.load()

for doc in documents:
    doc.metadata["filename"] = os.path.basename(doc.metadata["source"])

print(f"Завантажено документів: {len(documents)}")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    separators=["\n\n", "\n", ". ", " ", ""],
)

chunks = text_splitter.split_documents(documents)
print(f"Створено текстових фрагментів (чанків): {len(chunks)}")

if chunks:
    print(f"Приклад метаданих 1-го фрагмента: {chunks[0].metadata}")

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001"
)

vectorstore = FAISS.from_documents(
    documents=chunks,
    embedding=embeddings,
)

vectorstore.save_local(VECTORSTORE_PATH)
print(f"Векторне сховище успішно збережено у директорію '{VECTORSTORE_PATH}'!")