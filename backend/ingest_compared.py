import os
import re 
from PIL import Image
import pytesseract
from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader, UnstructuredExcelLoader , UnstructuredImageLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

DATA_PATH = "data_compared/"
DB_PATH = "chroma_db_compared/"

def get_loader(filepath):
    if filepath.endswith(".pdf"):
        return PyPDFLoader(filepath)
    elif filepath.endswith(".docx"):
        return UnstructuredWordDocumentLoader(filepath)
    elif filepath.endswith(".xlsx"):
        return UnstructuredExcelLoader(filepath)
    elif filepath.endswith((".png", ".jpg", ".jpeg")):
        return UnstructuredImageLoader(filepath)
    else:
        return None

# ✅ Text cleaning function
def clean_text(text: str) -> str:
    text = re.sub(r'\n+', '\n', text)          # remove extra newlines
    text = re.sub(r'\s+', ' ', text)           # normalize spaces
    return text.strip()

def ingest_data_compared():
    documents = []
    for file in os.listdir(DATA_PATH):
        filepath = os.path.join(DATA_PATH, file)
        loader = get_loader(filepath)

        if not loader:
            print(f"⚠️ Skipping unsupported file: {file}")
            continue

        try:
            file_docs = loader.load()

            # ✅ OCR fallback for images
            if filepath.endswith((".png", ".jpg", ".jpeg")):
                text = pytesseract.image_to_string(Image.open(filepath))
                if text.strip():
                    file_docs = [Document(page_content=clean_text(text),
                                          metadata={"source": file, "page": 1})]
                                          
            # ✅ Apply metadata to each doc from this file
            for i, doc in enumerate(file_docs):
                doc.metadata["source"] = file
                doc.metadata["page"] = i + 1

            documents.extend(file_docs)
            print(f"✅ Loaded {len(file_docs)} docs from {file}")

        except Exception as e:
            print(f"❌ Error loading {file}: {e}")

    if not documents:
        print("⚠️ No documents ingested. Check your data_compared folder.")
        return

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
     # 🔑 Apply cleaning before splitting
    chunks = []
    for doc in documents:
        # Clean again just to be safe, then split
        chunks.extend(splitter.split_text(clean_text(doc.page_content)))

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma.from_documents(chunks, embeddings, persist_directory=DB_PATH)
    db.persist()

    all_text = "\n".join([docs.page_content for docs in documents])
    os.makedirs("data_compared", exist_ok=True)
    with open("data_compared/combined.txt", "w", encoding="utf-8") as f:
        f.write(all_text)

    print("✅ Compared documents ingested successfully!")

if __name__ == "__main__":
    ingest_data_compared()
