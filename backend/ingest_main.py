import os
import re
from PIL import Image
from paddleocr import PaddleOCR
from pdf2image import convert_from_path
from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader, UnstructuredExcelLoader, UnstructuredImageLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
import fitz

DATA_PATH = "data_main/"
DB_PATH = "chroma_db_main/"

# ✅ Initialize PaddleOCR once
ocr_engine = PaddleOCR(use_angle_cls=True, lang='en')

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

# ✅ OCR fallback for PDFs using PaddleOCR
def ocr_pdf(filepath):
    print("🖼️ Running OCR on PDF pages with PaddleOCR...")
    docs = []
    pdf_doc = fitz.open(filepath)
    for i, page in enumerate(pdf_doc):
        pix = page.get_pixmap()
        img_path = f"temp_page_{i}.png"
        pix.save(img_path)

        result = ocr_engine.ocr(img_path, cls=True)
        if result and result[0]:
            text = "\n".join([line[1][0] for line in result[0]])
            docs.append(Document(
                page_content=clean_text(text),
                metadata={"source": os.path.basename(filepath), "page": i+1, "extraction_method": "ocr"}
            ))
        os.remove(img_path)
    return docs

def ingest_data_main():
    documents = []

    for file in os.listdir(DATA_PATH):
        filepath = os.path.join(DATA_PATH, file)
        print(f"\n📂 Processing: {file}")

        try:
            # ✅ Handle images with PaddleOCR
            if filepath.lower().endswith((".png", ".jpg", ".jpeg")):
                print("🖼️ Running OCR on image with PaddleOCR...")
                result = ocr_engine.ocr(filepath, cls=True)

                if not result or not result[0]:
                    print("❌ No text found in image")
                    continue

                text = "\n".join([line[1][0] for line in result[0]])
                file_docs = [Document(
                    page_content=clean_text(text),
                    metadata={
                        "source": file,
                        "page": 1,
                        "extraction_method": "ocr"
                    }
                )]

            elif filepath.lower().endswith(".pdf"):
                print("📄 Loading PDF with PyPDFLoader...")
                try:
                    loader = get_loader(filepath)
                    if loader is None:
                        raise ValueError("Loader returned None")

                    file_docs = loader.load()

                    if not file_docs:   # fallback if no text layer
                        print("⚠️ No text layer found, using PaddleOCR...")
                        file_docs = ocr_pdf(filepath)
                    else:
                        for doc in file_docs:
                            doc.metadata["extraction_method"] = "text_layer"

                except Exception as e:
                    print(f"⚠️ PyPDFLoader failed, using OCR fallback... Error: {e}")
                    file_docs = ocr_pdf(filepath)

                if not file_docs:
                    print("❌ No content extracted from PDF")
                    continue

            else:
                loader = get_loader(filepath)

                if not loader:
                    print(f"⚠️ Skipping unsupported file: {file}")
                    continue

                print("📄 Loading with loader...")
                try:
                    file_docs = loader.load()
                    for doc in file_docs:
                        doc.metadata["extraction_method"] = "text_layer"
                except Exception as e:
                    print(f"❌ Error loading {file}: {e}")
                    continue

                if not file_docs:
                    print("❌ No content extracted")
                    continue

            # ✅ Add metadata
            for i, doc in enumerate(file_docs):
                doc.metadata["source"] = file
                doc.metadata["page"] = i + 1

            documents.extend(file_docs)
            print(f"✅ Loaded {len(file_docs)} docs")

        except Exception as e:
            print(f"❌ Error processing {file}: {e}")

    # ❌ No documents
    if not documents:
        print("⚠️ No documents ingested.")
        return

    # ✅ Split
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    chunks = splitter.split_documents(documents)

    # ✅ Embeddings + DB
    embeddings = HuggingFaceEmbeddings(
        model_name="C:\\Users\\hp\\Documents\\Multi-fileChatBot\\models\\all-MiniLM-L6-v2"
    )
    db = Chroma.from_documents(chunks, embeddings, persist_directory=DB_PATH)
    db.persist()

    # ✅ Save combined text
    all_text = "\n".join([doc.page_content for doc in documents])

    with open(os.path.join(DATA_PATH, "combined.txt"), "w", encoding="utf-8") as f:
        f.write(all_text)

    print("\n🎉 Ingestion completed successfully!")
