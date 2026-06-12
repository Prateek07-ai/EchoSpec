from fastapi import FastAPI, UploadFile, File
import shutil
from backend.ingest_main import ingest_data_main
from backend.ingest_compared import ingest_data_compared
from backend.query import sop_check, ask

app = FastAPI(title="SOP Chatbot API")

# --- Upload Reference PDF ---
@app.post("/upload/reference/")
async def upload_reference(file: UploadFile = File(...)):
    file_path = f"data_main/{file.filename}"
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    ingest_data_main()
    return {"status": "success", "file": file.filename, "db": "chroma_db_main"}

# --- Upload Checking PDF ---
@app.post("/upload/checking/")
async def upload_checking(file: UploadFile = File(...)):
    file_path = f"data_compared/{file.filename}"
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    ingest_data_compared()
    return {"status": "success", "file": file.filename, "db": "chroma_db_compared"}

# --- Run SOP Check ---
@app.post("/sop_check/")
async def run_sop_check(queries: list[str]):
    results = sop_check(queries, ref_db="chroma_db_main", check_db="chroma_db_compared", use_qwen=True)
    return results

# --- Ask Query ---
@app.get("/ask/")
async def ask_query(query: str, db: str = "chroma_db_main"):
    answer, refs = ask(query, db_path=db, use_qwen=True)
    return {"query": query, "answer": answer, "references": refs}
