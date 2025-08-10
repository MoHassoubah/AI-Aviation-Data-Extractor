from fastapi import FastAPI, UploadFile, File, Form, Header, HTTPException, Depends
from fastapi.responses import JSONResponse
from typing import List
import shutil
import os
import time
from data_extractor import query_pdf

API_KEY = "044491923db599e7daf2613ff3391960"

def verify_token(authorization: str = Header(None)):
    if authorization != f"Bearer {API_KEY}":
        raise HTTPException(status_code=401, detail="Unauthorized")
        
app = FastAPI()


@app.get("/status")
def status():
    return {"status": "AI server is running"}

@app.post("/extract")
async def extract(
    query: str = Form(None),
    files: List[UploadFile] = File(...),  # 👈 Accept multiple files
    _ = Depends(verify_token)
):
    result = query_pdf(files)  # 👈 Pass the uploaded files to your logic
    return result
