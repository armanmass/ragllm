import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from scripts.rag_chain import answer_question

class QueryRequest(BaseModel):
    question: str
    top_k: int = 6

class QueryResponse(BaseModel):
    answer: str

app = FastAPI(
    tile="Local RAG API",
    version="1.0.0",
    description="Receive context from local files and answer questions using RAG."
)


@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/answer", response_model=QueryResponse)
def get_answer(request: QueryRequest):
    question = request.question.strip()

    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    
    try:
        answer = answer_question(question, k=request.top_k)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during generation: {str(e)}")
    return {"answer": answer}