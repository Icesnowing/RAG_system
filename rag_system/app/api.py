"""
RAG系统API服务 - FastAPI实现（带用户认证）
原理：JWT + bcrypt 认证，RESTful API 封装 RAG 核心功能
"""
import os
import json
import time
import logging
from pathlib import Path
from typing import Optional, List

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks, Request, Depends
from fastapi.responses import JSONResponse, StreamingResponse, HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.security import HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager

from rag_system import LocalRAGSystem, RAGEvaluator, DOCS_DIR, CHROMA_DB_DIR, DEFAULT_TEST_SET
from auth import (
    ensure_data_dir, authenticate_user, create_access_token,
    get_current_user, change_password, security, ACCESS_TOKEN_EXPIRE_MINUTES,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==================== 请求/响应模型 ====================
class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    stream: bool = Field(False)

class AskResponse(BaseModel):
    question: str
    answer: str
    contexts: Optional[List[str]] = None
    retrieval_time_ms: float
    generation_time_ms: float
    total_time_ms: float

# ==================== 全局实例 ====================
rag_system: LocalRAGSystem = None

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "templates")


def read_template(name: str) -> str:
    with open(os.path.join(TEMPLATE_DIR, name), "r", encoding="utf-8") as f:
        return f.read()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag_system
    ensure_data_dir()

    logger.info("Starting RAG system...")
    rag_system = LocalRAGSystem()
    rag_system.build_or_load_db()

    if rag_system.vector_store:
        try:
            rag_system.warmup_models(preload_chat=True)
            logger.info("Model warmup completed")
        except Exception as e:
            logger.warning(f"Model warmup failed: {e}")
        status = rag_system.get_document_status()
        logger.info(f"Knowledge base: {status['manifest']['total_documents']} docs, "
                    f"{status['manifest']['total_chunks']} chunks")
    else:
        logger.warning("Vector store is empty")

    yield
    logger.info("Shutting down RAG system...")


app = FastAPI(title="RAG Knowledge Base", version="2.0.0", lifespan=lifespan)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# ==================== 认证端点 ====================
@app.post("/api/auth/login")
async def login(request: Request):
    form = await request.form()
    username = form.get("username", "")
    password = form.get("password", "")

    user = authenticate_user(username, password)
    if not user:
        raise HTTPException(status_code=401, detail="用户名或密码错误")

    token = create_access_token(data={"sub": username})
    response = JSONResponse({
        "access_token": token,
        "token_type": "bearer",
        "username": username,
    })
    response.set_cookie(
        key="access_token",
        value=token,
        httponly=True,
        max_age=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        samesite="lax",
    )
    return response


@app.get("/api/auth/me")
async def get_me(user: Optional[dict] = Depends(get_current_user)):
    if not user:
        raise HTTPException(status_code=401, detail="未登录")
    return {"username": user["username"], "role": user.get("role")}


@app.post("/api/auth/logout")
async def logout():
    response = JSONResponse({"message": "已退出"})
    response.delete_cookie("access_token")
    return response


@app.post("/api/auth/change-password")
async def change_pwd(request: Request, user: Optional[dict] = Depends(get_current_user)):
    if not user:
        raise HTTPException(status_code=401, detail="请先登录")

    form = await request.form()
    old_pw = form.get("old_password", "")
    new_pw = form.get("new_password", "")

    if not old_pw or not new_pw:
        raise HTTPException(status_code=400, detail="密码不能为空")
    if len(new_pw) < 4:
        raise HTTPException(status_code=400, detail="新密码长度至少4位")

    if change_password(user["username"], old_pw, new_pw):
        return {"message": "密码修改成功"}
    raise HTTPException(status_code=400, detail="旧密码错误")


# ==================== 页面路由 ====================
@app.get("/login")
async def login_page():
    return HTMLResponse(read_template("login.html"))


@app.get("/app")
async def app_page(user: Optional[dict] = Depends(get_current_user)):
    if not user:
        return RedirectResponse(url="/login")
    return HTMLResponse(read_template("index.html"))


@app.get("/")
async def root():
    return RedirectResponse(url="/login")


# ==================== 健康检查 ====================
@app.get("/api/health")
async def health_check():
    rerank_ok = False
    if rag_system and rag_system._rerank_model is not None:
        rerank_ok = True
    elif rag_system:
        try:
            _ = rag_system.rerank_model
            rerank_ok = True
        except Exception:
            pass

    return {
        "status": "healthy",
        "vector_store_ready": rag_system is not None and rag_system.vector_store is not None,
        "rag_chain_ready": rag_system is not None and rag_system.rag_chain is not None,
        "rerank_available": rerank_ok,
    }


# ==================== 问答接口 ====================
@app.post("/api/ask", response_model=AskResponse)
async def ask_question(request: AskRequest, user: Optional[dict] = Depends(get_current_user)):
    if not rag_system or not rag_system.vector_store:
        raise HTTPException(status_code=503, detail="向量库未初始化")

    if not request.question or not request.question.strip():
        raise HTTPException(status_code=400, detail="问题不能为空")

    try:
        t_start = time.time()
        t_retrieve = time.time()
        contexts = rag_system._retrieve_with_rerank(request.question)
        retrieve_time = (time.time() - t_retrieve) * 1000

        t_generate = time.time()
        answer = rag_system._generate_answer(request.question, contexts)
        generate_time = (time.time() - t_generate) * 1000

        return AskResponse(
            question=request.question,
            answer=answer if isinstance(answer, str) else str(answer),
            contexts=[d.page_content for d in contexts[:5]],
            retrieval_time_ms=retrieve_time,
            generation_time_ms=generate_time,
            total_time_ms=(time.time() - t_start) * 1000,
        )
    except Exception as e:
        logger.error(f"问答失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/ask/stream")
async def ask_question_stream(request: AskRequest, user: Optional[dict] = Depends(get_current_user)):
    if not rag_system or not rag_system.vector_store:
        raise HTTPException(status_code=503, detail="向量库未初始化")
    if not request.question or not request.question.strip():
        raise HTTPException(status_code=400, detail="问题不能为空")

    async def generate():
        try:
            contexts = rag_system._retrieve_with_rerank(request.question)
            context_texts = [d.page_content for d in contexts[:5]]

            prompt = rag_system._get_rag_prompt()
            chain = prompt | rag_system.chat_model

            for chunk in chain.stream({"input": request.question, "context": contexts}):
                if hasattr(chunk, "content"):
                    yield f"data: {json.dumps({'content': chunk.content})}\n\n"

            yield f"data: {json.dumps({'contexts': context_texts})}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


# ==================== 文档管理接口 ====================
@app.post("/api/documents/upload")
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    user: Optional[dict] = Depends(get_current_user),
):
    if not rag_system or not rag_system.vector_store:
        raise HTTPException(status_code=503, detail="向量库未初始化")

    allowed = {".txt", ".pdf", ".docx", ".doc"}
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed:
        raise HTTPException(status_code=400, detail=f"不支持的文件类型: {ext}")

    os.makedirs(DOCS_DIR, exist_ok=True)
    file_path = os.path.join(DOCS_DIR, file.filename)
    content = await file.read()
    with open(file_path, "wb") as f:
        f.write(content)

    def add_task():
        rag_system.add_document(file_path)

    background_tasks.add_task(add_task)
    return {"message": f"{file.filename} 已上传，正在后台处理索引"}


@app.delete("/api/documents/{filename}")
async def delete_document(filename: str, user: Optional[dict] = Depends(get_current_user)):
    if not rag_system or not rag_system.vector_store:
        raise HTTPException(status_code=503, detail="向量库未初始化")

    file_path = os.path.join(DOCS_DIR, filename)
    success = rag_system.remove_document(file_path)
    if success:
        return {"message": f"文档 {filename} 已删除"}
    raise HTTPException(status_code=404, detail=f"文档 {filename} 不存在")


@app.post("/api/documents/sync")
async def sync_documents(user: Optional[dict] = Depends(get_current_user)):
    if not rag_system or not rag_system.vector_store:
        raise HTTPException(status_code=503, detail="向量库未初始化")
    return rag_system.sync_knowledge_base()


@app.get("/api/documents/status")
async def get_document_status(user: Optional[dict] = Depends(get_current_user)):
    if not rag_system or not rag_system.vector_store:
        return {"manifest": {"total_documents": 0, "total_chunks": 0, "vector_count": 0}, "documents": []}
    return rag_system.get_document_status()


# ==================== 评估接口 ====================
@app.post("/api/evaluate")
async def run_evaluation(
    background_tasks: BackgroundTasks,
    testset_path: str = Form(DEFAULT_TEST_SET),
    limit: Optional[int] = Form(None),
    user: Optional[dict] = Depends(get_current_user),
):
    if not rag_system or not rag_system.vector_store:
        raise HTTPException(status_code=503, detail="向量库未初始化")

    evaluator = RAGEvaluator(rag_system)
    background_tasks.add_task(lambda: evaluator.run_evaluation(testset_path, limit=limit))
    return {"message": "评估任务已启动", "testset": testset_path, "limit": limit}
