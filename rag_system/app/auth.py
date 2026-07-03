"""
认证模块 - JWT + bcrypt 用户认证
原理：使用 bcrypt 哈希密码存储，JWT Token 维持会话状态
"""
import os
import json
import hashlib
import secrets
from datetime import datetime, timedelta, timezone
from pathlib import Path
from functools import wraps
from typing import Optional, Dict

from fastapi import HTTPException, Request, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from passlib.context import CryptContext

AUTH_DIR = os.environ.get("AUTH_DIR", "/app/data")
USERS_FILE = os.path.join(AUTH_DIR, "users.json")
SECRET_KEY = os.environ.get("JWT_SECRET_KEY", secrets.token_urlsafe(32))
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.environ.get("JWT_EXPIRE_MINUTES", "480"))

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer(auto_error=False)


def ensure_data_dir():
    os.makedirs(AUTH_DIR, exist_ok=True)
    if not os.path.exists(USERS_FILE):
        admin_user = os.environ.get("ADMIN_USER", "admin")
        admin_pass = os.environ.get("ADMIN_PASSWORD", "admin123")
        users = {
            admin_user: {
                "username": admin_user,
                "password_hash": pwd_context.hash(admin_pass),
                "role": "admin",
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
        }
        with open(USERS_FILE, "w", encoding="utf-8") as f:
            json.dump(users, f, ensure_ascii=False, indent=2)
        print(f"初始管理员用户创建: {admin_user}")
        if admin_pass == "admin123":
            print("警告: 使用默认密码，请通过环境变量 ADMIN_PASSWORD 修改")


def _load_users() -> Dict:
    ensure_data_dir()
    with open(USERS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_users(users: Dict):
    ensure_data_dir()
    tmp = USERS_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(users, f, ensure_ascii=False, indent=2)
    os.replace(tmp, USERS_FILE)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def authenticate_user(username: str, password: str) -> Optional[Dict]:
    users = _load_users()
    user = users.get(username)
    if not user or not verify_password(password, user.get("password_hash", "")):
        return None
    return user


async def get_current_user(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> Optional[Dict]:
    token = None
    if credentials:
        token = credentials.credentials
    if not token:
        token = request.cookies.get("access_token")
    if not token:
        return None

    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            return None
    except JWTError:
        return None

    users = _load_users()
    return users.get(username)


def require_auth(func):
    """要求认证的装饰器"""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        request = kwargs.get("request")
        if request is None:
            for arg in args:
                if isinstance(arg, Request):
                    request = arg
                    break
        if request is None:
            raise HTTPException(status_code=500, detail="无法获取请求上下文")

        user = await get_current_user(request)
        if user is None:
            raise HTTPException(status_code=401, detail="请先登录")
        kwargs["current_user"] = user
        return await func(*args, **kwargs)
    return wrapper


def change_password(username: str, old_password: str, new_password: str) -> bool:
    users = _load_users()
    user = users.get(username)
    if not user or not verify_password(old_password, user.get("password_hash", "")):
        return False
    user["password_hash"] = pwd_context.hash(new_password)
    _save_users(users)
    return True
