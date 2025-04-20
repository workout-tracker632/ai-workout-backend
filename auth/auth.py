from fastapi import APIRouter, HTTPException, Depends
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, EmailStr
from typing import Optional
from datetime import datetime, timedelta
from jose import jwt
from passlib.context import CryptContext
import logging

from db.init_db import users_collection  # Already initialized due to init_db() on import
from bson.objectid import ObjectId

# Router for all authentication endpoints
auth_router = APIRouter()

SECRET_KEY = "123"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Pydantic models
class UserIn(BaseModel):
    username: str
    email: EmailStr
    password: str

class UserOut(BaseModel):
    username: str
    email: EmailStr

class UserInDB(UserOut):
    hashed_password: str

class Token(BaseModel):
    access_token: str
    token_type: str

# Helper functions
def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def get_user_by_username(username: str) -> Optional[UserInDB]:
    user = users_collection.find_one({"username": username})
    if user:
        return UserInDB(
            username=user["username"],
            email=user["email"],
            hashed_password=user["hashed_password"]
        )
    return None

# Register endpoint
@auth_router.post("/register", response_model=UserOut)
def register(user_in: UserIn):
    existing = users_collection.find_one({"username": user_in.username})
    if existing:
        raise HTTPException(status_code=400, detail="Username already registered")

    hashed_pass = get_password_hash(user_in.password)
    new_user = {
        "username": user_in.username,
        "email": user_in.email,
        "hashed_password": hashed_pass
    }
    users_collection.insert_one(new_user)
    return UserOut(username=user_in.username, email=user_in.email)

# Login endpoint
@auth_router.post("/login", response_model=Token)
def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = get_user_by_username(form_data.username)
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(status_code=400, detail="Incorrect username or password")

    token = create_access_token(data={"sub": user.username})
    return {"access_token": token, "token_type": "bearer"}
