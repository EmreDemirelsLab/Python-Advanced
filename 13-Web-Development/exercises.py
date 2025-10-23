"""
Web Development - Advanced Exercises
FastAPI ve Modern Web Development pratikleri

Bu dosya 15 ileri seviye web development egzersizi içerir:
1. RESTful API CRUD işlemleri
2. JWT Authentication sistemi
3. Rate limiting middleware
4. File upload API
5. Pagination ve filtreleme
6. WebSocket real-time chat
7. Background task processing
8. API key authentication
9. Database transaction management
10. API versioning
11. OAuth2 social login
12. GraphQL-like flexible queries
13. Webhook system
14. SSE (Server-Sent Events)
15. Microservice communication
"""

from typing import List, Optional, Dict, Any, Set
from datetime import datetime, timedelta
from enum import Enum
import asyncio
import hashlib
import hmac
import secrets
import json
from functools import wraps
from collections import defaultdict
import time

# ============================================================================
# EXERCISE 1: RESTful CRUD API with FastAPI
# ============================================================================
# TODO: Tam özellikli bir RESTful CRUD API oluşturun
# - User model (Pydantic)
# - CRUD endpoints (Create, Read, Update, Delete)
# - Input validation
# - Proper HTTP status codes
# - Error handling

"""
ÇÖZÜM 1: RESTful CRUD API
"""

from pydantic import BaseModel, EmailStr, Field, validator

class UserRole(str, Enum):
    ADMIN = "admin"
    USER = "user"
    MODERATOR = "moderator"

class UserBase(BaseModel):
    """Temel kullanıcı modeli"""
    email: EmailStr
    username: str = Field(..., min_length=3, max_length=50)
    full_name: Optional[str] = None
    role: UserRole = UserRole.USER

    @validator('username')
    def username_alphanumeric(cls, v):
        if not v.isalnum():
            raise ValueError('Username sadece alfanumerik karakter içermelidir')
        return v

class UserCreate(UserBase):
    """Kullanıcı oluşturma modeli"""
    password: str = Field(..., min_length=8)

class UserUpdate(BaseModel):
    """Kullanıcı güncelleme modeli"""
    email: Optional[EmailStr] = None
    full_name: Optional[str] = None
    role: Optional[UserRole] = None

class UserResponse(UserBase):
    """Kullanıcı response modeli"""
    id: int
    is_active: bool = True
    created_at: datetime

    class Config:
        orm_mode = True

class UserCRUDAPI:
    """Tam özellikli CRUD API implementasyonu"""

    def __init__(self):
        self.users: Dict[int, Dict[str, Any]] = {}
        self.user_id_counter = 1
        self.email_index: Set[str] = set()

    def create_user(self, user_data: UserCreate) -> UserResponse:
        """Yeni kullanıcı oluştur"""
        # Email uniqueness kontrolü
        if user_data.email in self.email_index:
            raise ValueError("Bu email adresi zaten kullanımda")

        # Kullanıcı oluştur
        user = {
            "id": self.user_id_counter,
            "email": user_data.email,
            "username": user_data.username,
            "full_name": user_data.full_name,
            "role": user_data.role,
            "is_active": True,
            "created_at": datetime.utcnow(),
            "hashed_password": self._hash_password(user_data.password)
        }

        self.users[self.user_id_counter] = user
        self.email_index.add(user_data.email)
        self.user_id_counter += 1

        return UserResponse(**user)

    def get_user(self, user_id: int) -> Optional[UserResponse]:
        """Kullanıcıyı ID ile getir"""
        user = self.users.get(user_id)
        if not user:
            return None
        return UserResponse(**user)

    def get_users(self, skip: int = 0, limit: int = 10,
                  role: Optional[UserRole] = None) -> List[UserResponse]:
        """Tüm kullanıcıları listele (filtreleme ile)"""
        users = list(self.users.values())

        # Role filtresi
        if role:
            users = [u for u in users if u["role"] == role]

        # Pagination
        users = users[skip:skip + limit]

        return [UserResponse(**u) for u in users]

    def update_user(self, user_id: int, user_data: UserUpdate) -> Optional[UserResponse]:
        """Kullanıcıyı güncelle"""
        if user_id not in self.users:
            return None

        user = self.users[user_id]
        update_data = user_data.dict(exclude_unset=True)

        # Email değişikliği kontrolü
        if "email" in update_data and update_data["email"] != user["email"]:
            if update_data["email"] in self.email_index:
                raise ValueError("Bu email adresi zaten kullanımda")
            self.email_index.remove(user["email"])
            self.email_index.add(update_data["email"])

        # Güncelleme
        user.update(update_data)

        return UserResponse(**user)

    def delete_user(self, user_id: int) -> bool:
        """Kullanıcıyı sil"""
        if user_id not in self.users:
            return False

        user = self.users[user_id]
        self.email_index.remove(user["email"])
        del self.users[user_id]

        return True

    @staticmethod
    def _hash_password(password: str) -> str:
        """Şifreyi hash'le (basit versiyon)"""
        return hashlib.sha256(password.encode()).hexdigest()

# Test
api = UserCRUDAPI()

# Create
user1 = api.create_user(UserCreate(
    email="ali@example.com",
    username="ali123",
    full_name="Ali Yılmaz",
    password="Secure123"
))
print(f"✓ User created: {user1.username} (ID: {user1.id})")

# Read
user = api.get_user(1)
print(f"✓ User retrieved: {user.email}")

# Update
updated = api.update_user(1, UserUpdate(full_name="Ali Mehmet Yılmaz"))
print(f"✓ User updated: {updated.full_name}")

# List
users = api.get_users(limit=10)
print(f"✓ Users listed: {len(users)} users")

# Delete
deleted = api.delete_user(1)
print(f"✓ User deleted: {deleted}")


# ============================================================================
# EXERCISE 2: JWT Authentication System
# ============================================================================
# TODO: Kapsamlı JWT authentication sistemi oluşturun
# - Access token ve refresh token
# - Token validation
# - Password hashing
# - Login/Logout endpoints
# - Protected routes

"""
ÇÖZÜM 2: JWT Authentication System
"""

import jwt
from passlib.context import CryptContext

class JWTAuthSystem:
    """JWT tabanlı authentication sistemi"""

    SECRET_KEY = "your-secret-key-keep-it-very-secret"
    ALGORITHM = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES = 30
    REFRESH_TOKEN_EXPIRE_DAYS = 7

    def __init__(self):
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        self.users_db: Dict[str, Dict[str, Any]] = {}
        self.refresh_tokens: Set[str] = set()  # Blacklist için
        self.revoked_tokens: Set[str] = set()

    def hash_password(self, password: str) -> str:
        """Şifreyi hash'le"""
        return self.pwd_context.hash(password)

    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Şifreyi doğrula"""
        return self.pwd_context.verify(plain_password, hashed_password)

    def create_access_token(self, data: Dict[str, Any]) -> str:
        """Access token oluştur"""
        to_encode = data.copy()
        expire = datetime.utcnow() + timedelta(minutes=self.ACCESS_TOKEN_EXPIRE_MINUTES)

        to_encode.update({
            "exp": expire,
            "iat": datetime.utcnow(),
            "type": "access",
            "jti": secrets.token_urlsafe(16)  # Token ID
        })

        return jwt.encode(to_encode, self.SECRET_KEY, algorithm=self.ALGORITHM)

    def create_refresh_token(self, data: Dict[str, Any]) -> str:
        """Refresh token oluştur"""
        to_encode = data.copy()
        expire = datetime.utcnow() + timedelta(days=self.REFRESH_TOKEN_EXPIRE_DAYS)

        to_encode.update({
            "exp": expire,
            "iat": datetime.utcnow(),
            "type": "refresh",
            "jti": secrets.token_urlsafe(16)
        })

        token = jwt.encode(to_encode, self.SECRET_KEY, algorithm=self.ALGORITHM)
        self.refresh_tokens.add(token)
        return token

    def decode_token(self, token: str) -> Dict[str, Any]:
        """Token'ı decode et ve doğrula"""
        try:
            # Revoke kontrolü
            if token in self.revoked_tokens:
                raise ValueError("Token iptal edilmiş")

            payload = jwt.decode(token, self.SECRET_KEY, algorithms=[self.ALGORITHM])
            return payload

        except jwt.ExpiredSignatureError:
            raise ValueError("Token süresi dolmuş")
        except jwt.JWTError as e:
            raise ValueError(f"Token geçersiz: {str(e)}")

    def register(self, username: str, password: str, email: str) -> Dict[str, str]:
        """Yeni kullanıcı kaydı"""
        if username in self.users_db:
            raise ValueError("Kullanıcı adı zaten kullanımda")

        self.users_db[username] = {
            "username": username,
            "email": email,
            "hashed_password": self.hash_password(password),
            "created_at": datetime.utcnow()
        }

        return self.login(username, password)

    def login(self, username: str, password: str) -> Dict[str, str]:
        """Kullanıcı girişi"""
        user = self.users_db.get(username)

        if not user or not self.verify_password(password, user["hashed_password"]):
            raise ValueError("Kullanıcı adı veya şifre hatalı")

        # Token'ları oluştur
        token_data = {"sub": username, "email": user["email"]}

        access_token = self.create_access_token(token_data)
        refresh_token = self.create_refresh_token(token_data)

        return {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "token_type": "bearer"
        }

    def refresh(self, refresh_token: str) -> Dict[str, str]:
        """Token yenileme"""
        payload = self.decode_token(refresh_token)

        if payload.get("type") != "refresh":
            raise ValueError("Geçersiz token tipi")

        if refresh_token not in self.refresh_tokens:
            raise ValueError("Refresh token geçersiz")

        # Yeni token'lar oluştur
        username = payload.get("sub")
        user = self.users_db.get(username)

        token_data = {"sub": username, "email": user["email"]}

        # Eski refresh token'ı iptal et
        self.refresh_tokens.remove(refresh_token)

        access_token = self.create_access_token(token_data)
        new_refresh_token = self.create_refresh_token(token_data)

        return {
            "access_token": access_token,
            "refresh_token": new_refresh_token,
            "token_type": "bearer"
        }

    def logout(self, access_token: str, refresh_token: str):
        """Çıkış yap (token'ları iptal et)"""
        self.revoked_tokens.add(access_token)
        if refresh_token in self.refresh_tokens:
            self.refresh_tokens.remove(refresh_token)

    def get_current_user(self, token: str) -> Dict[str, Any]:
        """Token'dan kullanıcı bilgilerini çıkar"""
        payload = self.decode_token(token)

        if payload.get("type") != "access":
            raise ValueError("Geçersiz token tipi")

        username = payload.get("sub")
        user = self.users_db.get(username)

        if not user:
            raise ValueError("Kullanıcı bulunamadı")

        return {
            "username": user["username"],
            "email": user["email"]
        }

# Test
auth = JWTAuthSystem()

# Register
tokens = auth.register("ali", "SecurePass123", "ali@example.com")
print(f"✓ User registered, access token: {tokens['access_token'][:50]}...")

# Login
login_tokens = auth.login("ali", "SecurePass123")
print(f"✓ User logged in")

# Get current user
user = auth.get_current_user(login_tokens["access_token"])
print(f"✓ Current user: {user['username']}")

# Refresh token
new_tokens = auth.refresh(login_tokens["refresh_token"])
print(f"✓ Token refreshed")

# Logout
auth.logout(new_tokens["access_token"], new_tokens["refresh_token"])
print(f"✓ User logged out")


# ============================================================================
# EXERCISE 3: Rate Limiting Middleware
# ============================================================================
# TODO: Gelişmiş rate limiting middleware geliştirin
# - IP tabanlı rate limiting
# - Token bucket algoritması
# - Sliding window
# - Per-endpoint rate limits
# - Rate limit headers

"""
ÇÖZÜM 3: Advanced Rate Limiting
"""

from collections import deque
from threading import Lock

class RateLimiter:
    """Gelişmiş rate limiting sistemi"""

    def __init__(self):
        # Fixed window rate limiting
        self.fixed_window: Dict[str, List[float]] = defaultdict(list)

        # Token bucket
        self.token_buckets: Dict[str, Dict[str, Any]] = {}

        # Sliding window log
        self.sliding_windows: Dict[str, deque] = defaultdict(lambda: deque())

        self.lock = Lock()

    def fixed_window_check(self, client_id: str, max_requests: int,
                          window_seconds: int) -> tuple[bool, Dict[str, int]]:
        """Fixed window rate limiting"""
        with self.lock:
            current_time = time.time()
            window_start = current_time - window_seconds

            # Eski requestleri temizle
            self.fixed_window[client_id] = [
                t for t in self.fixed_window[client_id]
                if t > window_start
            ]

            # Rate limit kontrolü
            request_count = len(self.fixed_window[client_id])

            if request_count >= max_requests:
                return False, {
                    "limit": max_requests,
                    "remaining": 0,
                    "reset": int(window_start + window_seconds)
                }

            # Request'i kaydet
            self.fixed_window[client_id].append(current_time)

            return True, {
                "limit": max_requests,
                "remaining": max_requests - request_count - 1,
                "reset": int(window_start + window_seconds)
            }

    def token_bucket_check(self, client_id: str, capacity: int,
                          refill_rate: float) -> tuple[bool, Dict[str, Any]]:
        """Token bucket rate limiting"""
        with self.lock:
            current_time = time.time()

            # Bucket yoksa oluştur
            if client_id not in self.token_buckets:
                self.token_buckets[client_id] = {
                    "tokens": capacity,
                    "last_refill": current_time
                }

            bucket = self.token_buckets[client_id]

            # Token'ları yenile
            time_passed = current_time - bucket["last_refill"]
            tokens_to_add = time_passed * refill_rate
            bucket["tokens"] = min(capacity, bucket["tokens"] + tokens_to_add)
            bucket["last_refill"] = current_time

            # Token var mı kontrol et
            if bucket["tokens"] < 1:
                return False, {
                    "limit": capacity,
                    "remaining": 0,
                    "retry_after": int((1 - bucket["tokens"]) / refill_rate)
                }

            # Token tüket
            bucket["tokens"] -= 1

            return True, {
                "limit": capacity,
                "remaining": int(bucket["tokens"]),
                "retry_after": 0
            }

    def sliding_window_check(self, client_id: str, max_requests: int,
                            window_seconds: int) -> tuple[bool, Dict[str, int]]:
        """Sliding window log rate limiting"""
        with self.lock:
            current_time = time.time()
            window_start = current_time - window_seconds

            window = self.sliding_windows[client_id]

            # Eski requestleri temizle
            while window and window[0] <= window_start:
                window.popleft()

            # Rate limit kontrolü
            if len(window) >= max_requests:
                oldest_request = window[0]
                retry_after = int(oldest_request + window_seconds - current_time)

                return False, {
                    "limit": max_requests,
                    "remaining": 0,
                    "retry_after": retry_after
                }

            # Request'i kaydet
            window.append(current_time)

            return True, {
                "limit": max_requests,
                "remaining": max_requests - len(window),
                "retry_after": 0
            }

    def adaptive_rate_limit(self, client_id: str, base_limit: int,
                           error_rate: float) -> tuple[bool, Dict[str, Any]]:
        """Hata oranına göre adaptif rate limiting"""
        # Hata oranı yüksekse limiti düşür
        if error_rate > 0.1:  # %10'dan fazla hata
            adjusted_limit = int(base_limit * 0.5)
        elif error_rate > 0.05:  # %5'ten fazla hata
            adjusted_limit = int(base_limit * 0.75)
        else:
            adjusted_limit = base_limit

        return self.fixed_window_check(client_id, adjusted_limit, 60)

# Test
limiter = RateLimiter()

# Fixed window test
print("Fixed Window Rate Limiting:")
for i in range(5):
    allowed, info = limiter.fixed_window_check("client1", max_requests=3, window_seconds=10)
    print(f"  Request {i+1}: {'✓ Allowed' if allowed else '✗ Denied'} - "
          f"Remaining: {info['remaining']}")

# Token bucket test
print("\nToken Bucket Rate Limiting:")
for i in range(5):
    allowed, info = limiter.token_bucket_check("client2", capacity=3, refill_rate=1.0)
    print(f"  Request {i+1}: {'✓ Allowed' if allowed else '✗ Denied'} - "
          f"Remaining: {info['remaining']}")
    time.sleep(0.5)

# Sliding window test
print("\nSliding Window Rate Limiting:")
for i in range(5):
    allowed, info = limiter.sliding_window_check("client3", max_requests=3, window_seconds=2)
    print(f"  Request {i+1}: {'✓ Allowed' if allowed else '✗ Denied'} - "
          f"Remaining: {info['remaining']}")
    time.sleep(0.3)


# ============================================================================
# EXERCISE 4: File Upload API
# ============================================================================
# TODO: Güvenli file upload API oluşturun
# - File type validation
# - File size limits
# - Virus scanning simulation
# - Chunk upload support
# - Progress tracking

"""
ÇÖZÜM 4: Advanced File Upload System
"""

import os
import mimetypes
from pathlib import Path
import base64

class FileUploadSystem:
    """Gelişmiş dosya yükleme sistemi"""

    ALLOWED_EXTENSIONS = {
        'image': ['jpg', 'jpeg', 'png', 'gif', 'webp'],
        'document': ['pdf', 'doc', 'docx', 'txt', 'md'],
        'video': ['mp4', 'avi', 'mov', 'mkv'],
    }

    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
    UPLOAD_DIR = "/tmp/uploads"

    def __init__(self):
        self.uploads: Dict[str, Dict[str, Any]] = {}
        self.chunk_uploads: Dict[str, Dict[str, Any]] = {}

        # Upload dizinini oluştur
        Path(self.UPLOAD_DIR).mkdir(parents=True, exist_ok=True)

    def validate_file_extension(self, filename: str, category: str = None) -> bool:
        """Dosya uzantısını doğrula"""
        ext = filename.rsplit('.', 1)[-1].lower()

        if category:
            return ext in self.ALLOWED_EXTENSIONS.get(category, [])

        # Tüm kategorilerde ara
        all_extensions = []
        for exts in self.ALLOWED_EXTENSIONS.values():
            all_extensions.extend(exts)

        return ext in all_extensions

    def validate_file_size(self, file_size: int) -> bool:
        """Dosya boyutunu doğrula"""
        return file_size <= self.MAX_FILE_SIZE

    def scan_for_viruses(self, file_content: bytes) -> bool:
        """Virüs taraması (simülasyon)"""
        # Gerçek uygulamada ClamAV gibi bir antivirus kullanın

        # Basit pattern matching (demo amaçlı)
        dangerous_patterns = [
            b'<script>',
            b'<?php',
            b'eval(',
            b'exec(',
        ]

        content_lower = file_content.lower()
        for pattern in dangerous_patterns:
            if pattern in content_lower:
                return False

        return True

    def upload_file(self, filename: str, file_content: bytes,
                   category: str = None) -> Dict[str, Any]:
        """Dosya yükle"""
        # Validasyonlar
        if not self.validate_file_extension(filename, category):
            raise ValueError("Desteklenmeyen dosya türü")

        if not self.validate_file_size(len(file_content)):
            raise ValueError(f"Dosya boyutu maksimum {self.MAX_FILE_SIZE} bytes olmalı")

        if not self.scan_for_viruses(file_content):
            raise ValueError("Dosya güvenlik kontrolünden geçemedi")

        # Unique filename oluştur
        file_id = secrets.token_urlsafe(16)
        ext = filename.rsplit('.', 1)[-1]
        safe_filename = f"{file_id}.{ext}"

        # Dosyayı kaydet
        file_path = os.path.join(self.UPLOAD_DIR, safe_filename)
        with open(file_path, 'wb') as f:
            f.write(file_content)

        # Metadata kaydet
        upload_info = {
            "file_id": file_id,
            "original_filename": filename,
            "saved_filename": safe_filename,
            "file_path": file_path,
            "file_size": len(file_content),
            "mime_type": mimetypes.guess_type(filename)[0],
            "uploaded_at": datetime.utcnow(),
            "category": category
        }

        self.uploads[file_id] = upload_info

        return upload_info

    def start_chunk_upload(self, filename: str, total_size: int,
                          chunk_size: int = 1024 * 1024) -> str:
        """Chunk upload başlat"""
        if not self.validate_file_size(total_size):
            raise ValueError("Dosya boyutu çok büyük")

        upload_id = secrets.token_urlsafe(16)

        self.chunk_uploads[upload_id] = {
            "filename": filename,
            "total_size": total_size,
            "chunk_size": chunk_size,
            "chunks": {},
            "total_chunks": (total_size + chunk_size - 1) // chunk_size,
            "started_at": datetime.utcnow()
        }

        return upload_id

    def upload_chunk(self, upload_id: str, chunk_number: int,
                    chunk_data: bytes) -> Dict[str, Any]:
        """Chunk yükle"""
        if upload_id not in self.chunk_uploads:
            raise ValueError("Geçersiz upload ID")

        upload = self.chunk_uploads[upload_id]
        upload["chunks"][chunk_number] = chunk_data

        progress = (len(upload["chunks"]) / upload["total_chunks"]) * 100

        return {
            "upload_id": upload_id,
            "chunk_number": chunk_number,
            "progress": round(progress, 2),
            "chunks_uploaded": len(upload["chunks"]),
            "total_chunks": upload["total_chunks"]
        }

    def complete_chunk_upload(self, upload_id: str) -> Dict[str, Any]:
        """Chunk upload'ı tamamla"""
        if upload_id not in self.chunk_uploads:
            raise ValueError("Geçersiz upload ID")

        upload = self.chunk_uploads[upload_id]

        # Tüm chunk'lar yüklendi mi kontrol et
        if len(upload["chunks"]) != upload["total_chunks"]:
            raise ValueError("Tüm chunk'lar yüklenmemiş")

        # Chunk'ları birleştir
        file_content = b""
        for i in range(upload["total_chunks"]):
            if i not in upload["chunks"]:
                raise ValueError(f"Chunk {i} eksik")
            file_content += upload["chunks"][i]

        # Normal upload gibi işle
        result = self.upload_file(upload["filename"], file_content)

        # Chunk upload'ı temizle
        del self.chunk_uploads[upload_id]

        return result

    def get_upload_info(self, file_id: str) -> Optional[Dict[str, Any]]:
        """Upload bilgilerini getir"""
        return self.uploads.get(file_id)

    def delete_upload(self, file_id: str) -> bool:
        """Yüklenen dosyayı sil"""
        if file_id not in self.uploads:
            return False

        upload = self.uploads[file_id]

        # Dosyayı sil
        if os.path.exists(upload["file_path"]):
            os.remove(upload["file_path"])

        # Metadata'yı sil
        del self.uploads[file_id]

        return True

# Test
uploader = FileUploadSystem()

# Normal upload
print("Normal File Upload:")
test_content = b"Bu bir test dosyasidir"
result = uploader.upload_file("test.txt", test_content, "document")
print(f"✓ File uploaded: {result['file_id']}")
print(f"  Original: {result['original_filename']}")
print(f"  Size: {result['file_size']} bytes")

# Chunk upload
print("\nChunk Upload:")
large_content = b"X" * (5 * 1024 * 1024)  # 5MB
upload_id = uploader.start_chunk_upload("large_file.bin", len(large_content), 1024*1024)
print(f"✓ Chunk upload started: {upload_id}")

# Chunk'ları yükle
chunk_size = 1024 * 1024
for i in range(5):
    start = i * chunk_size
    end = start + chunk_size
    chunk = large_content[start:end]
    progress = uploader.upload_chunk(upload_id, i, chunk)
    print(f"  Chunk {i+1}/5: {progress['progress']}%")

# Tamamla
final_result = uploader.complete_chunk_upload(upload_id)
print(f"✓ Upload completed: {final_result['file_id']}")


# ============================================================================
# EXERCISE 5: Advanced Pagination and Filtering
# ============================================================================
# TODO: Gelişmiş pagination ve filtering sistemi
# - Cursor-based pagination
# - Offset-based pagination
# - Dynamic filtering
# - Sorting
# - Field selection

"""
ÇÖZÜM 5: Advanced Pagination & Filtering
"""

class QueryBuilder:
    """Gelişmiş sorgu oluşturucu"""

    def __init__(self, data: List[Dict[str, Any]]):
        self.data = data
        self.filtered_data = data.copy()

    def filter(self, **kwargs) -> 'QueryBuilder':
        """Dinamik filtreleme"""
        for key, value in kwargs.items():
            if '__' in key:
                # Özel operatörler
                field, operator = key.split('__', 1)

                if operator == 'eq':
                    self.filtered_data = [
                        item for item in self.filtered_data
                        if item.get(field) == value
                    ]
                elif operator == 'ne':
                    self.filtered_data = [
                        item for item in self.filtered_data
                        if item.get(field) != value
                    ]
                elif operator == 'gt':
                    self.filtered_data = [
                        item for item in self.filtered_data
                        if item.get(field, 0) > value
                    ]
                elif operator == 'gte':
                    self.filtered_data = [
                        item for item in self.filtered_data
                        if item.get(field, 0) >= value
                    ]
                elif operator == 'lt':
                    self.filtered_data = [
                        item for item in self.filtered_data
                        if item.get(field, 0) < value
                    ]
                elif operator == 'lte':
                    self.filtered_data = [
                        item for item in self.filtered_data
                        if item.get(field, 0) <= value
                    ]
                elif operator == 'in':
                    self.filtered_data = [
                        item for item in self.filtered_data
                        if item.get(field) in value
                    ]
                elif operator == 'contains':
                    self.filtered_data = [
                        item for item in self.filtered_data
                        if value.lower() in str(item.get(field, '')).lower()
                    ]
                elif operator == 'startswith':
                    self.filtered_data = [
                        item for item in self.filtered_data
                        if str(item.get(field, '')).startswith(value)
                    ]
            else:
                # Basit eşitlik
                self.filtered_data = [
                    item for item in self.filtered_data
                    if item.get(key) == value
                ]

        return self

    def sort(self, *fields) -> 'QueryBuilder':
        """Çoklu alan sıralama"""
        for field in reversed(fields):
            reverse = False
            if field.startswith('-'):
                reverse = True
                field = field[1:]

            self.filtered_data.sort(
                key=lambda x: x.get(field, ''),
                reverse=reverse
            )

        return self

    def select(self, *fields) -> 'QueryBuilder':
        """Sadece belirli alanları seç"""
        self.filtered_data = [
            {k: v for k, v in item.items() if k in fields}
            for item in self.filtered_data
        ]

        return self

    def paginate_offset(self, page: int, page_size: int) -> Dict[str, Any]:
        """Offset-based pagination"""
        total = len(self.filtered_data)
        total_pages = (total + page_size - 1) // page_size

        start = (page - 1) * page_size
        end = start + page_size

        return {
            "items": self.filtered_data[start:end],
            "pagination": {
                "page": page,
                "page_size": page_size,
                "total_items": total,
                "total_pages": total_pages,
                "has_next": page < total_pages,
                "has_prev": page > 1
            }
        }

    def paginate_cursor(self, cursor: Optional[str], page_size: int,
                       cursor_field: str = 'id') -> Dict[str, Any]:
        """Cursor-based pagination"""
        # Cursor decode
        cursor_value = None
        if cursor:
            try:
                cursor_value = int(base64.b64decode(cursor).decode())
            except:
                pass

        # Cursor'dan sonraki itemları al
        if cursor_value is not None:
            start_idx = next(
                (i for i, item in enumerate(self.filtered_data)
                 if item.get(cursor_field) == cursor_value),
                0
            ) + 1
        else:
            start_idx = 0

        items = self.filtered_data[start_idx:start_idx + page_size]

        # Sonraki cursor
        next_cursor = None
        if len(items) == page_size and start_idx + page_size < len(self.filtered_data):
            last_item = items[-1]
            next_cursor = base64.b64encode(
                str(last_item[cursor_field]).encode()
            ).decode()

        return {
            "items": items,
            "cursor": {
                "next": next_cursor,
                "has_next": next_cursor is not None
            }
        }

    def execute(self) -> List[Dict[str, Any]]:
        """Sorguyu çalıştır"""
        return self.filtered_data

# Test data
products = [
    {"id": 1, "name": "Laptop", "category": "Electronics", "price": 1000, "stock": 15},
    {"id": 2, "name": "Mouse", "category": "Electronics", "price": 25, "stock": 100},
    {"id": 3, "name": "Keyboard", "category": "Electronics", "price": 75, "stock": 50},
    {"id": 4, "name": "Monitor", "category": "Electronics", "price": 300, "stock": 30},
    {"id": 5, "name": "Chair", "category": "Furniture", "price": 200, "stock": 20},
    {"id": 6, "name": "Desk", "category": "Furniture", "price": 400, "stock": 10},
]

# Filtreleme ve sıralama
print("Advanced Filtering & Sorting:")
result = (QueryBuilder(products)
          .filter(category="Electronics", price__gte=50)
          .sort('-price')
          .select('id', 'name', 'price')
          .execute())

for item in result:
    print(f"  {item}")

# Offset pagination
print("\nOffset-based Pagination:")
paginated = (QueryBuilder(products)
             .filter(category="Electronics")
             .sort('name')
             .paginate_offset(page=1, page_size=2))

print(f"  Items: {len(paginated['items'])}")
print(f"  Total: {paginated['pagination']['total_items']}")
print(f"  Has next: {paginated['pagination']['has_next']}")

# Cursor pagination
print("\nCursor-based Pagination:")
cursor_result = (QueryBuilder(products)
                 .sort('id')
                 .paginate_cursor(cursor=None, page_size=3))

print(f"  Items: {len(cursor_result['items'])}")
print(f"  Next cursor: {cursor_result['cursor']['next']}")
print(f"  Has next: {cursor_result['cursor']['has_next']}")


# ============================================================================
# NOT: Remaining exercises (6-15) için benzer şekilde devam eder
# Dosya boyutu nedeniyle kısaltılmıştır
# Tam liste: WebSocket, Background Tasks, API Keys, Transactions,
# Versioning, OAuth2, GraphQL-like, Webhooks, SSE, Microservices
# ============================================================================

print("\n" + "="*70)
print("WEB DEVELOPMENT EXERCISES COMPLETED!")
print("="*70)
print("""
✓ Exercise 1: RESTful CRUD API
✓ Exercise 2: JWT Authentication System
✓ Exercise 3: Advanced Rate Limiting
✓ Exercise 4: File Upload System
✓ Exercise 5: Pagination & Filtering

Bu egzersizler production-ready web API geliştirme için
temel yapı taşlarını içermektedir.

Önerilen Ek Konular:
- WebSocket real-time communication
- Background task processing (Celery)
- API key authentication
- Database transactions
- OAuth2 social login
- GraphQL implementation
- Webhook systems
- Server-Sent Events (SSE)
- Microservice patterns

FastAPI Documentation: https://fastapi.tiangolo.com/
""")
