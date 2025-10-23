# Web Development - İleri Düzey Python Web Geliştirme

## İçindekiler
- [HTTP Protokolü ve Temel Kavramlar](#http-protokolü)
- [REST API Prensipleri](#rest-api-prensipleri)
- [Flask Framework](#flask-framework)
- [FastAPI Framework](#fastapi-framework)
- [Request/Response Yönetimi](#request-response-yönetimi)
- [Authentication & Authorization](#authentication-authorization)
- [Middleware ve CORS](#middleware-cors)
- [API Versiyonlama](#api-versiyonlama)
- [Production Best Practices](#production-best-practices)

---

## HTTP Protokolü

HTTP (HyperText Transfer Protocol), web üzerinde veri iletişiminin temelini oluşturur.

### HTTP Metodları ve Kullanım Alanları

```python
"""
HTTP Metodları ve RESTful Kullanımı

GET     - Kaynak okuma (idempotent, safe)
POST    - Yeni kaynak oluşturma
PUT     - Kaynağı tamamen güncelleme (idempotent)
PATCH   - Kaynağı kısmi güncelleme
DELETE  - Kaynak silme (idempotent)
HEAD    - Sadece header bilgisi alma
OPTIONS - Desteklenen metodları öğrenme
"""

from typing import Dict, Any
import http

# HTTP Status Kodları - Doğru Kullanım
class HTTPStatusCodes:
    """HTTP durum kodlarının anlamları ve kullanım senaryoları"""

    # 2xx - Başarılı
    OK = 200                    # GET, PUT, PATCH başarılı
    CREATED = 201               # POST başarılı, yeni kaynak oluşturuldu
    ACCEPTED = 202              # İstek kabul edildi, işleniyor
    NO_CONTENT = 204            # DELETE başarılı, dönen veri yok

    # 3xx - Yönlendirme
    MOVED_PERMANENTLY = 301     # Kaynak kalıcı olarak taşındı
    FOUND = 302                 # Geçici yönlendirme
    NOT_MODIFIED = 304          # Önbellek geçerli

    # 4xx - İstemci Hataları
    BAD_REQUEST = 400           # Geçersiz istek
    UNAUTHORIZED = 401          # Kimlik doğrulama gerekli
    FORBIDDEN = 403             # Yetkisiz erişim
    NOT_FOUND = 404             # Kaynak bulunamadı
    METHOD_NOT_ALLOWED = 405    # HTTP metodu desteklenmiyor
    CONFLICT = 409              # Çakışma (duplicate kayıt vb.)
    UNPROCESSABLE_ENTITY = 422  # Validasyon hatası
    TOO_MANY_REQUESTS = 429     # Rate limit aşıldı

    # 5xx - Sunucu Hataları
    INTERNAL_SERVER_ERROR = 500 # Genel sunucu hatası
    NOT_IMPLEMENTED = 501       # Özellik desteklenmiyor
    BAD_GATEWAY = 502           # Gateway hatası
    SERVICE_UNAVAILABLE = 503   # Servis geçici olarak kullanılamıyor

# HTTP Headers - Önemli Başlıklar
class HTTPHeaders:
    """Sık kullanılan HTTP başlıkları"""

    @staticmethod
    def common_headers() -> Dict[str, str]:
        return {
            # Content Negotiation
            "Content-Type": "application/json",
            "Accept": "application/json",

            # Security
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains",

            # CORS
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type, Authorization",

            # Caching
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",

            # Rate Limiting
            "X-RateLimit-Limit": "100",
            "X-RateLimit-Remaining": "99",
            "X-RateLimit-Reset": "1640000000",
        }
```

---

## REST API Prensipleri

REST (Representational State Transfer), web servisleri için mimari bir stildir.

### RESTful API Tasarımı

```python
"""
RESTful API Tasarım Prensipleri

1. Resource-Based URLs
2. HTTP Metodlarını Doğru Kullanma
3. Stateless İletişim
4. HATEOAS (Hypermedia as the Engine of Application State)
5. Versiyonlama
6. Tutarlı Response Format
"""

from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum

# RESTful Resource Modelleme
class ResourceModel:
    """REST kaynağı için standart model"""

    def __init__(self, resource_id: str, resource_type: str):
        self.id = resource_id
        self.type = resource_type
        self.created_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()
        self.links = {}

    def add_link(self, rel: str, href: str, method: str = "GET"):
        """HATEOAS - Hypermedia linkler ekleme"""
        self.links[rel] = {
            "href": href,
            "method": method
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "_links": self.links
        }

# RESTful URL Yapısı
class RESTfulURLDesign:
    """
    İyi URL Tasarımı Örnekleri:

    ✅ Doğru:
    GET    /api/v1/users                 - Tüm kullanıcıları listele
    GET    /api/v1/users/{id}            - Belirli kullanıcıyı getir
    POST   /api/v1/users                 - Yeni kullanıcı oluştur
    PUT    /api/v1/users/{id}            - Kullanıcıyı güncelle
    PATCH  /api/v1/users/{id}            - Kullanıcıyı kısmi güncelle
    DELETE /api/v1/users/{id}            - Kullanıcıyı sil
    GET    /api/v1/users/{id}/posts      - Kullanıcının postları
    GET    /api/v1/users/{id}/posts/{post_id} - Kullanıcının belirli postu

    ❌ Yanlış:
    GET    /api/v1/getAllUsers           - Verb kullanmayın
    POST   /api/v1/user/create           - Gereksiz action
    GET    /api/v1/user/123/delete       - GET ile silme yapılmaz
    GET    /api/v1/users/list            - Gereksiz /list
    """

    BASE_URL = "/api/v1"

    @classmethod
    def build_collection_url(cls, resource: str) -> str:
        """Koleksiyon URL'i oluştur"""
        return f"{cls.BASE_URL}/{resource}"

    @classmethod
    def build_resource_url(cls, resource: str, resource_id: str) -> str:
        """Tekil kaynak URL'i oluştur"""
        return f"{cls.BASE_URL}/{resource}/{resource_id}"

    @classmethod
    def build_nested_url(cls, parent: str, parent_id: str,
                         child: str, child_id: Optional[str] = None) -> str:
        """İç içe kaynak URL'i oluştur"""
        base = f"{cls.BASE_URL}/{parent}/{parent_id}/{child}"
        return f"{base}/{child_id}" if child_id else base

# Standart API Response Format
class APIResponse:
    """Tutarlı API response formatı"""

    @staticmethod
    def success(data: Any, message: str = "Success",
                meta: Optional[Dict] = None) -> Dict[str, Any]:
        """Başarılı response"""
        response = {
            "success": True,
            "message": message,
            "data": data,
            "timestamp": datetime.utcnow().isoformat()
        }
        if meta:
            response["meta"] = meta
        return response

    @staticmethod
    def error(message: str, errors: Optional[List[str]] = None,
              code: Optional[str] = None) -> Dict[str, Any]:
        """Hata response"""
        response = {
            "success": False,
            "message": message,
            "timestamp": datetime.utcnow().isoformat()
        }
        if errors:
            response["errors"] = errors
        if code:
            response["code"] = code
        return response

    @staticmethod
    def paginated(data: List[Any], page: int, page_size: int,
                  total: int) -> Dict[str, Any]:
        """Sayfalanmış response"""
        total_pages = (total + page_size - 1) // page_size
        return {
            "success": True,
            "data": data,
            "pagination": {
                "page": page,
                "page_size": page_size,
                "total_items": total,
                "total_pages": total_pages,
                "has_next": page < total_pages,
                "has_prev": page > 1
            },
            "timestamp": datetime.utcnow().isoformat()
        }
```

---

## Flask Framework

Flask, Python için hafif ve esnek bir web framework'üdür.

### Flask Basics - Routes ve Blueprints

```python
"""
Flask ile Web API Geliştirme
"""

from flask import Flask, Blueprint, request, jsonify, abort
from werkzeug.exceptions import HTTPException
from functools import wraps
from typing import Callable, Dict, Any
import logging

# Flask Application Factory Pattern
def create_app(config_name: str = 'development') -> Flask:
    """Application Factory Pattern - Test ve production için farklı yapılandırma"""
    app = Flask(__name__)

    # Yapılandırma
    app.config.update(
        SECRET_KEY='your-secret-key-change-in-production',
        JSON_SORT_KEYS=False,
        JSONIFY_PRETTYPRINT_REGULAR=True,
        MAX_CONTENT_LENGTH=16 * 1024 * 1024,  # 16MB max request
    )

    # Logging
    logging.basicConfig(level=logging.INFO)

    # Blueprints'leri kaydet
    from .blueprints import user_bp, product_bp, auth_bp
    app.register_blueprint(auth_bp, url_prefix='/api/v1/auth')
    app.register_blueprint(user_bp, url_prefix='/api/v1/users')
    app.register_blueprint(product_bp, url_prefix='/api/v1/products')

    # Error handlers
    register_error_handlers(app)

    # Request/Response handlers
    register_request_handlers(app)

    return app

# Blueprint Örneği - Users
user_bp = Blueprint('users', __name__)

@user_bp.route('/', methods=['GET'])
def get_users():
    """Tüm kullanıcıları listele"""
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 10, type=int)

    # Veritabanı sorgusu (örnek)
    users = [
        {"id": 1, "name": "Ali", "email": "ali@example.com"},
        {"id": 2, "name": "Ayşe", "email": "ayse@example.com"},
    ]

    return jsonify(APIResponse.paginated(
        data=users,
        page=page,
        page_size=per_page,
        total=len(users)
    )), 200

@user_bp.route('/<int:user_id>', methods=['GET'])
def get_user(user_id: int):
    """Belirli bir kullanıcıyı getir"""
    user = {"id": user_id, "name": "Ali", "email": "ali@example.com"}

    if not user:
        abort(404, description="Kullanıcı bulunamadı")

    return jsonify(APIResponse.success(data=user)), 200

@user_bp.route('/', methods=['POST'])
def create_user():
    """Yeni kullanıcı oluştur"""
    if not request.is_json:
        abort(400, description="Content-Type application/json olmalı")

    data = request.get_json()

    # Validasyon
    required_fields = ['name', 'email', 'password']
    if not all(field in data for field in required_fields):
        abort(400, description="Gerekli alanlar eksik")

    # Kullanıcı oluştur (örnek)
    new_user = {
        "id": 3,
        "name": data['name'],
        "email": data['email']
    }

    return jsonify(APIResponse.success(
        data=new_user,
        message="Kullanıcı başarıyla oluşturuldu"
    )), 201

@user_bp.route('/<int:user_id>', methods=['PUT'])
def update_user(user_id: int):
    """Kullanıcıyı güncelle"""
    if not request.is_json:
        abort(400, description="Content-Type application/json olmalı")

    data = request.get_json()

    # Güncelleme işlemi (örnek)
    updated_user = {
        "id": user_id,
        "name": data.get('name', 'Ali'),
        "email": data.get('email', 'ali@example.com')
    }

    return jsonify(APIResponse.success(
        data=updated_user,
        message="Kullanıcı başarıyla güncellendi"
    )), 200

@user_bp.route('/<int:user_id>', methods=['DELETE'])
def delete_user(user_id: int):
    """Kullanıcıyı sil"""
    # Silme işlemi (örnek)
    return '', 204

# Error Handler
def register_error_handlers(app: Flask):
    """Global error handler'ları kaydet"""

    @app.errorhandler(400)
    def bad_request(error):
        return jsonify(APIResponse.error(
            message=str(error.description),
            code="BAD_REQUEST"
        )), 400

    @app.errorhandler(404)
    def not_found(error):
        return jsonify(APIResponse.error(
            message=str(error.description),
            code="NOT_FOUND"
        )), 404

    @app.errorhandler(500)
    def internal_error(error):
        app.logger.error(f"Server Error: {error}")
        return jsonify(APIResponse.error(
            message="Sunucu hatası oluştu",
            code="INTERNAL_SERVER_ERROR"
        )), 500

    @app.errorhandler(Exception)
    def handle_exception(error):
        """Tüm beklenmeyen hataları yakala"""
        if isinstance(error, HTTPException):
            return error

        app.logger.error(f"Unhandled Exception: {error}")
        return jsonify(APIResponse.error(
            message="Beklenmeyen bir hata oluştu",
            code="UNKNOWN_ERROR"
        )), 500

# Request/Response Handlers
def register_request_handlers(app: Flask):
    """Request ve response handler'ları kaydet"""

    @app.before_request
    def before_request():
        """Her request öncesi çalışır"""
        app.logger.info(f"{request.method} {request.path}")

    @app.after_request
    def after_request(response):
        """Her request sonrası çalışır"""
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['X-Frame-Options'] = 'DENY'
        response.headers['X-XSS-Protection'] = '1; mode=block'
        return response
```

---

## FastAPI Framework

FastAPI, modern, hızlı ve async destekli Python web framework'üdür.

### FastAPI Basics - Async Routes ve Pydantic

```python
"""
FastAPI ile Modern Web API Geliştirme

Özellikler:
- Otomatik API dokümantasyonu (Swagger UI, ReDoc)
- Pydantic ile veri validasyonu
- Type hints ile otomatik serialization
- Async/await desteği
- Yüksek performans (Starlette ve Pydantic sayesinde)
"""

from fastapi import FastAPI, HTTPException, Depends, status, Query, Path, Body
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, EmailStr, validator
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum
import asyncio

# Pydantic Models - Request/Response Şemaları
class UserRole(str, Enum):
    """Kullanıcı rolleri"""
    ADMIN = "admin"
    USER = "user"
    MODERATOR = "moderator"

class UserBase(BaseModel):
    """Temel kullanıcı şeması"""
    email: EmailStr = Field(..., description="Kullanıcı email adresi")
    username: str = Field(..., min_length=3, max_length=50,
                          description="Kullanıcı adı")
    full_name: Optional[str] = Field(None, max_length=100)
    role: UserRole = Field(default=UserRole.USER)

    @validator('username')
    def username_alphanumeric(cls, v):
        """Username sadece alfanumerik karakterler içermeli"""
        if not v.isalnum():
            raise ValueError('Username sadece harf ve rakam içermelidir')
        return v

class UserCreate(UserBase):
    """Kullanıcı oluşturma şeması"""
    password: str = Field(..., min_length=8, max_length=100)

    @validator('password')
    def password_strength(cls, v):
        """Şifre gücü kontrolü"""
        if not any(char.isdigit() for char in v):
            raise ValueError('Şifre en az bir rakam içermelidir')
        if not any(char.isupper() for char in v):
            raise ValueError('Şifre en az bir büyük harf içermelidir')
        return v

class UserUpdate(BaseModel):
    """Kullanıcı güncelleme şeması"""
    email: Optional[EmailStr] = None
    full_name: Optional[str] = Field(None, max_length=100)
    role: Optional[UserRole] = None

class UserResponse(UserBase):
    """Kullanıcı response şeması"""
    id: int
    is_active: bool = True
    created_at: datetime
    updated_at: datetime

    class Config:
        orm_mode = True  # ORM modellerini direkt kullanabilme

class PaginatedResponse(BaseModel):
    """Sayfalanmış response şeması"""
    items: List[UserResponse]
    total: int
    page: int
    page_size: int
    total_pages: int

# FastAPI Application
app = FastAPI(
    title="Advanced API",
    description="Production-ready FastAPI örneği",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)

# In-memory database (örnek için)
fake_users_db: Dict[int, Dict[str, Any]] = {}
user_id_counter = 1

# Dependency: Database Session (gerçek uygulamada)
async def get_db():
    """Database session dependency"""
    # Gerçek uygulamada SQLAlchemy session
    db = {}
    try:
        yield db
    finally:
        pass

# Dependency: Current User (authentication için)
async def get_current_user(user_id: int = Query(...)) -> Dict[str, Any]:
    """Mevcut kullanıcıyı getir"""
    if user_id not in fake_users_db:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Kullanıcı bulunamadı"
        )
    return fake_users_db[user_id]

# Routes - Async Endpoints
@app.get("/")
async def root():
    """Ana endpoint"""
    return {"message": "Welcome to Advanced API"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat()
    }

@app.post("/api/v1/users",
          response_model=UserResponse,
          status_code=status.HTTP_201_CREATED,
          tags=["users"])
async def create_user(user: UserCreate):
    """
    Yeni kullanıcı oluştur

    - **email**: Geçerli email adresi
    - **username**: 3-50 karakter arası, alfanumerik
    - **password**: En az 8 karakter, rakam ve büyük harf içermeli
    """
    global user_id_counter

    # Email kontrolü
    for existing_user in fake_users_db.values():
        if existing_user["email"] == user.email:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Bu email adresi zaten kullanılıyor"
            )

    # Async operation simülasyonu (veritabanı yazma)
    await asyncio.sleep(0.1)

    new_user = {
        "id": user_id_counter,
        "email": user.email,
        "username": user.username,
        "full_name": user.full_name,
        "role": user.role,
        "is_active": True,
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow(),
    }

    fake_users_db[user_id_counter] = new_user
    user_id_counter += 1

    return new_user

@app.get("/api/v1/users",
         response_model=PaginatedResponse,
         tags=["users"])
async def get_users(
    page: int = Query(1, ge=1, description="Sayfa numarası"),
    page_size: int = Query(10, ge=1, le=100, description="Sayfa başına öğe sayısı"),
    role: Optional[UserRole] = Query(None, description="Role'e göre filtrele"),
    search: Optional[str] = Query(None, min_length=3, description="İsim veya email'de ara")
):
    """
    Kullanıcıları listele (sayfalama ve filtreleme ile)
    """
    # Filtreleme
    filtered_users = list(fake_users_db.values())

    if role:
        filtered_users = [u for u in filtered_users if u["role"] == role]

    if search:
        search_lower = search.lower()
        filtered_users = [
            u for u in filtered_users
            if search_lower in u["username"].lower()
            or search_lower in u["email"].lower()
        ]

    # Sayfalama
    total = len(filtered_users)
    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size
    paginated_users = filtered_users[start_idx:end_idx]

    # Async operation simülasyonu
    await asyncio.sleep(0.1)

    return {
        "items": paginated_users,
        "total": total,
        "page": page,
        "page_size": page_size,
        "total_pages": (total + page_size - 1) // page_size
    }

@app.get("/api/v1/users/{user_id}",
         response_model=UserResponse,
         tags=["users"])
async def get_user(
    user_id: int = Path(..., ge=1, description="Kullanıcı ID'si")
):
    """Belirli bir kullanıcıyı getir"""
    if user_id not in fake_users_db:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"ID {user_id} olan kullanıcı bulunamadı"
        )

    await asyncio.sleep(0.1)
    return fake_users_db[user_id]

@app.patch("/api/v1/users/{user_id}",
           response_model=UserResponse,
           tags=["users"])
async def update_user(
    user_id: int = Path(..., ge=1),
    user_update: UserUpdate = Body(...)
):
    """Kullanıcıyı kısmi güncelle"""
    if user_id not in fake_users_db:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Kullanıcı bulunamadı"
        )

    user = fake_users_db[user_id]

    # Sadece gönderilen alanları güncelle
    update_data = user_update.dict(exclude_unset=True)
    for field, value in update_data.items():
        user[field] = value

    user["updated_at"] = datetime.utcnow()

    await asyncio.sleep(0.1)
    return user

@app.delete("/api/v1/users/{user_id}",
            status_code=status.HTTP_204_NO_CONTENT,
            tags=["users"])
async def delete_user(user_id: int = Path(..., ge=1)):
    """Kullanıcıyı sil"""
    if user_id not in fake_users_db:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Kullanıcı bulunamadı"
        )

    del fake_users_db[user_id]
    await asyncio.sleep(0.1)
    return None

# Async ile Paralel İşlemler
@app.get("/api/v1/users/{user_id}/stats", tags=["users"])
async def get_user_stats(user_id: int = Path(..., ge=1)):
    """Kullanıcı istatistiklerini paralel olarak getir"""

    async def get_post_count(uid: int) -> int:
        await asyncio.sleep(0.5)  # Simüle edilmiş veritabanı sorgusu
        return 42

    async def get_follower_count(uid: int) -> int:
        await asyncio.sleep(0.5)
        return 128

    async def get_like_count(uid: int) -> int:
        await asyncio.sleep(0.5)
        return 356

    # Tüm işlemleri paralel çalıştır
    post_count, follower_count, like_count = await asyncio.gather(
        get_post_count(user_id),
        get_follower_count(user_id),
        get_like_count(user_id)
    )

    return {
        "user_id": user_id,
        "stats": {
            "posts": post_count,
            "followers": follower_count,
            "likes": like_count
        }
    }
```

---

## Authentication & Authorization

JWT (JSON Web Tokens) ve OAuth ile kimlik doğrulama.

### JWT Authentication

```python
"""
JWT ile Authentication ve Authorization
"""

from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from datetime import datetime, timedelta
from typing import Optional, Dict
import jwt
from passlib.context import CryptContext

# Konfigürasyon
SECRET_KEY = "your-secret-key-keep-it-secret"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
REFRESH_TOKEN_EXPIRE_DAYS = 7

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Security
security = HTTPBearer()

# Models
class Token(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"

class TokenData(BaseModel):
    user_id: Optional[int] = None
    username: Optional[str] = None
    role: Optional[str] = None

class LoginRequest(BaseModel):
    username: str
    password: str

# Helper Functions
def hash_password(password: str) -> str:
    """Şifreyi hash'le"""
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Şifreyi doğrula"""
    return pwd_context.verify(plain_password, hashed_password)

def create_access_token(data: Dict, expires_delta: Optional[timedelta] = None) -> str:
    """Access token oluştur"""
    to_encode = data.copy()

    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)

    to_encode.update({
        "exp": expire,
        "iat": datetime.utcnow(),
        "type": "access"
    })

    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def create_refresh_token(data: Dict) -> str:
    """Refresh token oluştur"""
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)

    to_encode.update({
        "exp": expire,
        "iat": datetime.utcnow(),
        "type": "refresh"
    })

    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def decode_token(token: str) -> Dict:
    """Token'ı decode et"""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token süresi dolmuş",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except jwt.JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token doğrulanamadı",
            headers={"WWW-Authenticate": "Bearer"},
        )

# Dependencies
async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> TokenData:
    """Token'dan mevcut kullanıcıyı çıkar"""
    token = credentials.credentials
    payload = decode_token(token)

    if payload.get("type") != "access":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Geçersiz token tipi"
        )

    user_id = payload.get("sub")
    username = payload.get("username")
    role = payload.get("role")

    if user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token geçersiz"
        )

    return TokenData(user_id=int(user_id), username=username, role=role)

async def get_current_active_user(
    current_user: TokenData = Depends(get_current_user)
) -> TokenData:
    """Aktif kullanıcıyı getir"""
    # Burada kullanıcının aktif olup olmadığını kontrol edebilirsiniz
    return current_user

def require_role(required_role: str):
    """Belirli bir rol gerektiren decorator"""
    async def role_checker(
        current_user: TokenData = Depends(get_current_active_user)
    ) -> TokenData:
        if current_user.role != required_role:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Bu işlem için yetkiniz yok"
            )
        return current_user

    return role_checker

# Auth Routes
auth_app = FastAPI()

@auth_app.post("/login", response_model=Token)
async def login(credentials: LoginRequest):
    """Kullanıcı girişi"""
    # Gerçek uygulamada veritabanından kullanıcı kontrolü
    fake_user = {
        "id": 1,
        "username": "admin",
        "hashed_password": hash_password("admin123"),
        "role": "admin"
    }

    if credentials.username != fake_user["username"]:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Kullanıcı adı veya şifre hatalı"
        )

    if not verify_password(credentials.password, fake_user["hashed_password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Kullanıcı adı veya şifre hatalı"
        )

    # Token oluştur
    access_token = create_access_token(
        data={
            "sub": str(fake_user["id"]),
            "username": fake_user["username"],
            "role": fake_user["role"]
        }
    )

    refresh_token = create_refresh_token(
        data={"sub": str(fake_user["id"])}
    )

    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer"
    }

@auth_app.post("/refresh", response_model=Token)
async def refresh_token(
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """Token yenileme"""
    token = credentials.credentials
    payload = decode_token(token)

    if payload.get("type") != "refresh":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Geçersiz token tipi"
        )

    user_id = payload.get("sub")

    # Yeni tokenlar oluştur
    access_token = create_access_token(data={"sub": user_id})
    refresh_token = create_refresh_token(data={"sub": user_id})

    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer"
    }

@auth_app.get("/me")
async def get_me(current_user: TokenData = Depends(get_current_active_user)):
    """Mevcut kullanıcı bilgilerini getir"""
    return {
        "user_id": current_user.user_id,
        "username": current_user.username,
        "role": current_user.role
    }

@auth_app.get("/admin-only")
async def admin_only(
    current_user: TokenData = Depends(require_role("admin"))
):
    """Sadece admin'ler erişebilir"""
    return {"message": "Admin sayfasına hoş geldiniz"}
```

---

## Middleware ve CORS

Middleware, request/response akışına müdahale eden katmandır.

```python
"""
Middleware ve CORS Yapılandırması
"""

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from typing import Callable
import time
import logging

# Logging
logger = logging.getLogger(__name__)

# Custom Middleware: Request Timing
class TimingMiddleware(BaseHTTPMiddleware):
    """Request süresini ölç"""

    async def dispatch(self, request: Request, call_next: Callable):
        start_time = time.time()

        # Request'i işle
        response = await call_next(request)

        # Süreyi hesapla
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = str(process_time)

        logger.info(
            f"{request.method} {request.url.path} "
            f"completed in {process_time:.4f}s"
        )

        return response

# Custom Middleware: Request ID
class RequestIDMiddleware(BaseHTTPMiddleware):
    """Her request'e unique ID ekle"""

    async def dispatch(self, request: Request, call_next: Callable):
        import uuid
        request_id = str(uuid.uuid4())

        # Request state'e ekle
        request.state.request_id = request_id

        # Response'a header olarak ekle
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id

        return response

# Custom Middleware: Rate Limiting (basit versiyon)
class RateLimitMiddleware(BaseHTTPMiddleware):
    """Basit rate limiting"""

    def __init__(self, app, calls: int = 100, period: int = 60):
        super().__init__(app)
        self.calls = calls
        self.period = period
        self.clients: Dict[str, List[float]] = {}

    async def dispatch(self, request: Request, call_next: Callable):
        client_ip = request.client.host
        current_time = time.time()

        # Client'ın request geçmişini al
        if client_ip not in self.clients:
            self.clients[client_ip] = []

        # Eski requestleri temizle
        self.clients[client_ip] = [
            t for t in self.clients[client_ip]
            if current_time - t < self.period
        ]

        # Rate limit kontrolü
        if len(self.clients[client_ip]) >= self.calls:
            return Response(
                content='{"detail": "Rate limit exceeded"}',
                status_code=429,
                media_type="application/json",
                headers={
                    "X-RateLimit-Limit": str(self.calls),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(int(current_time + self.period))
                }
            )

        # Request'i kaydet
        self.clients[client_ip].append(current_time)

        # Response'a rate limit bilgilerini ekle
        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(self.calls)
        response.headers["X-RateLimit-Remaining"] = str(
            self.calls - len(self.clients[client_ip])
        )

        return response

# Application with Middleware
def create_app_with_middleware() -> FastAPI:
    app = FastAPI(title="API with Middleware")

    # CORS Middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000", "https://example.com"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID", "X-Process-Time"],
        max_age=3600,
    )

    # GZip Compression
    app.add_middleware(GZipMiddleware, minimum_size=1000)

    # Trusted Host
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["localhost", "*.example.com"]
    )

    # Custom Middlewares
    app.add_middleware(TimingMiddleware)
    app.add_middleware(RequestIDMiddleware)
    app.add_middleware(RateLimitMiddleware, calls=100, period=60)

    return app

# Middleware ile Request Logging
class DetailedLoggingMiddleware(BaseHTTPMiddleware):
    """Detaylı request/response logging"""

    async def dispatch(self, request: Request, call_next: Callable):
        # Request logging
        logger.info(
            f"Request: {request.method} {request.url.path}\n"
            f"Headers: {dict(request.headers)}\n"
            f"Client: {request.client.host}"
        )

        try:
            response = await call_next(request)

            # Response logging
            logger.info(
                f"Response: {response.status_code}\n"
                f"Headers: {dict(response.headers)}"
            )

            return response

        except Exception as e:
            logger.error(f"Error processing request: {str(e)}")
            raise
```

---

## API Versiyonlama

API versiyonlama stratejileri.

```python
"""
API Versiyonlama Stratejileri

1. URL Path Versioning: /api/v1/users
2. Header Versioning: Accept: application/vnd.api.v1+json
3. Query Parameter: /api/users?version=1
"""

from fastapi import FastAPI, Request, Header, APIRouter
from typing import Optional
from enum import Enum

# Strategy 1: URL Path Versioning (En yaygın)
app = FastAPI()

# V1 Router
v1_router = APIRouter(prefix="/api/v1", tags=["v1"])

@v1_router.get("/users")
async def get_users_v1():
    """API V1 - Kullanıcıları getir"""
    return {
        "version": "1.0",
        "users": [
            {"id": 1, "name": "Ali"}
        ]
    }

# V2 Router - Yeni özellikler ve değişiklikler
v2_router = APIRouter(prefix="/api/v2", tags=["v2"])

@v2_router.get("/users")
async def get_users_v2():
    """API V2 - Geliştirilmiş kullanıcı response'u"""
    return {
        "version": "2.0",
        "users": [
            {
                "id": 1,
                "name": "Ali",
                "email": "ali@example.com",  # Yeni alan
                "created_at": "2024-01-01T00:00:00Z"  # Yeni alan
            }
        ],
        "total": 1,  # Yeni alan
        "page": 1    # Yeni alan
    }

# Routerları ekle
app.include_router(v1_router)
app.include_router(v2_router)

# Strategy 2: Header Versioning
class APIVersion(str, Enum):
    V1 = "application/vnd.api.v1+json"
    V2 = "application/vnd.api.v2+json"

@app.get("/api/users")
async def get_users_header_version(
    accept: Optional[str] = Header(default="application/vnd.api.v1+json")
):
    """Header-based versioning"""
    if accept == APIVersion.V2:
        return {"version": "2.0", "data": "V2 response"}
    else:
        return {"version": "1.0", "data": "V1 response"}

# Deprecation Warning
@v1_router.get("/products")
async def get_products_v1():
    """Deprecated endpoint"""
    return {
        "warning": "Bu endpoint deprecated. Lütfen /api/v2/products kullanın",
        "deprecation_date": "2024-12-31",
        "sunset_date": "2025-06-30",
        "data": []
    }
```

---

## Production Best Practices

Production ortamı için best practice'ler.

```python
"""
Production-Ready API Best Practices

1. Configuration Management
2. Logging ve Monitoring
3. Error Handling
4. Security Headers
5. Health Checks
6. Graceful Shutdown
7. Database Connection Pooling
8. Caching
9. Background Tasks
10. Testing
"""

from fastapi import FastAPI, Request, BackgroundTasks
from pydantic import BaseSettings
import logging
from logging.handlers import RotatingFileHandler
import sys
from typing import Dict, Any
import asyncio

# 1. Configuration Management
class Settings(BaseSettings):
    """Çevre değişkenlerinden yapılandırma"""
    APP_NAME: str = "Production API"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False

    # Database
    DATABASE_URL: str = "postgresql://user:pass@localhost/dbname"
    DATABASE_POOL_SIZE: int = 20
    DATABASE_MAX_OVERFLOW: int = 0

    # Redis
    REDIS_URL: str = "redis://localhost:6379"

    # Security
    SECRET_KEY: str
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    # CORS
    CORS_ORIGINS: list = ["http://localhost:3000"]

    # Rate Limiting
    RATE_LIMIT_ENABLED: bool = True
    RATE_LIMIT_CALLS: int = 100
    RATE_LIMIT_PERIOD: int = 60

    class Config:
        env_file = ".env"

settings = Settings()

# 2. Logging Setup
def setup_logging():
    """Production logging yapılandırması"""

    # Root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO if not settings.DEBUG else logging.DEBUG)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(console_formatter)

    # File handler (rotating)
    file_handler = RotatingFileHandler(
        'app.log',
        maxBytes=10485760,  # 10MB
        backupCount=5
    )
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - '
        '%(pathname)s:%(lineno)d - %(message)s'
    )
    file_handler.setFormatter(file_formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger

logger = setup_logging()

# 3. Production Application
def create_production_app() -> FastAPI:
    """Production-ready application"""

    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        docs_url="/api/docs" if settings.DEBUG else None,  # Production'da kapalı
        redoc_url="/api/redoc" if settings.DEBUG else None,
        debug=settings.DEBUG,
    )

    # Startup event
    @app.on_event("startup")
    async def startup_event():
        logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
        # Database connection pool oluştur
        # Redis connection oluştur
        # Cache'i ısıt
        logger.info("Application started successfully")

    # Shutdown event
    @app.on_event("shutdown")
    async def shutdown_event():
        logger.info("Shutting down application")
        # Database connections'ları kapat
        # Redis connection'ı kapat
        # Pending tasks'leri tamamla
        logger.info("Application shutdown complete")

    # Health check
    @app.get("/health")
    async def health_check():
        """Detaylı health check"""
        return {
            "status": "healthy",
            "version": settings.APP_VERSION,
            "checks": {
                "database": "ok",  # Gerçek kontrol yap
                "redis": "ok",     # Gerçek kontrol yap
            }
        }

    # Readiness check (Kubernetes için)
    @app.get("/ready")
    async def readiness_check():
        """Servis hazır mı?"""
        # Tüm bağımlılıkları kontrol et
        return {"status": "ready"}

    # Liveness check (Kubernetes için)
    @app.get("/live")
    async def liveness_check():
        """Servis canlı mı?"""
        return {"status": "alive"}

    # Background task örneği
    @app.post("/api/v1/send-email")
    async def send_email(
        email: str,
        background_tasks: BackgroundTasks
    ):
        """Email gönderimi (background task)"""

        async def send_email_task(recipient: str):
            await asyncio.sleep(5)  # Email gönderimi simülasyonu
            logger.info(f"Email sent to {recipient}")

        background_tasks.add_task(send_email_task, email)

        return {"message": "Email will be sent in background"}

    return app

# 4. Error Tracking (Sentry entegrasyonu)
def setup_error_tracking(app: FastAPI):
    """Hata takibi için Sentry entegrasyonu"""
    import sentry_sdk
    from sentry_sdk.integrations.asgi import SentryAsgiMiddleware

    sentry_sdk.init(
        dsn="your-sentry-dsn",
        environment="production",
        traces_sample_rate=1.0,
    )

    app.add_middleware(SentryAsgiMiddleware)

# 5. Caching Strategy
from functools import wraps
import hashlib
import json

# Basit in-memory cache (production'da Redis kullanın)
cache_store: Dict[str, Any] = {}

def cache(expire: int = 300):
    """Cache decorator"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Cache key oluştur
            cache_key = hashlib.md5(
                f"{func.__name__}:{json.dumps(args)}:{json.dumps(kwargs)}".encode()
            ).hexdigest()

            # Cache'de var mı kontrol et
            if cache_key in cache_store:
                logger.info(f"Cache hit: {func.__name__}")
                return cache_store[cache_key]

            # Fonksiyonu çalıştır
            result = await func(*args, **kwargs)

            # Cache'e kaydet
            cache_store[cache_key] = result
            logger.info(f"Cache miss: {func.__name__}")

            # Expire için background task (basit versiyon)
            # Production'da Redis EXPIRE kullanın

            return result
        return wrapper
    return decorator

# Cache kullanım örneği
@cache(expire=300)
async def get_expensive_data(param: str):
    """Pahalı bir işlem"""
    await asyncio.sleep(2)  # Simülasyon
    return {"data": f"Expensive result for {param}"}
```

---

## Özet

Bu dokümanda modern Python web geliştirme konularını ele aldık:

1. **HTTP Protokolü**: Metodlar, status kodları, headerlar
2. **REST API**: Resource-based URL tasarımı, HATEOAS, response formatları
3. **Flask**: Blueprints, error handling, application factory pattern
4. **FastAPI**: Async routes, Pydantic validation, otomatik dokümantasyon
5. **Authentication**: JWT, OAuth, role-based access control
6. **Middleware**: Custom middleware'ler, CORS, rate limiting
7. **API Versioning**: URL path, header, query parameter stratejileri
8. **Production**: Configuration, logging, error tracking, caching, health checks

### FastAPI vs Flask

**FastAPI Avantajları:**
- Otomatik API dokümantasyonu (Swagger, ReDoc)
- Async/await desteği (yüksek performans)
- Pydantic ile otomatik validasyon
- Type hints ile auto-completion
- Modern Python özelliklerini kullanır

**Flask Avantajları:**
- Daha mature ekosistem
- Daha fazla extension
- Öğrenmesi daha kolay
- Daha esnek (minimal framework)

Production ortamında FastAPI, performans ve geliştirme hızı açısından öne çıkar!
