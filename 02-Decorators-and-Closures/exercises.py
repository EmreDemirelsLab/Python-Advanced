"""
DECORATORS VE CLOSURES - İLERI SEVİYE ALIŞTIRMALAR
==================================================

Bu dosya, decorator ve closure konularında ileri seviye pratik yapmak için
hazırlanmış 18 alıştırma içermektedir.

Her alıştırma:
1. Soru açıklaması
2. TODO kısmı (buraya kod yazılacak)
3. Detaylı çözüm

Zorluk Seviyeleri:
- Medium: Orta seviye (1-6)
- Hard: Zor seviye (7-12)
- Expert: Uzman seviye (13-18)
"""

import functools
import time
from typing import Callable, Any, Dict, List
from collections import defaultdict
import inspect
import json

# =============================================================================
# ALIŞTIRMA 1: Smart Cache Decorator (Medium)
# =============================================================================
"""
SORU 1: Smart Cache Decorator

Aşağıdaki özelliklere sahip bir cache decorator'ı yazın:
- Fonksiyon sonuçlarını cache'lesin
- TTL (Time To Live) desteği olsun
- Cache boyutu sınırı olsun (LRU mantığı)
- Cache istatistikleri tutabilsin (hit/miss)
- Manuel cache temizleme fonksiyonu olsun

Örnek kullanım:
    @smart_cache(ttl=60, maxsize=100)
    def expensive_function(x, y):
        time.sleep(1)
        return x + y
"""

# TODO: smart_cache decorator'ını buraya yazın


# ============= ÇÖZÜM 1 =============

def smart_cache(ttl: int = 60, maxsize: int = 128):
    """
    Smart cache decorator with TTL and LRU support

    Args:
        ttl: Cache'de kalma süresi (saniye)
        maxsize: Maksimum cache boyutu
    """
    def decorator(func: Callable) -> Callable:
        cache: Dict[tuple, tuple] = {}  # {key: (result, timestamp)}
        stats = {'hits': 0, 'misses': 0}
        access_order: List[tuple] = []  # LRU tracking

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Cache key oluştur
            key = (args, tuple(sorted(kwargs.items())))
            current_time = time.time()

            # Cache'de var mı ve geçerli mi kontrol et
            if key in cache:
                result, timestamp = cache[key]
                if current_time - timestamp < ttl:
                    stats['hits'] += 1
                    # LRU güncelle
                    access_order.remove(key)
                    access_order.append(key)
                    return result
                else:
                    # TTL expired
                    del cache[key]
                    access_order.remove(key)

            # Cache miss - fonksiyonu çalıştır
            stats['misses'] += 1
            result = func(*args, **kwargs)

            # Cache'e ekle
            cache[key] = (result, current_time)
            access_order.append(key)

            # Maxsize kontrolü - LRU mantığı
            if len(cache) > maxsize:
                oldest_key = access_order.pop(0)
                del cache[oldest_key]

            return result

        def cache_info():
            """Cache istatistiklerini döndür"""
            return {
                'hits': stats['hits'],
                'misses': stats['misses'],
                'size': len(cache),
                'maxsize': maxsize,
                'hit_rate': stats['hits'] / (stats['hits'] + stats['misses'])
                           if (stats['hits'] + stats['misses']) > 0 else 0
            }

        def cache_clear():
            """Cache'i temizle"""
            cache.clear()
            access_order.clear()
            stats['hits'] = stats['misses'] = 0

        wrapper.cache_info = cache_info
        wrapper.cache_clear = cache_clear

        return wrapper
    return decorator

# Test
@smart_cache(ttl=5, maxsize=3)
def add_numbers(a: int, b: int) -> int:
    """Toplama işlemi (simüle edilmiş yavaş işlem)"""
    time.sleep(0.1)
    return a + b

print("=== ALIŞTIRMA 1 TEST ===")
print(f"İlk çağrı: {add_numbers(2, 3)}")
print(f"İkinci çağrı (cache'den): {add_numbers(2, 3)}")
print(f"Cache info: {add_numbers.cache_info()}")
add_numbers.cache_clear()
print(f"Cache temizlendi: {add_numbers.cache_info()}")
print()


# =============================================================================
# ALIŞTIRMA 2: Async Retry Decorator (Medium)
# =============================================================================
"""
SORU 2: Async Retry Decorator

Asenkron fonksiyonlar için retry mekanizması olan bir decorator yazın:
- Belirtilen exception'lar için retry yapsın
- Exponential backoff desteği olsun
- Maksimum retry sayısı ayarlanabilsin
- Her retry'da callback fonksiyonu çağırabilsin

Örnek kullanım:
    @async_retry(max_attempts=3, backoff=2.0, exceptions=(ConnectionError,))
    async def fetch_data(url):
        # API call
        pass
"""

# TODO: async_retry decorator'ını buraya yazın


# ============= ÇÖZÜM 2 =============

import asyncio

def async_retry(
    max_attempts: int = 3,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,),
    on_retry: Callable = None
):
    """
    Async retry decorator with exponential backoff

    Args:
        max_attempts: Maksimum deneme sayısı
        backoff: Her denemede bekleme süresinin çarpanı
        exceptions: Yakalanacak exception'lar
        on_retry: Her retry'da çağrılacak callback
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            attempt = 0
            delay = 1.0

            while attempt < max_attempts:
                try:
                    attempt += 1
                    return await func(*args, **kwargs)
                except exceptions as e:
                    if attempt >= max_attempts:
                        raise

                    if on_retry:
                        on_retry(attempt, max_attempts, e, delay)
                    else:
                        print(f"Attempt {attempt}/{max_attempts} failed: {e}")
                        print(f"Retrying in {delay:.2f} seconds...")

                    await asyncio.sleep(delay)
                    delay *= backoff

        return wrapper
    return decorator

# Test
@async_retry(max_attempts=3, backoff=2.0, exceptions=(ValueError,))
async def unreliable_async_function(should_fail: bool = True):
    """Başarısız olabilecek async fonksiyon"""
    print("Async function called")
    if should_fail:
        raise ValueError("Simulated error")
    return "Success!"

print("=== ALIŞTIRMA 2 TEST ===")
# asyncio.run() kullanarak test edilebilir
# asyncio.run(unreliable_async_function(should_fail=False))
print("Async retry decorator tanımlandı (test için asyncio.run() kullanın)")
print()


# =============================================================================
# ALIŞTIRMA 3: Class Method Profiler (Medium)
# =============================================================================
"""
SORU 3: Class Method Profiler

Bir sınıftaki tüm public methodları otomatik olarak profiling eden bir
class decorator yazın:
- Her method'un çağrılma sayısını tutsun
- Execution time'ı ölçsün
- Memory kullanımını kaydetsin (opsiyonel)
- Rapor oluşturabilsin

Örnek kullanım:
    @profile_class
    class MyService:
        def method1(self):
            pass
"""

# TODO: profile_class decorator'ını buraya yazın


# ============= ÇÖZÜM 3 =============

import sys

class ProfilerStats:
    """Profiler istatistikleri için singleton class"""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.stats = defaultdict(lambda: {
                'calls': 0,
                'total_time': 0,
                'min_time': float('inf'),
                'max_time': 0
            })
        return cls._instance

    def record(self, method_name: str, execution_time: float):
        """İstatistik kaydet"""
        stats = self.stats[method_name]
        stats['calls'] += 1
        stats['total_time'] += execution_time
        stats['min_time'] = min(stats['min_time'], execution_time)
        stats['max_time'] = max(stats['max_time'], execution_time)

    def get_report(self, class_name: str = None):
        """Rapor oluştur"""
        report = []
        report.append("\n=== PROFILER REPORT ===")

        for method_name, stats in sorted(self.stats.items()):
            if class_name and not method_name.startswith(class_name):
                continue

            avg_time = stats['total_time'] / stats['calls'] if stats['calls'] > 0 else 0
            report.append(f"\n{method_name}:")
            report.append(f"  Calls: {stats['calls']}")
            report.append(f"  Total Time: {stats['total_time']:.4f}s")
            report.append(f"  Avg Time: {avg_time:.4f}s")
            report.append(f"  Min Time: {stats['min_time']:.4f}s")
            report.append(f"  Max Time: {stats['max_time']:.4f}s")

        return '\n'.join(report)

def profile_class(cls):
    """Class profiler decorator"""
    profiler = ProfilerStats()

    # Tüm public methodları bul ve wrap et
    for attr_name in dir(cls):
        if attr_name.startswith('_'):
            continue

        attr_value = getattr(cls, attr_name)

        if callable(attr_value):
            # Method'u wrap et
            def make_wrapper(method, name):
                @functools.wraps(method)
                def wrapper(*args, **kwargs):
                    full_name = f"{cls.__name__}.{name}"
                    start_time = time.perf_counter()
                    try:
                        return method(*args, **kwargs)
                    finally:
                        execution_time = time.perf_counter() - start_time
                        profiler.record(full_name, execution_time)
                return wrapper

            setattr(cls, attr_name, make_wrapper(attr_value, attr_name))

    # Rapor method'u ekle
    cls.get_profile_report = lambda self: profiler.get_report(cls.__name__)

    return cls

# Test
@profile_class
class Calculator:
    """Profiled calculator class"""

    def add(self, a: int, b: int) -> int:
        time.sleep(0.01)
        return a + b

    def multiply(self, a: int, b: int) -> int:
        time.sleep(0.02)
        return a * b

print("=== ALIŞTIRMA 3 TEST ===")
calc = Calculator()
calc.add(5, 3)
calc.add(10, 20)
calc.multiply(4, 5)
print(calc.get_profile_report())
print()


# =============================================================================
# ALIŞTIRMA 4: Transaction Decorator (Medium)
# =============================================================================
"""
SORU 4: Transaction Decorator

Database transaction'ları için bir decorator yazın:
- Fonksiyon başarılı olursa commit
- Exception durumunda rollback
- Nested transaction desteği
- Transaction log'lama

Örnek kullanım:
    @transaction(isolation_level='READ_COMMITTED')
    def update_balance(user_id, amount):
        # database operations
        pass
"""

# TODO: transaction decorator'ını buraya yazın


# ============= ÇÖZÜM 4 =============

class DatabaseMock:
    """Mock database sınıfı"""
    def __init__(self):
        self.in_transaction = False
        self.transaction_level = 0
        self.operations = []

    def begin_transaction(self, isolation_level: str):
        self.transaction_level += 1
        self.in_transaction = True
        print(f"[DB] BEGIN TRANSACTION (level {self.transaction_level}) - {isolation_level}")

    def commit(self):
        print(f"[DB] COMMIT (level {self.transaction_level})")
        self.transaction_level -= 1
        if self.transaction_level == 0:
            self.in_transaction = False
            self.operations.clear()

    def rollback(self):
        print(f"[DB] ROLLBACK (level {self.transaction_level})")
        self.transaction_level -= 1
        if self.transaction_level == 0:
            self.in_transaction = False
            self.operations.clear()

    def execute(self, query: str):
        self.operations.append(query)
        print(f"[DB] EXECUTE: {query}")

# Global mock database
db = DatabaseMock()

def transaction(isolation_level: str = 'READ_COMMITTED'):
    """
    Transaction decorator with nested transaction support

    Args:
        isolation_level: Transaction isolation level
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Transaction başlat
            db.begin_transaction(isolation_level)

            try:
                result = func(*args, **kwargs)
                db.commit()
                return result
            except Exception as e:
                db.rollback()
                print(f"[DB] Transaction failed: {e}")
                raise

        return wrapper
    return decorator

# Test
@transaction(isolation_level='READ_COMMITTED')
def transfer_money(from_user: int, to_user: int, amount: float):
    """Para transfer işlemi"""
    db.execute(f"UPDATE accounts SET balance = balance - {amount} WHERE user_id = {from_user}")

    if amount < 0:
        raise ValueError("Amount cannot be negative")

    db.execute(f"UPDATE accounts SET balance = balance + {amount} WHERE user_id = {to_user}")
    return f"Transferred {amount} from {from_user} to {to_user}"

print("=== ALIŞTIRMA 4 TEST ===")
try:
    result = transfer_money(1, 2, 100)
    print(f"Success: {result}\n")
except ValueError as e:
    print(f"Error: {e}\n")

try:
    transfer_money(1, 2, -50)
except ValueError:
    print("Negative amount rejected\n")


# =============================================================================
# ALIŞTIRMA 5: Middleware Chain Decorator (Hard)
# =============================================================================
"""
SORU 5: Middleware Chain Decorator

Web framework'lerindeki gibi middleware chain oluşturan bir decorator sistemi yazın:
- Middleware'ler sırayla çalışsın
- Request/response objelerini işleyebilsin
- Middleware'ler chain'i durdurabilsin
- Context passing desteği olsun

Örnek kullanım:
    @middleware_chain([auth_middleware, logging_middleware])
    def api_endpoint(request):
        return {"data": "response"}
"""

# TODO: middleware_chain decorator'ını ve middleware sistemi yazın


# ============= ÇÖZÜM 5 =============

class Request:
    """Mock request objesi"""
    def __init__(self, path: str, method: str = "GET", headers: dict = None):
        self.path = path
        self.method = method
        self.headers = headers or {}
        self.context = {}  # Middleware'ler arası veri paylaşımı

class Response:
    """Mock response objesi"""
    def __init__(self, data: Any, status: int = 200):
        self.data = data
        self.status = status

class MiddlewareChain:
    """Middleware chain manager"""
    def __init__(self, middlewares: List[Callable]):
        self.middlewares = middlewares

    def execute(self, request: Request, handler: Callable) -> Response:
        """Middleware chain'i çalıştır"""
        def process(index: int):
            if index >= len(self.middlewares):
                # Son middleware'den sonra asıl handler'ı çağır
                return handler(request)

            middleware = self.middlewares[index]
            return middleware(request, lambda: process(index + 1))

        return process(0)

def middleware_chain(middlewares: List[Callable]):
    """Middleware chain decorator"""
    def decorator(func: Callable) -> Callable:
        chain = MiddlewareChain(middlewares)

        @functools.wraps(func)
        def wrapper(request: Request) -> Response:
            return chain.execute(request, func)

        return wrapper
    return decorator

# Middleware örnekleri
def auth_middleware(request: Request, next_handler: Callable) -> Response:
    """Authentication middleware"""
    print(f"[Auth] Checking authentication for {request.path}")

    if "Authorization" not in request.headers:
        print("[Auth] No auth token, returning 401")
        return Response({"error": "Unauthorized"}, status=401)

    request.context['user_id'] = 123  # Mock user
    print("[Auth] Authentication successful")
    return next_handler()

def logging_middleware(request: Request, next_handler: Callable) -> Response:
    """Logging middleware"""
    print(f"[Logging] {request.method} {request.path}")
    start_time = time.time()

    response = next_handler()

    duration = time.time() - start_time
    print(f"[Logging] Response {response.status} in {duration:.4f}s")
    return response

def rate_limit_middleware(request: Request, next_handler: Callable) -> Response:
    """Rate limiting middleware"""
    print(f"[RateLimit] Checking rate limit for {request.path}")
    # Mock rate limit check
    return next_handler()

# Test
@middleware_chain([auth_middleware, logging_middleware, rate_limit_middleware])
def api_endpoint(request: Request) -> Response:
    """API endpoint handler"""
    user_id = request.context.get('user_id')
    return Response({"message": "Success", "user_id": user_id})

print("=== ALIŞTIRMA 5 TEST ===")
# Başarılı request
req1 = Request("/api/users", headers={"Authorization": "Bearer token123"})
resp1 = api_endpoint(req1)
print(f"Response 1: {resp1.data}\n")

# Unauthorized request
req2 = Request("/api/users")
resp2 = api_endpoint(req2)
print(f"Response 2: {resp2.data}\n")


# =============================================================================
# ALIŞTIRMA 6: Dependency Injection Decorator (Hard)
# =============================================================================
"""
SORU 6: Dependency Injection Decorator

Dependency injection pattern'i uygulayan bir decorator yazın:
- Fonksiyon parametrelerine otomatik dependency inject etsin
- Type hints kullanarak dependency'leri çözümlesin
- Singleton ve transient scope desteği olsun
- Circular dependency detection

Örnek kullanım:
    @inject
    def process_data(service: DataService, logger: Logger):
        # dependencies otomatik inject edilir
        pass
"""

# TODO: inject decorator'ını ve DI container'ını yazın


# ============= ÇÖZÜM 6 =============

from typing import Type, get_type_hints

class DIContainer:
    """Dependency Injection Container"""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.services = {}  # {Type: instance/factory}
            cls._instance.singletons = {}  # {Type: instance}
            cls._instance.resolving = set()  # Circular dependency detection
        return cls._instance

    def register(self, interface: Type, implementation: Type = None, singleton: bool = True):
        """Service kaydet"""
        if implementation is None:
            implementation = interface

        self.services[interface] = {
            'implementation': implementation,
            'singleton': singleton
        }

    def resolve(self, interface: Type) -> Any:
        """Dependency çözümle"""
        # Circular dependency check
        if interface in self.resolving:
            raise RuntimeError(f"Circular dependency detected: {interface}")

        # Singleton check
        if interface in self.singletons:
            return self.singletons[interface]

        if interface not in self.services:
            # Auto-registration: Eğer constructor varsa direkt oluştur
            try:
                instance = interface()
                return instance
            except:
                raise RuntimeError(f"Service not registered: {interface}")

        service_info = self.services[interface]
        implementation = service_info['implementation']

        # Circular dependency protection
        self.resolving.add(interface)

        try:
            # Constructor dependencies'i çözümle
            hints = get_type_hints(implementation.__init__) if hasattr(implementation, '__init__') else {}
            deps = {}

            for param_name, param_type in hints.items():
                if param_name != 'return':
                    deps[param_name] = self.resolve(param_type)

            # Instance oluştur
            instance = implementation(**deps)

            # Singleton ise kaydet
            if service_info['singleton']:
                self.singletons[interface] = instance

            return instance
        finally:
            self.resolving.discard(interface)

# Global DI container
container = DIContainer()

def inject(func: Callable) -> Callable:
    """Dependency injection decorator"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Type hints'i al
        hints = get_type_hints(func)

        # Her parametre için dependency çözümle
        for param_name, param_type in hints.items():
            if param_name not in kwargs and param_name != 'return':
                kwargs[param_name] = container.resolve(param_type)

        return func(*args, **kwargs)

    return wrapper

# Test services
class Logger:
    """Logger service"""
    def log(self, message: str):
        print(f"[LOG] {message}")

class Database:
    """Database service"""
    def __init__(self):
        print("Database initialized")

    def query(self, sql: str):
        return f"Query result: {sql}"

class UserService:
    """User service with dependencies"""
    def __init__(self, database: Database, logger: Logger):
        self.database = database
        self.logger = logger
        self.logger.log("UserService initialized")

    def get_user(self, user_id: int):
        result = self.database.query(f"SELECT * FROM users WHERE id={user_id}")
        self.logger.log(f"Fetched user {user_id}")
        return result

# Services'leri kaydet
container.register(Logger, singleton=True)
container.register(Database, singleton=True)
container.register(UserService, singleton=False)  # Transient

# Test function
@inject
def process_user_data(service: UserService, logger: Logger, user_id: int = 123):
    """User data işleme - dependencies inject edilir"""
    logger.log(f"Processing user {user_id}")
    return service.get_user(user_id)

print("=== ALIŞTIRMA 6 TEST ===")
result = process_user_data()
print(f"Result: {result}\n")


# =============================================================================
# ALIŞTIRMA 7: Method Dispatch Decorator (Hard)
# =============================================================================
"""
SORU 7: Method Dispatch Decorator

functools.singledispatch benzeri ama daha gelişmiş bir dispatch sistemi yazın:
- Multiple dispatch (birden fazla parametre tipine göre)
- Type annotation kullanarak otomatik dispatch
- Default handler desteği
- Dispatch öncelik sistemi

Örnek kullanım:
    @multidispatch
    def process(data):
        pass  # default

    @process.register(int, int)
    def _(x, y):
        return x + y
"""

# TODO: multidispatch decorator'ını yazın


# ============= ÇÖZÜM 7 =============

class MultiDispatch:
    """Multiple dispatch implementation"""
    def __init__(self, func: Callable):
        self.default_func = func
        self.registry: Dict[tuple, Callable] = {}
        functools.update_wrapper(self, func)

    def register(self, *types):
        """Tip kombinasyonu için handler kaydet"""
        def decorator(func: Callable) -> Callable:
            self.registry[types] = func
            return func
        return decorator

    def __call__(self, *args, **kwargs):
        # Argüman tiplerini al
        arg_types = tuple(type(arg) for arg in args)

        # Exact match ara
        if arg_types in self.registry:
            return self.registry[arg_types](*args, **kwargs)

        # Subclass match ara (inheritance desteği)
        for registered_types, func in self.registry.items():
            if len(registered_types) == len(arg_types):
                if all(isinstance(arg, reg_type) for arg, reg_type in zip(args, registered_types)):
                    return func(*args, **kwargs)

        # Default handler
        return self.default_func(*args, **kwargs)

def multidispatch(func: Callable) -> MultiDispatch:
    """Multi-dispatch decorator"""
    return MultiDispatch(func)

# Test
@multidispatch
def process_data(data):
    """Default handler"""
    return f"Generic processing: {data}"

@process_data.register(int, int)
def _(x: int, y: int):
    """İki integer için özel işlem"""
    return f"Adding integers: {x + y}"

@process_data.register(str, str)
def _(x: str, y: str):
    """İki string için özel işlem"""
    return f"Concatenating strings: {x + y}"

@process_data.register(list, int)
def _(lst: list, multiplier: int):
    """List ve integer için özel işlem"""
    return f"List repeated {multiplier} times: {lst * multiplier}"

print("=== ALIŞTIRMA 7 TEST ===")
print(process_data(5, 10))              # Adding integers: 15
print(process_data("Hello", "World"))   # Concatenating strings: HelloWorld
print(process_data([1, 2], 3))          # List repeated 3 times
print(process_data(42))                 # Generic processing: 42
print()


# =============================================================================
# ALIŞTIRMA 8: Context Aware Decorator (Hard)
# =============================================================================
"""
SORU 8: Context Aware Decorator

Context'e göre farklı davranış gösteren bir decorator yazın:
- Environment'a göre (dev/prod) farklı davranış
- User role'üne göre farklı davranış
- Time-based davranış (working hours, weekend, etc)
- Feature flag desteği

Örnek kullanım:
    @context_aware(
        dev=lambda: print("Dev mode"),
        prod=lambda: time.sleep(1)
    )
    def process():
        pass
"""

# TODO: context_aware decorator'ını yazın


# ============= ÇÖZÜM 8 =============

from datetime import datetime
from enum import Enum

class Environment(Enum):
    """Environment tipleri"""
    DEV = "dev"
    STAGING = "staging"
    PROD = "prod"

class AppContext:
    """Global application context"""
    def __init__(self):
        self.environment = Environment.DEV
        self.user_role = "user"
        self.feature_flags = {}

    def is_business_hours(self) -> bool:
        """İş saatleri kontrolü"""
        now = datetime.now()
        return 9 <= now.hour < 17 and now.weekday() < 5

    def is_feature_enabled(self, feature: str) -> bool:
        """Feature flag kontrolü"""
        return self.feature_flags.get(feature, False)

# Global context
app_context = AppContext()

def context_aware(
    dev: Callable = None,
    staging: Callable = None,
    prod: Callable = None,
    business_hours_only: bool = False,
    require_role: str = None,
    require_feature: str = None
):
    """
    Context-aware decorator

    Args:
        dev: Dev environment için özel davranış
        staging: Staging environment için özel davranış
        prod: Prod environment için özel davranış
        business_hours_only: Sadece iş saatlerinde çalışsın mı
        require_role: Gerekli kullanıcı rolü
        require_feature: Gerekli feature flag
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Business hours kontrolü
            if business_hours_only and not app_context.is_business_hours():
                raise RuntimeError("This function can only be called during business hours")

            # Role kontrolü
            if require_role and app_context.user_role != require_role:
                raise PermissionError(f"Required role: {require_role}, current: {app_context.user_role}")

            # Feature flag kontrolü
            if require_feature and not app_context.is_feature_enabled(require_feature):
                raise RuntimeError(f"Feature not enabled: {require_feature}")

            # Environment-specific davranış
            env_handlers = {
                Environment.DEV: dev,
                Environment.STAGING: staging,
                Environment.PROD: prod
            }

            handler = env_handlers.get(app_context.environment)
            if handler:
                handler()

            return func(*args, **kwargs)

        return wrapper
    return decorator

# Test
@context_aware(
    dev=lambda: print("[DEV] Skipping validation"),
    prod=lambda: print("[PROD] Full validation enabled")
)
def process_payment(amount: float):
    """Payment processing"""
    return f"Processing ${amount}"

@context_aware(
    business_hours_only=True,
    require_role="admin"
)
def admin_task():
    """Admin-only task during business hours"""
    return "Admin task completed"

print("=== ALIŞTIRMA 8 TEST ===")
# Dev mode test
app_context.environment = Environment.DEV
print(process_payment(100.0))

# Prod mode test
app_context.environment = Environment.PROD
print(process_payment(100.0))

# Admin role test
app_context.user_role = "admin"
try:
    print(admin_task())
except RuntimeError as e:
    print(f"Error: {e} (outside business hours)")
print()


# =============================================================================
# ALIŞTIRMA 9: Event Emitter Decorator (Hard)
# =============================================================================
"""
SORU 9: Event Emitter Decorator

Event-driven architecture için event emitter decorator yazın:
- Fonksiyon çalışmadan önce ve sonra event emit etsin
- Event listener'lar kaydedilebilsin
- Asenkron event handling desteği
- Event filtering ve transformation

Örnek kullanım:
    @emit_events('user.created', 'user.validated')
    def create_user(username, email):
        return User(username, email)
"""

# TODO: emit_events decorator'ını ve event system'i yazın


# ============= ÇÖZÜM 9 =============

class EventEmitter:
    """Event emitter singleton"""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.listeners: Dict[str, List[Callable]] = defaultdict(list)
            cls._instance.event_history: List[Dict] = []
        return cls._instance

    def on(self, event_name: str, handler: Callable):
        """Event listener kaydet"""
        self.listeners[event_name].append(handler)

    def emit(self, event_name: str, data: Any = None):
        """Event emit et"""
        event = {
            'name': event_name,
            'data': data,
            'timestamp': time.time()
        }
        self.event_history.append(event)

        print(f"[EVENT] {event_name} emitted with data: {data}")

        # Listener'ları çağır
        for handler in self.listeners.get(event_name, []):
            try:
                handler(event)
            except Exception as e:
                print(f"[EVENT ERROR] Handler for {event_name} failed: {e}")

    def get_history(self, event_name: str = None) -> List[Dict]:
        """Event geçmişini getir"""
        if event_name:
            return [e for e in self.event_history if e['name'] == event_name]
        return self.event_history

# Global emitter
emitter = EventEmitter()

def emit_events(*event_names, before: bool = True, after: bool = True):
    """
    Event emitter decorator

    Args:
        event_names: Emit edilecek event isimleri
        before: Fonksiyon öncesi event emit et
        after: Fonksiyon sonrası event emit et
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            func_name = func.__name__

            # Before event
            if before:
                for event_name in event_names:
                    emitter.emit(f"{event_name}.before", {
                        'function': func_name,
                        'args': args,
                        'kwargs': kwargs
                    })

            # Fonksiyonu çalıştır
            try:
                result = func(*args, **kwargs)

                # After event (success)
                if after:
                    for event_name in event_names:
                        emitter.emit(f"{event_name}.after", {
                            'function': func_name,
                            'result': result
                        })

                return result

            except Exception as e:
                # Error event
                for event_name in event_names:
                    emitter.emit(f"{event_name}.error", {
                        'function': func_name,
                        'error': str(e)
                    })
                raise

        return wrapper
    return decorator

# Event listeners
def log_user_creation(event):
    """User creation logger"""
    print(f"[LISTENER] User creation event: {event['data']}")

def notify_admins(event):
    """Admin notification"""
    print(f"[LISTENER] Notifying admins about: {event['name']}")

# Listener'ları kaydet
emitter.on('user.created.after', log_user_creation)
emitter.on('user.created.after', notify_admins)

# Test
@emit_events('user.created', 'user.validated')
def create_user(username: str, email: str):
    """User oluşturma fonksiyonu"""
    print(f"Creating user: {username}")
    return {'username': username, 'email': email, 'id': 123}

print("=== ALIŞTIRMA 9 TEST ===")
user = create_user("john_doe", "john@example.com")
print(f"Created user: {user}\n")


# =============================================================================
# ALIŞTIRMA 10: Distributed Lock Decorator (Expert)
# =============================================================================
"""
SORU 10: Distributed Lock Decorator

Distributed sistemler için lock decorator'ı yazın:
- Redis-like lock mekanizması (mock)
- Lock timeout desteği
- Auto-renewal için heartbeat
- Deadlock prevention

Örnek kullanım:
    @distributed_lock('resource:user:123', timeout=30)
    def update_user_balance(user_id, amount):
        # Thread-safe işlem
        pass
"""

# TODO: distributed_lock decorator'ını yazın


# ============= ÇÖZÜM 10 =============

import threading
import uuid

class DistributedLockManager:
    """Mock distributed lock manager (Redis benzeri)"""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.locks: Dict[str, Dict] = {}
            cls._instance.lock = threading.Lock()
        return cls._instance

    def acquire(self, resource_key: str, timeout: float = 30) -> str:
        """Lock al"""
        with self.lock:
            now = time.time()

            # Mevcut lock'u kontrol et
            if resource_key in self.locks:
                lock_info = self.locks[resource_key]
                # Timeout kontrolü
                if now - lock_info['acquired_at'] < timeout:
                    return None  # Lock alınamadı

            # Yeni lock oluştur
            lock_id = str(uuid.uuid4())
            self.locks[resource_key] = {
                'lock_id': lock_id,
                'acquired_at': now,
                'timeout': timeout
            }
            print(f"[LOCK] Acquired lock on {resource_key} (id: {lock_id[:8]}...)")
            return lock_id

    def release(self, resource_key: str, lock_id: str) -> bool:
        """Lock'u serbest bırak"""
        with self.lock:
            if resource_key in self.locks:
                if self.locks[resource_key]['lock_id'] == lock_id:
                    del self.locks[resource_key]
                    print(f"[LOCK] Released lock on {resource_key}")
                    return True
        return False

    def extend(self, resource_key: str, lock_id: str, extension: float) -> bool:
        """Lock süresini uzat"""
        with self.lock:
            if resource_key in self.locks:
                lock_info = self.locks[resource_key]
                if lock_info['lock_id'] == lock_id:
                    lock_info['acquired_at'] = time.time()
                    lock_info['timeout'] += extension
                    return True
        return False

# Global lock manager
lock_manager = DistributedLockManager()

def distributed_lock(resource_key: str, timeout: float = 30, wait: bool = False, wait_timeout: float = 60):
    """
    Distributed lock decorator

    Args:
        resource_key: Lock resource anahtarı
        timeout: Lock timeout süresi
        wait: Lock alınamazsa bekle
        wait_timeout: Maksimum bekleme süresi
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            lock_id = None
            start_wait = time.time()

            # Lock almayı dene
            while True:
                lock_id = lock_manager.acquire(resource_key, timeout)

                if lock_id:
                    break

                if not wait:
                    raise RuntimeError(f"Could not acquire lock on {resource_key}")

                if time.time() - start_wait > wait_timeout:
                    raise TimeoutError(f"Lock wait timeout on {resource_key}")

                print(f"[LOCK] Waiting for lock on {resource_key}...")
                time.sleep(0.5)

            # Fonksiyonu çalıştır
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                # Lock'u serbest bırak
                if lock_id:
                    lock_manager.release(resource_key, lock_id)

        return wrapper
    return decorator

# Test
@distributed_lock('account:balance', timeout=10)
def update_balance(account_id: int, amount: float):
    """Thread-safe balance update"""
    print(f"Updating balance for account {account_id}")
    time.sleep(1)  # Simüle edilmiş işlem
    return f"Balance updated: +${amount}"

print("=== ALIŞTIRMA 10 TEST ===")
result = update_balance(123, 50.0)
print(f"Result: {result}\n")


# =============================================================================
# ALIŞTIRMA 11: State Machine Decorator (Expert)
# =============================================================================
"""
SORU 11: State Machine Decorator

State machine pattern'i uygulayan decorator yazın:
- State transition'ları yönetsin
- Invalid transition'ları engellesin
- State history tutsun
- Event-driven state changes

Örnek kullanım:
    @state_machine(initial='draft', transitions={
        'draft': ['review', 'rejected'],
        'review': ['approved', 'rejected'],
        'approved': [],
        'rejected': ['draft']
    })
    class Document:
        pass
"""

# TODO: state_machine decorator'ını yazın


# ============= ÇÖZÜM 11 =============

class StateMachine:
    """State machine implementation"""
    def __init__(self, initial_state: str, transitions: Dict[str, List[str]]):
        self.current_state = initial_state
        self.transitions = transitions
        self.history: List[Dict] = [{
            'state': initial_state,
            'timestamp': time.time(),
            'transition': 'initial'
        }]

    def can_transition(self, new_state: str) -> bool:
        """Transition mümkün mü?"""
        return new_state in self.transitions.get(self.current_state, [])

    def transition(self, new_state: str, reason: str = None):
        """State transition yap"""
        if not self.can_transition(new_state):
            raise ValueError(
                f"Invalid transition from {self.current_state} to {new_state}. "
                f"Valid transitions: {self.transitions.get(self.current_state, [])}"
            )

        old_state = self.current_state
        self.current_state = new_state

        self.history.append({
            'from': old_state,
            'to': new_state,
            'timestamp': time.time(),
            'reason': reason
        })

        print(f"[STATE] Transitioned from {old_state} to {new_state}")

    def get_history(self) -> List[Dict]:
        """State geçmişini getir"""
        return self.history.copy()

def state_machine(initial: str, transitions: Dict[str, List[str]]):
    """
    State machine decorator for classes

    Args:
        initial: İlk state
        transitions: State transition map'i
    """
    def decorator(cls):
        # Original __init__'i sakla
        original_init = cls.__init__

        def new_init(self, *args, **kwargs):
            # State machine ekle
            self._state_machine = StateMachine(initial, transitions)
            original_init(self, *args, **kwargs)

        cls.__init__ = new_init

        # State yönetimi methodları ekle
        def transition_to(self, new_state: str, reason: str = None):
            """State transition"""
            self._state_machine.transition(new_state, reason)

        def get_state(self) -> str:
            """Mevcut state"""
            return self._state_machine.current_state

        def can_transition_to(self, new_state: str) -> bool:
            """Transition kontrolü"""
            return self._state_machine.can_transition(new_state)

        def get_state_history(self) -> List[Dict]:
            """State geçmişi"""
            return self._state_machine.get_history()

        cls.transition_to = transition_to
        cls.get_state = get_state
        cls.can_transition_to = can_transition_to
        cls.get_state_history = get_state_history

        return cls

    return decorator

# Test
@state_machine(
    initial='draft',
    transitions={
        'draft': ['in_review', 'rejected'],
        'in_review': ['approved', 'rejected', 'draft'],
        'approved': ['archived'],
        'rejected': ['draft'],
        'archived': []
    }
)
class Document:
    """Document with state machine"""
    def __init__(self, title: str):
        self.title = title

    def __repr__(self):
        return f"Document('{self.title}', state='{self.get_state()}')"

print("=== ALIŞTIRMA 11 TEST ===")
doc = Document("Project Proposal")
print(f"Initial: {doc}")

doc.transition_to('in_review', reason="Ready for review")
print(f"After review: {doc}")

doc.transition_to('approved', reason="Looks good")
print(f"After approval: {doc}")

try:
    doc.transition_to('draft', reason="Back to draft")  # Invalid!
except ValueError as e:
    print(f"Error: {e}")

print(f"\nState history: {doc.get_state_history()}")
print()


# =============================================================================
# ALIŞTIRMA 12: Circuit Breaker Decorator (Expert)
# =============================================================================
"""
SORU 12: Circuit Breaker Decorator

Microservices pattern'i olan circuit breaker uygulayın:
- Failure threshold'a göre circuit'i aç
- Half-open state desteği
- Success rate monitoring
- Auto-recovery mekanizması

Örnek kullanım:
    @circuit_breaker(failure_threshold=5, timeout=60)
    def external_api_call():
        # Başarısız olabilecek işlem
        pass
"""

# TODO: circuit_breaker decorator'ını yazın


# ============= ÇÖZÜM 12 =============

from enum import Enum

class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"       # Normal işlem
    OPEN = "open"           # Circuit açık, istekler reddediliyor
    HALF_OPEN = "half_open"  # Test ediliyor

class CircuitBreaker:
    """Circuit breaker implementation"""
    def __init__(self, failure_threshold: int = 5, timeout: float = 60, half_open_requests: int = 3):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.half_open_requests = half_open_requests

        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.half_open_attempts = 0

    def call(self, func: Callable, *args, **kwargs):
        """Circuit breaker ile fonksiyon çağır"""
        # Circuit open mı?
        if self.state == CircuitState.OPEN:
            # Timeout geçti mi?
            if time.time() - self.last_failure_time >= self.timeout:
                print("[CIRCUIT] Transitioning to HALF_OPEN state")
                self.state = CircuitState.HALF_OPEN
                self.half_open_attempts = 0
            else:
                raise RuntimeError("Circuit breaker is OPEN - request rejected")

        try:
            result = func(*args, **kwargs)

            # Success handling
            if self.state == CircuitState.HALF_OPEN:
                self.half_open_attempts += 1
                if self.half_open_attempts >= self.half_open_requests:
                    print("[CIRCUIT] Transitioning to CLOSED state")
                    self.state = CircuitState.CLOSED
                    self.failure_count = 0

            self.success_count += 1
            return result

        except Exception as e:
            # Failure handling
            self.failure_count += 1
            self.last_failure_time = time.time()

            if self.state == CircuitState.HALF_OPEN:
                print("[CIRCUIT] Failure in HALF_OPEN, returning to OPEN")
                self.state = CircuitState.OPEN

            elif self.failure_count >= self.failure_threshold:
                print(f"[CIRCUIT] Threshold reached ({self.failure_count}), opening circuit")
                self.state = CircuitState.OPEN

            raise

    def get_stats(self) -> Dict:
        """Circuit breaker istatistikleri"""
        return {
            'state': self.state.value,
            'failure_count': self.failure_count,
            'success_count': self.success_count,
            'failure_threshold': self.failure_threshold
        }

def circuit_breaker(failure_threshold: int = 5, timeout: float = 60, half_open_requests: int = 3):
    """
    Circuit breaker decorator

    Args:
        failure_threshold: Kaç hata sonrası circuit açılsın
        timeout: Circuit ne kadar süre açık kalsın
        half_open_requests: Half-open state'de kaç başarılı istek gerekli
    """
    breaker = CircuitBreaker(failure_threshold, timeout, half_open_requests)

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return breaker.call(func, *args, **kwargs)

        wrapper.get_stats = breaker.get_stats
        wrapper.breaker = breaker

        return wrapper
    return decorator

# Test
failure_counter = 0

@circuit_breaker(failure_threshold=3, timeout=5, half_open_requests=2)
def unreliable_service():
    """Başarısız olabilecek servis çağrısı"""
    global failure_counter
    failure_counter += 1

    print(f"[SERVICE] Attempt {failure_counter}")

    if failure_counter <= 5:
        raise ConnectionError("Service unavailable")

    return "Success!"

print("=== ALIŞTIRMA 12 TEST ===")
for i in range(10):
    try:
        result = unreliable_service()
        print(f"Result: {result}")
    except (ConnectionError, RuntimeError) as e:
        print(f"Error: {e}")

    print(f"Stats: {unreliable_service.get_stats()}")
    time.sleep(0.5)
    print()


# =============================================================================
# ALIŞTIRMA 13: Aspect-Oriented Programming (Expert)
# =============================================================================
"""
SORU 13: Aspect-Oriented Programming Decorator

AOP pattern'i uygulayan decorator sistemi yazın:
- Before, After, Around advice'lar
- Join point'leri tanımlayabilme
- Aspect composition
- Point-cut expressions

Örnek kullanım:
    @aspect(before=log_entry, after=log_exit, around=measure_time)
    def business_logic():
        pass
"""

# TODO: AOP decorator sistemini yazın


# ============= ÇÖZÜM 13 =============

from typing import Optional

class JoinPoint:
    """Join point - advice'ların erişebileceği method bilgisi"""
    def __init__(self, func: Callable, args: tuple, kwargs: dict):
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self.result = None
        self.exception = None

    def proceed(self):
        """Original method'u çalıştır"""
        return self.func(*self.args, **self.kwargs)

def aspect(
    before: Optional[Callable] = None,
    after: Optional[Callable] = None,
    around: Optional[Callable] = None,
    on_exception: Optional[Callable] = None
):
    """
    Aspect-oriented programming decorator

    Args:
        before: Method öncesi çalışacak advice
        after: Method sonrası çalışacak advice (return değerini alır)
        around: Method'u wrap eden advice (join_point alır)
        on_exception: Exception durumunda çalışacak advice
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            join_point = JoinPoint(func, args, kwargs)

            try:
                # Before advice
                if before:
                    before(join_point)

                # Around advice varsa, method çağrısını ona bırak
                if around:
                    result = around(join_point)
                else:
                    result = func(*args, **kwargs)

                join_point.result = result

                # After advice
                if after:
                    after(join_point)

                return result

            except Exception as e:
                join_point.exception = e

                # Exception advice
                if on_exception:
                    on_exception(join_point)

                raise

        return wrapper
    return decorator

# Advice örnekleri
def log_entry_advice(jp: JoinPoint):
    """Before advice - method giriş log'u"""
    print(f"[BEFORE] Entering {jp.func.__name__} with args={jp.args}, kwargs={jp.kwargs}")

def log_exit_advice(jp: JoinPoint):
    """After advice - method çıkış log'u"""
    print(f"[AFTER] Exiting {jp.func.__name__} with result={jp.result}")

def timing_advice(jp: JoinPoint):
    """Around advice - timing measurement"""
    print(f"[AROUND] Starting timer for {jp.func.__name__}")
    start_time = time.time()

    result = jp.proceed()

    duration = time.time() - start_time
    print(f"[AROUND] {jp.func.__name__} took {duration:.4f}s")

    return result

def exception_advice(jp: JoinPoint):
    """Exception advice - hata yönetimi"""
    print(f"[EXCEPTION] {jp.func.__name__} raised {type(jp.exception).__name__}: {jp.exception}")

# Test
@aspect(
    before=log_entry_advice,
    after=log_exit_advice,
    around=timing_advice,
    on_exception=exception_advice
)
def calculate_fibonacci(n: int) -> int:
    """Fibonacci hesaplama with aspects"""
    if n < 0:
        raise ValueError("n must be non-negative")
    if n < 2:
        return n
    return calculate_fibonacci(n - 1) + calculate_fibonacci(n - 2)

print("=== ALIŞTIRMA 13 TEST ===")
result = calculate_fibonacci(5)
print(f"Result: {result}\n")

try:
    calculate_fibonacci(-1)
except ValueError:
    print("Negative input handled\n")


# =============================================================================
# ALIŞTIRMA 14: Memory-Efficient Iterator Decorator (Expert)
# =============================================================================
"""
SORU 14: Memory-Efficient Iterator Decorator

Generator'ları optimize eden ve memory-efficient hale getiren decorator:
- Lazy evaluation
- Chunked processing
- Memory usage monitoring
- Automatic batching

Örnek kullanım:
    @memory_efficient(chunk_size=1000)
    def process_large_dataset():
        for item in huge_list:
            yield processed_item
"""

# TODO: memory_efficient decorator'ını yazın


# ============= ÇÖZÜM 14 =============

import sys

def memory_efficient(chunk_size: int = 1000, monitor: bool = True):
    """
    Memory-efficient iterator decorator

    Args:
        chunk_size: İşlenecek chunk boyutu
        monitor: Memory kullanımını izle
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Generator'ı al
            generator = func(*args, **kwargs)

            if not hasattr(generator, '__iter__'):
                return generator

            # Chunked processing
            chunk = []
            total_processed = 0

            for item in generator:
                chunk.append(item)

                if len(chunk) >= chunk_size:
                    # Chunk'ı işle ve yield et
                    for processed_item in chunk:
                        yield processed_item
                        total_processed += 1

                    if monitor:
                        memory_usage = sys.getsizeof(chunk)
                        print(f"[MEMORY] Processed {total_processed} items, "
                              f"chunk memory: {memory_usage} bytes")

                    chunk.clear()

            # Kalan itemları işle
            for item in chunk:
                yield item
                total_processed += 1

            if monitor:
                print(f"[MEMORY] Total processed: {total_processed} items")

        return wrapper
    return decorator

# Test
@memory_efficient(chunk_size=5, monitor=True)
def generate_numbers(n: int):
    """Büyük sayı dizisi generator'ı"""
    for i in range(n):
        # Simüle edilmiş ağır işlem
        yield i ** 2

print("=== ALIŞTIRMA 14 TEST ===")
result = list(generate_numbers(20))
print(f"Generated {len(result)} numbers\n")


# =============================================================================
# ALIŞTIRMA 15: Adaptive Decorator (Expert)
# =============================================================================
"""
SORU 15: Adaptive Decorator

Çalışma zamanında davranışını optimize eden adaptive decorator:
- Performance metriklerini izle
- Otomatik caching stratejisi seç
- Load-based throttling
- Self-tuning parameters

Örnek kullanım:
    @adaptive(optimize_for='throughput')
    def data_processing(data):
        return processed_data
"""

# TODO: adaptive decorator'ını yazın


# ============= ÇÖZÜM 15 =============

class AdaptiveOptimizer:
    """Adaptive optimization engine"""
    def __init__(self, optimize_for: str = 'throughput'):
        self.optimize_for = optimize_for
        self.metrics = {
            'call_count': 0,
            'total_time': 0,
            'avg_time': 0,
            'cache_enabled': False
        }
        self.cache = {}
        self.cache_threshold = 0.1  # Cache'i etkinleştirme eşiği (saniye)

    def should_cache(self) -> bool:
        """Cache'in etkinleştirilip etkinleştirilmeyeceğine karar ver"""
        if self.metrics['call_count'] < 5:
            return False

        avg_time = self.metrics['avg_time']
        return avg_time > self.cache_threshold

    def update_metrics(self, execution_time: float):
        """Metrikleri güncelle"""
        self.metrics['call_count'] += 1
        self.metrics['total_time'] += execution_time
        self.metrics['avg_time'] = self.metrics['total_time'] / self.metrics['call_count']

        # Cache stratejisini güncelle
        should_cache = self.should_cache()
        if should_cache and not self.metrics['cache_enabled']:
            print(f"[ADAPTIVE] Enabling cache (avg time: {self.metrics['avg_time']:.4f}s)")
            self.metrics['cache_enabled'] = True
        elif not should_cache and self.metrics['cache_enabled']:
            print(f"[ADAPTIVE] Disabling cache (avg time: {self.metrics['avg_time']:.4f}s)")
            self.metrics['cache_enabled'] = False
            self.cache.clear()

def adaptive(optimize_for: str = 'throughput'):
    """
    Adaptive optimization decorator

    Args:
        optimize_for: 'throughput' veya 'latency'
    """
    optimizer = AdaptiveOptimizer(optimize_for)

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Cache kontrolü
            if optimizer.metrics['cache_enabled']:
                cache_key = (args, tuple(sorted(kwargs.items())))
                if cache_key in optimizer.cache:
                    print(f"[ADAPTIVE] Cache hit")
                    return optimizer.cache[cache_key]

            # Fonksiyonu çalıştır ve timing al
            start_time = time.perf_counter()
            result = func(*args, **kwargs)
            execution_time = time.perf_counter() - start_time

            # Metrikleri güncelle
            optimizer.update_metrics(execution_time)

            # Cache'e ekle
            if optimizer.metrics['cache_enabled']:
                cache_key = (args, tuple(sorted(kwargs.items())))
                optimizer.cache[cache_key] = result

            return result

        wrapper.get_metrics = lambda: optimizer.metrics.copy()
        wrapper.optimizer = optimizer

        return wrapper
    return decorator

# Test
@adaptive(optimize_for='throughput')
def expensive_computation(n: int) -> int:
    """Pahalı hesaplama"""
    time.sleep(0.05)  # Simüle edilmiş ağır işlem
    return sum(range(n))

print("=== ALIŞTIRMA 15 TEST ===")
for i in range(10):
    result = expensive_computation(100)
    metrics = expensive_computation.get_metrics()
    print(f"Call {i+1}: result={result}, metrics={metrics}")

print()


# =============================================================================
# ALIŞTIRMA 16: Hot Reload Decorator (Expert)
# =============================================================================
"""
SORU 16: Hot Reload Decorator

Development sırasında fonksiyonun hot-reload edilmesini sağlayan decorator:
- Source code değişikliklerini izle
- Otomatik reload
- State preservation
- Reload hooks

Örnek kullanım:
    @hot_reload(watch_file=__file__)
    def api_handler(request):
        return response
"""

# TODO: hot_reload decorator'ını yazın


# ============= ÇÖZÜM 16 =============

import os
import importlib

class HotReloader:
    """Hot reload implementation"""
    def __init__(self, watch_file: str):
        self.watch_file = watch_file
        self.last_mtime = os.path.getmtime(watch_file) if os.path.exists(watch_file) else 0
        self.reload_count = 0

    def check_and_reload(self, func: Callable) -> Callable:
        """Dosya değişikliğini kontrol et ve gerekirse reload et"""
        if not os.path.exists(self.watch_file):
            return func

        current_mtime = os.path.getmtime(self.watch_file)

        if current_mtime > self.last_mtime:
            self.last_mtime = current_mtime
            self.reload_count += 1
            print(f"[HOT-RELOAD] File changed, reloading... (reload #{self.reload_count})")

            # Not: Gerçek bir hot-reload için module reload gerekir
            # Bu basit implementasyonda sadece değişiklik tespit edilir

        return func

def hot_reload(watch_file: str = None, check_interval: int = 5):
    """
    Hot reload decorator

    Args:
        watch_file: İzlenecek dosya
        check_interval: Kontrol aralığı (saniye)
    """
    if watch_file is None:
        watch_file = __file__

    reloader = HotReloader(watch_file)
    last_check = [time.time()]  # Mutable object for closure

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Periyodik kontrol
            current_time = time.time()
            if current_time - last_check[0] >= check_interval:
                last_check[0] = current_time
                reloader.check_and_reload(func)

            return func(*args, **kwargs)

        wrapper.reloader = reloader

        return wrapper
    return decorator

# Test
@hot_reload(watch_file=__file__, check_interval=1)
def api_handler(name: str) -> str:
    """API handler with hot reload"""
    return f"Hello, {name}! (Version 1.0)"

print("=== ALIŞTIRMA 16 TEST ===")
print(api_handler("Alice"))
print(f"Reload count: {api_handler.reloader.reload_count}")
print()


# =============================================================================
# ALIŞTIRMA 17: Plugin System Decorator (Expert)
# =============================================================================
"""
SORU 17: Plugin System Decorator

Pluggable architecture için plugin decorator yazın:
- Plugin registration
- Priority-based execution
- Plugin dependencies
- Enable/disable plugins dynamically

Örnek kullanım:
    @plugin('image_processor', priority=10, depends_on=['file_loader'])
    def process_image(image):
        return processed_image
"""

# TODO: plugin system decorator'ını yazın


# ============= ÇÖZÜM 17 =============

class PluginRegistry:
    """Plugin registry singleton"""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.plugins: Dict[str, Dict] = {}
            cls._instance.execution_order: List[str] = []
        return cls._instance

    def register(self, name: str, func: Callable, priority: int = 0, depends_on: List[str] = None):
        """Plugin kaydet"""
        self.plugins[name] = {
            'func': func,
            'priority': priority,
            'depends_on': depends_on or [],
            'enabled': True
        }

        # Execution order'ı güncelle (priority'ye göre sırala)
        self.execution_order = sorted(
            self.plugins.keys(),
            key=lambda x: self.plugins[x]['priority'],
            reverse=True
        )

        print(f"[PLUGIN] Registered '{name}' with priority {priority}")

    def execute(self, plugin_name: str, *args, **kwargs):
        """Plugin çalıştır"""
        if plugin_name not in self.plugins:
            raise ValueError(f"Plugin not found: {plugin_name}")

        plugin = self.plugins[plugin_name]

        if not plugin['enabled']:
            print(f"[PLUGIN] Plugin '{plugin_name}' is disabled")
            return None

        # Dependency kontrolü
        for dep in plugin['depends_on']:
            if dep not in self.plugins or not self.plugins[dep]['enabled']:
                raise RuntimeError(f"Plugin '{plugin_name}' depends on '{dep}' which is not available")

        return plugin['func'](*args, **kwargs)

    def execute_all(self, *args, **kwargs) -> Dict[str, Any]:
        """Tüm plugin'leri sırayla çalıştır"""
        results = {}

        for plugin_name in self.execution_order:
            plugin = self.plugins[plugin_name]

            if plugin['enabled']:
                print(f"[PLUGIN] Executing '{plugin_name}'")
                try:
                    result = self.execute(plugin_name, *args, **kwargs)
                    results[plugin_name] = result
                except Exception as e:
                    print(f"[PLUGIN] Error in '{plugin_name}': {e}")
                    results[plugin_name] = None

        return results

    def enable(self, plugin_name: str):
        """Plugin'i etkinleştir"""
        if plugin_name in self.plugins:
            self.plugins[plugin_name]['enabled'] = True
            print(f"[PLUGIN] Enabled '{plugin_name}'")

    def disable(self, plugin_name: str):
        """Plugin'i devre dışı bırak"""
        if plugin_name in self.plugins:
            self.plugins[plugin_name]['enabled'] = False
            print(f"[PLUGIN] Disabled '{plugin_name}'")

# Global registry
plugin_registry = PluginRegistry()

def plugin(name: str, priority: int = 0, depends_on: List[str] = None):
    """
    Plugin decorator

    Args:
        name: Plugin adı
        priority: Execution priority (yüksek önce çalışır)
        depends_on: Bağımlı olunan plugin'ler
    """
    def decorator(func: Callable) -> Callable:
        plugin_registry.register(name, func, priority, depends_on)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return plugin_registry.execute(name, *args, **kwargs)

        return wrapper
    return decorator

# Test plugins
@plugin('validator', priority=100)
def validate_input(data: dict) -> bool:
    """Input validation plugin"""
    print(f"Validating input: {data}")
    return 'value' in data

@plugin('transformer', priority=50, depends_on=['validator'])
def transform_data(data: dict) -> dict:
    """Data transformation plugin"""
    print(f"Transforming data: {data}")
    return {**data, 'transformed': True}

@plugin('logger', priority=10)
def log_data(data: dict) -> None:
    """Logging plugin"""
    print(f"Logging data: {data}")

print("=== ALIŞTIRMA 17 TEST ===")
test_data = {'value': 123}
results = plugin_registry.execute_all(test_data)
print(f"\nResults: {results}\n")

# Plugin'i disable et
plugin_registry.disable('logger')
results = plugin_registry.execute_all(test_data)
print(f"\nResults after disabling logger: {results}\n")


# =============================================================================
# ALIŞTIRMA 18: Comprehensive Monitoring Decorator (Expert)
# =============================================================================
"""
SORU 18: Comprehensive Monitoring Decorator

Production-ready kapsamlı monitoring decorator:
- Performance metrics (timing, memory, CPU)
- Error tracking ve alerting
- Distributed tracing support
- Real-time dashboarding
- Metric aggregation

Örnek kullanım:
    @monitor(metrics=['time', 'memory', 'errors'], alert_threshold=1.0)
    def critical_operation():
        pass
"""

# TODO: monitor decorator'ını yazın


# ============= ÇÖZÜM 18 =============

import traceback
from collections import deque

class MonitoringSystem:
    """Comprehensive monitoring system"""
    def __init__(self):
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.alert_handlers: List[Callable] = []
        self.trace_id = 0

    def record_metric(self, func_name: str, metric_type: str, value: Any):
        """Metric kaydet"""
        self.metrics[f"{func_name}.{metric_type}"].append({
            'value': value,
            'timestamp': time.time()
        })

    def add_alert_handler(self, handler: Callable):
        """Alert handler ekle"""
        self.alert_handlers.append(handler)

    def trigger_alert(self, func_name: str, alert_type: str, message: str):
        """Alert tetikle"""
        for handler in self.alert_handlers:
            handler(func_name, alert_type, message)

    def get_aggregated_metrics(self, func_name: str) -> Dict:
        """Aggregated metrikler"""
        result = {}

        for key, values in self.metrics.items():
            if key.startswith(func_name):
                metric_type = key.split('.', 1)[1]
                numeric_values = [v['value'] for v in values if isinstance(v['value'], (int, float))]

                if numeric_values:
                    result[metric_type] = {
                        'count': len(numeric_values),
                        'mean': sum(numeric_values) / len(numeric_values),
                        'min': min(numeric_values),
                        'max': max(numeric_values)
                    }

        return result

    def generate_trace_id(self) -> str:
        """Unique trace ID oluştur"""
        self.trace_id += 1
        return f"trace-{self.trace_id:06d}"

# Global monitoring system
monitoring_system = MonitoringSystem()

# Alert handler
def console_alert_handler(func_name: str, alert_type: str, message: str):
    """Console alert handler"""
    print(f"[ALERT] {alert_type.upper()} in {func_name}: {message}")

monitoring_system.add_alert_handler(console_alert_handler)

def monitor(
    metrics: List[str] = None,
    alert_threshold: float = None,
    trace: bool = True
):
    """
    Comprehensive monitoring decorator

    Args:
        metrics: İzlenecek metrikler ['time', 'memory', 'errors', 'calls']
        alert_threshold: Alert eşiği (saniye)
        trace: Distributed tracing etkinleştir
    """
    if metrics is None:
        metrics = ['time', 'calls']

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            func_name = func.__name__
            trace_id = monitoring_system.generate_trace_id() if trace else None

            if trace:
                print(f"[TRACE] {trace_id} - Starting {func_name}")

            # Memory before
            memory_before = sys.getsizeof(args) + sys.getsizeof(kwargs) if 'memory' in metrics else 0

            # Timing start
            start_time = time.perf_counter() if 'time' in metrics else 0

            try:
                result = func(*args, **kwargs)

                # Success metrics
                if 'time' in metrics:
                    execution_time = time.perf_counter() - start_time
                    monitoring_system.record_metric(func_name, 'time', execution_time)

                    # Alert kontrolü
                    if alert_threshold and execution_time > alert_threshold:
                        monitoring_system.trigger_alert(
                            func_name,
                            'performance',
                            f"Execution time {execution_time:.4f}s exceeded threshold {alert_threshold}s"
                        )

                if 'memory' in metrics:
                    memory_after = sys.getsizeof(result)
                    memory_delta = memory_after - memory_before
                    monitoring_system.record_metric(func_name, 'memory', memory_delta)

                if 'calls' in metrics:
                    monitoring_system.record_metric(func_name, 'calls', 1)

                if trace:
                    print(f"[TRACE] {trace_id} - Completed {func_name}")

                return result

            except Exception as e:
                # Error metrics
                if 'errors' in metrics:
                    monitoring_system.record_metric(func_name, 'errors', 1)
                    monitoring_system.trigger_alert(
                        func_name,
                        'error',
                        f"{type(e).__name__}: {str(e)}"
                    )

                if trace:
                    print(f"[TRACE] {trace_id} - Failed {func_name}")

                raise

        def get_metrics():
            """Aggregated metrikler"""
            return monitoring_system.get_aggregated_metrics(func.__name__)

        wrapper.get_metrics = get_metrics
        wrapper.monitoring = monitoring_system

        return wrapper
    return decorator

# Test
@monitor(metrics=['time', 'memory', 'calls', 'errors'], alert_threshold=0.5, trace=True)
def data_processing(size: int, should_fail: bool = False):
    """Monitored data processing"""
    time.sleep(0.3)

    if should_fail:
        raise ValueError("Simulated error")

    return list(range(size))

print("=== ALIŞTIRMA 18 TEST ===")

# Başarılı çağrılar
for i in range(3):
    result = data_processing(100)
    print(f"Processed {len(result)} items\n")

# Başarısız çağrı
try:
    data_processing(100, should_fail=True)
except ValueError:
    pass

# Metrics raporu
print("\n=== MONITORING METRICS ===")
metrics = data_processing.get_metrics()
for metric_name, values in metrics.items():
    print(f"\n{metric_name}:")
    for key, value in values.items():
        print(f"  {key}: {value}")

print("\n" + "="*50)
print("TÜM ALIŞTIRMALAR TAMAMLANDI!")
print("="*50)
