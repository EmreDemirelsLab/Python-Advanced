# Decorators ve Closures - İleri Seviye

## İçindekiler
1. [Closures (Kapanışlar) Derinlemesine](#closures)
2. [Function Decorators](#function-decorators)
3. [Class Decorators](#class-decorators)
4. [Decorator Chaining](#decorator-chaining)
5. [Parametrized Decorators](#parametrized-decorators)
6. [functools Modülü](#functools-modulu)
7. [Decorator Patterns](#decorator-patterns)
8. [Production Use Cases](#production-use-cases)

---

## Closures (Kapanışlar) Derinlemesine

Closure, bir iç fonksiyonun dış fonksiyonun yerel değişkenlerine erişebildiği ve bu değişkenleri "hatırladığı" bir programlama konseptidir.

### Temel Closure Örneği

```python
def outer_function(message):
    """Dış fonksiyon - enclosing scope oluşturur"""

    def inner_function():
        """İç fonksiyon - closure oluşturur"""
        print(message)  # Dış fonksiyonun değişkenine erişim

    return inner_function

# Closure oluşturma
greeting = outer_function("Merhaba Dünya!")
greeting()  # Output: Merhaba Dünya!

# Her closure kendi state'ini tutar
hello = outer_function("Hello")
hi = outer_function("Hi")
hello()  # Output: Hello
hi()     # Output: Hi
```

### Closure ile State Yönetimi

```python
def counter(start=0):
    """Closure kullanarak sayaç implementasyonu"""
    count = [start]  # Mutable nesne kullanımı

    def increment(step=1):
        count[0] += step
        return count[0]

    def decrement(step=1):
        count[0] -= step
        return count[0]

    def get_value():
        return count[0]

    def reset():
        count[0] = start
        return count[0]

    # Multiple closures döndürme
    increment.decrement = decrement
    increment.get = get_value
    increment.reset = reset

    return increment

# Kullanım
c = counter(10)
print(c())              # 11
print(c(5))             # 16
print(c.decrement(3))   # 13
print(c.get())          # 13
print(c.reset())        # 10
```

### nonlocal Keyword ile Closure

```python
def make_averager():
    """nonlocal kullanarak ortalam hesaplama"""
    count = 0
    total = 0

    def averager(new_value):
        nonlocal count, total
        count += 1
        total += new_value
        return total / count

    return averager

avg = make_averager()
print(avg(10))  # 10.0
print(avg(11))  # 10.5
print(avg(12))  # 11.0
```

### Gelişmiş Closure Factory Pattern

```python
def make_multiplier_factory():
    """Closure factory pattern - farklı çarpanlar üreten factory"""
    multipliers = {}

    def create_multiplier(factor, name=None):
        """Belirli bir faktör için multiplier oluşturur"""
        key = name or f"x{factor}"

        if key in multipliers:
            return multipliers[key]

        def multiply(x):
            """Closure - factor değerini hatırlar"""
            return x * factor

        multiply.__name__ = key
        multiply.factor = factor
        multipliers[key] = multiply

        return multiply

    create_multiplier.get_all = lambda: multipliers.copy()
    create_multiplier.clear = lambda: multipliers.clear()

    return create_multiplier

# Kullanım
factory = make_multiplier_factory()
double = factory(2, "double")
triple = factory(3, "triple")

print(double(5))   # 10
print(triple(5))   # 15
print(factory.get_all().keys())  # dict_keys(['double', 'triple'])
```

---

## Function Decorators

Decorator, bir fonksiyonu alıp davranışını değiştiren veya genişleten bir fonksiyondur.

### Temel Decorator Yapısı

```python
import functools
from typing import Callable, Any

def simple_decorator(func: Callable) -> Callable:
    """En basit decorator yapısı"""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        print(f"Fonksiyon çağrılıyor: {func.__name__}")
        result = func(*args, **kwargs)
        print(f"Fonksiyon tamamlandı: {func.__name__}")
        return result

    return wrapper

@simple_decorator
def greet(name: str) -> str:
    return f"Hello, {name}!"

print(greet("Alice"))
# Output:
# Fonksiyon çağrılıyor: greet
# Fonksiyon tamamlandı: greet
# Hello, Alice!
```

### Timing Decorator - Production Ready

```python
import time
import functools
from typing import Callable, Any
from collections import defaultdict

class PerformanceTracker:
    """Fonksiyon performansını izleyen singleton class"""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.stats = defaultdict(lambda: {
                'calls': 0,
                'total_time': 0,
                'min_time': float('inf'),
                'max_time': 0,
                'avg_time': 0
            })
        return cls._instance

    def record(self, func_name: str, execution_time: float):
        """Performans kaydı ekle"""
        stats = self.stats[func_name]
        stats['calls'] += 1
        stats['total_time'] += execution_time
        stats['min_time'] = min(stats['min_time'], execution_time)
        stats['max_time'] = max(stats['max_time'], execution_time)
        stats['avg_time'] = stats['total_time'] / stats['calls']

    def get_report(self, func_name: str = None) -> dict:
        """Performans raporu al"""
        if func_name:
            return self.stats.get(func_name, {})
        return dict(self.stats)

def timing(verbose: bool = True):
    """Gelişmiş timing decorator with tracking"""
    def decorator(func: Callable) -> Callable:
        tracker = PerformanceTracker()

        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            start_time = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                end_time = time.perf_counter()
                execution_time = end_time - start_time
                tracker.record(func.__name__, execution_time)

                if verbose:
                    print(f"{func.__name__} executed in {execution_time:.4f} seconds")

        wrapper.get_stats = lambda: tracker.get_report(func.__name__)
        wrapper.tracker = tracker

        return wrapper
    return decorator

@timing(verbose=True)
def fibonacci(n: int) -> int:
    """Fibonacci hesaplama"""
    if n < 2:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

# Kullanım
result = fibonacci(10)
print(f"\nStats: {fibonacci.get_stats()}")
```

### Memoization Decorator

```python
import functools
from typing import Callable, Any, Hashable

def memoize(maxsize: int = 128, typed: bool = False):
    """Gelişmiş memoization decorator"""
    def decorator(func: Callable) -> Callable:
        cache = {}
        hits = misses = 0

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal hits, misses

            # Cache key oluşturma
            key = (args, tuple(sorted(kwargs.items())))

            try:
                # Cache'den dön
                result = cache[key]
                hits += 1
                return result
            except KeyError:
                # Hesapla ve cache'e ekle
                result = func(*args, **kwargs)
                misses += 1

                # Maxsize kontrolü
                if len(cache) >= maxsize:
                    # LRU mantığı - en eskiyi sil
                    cache.pop(next(iter(cache)))

                cache[key] = result
                return result

        def cache_info():
            """Cache istatistikleri"""
            return {
                'hits': hits,
                'misses': misses,
                'maxsize': maxsize,
                'currsize': len(cache),
                'hit_rate': hits / (hits + misses) if (hits + misses) > 0 else 0
            }

        wrapper.cache_info = cache_info
        wrapper.cache_clear = lambda: cache.clear()

        return wrapper
    return decorator

@memoize(maxsize=256)
def expensive_calculation(n: int) -> int:
    """Pahalı hesaplama simülasyonu"""
    time.sleep(0.1)  # Simüle edilmiş gecikme
    return n ** 2

# Test
print(expensive_calculation(10))  # Yavaş - cache miss
print(expensive_calculation(10))  # Hızlı - cache hit
print(expensive_calculation.cache_info())
```

### Retry Decorator - Production Ready

```python
import functools
import time
from typing import Callable, Type, Tuple
import random

def retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    on_retry: Callable = None
):
    """
    Gelişmiş retry decorator with exponential backoff

    Args:
        max_attempts: Maksimum deneme sayısı
        delay: İlk deneme arası bekleme süresi (saniye)
        backoff: Her denemede bekleme süresinin çarpanı
        exceptions: Yakalanacak exception'lar
        on_retry: Her retry'da çağrılacak callback
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            attempt = 0
            current_delay = delay

            while attempt < max_attempts:
                try:
                    attempt += 1
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt >= max_attempts:
                        raise

                    if on_retry:
                        on_retry(attempt, max_attempts, e, current_delay)
                    else:
                        print(f"Attempt {attempt}/{max_attempts} failed: {e}")
                        print(f"Retrying in {current_delay:.2f} seconds...")

                    time.sleep(current_delay)
                    current_delay *= backoff

        return wrapper
    return decorator

# Kullanım örneği
@retry(max_attempts=5, delay=0.5, backoff=2.0, exceptions=(ConnectionError, TimeoutError))
def unreliable_network_call(fail_rate: float = 0.7):
    """Başarısız olabilecek network çağrısı simülasyonu"""
    if random.random() < fail_rate:
        raise ConnectionError("Network error occurred")
    return "Success!"

# Test
try:
    result = unreliable_network_call(fail_rate=0.5)
    print(f"Result: {result}")
except ConnectionError as e:
    print(f"All retries failed: {e}")
```

---

## Class Decorators

Class decorator'lar, sınıfları modifiye etmek veya genişletmek için kullanılır.

### Singleton Pattern Decorator

```python
import functools
from typing import Any, Type

def singleton(cls: Type) -> Type:
    """Thread-safe singleton decorator"""
    instances = {}
    lock = __import__('threading').Lock()

    @functools.wraps(cls)
    def get_instance(*args, **kwargs):
        if cls not in instances:
            with lock:
                if cls not in instances:
                    instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    return get_instance

@singleton
class DatabaseConnection:
    """Singleton database connection"""

    def __init__(self, host: str = "localhost"):
        self.host = host
        self.connected = False
        print(f"Creating connection to {host}")

    def connect(self):
        self.connected = True
        print(f"Connected to {self.host}")

# Test
db1 = DatabaseConnection("localhost")
db2 = DatabaseConnection("localhost")
print(db1 is db2)  # True - aynı instance
```

### Class Method Logger Decorator

```python
import functools
from typing import Type
import logging

def log_all_methods(logger_name: str = None):
    """Sınıftaki tüm methodları logla"""
    def decorator(cls: Type) -> Type:
        logger = logging.getLogger(logger_name or cls.__name__)

        for attr_name in dir(cls):
            # Private ve magic method'ları atla
            if attr_name.startswith('_'):
                continue

            attr_value = getattr(cls, attr_name)

            # Callable olan attribute'ları decorate et
            if callable(attr_value):
                decorated = _log_method(attr_value, logger)
                setattr(cls, attr_name, decorated)

        return cls

    return decorator

def _log_method(method, logger):
    """Method wrapper with logging"""
    @functools.wraps(method)
    def wrapper(*args, **kwargs):
        logger.info(f"Calling {method.__name__} with args={args[1:]}, kwargs={kwargs}")
        try:
            result = method(*args, **kwargs)
            logger.info(f"{method.__name__} returned {result}")
            return result
        except Exception as e:
            logger.error(f"{method.__name__} raised {type(e).__name__}: {e}")
            raise
    return wrapper

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')

@log_all_methods()
class Calculator:
    """Logged calculator class"""

    def add(self, a: float, b: float) -> float:
        return a + b

    def divide(self, a: float, b: float) -> float:
        if b == 0:
            raise ValueError("Cannot divide by zero")
        return a / b

# Test
calc = Calculator()
print(calc.add(5, 3))
try:
    calc.divide(10, 0)
except ValueError:
    pass
```

### DataClass-like Decorator

```python
from typing import Type, Any, get_type_hints

def dataclass_like(frozen: bool = False, repr: bool = True):
    """Basit dataclass benzeri decorator"""
    def decorator(cls: Type) -> Type:
        # Type hints'i al
        hints = get_type_hints(cls)

        # __init__ method'u oluştur
        def __init__(self, **kwargs):
            for name, type_ in hints.items():
                if name in kwargs:
                    setattr(self, name, kwargs[name])
                else:
                    raise TypeError(f"Missing required argument: {name}")

            if frozen:
                object.__setattr__(self, '_frozen', True)

        # __repr__ method'u
        if repr:
            def __repr__(self):
                attrs = ', '.join(f"{k}={getattr(self, k)!r}" for k in hints)
                return f"{cls.__name__}({attrs})"
            cls.__repr__ = __repr__

        # __setattr__ for frozen
        if frozen:
            original_setattr = cls.__setattr__ if hasattr(cls, '__setattr__') else object.__setattr__

            def __setattr__(self, name, value):
                if hasattr(self, '_frozen') and self._frozen:
                    raise AttributeError(f"Cannot modify frozen instance")
                original_setattr(self, name, value)

            cls.__setattr__ = __setattr__

        # __eq__ method'u
        def __eq__(self, other):
            if not isinstance(other, cls):
                return False
            return all(getattr(self, k) == getattr(other, k) for k in hints)

        cls.__init__ = __init__
        cls.__eq__ = __eq__

        return cls

    return decorator

@dataclass_like(frozen=True, repr=True)
class Point:
    x: int
    y: int

# Test
p1 = Point(x=10, y=20)
p2 = Point(x=10, y=20)
print(p1)           # Point(x=10, y=20)
print(p1 == p2)     # True

try:
    p1.x = 30       # AttributeError: Cannot modify frozen instance
except AttributeError as e:
    print(e)
```

---

## Decorator Chaining

Birden fazla decorator'ı aynı fonksiyon üzerinde kullanma.

### Decorator Execution Order

```python
def decorator_one(func):
    print("Decorator One - Wrapping")
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        print("Decorator One - Before")
        result = func(*args, **kwargs)
        print("Decorator One - After")
        return result
    return wrapper

def decorator_two(func):
    print("Decorator Two - Wrapping")
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        print("Decorator Two - Before")
        result = func(*args, **kwargs)
        print("Decorator Two - After")
        return result
    return wrapper

@decorator_one
@decorator_two
def my_function():
    print("Original Function")

# Output during decoration:
# Decorator Two - Wrapping
# Decorator One - Wrapping

print("\n--- Function Call ---")
my_function()
# Output:
# Decorator One - Before
# Decorator Two - Before
# Original Function
# Decorator Two - After
# Decorator One - After
```

### Production Decorator Chain Example

```python
import functools
import time
from typing import Callable

def validate_inputs(*validators):
    """Input validation decorator"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for validator in validators:
                if not validator(*args, **kwargs):
                    raise ValueError(f"Validation failed for {func.__name__}")
            return func(*args, **kwargs)
        return wrapper
    return decorator

def cache_result(ttl: int = 60):
    """Simple TTL cache decorator"""
    def decorator(func: Callable) -> Callable:
        cache = {}

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            key = (args, tuple(sorted(kwargs.items())))

            if key in cache:
                result, timestamp = cache[key]
                if time.time() - timestamp < ttl:
                    print(f"Cache hit for {func.__name__}")
                    return result

            result = func(*args, **kwargs)
            cache[key] = (result, time.time())
            return result

        return wrapper
    return decorator

def log_execution(func: Callable) -> Callable:
    """Execution logging decorator"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        print(f"Executing {func.__name__} with args={args}, kwargs={kwargs}")
        result = func(*args, **kwargs)
        print(f"{func.__name__} returned {result}")
        return result
    return wrapper

# Validator functions
def positive_numbers(*args, **kwargs):
    """Tüm argümanların pozitif olduğunu kontrol et"""
    return all(isinstance(x, (int, float)) and x > 0 for x in args)

# Chained decorators - aşağıdan yukarıya uygulanır
@log_execution
@cache_result(ttl=30)
@validate_inputs(positive_numbers)
def calculate_compound_interest(principal: float, rate: float, time: float) -> float:
    """Bileşik faiz hesaplama"""
    return principal * (1 + rate) ** time

# Test
try:
    print(calculate_compound_interest(1000, 0.05, 10))  # İlk çağrı
    print(calculate_compound_interest(1000, 0.05, 10))  # Cache'den
    print(calculate_compound_interest(-1000, 0.05, 10)) # Validation hatası
except ValueError as e:
    print(f"Error: {e}")
```

---

## Parametrized Decorators

Parametre alan decorator'lar - decorator factory pattern.

### Access Control Decorator

```python
import functools
from typing import Callable, Set, Any
from enum import Enum

class Role(Enum):
    """Kullanıcı rolleri"""
    ADMIN = "admin"
    USER = "user"
    GUEST = "guest"

class User:
    """Kullanıcı modeli"""
    def __init__(self, username: str, role: Role):
        self.username = username
        self.role = role

# Global context - gerçek uygulamada session/request context kullanılır
current_user = User("john", Role.USER)

def require_role(*allowed_roles: Role):
    """Role-based access control decorator"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            if current_user.role not in allowed_roles:
                raise PermissionError(
                    f"User {current_user.username} with role {current_user.role.value} "
                    f"cannot access {func.__name__}. Required roles: "
                    f"{[r.value for r in allowed_roles]}"
                )
            return func(*args, **kwargs)
        return wrapper
    return decorator

@require_role(Role.ADMIN)
def delete_user(user_id: int):
    """Sadece admin silebilir"""
    return f"User {user_id} deleted"

@require_role(Role.ADMIN, Role.USER)
def view_profile(user_id: int):
    """Admin ve user görüntüleyebilir"""
    return f"Profile {user_id} data"

@require_role(Role.ADMIN, Role.USER, Role.GUEST)
def view_public_content():
    """Herkes görüntüleyebilir"""
    return "Public content"

# Test
print(view_public_content())    # OK
print(view_profile(123))        # OK

try:
    print(delete_user(123))     # PermissionError - user cannot delete
except PermissionError as e:
    print(f"Error: {e}")
```

### Rate Limiting Decorator

```python
import functools
import time
from typing import Callable, Dict
from collections import deque

def rate_limit(max_calls: int, time_window: int):
    """
    Rate limiting decorator

    Args:
        max_calls: Zaman penceresi içinde maksimum çağrı sayısı
        time_window: Zaman penceresi (saniye)
    """
    def decorator(func: Callable) -> Callable:
        # Her fonksiyon için ayrı call history
        calls: Dict[str, deque] = {}

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Call history anahtarı - gerçek uygulamada user_id kullanılır
            key = "global"

            if key not in calls:
                calls[key] = deque()

            now = time.time()
            call_times = calls[key]

            # Eski çağrıları temizle
            while call_times and call_times[0] < now - time_window:
                call_times.popleft()

            # Rate limit kontrolü
            if len(call_times) >= max_calls:
                oldest_call = call_times[0]
                wait_time = time_window - (now - oldest_call)
                raise RuntimeError(
                    f"Rate limit exceeded for {func.__name__}. "
                    f"Try again in {wait_time:.2f} seconds. "
                    f"Limit: {max_calls} calls per {time_window} seconds."
                )

            # Çağrıyı kaydet
            call_times.append(now)

            return func(*args, **kwargs)

        def reset_limit(key: str = "global"):
            """Rate limit sayaçlarını sıfırla"""
            if key in calls:
                calls[key].clear()

        wrapper.reset_limit = reset_limit

        return wrapper
    return decorator

@rate_limit(max_calls=3, time_window=10)
def api_call(endpoint: str):
    """Rate limited API çağrısı"""
    return f"Calling {endpoint}"

# Test
try:
    for i in range(5):
        print(f"Call {i + 1}: {api_call('/users')}")
        time.sleep(1)
except RuntimeError as e:
    print(f"Error: {e}")
```

### Type Checking Decorator

```python
import functools
from typing import Callable, get_type_hints, Any

def enforce_types(func: Callable) -> Callable:
    """Runtime type checking decorator using type hints"""

    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        # Type hints'i al
        hints = get_type_hints(func)

        # Positional arguments kontrolü
        arg_names = func.__code__.co_varnames[:func.__code__.co_argcount]

        for arg_name, arg_value in zip(arg_names, args):
            if arg_name in hints:
                expected_type = hints[arg_name]
                if not isinstance(arg_value, expected_type):
                    raise TypeError(
                        f"Argument '{arg_name}' must be {expected_type.__name__}, "
                        f"got {type(arg_value).__name__}"
                    )

        # Keyword arguments kontrolü
        for arg_name, arg_value in kwargs.items():
            if arg_name in hints:
                expected_type = hints[arg_name]
                if not isinstance(arg_value, expected_type):
                    raise TypeError(
                        f"Argument '{arg_name}' must be {expected_type.__name__}, "
                        f"got {type(arg_value).__name__}"
                    )

        # Fonksiyonu çalıştır
        result = func(*args, **kwargs)

        # Return type kontrolü
        if 'return' in hints:
            expected_return = hints['return']
            if not isinstance(result, expected_return):
                raise TypeError(
                    f"Return value must be {expected_return.__name__}, "
                    f"got {type(result).__name__}"
                )

        return result

    return wrapper

@enforce_types
def calculate_area(width: int, height: int) -> int:
    """Alan hesaplama - type check'li"""
    return width * height

@enforce_types
def greet(name: str, age: int) -> str:
    """Selamlama - type check'li"""
    return f"Hello {name}, you are {age} years old"

# Test
print(calculate_area(5, 10))        # OK: 50
print(greet("Alice", 30))           # OK

try:
    calculate_area(5.5, 10)         # TypeError: width must be int
except TypeError as e:
    print(f"Error: {e}")

try:
    greet("Bob", "30")              # TypeError: age must be int
except TypeError as e:
    print(f"Error: {e}")
```

---

## functools Modülü

Python'un functools modülü, higher-order fonksiyonlar için güçlü araçlar sağlar.

### functools.wraps

```python
import functools

def without_wraps(func):
    """wraps kullanmayan decorator"""
    def wrapper(*args, **kwargs):
        """Wrapper function"""
        return func(*args, **kwargs)
    return wrapper

def with_wraps(func):
    """wraps kullanan decorator"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        """Wrapper function"""
        return func(*args, **kwargs)
    return wrapper

@without_wraps
def function_one():
    """Original function one docstring"""
    pass

@with_wraps
def function_two():
    """Original function two docstring"""
    pass

# Metadata karşılaştırması
print(f"Without wraps - Name: {function_one.__name__}")      # wrapper
print(f"Without wraps - Doc: {function_one.__doc__}")        # Wrapper function

print(f"With wraps - Name: {function_two.__name__}")         # function_two
print(f"With wraps - Doc: {function_two.__doc__}")           # Original function two docstring
```

### functools.lru_cache

```python
import functools
import time

@functools.lru_cache(maxsize=128, typed=False)
def fibonacci_cached(n: int) -> int:
    """LRU cache ile optimize edilmiş fibonacci"""
    if n < 2:
        return n
    return fibonacci_cached(n - 1) + fibonacci_cached(n - 2)

def fibonacci_normal(n: int) -> int:
    """Normal fibonacci"""
    if n < 2:
        return n
    return fibonacci_normal(n - 1) + fibonacci_normal(n - 2)

# Performans karşılaştırması
print("Normal fibonacci:")
start = time.time()
result = fibonacci_normal(30)
print(f"Result: {result}, Time: {time.time() - start:.4f}s")

print("\nCached fibonacci:")
start = time.time()
result = fibonacci_cached(30)
print(f"Result: {result}, Time: {time.time() - start:.4f}s")

# Cache bilgisi
print(f"\nCache info: {fibonacci_cached.cache_info()}")
# CacheInfo(hits=28, misses=31, maxsize=128, currsize=31)
```

### functools.singledispatch

```python
from functools import singledispatch
from typing import Any

@singledispatch
def process_data(data: Any) -> str:
    """Generic data processor - fallback"""
    return f"Processing generic data: {data}"

@process_data.register(int)
def _(data: int) -> str:
    """Integer processor"""
    return f"Processing integer: {data * 2}"

@process_data.register(str)
def _(data: str) -> str:
    """String processor"""
    return f"Processing string: {data.upper()}"

@process_data.register(list)
def _(data: list) -> str:
    """List processor"""
    return f"Processing list with {len(data)} items: {sum(data) if all(isinstance(x, (int, float)) for x in data) else 'N/A'}"

@process_data.register(dict)
def _(data: dict) -> str:
    """Dict processor"""
    return f"Processing dict with keys: {list(data.keys())}"

# Test
print(process_data(42))                          # Processing integer: 84
print(process_data("hello"))                     # Processing string: HELLO
print(process_data([1, 2, 3, 4]))               # Processing list with 4 items: 10
print(process_data({"a": 1, "b": 2}))           # Processing dict with keys: ['a', 'b']
print(process_data(3.14))                        # Processing generic data: 3.14
```

### functools.partial

```python
from functools import partial

def power(base: float, exponent: float) -> float:
    """Üs alma fonksiyonu"""
    return base ** exponent

# Partial application
square = partial(power, exponent=2)
cube = partial(power, exponent=3)
sqrt = partial(power, exponent=0.5)

print(square(5))    # 25.0
print(cube(3))      # 27.0
print(sqrt(16))     # 4.0

# Pratik kullanım - logging
import logging

debug_log = partial(logging.log, logging.DEBUG)
info_log = partial(logging.log, logging.INFO)
error_log = partial(logging.log, logging.ERROR)

logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')

debug_log("Debug message")
info_log("Info message")
error_log("Error message")
```

---

## Decorator Patterns

Gerçek dünya senaryolarında kullanılan decorator pattern'leri.

### Context Manager Decorator

```python
import functools
from contextlib import contextmanager
from typing import Callable

def with_context_manager(context_manager_func):
    """Context manager kullanan decorator"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with context_manager_func():
                return func(*args, **kwargs)
        return wrapper
    return decorator

@contextmanager
def database_transaction():
    """Database transaction context manager"""
    print("BEGIN TRANSACTION")
    try:
        yield
        print("COMMIT TRANSACTION")
    except Exception as e:
        print(f"ROLLBACK TRANSACTION: {e}")
        raise

@with_context_manager(database_transaction)
def update_user(user_id: int, name: str):
    """Transaction içinde user güncelleme"""
    print(f"Updating user {user_id} with name {name}")
    if name == "error":
        raise ValueError("Invalid name")
    return f"User {user_id} updated"

# Test
print(update_user(1, "Alice"))
print()
try:
    update_user(2, "error")
except ValueError:
    pass
```

### Async Decorator Pattern

```python
import functools
import asyncio
from typing import Callable

def async_retry(max_attempts: int = 3):
    """Async retry decorator"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts - 1:
                        raise
                    print(f"Attempt {attempt + 1} failed: {e}")
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
        return wrapper
    return decorator

@async_retry(max_attempts=3)
async def fetch_data(url: str):
    """Async data fetching"""
    print(f"Fetching {url}")
    await asyncio.sleep(1)

    # Simüle edilmiş hata
    import random
    if random.random() < 0.5:
        raise ConnectionError("Network error")

    return f"Data from {url}"

# Test
async def main():
    try:
        result = await fetch_data("https://api.example.com")
        print(result)
    except ConnectionError as e:
        print(f"Failed after retries: {e}")

# asyncio.run(main())  # Uncomment to run
```

### Deprecation Warning Decorator

```python
import functools
import warnings
from typing import Callable

def deprecated(reason: str = "", version: str = "", alternative: str = ""):
    """Deprecation warning decorator"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            message = f"{func.__name__} is deprecated"

            if version:
                message += f" since version {version}"
            if reason:
                message += f". Reason: {reason}"
            if alternative:
                message += f". Use {alternative} instead"

            warnings.warn(message, category=DeprecationWarning, stacklevel=2)
            return func(*args, **kwargs)

        return wrapper
    return decorator

@deprecated(
    reason="Performance issues with large datasets",
    version="2.0.0",
    alternative="fast_sort()"
)
def old_sort(data: list) -> list:
    """Eski sıralama fonksiyonu"""
    return sorted(data)

def fast_sort(data: list) -> list:
    """Yeni hızlı sıralama fonksiyonu"""
    return sorted(data)

# Test
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    result = old_sort([3, 1, 2])

    if w:
        print(f"Warning: {w[0].message}")
```

---

## Production Use Cases

Gerçek production ortamlarında kullanılan decorator örnekleri.

### API Request Validator

```python
import functools
from typing import Callable, Dict, Any
import re

class ValidationError(Exception):
    """Validation hatası"""
    pass

def validate_api_request(schema: Dict[str, Dict[str, Any]]):
    """
    API request validation decorator

    schema format:
    {
        'param_name': {
            'type': type,
            'required': bool,
            'pattern': str (regex),
            'min': int/float,
            'max': int/float
        }
    }
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Validate kwargs against schema
            for param_name, rules in schema.items():
                value = kwargs.get(param_name)

                # Required check
                if rules.get('required', False) and value is None:
                    raise ValidationError(f"Required parameter '{param_name}' is missing")

                if value is not None:
                    # Type check
                    if 'type' in rules and not isinstance(value, rules['type']):
                        raise ValidationError(
                            f"Parameter '{param_name}' must be {rules['type'].__name__}"
                        )

                    # Pattern check (for strings)
                    if 'pattern' in rules and isinstance(value, str):
                        if not re.match(rules['pattern'], value):
                            raise ValidationError(
                                f"Parameter '{param_name}' does not match pattern {rules['pattern']}"
                            )

                    # Range check (for numbers)
                    if 'min' in rules and value < rules['min']:
                        raise ValidationError(
                            f"Parameter '{param_name}' must be >= {rules['min']}"
                        )

                    if 'max' in rules and value > rules['max']:
                        raise ValidationError(
                            f"Parameter '{param_name}' must be <= {rules['max']}"
                        )

            return func(*args, **kwargs)

        return wrapper
    return decorator

@validate_api_request({
    'username': {
        'type': str,
        'required': True,
        'pattern': r'^[a-zA-Z0-9_]{3,20}$'
    },
    'email': {
        'type': str,
        'required': True,
        'pattern': r'^[\w\.-]+@[\w\.-]+\.\w+$'
    },
    'age': {
        'type': int,
        'required': False,
        'min': 0,
        'max': 150
    }
})
def create_user(username: str, email: str, age: int = None):
    """Kullanıcı oluşturma API endpoint'i"""
    return {
        'id': 1,
        'username': username,
        'email': email,
        'age': age
    }

# Test
try:
    print(create_user(username="john_doe", email="john@example.com", age=30))
    print(create_user(username="ab", email="john@example.com"))  # ValidationError
except ValidationError as e:
    print(f"Validation Error: {e}")
```

### Performance Monitoring Decorator

```python
import functools
import time
from typing import Callable, Dict
import statistics

class PerformanceMonitor:
    """Global performance monitor"""

    def __init__(self):
        self.metrics: Dict[str, list] = {}

    def record(self, func_name: str, duration: float, memory_usage: float = 0):
        """Metric kaydet"""
        if func_name not in self.metrics:
            self.metrics[func_name] = []

        self.metrics[func_name].append({
            'duration': duration,
            'memory': memory_usage,
            'timestamp': time.time()
        })

    def get_stats(self, func_name: str) -> Dict:
        """İstatistikleri al"""
        if func_name not in self.metrics or not self.metrics[func_name]:
            return {}

        durations = [m['duration'] for m in self.metrics[func_name]]

        return {
            'count': len(durations),
            'mean': statistics.mean(durations),
            'median': statistics.median(durations),
            'min': min(durations),
            'max': max(durations),
            'stdev': statistics.stdev(durations) if len(durations) > 1 else 0
        }

    def print_report(self):
        """Performans raporu yazdır"""
        print("\n=== Performance Report ===")
        for func_name in sorted(self.metrics.keys()):
            stats = self.get_stats(func_name)
            print(f"\n{func_name}:")
            print(f"  Calls: {stats['count']}")
            print(f"  Mean: {stats['mean']*1000:.2f}ms")
            print(f"  Median: {stats['median']*1000:.2f}ms")
            print(f"  Min: {stats['min']*1000:.2f}ms")
            print(f"  Max: {stats['max']*1000:.2f}ms")
            print(f"  StdDev: {stats['stdev']*1000:.2f}ms")

# Global monitor instance
monitor = PerformanceMonitor()

def monitor_performance(func: Callable) -> Callable:
    """Performance monitoring decorator"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()

        try:
            result = func(*args, **kwargs)
            return result
        finally:
            duration = time.perf_counter() - start_time
            monitor.record(func.__name__, duration)

    return wrapper

@monitor_performance
def process_data(size: int):
    """Data processing simülasyonu"""
    data = list(range(size))
    return sum(x ** 2 for x in data)

@monitor_performance
def io_operation(delay: float):
    """IO operation simülasyonu"""
    time.sleep(delay)
    return "Done"

# Test
for _ in range(10):
    process_data(1000)
    io_operation(0.01)

monitor.print_report()
```

Bu doküman, Python'da decorator ve closure konularının ileri seviye kullanımını kapsamlı bir şekilde ele almaktadır. Her örnek production-ready kod standartlarında yazılmış ve gerçek dünya senaryolarını yansıtmaktadır.
