# Type Hints ve Static Type Checking

## İçindekiler
- [Temel Kavramlar](#temel-kavramlar)
- [Typing Module ve İleri Seviye Tipler](#typing-module-ve-ileri-seviye-tipler)
- [Generics ve Type Variables](#generics-ve-type-variables)
- [Protocol ve Structural Subtyping](#protocol-ve-structural-subtyping)
- [TypedDict ve Literal Types](#typeddict-ve-literal-types)
- [Callable Types ve Overload](#callable-types-ve-overload)
- [NewType ve Type Aliases](#newtype-ve-type-aliases)
- [Mypy ile Static Type Checking](#mypy-ile-static-type-checking)
- [Production Best Practices](#production-best-practices)

## Temel Kavramlar

Type hints, Python kodunuza statik tip bilgisi eklemenizi sağlar. Runtime'da doğrudan bir etkisi olmasa da, IDE'ler, linter'lar ve mypy gibi araçlar tarafından kullanılarak kod kalitesini artırır.

### Modern Python 3.10+ Type Annotations

```python
# Python 3.10+ ile built-in collection'lar doğrudan kullanılabilir
from typing import Optional, Union

# Eski yöntem (Python 3.9 ve öncesi)
# from typing import List, Dict, Tuple

# Yeni yöntem (Python 3.10+)
def process_items(items: list[str]) -> dict[str, int]:
    """Liste elemanlarını işleyip dictionary döndürür."""
    return {item: len(item) for item in items}

# Union operatörü (Python 3.10+)
def parse_value(value: int | str | float) -> str:
    """Birden fazla tip kabul eder - Union yerine | operatörü."""
    return str(value)

# Optional tip (None olabilir)
def find_user(user_id: int) -> Optional[dict[str, str]]:
    """Kullanıcı bulunamazsa None döner."""
    users = {1: {"name": "Ali", "email": "ali@example.com"}}
    return users.get(user_id)

# Modern Optional syntax (Python 3.10+)
def get_config(key: str) -> str | None:
    """Config değeri bulunamazsa None döner."""
    config = {"host": "localhost"}
    return config.get(key)
```

### Function Annotations ve Return Types

```python
from collections.abc import Callable, Iterator
from typing import Any, NoReturn

def validate_email(email: str) -> bool:
    """Email formatını doğrular."""
    return "@" in email and "." in email.split("@")[1]

def process_with_callback(
    data: list[int],
    callback: Callable[[int], str]
) -> list[str]:
    """Her eleman için callback fonksiyonunu çalıştırır."""
    return [callback(item) for item in data]

def raise_error(message: str) -> NoReturn:
    """Bu fonksiyon asla normal şekilde dönmez."""
    raise ValueError(message)

def generate_numbers(n: int) -> Iterator[int]:
    """Sayı generator'ı döndürür."""
    for i in range(n):
        yield i

def flexible_function(*args: int, **kwargs: str) -> dict[str, Any]:
    """Variable arguments ile tip tanımları."""
    return {"args_sum": sum(args), "kwargs": kwargs}
```

## Typing Module ve İleri Seviye Tipler

### Union, Literal ve Final

```python
from typing import Literal, Final, get_args, get_origin
from enum import Enum

# Literal: Sadece belirtilen değerleri kabul eder
HttpMethod = Literal["GET", "POST", "PUT", "DELETE"]

def make_request(url: str, method: HttpMethod) -> dict[str, Any]:
    """HTTP request yapar - method sadece belirtilen değerler olabilir."""
    return {"url": url, "method": method, "status": 200}

# Final: Değiştirilemez sabitler
MAX_CONNECTIONS: Final[int] = 100
API_VERSION: Final[str] = "v2"

class Config:
    """Final attribute'lar subclass'larda override edilemez."""
    BASE_URL: Final[str] = "https://api.example.com"

    def __init__(self) -> None:
        self.timeout: Final[int] = 30  # Instance level final

# Literal ile tip güvenli status kodları
Status = Literal["pending", "active", "completed", "failed"]

class Task:
    def __init__(self, name: str, status: Status = "pending") -> None:
        self.name = name
        self.status: Status = status

    def update_status(self, new_status: Status) -> None:
        """Status günceller - sadece geçerli değerler kabul edilir."""
        self.status = new_status

# Type introspection
def inspect_type() -> None:
    """Tip bilgilerini runtime'da inceleme."""
    print(get_args(HttpMethod))  # ('GET', 'POST', 'PUT', 'DELETE')
    print(get_origin(list[int]))  # <class 'list'>
```

### TypedDict ile Structured Dictionaries

```python
from typing import TypedDict, NotRequired, Required
from datetime import datetime

# TypedDict: Dictionary'lere yapı kazandırır
class UserDict(TypedDict):
    """Kullanıcı dictionary'si için tip tanımı."""
    id: int
    username: str
    email: str
    is_active: bool

class ExtendedUserDict(UserDict):
    """UserDict'i extend eder."""
    created_at: datetime
    roles: list[str]

# Python 3.11+ NotRequired ve Required
class PartialUser(TypedDict):
    """Bazı alanlar opsiyonel."""
    id: Required[int]  # Zorunlu
    username: Required[str]  # Zorunlu
    email: NotRequired[str]  # Opsiyonel
    phone: NotRequired[str]  # Opsiyonel

def create_user(user_data: UserDict) -> int:
    """TypedDict kullanarak tip güvenli dictionary işleme."""
    # IDE tam olarak hangi key'lerin olduğunu bilir
    print(f"Creating user: {user_data['username']}")
    return user_data["id"]

def update_user(user_id: int, updates: PartialUser) -> None:
    """Partial update için NotRequired alanlar."""
    print(f"Updating user {user_id} with {updates}")

# Kullanım
user: UserDict = {
    "id": 1,
    "username": "johndoe",
    "email": "john@example.com",
    "is_active": True
}

create_user(user)
```

### Protocol ile Structural Subtyping

```python
from typing import Protocol, runtime_checkable
from collections.abc import Sized

# Protocol: Duck typing'i statik olarak kontrol eder
class Drawable(Protocol):
    """Çizilebilir nesneler için protocol."""
    def draw(self) -> str:
        ...

    def get_color(self) -> str:
        ...

class SupportsClose(Protocol):
    """Kapatılabilir nesneler için protocol."""
    def close(self) -> None:
        ...

# Runtime kontrolü için
@runtime_checkable
class SupportsRead(Protocol):
    """Okunabilir nesneler için protocol."""
    def read(self, n: int = -1) -> str:
        ...

# Protocol kullanımı - explicit inheritance gerekmez
class Circle:
    def __init__(self, radius: float, color: str) -> None:
        self.radius = radius
        self.color = color

    def draw(self) -> str:
        return f"Drawing circle with radius {self.radius}"

    def get_color(self) -> str:
        return self.color

class Rectangle:
    def __init__(self, width: float, height: float, color: str) -> None:
        self.width = width
        self.height = height
        self.color = color

    def draw(self) -> str:
        return f"Drawing rectangle {self.width}x{self.height}"

    def get_color(self) -> str:
        return self.color

def render_shape(shape: Drawable) -> None:
    """Protocol sayesinde Circle ve Rectangle kabul edilir."""
    print(f"{shape.draw()} in {shape.get_color()}")

# Generic protocol
class Comparable(Protocol):
    """Karşılaştırılabilir nesneler için protocol."""
    def __lt__(self, other: Any) -> bool:
        ...

    def __gt__(self, other: Any) -> bool:
        ...

def find_max(items: list[Comparable]) -> Comparable:
    """Comparable protocol'ü implement eden her tip ile çalışır."""
    return max(items)
```

## Generics ve Type Variables

### TypeVar ile Generic Functions

```python
from typing import TypeVar, Generic, Sequence, Mapping
from collections.abc import Iterable

# TypeVar: Generic tip parametreleri
T = TypeVar('T')  # Herhangi bir tip
S = TypeVar('S')
K = TypeVar('K')
V = TypeVar('V')

# Constrained TypeVar: Sadece belirtilen tipler
NumericT = TypeVar('NumericT', int, float, complex)

# Bound TypeVar: Belirtilen tipten türemiş
class Comparable(Protocol):
    def __lt__(self, other: Any) -> bool: ...

ComparableT = TypeVar('ComparableT', bound=Comparable)

def first_element(items: Sequence[T]) -> T | None:
    """Sequence'in ilk elemanını döndürür - generic."""
    return items[0] if items else None

def reverse_dict(d: dict[K, V]) -> dict[V, K]:
    """Dictionary'nin key-value'larını tersine çevirir."""
    return {v: k for k, v in d.items()}

def add_numbers(a: NumericT, b: NumericT) -> NumericT:
    """Sadece numeric tiplerle çalışır."""
    return a + b  # type: ignore

def find_min(items: Sequence[ComparableT]) -> ComparableT:
    """Comparable protocol'ü implement eden tiplerle çalışır."""
    return min(items)

# Multiple TypeVar kullanımı
def zip_dicts(dict1: dict[K, V], dict2: dict[K, S]) -> dict[K, tuple[V, S]]:
    """İki dictionary'yi birleştirir."""
    return {k: (dict1[k], dict2[k]) for k in dict1.keys() & dict2.keys()}
```

### Generic Classes

```python
from typing import Generic, TypeVar, Protocol
from collections.abc import Iterator

T = TypeVar('T')
K = TypeVar('K')
V = TypeVar('V')

class Stack(Generic[T]):
    """Generic stack implementasyonu."""

    def __init__(self) -> None:
        self._items: list[T] = []

    def push(self, item: T) -> None:
        """Stack'e eleman ekler."""
        self._items.append(item)

    def pop(self) -> T:
        """Stack'ten eleman çıkarır."""
        if not self._items:
            raise IndexError("Stack is empty")
        return self._items.pop()

    def peek(self) -> T | None:
        """Stack'in tepesindeki elemanı gösterir."""
        return self._items[-1] if self._items else None

    def is_empty(self) -> bool:
        """Stack boş mu kontrol eder."""
        return len(self._items) == 0

class Cache(Generic[K, V]):
    """Generic cache implementasyonu."""

    def __init__(self, max_size: int = 100) -> None:
        self._data: dict[K, V] = {}
        self._max_size = max_size

    def get(self, key: K) -> V | None:
        """Cache'ten değer alır."""
        return self._data.get(key)

    def set(self, key: K, value: V) -> None:
        """Cache'e değer ekler."""
        if len(self._data) >= self._max_size:
            # İlk elemanı sil (basit LRU benzeri)
            first_key = next(iter(self._data))
            del self._data[first_key]
        self._data[key] = value

    def clear(self) -> None:
        """Cache'i temizler."""
        self._data.clear()

# Generic class kullanımı
int_stack: Stack[int] = Stack()
int_stack.push(1)
int_stack.push(2)

str_stack: Stack[str] = Stack()
str_stack.push("hello")

user_cache: Cache[int, UserDict] = Cache(max_size=1000)
```

### Advanced Generic Patterns

```python
from typing import TypeVar, Generic, Callable, ParamSpec, Concatenate
from collections.abc import Awaitable

T = TypeVar('T')
P = ParamSpec('P')  # Function parameter specification
R = TypeVar('R')

class Repository(Generic[T]):
    """Generic repository pattern."""

    def __init__(self, item_type: type[T]) -> None:
        self._item_type = item_type
        self._items: dict[int, T] = {}
        self._next_id = 1

    def add(self, item: T) -> int:
        """Yeni item ekler ve ID döner."""
        item_id = self._next_id
        self._items[item_id] = item
        self._next_id += 1
        return item_id

    def get(self, item_id: int) -> T | None:
        """ID'ye göre item getirir."""
        return self._items.get(item_id)

    def get_all(self) -> list[T]:
        """Tüm item'ları listeler."""
        return list(self._items.values())

    def filter(self, predicate: Callable[[T], bool]) -> list[T]:
        """Predicate'e uyan item'ları filtreler."""
        return [item for item in self._items.values() if predicate(item)]

# Decorator için ParamSpec kullanımı
def log_function(func: Callable[P, R]) -> Callable[P, R]:
    """Function çağrılarını loglar - tip korunur."""
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        print(f"Calling {func.__name__}")
        result = func(*args, **kwargs)
        print(f"Finished {func.__name__}")
        return result
    return wrapper

@log_function
def calculate(x: int, y: int) -> int:
    return x + y

# Async generic patterns
class AsyncRepository(Generic[T]):
    """Async generic repository pattern."""

    async def get(self, item_id: int) -> T | None:
        """Async item getirme."""
        # Simulated async operation
        return None

    async def save(self, item: T) -> None:
        """Async item kaydetme."""
        pass
```

## Callable Types ve Overload

### Callable Type Hints

```python
from typing import Callable, TypeAlias, Protocol
from collections.abc import Awaitable

# Callable type aliases
IntBinaryOp: TypeAlias = Callable[[int, int], int]
StringProcessor: TypeAlias = Callable[[str], str]
ValidationFunc: TypeAlias = Callable[[Any], bool]

def apply_operation(x: int, y: int, operation: IntBinaryOp) -> int:
    """Binary operation uygular."""
    return operation(x, y)

def add(a: int, b: int) -> int:
    return a + b

def multiply(a: int, b: int) -> int:
    return a * b

# Kullanım
result1 = apply_operation(5, 3, add)
result2 = apply_operation(5, 3, multiply)

# Higher-order functions
def compose(
    f: Callable[[T], S],
    g: Callable[[S], R]
) -> Callable[[T], R]:
    """İki fonksiyonu compose eder: g(f(x))."""
    def composed(x: T) -> R:
        return g(f(x))
    return composed

# Async callable
AsyncFunc: TypeAlias = Callable[[T], Awaitable[R]]

async def process_async(
    data: list[T],
    processor: Callable[[T], Awaitable[R]]
) -> list[R]:
    """Async processing pipeline."""
    results: list[R] = []
    for item in data:
        result = await processor(item)
        results.append(result)
    return results

# Protocol ile callable class
class Processor(Protocol[T, R]):
    """Callable class protocol."""
    def __call__(self, value: T) -> R:
        ...

class StringUpper:
    """Callable class örneği."""
    def __call__(self, s: str) -> str:
        return s.upper()

def process_strings(
    strings: list[str],
    processor: Processor[str, str]
) -> list[str]:
    """Processor kullanarak string'leri işler."""
    return [processor(s) for s in strings]
```

### Function Overloading

```python
from typing import overload, Union, Literal

# Overload: Farklı parametre kombinasyonları için farklı return tipleri
@overload
def process(value: int) -> str: ...

@overload
def process(value: str) -> int: ...

@overload
def process(value: list[int]) -> list[str]: ...

def process(value: int | str | list[int]) -> str | int | list[str]:
    """Farklı tiplerle farklı şekilde çalışır."""
    if isinstance(value, int):
        return str(value)
    elif isinstance(value, str):
        return len(value)
    else:
        return [str(v) for v in value]

# Literal ile overload
@overload
def fetch_data(source: Literal["db"]) -> dict[str, Any]: ...

@overload
def fetch_data(source: Literal["api"]) -> list[dict[str, Any]]: ...

@overload
def fetch_data(source: Literal["cache"]) -> str: ...

def fetch_data(source: Literal["db", "api", "cache"]) -> dict[str, Any] | list[dict[str, Any]] | str:
    """Kaynağa göre farklı tip döner."""
    if source == "db":
        return {"id": 1, "name": "Test"}
    elif source == "api":
        return [{"id": 1}, {"id": 2}]
    else:
        return "cached_data"

# Generic overload
@overload
def get_items(ids: list[int], return_dict: Literal[True]) -> dict[int, str]: ...

@overload
def get_items(ids: list[int], return_dict: Literal[False]) -> list[str]: ...

def get_items(
    ids: list[int],
    return_dict: bool = False
) -> dict[int, str] | list[str]:
    """Return type parametreye göre değişir."""
    items = {id_: f"Item {id_}" for id_ in ids}
    return items if return_dict else list(items.values())
```

## NewType ve Type Aliases

### NewType ile Distinct Types

```python
from typing import NewType, TypeAlias

# NewType: Runtime'da overhead yok, sadece static type checking için
UserId = NewType('UserId', int)
ProductId = NewType('ProductId', int)
EmailAddress = NewType('EmailAddress', str)

def get_user(user_id: UserId) -> dict[str, Any]:
    """UserId tipinde parametre bekler."""
    return {"id": user_id, "name": "John"}

def get_product(product_id: ProductId) -> dict[str, Any]:
    """ProductId tipinde parametre bekler."""
    return {"id": product_id, "title": "Product"}

# Kullanım - explicit conversion gerekli
user_id = UserId(123)
product_id = ProductId(456)

get_user(user_id)  # OK
# get_user(product_id)  # Type error! Farklı NewType'lar
# get_user(123)  # Type error! int değil UserId gerekli

# Email validation ile güvenli tip
def validate_email(email: str) -> EmailAddress:
    """Email doğrular ve EmailAddress tipi döner."""
    if "@" not in email:
        raise ValueError("Invalid email")
    return EmailAddress(email)

def send_email(to: EmailAddress, subject: str, body: str) -> None:
    """EmailAddress tipi garanti eder ki email geçerli."""
    print(f"Sending to {to}: {subject}")

# Type Aliases vs NewType
# Type Alias: Sadece isim, tip kontrolü yok
UserId2: TypeAlias = int  # Alias, int ile aynı
ProductId2: TypeAlias = int  # Alias, int ile aynı

def func1(x: UserId2) -> None: ...
def func2(x: ProductId2) -> None: ...

value: int = 100
func1(value)  # OK - Type alias
func2(value)  # OK - Type alias
```

### Complex Type Aliases

```python
from typing import TypeAlias, TypeVar, Generic
from collections.abc import Callable, Awaitable

# Complex type aliases
JSON: TypeAlias = dict[str, 'JSON'] | list['JSON'] | str | int | float | bool | None
Handler: TypeAlias = Callable[[dict[str, Any]], dict[str, Any]]
AsyncHandler: TypeAlias = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]

# Nested type aliases
NestedDict: TypeAlias = dict[str, dict[str, list[int]]]
Matrix: TypeAlias = list[list[float]]

def parse_json(data: str) -> JSON:
    """JSON string'i parse eder."""
    import json
    return json.loads(data)

def process_matrix(matrix: Matrix) -> float:
    """Matrix işler."""
    return sum(sum(row) for row in matrix)

# Generic type alias
T = TypeVar('T')
Result: TypeAlias = tuple[T | None, str | None]  # (value, error)

def safe_divide(a: float, b: float) -> Result[float]:
    """Güvenli bölme işlemi."""
    if b == 0:
        return (None, "Division by zero")
    return (a / b, None)

# Function type aliases
Validator: TypeAlias = Callable[[T], bool]
Transformer: TypeAlias = Callable[[T], R]
Reducer: TypeAlias = Callable[[R, T], R]

class Pipeline(Generic[T, R]):
    """Generic pipeline with type aliases."""

    def __init__(
        self,
        validator: Validator[T],
        transformer: Transformer[T, R]
    ) -> None:
        self.validator = validator
        self.transformer = transformer

    def process(self, items: list[T]) -> list[R]:
        """Pipeline process."""
        return [
            self.transformer(item)
            for item in items
            if self.validator(item)
        ]
```

## Mypy ile Static Type Checking

### Mypy Configuration

```python
# mypy.ini veya pyproject.toml konfigürasyonu
"""
[mypy]
python_version = 3.10
warn_return_any = True
warn_unused_configs = True
disallow_untyped_defs = True
disallow_any_unimported = True
no_implicit_optional = True
warn_redundant_casts = True
warn_unused_ignores = True
warn_no_return = True
check_untyped_defs = True
strict_equality = True

[mypy-external_lib.*]
ignore_missing_imports = True
"""

# Type checking direktifleri
from typing import Any, cast

def process_data(data: Any) -> dict[str, int]:
    """Any kullanımı - dikkatli olunmalı."""
    # type: ignore - mypy'ı belirli satırda devre dışı bırakır
    result = data.some_method()  # type: ignore[attr-defined]
    return result

def unsafe_operation(data: object) -> str:
    """Cast ile runtime type conversion."""
    # cast sadece type checker için, runtime'da hiçbir etkisi yok
    return cast(str, data)

# reveal_type - mypy'nin tip çıkarımını gösterir
def debug_types() -> None:
    """Tip debugging için reveal_type kullanımı."""
    x = [1, 2, 3]
    # reveal_type(x)  # mypy: Revealed type is "builtins.list[builtins.int]"

    y = {"key": "value"}
    # reveal_type(y)  # mypy: Revealed type is "builtins.dict[builtins.str, builtins.str]"

# assert_type - Python 3.11+
from typing import assert_type

def verify_types() -> None:
    """Tip doğrulama için assert_type."""
    value: int | str = get_value()
    if isinstance(value, int):
        assert_type(value, int)  # Type checker'a int olduğunu garanti eder
```

### Type Narrowing ve Type Guards

```python
from typing import TypeGuard, TypeIs, Union
from collections.abc import Iterable

# Type narrowing with isinstance
def process_value(value: int | str | list[int]) -> str:
    """Type narrowing ile tip güvenliği."""
    if isinstance(value, int):
        # Bu blokta value kesinlikle int
        return f"Number: {value * 2}"
    elif isinstance(value, str):
        # Bu blokta value kesinlikle str
        return f"String: {value.upper()}"
    else:
        # Bu blokta value kesinlikle list[int]
        return f"List sum: {sum(value)}"

# Custom type guard
def is_string_list(value: list[Any]) -> TypeGuard[list[str]]:
    """Type guard - liste string'lerden mi kontrol eder."""
    return all(isinstance(item, str) for item in value)

def process_list(items: list[Any]) -> None:
    """Type guard kullanımı."""
    if is_string_list(items):
        # Bu blokta items kesinlikle list[str]
        result = ", ".join(items)  # Type safe
        print(result)

# TypeIs (Python 3.13+) - daha güçlü type narrowing
def is_non_empty_string(value: str | None) -> TypeIs[str]:
    """TypeIs ile None check."""
    return value is not None and len(value) > 0

def use_string(value: str | None) -> None:
    """TypeIs kullanımı."""
    if is_non_empty_string(value):
        # value kesinlikle str (None değil)
        print(value.upper())

# Discriminated unions
class Success(TypedDict):
    status: Literal["success"]
    data: dict[str, Any]

class Error(TypedDict):
    status: Literal["error"]
    message: str

Response = Success | Error

def handle_response(response: Response) -> None:
    """Discriminated union ile type narrowing."""
    if response["status"] == "success":
        # Type checker bilir ki bu Success
        print(response["data"])
    else:
        # Type checker bilir ki bu Error
        print(response["message"])
```

## Production Best Practices

### API Type Safety

```python
from typing import TypedDict, Protocol, Literal, NotRequired
from datetime import datetime
from enum import Enum

# API Request/Response types
class CreateUserRequest(TypedDict):
    """User creation request."""
    username: str
    email: str
    password: str
    role: Literal["admin", "user", "guest"]

class UpdateUserRequest(TypedDict):
    """User update request - all fields optional."""
    username: NotRequired[str]
    email: NotRequired[str]
    role: NotRequired[Literal["admin", "user", "guest"]]

class UserResponse(TypedDict):
    """User API response."""
    id: int
    username: str
    email: str
    role: str
    created_at: str
    is_active: bool

class APIError(TypedDict):
    """API error response."""
    error: str
    message: str
    status_code: int

# API result type
APIResult = UserResponse | APIError

class UserService:
    """Type-safe user service."""

    def create_user(self, request: CreateUserRequest) -> APIResult:
        """User oluşturur."""
        # Validation
        if len(request["password"]) < 8:
            return {
                "error": "ValidationError",
                "message": "Password too short",
                "status_code": 400
            }

        # Success response
        return {
            "id": 1,
            "username": request["username"],
            "email": request["email"],
            "role": request["role"],
            "created_at": datetime.now().isoformat(),
            "is_active": True
        }

    def update_user(
        self,
        user_id: int,
        request: UpdateUserRequest
    ) -> APIResult:
        """User günceller."""
        return {
            "id": user_id,
            "username": request.get("username", "default"),
            "email": request.get("email", "default@example.com"),
            "role": request.get("role", "user"),
            "created_at": datetime.now().isoformat(),
            "is_active": True
        }
```

### Type-Safe Builder Pattern

```python
from typing import Generic, TypeVar, Self
from dataclasses import dataclass

T = TypeVar('T')

@dataclass
class QueryBuilder:
    """Type-safe SQL query builder."""
    _table: str | None = None
    _columns: list[str] | None = None
    _where: list[str] | None = None
    _limit: int | None = None

    def table(self, name: str) -> Self:
        """Table seçer."""
        self._table = name
        return self

    def select(self, *columns: str) -> Self:
        """Column'ları seçer."""
        self._columns = list(columns)
        return self

    def where(self, condition: str) -> Self:
        """WHERE condition ekler."""
        if self._where is None:
            self._where = []
        self._where.append(condition)
        return self

    def limit(self, n: int) -> Self:
        """LIMIT ekler."""
        self._limit = n
        return self

    def build(self) -> str:
        """Query string oluşturur."""
        if not self._table:
            raise ValueError("Table not specified")

        columns = ", ".join(self._columns) if self._columns else "*"
        query = f"SELECT {columns} FROM {self._table}"

        if self._where:
            query += f" WHERE {' AND '.join(self._where)}"

        if self._limit:
            query += f" LIMIT {self._limit}"

        return query

# Kullanım - method chaining ile type-safe
query = (QueryBuilder()
    .table("users")
    .select("id", "username", "email")
    .where("is_active = true")
    .where("age > 18")
    .limit(10)
    .build())
```

### Dependency Injection with Types

```python
from typing import Protocol, Generic, TypeVar
from abc import abstractmethod

T = TypeVar('T')

# Protocol-based dependency injection
class Logger(Protocol):
    """Logger protocol."""
    def log(self, message: str) -> None: ...
    def error(self, message: str) -> None: ...

class DatabaseConnection(Protocol):
    """Database connection protocol."""
    def execute(self, query: str) -> list[dict[str, Any]]: ...
    def commit(self) -> None: ...

class UserRepository:
    """Repository with injected dependencies."""

    def __init__(
        self,
        db: DatabaseConnection,
        logger: Logger
    ) -> None:
        self._db = db
        self._logger = logger

    def find_by_id(self, user_id: int) -> UserDict | None:
        """User bulur."""
        self._logger.log(f"Finding user {user_id}")
        results = self._db.execute(f"SELECT * FROM users WHERE id = {user_id}")

        if not results:
            return None

        return results[0]  # type: ignore

    def save(self, user: UserDict) -> None:
        """User kaydeder."""
        self._logger.log(f"Saving user {user['username']}")
        self._db.execute(f"INSERT INTO users ...")
        self._db.commit()

# Generic service container
class ServiceContainer(Generic[T]):
    """Generic DI container."""

    def __init__(self) -> None:
        self._services: dict[type[T], T] = {}

    def register(self, service_type: type[T], instance: T) -> None:
        """Service kaydeder."""
        self._services[service_type] = instance

    def resolve(self, service_type: type[T]) -> T:
        """Service resolve eder."""
        if service_type not in self._services:
            raise ValueError(f"Service {service_type} not registered")
        return self._services[service_type]
```

### Type-Safe Event System

```python
from typing import Protocol, Generic, TypeVar, Callable
from dataclasses import dataclass
from datetime import datetime

# Event types
@dataclass
class UserCreatedEvent:
    """User creation event."""
    user_id: int
    username: str
    email: str
    timestamp: datetime

@dataclass
class UserUpdatedEvent:
    """User update event."""
    user_id: int
    changes: dict[str, Any]
    timestamp: datetime

@dataclass
class OrderPlacedEvent:
    """Order placed event."""
    order_id: int
    user_id: int
    total: float
    timestamp: datetime

# Generic event handler
EventT = TypeVar('EventT')

class EventHandler(Protocol[EventT]):
    """Event handler protocol."""
    def handle(self, event: EventT) -> None: ...

class EventBus:
    """Type-safe event bus."""

    def __init__(self) -> None:
        self._handlers: dict[type, list[Callable[[Any], None]]] = {}

    def subscribe(
        self,
        event_type: type[EventT],
        handler: Callable[[EventT], None]
    ) -> None:
        """Event handler subscribe eder."""
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        self._handlers[event_type].append(handler)

    def publish(self, event: EventT) -> None:
        """Event publish eder."""
        event_type = type(event)
        if event_type in self._handlers:
            for handler in self._handlers[event_type]:
                handler(event)

# Kullanım
def on_user_created(event: UserCreatedEvent) -> None:
    """User created event handler."""
    print(f"User created: {event.username}")

def on_order_placed(event: OrderPlacedEvent) -> None:
    """Order placed event handler."""
    print(f"Order placed: {event.order_id} for ${event.total}")

event_bus = EventBus()
event_bus.subscribe(UserCreatedEvent, on_user_created)
event_bus.subscribe(OrderPlacedEvent, on_order_placed)

# Type-safe event publishing
event_bus.publish(UserCreatedEvent(
    user_id=1,
    username="johndoe",
    email="john@example.com",
    timestamp=datetime.now()
))
```

## Özet

Type hints, modern Python geliştirmede kod kalitesi ve maintainability için kritik öneme sahiptir:

### Temel Prensipler
1. **Progressive Typing**: Kademeli olarak tip ekleyin, tüm kodu bir anda tip'lemeye çalışmayın
2. **Protocol Over Inheritance**: Yapısal subtyping kullanın, sıkı inheritance hiyerarşilerinden kaçının
3. **Generic Types**: Yeniden kullanılabilir, tip-güvenli kod için generic'leri kullanın
4. **Type Narrowing**: isinstance ve type guards ile runtime tip kontrolü yapın
5. **Mypy Integration**: CI/CD pipeline'ına mypy ekleyin

### Production Checklist
- [ ] Tüm public API'ler tip tanımlarına sahip
- [ ] Generic types uygun yerlerde kullanılıyor
- [ ] Protocol'ler duck typing için tercih ediliyor
- [ ] TypedDict API contract'ları için kullanılıyor
- [ ] Mypy strict mode'da çalışıyor
- [ ] Type stubs (*.pyi) third-party kütüphaneler için mevcut
- [ ] NewType domain-specific types için kullanılıyor

Type hints sadece documentation değil, kod kalitesini artıran ve bug'ları önleyen güçlü bir araçtır!
