"""
TYPE HINTS VE STATIC TYPE CHECKING - İLERİ SEVİYE ALIŞTIRMALAR

Bu dosya type hints, generics, protocols ve mypy kullanımı için
advanced seviye pratik örnekler içerir.

Her alıştırma:
- Gerçek dünya senaryoları (API typing, generic collections, protocols)
- Type-safe design patterns
- Production-ready kod örnekleri

Zorluk seviyeleri:
- MEDIUM: Generic types, TypedDict, basic protocols
- HARD: Complex generics, custom protocols, overloading
- EXPERT: Advanced patterns, type narrowing, DI with types
"""

from typing import (
    TypeVar, Generic, Protocol, TypedDict, NotRequired, Literal,
    TypeAlias, Callable, ParamSpec, Concatenate, overload, Final,
    Any, cast, TypeGuard, Self
)
from collections.abc import Iterator, Iterable, Awaitable
from dataclasses import dataclass, field
from datetime import datetime
from abc import abstractmethod


# ==================== ALIŞTIRMA 1: Generic Repository Pattern ====================
# Zorluk: MEDIUM
# Konu: Generic class, TypeVar, Protocol

"""
TODO: Generic bir Repository pattern implementasyonu yapın.

Gereksinimler:
1. Repository[T] generic class oluşturun
2. CRUD operasyonları ekleyin (create, read, update, delete)
3. Filter ve query metodları ekleyin
4. Type-safe operations sağlayın
"""

# ÇÖZÜM:
T = TypeVar('T')
K = TypeVar('K')

class Identifiable(Protocol):
    """ID'ye sahip nesneler için protocol."""
    @property
    def id(self) -> int:
        """Unique identifier."""
        ...

class Repository(Generic[T]):
    """Generic repository pattern implementation."""

    def __init__(self) -> None:
        """Repository'yi initialize eder."""
        self._items: dict[int, T] = {}
        self._next_id: int = 1

    def create(self, item: T) -> int:
        """
        Yeni item oluşturur ve ID döner.

        Args:
            item: Oluşturulacak item

        Returns:
            Oluşturulan item'ın ID'si
        """
        item_id = self._next_id
        self._items[item_id] = item
        self._next_id += 1
        return item_id

    def read(self, item_id: int) -> T | None:
        """
        ID'ye göre item getirir.

        Args:
            item_id: Item ID

        Returns:
            Item veya None
        """
        return self._items.get(item_id)

    def update(self, item_id: int, item: T) -> bool:
        """
        Item günceller.

        Args:
            item_id: Item ID
            item: Yeni item değeri

        Returns:
            Başarılı ise True
        """
        if item_id not in self._items:
            return False
        self._items[item_id] = item
        return True

    def delete(self, item_id: int) -> bool:
        """
        Item siler.

        Args:
            item_id: Item ID

        Returns:
            Başarılı ise True
        """
        if item_id not in self._items:
            return False
        del self._items[item_id]
        return True

    def find_all(self) -> list[T]:
        """Tüm item'ları döner."""
        return list(self._items.values())

    def filter(self, predicate: Callable[[T], bool]) -> list[T]:
        """
        Predicate'e uyan item'ları filtreler.

        Args:
            predicate: Filter fonksiyonu

        Returns:
            Filtrelenmiş item listesi
        """
        return [item for item in self._items.values() if predicate(item)]

    def count(self) -> int:
        """Toplam item sayısı."""
        return len(self._items)

# Test
@dataclass
class User:
    """User entity."""
    id: int
    username: str
    email: str

def test_repository() -> None:
    """Repository test."""
    repo: Repository[User] = Repository()

    # Create
    user_id = repo.create(User(0, "john", "john@example.com"))
    repo.create(User(0, "jane", "jane@example.com"))

    # Read
    user = repo.read(user_id)
    assert user is not None
    assert user.username == "john"

    # Filter
    johns = repo.filter(lambda u: u.username.startswith("j"))
    assert len(johns) == 2


# ==================== ALIŞTIRMA 2: TypedDict API Responses ====================
# Zorluk: MEDIUM
# Konu: TypedDict, NotRequired, Literal, Union types

"""
TODO: RESTful API için type-safe request/response definitions oluşturun.

Gereksinimler:
1. User API için TypedDict'ler tanımlayın
2. Create, Update, Response types oluşturun
3. Success ve Error response types ekleyin
4. Type-safe API service implement edin
"""

# ÇÖZÜM:
class CreateUserRequest(TypedDict):
    """User creation request."""
    username: str
    email: str
    password: str
    role: Literal["admin", "user", "guest"]
    phone: NotRequired[str]  # Optional field

class UpdateUserRequest(TypedDict):
    """User update request - tüm alanlar optional."""
    username: NotRequired[str]
    email: NotRequired[str]
    role: NotRequired[Literal["admin", "user", "guest"]]
    phone: NotRequired[str]

class UserResponse(TypedDict):
    """User API response."""
    id: int
    username: str
    email: str
    role: str
    phone: str | None
    created_at: str
    is_active: bool

class APIError(TypedDict):
    """API error response."""
    error: str
    message: str
    status_code: int
    details: NotRequired[dict[str, Any]]

# Result type - Success ya da Error
APIResult: TypeAlias = UserResponse | APIError

class UserAPIService:
    """Type-safe user API service."""

    def __init__(self) -> None:
        """Service initialize."""
        self._users: dict[int, UserResponse] = {}
        self._next_id: int = 1

    def create_user(self, request: CreateUserRequest) -> APIResult:
        """
        User oluşturur.

        Args:
            request: User creation request

        Returns:
            UserResponse veya APIError
        """
        # Validation
        if len(request["password"]) < 8:
            return {
                "error": "ValidationError",
                "message": "Password must be at least 8 characters",
                "status_code": 400
            }

        if "@" not in request["email"]:
            return {
                "error": "ValidationError",
                "message": "Invalid email format",
                "status_code": 400
            }

        # Create user
        user_id = self._next_id
        self._next_id += 1

        user: UserResponse = {
            "id": user_id,
            "username": request["username"],
            "email": request["email"],
            "role": request["role"],
            "phone": request.get("phone"),
            "created_at": datetime.now().isoformat(),
            "is_active": True
        }

        self._users[user_id] = user
        return user

    def update_user(self, user_id: int, request: UpdateUserRequest) -> APIResult:
        """
        User günceller.

        Args:
            user_id: User ID
            request: Update request

        Returns:
            UserResponse veya APIError
        """
        if user_id not in self._users:
            return {
                "error": "NotFoundError",
                "message": f"User {user_id} not found",
                "status_code": 404
            }

        user = self._users[user_id]

        # Update fields
        if "username" in request:
            user["username"] = request["username"]
        if "email" in request:
            user["email"] = request["email"]
        if "role" in request:
            user["role"] = request["role"]
        if "phone" in request:
            user["phone"] = request["phone"]

        return user

    def get_user(self, user_id: int) -> APIResult:
        """User getirir."""
        if user_id not in self._users:
            return {
                "error": "NotFoundError",
                "message": f"User {user_id} not found",
                "status_code": 404
            }
        return self._users[user_id]


# ==================== ALIŞTIRMA 3: Protocol-Based Plugins ====================
# Zorluk: MEDIUM
# Konu: Protocol, runtime_checkable, duck typing

"""
TODO: Plugin sistemi için Protocol-based architecture oluşturun.

Gereksinimler:
1. Plugin protocol tanımlayın
2. Runtime type checking ekleyin
3. Plugin manager implement edin
4. Farklı plugin implementasyonları yapın
"""

# ÇÖZÜM:
from typing import runtime_checkable

@runtime_checkable
class Plugin(Protocol):
    """Plugin interface - Protocol ile duck typing."""

    @property
    def name(self) -> str:
        """Plugin adı."""
        ...

    @property
    def version(self) -> str:
        """Plugin versiyonu."""
        ...

    def initialize(self) -> None:
        """Plugin başlatılır."""
        ...

    def execute(self, data: dict[str, Any]) -> dict[str, Any]:
        """Plugin execute edilir."""
        ...

    def cleanup(self) -> None:
        """Plugin temizlenir."""
        ...

class PluginManager:
    """Plugin manager - Protocol-based."""

    def __init__(self) -> None:
        """Manager initialize."""
        self._plugins: dict[str, Plugin] = {}

    def register(self, plugin: Plugin) -> None:
        """
        Plugin kaydeder.

        Args:
            plugin: Plugin instance

        Raises:
            TypeError: Plugin protocol'ü implement etmiyorsa
        """
        # Runtime type check
        if not isinstance(plugin, Plugin):
            raise TypeError(f"{plugin} does not implement Plugin protocol")

        plugin.initialize()
        self._plugins[plugin.name] = plugin

    def unregister(self, plugin_name: str) -> None:
        """Plugin'i kaldırır."""
        if plugin_name in self._plugins:
            self._plugins[plugin_name].cleanup()
            del self._plugins[plugin_name]

    def execute(self, plugin_name: str, data: dict[str, Any]) -> dict[str, Any]:
        """
        Plugin execute eder.

        Args:
            plugin_name: Plugin adı
            data: Input data

        Returns:
            Plugin output
        """
        if plugin_name not in self._plugins:
            raise ValueError(f"Plugin {plugin_name} not found")

        return self._plugins[plugin_name].execute(data)

    def list_plugins(self) -> list[tuple[str, str]]:
        """Kayıtlı plugin'leri listeler."""
        return [(p.name, p.version) for p in self._plugins.values()]

# Plugin implementations - Protocol sayesinde explicit inheritance yok
class LoggerPlugin:
    """Logger plugin implementation."""

    @property
    def name(self) -> str:
        return "logger"

    @property
    def version(self) -> str:
        return "1.0.0"

    def initialize(self) -> None:
        print(f"Initializing {self.name} v{self.version}")

    def execute(self, data: dict[str, Any]) -> dict[str, Any]:
        print(f"Logging: {data}")
        return {"logged": True, **data}

    def cleanup(self) -> None:
        print(f"Cleaning up {self.name}")

class ValidationPlugin:
    """Validation plugin implementation."""

    @property
    def name(self) -> str:
        return "validator"

    @property
    def version(self) -> str:
        return "2.0.0"

    def initialize(self) -> None:
        print(f"Initializing {self.name} v{self.version}")

    def execute(self, data: dict[str, Any]) -> dict[str, Any]:
        # Simple validation
        is_valid = all(isinstance(v, (str, int, float)) for v in data.values())
        return {"valid": is_valid, **data}

    def cleanup(self) -> None:
        print(f"Cleaning up {self.name}")


# ==================== ALIŞTIRMA 4: Type-Safe Decorator Factory ====================
# Zorluk: HARD
# Konu: ParamSpec, Callable, Generic decorators

"""
TODO: Type-safe decorator factory oluşturun.

Gereksinimler:
1. ParamSpec kullanarak parametre tiplerini koruyun
2. Retry decorator yapın
3. Cache decorator yapın
4. Composition support ekleyin
"""

# ÇÖZÜM:
P = ParamSpec('P')
R = TypeVar('R')

def retry(max_attempts: int = 3) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """
    Retry decorator factory - type-safe.

    Args:
        max_attempts: Maksimum deneme sayısı

    Returns:
        Decorator function
    """
    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        """Decorator."""
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            """Wrapper function."""
            last_exception: Exception | None = None

            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    print(f"Attempt {attempt + 1} failed: {e}")

            # Tüm denemeler başarısız
            raise last_exception or Exception("All retries failed")

        return wrapper
    return decorator

def cache(func: Callable[P, R]) -> Callable[P, R]:
    """
    Simple cache decorator - type-safe.

    Args:
        func: Cached function

    Returns:
        Wrapper function
    """
    _cache: dict[tuple[Any, ...], R] = {}

    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        """Wrapper with caching."""
        # Simple cache key (hashable args only)
        try:
            cache_key = (args, tuple(sorted(kwargs.items())))
        except TypeError:
            # Non-hashable arguments, skip caching
            return func(*args, **kwargs)

        if cache_key not in _cache:
            _cache[cache_key] = func(*args, **kwargs)

        return _cache[cache_key]

    return wrapper

# Kullanım - tip güvenliği korunur
@retry(max_attempts=3)
def fetch_data(url: str, timeout: int = 30) -> dict[str, Any]:
    """Data fetch eder - retry ile."""
    print(f"Fetching {url} with timeout {timeout}")
    # Simulated fetch
    return {"data": "test"}

@cache
def expensive_calculation(x: int, y: int) -> int:
    """Pahalı hesaplama - cached."""
    print(f"Calculating {x} + {y}")
    return x + y


# ==================== ALIŞTIRMA 5: Generic State Machine ====================
# Zorluk: HARD
# Konu: Generic class, Enum, Literal, Type narrowing

"""
TODO: Type-safe state machine oluşturun.

Gereksinimler:
1. Generic state machine implement edin
2. State transitions type-safe yapın
3. State-specific data ekleyin
4. Transition validation yapın
"""

# ÇÖZÜM:
from enum import Enum, auto

StateT = TypeVar('StateT', bound=Enum)
DataT = TypeVar('DataT')

class Transition(Generic[StateT]):
    """State transition."""

    def __init__(self, from_state: StateT, to_state: StateT) -> None:
        """Transition initialize."""
        self.from_state = from_state
        self.to_state = to_state

class StateMachine(Generic[StateT, DataT]):
    """Generic state machine."""

    def __init__(
        self,
        initial_state: StateT,
        initial_data: DataT,
        transitions: list[Transition[StateT]]
    ) -> None:
        """
        State machine initialize.

        Args:
            initial_state: Başlangıç state
            initial_data: Başlangıç data
            transitions: İzin verilen transition'lar
        """
        self.current_state = initial_state
        self.data = initial_data
        self._transitions = self._build_transition_map(transitions)
        self._history: list[StateT] = [initial_state]

    def _build_transition_map(
        self,
        transitions: list[Transition[StateT]]
    ) -> dict[StateT, set[StateT]]:
        """Transition map oluşturur."""
        transition_map: dict[StateT, set[StateT]] = {}

        for transition in transitions:
            if transition.from_state not in transition_map:
                transition_map[transition.from_state] = set()
            transition_map[transition.from_state].add(transition.to_state)

        return transition_map

    def can_transition(self, to_state: StateT) -> bool:
        """
        Transition mümkün mü kontrol eder.

        Args:
            to_state: Hedef state

        Returns:
            Transition mümkünse True
        """
        return (
            self.current_state in self._transitions and
            to_state in self._transitions[self.current_state]
        )

    def transition(self, to_state: StateT) -> None:
        """
        State transition yapar.

        Args:
            to_state: Hedef state

        Raises:
            ValueError: Invalid transition
        """
        if not self.can_transition(to_state):
            raise ValueError(
                f"Invalid transition from {self.current_state} to {to_state}"
            )

        self.current_state = to_state
        self._history.append(to_state)

    def get_history(self) -> list[StateT]:
        """State geçmişini döner."""
        return self._history.copy()

# Concrete implementation
class OrderState(Enum):
    """Order state enum."""
    CREATED = auto()
    PAID = auto()
    SHIPPED = auto()
    DELIVERED = auto()
    CANCELLED = auto()

@dataclass
class OrderData:
    """Order data."""
    order_id: int
    total: float
    items: list[str]

def test_state_machine() -> None:
    """State machine test."""
    # Define valid transitions
    transitions = [
        Transition(OrderState.CREATED, OrderState.PAID),
        Transition(OrderState.CREATED, OrderState.CANCELLED),
        Transition(OrderState.PAID, OrderState.SHIPPED),
        Transition(OrderState.SHIPPED, OrderState.DELIVERED),
    ]

    # Create state machine
    order_data = OrderData(order_id=1, total=100.0, items=["item1", "item2"])
    sm: StateMachine[OrderState, OrderData] = StateMachine(
        initial_state=OrderState.CREATED,
        initial_data=order_data,
        transitions=transitions
    )

    # Valid transitions
    sm.transition(OrderState.PAID)
    sm.transition(OrderState.SHIPPED)
    sm.transition(OrderState.DELIVERED)

    # Check history
    history = sm.get_history()
    assert history == [
        OrderState.CREATED,
        OrderState.PAID,
        OrderState.SHIPPED,
        OrderState.DELIVERED
    ]


# ==================== ALIŞTIRMA 6: Function Overloading ====================
# Zorluk: HARD
# Konu: @overload, Literal, Union return types

"""
TODO: Function overloading ile type-safe API oluşturun.

Gereksinimler:
1. Farklı parametre kombinasyonları için farklı return tipleri
2. Literal types ile dispatch
3. Generic overload patterns
4. Type narrowing support
"""

# ÇÖZÜM:
# Overload 1: Data source bazlı farklı return types
@overload
def fetch_data(source: Literal["database"]) -> dict[str, Any]: ...

@overload
def fetch_data(source: Literal["api"]) -> list[dict[str, Any]]: ...

@overload
def fetch_data(source: Literal["cache"]) -> str: ...

def fetch_data(
    source: Literal["database", "api", "cache"]
) -> dict[str, Any] | list[dict[str, Any]] | str:
    """
    Kaynağa göre farklı formatta data getirir.

    Args:
        source: Data kaynağı

    Returns:
        Kaynağa göre farklı tip
    """
    if source == "database":
        return {"id": 1, "name": "User"}
    elif source == "api":
        return [{"id": 1}, {"id": 2}]
    else:  # cache
        return "cached_data_string"

# Overload 2: Return format parametresi
@overload
def get_users(as_dict: Literal[True]) -> dict[int, str]: ...

@overload
def get_users(as_dict: Literal[False]) -> list[str]: ...

def get_users(as_dict: bool = False) -> dict[int, str] | list[str]:
    """
    User'ları farklı formatlarda döner.

    Args:
        as_dict: Dict formatında dönülsün mü

    Returns:
        Dict veya list
    """
    users = {1: "Alice", 2: "Bob", 3: "Charlie"}
    return users if as_dict else list(users.values())

# Overload 3: Optional parameters
@overload
def create_connection(host: str) -> str: ...

@overload
def create_connection(host: str, port: int) -> tuple[str, int]: ...

@overload
def create_connection(host: str, port: int, ssl: bool) -> dict[str, Any]: ...

def create_connection(
    host: str,
    port: int | None = None,
    ssl: bool | None = None
) -> str | tuple[str, int] | dict[str, Any]:
    """
    Parametrelere göre farklı connection info döner.

    Args:
        host: Host address
        port: Port number (optional)
        ssl: SSL enabled (optional)

    Returns:
        Connection info - farklı formatlarda
    """
    if port is None:
        return host
    elif ssl is None:
        return (host, port)
    else:
        return {"host": host, "port": port, "ssl": ssl}

# Type-safe usage
def test_overloads() -> None:
    """Overload test."""
    # Type checker bilir ki result1 dict
    result1 = fetch_data("database")
    assert isinstance(result1, dict)

    # Type checker bilir ki result2 list
    result2 = fetch_data("api")
    assert isinstance(result2, list)

    # Type checker bilir ki result3 dict
    result3 = get_users(as_dict=True)
    assert isinstance(result3, dict)

    # Type checker bilir ki result4 list
    result4 = get_users(as_dict=False)
    assert isinstance(result4, list)


# ==================== ALIŞTIRMA 7: Type Guards ve Narrowing ====================
# Zorluk: HARD
# Konu: TypeGuard, isinstance, Type narrowing

"""
TODO: Custom type guards ile type narrowing yapın.

Gereksinimler:
1. TypeGuard kullanarak custom type check'ler
2. Union types için narrowing
3. Runtime validation ile type safety
4. Nested type guards
"""

# ÇÖZÜM:
# Type definitions
@dataclass
class SuccessResponse:
    """Success response."""
    status: Literal["success"]
    data: dict[str, Any]

@dataclass
class ErrorResponse:
    """Error response."""
    status: Literal["error"]
    message: str
    code: int

Response = SuccessResponse | ErrorResponse

# Type guard for string list
def is_string_list(value: list[Any]) -> TypeGuard[list[str]]:
    """
    Liste string'lerden mi kontrol eder.

    Args:
        value: Kontrol edilecek liste

    Returns:
        Tüm elemanlar string ise True
    """
    return all(isinstance(item, str) for item in value)

# Type guard for int list
def is_int_list(value: list[Any]) -> TypeGuard[list[int]]:
    """Liste int'lerden mi kontrol eder."""
    return all(isinstance(item, int) for item in value)

# Type guard for non-empty string
def is_non_empty_string(value: str | None) -> TypeGuard[str]:
    """
    Non-empty string mi kontrol eder.

    Args:
        value: Kontrol edilecek değer

    Returns:
        Non-empty string ise True
    """
    return value is not None and len(value) > 0

# Type guard for success response
def is_success(response: Response) -> TypeGuard[SuccessResponse]:
    """
    Response success mi kontrol eder.

    Args:
        response: Response object

    Returns:
        Success response ise True
    """
    return response.status == "success"

def process_response(response: Response) -> dict[str, Any]:
    """
    Response'u process eder - type narrowing ile.

    Args:
        response: Response object

    Returns:
        Processed data
    """
    if is_success(response):
        # Type narrowing: response artık SuccessResponse
        return response.data
    else:
        # Type narrowing: response artık ErrorResponse
        return {"error": response.message, "code": response.code}

def process_list(items: list[Any]) -> str:
    """
    Listeyi type-safe process eder.

    Args:
        items: Mixed type list

    Returns:
        Processed result
    """
    if is_string_list(items):
        # Type narrowing: items artık list[str]
        return ", ".join(items)
    elif is_int_list(items):
        # Type narrowing: items artık list[int]
        return f"Sum: {sum(items)}"
    else:
        return "Mixed or unknown types"

def validate_input(value: str | None) -> str:
    """
    Input validate eder.

    Args:
        value: Input value

    Returns:
        Validated string

    Raises:
        ValueError: Input geçersizse
    """
    if not is_non_empty_string(value):
        raise ValueError("Input must be a non-empty string")

    # Type narrowing: value artık str (None değil)
    return value.upper()


# ==================== ALIŞTIRMA 8: Async Generic Repository ====================
# Zorluk: EXPERT
# Konu: Async + Generics, Awaitable, Protocol

"""
TODO: Async generic repository pattern oluşturun.

Gereksinimler:
1. Async CRUD operations
2. Generic type support
3. Transaction support
4. Connection pooling simulation
"""

# ÇÖZÜM:
import asyncio

class AsyncRepository(Generic[T]):
    """Async generic repository."""

    def __init__(self, pool_size: int = 10) -> None:
        """
        Repository initialize.

        Args:
            pool_size: Connection pool size
        """
        self._items: dict[int, T] = {}
        self._next_id: int = 1
        self._pool_size = pool_size
        self._lock = asyncio.Lock()

    async def create(self, item: T) -> int:
        """
        Async item creation.

        Args:
            item: Item to create

        Returns:
            Created item ID
        """
        async with self._lock:
            # Simulate async DB operation
            await asyncio.sleep(0.01)

            item_id = self._next_id
            self._items[item_id] = item
            self._next_id += 1
            return item_id

    async def read(self, item_id: int) -> T | None:
        """
        Async item read.

        Args:
            item_id: Item ID

        Returns:
            Item or None
        """
        # Simulate async DB operation
        await asyncio.sleep(0.01)
        return self._items.get(item_id)

    async def update(self, item_id: int, item: T) -> bool:
        """Async item update."""
        async with self._lock:
            await asyncio.sleep(0.01)

            if item_id not in self._items:
                return False

            self._items[item_id] = item
            return True

    async def delete(self, item_id: int) -> bool:
        """Async item delete."""
        async with self._lock:
            await asyncio.sleep(0.01)

            if item_id not in self._items:
                return False

            del self._items[item_id]
            return True

    async def find_all(self) -> list[T]:
        """Async get all items."""
        await asyncio.sleep(0.01)
        return list(self._items.values())

    async def transaction(
        self,
        operations: Callable[[Self], Awaitable[None]]
    ) -> None:
        """
        Transaction context.

        Args:
            operations: Transaction operations
        """
        async with self._lock:
            # Simulate transaction
            await operations(self)

async def test_async_repository() -> None:
    """Async repository test."""
    repo: AsyncRepository[User] = AsyncRepository()

    # Create users
    user_id1 = await repo.create(User(0, "alice", "alice@example.com"))
    user_id2 = await repo.create(User(0, "bob", "bob@example.com"))

    # Read user
    user = await repo.read(user_id1)
    assert user is not None
    assert user.username == "alice"

    # Get all
    all_users = await repo.find_all()
    assert len(all_users) == 2


# ==================== ALIŞTIRMA 9: Type-Safe Event System ====================
# Zorluk: EXPERT
# Konu: Generic events, Protocol, Callable types

"""
TODO: Type-safe event bus sistemi oluşturun.

Gereksinimler:
1. Generic event definitions
2. Type-safe event handlers
3. Multiple subscribers
4. Event filtering
"""

# ÇÖZÜM:
EventT = TypeVar('EventT')

@dataclass
class Event:
    """Base event class."""
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class UserEvent(Event):
    """User event."""
    user_id: int = 0
    username: str = ""

@dataclass
class UserCreatedEvent(UserEvent):
    """User created event."""
    email: str = ""

@dataclass
class UserUpdatedEvent(UserEvent):
    """User updated event."""
    changes: dict[str, Any] = field(default_factory=dict)

@dataclass
class OrderEvent(Event):
    """Order event."""
    order_id: int = 0
    user_id: int = 0

@dataclass
class OrderPlacedEvent(OrderEvent):
    """Order placed event."""
    total: float = 0.0
    items: list[str] = field(default_factory=list)

class EventBus:
    """Type-safe event bus."""

    def __init__(self) -> None:
        """EventBus initialize."""
        self._handlers: dict[type, list[Callable[[Any], None]]] = {}
        self._event_log: list[tuple[type, Event]] = []

    def subscribe(
        self,
        event_type: type[EventT],
        handler: Callable[[EventT], None]
    ) -> None:
        """
        Event handler subscribe eder.

        Args:
            event_type: Event tipi
            handler: Handler function
        """
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        self._handlers[event_type].append(handler)

    def publish(self, event: EventT) -> None:
        """
        Event publish eder.

        Args:
            event: Event object
        """
        event_type = type(event)

        # Log event
        self._event_log.append((event_type, event))  # type: ignore

        # Call handlers
        if event_type in self._handlers:
            for handler in self._handlers[event_type]:
                handler(event)

    def get_events(self, event_type: type[EventT]) -> list[EventT]:
        """
        Belirli tipteki event'ları getirir.

        Args:
            event_type: Event tipi

        Returns:
            Event listesi
        """
        return [
            event  # type: ignore
            for evt_type, event in self._event_log
            if evt_type == event_type
        ]

    def clear_log(self) -> None:
        """Event log'unu temizler."""
        self._event_log.clear()

# Event handlers
def on_user_created(event: UserCreatedEvent) -> None:
    """User created handler."""
    print(f"User created: {event.username} ({event.email})")

def on_user_updated(event: UserUpdatedEvent) -> None:
    """User updated handler."""
    print(f"User updated: {event.username} - Changes: {event.changes}")

def on_order_placed(event: OrderPlacedEvent) -> None:
    """Order placed handler."""
    print(f"Order placed: {event.order_id} - Total: ${event.total}")

def test_event_bus() -> None:
    """Event bus test."""
    bus = EventBus()

    # Subscribe handlers
    bus.subscribe(UserCreatedEvent, on_user_created)
    bus.subscribe(UserUpdatedEvent, on_user_updated)
    bus.subscribe(OrderPlacedEvent, on_order_placed)

    # Publish events
    bus.publish(UserCreatedEvent(
        user_id=1,
        username="alice",
        email="alice@example.com"
    ))

    bus.publish(OrderPlacedEvent(
        order_id=100,
        user_id=1,
        total=299.99,
        items=["laptop", "mouse"]
    ))

    # Get events
    user_events = bus.get_events(UserCreatedEvent)
    assert len(user_events) == 1


# ==================== ALIŞTIRMA 10: Builder Pattern with Types ====================
# Zorluk: EXPERT
# Konu: Self, method chaining, fluent interface

"""
TODO: Type-safe builder pattern oluşturun.

Gereksinimler:
1. Fluent interface (method chaining)
2. Self return type
3. Validation
4. Immutable result
"""

# ÇÖZÜM:
from typing import Self

@dataclass(frozen=True)
class Query:
    """Immutable query object."""
    table: str
    columns: tuple[str, ...]
    where_clauses: tuple[str, ...]
    order_by: tuple[str, ...]
    limit: int | None

    def to_sql(self) -> str:
        """SQL query string oluşturur."""
        columns_str = ", ".join(self.columns) if self.columns else "*"
        sql = f"SELECT {columns_str} FROM {self.table}"

        if self.where_clauses:
            sql += f" WHERE {' AND '.join(self.where_clauses)}"

        if self.order_by:
            sql += f" ORDER BY {', '.join(self.order_by)}"

        if self.limit is not None:
            sql += f" LIMIT {self.limit}"

        return sql

class QueryBuilder:
    """Type-safe query builder with fluent interface."""

    def __init__(self) -> None:
        """Builder initialize."""
        self._table: str | None = None
        self._columns: list[str] = []
        self._where_clauses: list[str] = []
        self._order_by: list[str] = []
        self._limit: int | None = None

    def table(self, name: str) -> Self:
        """
        Table seçer.

        Args:
            name: Table name

        Returns:
            Self for chaining
        """
        self._table = name
        return self

    def select(self, *columns: str) -> Self:
        """
        Column'ları seçer.

        Args:
            columns: Column names

        Returns:
            Self for chaining
        """
        self._columns.extend(columns)
        return self

    def where(self, condition: str) -> Self:
        """
        WHERE condition ekler.

        Args:
            condition: WHERE condition

        Returns:
            Self for chaining
        """
        self._where_clauses.append(condition)
        return self

    def order_by(self, *columns: str) -> Self:
        """
        ORDER BY ekler.

        Args:
            columns: Column names

        Returns:
            Self for chaining
        """
        self._order_by.extend(columns)
        return self

    def limit(self, n: int) -> Self:
        """
        LIMIT ekler.

        Args:
            n: Limit count

        Returns:
            Self for chaining
        """
        if n < 0:
            raise ValueError("Limit must be non-negative")
        self._limit = n
        return self

    def build(self) -> Query:
        """
        Immutable Query object oluşturur.

        Returns:
            Query object

        Raises:
            ValueError: Table belirtilmemişse
        """
        if not self._table:
            raise ValueError("Table must be specified")

        return Query(
            table=self._table,
            columns=tuple(self._columns),
            where_clauses=tuple(self._where_clauses),
            order_by=tuple(self._order_by),
            limit=self._limit
        )

def test_query_builder() -> None:
    """Query builder test."""
    # Fluent interface - type-safe method chaining
    query = (QueryBuilder()
        .table("users")
        .select("id", "username", "email")
        .where("is_active = true")
        .where("age > 18")
        .order_by("created_at DESC")
        .limit(10)
        .build())

    sql = query.to_sql()
    expected = (
        "SELECT id, username, email FROM users "
        "WHERE is_active = true AND age > 18 "
        "ORDER BY created_at DESC LIMIT 10"
    )
    assert sql == expected


# ==================== ALIŞTIRMA 11: Dependency Injection Container ====================
# Zorluk: EXPERT
# Konu: Protocol, Generic, Type-safe DI

"""
TODO: Type-safe dependency injection container oluşturun.

Gereksinimler:
1. Protocol-based dependencies
2. Automatic resolution
3. Singleton support
4. Type-safe registration
"""

# ÇÖZÜM:
class Logger(Protocol):
    """Logger protocol."""
    def log(self, message: str) -> None: ...
    def error(self, message: str) -> None: ...

class Database(Protocol):
    """Database protocol."""
    def query(self, sql: str) -> list[dict[str, Any]]: ...
    def execute(self, sql: str) -> None: ...

class ConsoleLogger:
    """Console logger implementation."""

    def log(self, message: str) -> None:
        print(f"[LOG] {message}")

    def error(self, message: str) -> None:
        print(f"[ERROR] {message}")

class InMemoryDatabase:
    """In-memory database implementation."""

    def __init__(self) -> None:
        self._data: list[dict[str, Any]] = []

    def query(self, sql: str) -> list[dict[str, Any]]:
        return self._data

    def execute(self, sql: str) -> None:
        print(f"Executing: {sql}")

class DIContainer:
    """Type-safe dependency injection container."""

    def __init__(self) -> None:
        """Container initialize."""
        self._services: dict[type, Any] = {}
        self._singletons: dict[type, Any] = {}

    def register(
        self,
        service_type: type[T],
        implementation: Callable[[], T],
        singleton: bool = False
    ) -> None:
        """
        Service kaydeder.

        Args:
            service_type: Service interface type
            implementation: Factory function
            singleton: Singleton instance mı
        """
        self._services[service_type] = (implementation, singleton)

    def register_instance(self, service_type: type[T], instance: T) -> None:
        """
        Instance kaydeder (singleton).

        Args:
            service_type: Service type
            instance: Service instance
        """
        self._singletons[service_type] = instance

    def resolve(self, service_type: type[T]) -> T:
        """
        Service resolve eder.

        Args:
            service_type: Service type

        Returns:
            Service instance

        Raises:
            ValueError: Service kayıtlı değilse
        """
        # Check singletons first
        if service_type in self._singletons:
            return self._singletons[service_type]

        if service_type not in self._services:
            raise ValueError(f"Service {service_type} not registered")

        factory, is_singleton = self._services[service_type]
        instance = factory()

        if is_singleton:
            self._singletons[service_type] = instance

        return instance

class UserService:
    """User service with DI."""

    def __init__(self, db: Database, logger: Logger) -> None:
        """
        Service initialize with dependencies.

        Args:
            db: Database instance
            logger: Logger instance
        """
        self._db = db
        self._logger = logger

    def get_users(self) -> list[dict[str, Any]]:
        """User'ları getirir."""
        self._logger.log("Fetching users")
        return self._db.query("SELECT * FROM users")

    def create_user(self, username: str) -> None:
        """User oluşturur."""
        self._logger.log(f"Creating user: {username}")
        self._db.execute(f"INSERT INTO users (username) VALUES ('{username}')")

def test_di_container() -> None:
    """DI container test."""
    container = DIContainer()

    # Register services
    container.register_instance(Logger, ConsoleLogger())
    container.register_instance(Database, InMemoryDatabase())

    # Resolve dependencies
    logger = container.resolve(Logger)
    db = container.resolve(Database)

    # Create service with dependencies
    user_service = UserService(db, logger)
    user_service.create_user("alice")


# ==================== ALIŞTIRMA 12: Advanced Type Validation ====================
# Zorluk: EXPERT
# Konu: Runtime validation, TypeGuard, Generic validators

"""
TODO: Runtime type validation sistemi oluşturun.

Gereksinimler:
1. Generic validators
2. Nested validation
3. Custom validation rules
4. Type-safe error messages
"""

# ÇÖZÜM:
@dataclass
class ValidationError:
    """Validation error."""
    field: str
    message: str
    value: Any

ValidationResult: TypeAlias = tuple[bool, list[ValidationError]]

class Validator(Generic[T]):
    """Generic validator."""

    def __init__(self, type_: type[T]) -> None:
        """
        Validator initialize.

        Args:
            type_: Expected type
        """
        self._type = type_
        self._rules: list[Callable[[T], ValidationResult]] = []

    def add_rule(self, rule: Callable[[T], ValidationResult]) -> Self:
        """
        Validation rule ekler.

        Args:
            rule: Validation function

        Returns:
            Self for chaining
        """
        self._rules.append(rule)
        return self

    def validate(self, value: Any) -> ValidationResult:
        """
        Value'yu validate eder.

        Args:
            value: Validate edilecek değer

        Returns:
            (is_valid, errors) tuple
        """
        errors: list[ValidationError] = []

        # Type check
        if not isinstance(value, self._type):
            errors.append(ValidationError(
                field="type",
                message=f"Expected {self._type}, got {type(value)}",
                value=value
            ))
            return (False, errors)

        # Apply rules
        for rule in self._rules:
            is_valid, rule_errors = rule(value)
            if not is_valid:
                errors.extend(rule_errors)

        return (len(errors) == 0, errors)

class StringValidator(Validator[str]):
    """String validator with common rules."""

    def min_length(self, length: int) -> Self:
        """Minimum length rule."""
        def rule(value: str) -> ValidationResult:
            if len(value) < length:
                return (False, [ValidationError(
                    field="length",
                    message=f"Must be at least {length} characters",
                    value=value
                )])
            return (True, [])

        return self.add_rule(rule)

    def max_length(self, length: int) -> Self:
        """Maximum length rule."""
        def rule(value: str) -> ValidationResult:
            if len(value) > length:
                return (False, [ValidationError(
                    field="length",
                    message=f"Must be at most {length} characters",
                    value=value
                )])
            return (True, [])

        return self.add_rule(rule)

    def pattern(self, pattern: str) -> Self:
        """Regex pattern rule."""
        import re
        compiled = re.compile(pattern)

        def rule(value: str) -> ValidationResult:
            if not compiled.match(value):
                return (False, [ValidationError(
                    field="pattern",
                    message=f"Must match pattern: {pattern}",
                    value=value
                )])
            return (True, [])

        return self.add_rule(rule)

class IntValidator(Validator[int]):
    """Int validator with common rules."""

    def min_value(self, min_val: int) -> Self:
        """Minimum value rule."""
        def rule(value: int) -> ValidationResult:
            if value < min_val:
                return (False, [ValidationError(
                    field="value",
                    message=f"Must be at least {min_val}",
                    value=value
                )])
            return (True, [])

        return self.add_rule(rule)

    def max_value(self, max_val: int) -> Self:
        """Maximum value rule."""
        def rule(value: int) -> ValidationResult:
            if value > max_val:
                return (False, [ValidationError(
                    field="value",
                    message=f"Must be at most {max_val}",
                    value=value
                )])
            return (True, [])

        return self.add_rule(rule)

def test_validators() -> None:
    """Validator test."""
    # String validation
    username_validator = (StringValidator(str)
        .min_length(3)
        .max_length(20)
        .pattern(r'^[a-zA-Z0-9_]+$'))

    is_valid, errors = username_validator.validate("alice")
    assert is_valid

    is_valid, errors = username_validator.validate("ab")
    assert not is_valid
    assert len(errors) == 1
    assert errors[0].field == "length"

    # Int validation
    age_validator = (IntValidator(int)
        .min_value(18)
        .max_value(100))

    is_valid, errors = age_validator.validate(25)
    assert is_valid

    is_valid, errors = age_validator.validate(15)
    assert not is_valid


# ==================== TEST RUNNER ====================
if __name__ == "__main__":
    print("=" * 60)
    print("TYPE HINTS ALIŞTIRMALARI - TEST")
    print("=" * 60)

    # Test 1
    print("\n1. Generic Repository Pattern")
    test_repository()
    print("✓ Repository tests passed")

    # Test 2
    print("\n2. TypedDict API Tests")
    service = UserAPIService()
    result = service.create_user({
        "username": "john",
        "email": "john@example.com",
        "password": "password123",
        "role": "user"
    })
    print(f"✓ API service tests passed: {result}")

    # Test 3
    print("\n3. Protocol-Based Plugins")
    manager = PluginManager()
    manager.register(LoggerPlugin())
    manager.register(ValidationPlugin())
    plugins = manager.list_plugins()
    print(f"✓ Plugin manager tests passed: {plugins}")

    # Test 4
    print("\n4. Type-Safe Decorators")
    result = fetch_data("https://api.example.com")
    calc_result = expensive_calculation(5, 3)
    print(f"✓ Decorator tests passed: {calc_result}")

    # Test 5
    print("\n5. Generic State Machine")
    test_state_machine()
    print("✓ State machine tests passed")

    # Test 6
    print("\n6. Function Overloading")
    test_overloads()
    print("✓ Overload tests passed")

    # Test 7
    print("\n7. Type Guards")
    response: Response = SuccessResponse(status="success", data={"key": "value"})
    result = process_response(response)
    print(f"✓ Type guard tests passed: {result}")

    # Test 8
    print("\n8. Async Repository")
    asyncio.run(test_async_repository())
    print("✓ Async repository tests passed")

    # Test 9
    print("\n9. Type-Safe Event System")
    test_event_bus()
    print("✓ Event bus tests passed")

    # Test 10
    print("\n10. Builder Pattern")
    test_query_builder()
    print("✓ Builder pattern tests passed")

    # Test 11
    print("\n11. Dependency Injection")
    test_di_container()
    print("✓ DI container tests passed")

    # Test 12
    print("\n12. Runtime Validation")
    test_validators()
    print("✓ Validation tests passed")

    print("\n" + "=" * 60)
    print("TÜM TESTLER BAŞARILI!")
    print("=" * 60)
