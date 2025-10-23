"""
DESIGN PATTERNS - ADVANCED EXERCISES
Her exercise için hem TODO kısmı hem de solution bulunmaktadır.
Önce TODO kısmını kendiniz implement etmeyi deneyin, sonra çözüme bakın.
"""

# ============================================================================
# EXERCISE 1: Singleton Pattern - Configuration Manager
# Seviye: Medium
# Konu: Thread-safe Singleton pattern implementation
# ============================================================================

print("=" * 80)
print("EXERCISE 1: Singleton Pattern - Configuration Manager")
print("=" * 80)

"""
TODO: Thread-safe bir Configuration Manager sınıfı oluşturun.
- Metaclass kullanarak Singleton pattern uygulayın
- set_config() ve get_config() methodları ekleyin
- Thread-safe olmalı (threading.Lock kullanın)
- Konfigürasyon ayarlarını dictionary'de saklayın
"""

# TODO: Kodunuzu buraya yazın
# import threading
#
# class SingletonMeta(type):
#     # Implement singleton metaclass
#     pass
#
# class ConfigurationManager(metaclass=SingletonMeta):
#     # Implement configuration manager
#     pass

# -------- SOLUTION --------
import threading

class SingletonMeta(type):
    """Thread-safe Singleton metaclass"""
    _instances = {}
    _lock = threading.Lock()

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            with cls._lock:
                if cls not in cls._instances:
                    cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]

class ConfigurationManager(metaclass=SingletonMeta):
    """Singleton configuration manager"""
    def __init__(self):
        self._config = {}
        self._lock = threading.Lock()

    def set_config(self, key: str, value):
        """Thread-safe config setter"""
        with self._lock:
            self._config[key] = value
            print(f"Config set: {key} = {value}")

    def get_config(self, key: str, default=None):
        """Thread-safe config getter"""
        with self._lock:
            return self._config.get(key, default)

    def get_all_config(self):
        """Get all configuration"""
        with self._lock:
            return self._config.copy()

# Test
config1 = ConfigurationManager()
config1.set_config("database_host", "localhost")
config1.set_config("database_port", 5432)

config2 = ConfigurationManager()
print(f"Config1 is Config2: {config1 is config2}")  # True
print(f"Database host: {config2.get_config('database_host')}")
print(f"All config: {config2.get_all_config()}")
print()


# ============================================================================
# EXERCISE 2: Factory Pattern - Notification System
# Seviye: Medium
# Konu: Factory Method Pattern ile farklı notification tipleri
# ============================================================================

print("=" * 80)
print("EXERCISE 2: Factory Pattern - Notification System")
print("=" * 80)

"""
TODO: Farklı tipte bildirimler gönderen bir Factory Pattern oluşturun.
- Notification abstract base class oluşturun (send metodu)
- EmailNotification, SMSNotification, PushNotification sınıfları
- NotificationFactory ile notification tipine göre ilgili sınıfı döndürün
- Enum ile notification tiplerini tanımlayın
"""

# TODO: Kodunuzu buraya yazın
# from abc import ABC, abstractmethod
# from enum import Enum
#
# class NotificationType(Enum):
#     # Define notification types
#     pass
#
# class Notification(ABC):
#     # Implement abstract notification class
#     pass

# -------- SOLUTION --------
from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict

class NotificationType(Enum):
    EMAIL = "email"
    SMS = "sms"
    PUSH = "push"

class Notification(ABC):
    """Abstract notification class"""
    @abstractmethod
    def send(self, recipient: str, message: str) -> bool:
        pass

class EmailNotification(Notification):
    def send(self, recipient: str, message: str) -> bool:
        print(f"[EMAIL] Sending to {recipient}: {message}")
        # Simülasyon: Email gönderme
        return True

class SMSNotification(Notification):
    def send(self, recipient: str, message: str) -> bool:
        print(f"[SMS] Sending to {recipient}: {message}")
        # Simülasyon: SMS gönderme
        return True

class PushNotification(Notification):
    def send(self, recipient: str, message: str) -> bool:
        print(f"[PUSH] Sending to {recipient}: {message}")
        # Simülasyon: Push notification
        return True

class NotificationFactory:
    """Factory for creating notification instances"""
    _notification_map: Dict[NotificationType, type] = {
        NotificationType.EMAIL: EmailNotification,
        NotificationType.SMS: SMSNotification,
        NotificationType.PUSH: PushNotification
    }

    @classmethod
    def create_notification(cls, notification_type: NotificationType) -> Notification:
        notification_class = cls._notification_map.get(notification_type)
        if not notification_class:
            raise ValueError(f"Unknown notification type: {notification_type}")
        return notification_class()

# Test
factory = NotificationFactory()

email_notif = factory.create_notification(NotificationType.EMAIL)
email_notif.send("user@example.com", "Your order has been shipped!")

sms_notif = factory.create_notification(NotificationType.SMS)
sms_notif.send("+1234567890", "Verification code: 123456")

push_notif = factory.create_notification(NotificationType.PUSH)
push_notif.send("user_device_123", "New message received!")
print()


# ============================================================================
# EXERCISE 3: Builder Pattern - HTTP Request Builder
# Seviye: Advanced
# Konu: Fluent Builder pattern ile HTTP request oluşturma
# ============================================================================

print("=" * 80)
print("EXERCISE 3: Builder Pattern - HTTP Request Builder")
print("=" * 80)

"""
TODO: HTTP request oluşturmak için Builder Pattern kullanın.
- HTTPRequest class (method, url, headers, body, params)
- HTTPRequestBuilder class ile method chaining
- set_method(), set_url(), add_header(), set_body(), add_param() metodları
- build() metodu HTTPRequest objesi döndürmeli
- __str__ metodu ile request'i string olarak yazdırın
"""

# TODO: Kodunuzu buraya yazın

# -------- SOLUTION --------
from typing import Dict, Optional, Any

class HTTPRequest:
    """HTTP Request object"""
    def __init__(self):
        self.method: str = "GET"
        self.url: str = ""
        self.headers: Dict[str, str] = {}
        self.body: Optional[str] = None
        self.params: Dict[str, str] = {}

    def __str__(self):
        lines = [f"{self.method} {self.url}"]

        # Add query parameters
        if self.params:
            param_str = "&".join(f"{k}={v}" for k, v in self.params.items())
            lines[0] += f"?{param_str}"

        # Add headers
        for key, value in self.headers.items():
            lines.append(f"{key}: {value}")

        # Add body
        if self.body:
            lines.append("")
            lines.append(self.body)

        return "\n".join(lines)

class HTTPRequestBuilder:
    """Builder for HTTP requests with method chaining"""
    def __init__(self):
        self.request = HTTPRequest()

    def set_method(self, method: str):
        """Set HTTP method (GET, POST, PUT, DELETE, etc.)"""
        self.request.method = method.upper()
        return self

    def set_url(self, url: str):
        """Set request URL"""
        self.request.url = url
        return self

    def add_header(self, key: str, value: str):
        """Add a header"""
        self.request.headers[key] = value
        return self

    def set_body(self, body: str):
        """Set request body"""
        self.request.body = body
        return self

    def add_param(self, key: str, value: str):
        """Add query parameter"""
        self.request.params[key] = value
        return self

    def build(self) -> HTTPRequest:
        """Build and return the HTTP request"""
        return self.request

# Test
request = (HTTPRequestBuilder()
    .set_method("POST")
    .set_url("https://api.example.com/users")
    .add_header("Content-Type", "application/json")
    .add_header("Authorization", "Bearer token123")
    .add_param("page", "1")
    .add_param("limit", "10")
    .set_body('{"name": "John Doe", "email": "john@example.com"}')
    .build())

print("Built HTTP Request:")
print(request)
print()


# ============================================================================
# EXERCISE 4: Observer Pattern - Event System
# Seviye: Advanced
# Konu: Event-driven architecture ile Observer pattern
# ============================================================================

print("=" * 80)
print("EXERCISE 4: Observer Pattern - Event System")
print("=" * 80)

"""
TODO: Event-driven bir sistem oluşturun.
- Event class (event_type, data)
- Observer abstract class (update metodu)
- EventManager class (subscribe, unsubscribe, notify)
- Farklı event tipleri için farklı observer'lar
- User login event örneği (LoggingObserver, EmailObserver, AnalyticsObserver)
"""

# TODO: Kodunuzu buraya yazın

# -------- SOLUTION --------
from abc import ABC, abstractmethod
from typing import List, Dict, Any
from dataclasses import dataclass
from datetime import datetime

@dataclass
class Event:
    """Event data class"""
    event_type: str
    data: Dict[str, Any]
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

class Observer(ABC):
    """Abstract observer"""
    @abstractmethod
    def update(self, event: Event):
        pass

class LoggingObserver(Observer):
    """Logs all events"""
    def update(self, event: Event):
        print(f"[LOG] {event.timestamp.strftime('%Y-%m-%d %H:%M:%S')} - "
              f"{event.event_type}: {event.data}")

class EmailObserver(Observer):
    """Sends email notifications"""
    def update(self, event: Event):
        if event.event_type == "user_login":
            user = event.data.get("username")
            print(f"[EMAIL] Sending welcome email to {user}")

class AnalyticsObserver(Observer):
    """Tracks analytics"""
    def __init__(self):
        self.event_count = {}

    def update(self, event: Event):
        self.event_count[event.event_type] = \
            self.event_count.get(event.event_type, 0) + 1
        print(f"[ANALYTICS] Event '{event.event_type}' occurred "
              f"{self.event_count[event.event_type]} time(s)")

class SecurityObserver(Observer):
    """Monitors security events"""
    def update(self, event: Event):
        if event.event_type == "user_login":
            ip = event.data.get("ip_address")
            user = event.data.get("username")
            print(f"[SECURITY] Login detected from {ip} for user {user}")

class EventManager:
    """Manages events and observers"""
    def __init__(self):
        self._observers: Dict[str, List[Observer]] = {}

    def subscribe(self, event_type: str, observer: Observer):
        """Subscribe an observer to an event type"""
        if event_type not in self._observers:
            self._observers[event_type] = []

        if observer not in self._observers[event_type]:
            self._observers[event_type].append(observer)
            print(f"Observer {observer.__class__.__name__} subscribed to '{event_type}'")

    def unsubscribe(self, event_type: str, observer: Observer):
        """Unsubscribe an observer from an event type"""
        if event_type in self._observers and observer in self._observers[event_type]:
            self._observers[event_type].remove(observer)
            print(f"Observer {observer.__class__.__name__} unsubscribed from '{event_type}'")

    def notify(self, event: Event):
        """Notify all observers of an event"""
        print(f"\n{'='*60}")
        print(f"Event triggered: {event.event_type}")
        print(f"{'='*60}")

        if event.event_type in self._observers:
            for observer in self._observers[event.event_type]:
                observer.update(event)

# Test
event_manager = EventManager()

# Create observers
logger = LoggingObserver()
email_notifier = EmailObserver()
analytics = AnalyticsObserver()
security = SecurityObserver()

# Subscribe to events
event_manager.subscribe("user_login", logger)
event_manager.subscribe("user_login", email_notifier)
event_manager.subscribe("user_login", analytics)
event_manager.subscribe("user_login", security)

# Trigger events
login_event = Event(
    event_type="user_login",
    data={
        "username": "john_doe",
        "ip_address": "192.168.1.100",
        "user_agent": "Mozilla/5.0"
    }
)

event_manager.notify(login_event)

# Another login
login_event2 = Event(
    event_type="user_login",
    data={
        "username": "jane_smith",
        "ip_address": "192.168.1.101",
        "user_agent": "Chrome/91.0"
    }
)

event_manager.notify(login_event2)
print()


# ============================================================================
# EXERCISE 5: Strategy Pattern - Compression Algorithms
# Seviye: Medium
# Konu: Runtime'da algoritma seçimi
# ============================================================================

print("=" * 80)
print("EXERCISE 5: Strategy Pattern - Compression Algorithms")
print("=" * 80)

"""
TODO: Farklı sıkıştırma algoritmalarını destekleyen bir sistem yapın.
- CompressionStrategy abstract class (compress, decompress)
- ZipCompression, GzipCompression, Bz2Compression sınıfları
- FileCompressor class (strategy pattern kullanarak)
- set_strategy() ile algoritma değiştirme
- compress_file() ve decompress_file() metodları
"""

# TODO: Kodunuzu buraya yazın

# -------- SOLUTION --------
from abc import ABC, abstractmethod
import zlib
import gzip
import bz2

class CompressionStrategy(ABC):
    """Abstract compression strategy"""
    @abstractmethod
    def compress(self, data: bytes) -> bytes:
        pass

    @abstractmethod
    def decompress(self, data: bytes) -> bytes:
        pass

    @abstractmethod
    def get_name(self) -> str:
        pass

class ZlibCompression(CompressionStrategy):
    """Zlib compression"""
    def compress(self, data: bytes) -> bytes:
        return zlib.compress(data)

    def decompress(self, data: bytes) -> bytes:
        return zlib.decompress(data)

    def get_name(self) -> str:
        return "ZLIB"

class GzipCompression(CompressionStrategy):
    """Gzip compression"""
    def compress(self, data: bytes) -> bytes:
        return gzip.compress(data)

    def decompress(self, data: bytes) -> bytes:
        return gzip.decompress(data)

    def get_name(self) -> str:
        return "GZIP"

class Bz2Compression(CompressionStrategy):
    """Bz2 compression"""
    def compress(self, data: bytes) -> bytes:
        return bz2.compress(data)

    def decompress(self, data: bytes) -> bytes:
        return bz2.decompress(data)

    def get_name(self) -> str:
        return "BZ2"

class FileCompressor:
    """File compressor using strategy pattern"""
    def __init__(self, strategy: CompressionStrategy):
        self._strategy = strategy

    def set_strategy(self, strategy: CompressionStrategy):
        """Change compression strategy"""
        self._strategy = strategy
        print(f"Compression strategy changed to: {strategy.get_name()}")

    def compress_data(self, data: str) -> bytes:
        """Compress data using current strategy"""
        data_bytes = data.encode('utf-8')
        compressed = self._strategy.compress(data_bytes)

        original_size = len(data_bytes)
        compressed_size = len(compressed)
        ratio = (1 - compressed_size / original_size) * 100

        print(f"[{self._strategy.get_name()}] Compressed: "
              f"{original_size} bytes -> {compressed_size} bytes "
              f"(Compression ratio: {ratio:.2f}%)")

        return compressed

    def decompress_data(self, compressed_data: bytes) -> str:
        """Decompress data using current strategy"""
        decompressed = self._strategy.decompress(compressed_data)
        print(f"[{self._strategy.get_name()}] Decompressed: "
              f"{len(compressed_data)} bytes -> {len(decompressed)} bytes")
        return decompressed.decode('utf-8')

# Test
test_data = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. " * 100

# Test with different compression algorithms
compressor = FileCompressor(ZlibCompression())
compressed_zlib = compressor.compress_data(test_data)

compressor.set_strategy(GzipCompression())
compressed_gzip = compressor.compress_data(test_data)

compressor.set_strategy(Bz2Compression())
compressed_bz2 = compressor.compress_data(test_data)

# Decompress
compressor.set_strategy(ZlibCompression())
decompressed = compressor.decompress_data(compressed_zlib)
print(f"Decompressed data length: {len(decompressed)}")
print()


# ============================================================================
# EXERCISE 6: Decorator Pattern - Logging Decorator
# Seviye: Advanced
# Konu: Function decorator ile logging, timing, caching
# ============================================================================

print("=" * 80)
print("EXERCISE 6: Decorator Pattern - Function Decorators")
print("=" * 80)

"""
TODO: Fonksiyonlar için decorator'lar oluşturun.
- @timeit - Fonksiyon çalışma süresini ölçer
- @memoize - Fonksiyon sonuçlarını cache'ler
- @retry - Hata durumunda yeniden dener (max_retries parametresi)
- @validate_types - Parametre tiplerini kontrol eder
- Birden fazla decorator birlikte kullanılabilmeli
"""

# TODO: Kodunuzu buraya yazın

# -------- SOLUTION --------
import functools
import time
from typing import Callable, Any, Dict

def timeit(func: Callable) -> Callable:
    """Measure function execution time"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"[TIMEIT] {func.__name__} took {end - start:.4f} seconds")
        return result
    return wrapper

def memoize(func: Callable) -> Callable:
    """Cache function results"""
    cache: Dict[tuple, Any] = {}

    @functools.wraps(func)
    def wrapper(*args):
        if args in cache:
            print(f"[CACHE] Returning cached result for {func.__name__}{args}")
            return cache[args]

        result = func(*args)
        cache[args] = result
        return result

    wrapper.cache = cache
    wrapper.cache_clear = lambda: cache.clear()
    return wrapper

def retry(max_retries: int = 3, delay: float = 1.0):
    """Retry function on failure"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    print(f"[RETRY] Attempt {attempt + 1}/{max_retries} failed: {e}")
                    if attempt < max_retries - 1:
                        time.sleep(delay)

            print(f"[RETRY] All {max_retries} attempts failed")
            raise last_exception

        return wrapper
    return decorator

def validate_types(**type_hints):
    """Validate function parameter types"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get function signature
            import inspect
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()

            # Validate types
            for param_name, expected_type in type_hints.items():
                if param_name in bound_args.arguments:
                    value = bound_args.arguments[param_name]
                    if not isinstance(value, expected_type):
                        raise TypeError(
                            f"Parameter '{param_name}' must be {expected_type.__name__}, "
                            f"got {type(value).__name__}"
                        )

            return func(*args, **kwargs)

        return wrapper
    return decorator

# Test functions
@timeit
@memoize
def fibonacci(n: int) -> int:
    """Calculate fibonacci number"""
    if n < 2:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

@retry(max_retries=3, delay=0.5)
def unstable_function(fail_count: int = 0):
    """Simulates an unstable function"""
    import random
    if random.random() < 0.7 and fail_count > 0:
        raise ValueError("Random failure!")
    return "Success!"

@validate_types(name=str, age=int, score=float)
def process_user(name: str, age: int, score: float):
    """Process user with type validation"""
    print(f"Processing user: {name}, age: {age}, score: {score}")
    return {"name": name, "age": age, "score": score}

# Test
print("Testing fibonacci with memoization:")
print(f"fib(10) = {fibonacci(10)}")
print(f"fib(10) = {fibonacci(10)}")  # Cached
print()

print("Testing retry decorator:")
try:
    result = unstable_function(fail_count=2)
    print(f"Result: {result}")
except Exception as e:
    print(f"Final error: {e}")
print()

print("Testing type validation:")
process_user("John", 25, 95.5)
try:
    process_user("Jane", "25", 95.5)  # Will fail - age is string
except TypeError as e:
    print(f"Type validation error: {e}")
print()


# ============================================================================
# EXERCISE 7: Command Pattern - Undo/Redo Text Editor
# Seviye: Advanced
# Konu: Command pattern ile undo/redo functionality
# ============================================================================

print("=" * 80)
print("EXERCISE 7: Command Pattern - Text Editor with Undo/Redo")
print("=" * 80)

"""
TODO: Undo/Redo özellikli bir text editor oluşturun.
- Command abstract class (execute, undo)
- InsertTextCommand, DeleteTextCommand, ReplaceTextCommand
- TextEditor class (insert, delete, replace metodları)
- CommandManager class (execute, undo, redo)
- Komut history'si tutun ve undo/redo desteği ekleyin
"""

# TODO: Kodunuzu buraya yazın

# -------- SOLUTION --------
from abc import ABC, abstractmethod
from typing import List

class Command(ABC):
    """Abstract command"""
    @abstractmethod
    def execute(self):
        pass

    @abstractmethod
    def undo(self):
        pass

class TextEditor:
    """Simple text editor"""
    def __init__(self):
        self.text = ""

    def insert(self, position: int, text: str):
        """Insert text at position"""
        self.text = self.text[:position] + text + self.text[position:]

    def delete(self, position: int, length: int):
        """Delete text from position"""
        self.text = self.text[:position] + self.text[position + length:]

    def replace(self, position: int, length: int, text: str):
        """Replace text at position"""
        self.text = self.text[:position] + text + self.text[position + length:]

    def get_text(self):
        """Get current text"""
        return self.text

class InsertTextCommand(Command):
    """Insert text command"""
    def __init__(self, editor: TextEditor, position: int, text: str):
        self.editor = editor
        self.position = position
        self.text = text

    def execute(self):
        self.editor.insert(self.position, self.text)
        print(f"Inserted '{self.text}' at position {self.position}")

    def undo(self):
        self.editor.delete(self.position, len(self.text))
        print(f"Undid insert of '{self.text}'")

class DeleteTextCommand(Command):
    """Delete text command"""
    def __init__(self, editor: TextEditor, position: int, length: int):
        self.editor = editor
        self.position = position
        self.length = length
        self.deleted_text = ""

    def execute(self):
        # Save deleted text for undo
        self.deleted_text = self.editor.get_text()[
            self.position:self.position + self.length
        ]
        self.editor.delete(self.position, self.length)
        print(f"Deleted '{self.deleted_text}' from position {self.position}")

    def undo(self):
        self.editor.insert(self.position, self.deleted_text)
        print(f"Undid delete of '{self.deleted_text}'")

class ReplaceTextCommand(Command):
    """Replace text command"""
    def __init__(self, editor: TextEditor, position: int, length: int, text: str):
        self.editor = editor
        self.position = position
        self.length = length
        self.new_text = text
        self.old_text = ""

    def execute(self):
        # Save old text for undo
        self.old_text = self.editor.get_text()[
            self.position:self.position + self.length
        ]
        self.editor.replace(self.position, self.length, self.new_text)
        print(f"Replaced '{self.old_text}' with '{self.new_text}' "
              f"at position {self.position}")

    def undo(self):
        self.editor.replace(self.position, len(self.new_text), self.old_text)
        print(f"Undid replace: restored '{self.old_text}'")

class CommandManager:
    """Manages command execution and history"""
    def __init__(self):
        self.history: List[Command] = []
        self.current_index = -1

    def execute(self, command: Command):
        """Execute a command and add to history"""
        # Remove any redo history
        self.history = self.history[:self.current_index + 1]

        command.execute()
        self.history.append(command)
        self.current_index += 1

    def undo(self):
        """Undo last command"""
        if self.current_index >= 0:
            command = self.history[self.current_index]
            command.undo()
            self.current_index -= 1
            return True
        print("Nothing to undo")
        return False

    def redo(self):
        """Redo previously undone command"""
        if self.current_index < len(self.history) - 1:
            self.current_index += 1
            command = self.history[self.current_index]
            command.execute()
            return True
        print("Nothing to redo")
        return False

    def get_history(self):
        """Get command history"""
        return [cmd.__class__.__name__ for cmd in self.history]

# Test
editor = TextEditor()
manager = CommandManager()

print("Creating document:")
manager.execute(InsertTextCommand(editor, 0, "Hello "))
manager.execute(InsertTextCommand(editor, 6, "World"))
manager.execute(InsertTextCommand(editor, 11, "!"))
print(f"Text: '{editor.get_text()}'")
print()

print("Deleting '!':")
manager.execute(DeleteTextCommand(editor, 11, 1))
print(f"Text: '{editor.get_text()}'")
print()

print("Replacing 'World' with 'Python':")
manager.execute(ReplaceTextCommand(editor, 6, 5, "Python"))
print(f"Text: '{editor.get_text()}'")
print()

print("Undo operations:")
manager.undo()
print(f"Text: '{editor.get_text()}'")
manager.undo()
print(f"Text: '{editor.get_text()}'")
print()

print("Redo operations:")
manager.redo()
print(f"Text: '{editor.get_text()}'")
print()


# ============================================================================
# EXERCISE 8: State Pattern - TCP Connection
# Seviye: Advanced
# Konu: State machine implementation
# ============================================================================

print("=" * 80)
print("EXERCISE 8: State Pattern - TCP Connection")
print("=" * 80)

"""
TODO: TCP connection için State Pattern kullanarak state machine oluşturun.
- ConnectionState abstract class (open, close, send, receive)
- ClosedState, ListeningState, EstablishedState sınıfları
- TCPConnection class (state tutacak)
- Her state'te farklı davranışlar göstermeli
- İnvalid state transitions hata vermeli
"""

# TODO: Kodunuzu buraya yazın

# -------- SOLUTION --------
from abc import ABC, abstractmethod
from enum import Enum

class ConnectionStatus(Enum):
    CLOSED = "closed"
    LISTENING = "listening"
    ESTABLISHED = "established"

class ConnectionState(ABC):
    """Abstract connection state"""
    @abstractmethod
    def open(self, connection):
        pass

    @abstractmethod
    def close(self, connection):
        pass

    @abstractmethod
    def send(self, connection, data: str):
        pass

    @abstractmethod
    def receive(self, connection):
        pass

class ClosedState(ConnectionState):
    """Connection is closed"""
    def open(self, connection):
        print("Opening connection...")
        connection.status = ConnectionStatus.LISTENING
        connection.set_state(connection.listening_state)

    def close(self, connection):
        print("Connection is already closed")

    def send(self, connection, data: str):
        print("ERROR: Cannot send data - connection is closed")

    def receive(self, connection):
        print("ERROR: Cannot receive data - connection is closed")

class ListeningState(ConnectionState):
    """Connection is listening for incoming connections"""
    def open(self, connection):
        print("Connection is already open and listening")

    def close(self, connection):
        print("Closing connection...")
        connection.status = ConnectionStatus.CLOSED
        connection.set_state(connection.closed_state)

    def send(self, connection, data: str):
        print("Establishing connection and sending data...")
        connection.status = ConnectionStatus.ESTABLISHED
        connection.set_state(connection.established_state)
        print(f"Sending: {data}")

    def receive(self, connection):
        print("Waiting for incoming connection...")
        connection.status = ConnectionStatus.ESTABLISHED
        connection.set_state(connection.established_state)
        print("Connection established")

class EstablishedState(ConnectionState):
    """Connection is established"""
    def open(self, connection):
        print("Connection is already established")

    def close(self, connection):
        print("Closing established connection...")
        connection.status = ConnectionStatus.CLOSED
        connection.set_state(connection.closed_state)

    def send(self, connection, data: str):
        print(f"Sending data: {data}")

    def receive(self, connection):
        print("Receiving data from established connection")
        return "Sample data"

class TCPConnection:
    """TCP Connection using state pattern"""
    def __init__(self):
        # Initialize all possible states
        self.closed_state = ClosedState()
        self.listening_state = ListeningState()
        self.established_state = EstablishedState()

        # Start in closed state
        self.state = self.closed_state
        self.status = ConnectionStatus.CLOSED

    def set_state(self, state: ConnectionState):
        """Change connection state"""
        self.state = state

    def open(self):
        """Open connection"""
        self.state.open(self)

    def close(self):
        """Close connection"""
        self.state.close(self)

    def send(self, data: str):
        """Send data"""
        self.state.send(self, data)

    def receive(self):
        """Receive data"""
        return self.state.receive(self)

    def get_status(self):
        """Get current connection status"""
        return f"Connection status: {self.status.value.upper()}"

# Test
connection = TCPConnection()
print(connection.get_status())
print()

print("--- Attempting to send without opening ---")
connection.send("Hello")
print()

print("--- Opening connection ---")
connection.open()
print(connection.get_status())
print()

print("--- Sending data (auto-establishes) ---")
connection.send("Hello, Server!")
print(connection.get_status())
print()

print("--- Receiving data ---")
data = connection.receive()
print()

print("--- Closing connection ---")
connection.close()
print(connection.get_status())
print()


# ============================================================================
# EXERCISE 9: Proxy Pattern - Lazy Loading Image Proxy
# Seviye: Medium
# Konu: Virtual Proxy with lazy loading
# ============================================================================

print("=" * 80)
print("EXERCISE 9: Proxy Pattern - Image Lazy Loading")
print("=" * 80)

"""
TODO: Büyük resimleri lazy load etmek için Proxy Pattern kullanın.
- Image interface (display, get_size)
- RealImage class (yükleme simülasyonu ile - time.sleep)
- ImageProxy class (lazy loading)
- İlk display() çağrısında yüklemeli
- Sonraki çağrılarda cache'den dönmeli
- get_size() metadata döndürmeli (yüklemeden)
"""

# TODO: Kodunuzu buraya yazın

# -------- SOLUTION --------
from abc import ABC, abstractmethod
import time

class Image(ABC):
    """Image interface"""
    @abstractmethod
    def display(self):
        pass

    @abstractmethod
    def get_size(self) -> tuple:
        pass

    @abstractmethod
    def get_filename(self) -> str:
        pass

class RealImage(Image):
    """Actual image that takes time to load"""
    def __init__(self, filename: str, width: int, height: int):
        self.filename = filename
        self.width = width
        self.height = height
        self._load_from_disk()

    def _load_from_disk(self):
        """Simulate loading image from disk (expensive operation)"""
        print(f"Loading image from disk: {self.filename}")
        print("Reading file...")
        time.sleep(1)  # Simulate slow disk I/O
        print(f"Image loaded: {self.filename} ({self.width}x{self.height})")

    def display(self):
        print(f"Displaying image: {self.filename}")

    def get_size(self) -> tuple:
        return (self.width, self.height)

    def get_filename(self) -> str:
        return self.filename

class ImageProxy(Image):
    """Proxy for lazy loading images"""
    def __init__(self, filename: str, width: int, height: int):
        self.filename = filename
        self.width = width
        self.height = height
        self._real_image = None

    def display(self):
        """Load image on first display"""
        if self._real_image is None:
            print(f"[PROXY] First access - loading image...")
            self._real_image = RealImage(self.filename, self.width, self.height)
        else:
            print(f"[PROXY] Using cached image")

        self._real_image.display()

    def get_size(self) -> tuple:
        """Return size without loading image"""
        print(f"[PROXY] Returning metadata (no loading)")
        return (self.width, self.height)

    def get_filename(self) -> str:
        """Return filename without loading image"""
        return self.filename

# Test
print("Creating image proxy (not loaded yet):")
image = ImageProxy("vacation_photo.jpg", 1920, 1080)
print()

print("Getting image metadata (no loading):")
print(f"Filename: {image.get_filename()}")
print(f"Size: {image.get_size()}")
print()

print("First display (loads from disk):")
image.display()
print()

print("Second display (uses cache):")
image.display()
print()

print("Creating multiple proxies:")
images = [
    ImageProxy("photo1.jpg", 1920, 1080),
    ImageProxy("photo2.jpg", 3840, 2160),
    ImageProxy("photo3.jpg", 1280, 720)
]

print("Displaying only photo2:")
images[1].display()
print()


# ============================================================================
# EXERCISE 10: Adapter Pattern - Legacy System Integration
# Seviye: Advanced
# Konu: Farklı API'ları birleştirme
# ============================================================================

print("=" * 80)
print("EXERCISE 10: Adapter Pattern - Legacy API Integration")
print("=" * 80)

"""
TODO: Eski ve yeni API'ları birleştiren adapter oluşturun.
- ModernPaymentAPI interface (process_payment metodu)
- LegacyPaymentSystem (make_payment, refund_payment)
- LegacyPaymentAdapter (LegacyPaymentSystem'i ModernPaymentAPI'ye uyarlar)
- PaymentResult dataclass (success, transaction_id, message)
- Farklı imzaları adapte edin
"""

# TODO: Kodunuzu buraya yazın

# -------- SOLUTION --------
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class PaymentResult:
    """Modern payment result"""
    success: bool
    transaction_id: str
    message: str
    amount: float

class ModernPaymentAPI(ABC):
    """Modern payment API interface"""
    @abstractmethod
    def process_payment(self, amount: float, currency: str,
                       customer_id: str) -> PaymentResult:
        pass

    @abstractmethod
    def refund(self, transaction_id: str) -> PaymentResult:
        pass

class LegacyPaymentSystem:
    """Old payment system with different interface"""
    def make_payment(self, customer_number: int, payment_amount: float) -> Dict:
        """Legacy payment method"""
        print(f"[LEGACY] Processing payment: Customer#{customer_number}, "
              f"Amount: ${payment_amount:.2f}")

        # Simulate legacy response format
        return {
            "status": 1,  # 1 = success, 0 = failure
            "ref_number": f"LEG_{customer_number}_{int(payment_amount)}",
            "msg": "Payment processed successfully"
        }

    def refund_payment(self, reference_number: str) -> Dict:
        """Legacy refund method"""
        print(f"[LEGACY] Processing refund: Ref#{reference_number}")

        return {
            "status": 1,
            "ref_number": reference_number,
            "msg": "Refund processed successfully"
        }

class LegacyPaymentAdapter(ModernPaymentAPI):
    """Adapter to make legacy system work with modern interface"""
    def __init__(self, legacy_system: LegacyPaymentSystem):
        self.legacy_system = legacy_system

    def process_payment(self, amount: float, currency: str,
                       customer_id: str) -> PaymentResult:
        """Adapt modern interface to legacy system"""
        print(f"[ADAPTER] Converting modern API call to legacy format")

        # Convert customer_id (string) to customer_number (int)
        customer_number = int(customer_id.replace("CUST_", ""))

        # Call legacy system
        legacy_result = self.legacy_system.make_payment(customer_number, amount)

        # Convert legacy response to modern format
        return PaymentResult(
            success=(legacy_result["status"] == 1),
            transaction_id=legacy_result["ref_number"],
            message=legacy_result["msg"],
            amount=amount
        )

    def refund(self, transaction_id: str) -> PaymentResult:
        """Adapt modern refund to legacy system"""
        print(f"[ADAPTER] Converting modern refund to legacy format")

        legacy_result = self.legacy_system.refund_payment(transaction_id)

        return PaymentResult(
            success=(legacy_result["status"] == 1),
            transaction_id=legacy_result["ref_number"],
            message=legacy_result["msg"],
            amount=0.0  # Legacy system doesn't return amount
        )

class ModernPaymentProcessor:
    """Modern payment processor expecting ModernPaymentAPI"""
    def __init__(self, payment_api: ModernPaymentAPI):
        self.payment_api = payment_api

    def charge_customer(self, amount: float, currency: str, customer_id: str):
        """Process payment using modern API"""
        print(f"\n{'='*60}")
        print(f"Processing payment: ${amount:.2f} {currency} for {customer_id}")
        print(f"{'='*60}")

        result = self.payment_api.process_payment(amount, currency, customer_id)

        if result.success:
            print(f"✓ Payment successful!")
            print(f"✓ Transaction ID: {result.transaction_id}")
            print(f"✓ Message: {result.message}")
        else:
            print(f"✗ Payment failed: {result.message}")

        return result

# Test
print("Using legacy system through adapter:")
legacy_system = LegacyPaymentSystem()
adapter = LegacyPaymentAdapter(legacy_system)

processor = ModernPaymentProcessor(adapter)
result = processor.charge_customer(99.99, "USD", "CUST_12345")
print()

# Refund
print("Processing refund:")
refund_result = adapter.refund(result.transaction_id)
print(f"Refund result: {refund_result}")
print()


# ============================================================================
# EXERCISE 11: Template Method Pattern - Data Pipeline
# Seviye: Advanced
# Konu: Template Method ile data processing pipeline
# ============================================================================

print("=" * 80)
print("EXERCISE 11: Template Method Pattern - Data Pipeline")
print("=" * 80)

"""
TODO: Farklı veri kaynaklarını işleyen bir pipeline oluşturun.
- DataPipeline abstract class (process metodu template method)
- extract_data, transform_data, validate_data, load_data metodları
- CSVPipeline, JSONPipeline, XMLPipeline sınıfları
- Her sınıf kendi formatına özgü metodları override etmeli
- validate_data() optional (default implementation)
"""

# TODO: Kodunuzu buraya yazın

# -------- SOLUTION --------
from abc import ABC, abstractmethod
from typing import List, Dict, Any
import json

class DataPipeline(ABC):
    """Template for data processing pipeline"""

    def process(self, source: str):
        """Template method - defines the algorithm skeleton"""
        print(f"\n{'='*60}")
        print(f"Starting {self.__class__.__name__} pipeline")
        print(f"{'='*60}\n")

        # Step 1: Extract
        raw_data = self.extract_data(source)

        # Step 2: Transform
        transformed_data = self.transform_data(raw_data)

        # Step 3: Validate (optional, has default implementation)
        if not self.validate_data(transformed_data):
            print("❌ Data validation failed!")
            return None

        # Step 4: Load
        result = self.load_data(transformed_data)

        print(f"\n{'='*60}")
        print(f"Pipeline completed successfully")
        print(f"{'='*60}\n")

        return result

    @abstractmethod
    def extract_data(self, source: str) -> Any:
        """Extract data from source - must be implemented"""
        pass

    @abstractmethod
    def transform_data(self, raw_data: Any) -> List[Dict]:
        """Transform data to common format - must be implemented"""
        pass

    def validate_data(self, data: List[Dict]) -> bool:
        """Validate data - default implementation"""
        print("Validating data...")
        if not data:
            return False
        print(f"✓ Validated {len(data)} records")
        return True

    @abstractmethod
    def load_data(self, data: List[Dict]) -> Dict:
        """Load data to destination - must be implemented"""
        pass

class CSVPipeline(DataPipeline):
    """Pipeline for CSV data"""

    def extract_data(self, source: str) -> str:
        """Extract CSV data"""
        print(f"Extracting data from CSV: {source}")
        # Simulating CSV data
        csv_data = "name,age,city\nJohn,25,NYC\nJane,30,LA\nBob,35,Chicago"
        print(f"✓ Extracted CSV data")
        return csv_data

    def transform_data(self, raw_data: str) -> List[Dict]:
        """Transform CSV to list of dicts"""
        print("Transforming CSV data...")
        lines = raw_data.strip().split('\n')
        headers = lines[0].split(',')

        result = []
        for line in lines[1:]:
            values = line.split(',')
            record = dict(zip(headers, values))
            result.append(record)

        print(f"✓ Transformed {len(result)} records")
        return result

    def load_data(self, data: List[Dict]) -> Dict:
        """Load CSV data"""
        print("Loading data to database...")
        print(f"✓ Loaded {len(data)} records")
        return {"status": "success", "count": len(data), "format": "CSV"}

class JSONPipeline(DataPipeline):
    """Pipeline for JSON data"""

    def extract_data(self, source: str) -> str:
        """Extract JSON data"""
        print(f"Extracting data from JSON: {source}")
        # Simulating JSON data
        json_data = '''
        {
            "users": [
                {"name": "Alice", "age": 28, "city": "Boston"},
                {"name": "Charlie", "age": 32, "city": "Seattle"}
            ]
        }
        '''
        print(f"✓ Extracted JSON data")
        return json_data

    def transform_data(self, raw_data: str) -> List[Dict]:
        """Transform JSON to list of dicts"""
        print("Transforming JSON data...")
        data = json.loads(raw_data)
        result = data.get("users", [])
        print(f"✓ Transformed {len(result)} records")
        return result

    def validate_data(self, data: List[Dict]) -> bool:
        """Custom validation for JSON data"""
        print("Validating JSON data with custom rules...")
        if not super().validate_data(data):
            return False

        # Additional JSON-specific validation
        for record in data:
            if "name" not in record or "age" not in record:
                print("❌ Missing required fields")
                return False

        print("✓ All records have required fields")
        return True

    def load_data(self, data: List[Dict]) -> Dict:
        """Load JSON data"""
        print("Loading data to API...")
        print(f"✓ Posted {len(data)} records to API")
        return {"status": "success", "count": len(data), "format": "JSON"}

class XMLPipeline(DataPipeline):
    """Pipeline for XML data"""

    def extract_data(self, source: str) -> str:
        """Extract XML data"""
        print(f"Extracting data from XML: {source}")
        # Simulating XML data
        xml_data = """
        <users>
            <user><name>Dave</name><age>29</age><city>Denver</city></user>
            <user><name>Eve</name><age>31</age><city>Miami</city></user>
        </users>
        """
        print(f"✓ Extracted XML data")
        return xml_data

    def transform_data(self, raw_data: str) -> List[Dict]:
        """Transform XML to list of dicts"""
        print("Transforming XML data...")
        # Simplified XML parsing (real implementation would use xml.etree)
        import re

        users = re.findall(r'<user>(.*?)</user>', raw_data, re.DOTALL)
        result = []

        for user in users:
            name = re.search(r'<name>(.*?)</name>', user).group(1)
            age = re.search(r'<age>(.*?)</age>', user).group(1)
            city = re.search(r'<city>(.*?)</city>', user).group(1)
            result.append({"name": name, "age": age, "city": city})

        print(f"✓ Transformed {len(result)} records")
        return result

    def load_data(self, data: List[Dict]) -> Dict:
        """Load XML data"""
        print("Loading data to file system...")
        print(f"✓ Saved {len(data)} records to file")
        return {"status": "success", "count": len(data), "format": "XML"}

# Test
csv_pipeline = CSVPipeline()
csv_result = csv_pipeline.process("users.csv")
print(f"Result: {csv_result}")

json_pipeline = JSONPipeline()
json_result = json_pipeline.process("users.json")
print(f"Result: {json_result}")

xml_pipeline = XMLPipeline()
xml_result = xml_pipeline.process("users.xml")
print(f"Result: {xml_result}")


# ============================================================================
# EXERCISE 12: Composite Pattern (BONUS)
# Seviye: Expert
# Konu: Tree structure ile component hierarchy
# ============================================================================

print("=" * 80)
print("EXERCISE 12: Composite Pattern - File System")
print("=" * 80)

"""
TODO: Dosya sistemi benzeri bir yapı oluşturun (Composite Pattern).
- FileSystemComponent abstract class (get_size, display)
- File class (leaf node)
- Directory class (composite node - içinde File ve Directory olabilir)
- add(), remove() metodları
- Recursive işlemler (total size, display hierarchy)
"""

# TODO: Kodunuzu buraya yazın

# -------- SOLUTION --------
from abc import ABC, abstractmethod
from typing import List

class FileSystemComponent(ABC):
    """Abstract component for file system"""
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def get_size(self) -> int:
        """Get size in bytes"""
        pass

    @abstractmethod
    def display(self, indent: int = 0):
        """Display component"""
        pass

class File(FileSystemComponent):
    """Leaf component - represents a file"""
    def __init__(self, name: str, size: int):
        super().__init__(name)
        self.size = size

    def get_size(self) -> int:
        return self.size

    def display(self, indent: int = 0):
        print("  " * indent + f"📄 {self.name} ({self.size} bytes)")

class Directory(FileSystemComponent):
    """Composite component - represents a directory"""
    def __init__(self, name: str):
        super().__init__(name)
        self.children: List[FileSystemComponent] = []

    def add(self, component: FileSystemComponent):
        """Add file or directory"""
        self.children.append(component)

    def remove(self, component: FileSystemComponent):
        """Remove file or directory"""
        self.children.remove(component)

    def get_size(self) -> int:
        """Get total size recursively"""
        return sum(child.get_size() for child in self.children)

    def display(self, indent: int = 0):
        """Display directory tree"""
        print("  " * indent + f"📁 {self.name}/ ({self.get_size()} bytes total)")
        for child in self.children:
            child.display(indent + 1)

# Test - Create file system structure
root = Directory("root")

# Documents directory
docs = Directory("documents")
docs.add(File("resume.pdf", 50000))
docs.add(File("cover_letter.docx", 25000))

# Photos directory
photos = Directory("photos")
photos.add(File("vacation1.jpg", 2000000))
photos.add(File("vacation2.jpg", 1800000))
photos.add(File("family.jpg", 1500000))

# Work directory with subdirectory
work = Directory("work")
work.add(File("report.xlsx", 100000))

projects = Directory("projects")
projects.add(File("project1.zip", 5000000))
projects.add(File("project2.zip", 4500000))
work.add(projects)

# Add all to root
root.add(docs)
root.add(photos)
root.add(work)
root.add(File("readme.txt", 1000))

print("File System Structure:")
print("=" * 60)
root.display()
print("=" * 60)
print(f"Total size: {root.get_size():,} bytes ({root.get_size() / 1024 / 1024:.2f} MB)")
print()


# ============================================================================
print("=" * 80)
print("TÜM EXERCISE'LAR TAMAMLANDI!")
print("=" * 80)
print("""
Design Patterns konusunda ele alınan pattern'ler:
1. ✓ Singleton Pattern (Thread-safe)
2. ✓ Factory Pattern (Notification System)
3. ✓ Builder Pattern (HTTP Request Builder)
4. ✓ Observer Pattern (Event System)
5. ✓ Strategy Pattern (Compression)
6. ✓ Decorator Pattern (Function Decorators)
7. ✓ Command Pattern (Undo/Redo)
8. ✓ State Pattern (TCP Connection)
9. ✓ Proxy Pattern (Lazy Loading)
10. ✓ Adapter Pattern (Legacy Integration)
11. ✓ Template Method Pattern (Data Pipeline)
12. ✓ Composite Pattern (File System) [BONUS]

Her pattern için:
- TODO kısmı (kendi implementasyonunuz için)
- Detaylı çözüm
- Real-world örnekler
- Test kodları

İyi çalışmalar!
""")
