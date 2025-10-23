# Context Managers (Bağlam Yöneticileri)

## İçindekiler
1. [Context Manager Temelleri](#context-manager-temelleri)
2. [with Statement](#with-statement)
3. [__enter__ ve __exit__ Metodları](#enter-ve-exit-metodları)
4. [contextlib Modülü](#contextlib-modülü)
5. [@contextmanager Decorator](#contextmanager-decorator)
6. [Nested Context Managers](#nested-context-managers)
7. [Exception Handling](#exception-handling)
8. [Resource Management Patterns](#resource-management-patterns)
9. [Database Transaction Management](#database-transaction-management)
10. [Advanced Patterns ve Best Practices](#advanced-patterns-ve-best-practices)

---

## Context Manager Temelleri

Context Manager'lar, kaynakların (resources) güvenli bir şekilde açılması, kullanılması ve kapatılması için Python'un sağladığı güçlü bir mekanizmadır. `with` statement ile birlikte kullanılarak kaynak sızıntılarını (resource leaks) önler ve exception durumlarında bile kaynakların düzgün şekilde temizlenmesini garanti eder.

### Temel Context Manager Protokolü

```python
# Örnek 1: Basit File Context Manager
class FileManager:
    """Dosya işlemleri için özel context manager."""

    def __init__(self, filename, mode):
        self.filename = filename
        self.mode = mode
        self.file = None

    def __enter__(self):
        """Context'e girildiğinde çağrılır."""
        print(f"Dosya açılıyor: {self.filename}")
        self.file = open(self.filename, self.mode)
        return self.file

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context'ten çıkılırken çağrılır."""
        if self.file:
            print(f"Dosya kapatılıyor: {self.filename}")
            self.file.close()

        # Exception handling
        if exc_type is not None:
            print(f"Hata oluştu: {exc_type.__name__}: {exc_val}")

        # False dönerse exception propagate olur, True dönerse suppress edilir
        return False

# Kullanım
with FileManager('test.txt', 'w') as f:
    f.write("Context Manager ile dosya yazma\n")
    f.write("Otomatik kaynak yönetimi\n")
# Dosya otomatik olarak kapatılır
```

---

## with Statement

`with` statement, context manager protokolünü kullanan nesnelerle çalışır ve kaynak yönetimini otomatikleştirir.

### Örnek 2: Multiple Resource Management

```python
import threading
import time
from contextlib import contextmanager

class TimedLock:
    """Zaman aşımı özelliği olan thread lock."""

    def __init__(self, timeout=5):
        self.lock = threading.Lock()
        self.timeout = timeout

    def __enter__(self):
        acquired = self.lock.acquire(timeout=self.timeout)
        if not acquired:
            raise TimeoutError(f"Lock {self.timeout} saniyede alınamadı")
        print(f"Lock alındı (Thread: {threading.current_thread().name})")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.lock.release()
        print(f"Lock serbest bırakıldı (Thread: {threading.current_thread().name})")
        return False

# Kullanım: Thread-safe operasyon
shared_resource = []
lock = TimedLock(timeout=3)

def worker(item):
    try:
        with lock:
            shared_resource.append(item)
            time.sleep(0.1)  # Simüle edilmiş iş yükü
            print(f"İşlendi: {item}")
    except TimeoutError as e:
        print(f"Timeout hatası: {e}")

# Thread'ler oluştur
threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
for t in threads:
    t.start()
for t in threads:
    t.join()
```

### Örnek 3: Database Connection Manager

```python
import sqlite3
from typing import Optional

class DatabaseConnection:
    """Otomatik commit/rollback özellikli veritabanı bağlantısı."""

    def __init__(self, db_path: str, auto_commit: bool = True):
        self.db_path = db_path
        self.auto_commit = auto_commit
        self.connection: Optional[sqlite3.Connection] = None
        self.cursor: Optional[sqlite3.Cursor] = None

    def __enter__(self):
        self.connection = sqlite3.connect(self.db_path)
        self.connection.row_factory = sqlite3.Row  # Dict-like access
        self.cursor = self.connection.cursor()
        print(f"Veritabanı bağlantısı açıldı: {self.db_path}")
        return self.cursor

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None and self.auto_commit:
            self.connection.commit()
            print("Transaction commit edildi")
        else:
            self.connection.rollback()
            print("Transaction rollback yapıldı")

        if self.cursor:
            self.cursor.close()
        if self.connection:
            self.connection.close()
            print("Veritabanı bağlantısı kapatıldı")

        return False  # Exception'ı propagate et

# Kullanım
with DatabaseConnection(':memory:') as cursor:
    cursor.execute('''
        CREATE TABLE users (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            email TEXT UNIQUE
        )
    ''')
    cursor.execute("INSERT INTO users (name, email) VALUES (?, ?)",
                   ("Ahmet", "ahmet@example.com"))
```

---

## __enter__ ve __exit__ Metodları

Context Manager protokolünü implement etmek için bu iki metod tanımlanmalıdır.

### Örnek 4: Advanced Resource Manager

```python
import os
import tempfile
import shutil
from pathlib import Path
from typing import Optional

class TemporaryDirectory:
    """Gelişmiş geçici dizin yöneticisi."""

    def __init__(self, prefix: str = "tmp_", cleanup: bool = True):
        self.prefix = prefix
        self.cleanup = cleanup
        self.path: Optional[Path] = None
        self._original_dir: Optional[Path] = None

    def __enter__(self) -> Path:
        """Geçici dizin oluştur ve döndür."""
        self.path = Path(tempfile.mkdtemp(prefix=self.prefix))
        self._original_dir = Path.cwd()
        print(f"Geçici dizin oluşturuldu: {self.path}")
        return self.path

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Geçici dizini temizle."""
        if self.cleanup and self.path and self.path.exists():
            try:
                shutil.rmtree(self.path)
                print(f"Geçici dizin temizlendi: {self.path}")
            except Exception as e:
                print(f"Temizleme hatası: {e}")

        # Exception durumunda bile cleanup yap
        return False

# Kullanım
with TemporaryDirectory(prefix="my_temp_") as temp_dir:
    # Geçici dosyalar oluştur
    test_file = temp_dir / "test.txt"
    test_file.write_text("Geçici içerik")
    print(f"Dosya oluşturuldu: {test_file}")

    # İşlemler yapılır
    assert test_file.exists()
# Blok bitiminde tüm dosyalar ve dizin silinir
```

### Örnek 5: Performance Monitoring Context Manager

```python
import time
import functools
from typing import Callable, Any
from dataclasses import dataclass, field
from collections import defaultdict

@dataclass
class PerformanceStats:
    """Performans istatistikleri."""
    call_count: int = 0
    total_time: float = 0.0
    min_time: float = float('inf')
    max_time: float = 0.0
    errors: int = 0

    def update(self, elapsed: float, error: bool = False):
        """İstatistikleri güncelle."""
        self.call_count += 1
        self.total_time += elapsed
        self.min_time = min(self.min_time, elapsed)
        self.max_time = max(self.max_time, elapsed)
        if error:
            self.errors += 1

    @property
    def avg_time(self) -> float:
        return self.total_time / self.call_count if self.call_count > 0 else 0.0

class PerformanceMonitor:
    """Kod bloklarının performansını ölçen context manager."""

    def __init__(self, name: str = "Operation",
                 print_stats: bool = True,
                 store_history: bool = False):
        self.name = name
        self.print_stats = print_stats
        self.store_history = store_history
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.elapsed: Optional[float] = None
        self.had_error = False

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.perf_counter()
        self.elapsed = self.end_time - self.start_time
        self.had_error = exc_type is not None

        if self.print_stats:
            status = "HATA" if self.had_error else "BAŞARILI"
            print(f"[{status}] {self.name}: {self.elapsed:.4f} saniye")

        return False  # Exception'ları propagate et

# Kullanım
with PerformanceMonitor("Veri İşleme"):
    # Simüle edilmiş iş yükü
    data = [i ** 2 for i in range(1000000)]
    time.sleep(0.1)

# Nested monitoring
with PerformanceMonitor("Ana İşlem"):
    with PerformanceMonitor("Adım 1"):
        time.sleep(0.05)

    with PerformanceMonitor("Adım 2"):
        time.sleep(0.03)
```

---

## contextlib Modülü

Python'un `contextlib` modülü, context manager'lar oluşturmak için yardımcı araçlar sağlar.

### Örnek 6: @contextmanager Decorator ile Basit Context Manager

```python
from contextlib import contextmanager
import sys
from io import StringIO

@contextmanager
def suppress_stdout():
    """stdout çıktısını geçici olarak bastır."""
    original_stdout = sys.stdout
    try:
        sys.stdout = StringIO()
        yield sys.stdout
    finally:
        # Yakalanan çıktıyı geri al
        captured = sys.stdout.getvalue()
        sys.stdout = original_stdout

# Kullanım
print("Bu görünür")
with suppress_stdout() as output:
    print("Bu gizli kalır")
    print("Bu da gizli")
print("Bu tekrar görünür")
```

### Örnek 7: contextmanager ile Transaction Manager

```python
from contextlib import contextmanager
from typing import List, Callable, Any
import traceback

class TransactionError(Exception):
    """Transaction hatası."""
    pass

@contextmanager
def transaction(*managers):
    """Birden fazla context manager'ı tek bir transaction'da yönet."""
    contexts = []

    try:
        # Tüm context'leri aç
        for mgr in managers:
            ctx = mgr.__enter__()
            contexts.append((mgr, ctx))

        # Tüm context'leri yield et
        yield [ctx for _, ctx in contexts]

    except Exception as e:
        # Hata durumunda tüm işlemleri geri al
        print(f"Transaction başarısız: {e}")
        raise

    finally:
        # Tüm context'leri ters sırada kapat
        for mgr, ctx in reversed(contexts):
            try:
                mgr.__exit__(None, None, None)
            except Exception as e:
                print(f"Cleanup hatası: {e}")

# Kullanım örneği
@contextmanager
def resource(name):
    print(f"  {name} açıldı")
    try:
        yield name
    finally:
        print(f"  {name} kapatıldı")

print("Transaction başlıyor:")
with transaction(resource("DB"), resource("Cache"), resource("File")):
    print("  İşlemler yapılıyor...")
print("Transaction tamamlandı")
```

### Örnek 8: contextlib.closing

```python
from contextlib import closing
import urllib.request

# closing: close() metodu olan herhangi bir nesneyi context manager'a çevirir
with closing(urllib.request.urlopen('http://www.python.org')) as page:
    # Page otomatik olarak kapatılır
    content = page.read()
    print(f"İçerik boyutu: {len(content)} byte")
```

### Örnek 9: contextlib.suppress

```python
from contextlib import suppress
import os

# suppress: Belirli exception'ları sessizce yoksay
filename = 'non_existent_file.txt'

# Eski yöntem
try:
    os.remove(filename)
except FileNotFoundError:
    pass

# contextlib.suppress ile
with suppress(FileNotFoundError):
    os.remove(filename)

# Birden fazla exception
with suppress(FileNotFoundError, PermissionError, IsADirectoryError):
    os.remove(filename)
```

---

## @contextmanager Decorator

Generator fonksiyonları context manager'a dönüştüren dekoratör.

### Örnek 10: Advanced @contextmanager Pattern

```python
from contextlib import contextmanager
import logging
from typing import Optional, Dict, Any
import json

@contextmanager
def logging_context(name: str,
                    level: int = logging.INFO,
                    log_args: bool = True,
                    log_result: bool = True):
    """Fonksiyon çağrılarını logla."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Handler yoksa ekle
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    context_data = {'name': name, 'errors': []}

    logger.info(f"Context başlatıldı: {name}")
    try:
        yield context_data
        logger.info(f"Context başarıyla tamamlandı: {name}")
    except Exception as e:
        logger.error(f"Context hatası: {name}", exc_info=True)
        context_data['errors'].append(str(e))
        raise
    finally:
        logger.info(f"Context sonlandırıldı: {name}")

# Kullanım
with logging_context("Veri İşleme") as ctx:
    print("İşlem yapılıyor...")
    # Hata durumu test
    # raise ValueError("Test hatası")
```

### Örnek 11: Reentrant Context Manager

```python
from contextlib import contextmanager
import threading

class ReentrantContextManager:
    """Yeniden girilebilir context manager."""

    def __init__(self, name: str):
        self.name = name
        self.lock = threading.RLock()  # Reentrant lock
        self.count = 0

    def __enter__(self):
        self.lock.acquire()
        self.count += 1
        print(f"{self.name}: Enter (depth={self.count})")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.count -= 1
        print(f"{self.name}: Exit (depth={self.count})")
        self.lock.release()
        return False

# Kullanım: İç içe aynı context manager
manager = ReentrantContextManager("Reentrant")

with manager:
    print("Dış seviye")
    with manager:
        print("İç seviye")
        with manager:
            print("En iç seviye")
```

---

## Nested Context Managers

Birden fazla context manager'ı yönetmek için çeşitli teknikler.

### Örnek 12: ExitStack ile Dynamic Context Management

```python
from contextlib import ExitStack, contextmanager
import tempfile
from pathlib import Path

@contextmanager
def create_temp_file(suffix: str = '.txt'):
    """Geçici dosya oluştur."""
    with tempfile.NamedTemporaryFile(mode='w', suffix=suffix, delete=False) as f:
        filepath = Path(f.name)
        print(f"Dosya oluşturuldu: {filepath.name}")
        try:
            yield filepath
        finally:
            if filepath.exists():
                filepath.unlink()
                print(f"Dosya silindi: {filepath.name}")

# ExitStack: Dinamik sayıda context manager yönetimi
def process_multiple_files(count: int):
    """Dinamik sayıda geçici dosya ile çalış."""
    with ExitStack() as stack:
        # Dinamik olarak context manager'lar ekle
        files = [
            stack.enter_context(create_temp_file(suffix=f'_{i}.txt'))
            for i in range(count)
        ]

        # Tüm dosyalara yaz
        for i, filepath in enumerate(files):
            filepath.write_text(f"Dosya {i} içeriği")

        # İşlemler yap
        print(f"\n{count} dosya ile işlem yapılıyor...")
        for filepath in files:
            content = filepath.read_text()
            print(f"  {filepath.name}: {content}")

        # ExitStack bitiminde tüm dosyalar otomatik temizlenir
        return files

print("ExitStack Örneği:")
files = process_multiple_files(3)
print("\nİşlem tamamlandı\n")
```

### Örnek 13: AsyncExitStack (Async Context Managers)

```python
import asyncio
from contextlib import asynccontextmanager
from typing import Optional

class AsyncDatabaseConnection:
    """Asenkron veritabanı bağlantısı simülasyonu."""

    def __init__(self, db_name: str):
        self.db_name = db_name
        self.connected = False

    async def __aenter__(self):
        """Async context enter."""
        await asyncio.sleep(0.1)  # Bağlantı simülasyonu
        self.connected = True
        print(f"Async DB bağlantısı açıldı: {self.db_name}")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context exit."""
        await asyncio.sleep(0.05)  # Kapatma simülasyonu
        self.connected = False
        print(f"Async DB bağlantısı kapatıldı: {self.db_name}")
        return False

    async def query(self, sql: str):
        """Async query."""
        if not self.connected:
            raise RuntimeError("Bağlantı açık değil")
        await asyncio.sleep(0.05)
        return f"Sonuç: {sql}"

@asynccontextmanager
async def async_transaction(db_conn):
    """Async transaction manager."""
    print("Transaction başladı")
    try:
        yield db_conn
        print("Transaction commit edildi")
    except Exception as e:
        print(f"Transaction rollback: {e}")
        raise
    finally:
        print("Transaction sonlandırıldı")

# Kullanım
async def main():
    async with AsyncDatabaseConnection("users_db") as db:
        async with async_transaction(db):
            result = await db.query("SELECT * FROM users")
            print(f"Query sonucu: {result}")

# asyncio.run(main())
```

---

## Exception Handling

Context manager'larda exception yönetimi ve pattern'ler.

### Örnek 14: Custom Exception Handling

```python
from contextlib import contextmanager
from typing import Type, Tuple, Callable, Optional
import sys

class RetryContext:
    """Hata durumunda yeniden deneme özellikli context manager."""

    def __init__(self,
                 max_retries: int = 3,
                 exceptions: Tuple[Type[Exception], ...] = (Exception,),
                 on_retry: Optional[Callable] = None):
        self.max_retries = max_retries
        self.exceptions = exceptions
        self.on_retry = on_retry
        self.attempt = 0
        self.last_exception: Optional[Exception] = None

    def __enter__(self):
        self.attempt += 1
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            return False

        # Belirtilen exception türlerinden biri mi?
        if not issubclass(exc_type, self.exceptions):
            return False

        self.last_exception = exc_val

        # Maksimum deneme sayısına ulaşıldı mı?
        if self.attempt >= self.max_retries:
            print(f"Maksimum deneme sayısına ulaşıldı ({self.max_retries})")
            return False

        # Retry callback
        if self.on_retry:
            self.on_retry(self.attempt, exc_val)

        print(f"Yeniden deneme {self.attempt}/{self.max_retries}: {exc_val}")
        return True  # Exception'ı suppress et

# Kullanım
def flaky_operation():
    """Bazen başarısız olan işlem."""
    import random
    if random.random() < 0.7:  # %70 başarısızlık
        raise ConnectionError("Bağlantı hatası")
    return "Başarılı!"

retry_manager = RetryContext(
    max_retries=5,
    exceptions=(ConnectionError, TimeoutError),
    on_retry=lambda attempt, e: print(f"  -> Retry callback: Attempt {attempt}")
)

while retry_manager.attempt < retry_manager.max_retries:
    with retry_manager:
        result = flaky_operation()
        print(f"Sonuç: {result}")
        break
```

### Örnek 15: Exception Logging ve Suppression

```python
from contextlib import contextmanager, suppress
from typing import Type, Tuple
import logging
import traceback

@contextmanager
def log_and_suppress(*exceptions: Type[Exception],
                     logger: Optional[logging.Logger] = None,
                     level: int = logging.ERROR):
    """Exception'ları logla ve suppress et."""
    if logger is None:
        logger = logging.getLogger(__name__)

    try:
        yield
    except exceptions as e:
        logger.log(
            level,
            f"Exception suppressed: {type(e).__name__}: {e}",
            exc_info=True
        )
        # Exception suppress edilir (propagate olmaz)

# Kullanım
logging.basicConfig(level=logging.INFO)

with log_and_suppress(ValueError, TypeError):
    print("İşlem başlıyor...")
    raise ValueError("Bu hata loglanır ama program çökmez")
    print("Bu satır çalışmaz")

print("Program devam ediyor...")
```

---

## Resource Management Patterns

Gerçek dünya kaynak yönetimi pattern'leri.

### Örnek 16: Connection Pool Manager

```python
from contextlib import contextmanager
from queue import Queue, Empty
from typing import Optional, Generic, TypeVar, Callable
import threading
import time

T = TypeVar('T')

class ConnectionPool(Generic[T]):
    """Generic connection pool implementasyonu."""

    def __init__(self,
                 create_connection: Callable[[], T],
                 max_size: int = 10,
                 timeout: float = 5.0):
        self.create_connection = create_connection
        self.max_size = max_size
        self.timeout = timeout
        self.pool: Queue[T] = Queue(maxsize=max_size)
        self.size = 0
        self.lock = threading.Lock()

    def acquire(self) -> T:
        """Pool'dan bağlantı al."""
        try:
            # Önce pool'dan almayı dene
            return self.pool.get(timeout=self.timeout)
        except Empty:
            # Pool boşsa ve limit dolmadıysa yeni bağlantı oluştur
            with self.lock:
                if self.size < self.max_size:
                    self.size += 1
                    return self.create_connection()
            raise RuntimeError("Connection pool dolu")

    def release(self, connection: T):
        """Bağlantıyı pool'a geri ver."""
        try:
            self.pool.put(connection, block=False)
        except:
            # Pool doluysa bağlantıyı kapat
            with self.lock:
                self.size -= 1

    @contextmanager
    def connection(self):
        """Context manager ile bağlantı yönetimi."""
        conn = self.acquire()
        try:
            yield conn
        finally:
            self.release(conn)

# Mock database connection
class MockDBConnection:
    _counter = 0

    def __init__(self):
        MockDBConnection._counter += 1
        self.id = MockDBConnection._counter
        print(f"  DB Connection {self.id} oluşturuldu")

    def query(self, sql: str):
        time.sleep(0.01)
        return f"Connection {self.id}: {sql} sonuçları"

# Kullanım
pool = ConnectionPool(MockDBConnection, max_size=3)

print("Connection pool kullanımı:")
with pool.connection() as conn:
    result = conn.query("SELECT * FROM users")
    print(f"  {result}")

# Paralel kullanım
def worker(pool, worker_id):
    with pool.connection() as conn:
        result = conn.query(f"Worker {worker_id} query")
        print(f"  {result}")

threads = [threading.Thread(target=worker, args=(pool, i)) for i in range(5)]
for t in threads:
    t.start()
for t in threads:
    t.join()
```

### Örnek 17: File Lock Manager

```python
import fcntl
import os
from contextlib import contextmanager
from pathlib import Path

class FileLock:
    """Cross-platform file locking."""

    def __init__(self, filepath: Path, timeout: float = 10.0):
        self.filepath = filepath
        self.timeout = timeout
        self.lock_file = None
        self.lock_filepath = Path(str(filepath) + '.lock')

    def __enter__(self):
        """Lock'u al."""
        self.lock_file = open(self.lock_filepath, 'w')

        try:
            # Non-blocking lock attempt
            fcntl.flock(self.lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            print(f"File lock alındı: {self.filepath.name}")
        except IOError:
            raise RuntimeError(
                f"Dosya kilitli: {self.filepath.name}. "
                f"Başka bir process kullanıyor."
            )

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Lock'u serbest bırak."""
        if self.lock_file:
            fcntl.flock(self.lock_file.fileno(), fcntl.LOCK_UN)
            self.lock_file.close()

            # Lock dosyasını temizle
            if self.lock_filepath.exists():
                self.lock_filepath.unlink()

            print(f"File lock serbest bırakıldı: {self.filepath.name}")

        return False

# Kullanım
@contextmanager
def safe_file_write(filepath: Path):
    """Thread-safe ve process-safe dosya yazma."""
    with FileLock(filepath):
        with open(filepath, 'a') as f:
            yield f

# Test
test_file = Path("shared_file.txt")
with safe_file_write(test_file) as f:
    f.write("Thread-safe yazma\n")
```

---

## Database Transaction Management

Veritabanı transaction'ları için advanced pattern'ler.

### Örnek 18: Savepoint Transaction Manager

```python
from contextlib import contextmanager
from typing import Optional, List
import sqlite3
from dataclasses import dataclass
from enum import Enum

class TransactionState(Enum):
    """Transaction durumları."""
    IDLE = "idle"
    ACTIVE = "active"
    COMMITTED = "committed"
    ROLLED_BACK = "rolled_back"

@dataclass
class Savepoint:
    """Transaction savepoint."""
    name: str
    state: TransactionState = TransactionState.ACTIVE

class TransactionManager:
    """Savepoint destekli gelişmiş transaction yönetimi."""

    def __init__(self, connection: sqlite3.Connection):
        self.connection = connection
        self.state = TransactionState.IDLE
        self.savepoints: List[Savepoint] = []
        self._savepoint_counter = 0

    def __enter__(self):
        """Transaction başlat."""
        if self.state != TransactionState.IDLE:
            raise RuntimeError(f"Transaction zaten aktif: {self.state}")

        self.connection.execute("BEGIN")
        self.state = TransactionState.ACTIVE
        print("Transaction başlatıldı")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Transaction'ı commit veya rollback yap."""
        try:
            if exc_type is None and self.state == TransactionState.ACTIVE:
                self.connection.commit()
                self.state = TransactionState.COMMITTED
                print("Transaction commit edildi")
            else:
                self.connection.rollback()
                self.state = TransactionState.ROLLED_BACK
                print("Transaction rollback yapıldı")
        finally:
            self.savepoints.clear()
            self.state = TransactionState.IDLE

        return False

    @contextmanager
    def savepoint(self, name: Optional[str] = None):
        """Savepoint oluştur ve yönet."""
        if self.state != TransactionState.ACTIVE:
            raise RuntimeError("Transaction aktif değil")

        if name is None:
            self._savepoint_counter += 1
            name = f"sp_{self._savepoint_counter}"

        # Savepoint oluştur
        self.connection.execute(f"SAVEPOINT {name}")
        sp = Savepoint(name=name)
        self.savepoints.append(sp)
        print(f"  Savepoint oluşturuldu: {name}")

        try:
            yield sp
            print(f"  Savepoint başarılı: {name}")
        except Exception as e:
            # Savepoint'e rollback
            self.connection.execute(f"ROLLBACK TO SAVEPOINT {name}")
            sp.state = TransactionState.ROLLED_BACK
            print(f"  Savepoint rollback: {name}")
            raise
        finally:
            # Savepoint'i serbest bırak
            self.connection.execute(f"RELEASE SAVEPOINT {name}")
            self.savepoints.remove(sp)

# Kullanım örneği
def demo_transaction_with_savepoints():
    """Savepoint'li transaction örneği."""
    conn = sqlite3.connect(':memory:')

    # Tablo oluştur
    conn.execute('''
        CREATE TABLE accounts (
            id INTEGER PRIMARY KEY,
            name TEXT,
            balance REAL
        )
    ''')
    conn.commit()

    # Transaction manager kullan
    tm = TransactionManager(conn)

    with tm:
        # İlk hesap
        conn.execute(
            "INSERT INTO accounts (name, balance) VALUES (?, ?)",
            ("Alice", 1000.0)
        )

        with tm.savepoint("first_insert"):
            # İkinci hesap
            conn.execute(
                "INSERT INTO accounts (name, balance) VALUES (?, ?)",
                ("Bob", 500.0)
            )

        # Bu savepoint başarısız olacak
        try:
            with tm.savepoint("failing_insert"):
                conn.execute(
                    "INSERT INTO accounts (name, balance) VALUES (?, ?)",
                    ("Charlie", -100.0)  # Negatif bakiye
                )
                # Kontrol: negatif bakiye kabul edilemez
                cursor = conn.execute(
                    "SELECT balance FROM accounts WHERE name = ?",
                    ("Charlie",)
                )
                balance = cursor.fetchone()[0]
                if balance < 0:
                    raise ValueError("Bakiye negatif olamaz")
        except ValueError as e:
            print(f"  Hata yakalandı: {e}")

        # Final insert
        conn.execute(
            "INSERT INTO accounts (name, balance) VALUES (?, ?)",
            ("David", 750.0)
        )

    # Sonuçları kontrol et
    cursor = conn.execute("SELECT * FROM accounts ORDER BY id")
    print("\nFinal hesaplar:")
    for row in cursor:
        print(f"  {row}")

    conn.close()

print("Transaction Savepoint Örneği:")
demo_transaction_with_savepoints()
```

### Örnek 19: Distributed Transaction Coordinator

```python
from contextlib import contextmanager
from typing import List, Protocol, Optional
from dataclasses import dataclass
from enum import Enum
import time

class TransactionParticipant(Protocol):
    """Transaction participant interface."""

    def prepare(self) -> bool:
        """Prepare phase - transaction hazır mı?"""
        ...

    def commit(self):
        """Commit phase - değişiklikleri kalıcı yap."""
        ...

    def rollback(self):
        """Rollback phase - değişiklikleri geri al."""
        ...

class MockDatabase:
    """Mock database transaction participant."""

    def __init__(self, name: str, fail_prepare: bool = False):
        self.name = name
        self.fail_prepare = fail_prepare
        self.state = "idle"
        self.operations = []

    def prepare(self) -> bool:
        """Prepare phase."""
        print(f"  [{self.name}] Prepare phase...")
        time.sleep(0.01)
        if self.fail_prepare:
            print(f"  [{self.name}] Prepare BAŞARISIZ!")
            return False
        self.state = "prepared"
        print(f"  [{self.name}] Prepare başarılı")
        return True

    def commit(self):
        """Commit phase."""
        print(f"  [{self.name}] Commit phase...")
        time.sleep(0.01)
        self.state = "committed"
        print(f"  [{self.name}] Commit başarılı")

    def rollback(self):
        """Rollback phase."""
        print(f"  [{self.name}] Rollback phase...")
        time.sleep(0.01)
        self.state = "rolled_back"
        print(f"  [{self.name}] Rollback tamamlandı")

class TwoPhaseCommit:
    """Two-Phase Commit (2PC) protocol implementation."""

    def __init__(self, participants: List[TransactionParticipant]):
        self.participants = participants
        self.prepared: List[TransactionParticipant] = []

    def __enter__(self):
        """Transaction başlat."""
        print("2PC Transaction başlatıldı")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Transaction'ı tamamla (2PC protocol)."""
        try:
            if exc_type is not None:
                print(f"Exception oluştu, rollback yapılıyor: {exc_val}")
                self._rollback_all()
                return False

            # Phase 1: Prepare
            print("\nPhase 1: PREPARE")
            for participant in self.participants:
                if not participant.prepare():
                    print("Prepare phase başarısız, rollback yapılıyor")
                    self._rollback_all()
                    raise RuntimeError("Prepare phase başarısız")
                self.prepared.append(participant)

            # Phase 2: Commit
            print("\nPhase 2: COMMIT")
            for participant in self.participants:
                participant.commit()

            print("\n2PC Transaction başarıyla tamamlandı")
            return False

        except Exception as e:
            print(f"\n2PC Transaction başarısız: {e}")
            self._rollback_all()
            raise

    def _rollback_all(self):
        """Tüm prepared participant'ları rollback yap."""
        if self.prepared:
            print("\nROLLBACK tüm participant'lar:")
            for participant in self.prepared:
                participant.rollback()

# Kullanım
def demo_two_phase_commit(fail_scenario: bool = False):
    """2PC protocol demonstration."""
    print(f"\n{'='*60}")
    print(f"Senaryo: {'BAŞARISIZ' if fail_scenario else 'BAŞARILI'} Transaction")
    print(f"{'='*60}\n")

    # Participant'lar oluştur
    participants = [
        MockDatabase("Database-1"),
        MockDatabase("Database-2", fail_prepare=fail_scenario),
        MockDatabase("Database-3"),
    ]

    try:
        with TwoPhaseCommit(participants):
            # İş mantığı
            print("İş mantığı çalışıyor...")
            time.sleep(0.05)
            print("İş mantığı tamamlandı")
    except Exception as e:
        print(f"\nTransaction hatası yakalandı: {e}")

# Test
demo_two_phase_commit(fail_scenario=False)
demo_two_phase_commit(fail_scenario=True)
```

---

## Advanced Patterns ve Best Practices

### Örnek 20: Composable Context Managers

```python
from contextlib import contextmanager, ExitStack
from typing import List, Any, Callable
import time

@contextmanager
def timer(name: str):
    """Basit timer context manager."""
    start = time.perf_counter()
    print(f"[{name}] Başladı")
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        print(f"[{name}] Tamamlandı: {elapsed:.4f}s")

@contextmanager
def profiler(name: str):
    """Basit profiler."""
    print(f"[PROFILER] {name} - Profiling başladı")
    try:
        yield
    finally:
        print(f"[PROFILER] {name} - Profiling bitti")

@contextmanager
def logger(name: str):
    """Basit logger."""
    print(f"[LOGGER] {name} - Logging başladı")
    try:
        yield
    finally:
        print(f"[LOGGER] {name} - Logging bitti")

def compose_contexts(*context_managers):
    """Context manager'ları compose et."""
    @contextmanager
    def composed():
        with ExitStack() as stack:
            for cm in context_managers:
                stack.enter_context(cm)
            yield
    return composed()

# Kullanım: Context manager'ları compose et
with compose_contexts(
    timer("Ana İşlem"),
    profiler("Ana İşlem"),
    logger("Ana İşlem")
):
    print("  İşlem çalışıyor...")
    time.sleep(0.1)
    print("  İşlem tamamlandı")
```

---

## Best Practices ve İpuçları

### 1. Her Zaman __exit__'te Cleanup Yap
```python
# İYİ: Exception durumunda bile cleanup
class Resource:
    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            self.cleanup()
        except Exception as e:
            logging.error(f"Cleanup hatası: {e}")
        return False  # Exception'ı propagate et
```

### 2. @contextmanager'da Try-Finally Kullan
```python
# İYİ: Finally ile cleanup garantisi
@contextmanager
def managed_resource():
    resource = acquire_resource()
    try:
        yield resource
    finally:
        release_resource(resource)
```

### 3. Exception Suppression Dikkatli Kullan
```python
# KÖTÜ: Tüm exception'ları gizleme
def __exit__(self, exc_type, exc_val, exc_tb):
    return True  # Tehlikeli!

# İYİ: Sadece beklenen exception'ları suppress et
def __exit__(self, exc_type, exc_val, exc_tb):
    if exc_type in (SpecificError, AnotherError):
        handle_specific_error(exc_val)
        return True
    return False
```

### 4. Reentrant Context Manager'ları Kullan
```python
# İYİ: Nested kullanım için reentrant lock
class ReentrantContext:
    def __init__(self):
        self.lock = threading.RLock()  # Reentrant

    def __enter__(self):
        self.lock.acquire()
        return self

    def __exit__(self, *args):
        self.lock.release()
```

### 5. Type Hints Kullan
```python
from typing import Optional, IO
from types import TracebackType

class TypedContext:
    def __enter__(self) -> 'TypedContext':
        return self

    def __exit__(
        self,
        exc_type: Optional[type],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType]
    ) -> bool:
        return False
```

---

## Özet

Context Manager'lar Python'da kaynak yönetimi için kritik öneme sahiptir:

1. **Otomatik Cleanup**: Kaynaklar her durumda temizlenir
2. **Exception Safety**: Hata durumlarında bile güvenli
3. **Okunabilir Kod**: `with` statement ile açık ve net
4. **Reusable**: Bir kez yaz, her yerde kullan
5. **Composable**: Birleştirilebilir ve genişletilebilir

**Production'da dikkat edilmesi gerekenler:**
- Exception handling'i dikkatlice yap
- Cleanup işlemlerini asla atlama
- Thread-safety düşün
- Resource leak'leri önle
- Logging ve monitoring ekle
- Timeout mekanizmaları kullan
- Test et (özellikle exception durumlarını)

Context Manager'lar, profesyonel Python kodunun temel taşlarından biridir!
