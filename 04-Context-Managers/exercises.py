"""
CONTEXT MANAGERS - ADVANCED EXERCISES
=====================================

Bu dosya Context Manager'lar konusunda ileri seviye alıştırmalar içerir.
Her alıştırma gerçek dünya senaryolarını yansıtır ve production-ready
çözümler gerektirir.

Zorluk Seviyeleri:
- ORTA: Temel context manager implementasyonları
- ZOR: Karmaşık kaynak yönetimi ve exception handling
- UZMAN: Distributed sistemler ve advanced patterns

Her alıştırma:
1. TODO kısmı (çözülmesi gereken problem)
2. Detaylı açıklama ve gereksinimler
3. Test kodları
4. Tam çözüm (ÇÖZÜM bölümünde)
"""

import time
import threading
from contextlib import contextmanager
from typing import Optional, List, Dict, Any, Callable
from pathlib import Path
import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
import tempfile
import shutil


# ============================================================================
# ALIŞTIRMA 1: Timer Context Manager (ORTA)
# ============================================================================
"""
GÖREV: Kod bloklarının çalışma süresini ölçen bir context manager oluştur.

Gereksinimler:
1. with bloğunun başlangıç ve bitiş zamanını kaydet
2. Blok bitiminde süreyi yazdır
3. Exception durumunda bile süreyi ölç
4. İç içe timer'lar desteklensin (nested timing)
5. Minimum süre eşiği altındaki işlemleri rapor etme

Kullanım:
    with Timer("İşlem 1"):
        # kod

    with Timer("İşlem 2", min_threshold=0.1):
        # 0.1 saniyeden kısa sürerse rapor edilmez
"""

# TODO: Timer class'ını implement edin
class Timer:
    """Çalışma süresini ölçen context manager."""

    def __init__(self, name: str, min_threshold: float = 0.0):
        """
        Args:
            name: Timer ismi
            min_threshold: Minimum süre eşiği (saniye)
        """
        pass

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


# Test Kodu
def test_timer():
    """Timer context manager testleri."""
    print("TEST: Timer Context Manager")
    print("-" * 50)

    # Test 1: Basit kullanım
    with Timer("Test İşlemi 1"):
        time.sleep(0.05)

    # Test 2: Nested timer'lar
    with Timer("Dış İşlem"):
        time.sleep(0.02)
        with Timer("İç İşlem"):
            time.sleep(0.03)

    # Test 3: Minimum threshold
    with Timer("Hızlı İşlem", min_threshold=0.1):
        time.sleep(0.01)  # Bu rapor edilmemeli

    # Test 4: Exception durumu
    try:
        with Timer("Hatalı İşlem"):
            time.sleep(0.02)
            raise ValueError("Test hatası")
    except ValueError:
        pass

    print("\n")


# ============================================================================
# ALIŞTIRMA 2: Database Transaction Manager (ZOR)
# ============================================================================
"""
GÖREV: Otomatik commit/rollback özellikli veritabanı transaction manager.

Gereksinimler:
1. Context içinde yapılan tüm işlemler bir transaction içinde olmalı
2. Exception olursa otomatik rollback
3. Exception olmazsa otomatik commit
4. Nested transaction'lar desteklenmeli (savepoint kullanarak)
5. Transaction timeout özelliği
6. Transaction isolation level ayarlanabilmeli

Kullanım:
    with DatabaseTransaction(conn) as cursor:
        cursor.execute("INSERT ...")
        # Otomatik commit veya rollback
"""

# TODO: DatabaseTransaction class'ını implement edin
class DatabaseTransaction:
    """Otomatik commit/rollback özellikli transaction manager."""

    def __init__(self,
                 connection: sqlite3.Connection,
                 isolation_level: Optional[str] = None,
                 timeout: Optional[float] = None):
        """
        Args:
            connection: SQLite connection
            isolation_level: Transaction isolation level
            timeout: Transaction timeout (saniye)
        """
        pass

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


# Test Kodu
def test_database_transaction():
    """Database transaction manager testleri."""
    print("TEST: Database Transaction Manager")
    print("-" * 50)

    # Test veritabanı oluştur
    conn = sqlite3.connect(':memory:')
    conn.execute('''
        CREATE TABLE users (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            email TEXT UNIQUE
        )
    ''')

    # Test 1: Başarılı transaction (auto commit)
    try:
        with DatabaseTransaction(conn) as cursor:
            cursor.execute("INSERT INTO users (name, email) VALUES (?, ?)",
                         ("Ahmet", "ahmet@test.com"))
            cursor.execute("INSERT INTO users (name, email) VALUES (?, ?)",
                         ("Mehmet", "mehmet@test.com"))
    except Exception as e:
        print(f"Hata: {e}")

    # Kayıtları kontrol et
    cursor = conn.execute("SELECT COUNT(*) FROM users")
    count = cursor.fetchone()[0]
    print(f"Başarılı transaction sonrası kayıt sayısı: {count}")

    # Test 2: Başarısız transaction (auto rollback)
    try:
        with DatabaseTransaction(conn) as cursor:
            cursor.execute("INSERT INTO users (name, email) VALUES (?, ?)",
                         ("Ayşe", "ayse@test.com"))
            # Duplicate email hatası oluştur
            cursor.execute("INSERT INTO users (name, email) VALUES (?, ?)",
                         ("Fatma", "ahmet@test.com"))  # Duplicate!
    except Exception as e:
        print(f"Beklenen hata yakalandı: {type(e).__name__}")

    # Kayıtları kontrol et (rollback olmalı)
    cursor = conn.execute("SELECT COUNT(*) FROM users")
    count = cursor.fetchone()[0]
    print(f"Rollback sonrası kayıt sayısı: {count}")

    conn.close()
    print("\n")


# ============================================================================
# ALIŞTIRMA 3: Temporary Directory Manager (ORTA)
# ============================================================================
"""
GÖREV: Geçici dizin oluşturan ve otomatik temizleyen context manager.

Gereksinimler:
1. Context başında geçici dizin oluştur
2. Context bitiminde tüm içeriği temizle
3. Cleanup başarısız olsa bile exception fırlatma
4. İsteğe bağlı olarak cleanup'ı atla (debug için)
5. Oluşturulan dosyaların listesini tut
6. Maksimum dizin boyutu limiti

Kullanım:
    with TemporaryDirectory(cleanup=True) as temp_dir:
        # Geçici dosyalar oluştur
        (temp_dir / "file.txt").write_text("data")
    # Dizin otomatik temizlenir
"""

# TODO: TemporaryDirectory class'ını implement edin
class TemporaryDirectory:
    """Otomatik temizleme özellikli geçici dizin manager."""

    def __init__(self,
                 prefix: str = "tmp_",
                 cleanup: bool = True,
                 max_size_mb: Optional[float] = None):
        """
        Args:
            prefix: Dizin ismi prefix'i
            cleanup: Context sonunda temizlik yap
            max_size_mb: Maksimum dizin boyutu (MB)
        """
        pass

    def __enter__(self) -> Path:
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def get_size_mb(self) -> float:
        """Dizinin toplam boyutunu MB cinsinden döndür."""
        pass


# Test Kodu
def test_temporary_directory():
    """Temporary directory manager testleri."""
    print("TEST: Temporary Directory Manager")
    print("-" * 50)

    # Test 1: Temel kullanım
    with TemporaryDirectory(prefix="test_") as temp_dir:
        print(f"Geçici dizin: {temp_dir}")

        # Dosya oluştur
        test_file = temp_dir / "test.txt"
        test_file.write_text("Test verisi")

        # Alt dizin oluştur
        sub_dir = temp_dir / "subdir"
        sub_dir.mkdir()
        (sub_dir / "file.txt").write_text("Alt dizin dosyası")

        print(f"Dizin var mı: {temp_dir.exists()}")

    # Context sonrası dizin silinmeli
    print(f"Cleanup sonrası var mı: {temp_dir.exists()}")

    # Test 2: Cleanup=False
    temp_path = None
    with TemporaryDirectory(cleanup=False) as temp_dir:
        temp_path = temp_dir
        (temp_dir / "persistent.txt").write_text("Bu dosya kalacak")

    print(f"Cleanup=False sonrası var mı: {temp_path.exists()}")
    if temp_path.exists():
        shutil.rmtree(temp_path)  # Manuel temizlik

    print("\n")


# ============================================================================
# ALIŞTIRMA 4: Connection Pool Manager (ZOR)
# ============================================================================
"""
GÖREV: Thread-safe connection pool yöneticisi.

Gereksinimler:
1. Maksimum bağlantı sayısı limiti
2. Bağlantı timeout kontrolü
3. Bağlantı health check
4. Otomatik bağlantı yenileme (unhealthy connection'lar için)
5. Connection usage statistics
6. Graceful shutdown

Kullanım:
    pool = ConnectionPool(max_size=5)
    with pool.get_connection() as conn:
        # Bağlantıyı kullan
    # Otomatik olarak pool'a geri döner
"""

# TODO: ConnectionPool class'ını implement edin
@dataclass
class ConnectionStats:
    """Connection pool istatistikleri."""
    total_connections: int = 0
    active_connections: int = 0
    idle_connections: int = 0
    total_requests: int = 0
    failed_requests: int = 0
    avg_wait_time: float = 0.0


class MockConnection:
    """Mock database connection."""
    _id_counter = 0

    def __init__(self):
        MockConnection._id_counter += 1
        self.id = MockConnection._id_counter
        self.is_healthy = True
        self.created_at = time.time()

    def query(self, sql: str):
        """Mock query."""
        if not self.is_healthy:
            raise RuntimeError("Connection unhealthy")
        time.sleep(0.01)  # Simulate query
        return f"[Conn-{self.id}] Query result"

    def health_check(self) -> bool:
        """Bağlantı sağlığını kontrol et."""
        return self.is_healthy

    def close(self):
        """Bağlantıyı kapat."""
        self.is_healthy = False


class ConnectionPool:
    """Thread-safe connection pool."""

    def __init__(self,
                 max_size: int = 10,
                 timeout: float = 5.0,
                 health_check_interval: float = 60.0):
        """
        Args:
            max_size: Maksimum bağlantı sayısı
            timeout: Bağlantı alma timeout'u
            health_check_interval: Health check aralığı (saniye)
        """
        pass

    @contextmanager
    def get_connection(self):
        """Pool'dan bağlantı al (context manager)."""
        pass

    def get_stats(self) -> ConnectionStats:
        """Pool istatistiklerini döndür."""
        pass

    def shutdown(self):
        """Pool'u kapat ve tüm bağlantıları temizle."""
        pass


# Test Kodu
def test_connection_pool():
    """Connection pool manager testleri."""
    print("TEST: Connection Pool Manager")
    print("-" * 50)

    pool = ConnectionPool(max_size=3, timeout=2.0)

    # Test 1: Basit kullanım
    with pool.get_connection() as conn:
        result = conn.query("SELECT * FROM users")
        print(f"Query sonucu: {result}")

    # Test 2: Paralel kullanım
    def worker(pool, worker_id):
        try:
            with pool.get_connection() as conn:
                result = conn.query(f"Worker {worker_id} query")
                print(f"  {result}")
        except Exception as e:
            print(f"  Worker {worker_id} hatası: {e}")

    threads = [threading.Thread(target=worker, args=(pool, i))
               for i in range(5)]

    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # İstatistikler
    stats = pool.get_stats()
    print(f"\nPool İstatistikleri:")
    print(f"  Toplam bağlantı: {stats.total_connections}")
    print(f"  Aktif bağlantı: {stats.active_connections}")
    print(f"  Toplam istek: {stats.total_requests}")

    pool.shutdown()
    print("\n")


# ============================================================================
# ALIŞTIRMA 5: Retry Context Manager (ZOR)
# ============================================================================
"""
GÖREV: Hata durumunda otomatik retry yapan context manager.

Gereksinimler:
1. Configurable retry sayısı
2. Exponential backoff stratejisi
3. Belirli exception türlerini retry et
4. Maksimum total timeout
5. Retry callback (loglama için)
6. Circuit breaker pattern implementasyonu

Kullanım:
    with RetryContext(max_retries=3, backoff=2.0) as retry:
        # Hata durumunda otomatik retry
        unreliable_operation()
"""

# TODO: RetryContext class'ını implement edin
@dataclass
class RetryStats:
    """Retry istatistikleri."""
    total_attempts: int = 0
    successful_attempts: int = 0
    failed_attempts: int = 0
    total_retry_time: float = 0.0
    exceptions: List[Exception] = field(default_factory=list)


class RetryContext:
    """Otomatik retry özellikli context manager."""

    def __init__(self,
                 max_retries: int = 3,
                 exceptions: tuple = (Exception,),
                 backoff_factor: float = 2.0,
                 max_timeout: Optional[float] = None,
                 on_retry: Optional[Callable] = None):
        """
        Args:
            max_retries: Maksimum retry sayısı
            exceptions: Retry yapılacak exception türleri
            backoff_factor: Exponential backoff çarpanı
            max_timeout: Maksimum toplam timeout
            on_retry: Retry callback fonksiyonu
        """
        pass

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def get_stats(self) -> RetryStats:
        """Retry istatistiklerini döndür."""
        pass


# Test Kodu
def test_retry_context():
    """Retry context manager testleri."""
    print("TEST: Retry Context Manager")
    print("-" * 50)

    # Test 1: Başarısız işlem (eventually succeeds)
    attempt_count = 0

    def flaky_operation():
        nonlocal attempt_count
        attempt_count += 1
        if attempt_count < 3:
            raise ConnectionError(f"Bağlantı hatası (deneme {attempt_count})")
        return "Başarılı!"

    retry_ctx = RetryContext(
        max_retries=5,
        exceptions=(ConnectionError,),
        backoff_factor=1.5,
        on_retry=lambda attempt, e: print(f"  Retry #{attempt}: {e}")
    )

    while retry_ctx.get_stats().total_attempts < 5:
        with retry_ctx:
            result = flaky_operation()
            print(f"Sonuç: {result}")
            break

    stats = retry_ctx.get_stats()
    print(f"\nRetry İstatistikleri:")
    print(f"  Toplam deneme: {stats.total_attempts}")
    print(f"  Başarılı: {stats.successful_attempts}")
    print(f"  Başarısız: {stats.failed_attempts}")

    print("\n")


# ============================================================================
# ALIŞTIRMA 6: File Lock Manager (ORTA)
# ============================================================================
"""
GÖREV: Process-safe file locking mekanizması.

Gereksinimler:
1. Cross-platform file locking
2. Blocking ve non-blocking modlar
3. Lock timeout
4. Deadlock detection
5. Lock info (hangi process tuttuğu, ne zaman alındığı)
6. Stale lock cleanup

Kullanım:
    with FileLock("shared_file.txt", timeout=5.0) as lock:
        # Dosya güvenli bir şekilde kullanılır
"""

# TODO: FileLock class'ını implement edin
@dataclass
class LockInfo:
    """Lock bilgileri."""
    owner_pid: int
    acquired_at: datetime
    lock_file: Path


class FileLock:
    """Process-safe file lock manager."""

    def __init__(self,
                 filepath: Path,
                 timeout: float = 10.0,
                 blocking: bool = True):
        """
        Args:
            filepath: Lock'lanacak dosya
            timeout: Lock alma timeout'u
            blocking: Blocking mode
        """
        pass

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def get_lock_info(self) -> Optional[LockInfo]:
        """Mevcut lock bilgilerini döndür."""
        pass


# Test Kodu
def test_file_lock():
    """File lock manager testleri."""
    print("TEST: File Lock Manager")
    print("-" * 50)

    test_file = Path("test_lock.txt")
    test_file.write_text("Shared data")

    # Test 1: Temel locking
    try:
        with FileLock(test_file, timeout=2.0) as lock:
            print("Lock alındı")
            content = test_file.read_text()
            test_file.write_text(content + "\nYeni satır")
            time.sleep(0.1)
    finally:
        print("Lock serbest bırakıldı")

    # Test 2: Concurrent access simulation
    def worker(filepath, worker_id):
        try:
            with FileLock(filepath, timeout=1.0) as lock:
                print(f"  Worker {worker_id}: Lock alındı")
                time.sleep(0.05)
        except Exception as e:
            print(f"  Worker {worker_id}: {e}")

    threads = [threading.Thread(target=worker, args=(test_file, i))
               for i in range(3)]

    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Cleanup
    test_file.unlink()
    lock_file = Path(str(test_file) + '.lock')
    if lock_file.exists():
        lock_file.unlink()

    print("\n")


# ============================================================================
# ALIŞTIRMA 7: Resource Monitor Context Manager (ZOR)
# ============================================================================
"""
GÖREV: Sistem kaynak kullanımını izleyen context manager.

Gereksinimler:
1. CPU kullanımını izle
2. Memory kullanımını izle
3. Disk I/O izleme
4. Network I/O izleme (optional)
5. Resource limit kontrolü (alarm)
6. Performans raporu oluştur

Kullanım:
    with ResourceMonitor(cpu_limit=80, memory_limit_mb=500) as monitor:
        # Kaynak yoğun işlem
        intensive_operation()
    # Otomatik rapor
"""

# TODO: ResourceMonitor class'ını implement edin
@dataclass
class ResourceSnapshot:
    """Kaynak kullanımı snapshot'ı."""
    timestamp: float
    cpu_percent: float
    memory_mb: float
    disk_io_mb: float = 0.0


class ResourceMonitor:
    """Sistem kaynak kullanımını izleyen context manager."""

    def __init__(self,
                 cpu_limit: Optional[float] = None,
                 memory_limit_mb: Optional[float] = None,
                 sample_interval: float = 0.1):
        """
        Args:
            cpu_limit: CPU kullanım limiti (%)
            memory_limit_mb: Memory kullanım limiti (MB)
            sample_interval: Örnekleme aralığı (saniye)
        """
        pass

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def get_snapshots(self) -> List[ResourceSnapshot]:
        """Tüm snapshot'ları döndür."""
        pass

    def print_report(self):
        """Performans raporunu yazdır."""
        pass


# Test Kodu
def test_resource_monitor():
    """Resource monitor testleri."""
    print("TEST: Resource Monitor")
    print("-" * 50)

    # Test: CPU ve memory intensive işlem
    try:
        with ResourceMonitor(cpu_limit=90, memory_limit_mb=1000) as monitor:
            # CPU intensive
            result = sum(i ** 2 for i in range(1000000))

            # Memory intensive
            data = [i for i in range(100000)]

            time.sleep(0.5)
    except Exception as e:
        print(f"Resource limit hatası: {e}")

    print("\n")


# ============================================================================
# ALIŞTIRMA 8: Distributed Lock Manager (UZMAN)
# ============================================================================
"""
GÖREV: Redis tabanlı distributed lock implementasyonu.

Gereksinimler:
1. Redis ile distributed locking
2. Lock timeout ve auto-renewal
3. Lock stealing prevention
4. Graceful degradation (Redis unavailable)
5. Lock statistics ve monitoring
6. Redlock algoritması (multiple Redis instances)

Not: Bu alıştırma için Redis mock implementasyonu kullanılacak.

Kullanım:
    with DistributedLock("resource_key", redis_client) as lock:
        # Distributed sistemde güvenli işlem
"""

# TODO: DistributedLock class'ını implement edin
class MockRedis:
    """Mock Redis client."""

    def __init__(self):
        self.data: Dict[str, Any] = {}
        self.lock = threading.Lock()

    def set(self, key: str, value: str, ex: Optional[int] = None):
        """Set key with optional expiry."""
        with self.lock:
            self.data[key] = {'value': value, 'expiry': ex}
            return True

    def get(self, key: str) -> Optional[str]:
        """Get key value."""
        with self.lock:
            item = self.data.get(key)
            return item['value'] if item else None

    def delete(self, key: str):
        """Delete key."""
        with self.lock:
            self.data.pop(key, None)
            return True


class DistributedLock:
    """Redis tabanlı distributed lock."""

    def __init__(self,
                 resource_name: str,
                 redis_client,
                 timeout: float = 10.0,
                 auto_renewal: bool = True):
        """
        Args:
            resource_name: Lock edilecek kaynak adı
            redis_client: Redis client
            timeout: Lock timeout (saniye)
            auto_renewal: Otomatik lock yenileme
        """
        pass

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def acquire(self) -> bool:
        """Lock'u al."""
        pass

    def release(self):
        """Lock'u serbest bırak."""
        pass


# Test Kodu
def test_distributed_lock():
    """Distributed lock manager testleri."""
    print("TEST: Distributed Lock Manager")
    print("-" * 50)

    redis_client = MockRedis()

    # Test 1: Temel locking
    with DistributedLock("resource_1", redis_client, timeout=5.0) as lock:
        print("Distributed lock alındı")
        time.sleep(0.1)
    print("Distributed lock serbest bırakıldı")

    # Test 2: Concurrent access
    success_count = 0
    lock_obj = threading.Lock()

    def worker(redis_client, worker_id):
        nonlocal success_count
        try:
            with DistributedLock("shared_resource", redis_client, timeout=1.0):
                print(f"  Worker {worker_id}: Lock alındı")
                time.sleep(0.05)
                with lock_obj:
                    success_count += 1
        except Exception as e:
            print(f"  Worker {worker_id}: {e}")

    threads = [threading.Thread(target=worker, args=(redis_client, i))
               for i in range(5)]

    for t in threads:
        t.start()
    for t in threads:
        t.join()

    print(f"\nBaşarılı lock sayısı: {success_count}")
    print("\n")


# ============================================================================
# ALIŞTIRMA 9: Transaction Coordinator (UZMAN)
# ============================================================================
"""
GÖREV: Multi-resource transaction coordinator (2PC - Two Phase Commit).

Gereksinimler:
1. Two-Phase Commit protokolü
2. Multiple resource coordination
3. Automatic rollback on failure
4. Transaction logging
5. Recovery mechanism
6. Timeout handling

Kullanım:
    coordinator = TransactionCoordinator()
    with coordinator.transaction(db1, db2, cache) as txn:
        # Tüm resource'lar transaction içinde
"""

# TODO: TransactionCoordinator class'ını implement edin
class TransactionParticipant:
    """Transaction participant interface."""

    def __init__(self, name: str):
        self.name = name
        self.state = "idle"

    def prepare(self) -> bool:
        """Prepare phase."""
        print(f"  [{self.name}] Prepare...")
        self.state = "prepared"
        return True

    def commit(self):
        """Commit phase."""
        print(f"  [{self.name}] Commit...")
        self.state = "committed"

    def rollback(self):
        """Rollback phase."""
        print(f"  [{self.name}] Rollback...")
        self.state = "rolled_back"


@dataclass
class TransactionLog:
    """Transaction log entry."""
    transaction_id: str
    timestamp: datetime
    participants: List[str]
    state: str
    error: Optional[str] = None


class TransactionCoordinator:
    """Two-Phase Commit coordinator."""

    def __init__(self):
        """Initialize coordinator."""
        pass

    @contextmanager
    def transaction(self, *participants):
        """Coordinate transaction across multiple participants."""
        pass

    def get_transaction_log(self) -> List[TransactionLog]:
        """Transaction log'unu döndür."""
        pass


# Test Kodu
def test_transaction_coordinator():
    """Transaction coordinator testleri."""
    print("TEST: Transaction Coordinator (2PC)")
    print("-" * 50)

    coordinator = TransactionCoordinator()

    # Test 1: Başarılı transaction
    print("\nSenaryo 1: Başarılı Transaction")
    db1 = TransactionParticipant("Database-1")
    db2 = TransactionParticipant("Database-2")
    cache = TransactionParticipant("Cache")

    try:
        with coordinator.transaction(db1, db2, cache):
            print("İş mantığı çalışıyor...")
            time.sleep(0.05)
    except Exception as e:
        print(f"Transaction hatası: {e}")

    print(f"DB1 durumu: {db1.state}")
    print(f"DB2 durumu: {db2.state}")
    print(f"Cache durumu: {cache.state}")

    print("\n")


# ============================================================================
# ALIŞTIRMA 10: Circuit Breaker Context Manager (ZOR)
# ============================================================================
"""
GÖREV: Circuit breaker pattern implementasyonu.

Gereksinimler:
1. Three states: Closed, Open, Half-Open
2. Failure threshold configuration
3. Timeout ve reset mekanizması
4. Fallback fonksiyon desteği
5. State change callback'leri
6. Metrics ve monitoring

Kullanım:
    breaker = CircuitBreaker(failure_threshold=5, timeout=10.0)
    with breaker.protect() as protected:
        # Circuit breaker koruması altında işlem
        unreliable_service_call()
"""

# TODO: CircuitBreaker class'ını implement edin
from enum import Enum

class CircuitState(Enum):
    """Circuit breaker durumları."""
    CLOSED = "closed"      # Normal operasyon
    OPEN = "open"          # Hata eşiği aşıldı, istekler reddediliyor
    HALF_OPEN = "half_open"  # Test durumu


@dataclass
class CircuitBreakerMetrics:
    """Circuit breaker metrikleri."""
    state: CircuitState
    failure_count: int = 0
    success_count: int = 0
    last_failure_time: Optional[float] = None
    state_changes: int = 0


class CircuitBreaker:
    """Circuit breaker pattern implementation."""

    def __init__(self,
                 failure_threshold: int = 5,
                 timeout: float = 60.0,
                 half_open_max_calls: int = 1,
                 on_state_change: Optional[Callable] = None):
        """
        Args:
            failure_threshold: Hata eşiği (Open state için)
            timeout: Open state timeout (saniye)
            half_open_max_calls: Half-open'da kaç deneme yapılacak
            on_state_change: State değişimi callback'i
        """
        pass

    @contextmanager
    def protect(self):
        """Circuit breaker koruması."""
        pass

    def get_metrics(self) -> CircuitBreakerMetrics:
        """Metrics döndür."""
        pass

    def reset(self):
        """Circuit breaker'ı sıfırla."""
        pass


# Test Kodu
def test_circuit_breaker():
    """Circuit breaker testleri."""
    print("TEST: Circuit Breaker")
    print("-" * 50)

    def state_change_callback(old_state, new_state):
        print(f"  State değişti: {old_state.value} -> {new_state.value}")

    breaker = CircuitBreaker(
        failure_threshold=3,
        timeout=2.0,
        on_state_change=state_change_callback
    )

    # Simüle edilmiş service
    call_count = 0

    def unreliable_service():
        nonlocal call_count
        call_count += 1
        if call_count <= 5:  # İlk 5 çağrı başarısız
            raise ConnectionError(f"Service unavailable (call #{call_count})")
        return "Success!"

    # Test: Failure threshold'a ulaş
    for i in range(10):
        try:
            with breaker.protect():
                result = unreliable_service()
                print(f"Çağrı #{i+1}: {result}")
        except Exception as e:
            print(f"Çağrı #{i+1}: {type(e).__name__}")

        time.sleep(0.3)

    # Metrics
    metrics = breaker.get_metrics()
    print(f"\nCircuit Breaker Metrics:")
    print(f"  State: {metrics.state.value}")
    print(f"  Failures: {metrics.failure_count}")
    print(f"  Successes: {metrics.success_count}")
    print(f"  State changes: {metrics.state_changes}")

    print("\n")


# ============================================================================
# ALIŞTIRMA 11: Async Context Manager with Timeout (ZOR)
# ============================================================================
"""
GÖREV: Timeout özellikli async context manager.

Gereksinimler:
1. Async context manager protocol (__aenter__, __aexit__)
2. Configurable timeout
3. Graceful cancellation
4. Resource cleanup guarantee
5. Nested async context support
6. Exception propagation

Kullanım:
    async with AsyncTimeout(5.0):
        await long_running_operation()
"""

# TODO: AsyncTimeout class'ını implement edin
import asyncio

class AsyncTimeout:
    """Timeout özellikli async context manager."""

    def __init__(self, timeout: float):
        """
        Args:
            timeout: Timeout süresi (saniye)
        """
        pass

    async def __aenter__(self):
        pass

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass


class AsyncResource:
    """Async resource simülasyonu."""

    def __init__(self, name: str):
        self.name = name
        self.is_open = False

    async def __aenter__(self):
        await asyncio.sleep(0.01)
        self.is_open = True
        print(f"  {self.name} açıldı")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await asyncio.sleep(0.01)
        self.is_open = False
        print(f"  {self.name} kapatıldı")
        return False

    async def process(self):
        """Async işlem."""
        if not self.is_open:
            raise RuntimeError("Resource açık değil")
        await asyncio.sleep(0.05)
        return f"{self.name} işlendi"


# Test Kodu
async def test_async_timeout():
    """Async timeout testleri."""
    print("TEST: Async Timeout Context Manager")
    print("-" * 50)

    # Test 1: Timeout içinde tamamlanan işlem
    try:
        async with AsyncTimeout(2.0):
            async with AsyncResource("Resource-1") as res:
                result = await res.process()
                print(f"Sonuç: {result}")
    except asyncio.TimeoutError:
        print("Timeout oluştu!")

    # Test 2: Timeout aşan işlem
    try:
        async with AsyncTimeout(0.1):
            async with AsyncResource("Resource-2") as res:
                await asyncio.sleep(1.0)  # Çok uzun
                result = await res.process()
    except asyncio.TimeoutError:
        print("Beklenen timeout oluştu")

    print("\n")


# ============================================================================
# ALIŞTIRMA 12: State Machine Context Manager (UZMAN)
# ============================================================================
"""
GÖREV: State machine tabanlı context manager.

Gereksinimler:
1. State tanımları ve transition'lar
2. State validation
3. Automatic state transitions
4. State history tracking
5. Rollback on error
6. State change hooks

Kullanım:
    fsm = StateMachine(initial_state="idle")
    with fsm.transition_to("processing"):
        # İşlem yapılırken state "processing"
    # Otomatik olarak önceki state'e dön veya yeni state'e geç
"""

# TODO: StateMachine ve ilgili class'ları implement edin
@dataclass
class StateTransition:
    """State geçiş kaydı."""
    from_state: str
    to_state: str
    timestamp: datetime
    success: bool
    error: Optional[str] = None


class StateMachine:
    """State machine tabanlı context manager."""

    def __init__(self,
                 initial_state: str,
                 valid_transitions: Optional[Dict[str, List[str]]] = None,
                 on_transition: Optional[Callable] = None):
        """
        Args:
            initial_state: Başlangıç state'i
            valid_transitions: Geçerli state transition'ları
            on_transition: Transition callback'i
        """
        pass

    @contextmanager
    def transition_to(self, new_state: str, rollback: bool = True):
        """State transition context manager."""
        pass

    def get_current_state(self) -> str:
        """Mevcut state'i döndür."""
        pass

    def get_history(self) -> List[StateTransition]:
        """State geçiş geçmişini döndür."""
        pass


# Test Kodu
def test_state_machine():
    """State machine context manager testleri."""
    print("TEST: State Machine Context Manager")
    print("-" * 50)

    # Valid state transitions tanımla
    transitions = {
        "idle": ["processing", "error"],
        "processing": ["completed", "error"],
        "completed": ["idle"],
        "error": ["idle"]
    }

    def on_transition(from_state, to_state):
        print(f"  State: {from_state} -> {to_state}")

    fsm = StateMachine(
        initial_state="idle",
        valid_transitions=transitions,
        on_transition=on_transition
    )

    # Test: Başarılı işlem akışı
    print(f"Başlangıç state: {fsm.get_current_state()}")

    try:
        with fsm.transition_to("processing"):
            print("  İşlem yapılıyor...")
            time.sleep(0.1)

            with fsm.transition_to("completed"):
                print("  İşlem tamamlandı")
    except Exception as e:
        print(f"Hata: {e}")

    print(f"Final state: {fsm.get_current_state()}")

    # State history
    print("\nState Geçmişi:")
    for transition in fsm.get_history():
        status = "✓" if transition.success else "✗"
        print(f"  {status} {transition.from_state} -> {transition.to_state}")

    print("\n")


# ============================================================================
# ALIŞTIRMA 13: Profiling Context Manager (ORTA)
# ============================================================================
"""
GÖREV: Kod profillemesi yapan context manager.

Gereksinimler:
1. CPU profiling
2. Memory profiling
3. Function call statistics
4. Line-by-line profiling
5. Profile sonuçlarını kaydet
6. Visual report generation (optional)

Kullanım:
    with Profiler(profile_type="cpu") as profiler:
        # Profillenen kod
    profiler.print_stats()
"""

# TODO: Profiler class'ını implement edin
@dataclass
class ProfileStats:
    """Profiling istatistikleri."""
    duration: float
    function_calls: int
    memory_peak_mb: float
    cpu_percent: float


class Profiler:
    """Kod profiling context manager."""

    def __init__(self,
                 profile_type: str = "cpu",  # "cpu", "memory", "both"
                 output_file: Optional[Path] = None):
        """
        Args:
            profile_type: Profiling türü
            output_file: Sonuçların kaydedileceği dosya
        """
        pass

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def get_stats(self) -> ProfileStats:
        """Profiling istatistikleri."""
        pass

    def print_stats(self):
        """İstatistikleri yazdır."""
        pass


# Test Kodu
def test_profiler():
    """Profiler context manager testleri."""
    print("TEST: Profiler Context Manager")
    print("-" * 50)

    with Profiler(profile_type="both") as profiler:
        # CPU intensive
        result = sum(i ** 2 for i in range(100000))

        # Memory intensive
        data = [i for i in range(50000)]

        # Function calls
        for i in range(100):
            str(i)

    profiler.print_stats()
    print("\n")


# ============================================================================
# ALIŞTIRMA 14: Cache Context Manager (ZOR)
# ============================================================================
"""
GÖREV: Otomatik cache yönetimi yapan context manager.

Gereksinimler:
1. Context içinde cache aktif
2. Cache hit/miss tracking
3. TTL (Time To Live) desteği
4. Cache invalidation
5. Multiple cache backend desteği (memory, redis, file)
6. Cache statistics

Kullanım:
    with CacheContext(backend="memory", ttl=60) as cache:
        value = cache.get_or_compute("key", expensive_function)
"""

# TODO: CacheContext class'ını implement edin
@dataclass
class CacheStats:
    """Cache istatistikleri."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    size: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


class CacheContext:
    """Otomatik cache yönetimi context manager."""

    def __init__(self,
                 backend: str = "memory",
                 ttl: Optional[float] = None,
                 max_size: Optional[int] = None):
        """
        Args:
            backend: Cache backend ("memory", "file")
            ttl: Time to live (saniye)
            max_size: Maksimum cache boyutu
        """
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def get(self, key: str) -> Optional[Any]:
        """Cache'den değer al."""
        pass

    def set(self, key: str, value: Any):
        """Cache'e değer yaz."""
        pass

    def get_or_compute(self, key: str, compute_fn: Callable) -> Any:
        """Cache'den al veya hesapla."""
        pass

    def get_stats(self) -> CacheStats:
        """Cache istatistiklerini döndür."""
        pass


# Test Kodu
def test_cache_context():
    """Cache context manager testleri."""
    print("TEST: Cache Context Manager")
    print("-" * 50)

    def expensive_computation(x):
        """Pahalı hesaplama simülasyonu."""
        time.sleep(0.1)
        return x ** 2

    with CacheContext(backend="memory", ttl=60, max_size=100) as cache:
        # İlk çağrı: cache miss
        result1 = cache.get_or_compute("key1", lambda: expensive_computation(10))
        print(f"Sonuç 1: {result1}")

        # İkinci çağrı: cache hit
        result2 = cache.get_or_compute("key1", lambda: expensive_computation(10))
        print(f"Sonuç 2: {result2}")

        # Farklı key: cache miss
        result3 = cache.get_or_compute("key2", lambda: expensive_computation(20))
        print(f"Sonuç 3: {result3}")

        # Stats
        stats = cache.get_stats()
        print(f"\nCache İstatistikleri:")
        print(f"  Hits: {stats.hits}")
        print(f"  Misses: {stats.misses}")
        print(f"  Hit rate: {stats.hit_rate:.2%}")
        print(f"  Size: {stats.size}")

    print("\n")


# ============================================================================
# ALIŞTIRMA 15: Composite Context Manager (UZMAN)
# ============================================================================
"""
GÖREV: Birden fazla context manager'ı yöneten composite pattern.

Gereksinimler:
1. Multiple context manager'ları tek bir context'te topla
2. Parallel ve sequential execution modları
3. Partial failure handling
4. Context manager dependencies
5. Priority-based execution order
6. Rollback strategy

Kullanım:
    composite = CompositeContext()
    composite.add(timer, priority=1)
    composite.add(profiler, priority=2)
    composite.add(logger, priority=3)

    with composite:
        # Tüm context manager'lar aktif
"""

# TODO: CompositeContext class'ını implement edin
@dataclass
class ContextManagerWrapper:
    """Context manager wrapper."""
    manager: Any
    priority: int = 0
    dependencies: List[str] = field(default_factory=list)
    name: str = ""


class CompositeContext:
    """Multiple context manager coordinator."""

    def __init__(self,
                 execution_mode: str = "sequential",  # "sequential", "parallel"
                 on_failure: str = "rollback_all"):  # "rollback_all", "continue"
        """
        Args:
            execution_mode: Execution mode
            on_failure: Failure handling strategy
        """
        pass

    def add(self,
            manager: Any,
            name: str,
            priority: int = 0,
            dependencies: Optional[List[str]] = None):
        """Context manager ekle."""
        pass

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


# Test Kodu
def test_composite_context():
    """Composite context manager testleri."""
    print("TEST: Composite Context Manager")
    print("-" * 50)

    # Basit context manager'lar
    @contextmanager
    def step(name, duration=0.01):
        print(f"  [{name}] Başladı")
        try:
            yield
            time.sleep(duration)
            print(f"  [{name}] Tamamlandı")
        except Exception as e:
            print(f"  [{name}] Hata: {e}")
            raise

    # Composite oluştur
    composite = CompositeContext(execution_mode="sequential")
    composite.add(step("Database", 0.02), name="db", priority=1)
    composite.add(step("Cache", 0.01), name="cache", priority=2)
    composite.add(step("Logger", 0.01), name="logger", priority=3)

    # Kullan
    with composite:
        print("  Ana işlem çalışıyor...")
        time.sleep(0.05)

    print("\n")


# ============================================================================
# ÇÖZÜMLER
# ============================================================================
"""
Aşağıda tüm alıştırmaların detaylı çözümleri bulunmaktadır.
Önce kendiniz çözmeyi deneyin, sonra çözümleri inceleyin!
"""


# ============================================================================
# ÇÖZÜM 1: Timer Context Manager
# ============================================================================
class TimerSolution:
    """Çalışma süresini ölçen context manager - ÇÖZÜM."""

    _depth = 0  # Nested timer depth tracking

    def __init__(self, name: str, min_threshold: float = 0.0):
        self.name = name
        self.min_threshold = min_threshold
        self.start_time: Optional[float] = None
        self.elapsed: Optional[float] = None
        self.depth = 0

    def __enter__(self):
        self.depth = TimerSolution._depth
        TimerSolution._depth += 1
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.elapsed = time.perf_counter() - self.start_time
        TimerSolution._depth -= 1

        # Minimum threshold kontrolü
        if self.elapsed >= self.min_threshold:
            indent = "  " * self.depth
            status = "[HATA]" if exc_type else "[OK]"
            print(f"{indent}{status} {self.name}: {self.elapsed:.4f}s")

        return False  # Exception'ı propagate et


# ============================================================================
# ÇÖZÜM 2: Database Transaction Manager
# ============================================================================
class DatabaseTransactionSolution:
    """Otomatik commit/rollback özellikli transaction manager - ÇÖZÜM."""

    def __init__(self,
                 connection: sqlite3.Connection,
                 isolation_level: Optional[str] = None,
                 timeout: Optional[float] = None):
        self.connection = connection
        self.isolation_level = isolation_level
        self.timeout = timeout
        self.cursor: Optional[sqlite3.Cursor] = None
        self.start_time: Optional[float] = None
        self._original_isolation = None

    def __enter__(self):
        self.start_time = time.time()

        # Isolation level ayarla
        if self.isolation_level:
            self._original_isolation = self.connection.isolation_level
            self.connection.isolation_level = self.isolation_level

        # Transaction başlat
        self.connection.execute("BEGIN")
        self.cursor = self.connection.cursor()
        print("Transaction başlatıldı")

        return self.cursor

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            # Timeout kontrolü
            if self.timeout:
                elapsed = time.time() - self.start_time
                if elapsed > self.timeout:
                    raise TimeoutError(f"Transaction timeout: {elapsed:.2f}s > {self.timeout}s")

            # Exception kontrolü
            if exc_type is None:
                self.connection.commit()
                print("Transaction commit edildi")
            else:
                self.connection.rollback()
                print(f"Transaction rollback yapıldı: {exc_type.__name__}")

        finally:
            # Cursor'u kapat
            if self.cursor:
                self.cursor.close()

            # Isolation level'ı geri yükle
            if self._original_isolation is not None:
                self.connection.isolation_level = self._original_isolation

        return False


# ============================================================================
# ÇÖZÜM 3: Temporary Directory Manager
# ============================================================================
class TemporaryDirectorySolution:
    """Otomatik temizleme özellikli geçici dizin manager - ÇÖZÜM."""

    def __init__(self,
                 prefix: str = "tmp_",
                 cleanup: bool = True,
                 max_size_mb: Optional[float] = None):
        self.prefix = prefix
        self.cleanup = cleanup
        self.max_size_mb = max_size_mb
        self.path: Optional[Path] = None
        self.created_files: List[Path] = []

    def __enter__(self) -> Path:
        # Geçici dizin oluştur
        self.path = Path(tempfile.mkdtemp(prefix=self.prefix))
        print(f"Geçici dizin oluşturuldu: {self.path}")
        return self.path

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.cleanup and self.path and self.path.exists():
            try:
                # Boyut kontrolü
                size_mb = self.get_size_mb()
                if self.max_size_mb and size_mb > self.max_size_mb:
                    print(f"UYARI: Dizin boyutu limiti aşıldı: {size_mb:.2f}MB > {self.max_size_mb}MB")

                # Dizini sil
                shutil.rmtree(self.path)
                print(f"Geçici dizin temizlendi: {self.path}")

            except Exception as e:
                # Cleanup başarısız olsa bile exception fırlatma
                print(f"Cleanup hatası (görmezden gelindi): {e}")

        return False

    def get_size_mb(self) -> float:
        """Dizinin toplam boyutunu MB cinsinden döndür."""
        if not self.path or not self.path.exists():
            return 0.0

        total_size = 0
        for item in self.path.rglob('*'):
            if item.is_file():
                total_size += item.stat().st_size

        return total_size / (1024 * 1024)


# ============================================================================
# ÇÖZÜM 4: Connection Pool Manager
# ============================================================================
class ConnectionPoolSolution:
    """Thread-safe connection pool - ÇÖZÜM."""

    def __init__(self,
                 max_size: int = 10,
                 timeout: float = 5.0,
                 health_check_interval: float = 60.0):
        self.max_size = max_size
        self.timeout = timeout
        self.health_check_interval = health_check_interval

        from queue import Queue
        self.pool: Queue = Queue(maxsize=max_size)
        self.size = 0
        self.lock = threading.Lock()

        # Statistics
        self.stats = ConnectionStats()

    @contextmanager
    def get_connection(self):
        """Pool'dan bağlantı al (context manager)."""
        conn = None
        start_time = time.time()

        try:
            # Bağlantı al
            conn = self._acquire()
            wait_time = time.time() - start_time

            # İstatistikleri güncelle
            with self.lock:
                self.stats.total_requests += 1
                self.stats.active_connections += 1
                self.stats.avg_wait_time = (
                    (self.stats.avg_wait_time * (self.stats.total_requests - 1) + wait_time) /
                    self.stats.total_requests
                )

            yield conn

        except Exception as e:
            with self.lock:
                self.stats.failed_requests += 1
            raise

        finally:
            if conn:
                # Bağlantıyı geri ver
                self._release(conn)
                with self.lock:
                    self.stats.active_connections -= 1

    def _acquire(self) -> MockConnection:
        """Pool'dan bağlantı al."""
        import queue

        try:
            # Önce pool'dan almayı dene
            conn = self.pool.get(timeout=self.timeout)

            # Health check
            if not conn.health_check():
                print(f"Unhealthy connection bulundu, yenisi oluşturuluyor")
                conn = MockConnection()

            return conn

        except queue.Empty:
            # Pool boş, yeni bağlantı oluştur
            with self.lock:
                if self.size < self.max_size:
                    self.size += 1
                    self.stats.total_connections += 1
                    return MockConnection()

            raise RuntimeError("Connection pool dolu, timeout")

    def _release(self, connection: MockConnection):
        """Bağlantıyı pool'a geri ver."""
        try:
            self.pool.put(connection, block=False)
            with self.lock:
                self.stats.idle_connections += 1
        except:
            # Pool doluysa bağlantıyı kapat
            with self.lock:
                self.size -= 1
                self.stats.total_connections -= 1
            connection.close()

    def get_stats(self) -> ConnectionStats:
        """Pool istatistiklerini döndür."""
        with self.lock:
            return ConnectionStats(
                total_connections=self.stats.total_connections,
                active_connections=self.stats.active_connections,
                idle_connections=self.stats.idle_connections,
                total_requests=self.stats.total_requests,
                failed_requests=self.stats.failed_requests,
                avg_wait_time=self.stats.avg_wait_time
            )

    def shutdown(self):
        """Pool'u kapat ve tüm bağlantıları temizle."""
        print("Connection pool kapatılıyor...")
        while not self.pool.empty():
            try:
                conn = self.pool.get(block=False)
                conn.close()
            except:
                break
        print("Connection pool kapatıldı")


# ============================================================================
# ÇÖZÜM 5: Retry Context Manager
# ============================================================================
class RetryContextSolution:
    """Otomatik retry özellikli context manager - ÇÖZÜM."""

    def __init__(self,
                 max_retries: int = 3,
                 exceptions: tuple = (Exception,),
                 backoff_factor: float = 2.0,
                 max_timeout: Optional[float] = None,
                 on_retry: Optional[Callable] = None):
        self.max_retries = max_retries
        self.exceptions = exceptions
        self.backoff_factor = backoff_factor
        self.max_timeout = max_timeout
        self.on_retry = on_retry

        # State
        self.stats = RetryStats()
        self.start_time: Optional[float] = None

    def __enter__(self):
        if self.stats.total_attempts == 0:
            self.start_time = time.time()

        self.stats.total_attempts += 1
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Exception yok, başarılı
        if exc_type is None:
            self.stats.successful_attempts += 1
            return False

        # Retry yapılacak exception türü mü?
        if not issubclass(exc_type, self.exceptions):
            return False

        # Exception'ı kaydet
        self.stats.exceptions.append(exc_val)
        self.stats.failed_attempts += 1

        # Maksimum retry sayısına ulaşıldı mı?
        if self.stats.total_attempts >= self.max_retries:
            print(f"Maksimum retry sayısına ulaşıldı ({self.max_retries})")
            return False

        # Timeout kontrolü
        if self.max_timeout:
            elapsed = time.time() - self.start_time
            if elapsed >= self.max_timeout:
                print(f"Maksimum timeout aşıldı ({elapsed:.2f}s)")
                return False

        # Exponential backoff
        wait_time = (self.backoff_factor ** (self.stats.total_attempts - 1)) * 0.1
        self.stats.total_retry_time += wait_time

        # Retry callback
        if self.on_retry:
            self.on_retry(self.stats.total_attempts, exc_val)

        print(f"Retry #{self.stats.total_attempts}/{self.max_retries} "
              f"(bekleme: {wait_time:.2f}s)")
        time.sleep(wait_time)

        return True  # Exception'ı suppress et (retry)

    def get_stats(self) -> RetryStats:
        """Retry istatistiklerini döndür."""
        return self.stats


# Test fonksiyonlarını çalıştır
if __name__ == "__main__":
    print("=" * 70)
    print("CONTEXT MANAGERS - ADVANCED EXERCISES")
    print("=" * 70)
    print()

    # Her testi çalıştır
    # test_timer()
    # test_database_transaction()
    # test_temporary_directory()
    # test_connection_pool()
    # test_retry_context()
    # test_file_lock()
    # test_resource_monitor()
    # test_distributed_lock()
    # test_transaction_coordinator()
    # test_circuit_breaker()
    # asyncio.run(test_async_timeout())
    # test_state_machine()
    # test_profiler()
    # test_cache_context()
    # test_composite_context()

    print("=" * 70)
    print("Tüm testler tamamlandı!")
    print("=" * 70)
    print("\nÖNERİ: Her alıştırmayı tek tek uncomment ederek test edin.")
    print("Önce kendiniz çözmeyi deneyin, sonra ÇÖZÜM bölümüne bakın!")
