"""
ASYNC PROGRAMMING - İLERİ SEVİYE ALIŞTIRMALAR
============================================

Bu dosya, async programming konusunda ileri seviye alıştırmalar içerir.
Her alıştırma gerçek dünya senaryolarını simüle eder.

Konular:
- Async API istekleri ve paralel işleme
- WebSocket bağlantıları
- Async veritabanı işlemleri
- Concurrent file operations
- Rate limiting ve throttling
- Circuit breaker pattern
- Producer-Consumer pattern
- Async context managers
- Error handling ve retry logic
- Production patterns
"""

import asyncio
import aiohttp
import aiofiles
from typing import List, Dict, Optional, Any
import time
import random
from dataclasses import dataclass
from enum import Enum
import json


# ============================================================================
# ALIŞTIRMA 1: Paralel API İstekleri ve Veri Toplama (MEDIUM)
# ============================================================================
"""
TODO: Birden fazla API endpoint'inden paralel olarak veri çekin ve birleştirin.
- Her endpoint'e async HTTP GET isteği gönderin
- Tüm istekleri paralel çalıştırın
- Hata durumlarını handle edin (return_exceptions kullanın)
- Sonuçları birleştirip döndürün
- Toplam çalışma süresini ölçün

Test URL'leri:
- https://jsonplaceholder.typicode.com/posts/1
- https://jsonplaceholder.typicode.com/posts/2
- https://jsonplaceholder.typicode.com/posts/3
"""

async def fetch_multiple_apis(urls: List[str]) -> Dict[str, Any]:
    """
    Birden fazla API'den paralel veri çekme

    Args:
        urls: API URL listesi

    Returns:
        Dict containing success_count, error_count, data, duration
    """
    # TODO: Implementasyonu yapın
    pass


# ÇÖZÜM:
async def fetch_multiple_apis_solution(urls: List[str]) -> Dict[str, Any]:
    """
    Birden fazla API'den paralel veri çekme

    Args:
        urls: API URL listesi

    Returns:
        Dict containing success_count, error_count, data, duration
    """
    start_time = time.time()

    async def fetch_url(session: aiohttp.ClientSession, url: str) -> Dict:
        """Tek bir URL'den veri çek"""
        try:
            async with session.get(url, timeout=10) as response:
                data = await response.json()
                return {
                    "url": url,
                    "status": response.status,
                    "data": data,
                    "success": True
                }
        except Exception as e:
            return {
                "url": url,
                "error": str(e),
                "success": False
            }

    # Tüm istekleri paralel gönder
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_url(session, url) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    # Sonuçları analiz et
    successful = [r for r in results if not isinstance(r, Exception) and r.get("success")]
    failed = [r for r in results if isinstance(r, Exception) or not r.get("success")]

    duration = time.time() - start_time

    return {
        "success_count": len(successful),
        "error_count": len(failed),
        "data": successful,
        "errors": failed,
        "duration": duration,
        "total_requests": len(urls)
    }


# ============================================================================
# ALIŞTIRMA 2: Async Rate Limiter (MEDIUM-HARD)
# ============================================================================
"""
TODO: Saniyede maksimum N istek gönderen bir rate limiter implementasyonu yapın.
- Context manager olarak çalışmalı (async with)
- Sliding window algoritması kullanın
- Thread-safe olmalı (asyncio.Lock)
- Aşıldığında otomatik bekleme yapmalı
"""

class AsyncRateLimiter:
    """
    Asenkron rate limiter

    Args:
        max_calls: Zaman penceresi içinde maksimum çağrı sayısı
        time_window: Zaman penceresi (saniye)
    """

    def __init__(self, max_calls: int, time_window: float):
        # TODO: Implementasyonu yapın
        pass

    async def __aenter__(self):
        # TODO: Rate limiting logic
        pass

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        # TODO: Cleanup
        pass


# ÇÖZÜM:
from collections import deque

class AsyncRateLimiter_Solution:
    """Asenkron rate limiter"""

    def __init__(self, max_calls: int, time_window: float):
        self.max_calls = max_calls
        self.time_window = time_window
        self.calls = deque()  # Çağrı zamanlarını sakla
        self.lock = asyncio.Lock()

    async def __aenter__(self):
        """Rate limiter ile korunan bölgeye gir"""
        async with self.lock:
            now = time.time()

            # Eski çağrıları temizle (sliding window)
            while self.calls and self.calls[0] < now - self.time_window:
                self.calls.popleft()

            # Limit aşılmış mı?
            if len(self.calls) >= self.max_calls:
                # En eski çağrının zaman penceresi dolana kadar bekle
                sleep_time = self.calls[0] + self.time_window - now
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)

                    # Tekrar temizle
                    now = time.time()
                    while self.calls and self.calls[0] < now - self.time_window:
                        self.calls.popleft()

            # Yeni çağrıyı kaydet
            self.calls.append(time.time())

        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return False


async def test_rate_limiter():
    """Rate limiter testi"""
    limiter = AsyncRateLimiter_Solution(max_calls=3, time_window=2.0)

    async def api_call(call_id: int):
        async with limiter:
            print(f"[{time.strftime('%H:%M:%S')}] API call {call_id}")
            await asyncio.sleep(0.1)

    # 10 çağrı yap (rate limit devreye girer)
    start = time.time()
    await asyncio.gather(*[api_call(i) for i in range(10)])
    print(f"Toplam süre: {time.time() - start:.2f}s")


# ============================================================================
# ALIŞTIRMA 3: Circuit Breaker Pattern (HARD)
# ============================================================================
"""
TODO: Circuit breaker pattern implementasyonu yapın.
- CLOSED: Normal çalışma
- OPEN: Hata threshold aşıldı, istekleri engelle
- HALF_OPEN: Recovery timeout sonrası test durumu
- Başarılı test sonrası CLOSED'a dön
- Başarısız test OPEN'da kal
"""

class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """
    Circuit breaker pattern

    Args:
        failure_threshold: Kaç hatadan sonra circuit açılsın
        recovery_timeout: Circuit açıldıktan kaç saniye sonra test edilsin
        expected_exception: Hangi exception'ları say
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: type = Exception
    ):
        # TODO: Implementasyonu yapın
        pass

    async def call(self, func, *args, **kwargs):
        """
        Korumalı fonksiyon çağrısı

        Returns:
            Fonksiyon sonucu

        Raises:
            Exception: Circuit OPEN ise veya fonksiyon hata verirse
        """
        # TODO: Implementasyonu yapın
        pass


# ÇÖZÜM:
class CircuitBreaker_Solution:
    """Circuit breaker pattern implementation"""

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: type = Exception
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception

        self.failure_count = 0
        self.last_failure_time: Optional[float] = None
        self.state = CircuitState.CLOSED
        self.lock = asyncio.Lock()

    async def call(self, func, *args, **kwargs):
        """Korumalı fonksiyon çağrısı"""
        async with self.lock:
            # Circuit OPEN mı?
            if self.state == CircuitState.OPEN:
                # Recovery timeout doldu mu?
                if (time.time() - self.last_failure_time) >= self.recovery_timeout:
                    self.state = CircuitState.HALF_OPEN
                    print(f"Circuit HALF_OPEN (test durumu)")
                else:
                    raise Exception(
                        f"Circuit OPEN! "
                        f"{self.recovery_timeout - (time.time() - self.last_failure_time):.1f}s sonra denenebilir."
                    )

        # Fonksiyonu çağır
        try:
            result = await func(*args, **kwargs)

            # Başarılı - circuit'i sıfırla
            async with self.lock:
                if self.state == CircuitState.HALF_OPEN:
                    print("Circuit CLOSED (başarılı test)")
                    self.state = CircuitState.CLOSED
                    self.failure_count = 0

            return result

        except self.expected_exception as e:
            # Hata - sayacı artır
            async with self.lock:
                self.failure_count += 1
                self.last_failure_time = time.time()

                print(f"Hata #{self.failure_count}: {e}")

                # Threshold aşıldı mı?
                if self.failure_count >= self.failure_threshold:
                    self.state = CircuitState.OPEN
                    print(f"Circuit OPEN! (threshold: {self.failure_threshold})")
                elif self.state == CircuitState.HALF_OPEN:
                    # Test başarısız - tekrar OPEN
                    self.state = CircuitState.OPEN
                    print("Circuit tekrar OPEN (test başarısız)")

            raise


async def test_circuit_breaker():
    """Circuit breaker testi"""
    circuit = CircuitBreaker_Solution(
        failure_threshold=3,
        recovery_timeout=5.0
    )

    # Hatalı servis simülasyonu
    failure_mode = True

    async def unreliable_service():
        if failure_mode:
            raise ConnectionError("Service unavailable!")
        return "Success"

    # Test senaryosu
    for i in range(10):
        try:
            result = await circuit.call(unreliable_service)
            print(f"Call {i+1}: {result}")
        except Exception as e:
            print(f"Call {i+1} failed: {e}")

        await asyncio.sleep(1)

        # 6. çağrıdan sonra servisi düzelt
        if i == 5:
            failure_mode = False
            print("\n>>> Service recovered <<<\n")


# ============================================================================
# ALIŞTIRMA 4: Async Producer-Consumer Queue (MEDIUM)
# ============================================================================
"""
TODO: Producer-Consumer pattern implementasyonu yapın.
- Birden fazla producer farklı hızlarda veri üretsin
- Birden fazla consumer farklı hızlarda veri tüketsin
- Queue size limit olsun
- Tüm producer'lar bitince consumer'lar graceful shutdown yapmalı
"""

async def producer(queue: asyncio.Queue, producer_id: int, item_count: int):
    """
    Veri üreten coroutine

    Args:
        queue: Asyncio queue
        producer_id: Producer ID
        item_count: Kaç item üretilecek
    """
    # TODO: Implementasyonu yapın
    pass


async def consumer(queue: asyncio.Queue, consumer_id: int):
    """
    Veri tüketen coroutine

    Args:
        queue: Asyncio queue
        consumer_id: Consumer ID
    """
    # TODO: Implementasyonu yapın
    pass


async def run_producer_consumer(producer_count: int, consumer_count: int, items_per_producer: int):
    """Producer-Consumer pattern çalıştır"""
    # TODO: Implementasyonu yapın
    pass


# ÇÖZÜM:
async def producer_solution(queue: asyncio.Queue, producer_id: int, item_count: int):
    """Veri üreten coroutine"""
    for i in range(item_count):
        item = f"P{producer_id}-Item{i}"
        await asyncio.sleep(random.uniform(0.1, 0.5))  # Üretim simülasyonu
        await queue.put(item)
        print(f"[Producer-{producer_id}] Üretti: {item} (queue size: {queue.qsize()})")

    print(f"[Producer-{producer_id}] Tamamlandı")


async def consumer_solution(queue: asyncio.Queue, consumer_id: int):
    """Veri tüketen coroutine"""
    while True:
        item = await queue.get()

        # Poison pill kontrolü
        if item is None:
            # Diğer consumer'lar için tekrar koy
            await queue.put(None)
            queue.task_done()
            break

        # İşleme simülasyonu
        await asyncio.sleep(random.uniform(0.2, 0.8))
        print(f"[Consumer-{consumer_id}] İşledi: {item}")
        queue.task_done()

    print(f"[Consumer-{consumer_id}] Durdu")


async def run_producer_consumer_solution(
    producer_count: int,
    consumer_count: int,
    items_per_producer: int
):
    """Producer-Consumer pattern çalıştır"""
    queue = asyncio.Queue(maxsize=10)

    # Producer'ları başlat
    producers = [
        asyncio.create_task(producer_solution(queue, i, items_per_producer))
        for i in range(producer_count)
    ]

    # Consumer'ları başlat
    consumers = [
        asyncio.create_task(consumer_solution(queue, i))
        for i in range(consumer_count)
    ]

    # Producer'ların bitmesini bekle
    await asyncio.gather(*producers)
    print("\nTüm producer'lar tamamlandı")

    # Queue'nun boşalmasını bekle
    await queue.join()
    print("Queue boşaldı")

    # Consumer'ları durdur (poison pill)
    await queue.put(None)

    # Consumer'ların bitmesini bekle
    await asyncio.gather(*consumers)
    print("Tüm consumer'lar durdu")


# ============================================================================
# ALIŞTIRMA 5: Async File Processor (MEDIUM)
# ============================================================================
"""
TODO: Birden fazla dosyayı paralel olarak işleyin.
- Dosyaları async okuma
- Her satırda transformation yapma
- Sonuçları async yazma
- Hata durumlarını handle etme
"""

async def process_file_async(
    input_file: str,
    output_file: str,
    transform_func
) -> Dict[str, Any]:
    """
    Dosyayı async işle

    Args:
        input_file: Girdi dosyası
        output_file: Çıktı dosyası
        transform_func: Satır transformation fonksiyonu

    Returns:
        Dict containing lines_processed, duration, etc.
    """
    # TODO: Implementasyonu yapın
    pass


# ÇÖZÜM:
async def process_file_async_solution(
    input_file: str,
    output_file: str,
    transform_func
) -> Dict[str, Any]:
    """Dosyayı async işle"""
    start_time = time.time()
    lines_processed = 0

    try:
        # Dosyayı aç (async)
        async with aiofiles.open(input_file, 'r') as infile:
            async with aiofiles.open(output_file, 'w') as outfile:
                # Satır satır işle
                async for line in infile:
                    # Transformation yap
                    transformed = transform_func(line.strip())

                    # Yaz
                    await outfile.write(transformed + '\n')
                    lines_processed += 1

                    # Async operation simülasyonu
                    await asyncio.sleep(0.01)

        return {
            "success": True,
            "lines_processed": lines_processed,
            "duration": time.time() - start_time,
            "input_file": input_file,
            "output_file": output_file
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "lines_processed": lines_processed,
            "duration": time.time() - start_time
        }


async def process_multiple_files(file_pairs: List[tuple], transform_func) -> List[Dict]:
    """
    Birden fazla dosyayı paralel işle

    Args:
        file_pairs: [(input, output), ...] listesi
        transform_func: Transformation fonksiyonu
    """
    tasks = [
        process_file_async_solution(input_f, output_f, transform_func)
        for input_f, output_f in file_pairs
    ]

    results = await asyncio.gather(*tasks, return_exceptions=True)
    return results


# ============================================================================
# ALIŞTIRMA 6: Async Database Connection Pool (HARD)
# ============================================================================
"""
TODO: Async veritabanı bağlantı havuzu implementasyonu yapın.
- Maksimum N bağlantı
- Bağlantı yeniden kullanımı
- Context manager desteği
- Timeout handling
- Health check
"""

class AsyncDatabasePool:
    """
    Async database connection pool

    Args:
        max_connections: Maksimum bağlantı sayısı
        connection_timeout: Bağlantı timeout (saniye)
    """

    def __init__(self, max_connections: int, connection_timeout: float = 10.0):
        # TODO: Implementasyonu yapın
        pass

    async def acquire(self):
        """Bağlantı al"""
        # TODO: Implementasyonu yapın
        pass

    async def release(self, connection):
        """Bağlantıyı geri ver"""
        # TODO: Implementasyonu yapın
        pass

    async def close(self):
        """Pool'u kapat"""
        # TODO: Implementasyonu yapın
        pass


# ÇÖZÜM:
@dataclass
class DBConnection:
    """Veritabanı bağlantısı (simüle edilmiş)"""
    connection_id: int
    created_at: float
    last_used: float

    async def execute(self, query: str) -> List[Dict]:
        """SQL sorgusu çalıştır (simüle edilmiş)"""
        await asyncio.sleep(random.uniform(0.1, 0.3))
        return [{"result": f"Query: {query}"}]


class AsyncDatabasePool_Solution:
    """Async database connection pool"""

    def __init__(self, max_connections: int, connection_timeout: float = 10.0):
        self.max_connections = max_connections
        self.connection_timeout = connection_timeout
        self.pool: asyncio.Queue[DBConnection] = asyncio.Queue(maxsize=max_connections)
        self.created_connections = 0
        self.lock = asyncio.Lock()

    async def _create_connection(self) -> DBConnection:
        """Yeni bağlantı oluştur"""
        await asyncio.sleep(0.5)  # Bağlantı simülasyonu
        async with self.lock:
            self.created_connections += 1
            conn = DBConnection(
                connection_id=self.created_connections,
                created_at=time.time(),
                last_used=time.time()
            )
        print(f"[Pool] Yeni bağlantı oluşturuldu: Connection-{conn.connection_id}")
        return conn

    async def acquire(self) -> DBConnection:
        """Bağlantı al"""
        try:
            # Pool'dan hazır bağlantı almayı dene
            conn = self.pool.get_nowait()
            conn.last_used = time.time()
            print(f"[Pool] Havuzdan bağlantı alındı: Connection-{conn.connection_id}")
            return conn
        except asyncio.QueueEmpty:
            # Pool boş - yeni bağlantı oluştur veya bekle
            async with self.lock:
                if self.created_connections < self.max_connections:
                    # Yeni bağlantı oluşturabilir
                    return await self._create_connection()

            # Maksimum bağlantı sayısına ulaşıldı - bekle
            print("[Pool] Maksimum bağlantı sayısına ulaşıldı, bekleniyor...")
            try:
                conn = await asyncio.wait_for(
                    self.pool.get(),
                    timeout=self.connection_timeout
                )
                conn.last_used = time.time()
                print(f"[Pool] Bekleme sonrası bağlantı alındı: Connection-{conn.connection_id}")
                return conn
            except asyncio.TimeoutError:
                raise TimeoutError("Bağlantı alma timeout!")

    async def release(self, connection: DBConnection):
        """Bağlantıyı geri ver"""
        connection.last_used = time.time()
        await self.pool.put(connection)
        print(f"[Pool] Bağlantı geri verildi: Connection-{connection.connection_id}")

    async def close(self):
        """Pool'u kapat"""
        print("[Pool] Kapatılıyor...")
        while not self.pool.empty():
            conn = await self.pool.get()
            print(f"[Pool] Bağlantı kapatıldı: Connection-{conn.connection_id}")
        print("[Pool] Kapatıldı")


async def test_database_pool():
    """Database pool testi"""
    pool = AsyncDatabasePool_Solution(max_connections=3)

    async def db_worker(worker_id: int):
        """Veritabanı işlemi yapan worker"""
        conn = await pool.acquire()

        try:
            # Sorgu çalıştır
            result = await conn.execute(f"SELECT * FROM users WHERE id={worker_id}")
            print(f"[Worker-{worker_id}] Sorgu sonucu: {result}")

            # İşlem simülasyonu
            await asyncio.sleep(random.uniform(0.5, 2.0))

        finally:
            await pool.release(conn)

    # 6 worker ama sadece 3 bağlantı
    await asyncio.gather(*[db_worker(i) for i in range(6)])

    await pool.close()


# ============================================================================
# ALIŞTIRMA 7: Async Retry Decorator (MEDIUM)
# ============================================================================
"""
TODO: Async fonksiyonlar için retry decorator yazın.
- Belirli exception'larda retry yapsın
- Max retry sayısı
- Exponential backoff
- Timeout
"""

def async_retry(
    max_retries: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,)
):
    """
    Async retry decorator

    Args:
        max_retries: Maksimum deneme sayısı
        delay: İlk bekleme süresi
        backoff: Her denemede bekleme süresini çarpan
        exceptions: Hangi exception'larda retry yapılacak
    """
    # TODO: Implementasyonu yapın
    pass


# ÇÖZÜM:
def async_retry_solution(
    max_retries: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,)
):
    """Async retry decorator"""

    def decorator(func):
        async def wrapper(*args, **kwargs):
            current_delay = delay
            last_exception = None

            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    print(f"[Retry] Deneme {attempt + 1}/{max_retries} başarısız: {e}")

                    if attempt < max_retries - 1:
                        print(f"[Retry] {current_delay:.2f}s bekleniyor...")
                        await asyncio.sleep(current_delay)
                        current_delay *= backoff

            # Tüm denemeler başarısız
            raise last_exception

        return wrapper
    return decorator


@async_retry_solution(max_retries=5, delay=0.5, backoff=2.0, exceptions=(ConnectionError,))
async def unreliable_api_call(success_rate: float = 0.3):
    """Güvenilmez API çağrısı"""
    await asyncio.sleep(0.2)
    if random.random() > success_rate:
        raise ConnectionError("API bağlantı hatası!")
    return "API başarılı!"


# ============================================================================
# ALIŞTIRMA 8: Async Iterator ile Data Stream (MEDIUM)
# ============================================================================
"""
TODO: API'den paginated veri çeken async iterator yazın.
- Her sayfayı async olarak çek
- __aiter__ ve __anext__ implement edin
- Context manager desteği
- Automatic pagination
"""

class AsyncPaginatedAPI:
    """
    Paginated API data fetcher

    Args:
        base_url: API base URL
        page_size: Her sayfada kaç item
        max_pages: Maksimum sayfa sayısı (None = sınırsız)
    """

    def __init__(self, base_url: str, page_size: int = 10, max_pages: Optional[int] = None):
        # TODO: Implementasyonu yapın
        pass

    async def __aenter__(self):
        # TODO: Session oluştur
        pass

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        # TODO: Session kapat
        pass

    def __aiter__(self):
        # TODO: Iterator döndür
        pass

    async def __anext__(self):
        # TODO: Sonraki sayfayı çek
        pass


# ÇÖZÜM:
class AsyncPaginatedAPI_Solution:
    """Paginated API data fetcher"""

    def __init__(self, base_url: str, page_size: int = 10, max_pages: Optional[int] = None):
        self.base_url = base_url
        self.page_size = page_size
        self.max_pages = max_pages
        self.current_page = 1
        self.session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        """Session oluştur"""
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Session kapat"""
        if self.session:
            await self.session.close()
        return False

    def __aiter__(self):
        return self

    async def __anext__(self):
        """Sonraki sayfayı çek"""
        # Max page kontrolü
        if self.max_pages and self.current_page > self.max_pages:
            raise StopAsyncIteration

        if not self.session:
            raise RuntimeError("Session açık değil - async with kullanın")

        # API'den sayfa çek
        url = f"{self.base_url}?page={self.current_page}&limit={self.page_size}"

        try:
            async with self.session.get(url) as response:
                if response.status != 200:
                    raise StopAsyncIteration

                data = await response.json()

                # Veri yoksa dur
                if not data:
                    raise StopAsyncIteration

                self.current_page += 1
                return data

        except Exception as e:
            print(f"API error: {e}")
            raise StopAsyncIteration


async def test_paginated_api():
    """Paginated API testi"""
    async with AsyncPaginatedAPI_Solution(
        "https://jsonplaceholder.typicode.com/posts",
        page_size=5,
        max_pages=3
    ) as api:
        async for page_data in api:
            print(f"Sayfa {api.current_page - 1}: {len(page_data)} item")


# ============================================================================
# ALIŞTIRMA 9: Async Task Monitor (MEDIUM-HARD)
# ============================================================================
"""
TODO: Async task'ları izleyen ve yöneten bir sistem yazın.
- Task oluşturma ve takip
- Progress monitoring
- Timeout handling
- Cancellation
- Result collection
"""

class AsyncTaskMonitor:
    """
    Async task monitor ve yönetici
    """

    def __init__(self):
        # TODO: Implementasyonu yapın
        pass

    async def submit_task(self, coro, timeout: Optional[float] = None) -> str:
        """
        Task submit et

        Returns:
            Task ID
        """
        # TODO: Implementasyonu yapın
        pass

    async def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """Task durumunu al"""
        # TODO: Implementasyonu yapın
        pass

    async def cancel_task(self, task_id: str) -> bool:
        """Task'ı iptal et"""
        # TODO: Implementasyonu yapın
        pass

    async def wait_all(self) -> Dict[str, Any]:
        """Tüm task'ların bitmesini bekle"""
        # TODO: Implementasyonu yapın
        pass


# ÇÖZÜM:
import uuid

class AsyncTaskMonitor_Solution:
    """Async task monitor ve yönetici"""

    def __init__(self):
        self.tasks: Dict[str, asyncio.Task] = {}
        self.results: Dict[str, Any] = {}
        self.lock = asyncio.Lock()

    async def submit_task(self, coro, timeout: Optional[float] = None) -> str:
        """Task submit et"""
        task_id = str(uuid.uuid4())[:8]

        async def wrapped_task():
            """Timeout ve result tracking ile wrapped task"""
            try:
                if timeout:
                    result = await asyncio.wait_for(coro, timeout=timeout)
                else:
                    result = await coro

                async with self.lock:
                    self.results[task_id] = {
                        "status": "completed",
                        "result": result,
                        "error": None
                    }

            except asyncio.TimeoutError:
                async with self.lock:
                    self.results[task_id] = {
                        "status": "timeout",
                        "result": None,
                        "error": "Task timeout"
                    }

            except asyncio.CancelledError:
                async with self.lock:
                    self.results[task_id] = {
                        "status": "cancelled",
                        "result": None,
                        "error": "Task cancelled"
                    }
                raise

            except Exception as e:
                async with self.lock:
                    self.results[task_id] = {
                        "status": "failed",
                        "result": None,
                        "error": str(e)
                    }

        # Task oluştur ve kaydet
        task = asyncio.create_task(wrapped_task())
        async with self.lock:
            self.tasks[task_id] = task
            self.results[task_id] = {
                "status": "running",
                "result": None,
                "error": None
            }

        print(f"[Monitor] Task submitted: {task_id}")
        return task_id

    async def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """Task durumunu al"""
        async with self.lock:
            if task_id not in self.results:
                return {"error": "Task not found"}

            return {
                "task_id": task_id,
                "done": self.tasks[task_id].done(),
                **self.results[task_id]
            }

    async def cancel_task(self, task_id: str) -> bool:
        """Task'ı iptal et"""
        async with self.lock:
            if task_id not in self.tasks:
                return False

            task = self.tasks[task_id]
            if not task.done():
                task.cancel()
                print(f"[Monitor] Task cancelled: {task_id}")
                return True

            return False

    async def wait_all(self) -> Dict[str, Any]:
        """Tüm task'ların bitmesini bekle"""
        async with self.lock:
            tasks = list(self.tasks.values())

        # Tüm task'ları bekle
        await asyncio.gather(*tasks, return_exceptions=True)

        async with self.lock:
            return dict(self.results)


async def test_task_monitor():
    """Task monitor testi"""
    monitor = AsyncTaskMonitor_Solution()

    async def long_task(duration: float, should_fail: bool = False):
        """Uzun süren task"""
        await asyncio.sleep(duration)
        if should_fail:
            raise ValueError("Task failed!")
        return f"Completed after {duration}s"

    # Task'ları submit et
    task1 = await monitor.submit_task(long_task(2, False), timeout=5)
    task2 = await monitor.submit_task(long_task(1, False), timeout=5)
    task3 = await monitor.submit_task(long_task(10, False), timeout=3)  # Timeout
    task4 = await monitor.submit_task(long_task(2, True), timeout=5)    # Fail

    # Bir task'ı iptal et
    await asyncio.sleep(0.5)
    await monitor.cancel_task(task4)

    # Status'leri kontrol et
    await asyncio.sleep(1)
    for task_id in [task1, task2, task3, task4]:
        status = await monitor.get_task_status(task_id)
        print(f"Task {task_id}: {status}")

    # Tüm task'ların bitmesini bekle
    results = await monitor.wait_all()
    print("\nFinal Results:")
    for task_id, result in results.items():
        print(f"  {task_id}: {result['status']}")


# ============================================================================
# ALIŞTIRMA 10: Async WebSocket Client (HARD)
# ============================================================================
"""
TODO: WebSocket client implementasyonu yapın (simüle edilmiş).
- Bağlantı yönetimi
- Mesaj gönderme/alma
- Automatic reconnection
- Heartbeat/ping-pong
- Graceful shutdown
"""

class AsyncWebSocketClient:
    """
    Async WebSocket client (simüle edilmiş)

    Args:
        url: WebSocket URL
        reconnect_interval: Tekrar bağlanma aralığı
        heartbeat_interval: Heartbeat aralığı
    """

    def __init__(
        self,
        url: str,
        reconnect_interval: float = 5.0,
        heartbeat_interval: float = 30.0
    ):
        # TODO: Implementasyonu yapın
        pass

    async def connect(self):
        """WebSocket'e bağlan"""
        # TODO: Implementasyonu yapın
        pass

    async def send_message(self, message: str):
        """Mesaj gönder"""
        # TODO: Implementasyonu yapın
        pass

    async def receive_messages(self):
        """Mesajları al (generator)"""
        # TODO: Implementasyonu yapın
        pass

    async def close(self):
        """Bağlantıyı kapat"""
        # TODO: Implementasyonu yapın
        pass


# ÇÖZÜM:
class AsyncWebSocketClient_Solution:
    """Async WebSocket client (simüle edilmiş)"""

    def __init__(
        self,
        url: str,
        reconnect_interval: float = 5.0,
        heartbeat_interval: float = 30.0
    ):
        self.url = url
        self.reconnect_interval = reconnect_interval
        self.heartbeat_interval = heartbeat_interval

        self.connected = False
        self.should_reconnect = True
        self.message_queue = asyncio.Queue()
        self.heartbeat_task: Optional[asyncio.Task] = None

    async def connect(self):
        """WebSocket'e bağlan"""
        print(f"[WS] Connecting to {self.url}...")
        await asyncio.sleep(1)  # Bağlantı simülasyonu

        self.connected = True
        print(f"[WS] Connected!")

        # Heartbeat başlat
        self.heartbeat_task = asyncio.create_task(self._heartbeat())

    async def _heartbeat(self):
        """Heartbeat gönder"""
        try:
            while self.connected:
                await asyncio.sleep(self.heartbeat_interval)
                if self.connected:
                    print("[WS] Sending heartbeat...")
        except asyncio.CancelledError:
            print("[WS] Heartbeat stopped")

    async def send_message(self, message: str):
        """Mesaj gönder"""
        if not self.connected:
            raise RuntimeError("Not connected!")

        print(f"[WS] Sending: {message}")
        await asyncio.sleep(0.1)  # Gönderim simülasyonu

    async def receive_messages(self):
        """Mesajları al (generator)"""
        while self.connected:
            # Simüle edilmiş mesaj alma
            await asyncio.sleep(random.uniform(1, 3))

            if self.connected:
                message = f"Message-{random.randint(1000, 9999)}"
                yield message

    async def close(self):
        """Bağlantıyı kapat"""
        print("[WS] Closing connection...")
        self.connected = False
        self.should_reconnect = False

        if self.heartbeat_task:
            self.heartbeat_task.cancel()
            try:
                await self.heartbeat_task
            except asyncio.CancelledError:
                pass

        await asyncio.sleep(0.5)  # Kapatma simülasyonu
        print("[WS] Closed")


async def test_websocket():
    """WebSocket client testi"""
    client = AsyncWebSocketClient_Solution("wss://example.com/ws")

    # Bağlan
    await client.connect()

    # Mesaj gönder
    await client.send_message("Hello WebSocket!")

    # Mesajları dinle (5 saniye)
    async def listen():
        count = 0
        async for message in client.receive_messages():
            print(f"[WS] Received: {message}")
            count += 1
            if count >= 3:
                break

    # Listen task'ı oluştur
    listen_task = asyncio.create_task(listen())

    # 5 saniye bekle
    await asyncio.sleep(5)

    # Kapat
    await client.close()


# ============================================================================
# ALIŞTIRMA 11: Async Batch Processor (MEDIUM)
# ============================================================================
"""
TODO: Batch processing sistemi yazın.
- Items'ı belirli batch size'da grupla
- Her batch'i paralel işle
- Max concurrent batches sınırı
- Progress tracking
"""

async def process_items_in_batches(
    items: List[Any],
    batch_size: int,
    max_concurrent_batches: int,
    process_func
) -> Dict[str, Any]:
    """
    Items'ı batch'ler halinde işle

    Args:
        items: İşlenecek item listesi
        batch_size: Her batch'teki item sayısı
        max_concurrent_batches: Aynı anda maksimum batch sayısı
        process_func: Her item için çağrılacak async fonksiyon

    Returns:
        Dict containing processed_count, failed_count, duration
    """
    # TODO: Implementasyonu yapın
    pass


# ÇÖZÜM:
async def process_items_in_batches_solution(
    items: List[Any],
    batch_size: int,
    max_concurrent_batches: int,
    process_func
) -> Dict[str, Any]:
    """Items'ı batch'ler halinde işle"""
    start_time = time.time()
    processed_count = 0
    failed_count = 0

    # Batch'leri oluştur
    batches = [
        items[i:i + batch_size]
        for i in range(0, len(items), batch_size)
    ]

    print(f"[Batch] {len(items)} item, {len(batches)} batch oluşturuldu")

    # Semaphore ile concurrent batch limitini kontrol et
    semaphore = asyncio.Semaphore(max_concurrent_batches)

    async def process_batch(batch_id: int, batch: List[Any]):
        """Tek bir batch'i işle"""
        nonlocal processed_count, failed_count

        async with semaphore:
            print(f"[Batch-{batch_id}] İşleniyor ({len(batch)} item)...")

            # Batch içindeki tüm item'ları paralel işle
            tasks = [process_func(item) for item in batch]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Sonuçları say
            for result in results:
                if isinstance(result, Exception):
                    failed_count += 1
                else:
                    processed_count += 1

            print(f"[Batch-{batch_id}] Tamamlandı")

    # Tüm batch'leri işle
    batch_tasks = [
        process_batch(i, batch)
        for i, batch in enumerate(batches)
    ]

    await asyncio.gather(*batch_tasks)

    return {
        "total_items": len(items),
        "processed_count": processed_count,
        "failed_count": failed_count,
        "duration": time.time() - start_time,
        "batches": len(batches)
    }


async def test_batch_processor():
    """Batch processor testi"""

    async def process_item(item: int):
        """Item işleme fonksiyonu"""
        await asyncio.sleep(random.uniform(0.1, 0.5))
        if random.random() < 0.1:  # %10 başarısızlık
            raise ValueError(f"Item {item} failed!")
        return f"Processed-{item}"

    items = list(range(50))
    result = await process_items_in_batches_solution(
        items=items,
        batch_size=10,
        max_concurrent_batches=3,
        process_func=process_item
    )

    print(f"\nBatch Processing Results:")
    print(f"  Total: {result['total_items']}")
    print(f"  Processed: {result['processed_count']}")
    print(f"  Failed: {result['failed_count']}")
    print(f"  Duration: {result['duration']:.2f}s")


# ============================================================================
# ALIŞTIRMA 12: Async Cache System (HARD)
# ============================================================================
"""
TODO: Async cache sistemi yazın.
- TTL (Time To Live) desteği
- Async get/set
- Cache invalidation
- Max size limit (LRU eviction)
- Thread-safe
"""

class AsyncCache:
    """
    Async cache system

    Args:
        max_size: Maksimum cache boyutu
        default_ttl: Default TTL (saniye)
    """

    def __init__(self, max_size: int = 100, default_ttl: float = 300):
        # TODO: Implementasyonu yapın
        pass

    async def get(self, key: str) -> Optional[Any]:
        """Cache'den değer al"""
        # TODO: Implementasyonu yapın
        pass

    async def set(self, key: str, value: Any, ttl: Optional[float] = None):
        """Cache'e değer ekle"""
        # TODO: Implementasyonu yapın
        pass

    async def delete(self, key: str) -> bool:
        """Cache'den sil"""
        # TODO: Implementasyonu yapın
        pass

    async def clear(self):
        """Cache'i temizle"""
        # TODO: Implementasyonu yapın
        pass


# ÇÖZÜM:
from collections import OrderedDict

@dataclass
class CacheEntry:
    """Cache entry"""
    value: Any
    expires_at: float

class AsyncCache_Solution:
    """Async cache system"""

    def __init__(self, max_size: int = 100, default_ttl: float = 300):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.lock = asyncio.Lock()

    async def get(self, key: str) -> Optional[Any]:
        """Cache'den değer al"""
        async with self.lock:
            # Key var mı?
            if key not in self.cache:
                return None

            entry = self.cache[key]

            # Expire olmuş mu?
            if time.time() > entry.expires_at:
                del self.cache[key]
                return None

            # LRU için sona taşı
            self.cache.move_to_end(key)

            return entry.value

    async def set(self, key: str, value: Any, ttl: Optional[float] = None):
        """Cache'e değer ekle"""
        async with self.lock:
            # TTL hesapla
            if ttl is None:
                ttl = self.default_ttl

            expires_at = time.time() + ttl

            # Entry oluştur
            entry = CacheEntry(value=value, expires_at=expires_at)

            # Var olan key'i güncelle
            if key in self.cache:
                self.cache[key] = entry
                self.cache.move_to_end(key)
            else:
                # Yeni key ekle
                self.cache[key] = entry

                # Max size kontrolü (LRU eviction)
                if len(self.cache) > self.max_size:
                    # En eski (least recently used) item'ı çıkar
                    oldest_key = next(iter(self.cache))
                    del self.cache[oldest_key]
                    print(f"[Cache] LRU eviction: {oldest_key}")

    async def delete(self, key: str) -> bool:
        """Cache'den sil"""
        async with self.lock:
            if key in self.cache:
                del self.cache[key]
                return True
            return False

    async def clear(self):
        """Cache'i temizle"""
        async with self.lock:
            self.cache.clear()

    async def cleanup_expired(self):
        """Expire olmuş entry'leri temizle"""
        async with self.lock:
            now = time.time()
            expired_keys = [
                key for key, entry in self.cache.items()
                if now > entry.expires_at
            ]

            for key in expired_keys:
                del self.cache[key]

            if expired_keys:
                print(f"[Cache] {len(expired_keys)} expired entry temizlendi")


async def test_cache():
    """Cache testi"""
    cache = AsyncCache_Solution(max_size=5, default_ttl=3.0)

    # Set
    await cache.set("user:1", {"name": "Alice"})
    await cache.set("user:2", {"name": "Bob"}, ttl=1.0)
    await cache.set("user:3", {"name": "Charlie"})

    # Get
    user1 = await cache.get("user:1")
    print(f"User 1: {user1}")

    # TTL testi
    await asyncio.sleep(2)
    user2 = await cache.get("user:2")
    print(f"User 2 (expired): {user2}")

    # LRU testi
    for i in range(4, 10):
        await cache.set(f"user:{i}", {"name": f"User{i}"})

    # user:1 evict edildi mi?
    user1_after = await cache.get("user:1")
    print(f"User 1 after LRU: {user1_after}")


# ============================================================================
# ALIŞTIRMA 13: Async Event System (MEDIUM-HARD)
# ============================================================================
"""
TODO: Event-driven async sistem yazın.
- Event subscription/publishing
- Async event handlers
- Priority handling
- Error handling
"""

class AsyncEventSystem:
    """
    Async event system
    """

    def __init__(self):
        # TODO: Implementasyonu yapın
        pass

    async def subscribe(self, event_name: str, handler, priority: int = 0):
        """Event'e handler ekle"""
        # TODO: Implementasyonu yapın
        pass

    async def unsubscribe(self, event_name: str, handler):
        """Handler'ı kaldır"""
        # TODO: Implementasyonu yapın
        pass

    async def publish(self, event_name: str, **kwargs):
        """Event yayınla"""
        # TODO: Implementasyonu yapın
        pass


# ÇÖZÜM:
class AsyncEventSystem_Solution:
    """Async event system"""

    def __init__(self):
        self.handlers: Dict[str, List[tuple]] = {}  # event_name -> [(priority, handler), ...]
        self.lock = asyncio.Lock()

    async def subscribe(self, event_name: str, handler, priority: int = 0):
        """Event'e handler ekle"""
        async with self.lock:
            if event_name not in self.handlers:
                self.handlers[event_name] = []

            self.handlers[event_name].append((priority, handler))

            # Priority'ye göre sırala (yüksek priority önce)
            self.handlers[event_name].sort(key=lambda x: x[0], reverse=True)

        print(f"[Event] Handler subscribed to '{event_name}' (priority: {priority})")

    async def unsubscribe(self, event_name: str, handler):
        """Handler'ı kaldır"""
        async with self.lock:
            if event_name in self.handlers:
                self.handlers[event_name] = [
                    (p, h) for p, h in self.handlers[event_name]
                    if h != handler
                ]
                print(f"[Event] Handler unsubscribed from '{event_name}'")

    async def publish(self, event_name: str, **kwargs):
        """Event yayınla"""
        async with self.lock:
            handlers = self.handlers.get(event_name, []).copy()

        if not handlers:
            return

        print(f"[Event] Publishing '{event_name}' to {len(handlers)} handlers")

        # Tüm handler'ları paralel çağır
        tasks = []
        for priority, handler in handlers:
            try:
                task = asyncio.create_task(handler(**kwargs))
                tasks.append(task)
            except Exception as e:
                print(f"[Event] Handler error: {e}")

        # Tüm handler'ların bitmesini bekle
        await asyncio.gather(*tasks, return_exceptions=True)


async def test_event_system():
    """Event system testi"""
    events = AsyncEventSystem_Solution()

    # Handler'ları tanımla
    async def user_created_handler(user_id: int, **kwargs):
        await asyncio.sleep(0.5)
        print(f"[Handler1] User created: {user_id}")

    async def send_welcome_email(user_id: int, **kwargs):
        await asyncio.sleep(0.3)
        print(f"[Handler2] Welcome email sent to user: {user_id}")

    async def update_analytics(user_id: int, **kwargs):
        await asyncio.sleep(0.2)
        print(f"[Handler3] Analytics updated for user: {user_id}")

    # Subscribe
    await events.subscribe("user.created", user_created_handler, priority=0)
    await events.subscribe("user.created", send_welcome_email, priority=10)
    await events.subscribe("user.created", update_analytics, priority=5)

    # Publish
    await events.publish("user.created", user_id=123, email="test@example.com")


# ============================================================================
# TEST RUNNER
# ============================================================================

async def main():
    """Tüm testleri çalıştır"""

    print("=" * 70)
    print("ASYNC PROGRAMMING - İLERİ SEVİYE ALIŞTIRMALAR")
    print("=" * 70)

    # Test 1: Paralel API istekleri
    print("\n" + "=" * 70)
    print("TEST 1: Paralel API İstekleri")
    print("=" * 70)
    urls = [
        "https://jsonplaceholder.typicode.com/posts/1",
        "https://jsonplaceholder.typicode.com/posts/2",
        "https://jsonplaceholder.typicode.com/posts/3",
    ]
    result = await fetch_multiple_apis_solution(urls)
    print(f"Sonuç: {result['success_count']} başarılı, {result['error_count']} hata")
    print(f"Süre: {result['duration']:.2f}s")

    # Test 2: Rate Limiter
    print("\n" + "=" * 70)
    print("TEST 2: Rate Limiter")
    print("=" * 70)
    await test_rate_limiter()

    # Test 3: Circuit Breaker
    print("\n" + "=" * 70)
    print("TEST 3: Circuit Breaker")
    print("=" * 70)
    await test_circuit_breaker()

    # Test 4: Producer-Consumer
    print("\n" + "=" * 70)
    print("TEST 4: Producer-Consumer")
    print("=" * 70)
    await run_producer_consumer_solution(
        producer_count=2,
        consumer_count=3,
        items_per_producer=5
    )

    # Test 5: Database Pool
    print("\n" + "=" * 70)
    print("TEST 5: Database Connection Pool")
    print("=" * 70)
    await test_database_pool()

    # Test 6: Retry Decorator
    print("\n" + "=" * 70)
    print("TEST 6: Retry Decorator")
    print("=" * 70)
    try:
        result = await unreliable_api_call(success_rate=0.4)
        print(f"API Sonucu: {result}")
    except Exception as e:
        print(f"API başarısız: {e}")

    # Test 7: Paginated API
    print("\n" + "=" * 70)
    print("TEST 7: Paginated API Iterator")
    print("=" * 70)
    await test_paginated_api()

    # Test 8: Task Monitor
    print("\n" + "=" * 70)
    print("TEST 8: Task Monitor")
    print("=" * 70)
    await test_task_monitor()

    # Test 9: WebSocket (simüle edilmiş)
    print("\n" + "=" * 70)
    print("TEST 9: WebSocket Client")
    print("=" * 70)
    await test_websocket()

    # Test 10: Batch Processor
    print("\n" + "=" * 70)
    print("TEST 10: Batch Processor")
    print("=" * 70)
    await test_batch_processor()

    # Test 11: Cache System
    print("\n" + "=" * 70)
    print("TEST 11: Cache System")
    print("=" * 70)
    await test_cache()

    # Test 12: Event System
    print("\n" + "=" * 70)
    print("TEST 12: Event System")
    print("=" * 70)
    await test_event_system()

    print("\n" + "=" * 70)
    print("TÜM TESTLER TAMAMLANDI!")
    print("=" * 70)


if __name__ == "__main__":
    # Not: aiohttp ve aiofiles kütüphaneleri gereklidir
    # pip install aiohttp aiofiles
    asyncio.run(main())
