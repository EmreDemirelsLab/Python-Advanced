"""
MULTITHREADING & MULTIPROCESSING - İLERİ SEVİYE ALIŞTIRMALAR
=============================================================

Bu dosya, threading, multiprocessing, GIL, thread safety, senkronizasyon
ve paralel programlama konularında ileri seviye alıştırmalar içerir.

Zorluk Seviyeleri:
- MEDIUM: Threading ve multiprocessing temelleri
- HARD: Senkronizasyon ve thread safety
- EXPERT: Karmaşık paralel sistemler ve optimizasyon

Her alıştırma için:
1. Önce TODO kısmını kendin yapmaya çalış
2. Sonra SOLUTION ile karşılaştır
3. Farklı senaryoları dene
"""

import threading
import multiprocessing
import queue
import time
import random
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from typing import List, Dict, Any, Optional
import os


# ============================================================================
# ALIŞTIRMA 1: Thread-Safe Counter (MEDIUM)
# ============================================================================
"""
AÇIKLAMA:
Thread-safe bir sayaç sınıfı oluşturun. Birden fazla thread aynı anda
increment ve decrement yapabilmeli, ancak race condition olmamalı.

GEREKSİNİMLER:
- increment(), decrement(), get_value() metodları
- Lock kullanarak thread safety
- 1000 thread ile test edilebilmeli
"""

# TODO: Thread-safe Counter sınıfını tamamla


class ThreadSafeCounter:
    def __init__(self):
        # TODO: Lock ve counter değişkeni ekle
        pass

    def increment(self):
        # TODO: Thread-safe increment
        pass

    def decrement(self):
        # TODO: Thread-safe decrement
        pass

    def get_value(self):
        # TODO: Thread-safe get
        pass


# SOLUTION:
class ThreadSafeCounterSolution:
    """Thread-safe sayaç implementasyonu"""

    def __init__(self):
        self._value = 0
        self._lock = threading.Lock()

    def increment(self):
        """Değeri 1 artır"""
        with self._lock:
            self._value += 1

    def decrement(self):
        """Değeri 1 azalt"""
        with self._lock:
            self._value -= 1

    def get_value(self):
        """Mevcut değeri al"""
        with self._lock:
            return self._value


def test_thread_safe_counter():
    """Counter'ı test et"""
    counter = ThreadSafeCounterSolution()

    def worker():
        for _ in range(1000):
            counter.increment()
            counter.decrement()
            counter.increment()

    threads = [threading.Thread(target=worker) for _ in range(10)]

    for t in threads:
        t.start()

    for t in threads:
        t.join()

    # 10 thread * 1000 iteration = 10000 increment net
    print(f"✓ Test başarılı! Final değer: {counter.get_value()}")
    assert counter.get_value() == 10000, "Counter değeri yanlış!"


# ============================================================================
# ALIŞTIRMA 2: Producer-Consumer Pattern (MEDIUM)
# ============================================================================
"""
AÇIKLAMA:
Queue kullanarak producer-consumer pattern implementasyonu.
Producer'lar veri üretir, consumer'lar tüketir.

GEREKSİNİMLER:
- Queue ile thread-safe iletişim
- Birden fazla producer ve consumer
- Graceful shutdown (poison pill pattern)
- İstatistik toplama (kaç item üretildi/tüketildi)
"""

# TODO: Producer-Consumer sistemini tamamla


class ProducerConsumerSystem:
    def __init__(self, num_producers=2, num_consumers=3):
        # TODO: Queue, thread listesi ve istatistikler
        pass

    def producer(self, producer_id):
        # TODO: Veri üret ve queue'ya ekle
        pass

    def consumer(self, consumer_id):
        # TODO: Queue'dan veri al ve işle
        pass

    def start(self):
        # TODO: Producer ve consumer thread'leri başlat
        pass

    def stop(self):
        # TODO: Poison pill gönder ve bekle
        pass

    def get_stats(self):
        # TODO: İstatistikleri döndür
        pass


# SOLUTION:
class ProducerConsumerSystemSolution:
    """Producer-Consumer pattern implementasyonu"""

    def __init__(self, num_producers=2, num_consumers=3):
        self.queue = queue.Queue(maxsize=10)
        self.num_producers = num_producers
        self.num_consumers = num_consumers
        self.producers = []
        self.consumers = []
        self.stats = {"produced": 0, "consumed": 0}
        self.stats_lock = threading.Lock()

    def producer(self, producer_id, num_items=20):
        """Veri üretici"""
        for i in range(num_items):
            item = random.randint(1, 100)
            self.queue.put(item)

            with self.stats_lock:
                self.stats["produced"] += 1

            print(f"[Producer-{producer_id}] Üretildi: {item}")
            time.sleep(random.uniform(0.1, 0.3))

        print(f"[Producer-{producer_id}] Tamamlandı")

    def consumer(self, consumer_id):
        """Veri tüketici"""
        while True:
            try:
                item = self.queue.get(timeout=2)

                if item is None:  # Poison pill
                    self.queue.put(None)  # Diğerleri için geri koy
                    break

                # İşlemi simüle et
                time.sleep(random.uniform(0.1, 0.4))

                with self.stats_lock:
                    self.stats["consumed"] += 1

                print(f"[Consumer-{consumer_id}] Tüketildi: {item}")
                self.queue.task_done()

            except queue.Empty:
                break

        print(f"[Consumer-{consumer_id}] Tamamlandı")

    def start(self):
        """Sistemi başlat"""
        # Producer'ları başlat
        for i in range(self.num_producers):
            t = threading.Thread(target=self.producer, args=(i, 20))
            t.start()
            self.producers.append(t)

        # Consumer'ları başlat
        for i in range(self.num_consumers):
            t = threading.Thread(target=self.consumer, args=(i,))
            t.start()
            self.consumers.append(t)

    def stop(self):
        """Sistemi durdur"""
        # Producer'ların bitmesini bekle
        for t in self.producers:
            t.join()

        # Poison pill gönder
        self.queue.put(None)

        # Consumer'ların bitmesini bekle
        for t in self.consumers:
            t.join()

    def get_stats(self):
        """İstatistikleri al"""
        with self.stats_lock:
            return self.stats.copy()


def test_producer_consumer():
    """Producer-Consumer sistemini test et"""
    system = ProducerConsumerSystemSolution(num_producers=2, num_consumers=3)

    system.start()
    system.stop()

    stats = system.get_stats()
    print(f"\n✓ Test başarılı!")
    print(f"Üretilen: {stats['produced']}, Tüketilen: {stats['consumed']}")
    assert stats["produced"] == stats["consumed"], "Üretilen ve tüketilen sayısı eşit değil!"


# ============================================================================
# ALIŞTIRMA 3: Thread Pool for Web Scraping (MEDIUM)
# ============================================================================
"""
AÇIKLAMA:
ThreadPoolExecutor kullanarak paralel web scraping simülasyonu.
Birden fazla URL'den veri çekin ve sonuçları toplayın.

GEREKSİNİMLER:
- ThreadPoolExecutor kullanımı
- Timeout handling
- Hata yönetimi
- Başarı/hata oranı istatistiği
"""

# TODO: Web scraper tamamla


def fetch_url(url: str) -> Dict[str, Any]:
    """URL'den veri çek (simülasyon)"""
    # TODO: URL fetch simülasyonu ve hata handling
    pass


def parallel_scraper(urls: List[str], max_workers=5) -> Dict[str, Any]:
    """Paralel scraping"""
    # TODO: ThreadPoolExecutor ile paralel scraping
    pass


# SOLUTION:
def fetch_url_solution(url: str) -> Dict[str, Any]:
    """URL'den veri çek (simülasyon)"""
    print(f"[Fetching] {url}")

    # Simüle edilmiş gecikme
    time.sleep(random.uniform(0.5, 2))

    # %20 hata ihtimali
    if random.random() < 0.2:
        raise Exception(f"Failed to fetch {url}")

    # Başarılı sonuç
    return {
        "url": url,
        "status": 200,
        "content_length": random.randint(1000, 50000),
        "title": f"Page from {url}",
    }


def parallel_scraper_solution(urls: List[str], max_workers=5) -> Dict[str, Any]:
    """Paralel scraping yapıcı"""
    results = []
    errors = []

    start_time = time.time()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit tüm task'ları
        future_to_url = {executor.submit(fetch_url_solution, url): url for url in urls}

        # Sonuçları topla
        from concurrent.futures import as_completed

        for future in as_completed(future_to_url):
            url = future_to_url[future]

            try:
                result = future.result(timeout=5)
                results.append(result)
                print(f"✓ {url}: {result['content_length']} bytes")

            except Exception as e:
                errors.append({"url": url, "error": str(e)})
                print(f"✗ {url}: {e}")

    elapsed = time.time() - start_time

    return {
        "total": len(urls),
        "successful": len(results),
        "failed": len(errors),
        "results": results,
        "errors": errors,
        "elapsed": elapsed,
    }


def test_parallel_scraper():
    """Parallel scraper'ı test et"""
    urls = [f"https://example.com/page{i}" for i in range(20)]

    result = parallel_scraper_solution(urls, max_workers=5)

    print(f"\n✓ Scraping tamamlandı!")
    print(f"Toplam: {result['total']}")
    print(f"Başarılı: {result['successful']}")
    print(f"Hatalı: {result['failed']}")
    print(f"Süre: {result['elapsed']:.2f}s")


# ============================================================================
# ALIŞTIRMA 4: CPU-Bound Task with Multiprocessing (MEDIUM)
# ============================================================================
"""
AÇIKLAMA:
ProcessPoolExecutor ile CPU-yoğun işlemleri paralel çalıştırın.
Threading vs Multiprocessing performans karşılaştırması yapın.

GEREKSİNİMLER:
- CPU-bound task (prime sayı bulma)
- Threading ile test
- Multiprocessing ile test
- Performans karşılaştırması
"""

# TODO: CPU-bound task ve karşılaştırma


def find_primes(start: int, end: int) -> List[int]:
    """Aralıktaki asal sayıları bul"""
    # TODO: Asal sayıları bulma algoritması
    pass


def compare_threading_vs_multiprocessing(ranges: List[tuple]):
    """Threading vs Multiprocessing karşılaştırması"""
    # TODO: Her iki yöntemle de test et ve karşılaştır
    pass


# SOLUTION:
def is_prime(n: int) -> bool:
    """Sayının asal olup olmadığını kontrol et"""
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False

    for i in range(3, int(n**0.5) + 1, 2):
        if n % i == 0:
            return False

    return True


def find_primes_solution(start: int, end: int) -> List[int]:
    """Aralıktaki asal sayıları bul"""
    primes = [n for n in range(start, end) if is_prime(n)]
    return primes


def compare_threading_vs_multiprocessing_solution(ranges: List[tuple]):
    """Threading vs Multiprocessing karşılaştırması"""

    # Threading ile
    print("Threading ile CPU-bound task:")
    start = time.time()

    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(lambda r: find_primes_solution(*r), ranges))

    threading_time = time.time() - start
    threading_count = sum(len(r) for r in results)

    print(f"Süre: {threading_time:.2f}s, Bulunan asal: {threading_count}")

    # Multiprocessing ile
    print("\nMultiprocessing ile CPU-bound task:")
    start = time.time()

    with ProcessPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(lambda r: find_primes_solution(*r), ranges))

    multiprocessing_time = time.time() - start
    multiprocessing_count = sum(len(r) for r in results)

    print(f"Süre: {multiprocessing_time:.2f}s, Bulunan asal: {multiprocessing_count}")

    # Karşılaştırma
    speedup = threading_time / multiprocessing_time
    print(f"\n✓ Multiprocessing {speedup:.2f}x daha hızlı!")

    return {"threading": threading_time, "multiprocessing": multiprocessing_time, "speedup": speedup}


def test_cpu_bound():
    """CPU-bound task testi"""
    # 4 ayrı aralıkta asal sayı ara
    ranges = [(100000, 150000), (150000, 200000), (200000, 250000), (250000, 300000)]

    compare_threading_vs_multiprocessing_solution(ranges)


# ============================================================================
# ALIŞTIRMA 5: Thread-Safe Cache (HARD)
# ============================================================================
"""
AÇIKLAMA:
Thread-safe LRU cache implementasyonu. Birden fazla thread aynı anda
cache'e erişebilmeli, thread-safe olmalı.

GEREKSİNİMLER:
- get(), put() metodları
- LRU eviction policy
- Max size limiti
- Thread-safety (RLock kullan)
- Cache hit/miss istatistikleri
"""

# TODO: Thread-safe LRU Cache


class ThreadSafeLRUCache:
    def __init__(self, capacity: int):
        # TODO: Cache yapısını oluştur
        pass

    def get(self, key: Any) -> Optional[Any]:
        # TODO: Thread-safe get
        pass

    def put(self, key: Any, value: Any):
        # TODO: Thread-safe put with LRU eviction
        pass

    def stats(self) -> Dict[str, int]:
        # TODO: Hit/miss istatistikleri
        pass


# SOLUTION:
from collections import OrderedDict


class ThreadSafeLRUCacheSolution:
    """Thread-safe LRU cache implementasyonu"""

    def __init__(self, capacity: int):
        self.cache = OrderedDict()
        self.capacity = capacity
        self.lock = threading.RLock()
        self.hits = 0
        self.misses = 0

    def get(self, key: Any) -> Optional[Any]:
        """Key'in değerini al"""
        with self.lock:
            if key not in self.cache:
                self.misses += 1
                return None

            # LRU: En sona taşı
            self.cache.move_to_end(key)
            self.hits += 1
            return self.cache[key]

    def put(self, key: Any, value: Any):
        """Key-value çifti ekle"""
        with self.lock:
            if key in self.cache:
                # Güncelle ve en sona taşı
                self.cache.move_to_end(key)

            self.cache[key] = value

            # Capacity aşıldıysa en eski öğeyi çıkar
            if len(self.cache) > self.capacity:
                oldest = next(iter(self.cache))
                del self.cache[oldest]

    def stats(self) -> Dict[str, int]:
        """İstatistikleri döndür"""
        with self.lock:
            total = self.hits + self.misses
            hit_rate = (self.hits / total * 100) if total > 0 else 0

            return {"hits": self.hits, "misses": self.misses, "hit_rate": hit_rate, "size": len(self.cache)}


def test_lru_cache():
    """LRU cache'i test et"""
    cache = ThreadSafeLRUCacheSolution(capacity=100)

    def worker(worker_id):
        """Cache'e erişen worker"""
        for i in range(1000):
            key = f"key_{random.randint(1, 150)}"

            # %50 okuma, %50 yazma
            if random.random() < 0.5:
                value = cache.get(key)
            else:
                cache.put(key, f"value_{i}")

    # 10 thread ile test
    threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]

    for t in threads:
        t.start()

    for t in threads:
        t.join()

    # İstatistikler
    stats = cache.stats()
    print(f"✓ LRU Cache test başarılı!")
    print(f"Hits: {stats['hits']}")
    print(f"Misses: {stats['misses']}")
    print(f"Hit Rate: {stats['hit_rate']:.2f}%")
    print(f"Cache Size: {stats['size']}")


# ============================================================================
# ALIŞTIRMA 6: Parallel File Processor (HARD)
# ============================================================================
"""
AÇIKLAMA:
Birden fazla dosyayı paralel olarak işleyin. Her dosya için:
1. Dosyayı oku
2. İşle (satır sayısı, kelime sayısı, karakter sayısı)
3. Sonucu kaydet

GEREKSİNİMLER:
- ProcessPoolExecutor kullan
- Dosya oluşturma ve işleme
- Sonuçları toplama
- Progress tracking
"""

# TODO: Parallel file processor


def process_file(filepath: str) -> Dict[str, Any]:
    """Dosyayı işle"""
    # TODO: Dosya okuma ve analiz
    pass


def parallel_file_processor(filepaths: List[str], max_workers=4) -> List[Dict]:
    """Dosyaları paralel işle"""
    # TODO: ProcessPoolExecutor ile paralel işleme
    pass


# SOLUTION:
def create_test_files(num_files=10, base_path="/tmp/test_files"):
    """Test dosyaları oluştur"""
    import os

    os.makedirs(base_path, exist_ok=True)
    filepaths = []

    for i in range(num_files):
        filepath = os.path.join(base_path, f"file_{i}.txt")

        with open(filepath, "w") as f:
            # Rastgele içerik
            num_lines = random.randint(100, 1000)
            for _ in range(num_lines):
                words = [f"word{random.randint(1, 100)}" for _ in range(random.randint(5, 20))]
                f.write(" ".join(words) + "\n")

        filepaths.append(filepath)

    return filepaths


def process_file_solution(filepath: str) -> Dict[str, Any]:
    """Dosyayı işle ve analiz et"""
    with open(filepath, "r") as f:
        content = f.read()

    lines = content.split("\n")
    words = content.split()
    chars = len(content)

    return {
        "filepath": filepath,
        "lines": len(lines),
        "words": len(words),
        "chars": chars,
        "avg_line_length": chars / len(lines) if lines else 0,
    }


def parallel_file_processor_solution(filepaths: List[str], max_workers=4) -> List[Dict]:
    """Dosyaları paralel işle"""
    results = []

    print(f"Processing {len(filepaths)} files with {max_workers} workers...")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit tüm task'ları
        future_to_path = {executor.submit(process_file_solution, path): path for path in filepaths}

        from concurrent.futures import as_completed

        completed = 0
        for future in as_completed(future_to_path):
            try:
                result = future.result()
                results.append(result)

                completed += 1
                progress = (completed / len(filepaths)) * 100
                print(f"Progress: {progress:.1f}% ({completed}/{len(filepaths)})")

            except Exception as e:
                print(f"Error processing file: {e}")

    return results


def test_parallel_file_processor():
    """File processor'ı test et"""
    # Test dosyaları oluştur
    filepaths = create_test_files(num_files=20)

    # Paralel işle
    start = time.time()
    results = parallel_file_processor_solution(filepaths, max_workers=4)
    elapsed = time.time() - start

    # Sonuçları analiz et
    total_lines = sum(r["lines"] for r in results)
    total_words = sum(r["words"] for r in results)
    total_chars = sum(r["chars"] for r in results)

    print(f"\n✓ Processing tamamlandı!")
    print(f"Süre: {elapsed:.2f}s")
    print(f"Toplam satır: {total_lines}")
    print(f"Toplam kelime: {total_words}")
    print(f"Toplam karakter: {total_chars}")

    # Temizlik
    import shutil

    shutil.rmtree("/tmp/test_files")


# ============================================================================
# ALIŞTIRMA 7: Thread-Safe Singleton (HARD)
# ============================================================================
"""
AÇIKLAMA:
Thread-safe Singleton pattern implementasyonu. Double-checked locking
kullanarak performanslı bir singleton oluşturun.

GEREKSİNİMLER:
- Singleton pattern
- Thread-safety
- Lazy initialization
- Double-checked locking
"""

# TODO: Thread-safe Singleton


class Singleton:
    # TODO: Singleton implementasyonu
    pass


# SOLUTION:
class SingletonSolution:
    """Thread-safe Singleton pattern"""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        # Double-checked locking
        if cls._instance is None:
            with cls._lock:
                # İkinci kontrol (lock içinde)
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False

        return cls._instance

    def __init__(self):
        # Sadece bir kez initialize et
        if not self._initialized:
            with self._lock:
                if not self._initialized:
                    self.data = {}
                    self.counter = 0
                    self._initialized = True

    def increment(self):
        """Thread-safe increment"""
        with self._lock:
            self.counter += 1

    def get_counter(self):
        """Counter değerini al"""
        with self._lock:
            return self.counter


def test_singleton():
    """Singleton'ı test et"""
    instances = []

    def worker():
        """Her thread bir instance alır"""
        instance = SingletonSolution()
        instances.append(id(instance))
        instance.increment()

    # 100 thread ile test
    threads = [threading.Thread(target=worker) for _ in range(100)]

    for t in threads:
        t.start()

    for t in threads:
        t.join()

    # Tüm instance'lar aynı olmalı
    unique_instances = set(instances)
    print(f"✓ Singleton test başarılı!")
    print(f"Unique instance sayısı: {len(unique_instances)} (1 olmalı)")
    print(f"Counter değeri: {SingletonSolution().get_counter()} (100 olmalı)")

    assert len(unique_instances) == 1, "Birden fazla instance oluştu!"
    assert SingletonSolution().get_counter() == 100, "Counter değeri yanlış!"


# ============================================================================
# ALIŞTIRMA 8: Barrier Synchronization (HARD)
# ============================================================================
"""
AÇIKLAMA:
Barrier kullanarak thread senkronizasyonu. Tüm thread'ler belirli bir
noktaya gelene kadar birbirlerini beklerler.

GEREKSİNİMLER:
- threading.Barrier kullanımı
- Çok aşamalı işlem
- Tüm thread'lerin senkronize çalışması
"""

# TODO: Barrier synchronization


class BarrierExample:
    def __init__(self, num_threads=5):
        # TODO: Barrier ve gerekli değişkenler
        pass

    def worker(self, worker_id):
        # TODO: Çok aşamalı işlem, her aşamada barrier
        pass


# SOLUTION:
class BarrierExampleSolution:
    """Barrier ile senkronizasyon örneği"""

    def __init__(self, num_threads=5):
        self.num_threads = num_threads
        # Her aşama için barrier
        self.barrier = threading.Barrier(num_threads)
        self.results = []
        self.results_lock = threading.Lock()

    def worker(self, worker_id):
        """Çok aşamalı işlem"""
        print(f"[Thread-{worker_id}] Faz 1 başladı")
        time.sleep(random.uniform(0.5, 2))
        print(f"[Thread-{worker_id}] Faz 1 tamamlandı, barrier'da bekleniyor...")

        # Faz 1 barrier - Tüm thread'ler buraya gelene kadar bekle
        self.barrier.wait()
        print(f"[Thread-{worker_id}] Tüm thread'ler Faz 1'i tamamladı!")

        # Faz 2
        print(f"[Thread-{worker_id}] Faz 2 başladı")
        time.sleep(random.uniform(0.5, 2))
        print(f"[Thread-{worker_id}] Faz 2 tamamlandı, barrier'da bekleniyor...")

        # Faz 2 barrier
        self.barrier.wait()
        print(f"[Thread-{worker_id}] Tüm thread'ler Faz 2'yi tamamladı!")

        # Faz 3 - Final
        print(f"[Thread-{worker_id}] Faz 3 (Final) başladı")
        result = worker_id * 10
        time.sleep(random.uniform(0.5, 1))

        with self.results_lock:
            self.results.append(result)

        print(f"[Thread-{worker_id}] Faz 3 tamamlandı")

        # Faz 3 barrier
        self.barrier.wait()
        print(f"[Thread-{worker_id}] İşlem tamamlandı!")

    def run(self):
        """Tüm worker'ları çalıştır"""
        threads = [threading.Thread(target=self.worker, args=(i,)) for i in range(self.num_threads)]

        for t in threads:
            t.start()

        for t in threads:
            t.join()

        return self.results


def test_barrier():
    """Barrier'ı test et"""
    example = BarrierExampleSolution(num_threads=5)

    print("Barrier synchronization başlıyor...\n")
    results = example.run()

    print(f"\n✓ Barrier test başarılı!")
    print(f"Sonuçlar: {results}")


# ============================================================================
# ALIŞTIRMA 9: Process Pool with Shared Memory (EXPERT)
# ============================================================================
"""
AÇIKLAMA:
Multiprocessing ile shared memory kullanarak process'ler arası veri paylaşımı.
Büyük bir array'i paralel olarak işleyin.

GEREKSİNİMLER:
- multiprocessing.Array kullanımı
- Process pool ile paralel işlem
- Shared memory'de in-place modification
- Lock ile senkronizasyon
"""

# TODO: Shared memory ile parallel processing


def process_chunk_shared(shared_array, lock, start, end):
    """Shared array'in bir bölümünü işle"""
    # TODO: Shared array'de işlem yap
    pass


def parallel_array_processor(size=1000000, num_processes=4):
    """Büyük array'i paralel işle"""
    # TODO: Shared array oluştur ve paralel işle
    pass


# SOLUTION:
def process_chunk_shared_solution(shared_array, lock, start, end, operation="square"):
    """Shared array'in bir bölümünü işle"""
    print(f"[Process-{os.getpid()}] Processing [{start}:{end}]")

    # Lock gerekmez çünkü her process farklı aralıkta çalışıyor
    # Ama güvenlik için eklenebilir
    for i in range(start, end):
        if operation == "square":
            shared_array[i] = shared_array[i] ** 2
        elif operation == "double":
            shared_array[i] = shared_array[i] * 2
        elif operation == "increment":
            shared_array[i] = shared_array[i] + 1

    print(f"[Process-{os.getpid()}] Completed [{start}:{end}]")


def parallel_array_processor_solution(size=1000000, num_processes=4):
    """Büyük array'i paralel işle"""
    # Shared array oluştur
    shared_array = multiprocessing.Array("i", size)

    # Array'i doldur
    print(f"Initializing array of size {size}...")
    for i in range(size):
        shared_array[i] = i

    # Lock (gerekirse)
    lock = multiprocessing.Lock()

    # Chunk'lara böl
    chunk_size = size // num_processes
    processes = []

    print(f"\nProcessing with {num_processes} processes...")
    start_time = time.time()

    for i in range(num_processes):
        start = i * chunk_size
        end = start + chunk_size if i < num_processes - 1 else size

        p = multiprocessing.Process(
            target=process_chunk_shared_solution, args=(shared_array, lock, start, end, "square")
        )

        processes.append(p)
        p.start()

    # Tüm process'lerin bitmesini bekle
    for p in processes:
        p.join()

    elapsed = time.time() - start_time

    # Sonuçları kontrol et
    print(f"\n✓ Processing tamamlandı!")
    print(f"Süre: {elapsed:.2f}s")
    print(f"İlk 10 eleman: {shared_array[:10]}")
    print(f"Son 10 eleman: {shared_array[-10:]}")

    return list(shared_array[:100])  # İlk 100 elemanı döndür


def test_shared_memory():
    """Shared memory'yi test et"""
    if __name__ == "__main__":
        result = parallel_array_processor_solution(size=100000, num_processes=4)
        print(f"Sonuç (ilk 100): {result[:10]}...")


# ============================================================================
# ALIŞTIRMA 10: Advanced Thread Pool (EXPERT)
# ============================================================================
"""
AÇIKLAMA:
Özelleştirilmiş thread pool implementasyonu. Dynamic resizing,
priority queue, ve graceful shutdown özellikleri ekleyin.

GEREKSİNİMLER:
- Custom thread pool
- Priority queue (yüksek öncelikli task'lar önce çalışır)
- Dynamic worker sayısı
- Graceful shutdown
- İstatistik toplama
"""

# TODO: Advanced thread pool


class AdvancedThreadPool:
    def __init__(self, min_workers=2, max_workers=10):
        # TODO: Pool yapısını oluştur
        pass

    def submit(self, func, *args, priority=5, **kwargs):
        # TODO: Priority ile task gönder
        pass

    def shutdown(self, wait=True):
        # TODO: Pool'u kapat
        pass

    def stats(self):
        # TODO: İstatistikler
        pass


# SOLUTION:
import heapq
from typing import Callable, Tuple


class AdvancedThreadPoolSolution:
    """Özelleştirilmiş thread pool"""

    def __init__(self, min_workers=2, max_workers=10):
        self.min_workers = min_workers
        self.max_workers = max_workers

        # Priority queue: (priority, counter, func, args, kwargs)
        self.task_queue = []
        self.queue_lock = threading.Lock()
        self.queue_condition = threading.Condition(self.queue_lock)

        self.workers = []
        self.shutdown_flag = False

        # İstatistikler
        self.stats_lock = threading.Lock()
        self.completed_tasks = 0
        self.failed_tasks = 0
        self.task_counter = 0

        # Initial workers oluştur
        for _ in range(min_workers):
            self._add_worker()

    def _add_worker(self):
        """Yeni worker ekle"""
        worker = threading.Thread(target=self._worker, daemon=True)
        worker.start()
        self.workers.append(worker)

    def _worker(self):
        """Worker thread"""
        while True:
            task = None

            with self.queue_condition:
                # Task bekle
                while not self.task_queue and not self.shutdown_flag:
                    self.queue_condition.wait()

                if self.shutdown_flag and not self.task_queue:
                    break

                if self.task_queue:
                    # En yüksek öncelikli task'ı al
                    _, _, func, args, kwargs = heapq.heappop(self.task_queue)
                    task = (func, args, kwargs)

            # Task'ı çalıştır
            if task:
                func, args, kwargs = task
                try:
                    func(*args, **kwargs)

                    with self.stats_lock:
                        self.completed_tasks += 1

                except Exception as e:
                    print(f"Task failed: {e}")

                    with self.stats_lock:
                        self.failed_tasks += 1

    def submit(self, func: Callable, *args, priority: int = 5, **kwargs):
        """Task gönder (düşük priority = yüksek öncelik)"""
        with self.queue_condition:
            if self.shutdown_flag:
                raise RuntimeError("Pool is shutting down")

            # Priority queue'ya ekle
            self.task_counter += 1
            heapq.heappush(self.task_queue, (priority, self.task_counter, func, args, kwargs))

            # Worker'ları uyandır
            self.queue_condition.notify()

            # Dynamic scaling: çok task varsa worker ekle
            if len(self.task_queue) > len(self.workers) * 2 and len(self.workers) < self.max_workers:
                self._add_worker()
                print(f"Added worker, total: {len(self.workers)}")

    def shutdown(self, wait=True):
        """Pool'u kapat"""
        with self.queue_condition:
            self.shutdown_flag = True
            self.queue_condition.notify_all()

        if wait:
            for worker in self.workers:
                worker.join()

    def stats(self):
        """İstatistikleri al"""
        with self.stats_lock:
            return {
                "workers": len(self.workers),
                "pending_tasks": len(self.task_queue),
                "completed": self.completed_tasks,
                "failed": self.failed_tasks,
            }


def test_advanced_thread_pool():
    """Advanced thread pool'u test et"""

    def task(task_id, duration, priority):
        """Test task'ı"""
        print(f"[Task-{task_id}] (priority={priority}) başladı")
        time.sleep(duration)
        print(f"[Task-{task_id}] (priority={priority}) tamamlandı")

    pool = AdvancedThreadPoolSolution(min_workers=2, max_workers=8)

    # Farklı önceliklerde task'lar gönder
    for i in range(20):
        priority = random.randint(1, 10)
        duration = random.uniform(0.5, 2)
        pool.submit(task, i, duration, priority, priority=priority)

    print(f"Pool stats: {pool.stats()}")

    # Biraz bekle
    time.sleep(5)

    print(f"\nMid-execution stats: {pool.stats()}")

    # Shutdown
    pool.shutdown(wait=True)

    print(f"\n✓ Advanced thread pool test başarılı!")
    print(f"Final stats: {pool.stats()}")


# ============================================================================
# ALIŞTIRMA 11: Distributed Task Queue (EXPERT)
# ============================================================================
"""
AÇIKLAMA:
Process'ler arası iletişim ile distributed task queue sistemi.
Producer process'ler task üretir, worker process'ler tüketir.

GEREKSİNİMLER:
- multiprocessing.Queue ile IPC
- Birden fazla producer ve worker process
- Task retry mekanizması
- Result collection
- Graceful shutdown
"""

# TODO: Distributed task queue


class DistributedTaskQueue:
    def __init__(self, num_workers=3):
        # TODO: Queue'ları ve process'leri oluştur
        pass

    def producer(self, producer_id, num_tasks):
        # TODO: Task üretici
        pass

    def worker(self, worker_id):
        # TODO: Task çalıştırıcı
        pass


# SOLUTION:
class DistributedTaskQueueSolution:
    """Distributed task queue sistemi"""

    def __init__(self, num_workers=3):
        self.num_workers = num_workers
        self.task_queue = multiprocessing.Queue()
        self.result_queue = multiprocessing.Queue()
        self.workers = []
        self.producers = []

    def producer(self, producer_id, num_tasks):
        """Task üretici process"""
        print(f"[Producer-{producer_id}] Başladı")

        for i in range(num_tasks):
            # Task oluştur
            task = {
                "id": f"P{producer_id}-T{i}",
                "data": random.randint(1, 100),
                "operation": random.choice(["square", "double", "triple"]),
                "retries": 0,
            }

            self.task_queue.put(task)
            print(f"[Producer-{producer_id}] Task gönderildi: {task['id']}")

            time.sleep(random.uniform(0.1, 0.5))

        print(f"[Producer-{producer_id}] Tamamlandı")

    def worker(self, worker_id):
        """Task çalıştırıcı process"""
        print(f"[Worker-{worker_id}] Başladı (PID: {os.getpid()})")

        while True:
            try:
                # Task al
                task = self.task_queue.get(timeout=2)

                if task is None:  # Poison pill
                    print(f"[Worker-{worker_id}] Poison pill alındı, kapanıyor")
                    self.task_queue.put(None)  # Diğerleri için geri koy
                    break

                # Task'ı işle
                print(f"[Worker-{worker_id}] İşleniyor: {task['id']}")

                try:
                    # İşlemi yap
                    data = task["data"]
                    operation = task["operation"]

                    # %20 başarısızlık ihtimali
                    if random.random() < 0.2:
                        raise Exception("Simulated failure")

                    if operation == "square":
                        result = data**2
                    elif operation == "double":
                        result = data * 2
                    elif operation == "triple":
                        result = data * 3

                    time.sleep(random.uniform(0.5, 1.5))

                    # Sonucu kaydet
                    self.result_queue.put({"task_id": task["id"], "result": result, "status": "success"})

                    print(f"[Worker-{worker_id}] Tamamlandı: {task['id']} = {result}")

                except Exception as e:
                    # Retry mekanizması
                    task["retries"] += 1

                    if task["retries"] < 3:
                        print(f"[Worker-{worker_id}] Hata: {task['id']}, retry {task['retries']}")
                        self.task_queue.put(task)  # Tekrar kuyruğa ekle
                    else:
                        print(f"[Worker-{worker_id}] Başarısız: {task['id']}, max retry aşıldı")
                        self.result_queue.put({"task_id": task["id"], "error": str(e), "status": "failed"})

            except queue.Empty:
                print(f"[Worker-{worker_id}] Queue boş, bekleniyor...")
                continue

        print(f"[Worker-{worker_id}] Kapatıldı")

    def start(self, num_producers=2, tasks_per_producer=10):
        """Sistemi başlat"""
        # Producer'ları başlat
        for i in range(num_producers):
            p = multiprocessing.Process(target=self.producer, args=(i, tasks_per_producer))
            p.start()
            self.producers.append(p)

        # Worker'ları başlat
        for i in range(self.num_workers):
            w = multiprocessing.Process(target=self.worker, args=(i,))
            w.start()
            self.workers.append(w)

    def wait_producers(self):
        """Producer'ların bitmesini bekle"""
        for p in self.producers:
            p.join()

    def shutdown(self):
        """Worker'ları kapat"""
        # Poison pill gönder
        self.task_queue.put(None)

        # Worker'ları bekle
        for w in self.workers:
            w.join(timeout=5)

    def get_results(self):
        """Tüm sonuçları topla"""
        results = []
        while not self.result_queue.empty():
            results.append(self.result_queue.get())
        return results


def test_distributed_task_queue():
    """Distributed task queue'yu test et"""
    if __name__ == "__main__":
        print("Distributed Task Queue başlatılıyor...\n")

        system = DistributedTaskQueueSolution(num_workers=3)
        system.start(num_producers=2, tasks_per_producer=10)

        # Producer'ların bitmesini bekle
        system.wait_producers()
        print("\nTüm producer'lar tamamlandı, worker'lar işliyor...\n")

        # Biraz bekle (worker'lar işsin)
        time.sleep(5)

        # Kapat
        system.shutdown()

        # Sonuçları topla
        results = system.get_results()

        successful = [r for r in results if r.get("status") == "success"]
        failed = [r for r in results if r.get("status") == "failed"]

        print(f"\n✓ Distributed Task Queue test tamamlandı!")
        print(f"Toplam task: {len(results)}")
        print(f"Başarılı: {len(successful)}")
        print(f"Başarısız: {len(failed)}")


# ============================================================================
# ALIŞTIRMA 12: Parallel Monte Carlo Simulation (EXPERT)
# ============================================================================
"""
AÇIKLAMA:
Monte Carlo simülasyonu ile Pi sayısını hesaplama. Multiprocessing ile
milyonlarca rastgele nokta oluşturun ve paralel hesaplayın.

GEREKSİNİMLER:
- ProcessPoolExecutor kullanımı
- Büyük ölçekli hesaplama (10M+ nokta)
- Sonuçları birleştirme
- Performans ölçümü
"""

# TODO: Parallel Monte Carlo


def monte_carlo_chunk(num_samples):
    """Monte Carlo chunk hesaplama"""
    # TODO: Rastgele noktalar oluştur ve daire içinde kaç tane olduğunu say
    pass


def parallel_monte_carlo(total_samples, num_processes=4):
    """Paralel Monte Carlo simülasyonu"""
    # TODO: Chunk'lara böl ve paralel hesapla
    pass


# SOLUTION:
def monte_carlo_chunk_solution(num_samples):
    """Monte Carlo chunk hesaplama"""
    import random

    inside_circle = 0

    for _ in range(num_samples):
        x = random.random()
        y = random.random()

        # Daire içinde mi?
        if x * x + y * y <= 1:
            inside_circle += 1

    return inside_circle


def parallel_monte_carlo_solution(total_samples=10_000_000, num_processes=4):
    """Paralel Monte Carlo simülasyonu ile Pi hesaplama"""
    print(f"Monte Carlo simülasyonu: {total_samples:,} sample")
    print(f"Process sayısı: {num_processes}\n")

    # Chunk'lara böl
    chunk_size = total_samples // num_processes
    chunks = [chunk_size] * num_processes

    # Son chunk'a kalanı ekle
    chunks[-1] += total_samples - sum(chunks)

    start = time.time()

    # Paralel hesapla
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        results = list(executor.map(monte_carlo_chunk_solution, chunks))

    elapsed = time.time() - start

    # Sonuçları birleştir
    total_inside = sum(results)

    # Pi hesapla: Pi = 4 * (daire içindeki noktalar / toplam noktalar)
    pi_estimate = 4 * total_inside / total_samples

    print(f"✓ Hesaplama tamamlandı!")
    print(f"Süre: {elapsed:.2f}s")
    print(f"Samples per second: {total_samples / elapsed:,.0f}")
    print(f"Daire içindeki noktalar: {total_inside:,}")
    print(f"Pi tahmini: {pi_estimate:.10f}")
    print(f"Gerçek Pi: {3.141592653589793:.10f}")
    print(f"Hata: {abs(pi_estimate - 3.141592653589793):.10f}")

    return pi_estimate


def test_monte_carlo():
    """Monte Carlo simülasyonunu test et"""
    if __name__ == "__main__":
        # Farklı process sayıları ile test
        for num_proc in [1, 2, 4]:
            print(f"\n{'=' * 60}")
            print(f"Testing with {num_proc} process(es)")
            print(f"{'=' * 60}")

            pi = parallel_monte_carlo_solution(total_samples=10_000_000, num_processes=num_proc)

            print()


# ============================================================================
# TEST RUNNER
# ============================================================================


def run_all_tests():
    """Tüm testleri çalıştır"""
    print("=" * 70)
    print("MULTITHREADING & MULTIPROCESSING - TEST SUITE")
    print("=" * 70)

    tests = [
        ("Thread-Safe Counter", test_thread_safe_counter),
        ("Producer-Consumer", test_producer_consumer),
        ("Parallel Web Scraper", test_parallel_scraper),
        ("CPU-Bound (Threading vs Multiprocessing)", test_cpu_bound),
        ("Thread-Safe LRU Cache", test_lru_cache),
        ("Parallel File Processor", test_parallel_file_processor),
        ("Thread-Safe Singleton", test_singleton),
        ("Barrier Synchronization", test_barrier),
        ("Advanced Thread Pool", test_advanced_thread_pool),
    ]

    for name, test_func in tests:
        print(f"\n{'=' * 70}")
        print(f"TEST: {name}")
        print(f"{'=' * 70}")

        try:
            test_func()
            print(f"\n✓ {name} - PASSED")
        except Exception as e:
            print(f"\n✗ {name} - FAILED: {e}")
            import traceback

            traceback.print_exc()

    # Multiprocessing testleri ayrı çalıştırılmalı (if __name__ == '__main__')
    print(f"\n{'=' * 70}")
    print("Multiprocessing testleri için ayrı çalıştırın:")
    print("- test_shared_memory()")
    print("- test_distributed_task_queue()")
    print("- test_monte_carlo()")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    # Threading testlerini çalıştır
    run_all_tests()

    # Multiprocessing testlerini ayrı çalıştırın:
    print("\n\n" + "=" * 70)
    print("MULTIPROCESSING TESTS")
    print("=" * 70)

    # Test 1: Shared Memory
    print("\nTest 1: Shared Memory")
    print("-" * 70)
    test_shared_memory()

    # Test 2: Distributed Task Queue
    print("\n\nTest 2: Distributed Task Queue")
    print("-" * 70)
    test_distributed_task_queue()

    # Test 3: Monte Carlo
    print("\n\nTest 3: Parallel Monte Carlo")
    print("-" * 70)
    test_monte_carlo()

    print("\n" + "=" * 70)
    print("TÜM TESTLER TAMAMLANDI!")
    print("=" * 70)
