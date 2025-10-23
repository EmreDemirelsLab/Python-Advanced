# Performance Optimization (Performans Optimizasyonu)

## İçindekiler
1. [Profiling (Profilleme)](#profiling)
2. [Memory Profiling](#memory-profiling)
3. [Algorithm Optimization](#algorithm-optimization)
4. [Caching Strategies](#caching-strategies)
5. [Lazy Evaluation](#lazy-evaluation)
6. [NumPy for Performance](#numpy-for-performance)
7. [Cython Basics](#cython-basics)
8. [Just-In-Time Compilation (Numba)](#jit-compilation)
9. [Performance Patterns](#performance-patterns)

---

## Profiling (Profilleme)

Profiling, kodun hangi kısımlarının en çok zaman aldığını belirlemek için kullanılır. Python'da birkaç güçlü profiling aracı vardır.

### 1. cProfile - Standard Profiler

```python
import cProfile
import pstats
from io import StringIO

def slow_function():
    """Yavaş çalışan örnek fonksiyon"""
    total = 0
    for i in range(1000000):
        total += i ** 2
    return total

def medium_function():
    """Orta hızda çalışan fonksiyon"""
    return sum(i ** 2 for i in range(100000))

def fast_function():
    """Hızlı çalışan fonksiyon"""
    return sum(range(10000))

def main():
    slow_function()
    medium_function()
    fast_function()

# Profiling yapma
profiler = cProfile.Profile()
profiler.enable()
main()
profiler.disable()

# Sonuçları analiz etme
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(10)  # İlk 10 sonucu göster
```

**Açıklama**: cProfile, Python'ın standart profiler'ıdır. Her fonksiyonun çağrılma sayısı ve harcadığı toplam/ortalama süreyi gösterir.

### 2. line_profiler - Satır Bazında Profiling

```python
"""
line_profiler kullanımı için önce kurulum gerekir:
pip install line-profiler

Kullanım:
1. @profile decorator ekle
2. kernprof -l -v script.py ile çalıştır
"""

# @profile  # Uncomment when using kernprof
def analyze_data(data):
    """Veri analizi fonksiyonu - satır bazında profiling"""
    # Her satırın ne kadar sürdüğünü görebiliriz
    result = []

    # Bu satır çok zaman alabilir
    squared = [x ** 2 for x in data]

    # Bu satır daha az zaman alır
    filtered = [x for x in squared if x > 100]

    # Aggregation işlemi
    total = sum(filtered)
    average = total / len(filtered) if filtered else 0

    return {
        'total': total,
        'average': average,
        'count': len(filtered)
    }

# Manuel profiling için alternatif
import time
from functools import wraps

def profile_lines(func):
    """Custom line profiler decorator"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        import sys
        import linecache

        def trace_lines(frame, event, arg):
            if event == 'line':
                lineno = frame.f_lineno
                filename = frame.f_code.co_filename
                line = linecache.getline(filename, lineno).strip()
                print(f"Line {lineno}: {line}")
            return trace_lines

        sys.settrace(trace_lines)
        result = func(*args, **kwargs)
        sys.settrace(None)
        return result
    return wrapper
```

**Açıklama**: line_profiler, fonksiyon içindeki her satırın harcadığı süreyi gösterir. Hangi satırların bottleneck olduğunu bulmak için idealdir.

### 3. py-spy - Production Profiler

```python
"""
py-spy, çalışan bir Python process'ini profiling yapar.
Avantajı: Kodu değiştirmeye gerek yok!

Kurulum:
pip install py-spy

Kullanım örnekleri:
1. Top-like interface:
   py-spy top --pid 12345

2. Flame graph oluşturma:
   py-spy record -o profile.svg --pid 12345

3. Script'i profile ederek çalıştırma:
   py-spy record -o profile.svg -- python script.py
"""

import time
import threading

def cpu_intensive_task():
    """CPU-intensive görev"""
    total = 0
    for i in range(10000000):
        total += i ** 0.5
    return total

def io_intensive_task():
    """I/O-intensive görev"""
    time.sleep(2)
    return "IO completed"

def mixed_workload():
    """Karışık workload - py-spy ile analiz için"""
    threads = []

    # CPU-intensive thread
    t1 = threading.Thread(target=cpu_intensive_task)
    threads.append(t1)

    # I/O-intensive thread
    t2 = threading.Thread(target=io_intensive_task)
    threads.append(t2)

    for t in threads:
        t.start()

    for t in threads:
        t.join()

if __name__ == "__main__":
    # Bu script'i py-spy ile profiling yapabilirsiniz
    for _ in range(5):
        mixed_workload()
```

**Açıklama**: py-spy, production ortamında çalışan Python uygulamalarını profiling yapmak için kullanılır. Kod değişikliği gerektirmez.

---

## Memory Profiling

Memory profiling, uygulamanızın ne kadar bellek kullandığını ve bellek sızıntılarını tespit etmek için kullanılır.

### 4. memory_profiler - Bellek Kullanımı Analizi

```python
"""
Kurulum:
pip install memory-profiler

Kullanım:
python -m memory_profiler script.py
"""

# from memory_profiler import profile

# @profile  # Uncomment when using memory_profiler
def memory_intensive_function():
    """Bellek yoğun fonksiyon"""
    # Büyük liste oluşturma
    large_list = [i for i in range(1000000)]

    # Liste comprehension ile kopyalama
    squared_list = [x ** 2 for x in large_list]

    # Dictionary oluşturma
    large_dict = {i: i ** 2 for i in range(1000000)}

    return len(large_list), len(large_dict)

# Manuel memory tracking
import tracemalloc

def track_memory_usage():
    """Memory kullanımını manuel olarak takip etme"""
    # Memory tracking başlat
    tracemalloc.start()

    # Başlangıç snapshot
    snapshot1 = tracemalloc.take_snapshot()

    # Memory kullanan işlemler
    data = []
    for i in range(100000):
        data.append({
            'id': i,
            'value': i ** 2,
            'text': f'Item {i}'
        })

    # Son snapshot
    snapshot2 = tracemalloc.take_snapshot()

    # Farkları analiz et
    top_stats = snapshot2.compare_to(snapshot1, 'lineno')

    print("[ Top 10 Memory Consumers ]")
    for stat in top_stats[:10]:
        print(stat)

    # Toplam bellek kullanımı
    current, peak = tracemalloc.get_traced_memory()
    print(f"\nCurrent memory: {current / 1024 / 1024:.2f} MB")
    print(f"Peak memory: {peak / 1024 / 1024:.2f} MB")

    tracemalloc.stop()
    return data
```

**Açıklama**: memory_profiler ve tracemalloc, bellek kullanımını satır bazında analiz etmeyi sağlar. Bellek sızıntılarını bulmak için kritiktir.

### 5. tracemalloc - Built-in Memory Tracking

```python
import tracemalloc
from collections import defaultdict

class MemoryLeakDetector:
    """Memory leak tespiti için yardımcı sınıf"""

    def __init__(self):
        self.snapshots = []

    def start_tracking(self):
        """Memory tracking başlat"""
        tracemalloc.start()

    def take_snapshot(self, name=""):
        """Snapshot al"""
        snapshot = tracemalloc.take_snapshot()
        self.snapshots.append((name, snapshot))
        return snapshot

    def compare_snapshots(self, idx1=0, idx2=-1):
        """İki snapshot'ı karşılaştır"""
        if len(self.snapshots) < 2:
            print("Not enough snapshots to compare")
            return

        name1, snap1 = self.snapshots[idx1]
        name2, snap2 = self.snapshots[idx2]

        top_stats = snap2.compare_to(snap1, 'lineno')

        print(f"\n[ Comparing '{name1}' vs '{name2}' ]")
        print("Top 10 memory increases:")
        for stat in top_stats[:10]:
            print(stat)

    def get_top_allocations(self, limit=10):
        """En çok bellek ayıran yerleri göster"""
        if not self.snapshots:
            print("No snapshots available")
            return

        _, snapshot = self.snapshots[-1]
        top_stats = snapshot.statistics('lineno')

        print(f"\n[ Top {limit} Memory Allocations ]")
        for stat in top_stats[:limit]:
            print(stat)

    def stop_tracking(self):
        """Tracking durdur"""
        tracemalloc.stop()

# Kullanım örneği
def simulate_memory_leak():
    """Memory leak simülasyonu"""
    leak = []

    for i in range(1000):
        # Her iterasyonda bellek birikiyor
        leak.append([0] * 10000)

    return leak

def proper_memory_usage():
    """Düzgün memory kullanımı"""
    data = []

    for i in range(1000):
        # Sadece gerekli veriyi tut
        data.append(i)

    return data

# Test
detector = MemoryLeakDetector()
detector.start_tracking()

detector.take_snapshot("Start")
result1 = proper_memory_usage()
detector.take_snapshot("After proper usage")

result2 = simulate_memory_leak()
detector.take_snapshot("After memory leak")

detector.compare_snapshots(0, 1)
detector.compare_snapshots(1, 2)
detector.get_top_allocations()
detector.stop_tracking()
```

**Açıklama**: tracemalloc, Python'ın built-in memory tracking modülüdür. Bellek kullanımını zaman içinde takip etmek ve leak'leri tespit etmek için kullanılır.

---

## Algorithm Optimization

Algoritma seçimi ve veri yapıları, performans üzerinde en büyük etkiye sahiptir.

### 6. Time Complexity Optimization

```python
from typing import List, Set
import time
from functools import wraps

def timing_decorator(func):
    """Fonksiyon çalışma süresini ölçen decorator"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"{func.__name__}: {end - start:.6f} seconds")
        return result
    return wrapper

# O(n²) - Yavaş yaklaşım
@timing_decorator
def find_duplicates_slow(numbers: List[int]) -> Set[int]:
    """O(n²) - Nested loop kullanarak duplicate bulma"""
    duplicates = set()
    n = len(numbers)

    for i in range(n):
        for j in range(i + 1, n):
            if numbers[i] == numbers[j]:
                duplicates.add(numbers[i])

    return duplicates

# O(n) - Hızlı yaklaşım
@timing_decorator
def find_duplicates_fast(numbers: List[int]) -> Set[int]:
    """O(n) - Set kullanarak duplicate bulma"""
    seen = set()
    duplicates = set()

    for num in numbers:
        if num in seen:
            duplicates.add(num)
        else:
            seen.add(num)

    return duplicates

# O(n log n) - Sorting yaklaşımı
@timing_decorator
def find_duplicates_sorted(numbers: List[int]) -> Set[int]:
    """O(n log n) - Sıralama kullanarak duplicate bulma"""
    if not numbers:
        return set()

    sorted_numbers = sorted(numbers)
    duplicates = set()

    for i in range(len(sorted_numbers) - 1):
        if sorted_numbers[i] == sorted_numbers[i + 1]:
            duplicates.add(sorted_numbers[i])

    return duplicates

# Test
test_data = list(range(10000)) * 2  # 20000 eleman, hepsi duplicate
print("Testing with 20000 elements...")
result1 = find_duplicates_fast(test_data)
result2 = find_duplicates_sorted(test_data)
# result3 = find_duplicates_slow(test_data)  # Çok yavaş, yorum satırında bırakıldı
```

**Açıklama**: Algoritma seçimi performansı büyük ölçüde etkiler. O(n) algoritma, O(n²) algoritmadan binlerce kat daha hızlıdır.

### 7. Data Structure Optimization

```python
from collections import deque, defaultdict, Counter
import bisect
from typing import Any, List

class OptimizedDataStructures:
    """Optimize edilmiş veri yapıları kullanımı"""

    @staticmethod
    @timing_decorator
    def list_vs_deque_append():
        """List vs Deque: Başa eleman ekleme"""
        # List ile (yavaş)
        data_list = []
        for i in range(10000):
            data_list.insert(0, i)  # O(n)

        return len(data_list)

    @staticmethod
    @timing_decorator
    def deque_append():
        """Deque ile başa eleman ekleme"""
        data_deque = deque()
        for i in range(10000):
            data_deque.appendleft(i)  # O(1)

        return len(data_deque)

    @staticmethod
    @timing_decorator
    def dict_vs_defaultdict():
        """Dict vs DefaultDict"""
        # Normal dict ile
        word_count = {}
        words = ['apple', 'banana', 'apple', 'cherry', 'banana', 'apple'] * 1000

        for word in words:
            if word in word_count:
                word_count[word] += 1
            else:
                word_count[word] = 1

        return word_count

    @staticmethod
    @timing_decorator
    def defaultdict_counting():
        """DefaultDict ile sayma"""
        word_count = defaultdict(int)
        words = ['apple', 'banana', 'apple', 'cherry', 'banana', 'apple'] * 1000

        for word in words:
            word_count[word] += 1

        return word_count

    @staticmethod
    @timing_decorator
    def counter_counting():
        """Counter ile sayma (en hızlı)"""
        words = ['apple', 'banana', 'apple', 'cherry', 'banana', 'apple'] * 1000
        word_count = Counter(words)
        return word_count

    @staticmethod
    @timing_decorator
    def linear_search(data: List[int], target: int) -> int:
        """Linear search - O(n)"""
        for i, value in enumerate(data):
            if value == target:
                return i
        return -1

    @staticmethod
    @timing_decorator
    def binary_search_sorted(data: List[int], target: int) -> int:
        """Binary search - O(log n)"""
        # data sorted olmalı
        idx = bisect.bisect_left(data, target)
        if idx < len(data) and data[idx] == target:
            return idx
        return -1

# Test
print("\n=== Deque vs List ===")
OptimizedDataStructures.list_vs_deque_append()
OptimizedDataStructures.deque_append()

print("\n=== Dict vs DefaultDict vs Counter ===")
OptimizedDataStructures.dict_vs_defaultdict()
OptimizedDataStructures.defaultdict_counting()
OptimizedDataStructures.counter_counting()

print("\n=== Linear vs Binary Search ===")
sorted_data = list(range(100000))
OptimizedDataStructures.linear_search(sorted_data, 99999)
OptimizedDataStructures.binary_search_sorted(sorted_data, 99999)
```

**Açıklama**: Doğru veri yapısını seçmek kritiktir. Deque, başa/sona ekleme için; Counter, sayma için; binary search, arama için optimize edilmiştir.

---

## Caching Strategies

Caching, pahalı hesaplamaların sonuçlarını saklar ve tekrar kullanır.

### 8. LRU Cache (Least Recently Used)

```python
from functools import lru_cache, wraps
import time
from typing import Dict, Any, Optional
from collections import OrderedDict

# Built-in LRU Cache
@lru_cache(maxsize=128)
def fibonacci_cached(n: int) -> int:
    """LRU cache ile Fibonacci hesaplama"""
    if n < 2:
        return n
    return fibonacci_cached(n - 1) + fibonacci_cached(n - 2)

def fibonacci_uncached(n: int) -> int:
    """Cache olmadan Fibonacci hesaplama"""
    if n < 2:
        return n
    return fibonacci_uncached(n - 1) + fibonacci_uncached(n - 2)

# Test
print("Without cache:")
start = time.perf_counter()
result1 = fibonacci_uncached(30)
print(f"Time: {time.perf_counter() - start:.4f}s, Result: {result1}")

print("\nWith LRU cache:")
start = time.perf_counter()
result2 = fibonacci_cached(30)
print(f"Time: {time.perf_counter() - start:.6f}s, Result: {result2}")

# Cache istatistikleri
print(f"\nCache info: {fibonacci_cached.cache_info()}")

# Custom LRU Cache Implementation
class LRUCache:
    """Custom LRU Cache implementasyonu"""

    def __init__(self, capacity: int):
        self.cache: OrderedDict = OrderedDict()
        self.capacity = capacity

    def get(self, key: Any) -> Optional[Any]:
        """Cache'den değer al"""
        if key not in self.cache:
            return None

        # En son kullanılan olarak işaretle
        self.cache.move_to_end(key)
        return self.cache[key]

    def put(self, key: Any, value: Any) -> None:
        """Cache'e değer ekle"""
        if key in self.cache:
            # Güncelle ve en sona taşı
            self.cache.move_to_end(key)
        else:
            # Yeni ekle
            if len(self.cache) >= self.capacity:
                # En eski elemanı çıkar (FIFO)
                self.cache.popitem(last=False)

        self.cache[key] = value

    def clear(self) -> None:
        """Cache'i temizle"""
        self.cache.clear()

    def __len__(self) -> int:
        return len(self.cache)

    def __repr__(self) -> str:
        return f"LRUCache(capacity={self.capacity}, size={len(self.cache)})"

# LRU Cache decorator
def lru_cache_custom(maxsize: int = 128):
    """Custom LRU cache decorator"""
    def decorator(func):
        cache = LRUCache(maxsize)

        @wraps(func)
        def wrapper(*args, **kwargs):
            # Key oluştur
            key = str(args) + str(sorted(kwargs.items()))

            # Cache'de var mı kontrol et
            result = cache.get(key)
            if result is not None:
                return result

            # Hesapla ve cache'le
            result = func(*args, **kwargs)
            cache.put(key, result)
            return result

        wrapper.cache = cache
        return wrapper

    return decorator

@lru_cache_custom(maxsize=100)
def expensive_computation(x: int, y: int) -> int:
    """Pahalı hesaplama simülasyonu"""
    time.sleep(0.1)  # Simüle edilmiş gecikme
    return x ** y

# Test
print("\n=== Custom LRU Cache Test ===")
start = time.perf_counter()
result = expensive_computation(2, 10)
print(f"First call: {time.perf_counter() - start:.4f}s")

start = time.perf_counter()
result = expensive_computation(2, 10)  # Cache'den gelecek
print(f"Second call (cached): {time.perf_counter() - start:.6f}s")
```

**Açıklama**: LRU Cache, en sık kullanılan değerleri bellekte tutar. Fibonacci gibi recursive fonksiyonlarda dramatik performans artışı sağlar.

### 9. TTL Cache (Time-To-Live)

```python
import time
from typing import Any, Optional, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta

@dataclass
class CacheEntry:
    """Cache entry with expiration"""
    value: Any
    expires_at: float

class TTLCache:
    """Time-to-live cache implementasyonu"""

    def __init__(self, ttl_seconds: int = 300):
        self.ttl_seconds = ttl_seconds
        self.cache: Dict[str, CacheEntry] = {}
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Optional[Any]:
        """Cache'den değer al"""
        if key not in self.cache:
            self.misses += 1
            return None

        entry = self.cache[key]

        # Expired mı kontrol et
        if time.time() > entry.expires_at:
            del self.cache[key]
            self.misses += 1
            return None

        self.hits += 1
        return entry.value

    def put(self, key: str, value: Any) -> None:
        """Cache'e değer ekle"""
        expires_at = time.time() + self.ttl_seconds
        self.cache[key] = CacheEntry(value=value, expires_at=expires_at)

    def clear(self) -> None:
        """Cache'i temizle"""
        self.cache.clear()
        self.hits = 0
        self.misses = 0

    def cleanup_expired(self) -> int:
        """Expired entry'leri temizle"""
        current_time = time.time()
        expired_keys = [
            key for key, entry in self.cache.items()
            if current_time > entry.expires_at
        ]

        for key in expired_keys:
            del self.cache[key]

        return len(expired_keys)

    def stats(self) -> Dict[str, Any]:
        """Cache istatistikleri"""
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total > 0 else 0

        return {
            'size': len(self.cache),
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': f"{hit_rate:.2f}%"
        }

def ttl_cache(ttl_seconds: int = 300):
    """TTL cache decorator"""
    cache = TTLCache(ttl_seconds)

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Key oluştur
            key = f"{func.__name__}:{args}:{sorted(kwargs.items())}"

            # Cache'de kontrol et
            result = cache.get(key)
            if result is not None:
                return result

            # Hesapla ve cache'le
            result = func(*args, **kwargs)
            cache.put(key, result)
            return result

        wrapper.cache = cache
        return wrapper

    return decorator

# API çağrısı simülasyonu
@ttl_cache(ttl_seconds=5)
def fetch_user_data(user_id: int) -> Dict[str, Any]:
    """API çağrısı simülasyonu (5 saniye TTL)"""
    time.sleep(0.5)  # Network gecikmesi simülasyonu
    return {
        'id': user_id,
        'name': f'User {user_id}',
        'timestamp': datetime.now().isoformat()
    }

# Test
print("\n=== TTL Cache Test ===")
print("First call (cache miss):")
data1 = fetch_user_data(1)
print(f"Data: {data1}")
print(f"Stats: {fetch_user_data.cache.stats()}")

print("\nSecond call (cache hit):")
data2 = fetch_user_data(1)
print(f"Data: {data2}")
print(f"Stats: {fetch_user_data.cache.stats()}")

print("\nWaiting 6 seconds for expiration...")
time.sleep(6)

print("\nThird call (cache expired, miss):")
data3 = fetch_user_data(1)
print(f"Data: {data3}")
print(f"Stats: {fetch_user_data.cache.stats()}")
```

**Açıklama**: TTL Cache, belirli bir süre sonra expire olan cache'ler için kullanılır. API çağrıları, veritabanı sorguları gibi operasyonlar için idealdir.

---

## Lazy Evaluation

Lazy evaluation, değerleri sadece ihtiyaç duyulduğunda hesaplar, böylece gereksiz hesaplamalardan kaçınır.

### 10. Generator-based Lazy Evaluation

```python
from typing import Iterator, List, Any
import sys

# Eager evaluation (tüm veriyi bellekte tutar)
def eager_range_squared(n: int) -> List[int]:
    """Eager evaluation - tüm listeyi oluşturur"""
    return [i ** 2 for i in range(n)]

# Lazy evaluation (sadece ihtiyaç duyulduğunda hesaplar)
def lazy_range_squared(n: int) -> Iterator[int]:
    """Lazy evaluation - generator kullanır"""
    for i in range(n):
        yield i ** 2

# Bellek karşılaştırması
print("=== Memory Comparison ===")
n = 1_000_000

# Eager
eager_result = eager_range_squared(n)
eager_size = sys.getsizeof(eager_result)
print(f"Eager list size: {eager_size:,} bytes ({eager_size / 1024 / 1024:.2f} MB)")

# Lazy
lazy_result = lazy_range_squared(n)
lazy_size = sys.getsizeof(lazy_result)
print(f"Lazy generator size: {lazy_size:,} bytes")

# Lazy evaluation ile pipeline
class LazyPipeline:
    """Lazy evaluation pipeline"""

    def __init__(self, data: Iterator[Any]):
        self.data = data

    def map(self, func: Callable) -> 'LazyPipeline':
        """Map operasyonu (lazy)"""
        def mapped():
            for item in self.data:
                yield func(item)
        return LazyPipeline(mapped())

    def filter(self, predicate: Callable) -> 'LazyPipeline':
        """Filter operasyonu (lazy)"""
        def filtered():
            for item in self.data:
                if predicate(item):
                    yield item
        return LazyPipeline(filtered())

    def take(self, n: int) -> 'LazyPipeline':
        """İlk n elemanı al (lazy)"""
        def taken():
            count = 0
            for item in self.data:
                if count >= n:
                    break
                yield item
                count += 1
        return LazyPipeline(taken())

    def collect(self) -> List[Any]:
        """Sonuçları topla (eager)"""
        return list(self.data)

# Pipeline kullanımı
print("\n=== Lazy Pipeline ===")
result = (
    LazyPipeline(range(1_000_000))
    .map(lambda x: x ** 2)
    .filter(lambda x: x % 3 == 0)
    .take(10)
    .collect()
)
print(f"First 10 squares divisible by 3: {result}")

# Lazy property decorator
class LazyProperty:
    """Lazy property descriptor"""

    def __init__(self, func: Callable):
        self.func = func
        self.attr_name = f"_lazy_{func.__name__}"

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self

        # Daha önce hesaplandı mı?
        if not hasattr(obj, self.attr_name):
            # İlk kez hesapla ve sakla
            value = self.func(obj)
            setattr(obj, self.attr_name, value)

        return getattr(obj, self.attr_name)

class DataProcessor:
    """Lazy property kullanan sınıf"""

    def __init__(self, data: List[int]):
        self.data = data

    @LazyProperty
    def total(self) -> int:
        """Toplam (lazy computed)"""
        print("Computing total...")
        return sum(self.data)

    @LazyProperty
    def average(self) -> float:
        """Ortalama (lazy computed)"""
        print("Computing average...")
        return sum(self.data) / len(self.data) if self.data else 0

    @LazyProperty
    def sorted_data(self) -> List[int]:
        """Sıralı veri (lazy computed)"""
        print("Sorting data...")
        return sorted(self.data)

# Test
print("\n=== Lazy Properties ===")
processor = DataProcessor([5, 2, 8, 1, 9])
print("DataProcessor created, nothing computed yet")

print(f"\nFirst access to total: {processor.total}")
print(f"Second access to total: {processor.total}")  # Cached

print(f"\nFirst access to average: {processor.average}")
print(f"Second access to average: {processor.average}")  # Cached
```

**Açıklama**: Lazy evaluation, büyük veri setleriyle çalışırken bellek kullanımını minimize eder ve gereksiz hesaplamaları önler.

---

## NumPy for Performance

NumPy, vectorized operasyonlar sayesinde Python'dan çok daha hızlı hesaplamalar yapar.

### 11. NumPy Vectorization

```python
import numpy as np
import time

# Pure Python vs NumPy karşılaştırması
def python_sum_squares(n: int) -> float:
    """Pure Python ile sum of squares"""
    return sum(i ** 2 for i in range(n))

def numpy_sum_squares(n: int) -> float:
    """NumPy ile sum of squares"""
    arr = np.arange(n)
    return np.sum(arr ** 2)

# Benchmark
n = 1_000_000
print("=== Python vs NumPy ===")

start = time.perf_counter()
py_result = python_sum_squares(n)
py_time = time.perf_counter() - start
print(f"Python: {py_time:.4f}s")

start = time.perf_counter()
np_result = numpy_sum_squares(n)
np_time = time.perf_counter() - start
print(f"NumPy: {np_time:.4f}s")
print(f"Speedup: {py_time / np_time:.2f}x")

# Vectorized operations
class VectorizedOperations:
    """NumPy vectorized operasyonları"""

    @staticmethod
    @timing_decorator
    def distance_calculation_python(points1: List, points2: List) -> List[float]:
        """Python ile Euclidean distance"""
        distances = []
        for p1, p2 in zip(points1, points2):
            dist = ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5
            distances.append(dist)
        return distances

    @staticmethod
    @timing_decorator
    def distance_calculation_numpy(points1: np.ndarray, points2: np.ndarray) -> np.ndarray:
        """NumPy ile Euclidean distance (vectorized)"""
        return np.sqrt(np.sum((points1 - points2) ** 2, axis=1))

    @staticmethod
    @timing_decorator
    def matrix_operations_python(size: int):
        """Python ile matrix çarpımı"""
        matrix_a = [[i + j for j in range(size)] for i in range(size)]
        matrix_b = [[i - j for j in range(size)] for i in range(size)]

        result = [[0] * size for _ in range(size)]
        for i in range(size):
            for j in range(size):
                for k in range(size):
                    result[i][j] += matrix_a[i][k] * matrix_b[k][j]

        return result

    @staticmethod
    @timing_decorator
    def matrix_operations_numpy(size: int):
        """NumPy ile matrix çarpımı"""
        matrix_a = np.arange(size * size).reshape(size, size)
        matrix_b = np.arange(size * size).reshape(size, size)

        return np.dot(matrix_a, matrix_b)

# Test
print("\n=== Distance Calculation ===")
n_points = 100000
points1_py = [(i, i * 2) for i in range(n_points)]
points2_py = [(i + 1, i * 2 + 1) for i in range(n_points)]

points1_np = np.array(points1_py)
points2_np = np.array(points2_py)

VectorizedOperations.distance_calculation_python(points1_py, points2_py)
VectorizedOperations.distance_calculation_numpy(points1_np, points2_np)

print("\n=== Matrix Multiplication ===")
matrix_size = 200
VectorizedOperations.matrix_operations_python(matrix_size)
VectorizedOperations.matrix_operations_numpy(matrix_size)

# Broadcasting
print("\n=== NumPy Broadcasting ===")
# Her satıra farklı değer eklemek
matrix = np.arange(12).reshape(3, 4)
print("Original matrix:")
print(matrix)

# Broadcasting ile tek satırda
row_to_add = np.array([1, 2, 3, 4])
result = matrix + row_to_add
print("\nAfter adding [1,2,3,4] to each row:")
print(result)

# Her sütuna farklı değer eklemek
col_to_add = np.array([[10], [20], [30]])
result = matrix + col_to_add
print("\nAfter adding [10,20,30] to each column:")
print(result)
```

**Açıklama**: NumPy, C ile yazılmış vectorized operasyonlar kullanır. Pure Python'a göre 10-100x daha hızlıdır.

### 12. NumPy Memory Optimization

```python
import numpy as np

class NumPyMemoryOptimization:
    """NumPy memory optimizasyon teknikleri"""

    @staticmethod
    def dtype_optimization():
        """Doğru dtype seçimi ile memory tasarrufu"""
        n = 1_000_000

        # Default dtype (float64)
        arr_float64 = np.arange(n, dtype=np.float64)

        # Optimize edilmiş dtype (float32)
        arr_float32 = np.arange(n, dtype=np.float32)

        # Integer kullanımı (int32)
        arr_int32 = np.arange(n, dtype=np.int32)

        # Integer kullanımı (int16 - küçük sayılar için)
        arr_int16 = np.arange(min(n, 32767), dtype=np.int16)

        print("=== Memory Usage by dtype ===")
        print(f"float64: {arr_float64.nbytes / 1024 / 1024:.2f} MB")
        print(f"float32: {arr_float32.nbytes / 1024 / 1024:.2f} MB")
        print(f"int32: {arr_int32.nbytes / 1024 / 1024:.2f} MB")
        print(f"int16: {arr_int16.nbytes / 1024:.2f} KB")

    @staticmethod
    def view_vs_copy():
        """View vs Copy - memory efficiency"""
        arr = np.arange(1000000)

        # View (memory paylaşır)
        view = arr[::2]  # Slicing bir view oluşturur
        print(f"\nOriginal array shares memory with view: {np.shares_memory(arr, view)}")

        # Copy (yeni memory allocate eder)
        copy = arr[::2].copy()
        print(f"Original array shares memory with copy: {np.shares_memory(arr, copy)}")

        # View kullanımının avantajı
        print(f"\nView overhead: {view.nbytes / 1024 / 1024:.2f} MB (shares memory)")
        print(f"Copy overhead: {copy.nbytes / 1024 / 1024:.2f} MB (new allocation)")

    @staticmethod
    def in_place_operations():
        """In-place operasyonlar ile memory tasarrufu"""
        arr = np.random.rand(1000000)

        print("\n=== In-place Operations ===")

        # Yeni array oluşturur (2x memory)
        start = time.perf_counter()
        result = arr * 2
        print(f"New array: {time.perf_counter() - start:.6f}s")

        # In-place, aynı array'i değiştirir (1x memory)
        start = time.perf_counter()
        arr *= 2
        print(f"In-place: {time.perf_counter() - start:.6f}s")

# Test
NumPyMemoryOptimization.dtype_optimization()
NumPyMemoryOptimization.view_vs_copy()
NumPyMemoryOptimization.in_place_operations()
```

**Açıklama**: NumPy'da doğru dtype seçimi, view kullanımı ve in-place operasyonlar bellek kullanımını optimize eder.

---

## Cython Basics

Cython, Python kodunu C'ye çevirerek performansı artırır.

### 13. Cython Optimizasyon Örneği

```python
"""
Cython kullanımı için:
1. Kurulum: pip install cython
2. .pyx dosyası oluştur
3. setup.py ile compile et
4. Import edip kullan

# example.pyx dosyası:
def fibonacci_cython(int n):
    cdef int a = 0, b = 1, i
    for i in range(n):
        a, b = b, a + b
    return a

# setup.py dosyası:
from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize("example.pyx")
)

# Compile:
python setup.py build_ext --inplace
"""

# Pure Python versiyonu
def fibonacci_python(n: int) -> int:
    """Pure Python Fibonacci"""
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a

# Cython benzeri optimizasyon (type hints)
def fibonacci_typed(n: int) -> int:
    """Type hints ile optimize edilmiş versiyon"""
    a: int = 0
    b: int = 1
    for _ in range(n):
        a, b = b, a + b
    return a

# Cython optimizasyon önerileri
"""
1. Değişken tiplerini tanımlayın (cdef):
   cdef int x = 5
   cdef double y = 3.14
   cdef list items = []

2. Fonksiyon parametrelerini tipleyin:
   def func(int x, double y):
       ...

3. C fonksiyonlarını kullanın:
   cdef int internal_func(int x):  # C fonksiyonu
       return x * 2

   def public_func(int x):  # Python'dan çağrılabilir
       return internal_func(x)

4. NumPy array'leri için typed memoryviews:
   def process_array(double[:] arr):
       cdef int i
       cdef double total = 0
       for i in range(arr.shape[0]):
           total += arr[i]
       return total

5. Bounds checking'i devre dışı bırakın (dikkatli):
   @cython.boundscheck(False)
   @cython.wraparound(False)
   def fast_function(double[:] arr):
       ...
"""

# Performans karşılaştırması
print("=== Fibonacci Performance ===")
n = 100000

start = time.perf_counter()
result_py = fibonacci_python(n)
py_time = time.perf_counter() - start
print(f"Python: {py_time:.6f}s")

start = time.perf_counter()
result_typed = fibonacci_typed(n)
typed_time = time.perf_counter() - start
print(f"Typed Python: {typed_time:.6f}s")

print("\nNote: Cython ile 10-100x speedup elde edilebilir")
print("Gerçek Cython .pyx dosyası compile edildiğinde çok daha hızlı olur")
```

**Açıklama**: Cython, Python kodunu C'ye çevirerek önemli performans artışı sağlar. Özellikle döngüler ve sayısal hesaplamalar için etkilidir.

---

## Just-In-Time Compilation (Numba)

Numba, Python fonksiyonlarını runtime'da makine koduna compile eder.

### 14. Numba JIT Optimization

```python
"""
Numba kurulumu:
pip install numba
"""

# Numba import (opsiyonel, yüklü değilse normal Python kullanır)
try:
    from numba import jit, njit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    print("Numba not installed. Install with: pip install numba")

    # Dummy decorator
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

    njit = jit
    prange = range

# Pure Python
def monte_carlo_pi_python(n: int) -> float:
    """Monte Carlo ile Pi hesaplama (Python)"""
    inside = 0
    for i in range(n):
        x = np.random.random()
        y = np.random.random()
        if x*x + y*y <= 1.0:
            inside += 1
    return 4.0 * inside / n

# Numba JIT
@njit
def monte_carlo_pi_numba(n: int) -> float:
    """Monte Carlo ile Pi hesaplama (Numba JIT)"""
    inside = 0
    for i in range(n):
        x = np.random.random()
        y = np.random.random()
        if x*x + y*y <= 1.0:
            inside += 1
    return 4.0 * inside / n

# Numba parallel
@njit(parallel=True)
def monte_carlo_pi_parallel(n: int) -> float:
    """Monte Carlo ile Pi hesaplama (Numba Parallel)"""
    inside = 0
    for i in prange(n):
        x = np.random.random()
        y = np.random.random()
        if x*x + y*y <= 1.0:
            inside += 1
    return 4.0 * inside / n

if NUMBA_AVAILABLE:
    print("=== Monte Carlo Pi Calculation ===")
    n = 10_000_000

    print("Python version:")
    start = time.perf_counter()
    pi_python = monte_carlo_pi_python(n)
    py_time = time.perf_counter() - start
    print(f"Pi ≈ {pi_python:.6f}, Time: {py_time:.4f}s")

    print("\nNumba JIT version:")
    # İlk çağrı compilation içerir
    _ = monte_carlo_pi_numba(100)  # Warm-up
    start = time.perf_counter()
    pi_numba = monte_carlo_pi_numba(n)
    numba_time = time.perf_counter() - start
    print(f"Pi ≈ {pi_numba:.6f}, Time: {numba_time:.4f}s")
    print(f"Speedup: {py_time / numba_time:.2f}x")

    print("\nNumba Parallel version:")
    _ = monte_carlo_pi_parallel(100)  # Warm-up
    start = time.perf_counter()
    pi_parallel = monte_carlo_pi_parallel(n)
    parallel_time = time.perf_counter() - start
    print(f"Pi ≈ {pi_parallel:.6f}, Time: {parallel_time:.4f}s")
    print(f"Speedup: {py_time / parallel_time:.2f}x")

# Numba optimizasyon önerileri
"""
1. @jit decorator kullanın:
   @jit
   def my_function(x):
       return x ** 2

2. nopython mode (@njit) - daha hızlı:
   @njit
   def my_function(x):
       return x ** 2

3. Parallel execution:
   @njit(parallel=True)
   def my_function(arr):
       for i in prange(len(arr)):
           arr[i] = arr[i] ** 2

4. FastMath (daha az kesin, daha hızlı):
   @njit(fastmath=True)
   def my_function(x):
       return np.sqrt(x)

5. Cache compilation results:
   @njit(cache=True)
   def my_function(x):
       return x ** 2

Numba desteklenen tipler:
- NumPy arrays
- Numerical types (int, float, complex)
- Tuples
- Basic control flow

Numba desteklenmeyen:
- Lists (NumPy arrays kullanın)
- Dictionaries
- Custom classes (çoğu durumda)
- String operations
"""
```

**Açıklama**: Numba, Python fonksiyonlarını JIT compilation ile optimize eder. NumPy operasyonları ve numerical computing için mükemmeldir.

---

## Performance Patterns

Production ortamında kullanılan performans pattern'leri.

### 15. Connection Pooling

```python
from typing import List, Optional, Any
import time
from queue import Queue, Empty
from threading import Lock
from contextlib import contextmanager

class Connection:
    """Simulated database connection"""

    def __init__(self, conn_id: int):
        self.conn_id = conn_id
        self.in_use = False
        # Simulate connection overhead
        time.sleep(0.1)

    def execute(self, query: str) -> Any:
        """Execute a query"""
        time.sleep(0.01)  # Simulated query time
        return f"Result from connection {self.conn_id}"

    def close(self):
        """Close connection"""
        pass

class ConnectionPool:
    """Connection pool implementation"""

    def __init__(self, min_size: int = 5, max_size: int = 20):
        self.min_size = min_size
        self.max_size = max_size
        self.pool: Queue = Queue(maxsize=max_size)
        self.size = 0
        self.lock = Lock()

        # Initialize minimum connections
        for _ in range(min_size):
            self._create_connection()

    def _create_connection(self) -> Connection:
        """Create a new connection"""
        with self.lock:
            if self.size >= self.max_size:
                raise Exception("Maximum pool size reached")

            conn = Connection(self.size)
            self.size += 1
            return conn

    def get_connection(self, timeout: float = 5.0) -> Connection:
        """Get a connection from pool"""
        try:
            # Try to get existing connection
            conn = self.pool.get(timeout=timeout)
            return conn
        except Empty:
            # Create new connection if pool is empty and size < max
            if self.size < self.max_size:
                return self._create_connection()
            raise Exception("No connections available")

    def return_connection(self, conn: Connection):
        """Return connection to pool"""
        self.pool.put(conn)

    @contextmanager
    def connection(self):
        """Context manager for connection"""
        conn = self.get_connection()
        try:
            yield conn
        finally:
            self.return_connection(conn)

    def close_all(self):
        """Close all connections"""
        while not self.pool.empty():
            try:
                conn = self.pool.get_nowait()
                conn.close()
            except Empty:
                break

# Kullanım örneği
print("=== Connection Pooling ===")

# Without pooling
start = time.perf_counter()
for i in range(10):
    conn = Connection(i)
    conn.execute("SELECT * FROM users")
    conn.close()
no_pool_time = time.perf_counter() - start
print(f"Without pooling: {no_pool_time:.4f}s")

# With pooling
pool = ConnectionPool(min_size=5, max_size=10)
start = time.perf_counter()
for i in range(10):
    with pool.connection() as conn:
        conn.execute("SELECT * FROM users")
pool_time = time.perf_counter() - start
print(f"With pooling: {pool_time:.4f}s")
print(f"Speedup: {no_pool_time / pool_time:.2f}x")

pool.close_all()
```

**Açıklama**: Connection pooling, veritabanı bağlantıları gibi pahalı kaynakları yeniden kullanır. Her seferinde yeni bağlantı oluşturmaktan kaçınır.

### 16. Batch Processing

```python
from typing import List, Callable, Any, Iterator
from collections import deque

class BatchProcessor:
    """Batch processing for efficiency"""

    def __init__(self, batch_size: int = 100):
        self.batch_size = batch_size
        self.buffer: deque = deque()

    def add(self, item: Any):
        """Add item to buffer"""
        self.buffer.append(item)

    def process_batch(self, processor: Callable[[List[Any]], None]):
        """Process accumulated batch"""
        if len(self.buffer) >= self.batch_size:
            batch = [self.buffer.popleft() for _ in range(self.batch_size)]
            processor(batch)

    def flush(self, processor: Callable[[List[Any]], None]):
        """Process remaining items"""
        while self.buffer:
            batch_size = min(len(self.buffer), self.batch_size)
            batch = [self.buffer.popleft() for _ in range(batch_size)]
            processor(batch)

    @staticmethod
    def batch_iterator(items: Iterator[Any], batch_size: int) -> Iterator[List[Any]]:
        """Create batches from iterator"""
        batch = []
        for item in items:
            batch.append(item)
            if len(batch) >= batch_size:
                yield batch
                batch = []

        if batch:
            yield batch

# Simulated database operations
def save_to_db_single(item: Any):
    """Save single item to database"""
    time.sleep(0.01)  # Simulated DB latency

def save_to_db_batch(items: List[Any]):
    """Save batch to database (more efficient)"""
    time.sleep(0.05)  # Simulated DB latency (much less than N * 0.01)

# Test
print("\n=== Batch Processing ===")
items = list(range(1000))

# Without batching
start = time.perf_counter()
for item in items[:100]:  # Only 100 for demo
    save_to_db_single(item)
no_batch_time = time.perf_counter() - start
print(f"Without batching (100 items): {no_batch_time:.4f}s")

# With batching
processor = BatchProcessor(batch_size=10)
start = time.perf_counter()
for item in items[:100]:
    processor.add(item)
    processor.process_batch(save_to_db_batch)
processor.flush(save_to_db_batch)
batch_time = time.perf_counter() - start
print(f"With batching (100 items): {batch_time:.4f}s")
print(f"Speedup: {no_batch_time / batch_time:.2f}x")

# Batch iterator kullanımı
print("\n=== Batch Iterator ===")
for batch in BatchProcessor.batch_iterator(range(25), batch_size=10):
    print(f"Processing batch of {len(batch)} items: {batch}")
```

**Açıklama**: Batch processing, birden fazla işlemi gruplar ve tek seferde işler. Network/disk I/O overhead'ini önemli ölçüde azaltır.

### 17. Async I/O for Performance

```python
import asyncio
from typing import List
import aiohttp

# Senkron yaklaşım (yavaş)
def fetch_url_sync(url: str) -> str:
    """Senkron URL fetch (simülasyon)"""
    time.sleep(1)  # Simulated network delay
    return f"Content from {url}"

def fetch_all_sync(urls: List[str]) -> List[str]:
    """Tüm URL'leri senkron olarak fetch et"""
    results = []
    for url in urls:
        results.append(fetch_url_sync(url))
    return results

# Async yaklaşım (hızlı)
async def fetch_url_async(url: str) -> str:
    """Async URL fetch (simülasyon)"""
    await asyncio.sleep(1)  # Simulated network delay
    return f"Content from {url}"

async def fetch_all_async(urls: List[str]) -> List[str]:
    """Tüm URL'leri async olarak fetch et"""
    tasks = [fetch_url_async(url) for url in urls]
    results = await asyncio.gather(*tasks)
    return results

# Test
print("\n=== Async I/O Performance ===")
urls = [f"https://example.com/page{i}" for i in range(5)]

# Senkron
print("Synchronous approach:")
start = time.perf_counter()
results_sync = fetch_all_sync(urls)
sync_time = time.perf_counter() - start
print(f"Time: {sync_time:.4f}s")

# Async
print("\nAsynchronous approach:")
start = time.perf_counter()
results_async = asyncio.run(fetch_all_async(urls))
async_time = time.perf_counter() - start
print(f"Time: {async_time:.4f}s")
print(f"Speedup: {sync_time / async_time:.2f}x")

# Real-world async HTTP example (requires aiohttp)
"""
async def fetch_real_url(session, url):
    async with session.get(url) as response:
        return await response.text()

async def fetch_multiple_urls(urls):
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_real_url(session, url) for url in urls]
        results = await asyncio.gather(*tasks)
        return results

# Usage:
urls = ['https://api.github.com/users/python', ...]
results = asyncio.run(fetch_multiple_urls(urls))
"""
```

**Açıklama**: Async I/O, I/O-bound işlemler için idealdir. Network requests, file I/O gibi işlemleri paralel olarak gerçekleştirir.

### 18. String Optimization

```python
# String concatenation optimization
@timing_decorator
def string_concat_slow(n: int) -> str:
    """Yavaş string concatenation (+ operatörü)"""
    result = ""
    for i in range(n):
        result += str(i)  # Her seferinde yeni string oluşturur
    return result

@timing_decorator
def string_concat_fast(n: int) -> str:
    """Hızlı string concatenation (join)"""
    parts = []
    for i in range(n):
        parts.append(str(i))
    return "".join(parts)  # Tek seferde birleştirir

@timing_decorator
def string_concat_fastest(n: int) -> str:
    """En hızlı (generator + join)"""
    return "".join(str(i) for i in range(n))

# Test
print("\n=== String Concatenation ===")
n = 10000
string_concat_slow(n)
string_concat_fast(n)
string_concat_fastest(n)

# String formatting optimization
@timing_decorator
def format_with_percent(n: int):
    """% formatting"""
    for i in range(n):
        s = "Value: %d, Name: %s" % (i, f"item_{i}")

@timing_decorator
def format_with_format(n: int):
    """str.format()"""
    for i in range(n):
        s = "Value: {}, Name: {}".format(i, f"item_{i}")

@timing_decorator
def format_with_fstring(n: int):
    """f-string (en hızlı)"""
    for i in range(n):
        s = f"Value: {i}, Name: item_{i}"

print("\n=== String Formatting ===")
n = 100000
format_with_percent(n)
format_with_format(n)
format_with_fstring(n)
```

**Açıklama**: String operasyonları Python'da sık yapılır. join() kullanımı ve f-string'ler, + operatörü ve % formatting'den çok daha hızlıdır.

### 19. Dictionary and Set Optimization

```python
# Membership testing
@timing_decorator
def membership_list(items: List[int], search_items: List[int]):
    """List ile membership test (O(n))"""
    results = []
    for item in search_items:
        results.append(item in items)
    return results

@timing_decorator
def membership_set(items: Set[int], search_items: List[int]):
    """Set ile membership test (O(1))"""
    results = []
    for item in search_items:
        results.append(item in items)
    return results

print("\n=== Membership Testing ===")
data_list = list(range(10000))
data_set = set(data_list)
search_items = [100, 5000, 9999] * 1000

membership_list(data_list, search_items)
membership_set(data_set, search_items)

# Dictionary get vs exception handling
@timing_decorator
def dict_get_exception(d: dict, keys: List[str]):
    """Exception handling ile"""
    results = []
    for key in keys:
        try:
            results.append(d[key])
        except KeyError:
            results.append(None)
    return results

@timing_decorator
def dict_get_method(d: dict, keys: List[str]):
    """get() method ile (daha hızlı)"""
    results = []
    for key in keys:
        results.append(d.get(key))
    return results

print("\n=== Dictionary Access ===")
data_dict = {f"key_{i}": i for i in range(1000)}
search_keys = [f"key_{i}" for i in range(0, 2000, 2)]  # Half missing

dict_get_exception(data_dict, search_keys)
dict_get_method(data_dict, search_keys)
```

**Açıklama**: Set ve dict, hash-based veri yapılarıdır. Membership testing ve lookup işlemleri için list'ten çok daha hızlıdır.

### 20. Comprehension vs Loop Optimization

```python
# List comprehension vs loop
@timing_decorator
def create_list_loop(n: int) -> List[int]:
    """For loop ile liste oluşturma"""
    result = []
    for i in range(n):
        result.append(i ** 2)
    return result

@timing_decorator
def create_list_comprehension(n: int) -> List[int]:
    """List comprehension (daha hızlı)"""
    return [i ** 2 for i in range(n)]

@timing_decorator
def create_list_map(n: int) -> List[int]:
    """Map ile oluşturma"""
    return list(map(lambda x: x ** 2, range(n)))

print("\n=== List Creation ===")
n = 1_000_000
create_list_loop(n)
create_list_comprehension(n)
create_list_map(n)

# Filter operations
@timing_decorator
def filter_loop(data: List[int]) -> List[int]:
    """For loop ile filtreleme"""
    result = []
    for item in data:
        if item % 2 == 0:
            result.append(item)
    return result

@timing_decorator
def filter_comprehension(data: List[int]) -> List[int]:
    """List comprehension ile filtreleme (daha hızlı)"""
    return [item for item in data if item % 2 == 0]

@timing_decorator
def filter_builtin(data: List[int]) -> List[int]:
    """Built-in filter"""
    return list(filter(lambda x: x % 2 == 0, data))

print("\n=== Filtering ===")
data = list(range(1_000_000))
filter_loop(data)
filter_comprehension(data)
filter_builtin(data)
```

**Açıklama**: List comprehension, C seviyesinde optimize edilmiştir. Normal for loop'tan %20-30 daha hızlıdır.

---

## Özet ve Best Practices

### Performans Optimizasyon Stratejisi

1. **Önce Profiling Yapın**: Optimize etmeden önce bottleneck'leri bulun
2. **Algoritma Seçimi**: En büyük kazanç doğru algoritma seçiminden gelir
3. **Veri Yapıları**: İşleminize uygun veri yapısını seçin
4. **Caching**: Pahalı hesaplamaları cache'leyin
5. **Lazy Evaluation**: Gereksiz hesaplamalardan kaçının
6. **Vectorization**: NumPy ile vectorized operasyonlar kullanın
7. **JIT/Cython**: Critical path'leri compile edin
8. **Async I/O**: I/O-bound işlemler için async kullanın
9. **Batch Processing**: Birden fazla işlemi grupla
10. **Built-in Kullanımı**: Python built-in'leri C ile yazılmıştır, kullanın

### Production Checklist

- [ ] Profiling yapıldı (cProfile, line_profiler)
- [ ] Memory profiling yapıldı (memory_profiler, tracemalloc)
- [ ] Algoritma complexity analizi yapıldı
- [ ] Uygun veri yapıları seçildi
- [ ] Caching stratejisi uygulandı
- [ ] NumPy vectorization kullanıldı (sayısal hesaplamalar için)
- [ ] Critical path'ler Cython/Numba ile optimize edildi
- [ ] Async I/O kullanıldı (I/O-bound işlemler için)
- [ ] Connection pooling uygulandı
- [ ] Batch processing kullanıldı
- [ ] String operasyonları optimize edildi
- [ ] List comprehension kullanıldı
- [ ] Performance regression testleri yazıldı

### Performans İyileştirme Sırası

1. **Algorithm** (100-1000x improvement)
2. **Data Structures** (10-100x improvement)
3. **Caching** (10-100x improvement)
4. **Compilation (Cython/Numba)** (10-100x improvement)
5. **Vectorization (NumPy)** (10-50x improvement)
6. **Code-level optimizations** (2-5x improvement)

**Altın Kural**: Premature optimization is the root of all evil. Önce çalışan kod yazın, sonra profiling yapıp optimize edin.
