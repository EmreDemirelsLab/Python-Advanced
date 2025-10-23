"""
PERFORMANCE OPTIMIZATION EXERCISES
Her egzersiz için TODO kısmını tamamlayın, ardından solution ile karşılaştırın.
Performance benchmark'ları dahildir.
"""

import time
import sys
from typing import List, Dict, Any, Callable, Iterator
from functools import wraps
from collections import defaultdict, Counter, deque
import tracemalloc

# Utility: Timing decorator
def benchmark(func: Callable) -> Callable:
    """Fonksiyon performansını ölçen decorator"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"{func.__name__}: {end - start:.6f} seconds")
        return result
    return wrapper


# ============================================================================
# EXERCISE 1: Fibonacci with Memoization (Medium)
# ============================================================================
print("=" * 70)
print("EXERCISE 1: Fibonacci with Memoization")
print("=" * 70)
print("Görev: LRU cache kullanarak Fibonacci hesaplamasını optimize edin")
print()

# TODO: @lru_cache decorator ekleyerek bu fonksiyonu optimize edin
def fibonacci_todo(n: int) -> int:
    """
    TODO: functools.lru_cache kullanarak optimize edin

    Fibonacci sayısını hesaplar. Şu anda çok yavaş (exponential time).
    LRU cache ekleyerek dramatik hızlanma sağlayın.
    """
    if n < 2:
        return n
    return fibonacci_todo(n - 1) + fibonacci_todo(n - 2)


# SOLUTION
from functools import lru_cache

@lru_cache(maxsize=None)
def fibonacci_solution(n: int) -> int:
    """
    SOLUTION: LRU cache ile optimize edilmiş Fibonacci

    Cache sayesinde her değer sadece bir kez hesaplanır.
    Time complexity: O(n) instead of O(2^n)
    """
    if n < 2:
        return n
    return fibonacci_solution(n - 1) + fibonacci_solution(n - 2)


# Test & Benchmark
print("\n--- Benchmark ---")
# fibonacci_todo(35)  # Çok yavaş, yorum satırında
# benchmark(lambda: fibonacci_todo(35))()

@benchmark
def test_fibonacci_solution():
    return fibonacci_solution(35)

result = test_fibonacci_solution()
print(f"Result: {result}")
print(f"Cache info: {fibonacci_solution.cache_info()}")


# ============================================================================
# EXERCISE 2: Optimize List Operations (Medium)
# ============================================================================
print("\n" + "=" * 70)
print("EXERCISE 2: List Operations Optimization")
print("=" * 70)
print("Görev: Liste işlemlerini optimize edin - başa eleman ekleme")
print()

# TODO: Bu fonksiyonu deque kullanarak optimize edin
def prepend_items_todo(n: int) -> List[int]:
    """
    TODO: list yerine collections.deque kullanarak optimize edin

    list.insert(0, item) her seferinde O(n) time alır.
    Daha iyi bir veri yapısı seçin.
    """
    result = []
    for i in range(n):
        result.insert(0, i)  # O(n) operation!
    return result


# SOLUTION
def prepend_items_solution(n: int) -> List[int]:
    """
    SOLUTION: deque kullanarak O(1) prepend

    deque.appendleft() O(1) time complexity'ye sahiptir.
    List'e göre çok daha hızlıdır.
    """
    from collections import deque
    result = deque()
    for i in range(n):
        result.appendleft(i)  # O(1) operation
    return list(result)


# Test & Benchmark
print("\n--- Benchmark ---")
n = 10000

# TODO versiyonu yavaş, küçük n ile test
# @benchmark
# def test_todo():
#     return prepend_items_todo(1000)
# test_todo()

@benchmark
def test_solution():
    return prepend_items_solution(n)

test_solution()


# ============================================================================
# EXERCISE 3: Optimize Dictionary Lookup (Medium)
# ============================================================================
print("\n" + "=" * 70)
print("EXERCISE 3: Dictionary Lookup Optimization")
print("=" * 70)
print("Görev: Exception handling yerine dict.get() kullanarak optimize edin")
print()

# TODO: Exception handling yerine dict.get() kullanın
def safe_dict_access_todo(data: Dict[str, int], keys: List[str]) -> List[int]:
    """
    TODO: try-except yerine dict.get() kullanarak optimize edin

    Exception handling pahalıdır. dict.get() çok daha hızlıdır.
    Bulunmayan key'ler için 0 döndürün.
    """
    results = []
    for key in keys:
        try:
            results.append(data[key])
        except KeyError:
            results.append(0)
    return results


# SOLUTION
def safe_dict_access_solution(data: Dict[str, int], keys: List[str]) -> List[int]:
    """
    SOLUTION: dict.get() ile optimize edilmiş erişim

    dict.get(key, default) exception handling'den çok daha hızlıdır.
    """
    results = []
    for key in keys:
        results.append(data.get(key, 0))
    return results

# Daha da optimize: list comprehension
def safe_dict_access_optimized(data: Dict[str, int], keys: List[str]) -> List[int]:
    """En optimize versiyonu: list comprehension"""
    return [data.get(key, 0) for key in keys]


# Test & Benchmark
print("\n--- Benchmark ---")
test_data = {f"key_{i}": i for i in range(1000)}
test_keys = [f"key_{i}" for i in range(2000)]  # Yarısı missing

@benchmark
def test_todo():
    return safe_dict_access_todo(test_data, test_keys)

@benchmark
def test_solution():
    return safe_dict_access_solution(test_data, test_keys)

@benchmark
def test_optimized():
    return safe_dict_access_optimized(test_data, test_keys)

test_todo()
test_solution()
test_optimized()


# ============================================================================
# EXERCISE 4: String Concatenation Optimization (Medium)
# ============================================================================
print("\n" + "=" * 70)
print("EXERCISE 4: String Concatenation Optimization")
print("=" * 70)
print("Görev: String concatenation'ı join() ile optimize edin")
print()

# TODO: + operatörü yerine join() kullanın
def build_csv_todo(data: List[List[str]]) -> str:
    """
    TODO: + operatörü yerine str.join() kullanarak optimize edin

    String concatenation + ile çok yavaştır (her seferinde yeni string).
    join() kullanarak optimize edin.
    """
    result = ""
    for row in data:
        line = ""
        for i, item in enumerate(row):
            line += item
            if i < len(row) - 1:
                line += ","
        result += line + "\n"
    return result


# SOLUTION
def build_csv_solution(data: List[List[str]]) -> str:
    """
    SOLUTION: join() ile optimize edilmiş string building

    join() tek seferde birleştirme yapar, çok daha hızlıdır.
    """
    lines = []
    for row in data:
        line = ",".join(row)
        lines.append(line)
    return "\n".join(lines)

# Daha da optimize: nested comprehension
def build_csv_optimized(data: List[List[str]]) -> str:
    """En optimize versiyonu: nested comprehension"""
    return "\n".join(",".join(row) for row in data)


# Test & Benchmark
print("\n--- Benchmark ---")
test_data = [[f"col{j}_{i}" for j in range(10)] for i in range(1000)]

@benchmark
def test_todo():
    return build_csv_todo(test_data)

@benchmark
def test_solution():
    return build_csv_solution(test_data)

@benchmark
def test_optimized():
    return build_csv_optimized(test_data)

test_todo()
test_solution()
test_optimized()


# ============================================================================
# EXERCISE 5: Algorithm Optimization - Find Duplicates (Hard)
# ============================================================================
print("\n" + "=" * 70)
print("EXERCISE 5: Algorithm Optimization - Find Duplicates")
print("=" * 70)
print("Görev: O(n²) algoritmayı O(n)'e düşürün")
print()

# TODO: O(n²) algoritmayı O(n)'e optimize edin
def find_duplicates_todo(numbers: List[int]) -> List[int]:
    """
    TODO: O(n²) nested loop'u O(n) set-based çözüme çevirin

    Şu anki complexity: O(n²)
    Hedef complexity: O(n)

    İpucu: Set kullanın
    """
    duplicates = []
    n = len(numbers)

    for i in range(n):
        for j in range(i + 1, n):
            if numbers[i] == numbers[j] and numbers[i] not in duplicates:
                duplicates.append(numbers[i])

    return duplicates


# SOLUTION
def find_duplicates_solution(numbers: List[int]) -> List[int]:
    """
    SOLUTION: O(n) set-based duplicate finding

    Set lookup O(1) olduğu için toplam complexity O(n)'dir.
    Nested loop yerine tek pass ile çözüm.
    """
    seen = set()
    duplicates = set()

    for num in numbers:
        if num in seen:
            duplicates.add(num)
        else:
            seen.add(num)

    return list(duplicates)

# Alternative: Counter kullanarak
def find_duplicates_counter(numbers: List[int]) -> List[int]:
    """Counter ile alternatif çözüm"""
    from collections import Counter
    counts = Counter(numbers)
    return [num for num, count in counts.items() if count > 1]


# Test & Benchmark
print("\n--- Benchmark ---")
test_data = list(range(5000)) * 2  # 10000 elements, all duplicates

# TODO çok yavaş, küçük veri ile test
# @benchmark
# def test_todo():
#     return find_duplicates_todo(test_data[:1000])
# test_todo()

@benchmark
def test_solution():
    return find_duplicates_solution(test_data)

@benchmark
def test_counter():
    return find_duplicates_counter(test_data)

test_solution()
test_counter()


# ============================================================================
# EXERCISE 6: Memory Optimization - Generator (Medium)
# ============================================================================
print("\n" + "=" * 70)
print("EXERCISE 6: Memory Optimization with Generators")
print("=" * 70)
print("Görev: List comprehension'ı generator'a çevirerek memory optimize edin")
print()

# TODO: List yerine generator kullanarak memory optimize edin
def process_large_file_todo(filename: str, chunk_size: int = 1000) -> int:
    """
    TODO: List comprehension yerine generator expression kullanın

    Büyük dosyalarda list comprehension tüm veriyi memory'de tutar.
    Generator kullanarak memory kullanımını düşürün.
    """
    # Simulated large file
    lines = [f"line {i}: " + "x" * 100 for i in range(100000)]

    # TODO: Bu list comprehension'ı generator'a çevirin
    processed = [line.upper() for line in lines if len(line) > 10]

    return len(processed)


# SOLUTION
def process_large_file_solution(filename: str, chunk_size: int = 1000) -> int:
    """
    SOLUTION: Generator ile memory-efficient processing

    Generator expression sadece ihtiyaç duyulduğunda değer üretir.
    Bellek kullanımı sabit kalır (O(1) space).
    """
    # Simulated large file
    lines = (f"line {i}: " + "x" * 100 for i in range(100000))

    # Generator expression - lazy evaluation
    processed = (line.upper() for line in lines if len(line) > 10)

    # Sum generator without storing all results
    return sum(1 for _ in processed)


# Test & Benchmark
print("\n--- Benchmark ---")

print("TODO version (list):")
tracemalloc.start()
start = time.perf_counter()
result1 = process_large_file_todo("dummy.txt")
current1, peak1 = tracemalloc.get_traced_memory()
time1 = time.perf_counter() - start
tracemalloc.stop()
print(f"Time: {time1:.4f}s, Peak Memory: {peak1 / 1024 / 1024:.2f} MB")

print("\nSOLUTION version (generator):")
tracemalloc.start()
start = time.perf_counter()
result2 = process_large_file_solution("dummy.txt")
current2, peak2 = tracemalloc.get_traced_memory()
time2 = time.perf_counter() - start
tracemalloc.stop()
print(f"Time: {time2:.4f}s, Peak Memory: {peak2 / 1024 / 1024:.2f} MB")

print(f"\nMemory savings: {((peak1 - peak2) / peak1 * 100):.1f}%")


# ============================================================================
# EXERCISE 7: Caching Strategy - API Calls (Hard)
# ============================================================================
print("\n" + "=" * 70)
print("EXERCISE 7: TTL Cache for API Calls")
print("=" * 70)
print("Görev: API çağrıları için TTL cache implementasyonu yazın")
print()

# TODO: TTL cache implementasyonu yapın
class APIClientTodo:
    """
    TODO: TTL (Time-To-Live) cache ekleyin

    API çağrıları pahalıdır. Cache ekleyerek:
    - Aynı request 5 saniye içinde tekrar yapılırsa cache'den dön
    - 5 saniye sonra cache expire olsun, yeni request yap
    """

    def __init__(self):
        self.call_count = 0

    def fetch_user(self, user_id: int) -> Dict[str, Any]:
        """
        TODO: TTL cache ekleyin (5 saniye)
        """
        # Simulated API call
        time.sleep(0.1)
        self.call_count += 1
        return {"id": user_id, "name": f"User {user_id}"}


# SOLUTION
class APIClientSolution:
    """
    SOLUTION: TTL cache ile optimize edilmiş API client

    Cache entry'ler expire time ile saklanır.
    Expired entry'ler yeniden fetch edilir.
    """

    def __init__(self, ttl_seconds: int = 5):
        self.ttl_seconds = ttl_seconds
        self.cache: Dict[int, tuple] = {}  # {user_id: (data, expire_time)}
        self.call_count = 0

    def fetch_user(self, user_id: int) -> Dict[str, Any]:
        """TTL cache ile user fetch"""
        current_time = time.time()

        # Check cache
        if user_id in self.cache:
            data, expire_time = self.cache[user_id]
            if current_time < expire_time:
                print(f"  Cache HIT for user {user_id}")
                return data
            else:
                print(f"  Cache EXPIRED for user {user_id}")
        else:
            print(f"  Cache MISS for user {user_id}")

        # Fetch from API
        time.sleep(0.1)  # Simulated API call
        self.call_count += 1
        data = {"id": user_id, "name": f"User {user_id}"}

        # Store in cache
        expire_time = current_time + self.ttl_seconds
        self.cache[user_id] = (data, expire_time)

        return data

    def clear_expired(self):
        """Expired cache entry'leri temizle"""
        current_time = time.time()
        expired_keys = [
            key for key, (_, expire_time) in self.cache.items()
            if current_time >= expire_time
        ]
        for key in expired_keys:
            del self.cache[key]


# Test & Benchmark
print("\n--- Benchmark ---")
client = APIClientSolution(ttl_seconds=2)

print("First call (cache miss):")
start = time.perf_counter()
data1 = client.fetch_user(1)
print(f"Time: {time.perf_counter() - start:.4f}s\n")

print("Second call immediately (cache hit):")
start = time.perf_counter()
data2 = client.fetch_user(1)
print(f"Time: {time.perf_counter() - start:.4f}s\n")

print("Waiting 3 seconds for expiration...")
time.sleep(3)

print("Third call after expiration (cache expired):")
start = time.perf_counter()
data3 = client.fetch_user(1)
print(f"Time: {time.perf_counter() - start:.4f}s\n")

print(f"Total API calls made: {client.call_count}")


# ============================================================================
# EXERCISE 8: Batch Processing Optimization (Hard)
# ============================================================================
print("\n" + "=" * 70)
print("EXERCISE 8: Batch Processing for Database Operations")
print("=" * 70)
print("Görev: Tek tek insert yerine batch insert kullanın")
print()

# TODO: Batch processing implementasyonu
def save_records_todo(records: List[Dict[str, Any]]) -> None:
    """
    TODO: Tek tek insert yerine batch insert yapın

    Her kayıt için ayrı DB call yapmak çok yavaştır.
    Batch'ler halinde (örn. 100'lük gruplar) insert yapın.
    """
    for record in records:
        # Simulated DB insert
        time.sleep(0.001)  # Her insert 1ms
        pass


# SOLUTION
def save_records_solution(records: List[Dict[str, Any]], batch_size: int = 100) -> None:
    """
    SOLUTION: Batch processing ile optimize edilmiş insert

    Kayıtları batch'ler halinde gruplar ve tek seferde insert eder.
    Network overhead'i büyük ölçüde azaltır.
    """
    def insert_batch(batch: List[Dict[str, Any]]):
        """Batch insert - tek network call"""
        time.sleep(0.01)  # Batch insert 10ms (100 kayıt için)

    # Process in batches
    for i in range(0, len(records), batch_size):
        batch = records[i:i + batch_size]
        insert_batch(batch)


# Alternative: Generator-based batching
def batch_iterator(items: List[Any], batch_size: int) -> Iterator[List[Any]]:
    """Iterator'ü batch'lere böler"""
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]

def save_records_optimized(records: List[Dict[str, Any]], batch_size: int = 100) -> None:
    """Generator ile batch processing"""
    def insert_batch(batch: List[Dict[str, Any]]):
        time.sleep(0.01)

    for batch in batch_iterator(records, batch_size):
        insert_batch(batch)


# Test & Benchmark
print("\n--- Benchmark ---")
test_records = [{"id": i, "data": f"record_{i}"} for i in range(1000)]

@benchmark
def test_todo():
    save_records_todo(test_records)

@benchmark
def test_solution():
    save_records_solution(test_records, batch_size=100)

@benchmark
def test_optimized():
    save_records_optimized(test_records, batch_size=100)

test_todo()
test_solution()
test_optimized()


# ============================================================================
# EXERCISE 9: Lazy Evaluation - Data Pipeline (Hard)
# ============================================================================
print("\n" + "=" * 70)
print("EXERCISE 9: Lazy Evaluation Pipeline")
print("=" * 70)
print("Görev: Eager evaluation'ı lazy evaluation'a çevirin")
print()

# TODO: Eager pipeline'ı lazy'ye çevirin
def data_pipeline_todo(data: List[int]) -> List[int]:
    """
    TODO: Eager evaluation (list) yerine lazy evaluation (generator) kullanın

    Her adım yeni bir list oluşturuyor. Memory kullanımı yüksek.
    Generator chain kullanarak lazy evaluation yapın.
    """
    # Step 1: Square
    squared = [x ** 2 for x in data]

    # Step 2: Filter evens
    evens = [x for x in squared if x % 2 == 0]

    # Step 3: Divide by 2
    halved = [x // 2 for x in evens]

    # Step 4: Take first 100
    result = halved[:100]

    return result


# SOLUTION
def data_pipeline_solution(data: List[int]) -> List[int]:
    """
    SOLUTION: Lazy evaluation ile memory-efficient pipeline

    Generator chain kullanarak sadece gerektiğinde değer üretir.
    Intermediate list'ler oluşturmaz.
    """
    # Lazy generators - sadece consume edildiğinde çalışır
    squared = (x ** 2 for x in data)
    evens = (x for x in squared if x % 2 == 0)
    halved = (x // 2 for x in evens)

    # Take first 100 - sadece 100 değer için işlem yapılır
    result = []
    for i, value in enumerate(halved):
        if i >= 100:
            break
        result.append(value)

    return result


# Class-based lazy pipeline
class LazyPipeline:
    """Fluent interface ile lazy pipeline"""

    def __init__(self, data: Iterator[Any]):
        self.data = data

    def map(self, func: Callable) -> 'LazyPipeline':
        """Map operation (lazy)"""
        return LazyPipeline(func(x) for x in self.data)

    def filter(self, predicate: Callable) -> 'LazyPipeline':
        """Filter operation (lazy)"""
        return LazyPipeline(x for x in self.data if predicate(x))

    def take(self, n: int) -> 'LazyPipeline':
        """Take first n elements (lazy)"""
        def taken():
            for i, x in enumerate(self.data):
                if i >= n:
                    break
                yield x
        return LazyPipeline(taken())

    def collect(self) -> List[Any]:
        """Collect results (eager)"""
        return list(self.data)

def data_pipeline_optimized(data: List[int]) -> List[int]:
    """Fluent interface ile lazy pipeline"""
    return (
        LazyPipeline(iter(data))
        .map(lambda x: x ** 2)
        .filter(lambda x: x % 2 == 0)
        .map(lambda x: x // 2)
        .take(100)
        .collect()
    )


# Test & Benchmark
print("\n--- Benchmark ---")
test_data = list(range(1_000_000))

print("TODO version (eager):")
tracemalloc.start()
start = time.perf_counter()
result1 = data_pipeline_todo(test_data)
current1, peak1 = tracemalloc.get_traced_memory()
time1 = time.perf_counter() - start
tracemalloc.stop()
print(f"Time: {time1:.4f}s, Peak Memory: {peak1 / 1024 / 1024:.2f} MB")

print("\nSOLUTION version (lazy):")
tracemalloc.start()
start = time.perf_counter()
result2 = data_pipeline_solution(test_data)
current2, peak2 = tracemalloc.get_traced_memory()
time2 = time.perf_counter() - start
tracemalloc.stop()
print(f"Time: {time2:.4f}s, Peak Memory: {peak2 / 1024 / 1024:.2f} MB")

print("\nOPTIMIZED version (fluent lazy):")
tracemalloc.start()
start = time.perf_counter()
result3 = data_pipeline_optimized(test_data)
current3, peak3 = tracemalloc.get_traced_memory()
time3 = time.perf_counter() - start
tracemalloc.stop()
print(f"Time: {time3:.4f}s, Peak Memory: {peak3 / 1024 / 1024:.2f} MB")


# ============================================================================
# EXERCISE 10: NumPy Vectorization (Hard)
# ============================================================================
print("\n" + "=" * 70)
print("EXERCISE 10: NumPy Vectorization")
print("=" * 70)
print("Görev: Pure Python'u NumPy vectorization'a çevirin")
print()

# TODO: NumPy vectorization kullanın
def calculate_distances_todo(points1: List[tuple], points2: List[tuple]) -> List[float]:
    """
    TODO: Pure Python yerine NumPy vectorization kullanın

    Euclidean distance hesaplama Python loop'la çok yavaş.
    NumPy array operations ile vectorize edin.
    """
    distances = []
    for (x1, y1), (x2, y2) in zip(points1, points2):
        dist = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
        distances.append(dist)
    return distances


# SOLUTION
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    print("NumPy not installed. Install with: pip install numpy")

if NUMPY_AVAILABLE:
    def calculate_distances_solution(points1: np.ndarray, points2: np.ndarray) -> np.ndarray:
        """
        SOLUTION: NumPy vectorization ile optimize edilmiş distance

        Vectorized operation C seviyesinde çalışır, çok daha hızlıdır.
        """
        return np.sqrt(np.sum((points2 - points1) ** 2, axis=1))

    # Test & Benchmark
    print("\n--- Benchmark ---")
    n = 100000
    points1_py = [(i, i * 2) for i in range(n)]
    points2_py = [(i + 1, i * 2 + 1) for i in range(n)]

    points1_np = np.array(points1_py)
    points2_np = np.array(points2_py)

    @benchmark
    def test_todo():
        return calculate_distances_todo(points1_py, points2_py)

    @benchmark
    def test_solution():
        return calculate_distances_solution(points1_np, points2_np)

    test_todo()
    test_solution()


# ============================================================================
# EXERCISE 11: Set vs List for Membership Testing (Medium)
# ============================================================================
print("\n" + "=" * 70)
print("EXERCISE 11: Set vs List Membership Testing")
print("=" * 70)
print("Görev: List membership testing'i set ile optimize edin")
print()

# TODO: List yerine set kullanın
def filter_duplicates_todo(source: List[int], exclude: List[int]) -> List[int]:
    """
    TODO: exclude list'ini set'e çevirerek optimize edin

    'in' operatörü list'te O(n), set'te O(1)
    Büyük exclude list'lerinde dramatik fark yaratır.
    """
    result = []
    for item in source:
        if item not in exclude:  # O(n) for list!
            result.append(item)
    return result


# SOLUTION
def filter_duplicates_solution(source: List[int], exclude: List[int]) -> List[int]:
    """
    SOLUTION: Set ile O(1) membership testing

    exclude'u set'e çevirerek lookup'ı hızlandırır.
    """
    exclude_set = set(exclude)
    result = []
    for item in source:
        if item not in exclude_set:  # O(1) for set!
            result.append(item)
    return result

# Daha optimize: list comprehension
def filter_duplicates_optimized(source: List[int], exclude: List[int]) -> List[int]:
    """En optimize: set + list comprehension"""
    exclude_set = set(exclude)
    return [item for item in source if item not in exclude_set]


# Test & Benchmark
print("\n--- Benchmark ---")
source_data = list(range(10000))
exclude_data = list(range(5000, 15000))  # 5000 eleman overlap

@benchmark
def test_todo():
    return filter_duplicates_todo(source_data, exclude_data)

@benchmark
def test_solution():
    return filter_duplicates_solution(source_data, exclude_data)

@benchmark
def test_optimized():
    return filter_duplicates_optimized(source_data, exclude_data)

test_todo()
test_solution()
test_optimized()


# ============================================================================
# EXERCISE 12: Profiling and Optimization (Expert)
# ============================================================================
print("\n" + "=" * 70)
print("EXERCISE 12: Complete Optimization Challenge")
print("=" * 70)
print("Görev: Tüm öğrendiklerinizi kullanarak bu fonksiyonu optimize edin")
print()

# TODO: Bu fonksiyonu optimize edin (birden fazla problem var!)
def process_user_data_todo(users: List[Dict], transactions: List[Dict]) -> Dict[str, Any]:
    """
    TODO: Bu fonksiyondaki tüm performance problemlerini bulun ve düzeltin

    Problemler:
    1. String concatenation
    2. List membership testing
    3. Nested loops
    4. Dictionary lookup
    5. Eager evaluation

    Bu fonksiyonu analiz edin ve optimize edin!
    """
    # Build user index (PROBLEM: O(n) lookup her seferinde)
    result = []

    for transaction in transactions:
        user = None
        for u in users:  # O(n) lookup!
            if u['id'] == transaction['user_id']:
                user = u
                break

        if user:
            # PROBLEM: String concatenation
            summary = ""
            summary += f"User: {user['name']}, "
            summary += f"Transaction: {transaction['amount']}, "
            summary += f"Status: {transaction['status']}"

            result.append(summary)

    # PROBLEM: List'te unique bulma
    unique_users = []
    for transaction in transactions:
        if transaction['user_id'] not in unique_users:  # O(n)!
            unique_users.append(transaction['user_id'])

    return {
        'summaries': result,
        'unique_users': len(unique_users),
        'total_transactions': len(transactions)
    }


# SOLUTION
def process_user_data_solution(users: List[Dict], transactions: List[Dict]) -> Dict[str, Any]:
    """
    SOLUTION: Tamamen optimize edilmiş versiyon

    Optimizasyonlar:
    1. Dictionary index (O(1) lookup)
    2. f-string kullanımı (string concat yerine)
    3. List comprehension
    4. Set kullanımı (unique için)
    """
    # 1. User index oluştur - O(1) lookup için
    user_index = {user['id']: user for user in users}

    # 2. Generator + f-string ile summaries
    summaries = [
        f"User: {user_index[t['user_id']]['name']}, "
        f"Transaction: {t['amount']}, "
        f"Status: {t['status']}"
        for t in transactions
        if t['user_id'] in user_index
    ]

    # 3. Set ile unique users (O(n) time)
    unique_users = len(set(t['user_id'] for t in transactions))

    return {
        'summaries': summaries,
        'unique_users': unique_users,
        'total_transactions': len(transactions)
    }


# Test & Benchmark
print("\n--- Benchmark ---")
test_users = [
    {'id': i, 'name': f'User {i}'}
    for i in range(1000)
]
test_transactions = [
    {'user_id': i % 1000, 'amount': i * 10, 'status': 'completed'}
    for i in range(10000)
]

@benchmark
def test_todo():
    return process_user_data_todo(test_users, test_transactions)

@benchmark
def test_solution():
    return process_user_data_solution(test_users, test_transactions)

result_todo = test_todo()
result_solution = test_solution()

print(f"\nResults match: {result_todo['unique_users'] == result_solution['unique_users']}")


# ============================================================================
# EXERCISE 13: Memory Profiling Challenge (Expert)
# ============================================================================
print("\n" + "=" * 70)
print("EXERCISE 13: Memory Leak Detection and Fix")
print("=" * 70)
print("Görev: Memory leak'i bulun ve düzeltin")
print()

# TODO: Bu class'ta memory leak var, bulun ve düzeltin!
class DataProcessorTodo:
    """
    TODO: Bu class'ta memory leak var

    Process çağrıldıkça memory birikiyor ve temizlenmiyor.
    Memory leak'i bulun ve düzeltin.
    """

    def __init__(self):
        self.cache = []
        self.results = []

    def process(self, data: List[int]) -> List[int]:
        """
        TODO: Memory leak'i düzeltin

        Her process çağrısında veri birikiyor!
        """
        # Process data
        processed = [x ** 2 for x in data]

        # PROBLEM: Cache sürekli büyüyor!
        self.cache.append(processed)

        # PROBLEM: Results da sürekli büyüyor!
        self.results.extend(processed)

        return processed


# SOLUTION
class DataProcessorSolution:
    """
    SOLUTION: Memory leak düzeltilmiş versiyon

    Çözümler:
    1. Cache için maksimum boyut sınırı
    2. LRU cache kullanımı
    3. Results yerine generator
    4. Clear metodu
    """

    def __init__(self, max_cache_size: int = 10):
        self.cache = deque(maxlen=max_cache_size)  # Auto-limit
        self.processed_count = 0

    def process(self, data: List[int]) -> List[int]:
        """Memory-safe processing"""
        # Process data
        processed = [x ** 2 for x in data]

        # Cache with size limit
        self.cache.append(len(processed))  # Sadece metadata sakla

        # Counter, full data saklamıyor
        self.processed_count += len(processed)

        return processed

    def clear_cache(self):
        """Manual cache temizleme"""
        self.cache.clear()
        self.processed_count = 0


# Test & Benchmark
print("\n--- Memory Benchmark ---")

print("TODO version (memory leak):")
tracemalloc.start()
processor_todo = DataProcessorTodo()
for i in range(100):
    data = list(range(10000))
    processor_todo.process(data)
current1, peak1 = tracemalloc.get_traced_memory()
tracemalloc.stop()
print(f"Peak Memory: {peak1 / 1024 / 1024:.2f} MB")
print(f"Cache size: {len(processor_todo.cache)} items")
print(f"Results size: {len(processor_todo.results)} items")

print("\nSOLUTION version (memory safe):")
tracemalloc.start()
processor_solution = DataProcessorSolution(max_cache_size=10)
for i in range(100):
    data = list(range(10000))
    processor_solution.process(data)
current2, peak2 = tracemalloc.get_traced_memory()
tracemalloc.stop()
print(f"Peak Memory: {peak2 / 1024 / 1024:.2f} MB")
print(f"Cache size: {len(processor_solution.cache)} items (limited)")
print(f"Processed count: {processor_solution.processed_count} items")

print(f"\nMemory saved: {((peak1 - peak2) / peak1 * 100):.1f}%")


# ============================================================================
# EXERCISE 14: Connection Pooling (Expert)
# ============================================================================
print("\n" + "=" * 70)
print("EXERCISE 14: Connection Pool Implementation")
print("=" * 70)
print("Görev: Connection pool implementasyonu yapın")
print()

# TODO: Connection pool implementasyonu
class DatabaseConnectionTodo:
    """
    TODO: Connection pool pattern implementasyonu

    Her defasında yeni connection oluşturmak yerine,
    connection pool'dan connection al/ver mantığı yapın.
    """

    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        # Simulate connection overhead
        time.sleep(0.1)

    def execute(self, query: str) -> Any:
        time.sleep(0.01)
        return "Result"

    def close(self):
        pass


def execute_queries_todo(queries: List[str]) -> List[Any]:
    """
    TODO: Her query için yeni connection açmak yerine pool kullanın
    """
    results = []
    for query in queries:
        # PROBLEM: Her seferinde yeni connection!
        conn = DatabaseConnectionTodo("postgresql://localhost")
        result = conn.execute(query)
        results.append(result)
        conn.close()
    return results


# SOLUTION
from queue import Queue
from contextlib import contextmanager

class ConnectionPool:
    """
    SOLUTION: Connection pool implementation

    Connection'ları pool'da tutar ve yeniden kullanır.
    """

    def __init__(self, connection_string: str, pool_size: int = 5):
        self.connection_string = connection_string
        self.pool = Queue(maxsize=pool_size)

        # Initialize pool
        for _ in range(pool_size):
            conn = DatabaseConnectionTodo(connection_string)
            self.pool.put(conn)

    @contextmanager
    def get_connection(self):
        """Context manager ile connection al/ver"""
        conn = self.pool.get()
        try:
            yield conn
        finally:
            self.pool.put(conn)

    def close_all(self):
        """Tüm connection'ları kapat"""
        while not self.pool.empty():
            conn = self.pool.get()
            conn.close()


def execute_queries_solution(queries: List[str]) -> List[Any]:
    """Connection pool ile query execution"""
    pool = ConnectionPool("postgresql://localhost", pool_size=5)
    results = []

    for query in queries:
        with pool.get_connection() as conn:
            result = conn.execute(query)
            results.append(result)

    pool.close_all()
    return results


# Test & Benchmark
print("\n--- Benchmark ---")
test_queries = [f"SELECT * FROM table WHERE id = {i}" for i in range(20)]

@benchmark
def test_todo():
    return execute_queries_todo(test_queries)

@benchmark
def test_solution():
    return execute_queries_solution(test_queries)

test_todo()
test_solution()


# ============================================================================
# EXERCISE 15: Complete Optimization (Expert)
# ============================================================================
print("\n" + "=" * 70)
print("EXERCISE 15: Real-World Optimization Challenge")
print("=" * 70)
print("Görev: Gerçek dünya senaryosunu tüm teknikleri kullanarak optimize edin")
print()

# TODO: Bu analitik pipeline'ı optimize edin
def analytics_pipeline_todo(users: List[Dict], events: List[Dict]) -> Dict[str, Any]:
    """
    TODO: Bu analytics pipeline'ında birçok optimization fırsatı var

    Pipeline:
    1. User ID'ye göre event'leri gruplama
    2. Her user için metrics hesaplama
    3. Top 10 user bulma
    4. Summary oluşturma

    Optimize edilecekler:
    - Algorithm complexity
    - Data structure seçimi
    - Memory usage
    - String operations
    """
    # Group events by user (PROBLEM: O(n²))
    user_events = {}
    for event in events:
        user_id = event['user_id']
        found = False
        for uid in user_events:
            if uid == user_id:
                user_events[uid].append(event)
                found = True
                break
        if not found:
            user_events[user_id] = [event]

    # Calculate metrics (PROBLEM: Inefficient)
    user_metrics = []
    for user_id, events_list in user_events.items():
        total_value = 0
        for event in events_list:
            total_value += event['value']

        # PROBLEM: String concatenation
        user_name = ""
        for user in users:
            if user['id'] == user_id:
                user_name = user['name']
                break

        user_metrics.append({
            'user_id': user_id,
            'user_name': user_name,
            'event_count': len(events_list),
            'total_value': total_value
        })

    # Sort and get top 10 (OK)
    user_metrics.sort(key=lambda x: x['total_value'], reverse=True)
    top_10 = user_metrics[:10]

    # Build summary (PROBLEM: String concatenation)
    summary = ""
    for metric in top_10:
        summary += f"User {metric['user_name']}: {metric['total_value']}\n"

    return {
        'top_10': top_10,
        'summary': summary,
        'total_users': len(user_metrics)
    }


# SOLUTION
def analytics_pipeline_solution(users: List[Dict], events: List[Dict]) -> Dict[str, Any]:
    """
    SOLUTION: Fully optimized analytics pipeline

    Optimizations:
    1. defaultdict for O(1) grouping
    2. User index for O(1) lookup
    3. Generator expressions
    4. join() for string building
    """
    # 1. User index - O(1) lookup
    user_index = {user['id']: user for user in users}

    # 2. Group events with defaultdict - O(n) instead of O(n²)
    user_events = defaultdict(list)
    for event in events:
        user_events[event['user_id']].append(event)

    # 3. Calculate metrics efficiently
    user_metrics = [
        {
            'user_id': user_id,
            'user_name': user_index.get(user_id, {}).get('name', 'Unknown'),
            'event_count': len(events_list),
            'total_value': sum(e['value'] for e in events_list)  # Generator
        }
        for user_id, events_list in user_events.items()
    ]

    # 4. Sort and get top 10
    user_metrics.sort(key=lambda x: x['total_value'], reverse=True)
    top_10 = user_metrics[:10]

    # 5. Build summary with join
    summary_lines = [
        f"User {metric['user_name']}: {metric['total_value']}"
        for metric in top_10
    ]
    summary = "\n".join(summary_lines)

    return {
        'top_10': top_10,
        'summary': summary,
        'total_users': len(user_metrics)
    }


# Test & Benchmark
print("\n--- Final Challenge Benchmark ---")
test_users = [{'id': i, 'name': f'User {i}'} for i in range(1000)]
test_events = [
    {'user_id': i % 1000, 'value': i % 100, 'type': 'click'}
    for i in range(50000)
]

print("TODO version:")
start = time.perf_counter()
result_todo = analytics_pipeline_todo(test_users, test_events)
time_todo = time.perf_counter() - start
print(f"Time: {time_todo:.4f}s")

print("\nSOLUTION version:")
start = time.perf_counter()
result_solution = analytics_pipeline_solution(test_users, test_events)
time_solution = time.perf_counter() - start
print(f"Time: {time_solution:.4f}s")

print(f"\nSpeedup: {time_todo / time_solution:.2f}x")
print(f"Top user (TODO): {result_todo['top_10'][0]['user_name']}")
print(f"Top user (SOLUTION): {result_solution['top_10'][0]['user_name']}")


print("\n" + "=" * 70)
print("TÜM EXERCISE'LAR TAMAMLANDI!")
print("=" * 70)
print("""
Performance Optimization Özeti:
1. Profiling yaparak bottleneck'leri bulun
2. Doğru algoritma ve veri yapısı seçin
3. Caching kullanın
4. Lazy evaluation tercih edin
5. NumPy ile vectorize edin
6. Built-in fonksiyonları kullanın
7. Generator'ları tercih edin (memory)
8. Set/Dict kullanın (O(1) lookup)
9. String işlemlerinde join() kullanın
10. Batch processing yapın

Unutmayın: "Premature optimization is the root of all evil"
Önce çalışan kod yazın, sonra profiling yapıp optimize edin!
""")
