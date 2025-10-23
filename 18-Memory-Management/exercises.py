"""
MEMORY MANAGEMENT EXERCISES
Her exercise için önce TODO kısmını kendiniz implement etmeye çalışın.
Sonra solution ile karşılaştırın.
"""

import gc
import sys
import weakref
import tracemalloc
from typing import Any, Dict, List, Optional
import time


# ============================================================================
# EXERCISE 1: Memory Leak Detector (HARD)
# ============================================================================
"""
TODO: Bir sınıf yazın ki:
1. Belirli tipteki nesnelerin oluşturulmasını izlesin
2. GC collect sonrası kaç tanesinin silindiğini raporlasın
3. Silenmemiş nesneleri (leak) tespit etsin
4. Referans zincirini göstersin

İpucu: gc.get_objects(), gc.get_referrers() kullanın
"""

class MemoryLeakDetector:
    def __init__(self, target_type):
        # TODO: Initialize detector
        pass

    def start_tracking(self):
        # TODO: Başlangıç snapshot'ı al
        pass

    def stop_tracking(self):
        # TODO: Bitiş snapshot'ı al ve analiz et
        pass

    def detect_leaks(self):
        # TODO: Leak'leri tespit et
        pass

    def show_referrers(self, obj, max_depth=2):
        # TODO: Referans zincirini göster
        pass


# SOLUTION:
class MemoryLeakDetectorSolution:
    """Bellek sızıntılarını tespit eden gelişmiş araç"""

    def __init__(self, target_type):
        self.target_type = target_type
        self.initial_objects = set()
        self.final_objects = set()
        self.leaked_objects = []

    def start_tracking(self):
        """Tracking'i başlat"""
        gc.collect()  # Önce temizlik yap
        self.initial_objects = set(
            id(obj) for obj in gc.get_objects()
            if isinstance(obj, self.target_type)
        )
        print(f"Tracking başladı: {len(self.initial_objects)} {self.target_type.__name__} nesnesi")

    def stop_tracking(self):
        """Tracking'i durdur ve analiz et"""
        gc.collect()  # GC çalıştır
        self.final_objects = set(
            id(obj) for obj in gc.get_objects()
            if isinstance(obj, self.target_type)
        )
        print(f"Tracking durdu: {len(self.final_objects)} {self.target_type.__name__} nesnesi")
        return self.detect_leaks()

    def detect_leaks(self):
        """Leak'leri tespit et"""
        # Silinmeyen yeni nesneler
        new_objects = self.final_objects - self.initial_objects

        if new_objects:
            print(f"\n⚠️  Leak tespit edildi: {len(new_objects)} nesne silinmedi!")

            # Nesneleri al
            all_objects = {id(obj): obj for obj in gc.get_objects()
                          if isinstance(obj, self.target_type)}

            self.leaked_objects = [all_objects[obj_id] for obj_id in new_objects
                                  if obj_id in all_objects]

            # Detaylı rapor
            for i, obj in enumerate(self.leaked_objects[:5], 1):  # İlk 5
                print(f"\n{i}. Leak: {obj}")
                print(f"   RefCount: {sys.getrefcount(obj) - 1}")
                self.show_referrers(obj, max_depth=2)

            return True
        else:
            print("\n✓ Leak yok, tüm nesneler temizlendi")
            return False

    def show_referrers(self, obj, max_depth=2, _depth=0, _seen=None):
        """Referans zincirini göster"""
        if _seen is None:
            _seen = set()

        if id(obj) in _seen or _depth > max_depth:
            return

        _seen.add(id(obj))
        indent = "   " * (_depth + 1)

        referrers = gc.get_referrers(obj)
        # Frame ve module'leri filtrele
        referrers = [r for r in referrers
                    if not isinstance(r, type(sys._getframe()))]

        if referrers:
            print(f"{indent}└─ Referanslar ({len(referrers)}):")
            for ref in referrers[:3]:  # İlk 3
                ref_type = type(ref).__name__
                if isinstance(ref, dict):
                    print(f"{indent}   - dict (keys: {list(ref.keys())[:3]})")
                elif isinstance(ref, list):
                    print(f"{indent}   - list (len: {len(ref)})")
                else:
                    print(f"{indent}   - {ref_type}")


def test_memory_leak_detector():
    """Memory leak detector test"""
    print("=" * 60)
    print("TEST: Memory Leak Detector")
    print("=" * 60)

    class TestObject:
        def __init__(self, name):
            self.name = name

        def __repr__(self):
            return f"TestObject({self.name})"

    detector = MemoryLeakDetectorSolution(TestObject)

    # Test 1: Leak var
    print("\n--- Test 1: Leak ile ---")
    leak_container = []  # Global container

    detector.start_tracking()
    for i in range(5):
        obj = TestObject(f"leak-{i}")
        leak_container.append(obj)  # Leak!
    detector.stop_tracking()

    # Test 2: Leak yok
    print("\n\n--- Test 2: Leak yok ---")
    leak_container.clear()
    gc.collect()

    detector = MemoryLeakDetectorSolution(TestObject)
    detector.start_tracking()
    for i in range(5):
        obj = TestObject(f"temp-{i}")
        # Fonksiyon bitince silinecek
    detector.stop_tracking()


# ============================================================================
# EXERCISE 2: Smart Cache with Memory Limit (MEDIUM-HARD)
# ============================================================================
"""
TODO: Bellek limiti olan bir cache implementasyonu:
1. LRU eviction stratejisi
2. Maksimum bellek limiti (MB cinsinden)
3. Bellek kullanımını izleme
4. Weak reference desteği (optional items için)
5. Cache hit/miss istatistikleri

İpucu: sys.getsizeof(), weakref.WeakValueDictionary
"""

class SmartCache:
    def __init__(self, max_memory_mb: float):
        # TODO: Initialize cache
        pass

    def get(self, key):
        # TODO: Cache'ten al, hit/miss say
        pass

    def set(self, key, value, weak=False):
        # TODO: Cache'e ekle, limit kontrolü yap
        pass

    def _evict_if_needed(self):
        # TODO: Gerekirse LRU item'ları çıkar
        pass

    def get_stats(self):
        # TODO: İstatistikleri döndür
        pass


# SOLUTION:
class SmartCacheSolution:
    """Bellek limiti olan akıllı cache"""

    def __init__(self, max_memory_mb: float):
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self._cache: Dict[Any, Any] = {}
        self._weak_cache: weakref.WeakValueDictionary = weakref.WeakValueDictionary()
        self._access_order: List[Any] = []  # LRU tracking
        self._hits = 0
        self._misses = 0

    def get(self, key):
        """Cache'ten değer al"""
        # Önce normal cache'e bak
        if key in self._cache:
            self._hits += 1
            # LRU güncelle
            self._access_order.remove(key)
            self._access_order.append(key)
            return self._cache[key]

        # Weak cache'e bak
        if key in self._weak_cache:
            self._hits += 1
            return self._weak_cache[key]

        self._misses += 1
        return None

    def set(self, key, value, weak=False):
        """Cache'e değer ekle"""
        if weak:
            # Weak cache'e ekle
            try:
                self._weak_cache[key] = value
            except TypeError:
                print(f"⚠️  {type(value)} weak reference desteklemiyor")
                weak = False

        if not weak:
            # Normal cache'e ekle
            # Önce mevcut değeri sil (varsa)
            if key in self._cache:
                self._access_order.remove(key)

            self._cache[key] = value
            self._access_order.append(key)

            # Bellek limiti kontrolü
            self._evict_if_needed()

    def _evict_if_needed(self):
        """Bellek limiti aşılırsa LRU item'ları çıkar"""
        current_size = self._get_current_size()

        while current_size > self.max_memory_bytes and self._access_order:
            # En eski item'ı çıkar
            oldest_key = self._access_order.pop(0)
            if oldest_key in self._cache:
                del self._cache[oldest_key]

            current_size = self._get_current_size()

    def _get_current_size(self):
        """Mevcut cache boyutu"""
        total = 0
        for key, value in self._cache.items():
            total += sys.getsizeof(key) + sys.getsizeof(value)
        return total

    def get_stats(self):
        """Cache istatistikleri"""
        total_requests = self._hits + self._misses
        hit_rate = self._hits / total_requests if total_requests > 0 else 0

        return {
            'size': len(self._cache),
            'weak_size': len(self._weak_cache),
            'memory_mb': self._get_current_size() / 1024 / 1024,
            'max_memory_mb': self.max_memory_bytes / 1024 / 1024,
            'hits': self._hits,
            'misses': self._misses,
            'hit_rate': hit_rate,
        }

    def clear(self):
        """Cache'i temizle"""
        self._cache.clear()
        self._weak_cache = weakref.WeakValueDictionary()
        self._access_order.clear()


def test_smart_cache():
    """Smart cache test"""
    print("\n" + "=" * 60)
    print("TEST: Smart Cache")
    print("=" * 60)

    cache = SmartCacheSolution(max_memory_mb=1.0)  # 1MB limit

    # Test 1: Normal cache
    print("\n--- Test 1: Normal cache operations ---")
    for i in range(10):
        cache.set(f"key{i}", [0] * 10000)  # Her biri ~80KB

    stats = cache.get_stats()
    print(f"Cache size: {stats['size']} items")
    print(f"Memory: {stats['memory_mb']:.2f} MB")

    # Test 2: LRU eviction
    print("\n--- Test 2: LRU eviction ---")
    result = cache.get("key0")
    print(f"key0 (evicted): {result}")
    result = cache.get("key9")
    print(f"key9 (newest): {result is not None}")

    # Test 3: Weak reference
    print("\n--- Test 3: Weak reference ---")

    class CacheableObject:
        def __init__(self, data):
            self.data = data

    obj = CacheableObject("test data")
    cache.set("weak_key", obj, weak=True)
    print(f"Weak cache size: {cache.get_stats()['weak_size']}")

    result = cache.get("weak_key")
    print(f"Got from weak cache: {result is not None}")

    del obj
    gc.collect()
    result = cache.get("weak_key")
    print(f"After del: {result}")

    # Test 4: Statistics
    print("\n--- Test 4: Statistics ---")
    for i in range(20):
        cache.get(f"key{i}")

    stats = cache.get_stats()
    print(f"Hits: {stats['hits']}")
    print(f"Misses: {stats['misses']}")
    print(f"Hit rate: {stats['hit_rate']:.2%}")


# ============================================================================
# EXERCISE 3: Memory Profiler Decorator (MEDIUM)
# ============================================================================
"""
TODO: Fonksiyonun bellek kullanımını ölçen decorator:
1. Fonksiyon çağrısı öncesi/sonrası bellek kullanımı
2. Peak memory kullanımı
3. Memory allocation detayları
4. Execution time

İpucu: tracemalloc modülünü kullanın
"""

def memory_profile(func):
    # TODO: Decorator implementation
    pass


# SOLUTION:
def memory_profile_solution(func):
    """Fonksiyonun bellek kullanımını profillayan decorator"""

    def wrapper(*args, **kwargs):
        # tracemalloc başlat
        tracemalloc.start()

        # Başlangıç snapshot
        snapshot_start = tracemalloc.take_snapshot()
        current_start, peak_start = tracemalloc.get_traced_memory()
        time_start = time.perf_counter()

        # Fonksiyonu çalıştır
        result = func(*args, **kwargs)

        # Bitiş ölçümleri
        time_end = time.perf_counter()
        current_end, peak_end = tracemalloc.get_traced_memory()
        snapshot_end = tracemalloc.take_snapshot()

        # İstatistikleri hesapla
        exec_time = time_end - time_start
        memory_diff = current_end - current_start
        peak_memory = peak_end

        # Rapor
        print(f"\n{'=' * 60}")
        print(f"Memory Profile: {func.__name__}")
        print(f"{'=' * 60}")
        print(f"Execution time: {exec_time:.4f}s")
        print(f"Memory used: {memory_diff / 1024 / 1024:.2f} MB")
        print(f"Peak memory: {peak_memory / 1024 / 1024:.2f} MB")

        # Top allocations
        top_stats = snapshot_end.compare_to(snapshot_start, 'lineno')
        print(f"\nTop 5 memory allocations:")
        for i, stat in enumerate(top_stats[:5], 1):
            print(f"{i}. {stat}")

        tracemalloc.stop()

        return result

    return wrapper


def test_memory_profiler():
    """Memory profiler test"""
    print("\n" + "=" * 60)
    print("TEST: Memory Profiler Decorator")
    print("=" * 60)

    @memory_profile_solution
    def memory_intensive_function():
        """Bellek yoğun fonksiyon"""
        # Büyük liste
        big_list = [i for i in range(1_000_000)]

        # Büyük dict
        big_dict = {i: str(i) * 10 for i in range(100_000)}

        # Nested yapı
        nested = [[j for j in range(100)] for i in range(10_000)]

        return len(big_list) + len(big_dict) + len(nested)

    result = memory_intensive_function()
    print(f"\nResult: {result}")


# ============================================================================
# EXERCISE 4: Object Pool with __slots__ (MEDIUM)
# ============================================================================
"""
TODO: __slots__ kullanan efficient object pool:
1. Sabit sayıda pre-allocated object
2. acquire() ve release() metodları
3. Pool dolu olduğunda bekleme veya hata
4. Kullanım istatistikleri

İpucu: __slots__ ile bellek optimize edin
"""

class ObjectPool:
    class PooledObject:
        # TODO: __slots__ kullan
        pass

    def __init__(self, size: int):
        # TODO: Pool'u initialize et
        pass

    def acquire(self):
        # TODO: Pool'dan object al
        pass

    def release(self, obj):
        # TODO: Object'i pool'a geri ver
        pass


# SOLUTION:
class ObjectPoolSolution:
    """__slots__ kullanan efficient object pool"""

    class PooledObject:
        """Pool'daki nesneler için optimize edilmiş sınıf"""
        __slots__ = ['id', 'data', 'in_use', '_created_at']

        def __init__(self, pool_id):
            self.id = pool_id
            self.data = None
            self.in_use = False
            self._created_at = time.time()

        def reset(self):
            """Nesneyi temizle"""
            self.data = None
            self.in_use = False

        def __repr__(self):
            return f"PooledObject(id={self.id}, in_use={self.in_use})"

    def __init__(self, size: int):
        self.size = size
        # Pre-allocate objects
        self._pool = [self.PooledObject(i) for i in range(size)]
        self._available = list(self._pool)
        self._stats = {
            'acquires': 0,
            'releases': 0,
            'waits': 0,
        }

    def acquire(self, timeout: float = None) -> PooledObject:
        """Pool'dan nesne al"""
        start_time = time.time()

        while True:
            if self._available:
                obj = self._available.pop()
                obj.in_use = True
                self._stats['acquires'] += 1
                return obj

            # Pool boş
            if timeout is None:
                raise RuntimeError("Object pool boş")

            if time.time() - start_time > timeout:
                raise TimeoutError(f"Pool'dan nesne alınamadı ({timeout}s)")

            self._stats['waits'] += 1
            time.sleep(0.01)  # Kısa bekle

    def release(self, obj: PooledObject):
        """Nesneyi pool'a geri ver"""
        if not obj.in_use:
            raise ValueError("Nesne zaten pool'da")

        if obj not in self._pool:
            raise ValueError("Nesne bu pool'a ait değil")

        obj.reset()
        self._available.append(obj)
        self._stats['releases'] += 1

    def get_stats(self):
        """Pool istatistikleri"""
        return {
            'size': self.size,
            'available': len(self._available),
            'in_use': self.size - len(self._available),
            **self._stats,
        }

    def __enter__(self):
        """Context manager support"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Cleanup"""
        # Tüm nesneleri temizle
        for obj in self._pool:
            obj.reset()
        self._available = list(self._pool)


def test_object_pool():
    """Object pool test"""
    print("\n" + "=" * 60)
    print("TEST: Object Pool with __slots__")
    print("=" * 60)

    # Bellek karşılaştırması
    print("\n--- Memory comparison ---")
    print(f"PooledObject size: {sys.getsizeof(ObjectPoolSolution.PooledObject(0))} bytes")

    # Pool oluştur
    with ObjectPoolSolution(size=5) as pool:
        print(f"\n--- Initial stats ---")
        print(pool.get_stats())

        # Test 1: Normal kullanım
        print("\n--- Test 1: Acquire/Release ---")
        obj1 = pool.acquire()
        obj2 = pool.acquire()
        obj1.data = "test data"

        print(f"Acquired: {obj1}")
        print(f"Stats: {pool.get_stats()}")

        pool.release(obj1)
        pool.release(obj2)
        print(f"After release: {pool.get_stats()}")

        # Test 2: Pool dolu
        print("\n--- Test 2: Pool exhaustion ---")
        objects = []
        for i in range(5):
            obj = pool.acquire()
            objects.append(obj)

        print(f"All acquired: {pool.get_stats()}")

        try:
            pool.acquire()
        except RuntimeError as e:
            print(f"Expected error: {e}")

        # Temizle
        for obj in objects:
            pool.release(obj)

    print("\n--- Final stats ---")
    print(pool.get_stats())


# ============================================================================
# EXERCISE 5: Circular Reference Detector (HARD)
# ============================================================================
"""
TODO: Döngüsel referansları tespit eden araç:
1. Nesne grafını oluştur
2. Döngüleri tespit et (cycle detection)
3. Döngüdeki nesneleri ve referans zincirini göster
4. Döngü kırma önerileri

İpucu: gc.get_referents(), gc.get_referrers(), graph traversal
"""

class CircularReferenceDetector:
    def __init__(self, root_obj):
        # TODO: Initialize detector
        pass

    def detect_cycles(self):
        # TODO: Döngüleri tespit et
        pass

    def visualize_graph(self):
        # TODO: Nesne grafını görselleştir
        pass


# SOLUTION:
class CircularReferenceDetectorSolution:
    """Döngüsel referansları tespit eden araç"""

    def __init__(self, root_obj):
        self.root_obj = root_obj
        self.graph = {}  # obj_id -> [referent_ids]
        self.cycles = []

    def build_graph(self, obj, max_depth=5, _depth=0, _seen=None):
        """Nesne grafını oluştur"""
        if _seen is None:
            _seen = set()

        obj_id = id(obj)

        if obj_id in _seen or _depth > max_depth:
            return

        _seen.add(obj_id)

        # Referents (bu nesnenin referans verdiği nesneler)
        referents = gc.get_referents(obj)
        referents = [r for r in referents
                    if not isinstance(r, (type, type(sys._getframe())))]

        self.graph[obj_id] = [id(r) for r in referents]

        # Recursive olarak devam et
        for ref in referents:
            self.build_graph(ref, max_depth, _depth + 1, _seen)

    def detect_cycles(self):
        """DFS ile döngüleri tespit et"""
        self.cycles = []
        visited = set()
        rec_stack = []

        def dfs(node_id, path):
            if node_id in rec_stack:
                # Döngü bulundu!
                cycle_start = rec_stack.index(node_id)
                cycle = rec_stack[cycle_start:] + [node_id]
                self.cycles.append(cycle)
                return True

            if node_id in visited:
                return False

            visited.add(node_id)
            rec_stack.append(node_id)

            # Komşuları ziyaret et
            for neighbor_id in self.graph.get(node_id, []):
                if neighbor_id in self.graph:  # Sadece track edilen nesneler
                    dfs(neighbor_id, path + [node_id])

            rec_stack.pop()
            return False

        # Her node'dan başlat
        for node_id in self.graph:
            if node_id not in visited:
                dfs(node_id, [])

        return self.cycles

    def analyze(self):
        """Tam analiz yap"""
        print(f"\n{'=' * 60}")
        print("Circular Reference Analysis")
        print(f"{'=' * 60}")

        # Graf oluştur
        self.build_graph(self.root_obj)
        print(f"Graph nodes: {len(self.graph)}")

        # Döngüleri tespit et
        cycles = self.detect_cycles()

        if cycles:
            print(f"\n⚠️  {len(cycles)} döngü tespit edildi!\n")

            # Her döngüyü göster
            all_objects = {id(obj): obj for obj in gc.get_objects()}

            for i, cycle in enumerate(cycles[:5], 1):  # İlk 5
                print(f"Döngü #{i}:")
                print(f"  Uzunluk: {len(cycle) - 1}")

                # Döngüdeki nesneleri göster
                for j, obj_id in enumerate(cycle[:-1]):
                    if obj_id in all_objects:
                        obj = all_objects[obj_id]
                        obj_type = type(obj).__name__
                        print(f"  [{j}] {obj_type} (id: {obj_id})")

                        # Referans göster
                        next_id = cycle[j + 1]
                        print(f"      └─> references [{j+1}]")

                print()

            # Öneriler
            print("Çözüm önerileri:")
            print("1. weakref.ref() veya weakref.proxy() kullanın")
            print("2. Referansları manuel olarak temizleyin (__del__ veya cleanup method)")
            print("3. Context manager kullanarak otomatik cleanup yapın")

        else:
            print("\n✓ Döngüsel referans bulunamadı")

        return len(cycles) > 0


def test_circular_reference_detector():
    """Circular reference detector test"""
    print("\n" + "=" * 60)
    print("TEST: Circular Reference Detector")
    print("=" * 60)

    # Test 1: Basit döngü
    print("\n--- Test 1: Simple cycle ---")

    class Node:
        def __init__(self, value):
            self.value = value
            self.next = None

    node1 = Node(1)
    node2 = Node(2)
    node1.next = node2
    node2.next = node1  # Cycle!

    detector = CircularReferenceDetectorSolution(node1)
    detector.analyze()

    # Test 2: Karmaşık döngü
    print("\n--- Test 2: Complex cycle ---")

    class Parent:
        def __init__(self, name):
            self.name = name
            self.children = []

    class Child:
        def __init__(self, name):
            self.name = name
            self.parent = None

    parent = Parent("Parent")
    child1 = Child("Child1")
    child2 = Child("Child2")

    parent.children = [child1, child2]
    child1.parent = parent  # Back reference
    child2.parent = parent

    detector = CircularReferenceDetectorSolution(parent)
    detector.analyze()


# ============================================================================
# EXERCISE 6: Memory-Efficient Data Structure (MEDIUM-HARD)
# ============================================================================
"""
TODO: Büyük veri setleri için memory-efficient list:
1. Dahili olarak array.array kullan
2. Dinamik tip desteği (int, float, etc.)
3. Compression desteği (tekrar eden değerler)
4. Iterator protocol
5. Normal list ile bellek karşılaştırması

İpucu: array.array, struct modülü
"""

class EfficientList:
    def __init__(self, typecode='i'):
        # TODO: Implementation
        pass

    def append(self, value):
        # TODO: Append implementation
        pass

    def __getitem__(self, index):
        # TODO: Get item
        pass

    def __len__(self):
        # TODO: Length
        pass


# SOLUTION:
import array
from collections import defaultdict


class EfficientListSolution:
    """Memory-efficient list implementation"""

    # Type code mapping
    TYPECODES = {
        'int8': 'b',
        'int16': 'h',
        'int32': 'i',
        'int64': 'l',
        'float': 'f',
        'double': 'd',
    }

    def __init__(self, typecode='i', compress=False):
        """
        Args:
            typecode: array type code veya type name
            compress: Tekrar eden değerleri sıkıştır
        """
        if typecode in self.TYPECODES:
            typecode = self.TYPECODES[typecode]

        self.typecode = typecode
        self.compress = compress

        if compress:
            # Run-length encoding: (value, count)
            self._data = []
            self._length = 0
        else:
            self._data = array.array(typecode)

    def append(self, value):
        """Değer ekle"""
        if self.compress:
            # RLE compression
            if self._data and self._data[-1][0] == value:
                # Son değerle aynı, count artır
                self._data[-1] = (value, self._data[-1][1] + 1)
            else:
                # Yeni değer
                self._data.append((value, 1))
            self._length += 1
        else:
            self._data.append(value)

    def extend(self, values):
        """Çoklu değer ekle"""
        for value in values:
            self.append(value)

    def __getitem__(self, index):
        """Index ile erişim"""
        if self.compress:
            if index < 0 or index >= self._length:
                raise IndexError("Index out of range")

            # RLE'den değeri bul
            current_pos = 0
            for value, count in self._data:
                if current_pos + count > index:
                    return value
                current_pos += count

            raise IndexError("Index not found")
        else:
            return self._data[index]

    def __len__(self):
        """Uzunluk"""
        if self.compress:
            return self._length
        else:
            return len(self._data)

    def __iter__(self):
        """Iterator"""
        if self.compress:
            for value, count in self._data:
                for _ in range(count):
                    yield value
        else:
            yield from self._data

    def memory_usage(self):
        """Bellek kullanımı (bytes)"""
        if self.compress:
            return sys.getsizeof(self._data)
        else:
            return sys.getsizeof(self._data)

    def compression_ratio(self):
        """Sıkıştırma oranı"""
        if not self.compress:
            return 1.0

        compressed_items = len(self._data)
        actual_items = self._length
        return actual_items / compressed_items if compressed_items > 0 else 1.0


def test_efficient_list():
    """Efficient list test"""
    print("\n" + "=" * 60)
    print("TEST: Memory-Efficient List")
    print("=" * 60)

    # Test 1: Normal vs Efficient
    print("\n--- Test 1: Memory comparison ---")
    n = 100_000

    normal_list = list(range(n))
    efficient_list = EfficientListSolution('int32')
    efficient_list.extend(range(n))

    print(f"Normal list: {sys.getsizeof(normal_list) / 1024:.2f} KB")
    print(f"Efficient list: {efficient_list.memory_usage() / 1024:.2f} KB")
    print(f"Savings: {(sys.getsizeof(normal_list) - efficient_list.memory_usage()) / 1024:.2f} KB")

    # Test 2: Compression
    print("\n--- Test 2: RLE Compression ---")

    # Tekrar eden değerler
    repeated_data = [1] * 10000 + [2] * 10000 + [3] * 10000

    normal_list2 = repeated_data
    compressed_list = EfficientListSolution('int32', compress=True)
    compressed_list.extend(repeated_data)

    print(f"Normal list: {sys.getsizeof(normal_list2) / 1024:.2f} KB")
    print(f"Compressed list: {compressed_list.memory_usage() / 1024:.2f} KB")
    print(f"Compression ratio: {compressed_list.compression_ratio():.1f}x")
    print(f"Savings: {(sys.getsizeof(normal_list2) - compressed_list.memory_usage()) / 1024:.2f} KB")

    # Test 3: Access speed
    print("\n--- Test 3: Access performance ---")

    start = time.perf_counter()
    for i in range(1000):
        _ = normal_list[i]
    normal_time = time.perf_counter() - start

    start = time.perf_counter()
    for i in range(1000):
        _ = efficient_list[i]
    efficient_time = time.perf_counter() - start

    print(f"Normal list access: {normal_time:.6f}s")
    print(f"Efficient list access: {efficient_time:.6f}s")


# ============================================================================
# EXERCISE 7: Weak Reference Cache (MEDIUM)
# ============================================================================
"""
TODO: Weak reference kullanan self-cleaning cache:
1. WeakValueDictionary tabanlı
2. Otomatik cleanup callback'leri
3. Cache statistics (evictions, hits, misses)
4. Size limit ve TTL (time-to-live)

İpucu: weakref.WeakValueDictionary, weakref.finalize
"""

class WeakCache:
    def __init__(self, max_size=None, ttl=None):
        # TODO: Implementation
        pass

    def set(self, key, value):
        # TODO: Set with weak reference
        pass

    def get(self, key):
        # TODO: Get from cache
        pass


# SOLUTION:
import time
from typing import Optional


class WeakCacheSolution:
    """Weak reference ile self-cleaning cache"""

    def __init__(self, max_size: Optional[int] = None, ttl: Optional[float] = None):
        """
        Args:
            max_size: Maksimum cache boyutu
            ttl: Time-to-live (seconds)
        """
        self.max_size = max_size
        self.ttl = ttl
        self._cache = weakref.WeakValueDictionary()
        self._strong_refs = {}  # TTL için strong reference'lar
        self._access_times = {}  # TTL tracking
        self._stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'expirations': 0,
        }

    def set(self, key, value):
        """Cache'e ekle"""
        # Size limit kontrolü
        if self.max_size and len(self._cache) >= self.max_size:
            self._evict_oldest()

        try:
            # Weak reference ile cache'e ekle
            self._cache[key] = value

            # TTL varsa strong reference tut
            if self.ttl:
                self._strong_refs[key] = value
                self._access_times[key] = time.time()

                # Finalize ile cleanup
                def cleanup_callback(k=key):
                    self._stats['expirations'] += 1
                    self._strong_refs.pop(k, None)
                    self._access_times.pop(k, None)

                weakref.finalize(value, cleanup_callback)

        except TypeError:
            # Weak reference desteklemiyor
            raise TypeError(f"{type(value).__name__} weak reference desteklemiyor")

    def get(self, key):
        """Cache'ten al"""
        # TTL kontrolü
        if self.ttl and key in self._access_times:
            elapsed = time.time() - self._access_times[key]
            if elapsed > self.ttl:
                # Expired
                self._expire(key)
                self._stats['misses'] += 1
                return None

        # Cache'ten al
        value = self._cache.get(key)

        if value is not None:
            self._stats['hits'] += 1
            # TTL güncelle
            if self.ttl and key in self._access_times:
                self._access_times[key] = time.time()
        else:
            self._stats['misses'] += 1

        return value

    def _evict_oldest(self):
        """En eski girdiyi çıkar"""
        if not self._access_times:
            # Random bir key çıkar
            if self._cache:
                key = next(iter(self._cache))
                del self._cache[key]
                self._stats['evictions'] += 1
            return

        # En eski access time'a sahip key'i bul
        oldest_key = min(self._access_times.items(), key=lambda x: x[1])[0]
        self._expire(oldest_key)
        self._stats['evictions'] += 1

    def _expire(self, key):
        """Key'i expire et"""
        self._cache.pop(key, None)
        self._strong_refs.pop(key, None)
        self._access_times.pop(key, None)

    def get_stats(self):
        """İstatistikler"""
        total = self._stats['hits'] + self._stats['misses']
        hit_rate = self._stats['hits'] / total if total > 0 else 0

        return {
            'size': len(self._cache),
            'hit_rate': hit_rate,
            **self._stats,
        }

    def clear(self):
        """Cache'i temizle"""
        self._cache.clear()
        self._strong_refs.clear()
        self._access_times.clear()


def test_weak_cache():
    """Weak cache test"""
    print("\n" + "=" * 60)
    print("TEST: Weak Reference Cache")
    print("=" * 60)

    class CacheableData:
        def __init__(self, value):
            self.value = value

        def __repr__(self):
            return f"CacheableData({self.value})"

    # Test 1: Basic weak cache
    print("\n--- Test 1: Basic operations ---")
    cache = WeakCacheSolution()

    obj1 = CacheableData("data1")
    obj2 = CacheableData("data2")

    cache.set("key1", obj1)
    cache.set("key2", obj2)

    print(f"Cache size: {cache.get_stats()['size']}")
    print(f"Get key1: {cache.get('key1')}")

    # Obj1'i sil
    del obj1
    gc.collect()

    print(f"After del obj1: {cache.get('key1')}")
    print(f"Cache size: {cache.get_stats()['size']}")

    # Test 2: TTL
    print("\n--- Test 2: TTL (Time-To-Live) ---")
    cache2 = WeakCacheSolution(ttl=2.0)  # 2 saniye TTL

    obj3 = CacheableData("ttl_data")
    cache2.set("key3", obj3)

    print(f"Immediately: {cache2.get('key3')}")

    time.sleep(1)
    print(f"After 1s: {cache2.get('key3')}")

    time.sleep(1.5)
    print(f"After 2.5s: {cache2.get('key3')}")

    # Test 3: Size limit
    print("\n--- Test 3: Size limit ---")
    cache3 = WeakCacheSolution(max_size=3)

    objects = [CacheableData(f"data{i}") for i in range(5)]
    for i, obj in enumerate(objects):
        cache3.set(f"key{i}", obj)

    print(f"Cache size (max=3): {cache3.get_stats()['size']}")
    print(f"Stats: {cache3.get_stats()}")


# ============================================================================
# EXERCISE 8: Memory Monitor (MEDIUM)
# ============================================================================
"""
TODO: Real-time bellek monitörü:
1. Periyodik bellek kullanımı ölçümü
2. Memory spike detection
3. Top memory consumers
4. Alert sistemi (threshold aşımı)
5. Logging ve reporting

İpucu: threading, tracemalloc, gc.get_objects()
"""

class MemoryMonitor:
    def __init__(self, interval=1.0, threshold_mb=100):
        # TODO: Implementation
        pass

    def start(self):
        # TODO: Monitoring'i başlat
        pass

    def stop(self):
        # TODO: Monitoring'i durdur
        pass

    def get_report(self):
        # TODO: Rapor oluştur
        pass


# SOLUTION:
import threading
from collections import deque


class MemoryMonitorSolution:
    """Real-time bellek monitörü"""

    def __init__(self, interval: float = 1.0, threshold_mb: float = 100,
                 history_size: int = 100):
        """
        Args:
            interval: Ölçüm aralığı (seconds)
            threshold_mb: Alert threshold (MB)
            history_size: Tutulacak ölçüm sayısı
        """
        self.interval = interval
        self.threshold_bytes = threshold_mb * 1024 * 1024
        self.history_size = history_size

        self._monitoring = False
        self._thread = None
        self._history = deque(maxlen=history_size)
        self._alerts = []
        self._top_consumers = []

    def start(self):
        """Monitoring'i başlat"""
        if self._monitoring:
            print("Monitor zaten çalışıyor")
            return

        self._monitoring = True
        tracemalloc.start()

        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()

        print(f"Memory monitor başladı (interval={self.interval}s)")

    def stop(self):
        """Monitoring'i durdur"""
        if not self._monitoring:
            return

        self._monitoring = False
        if self._thread:
            self._thread.join(timeout=self.interval * 2)

        tracemalloc.stop()
        print("Memory monitor durdu")

    def _monitor_loop(self):
        """Monitoring loop"""
        while self._monitoring:
            try:
                # Mevcut bellek kullanımı
                current, peak = tracemalloc.get_traced_memory()

                # Snapshot al
                snapshot = tracemalloc.take_snapshot()
                top_stats = snapshot.statistics('lineno')

                # Kaydet
                measurement = {
                    'timestamp': time.time(),
                    'current_mb': current / 1024 / 1024,
                    'peak_mb': peak / 1024 / 1024,
                }
                self._history.append(measurement)

                # Top consumers
                self._top_consumers = [
                    {
                        'file': str(stat.traceback[0].filename).split('/')[-1],
                        'line': stat.traceback[0].lineno,
                        'size_mb': stat.size / 1024 / 1024,
                    }
                    for stat in top_stats[:5]
                ]

                # Threshold kontrolü
                if current > self.threshold_bytes:
                    alert = {
                        'timestamp': time.time(),
                        'current_mb': current / 1024 / 1024,
                        'threshold_mb': self.threshold_bytes / 1024 / 1024,
                    }
                    self._alerts.append(alert)
                    print(f"\n⚠️  MEMORY ALERT: {alert['current_mb']:.2f} MB "
                          f"(threshold: {alert['threshold_mb']:.2f} MB)")

                time.sleep(self.interval)

            except Exception as e:
                print(f"Monitor error: {e}")
                time.sleep(self.interval)

    def get_report(self):
        """Detaylı rapor"""
        if not self._history:
            return "Henüz veri yok"

        # İstatistikler
        current_usage = self._history[-1]['current_mb']
        peak_usage = max(m['peak_mb'] for m in self._history)
        avg_usage = sum(m['current_mb'] for m in self._history) / len(self._history)

        # Memory trend
        if len(self._history) > 10:
            recent_avg = sum(m['current_mb'] for m in list(self._history)[-10:]) / 10
            old_avg = sum(m['current_mb'] for m in list(self._history)[:10]) / 10
            trend = "↑ Artıyor" if recent_avg > old_avg * 1.1 else "→ Stabil"
        else:
            trend = "→ Yeterli veri yok"

        report = f"""
{'=' * 60}
Memory Monitor Report
{'=' * 60}

Current Usage: {current_usage:.2f} MB
Peak Usage: {peak_usage:.2f} MB
Average Usage: {avg_usage:.2f} MB
Trend: {trend}

Alerts: {len(self._alerts)}
Measurements: {len(self._history)}

Top 5 Memory Consumers:
"""
        for i, consumer in enumerate(self._top_consumers, 1):
            report += f"  {i}. {consumer['file']}:{consumer['line']} - {consumer['size_mb']:.2f} MB\n"

        if self._alerts:
            report += f"\nRecent Alerts (last 5):\n"
            for alert in self._alerts[-5:]:
                ts = time.strftime('%H:%M:%S', time.localtime(alert['timestamp']))
                report += f"  [{ts}] {alert['current_mb']:.2f} MB\n"

        return report

    def get_history(self):
        """Measurement history"""
        return list(self._history)


def test_memory_monitor():
    """Memory monitor test"""
    print("\n" + "=" * 60)
    print("TEST: Memory Monitor")
    print("=" * 60)

    monitor = MemoryMonitorSolution(interval=0.5, threshold_mb=10)
    monitor.start()

    # Bellek kullan
    print("\n--- Allocating memory ---")
    data = []
    for i in range(10):
        chunk = [0] * 100_000  # ~800KB
        data.append(chunk)
        time.sleep(0.3)

    time.sleep(2)

    # Rapor
    print(monitor.get_report())

    # Temizle
    del data
    gc.collect()

    time.sleep(1)
    print("\n--- After cleanup ---")
    print(monitor.get_report())

    monitor.stop()


# ============================================================================
# MAIN: TÜM TESTLERİ ÇALIŞTIR
# ============================================================================

def run_all_tests():
    """Tüm exercise'ların testlerini çalıştır"""

    tests = [
        ("Memory Leak Detector", test_memory_leak_detector),
        ("Smart Cache", test_smart_cache),
        ("Memory Profiler", test_memory_profiler),
        ("Object Pool", test_object_pool),
        ("Circular Reference Detector", test_circular_reference_detector),
        ("Efficient List", test_efficient_list),
        ("Weak Cache", test_weak_cache),
        ("Memory Monitor", test_memory_monitor),
    ]

    print("\n" + "=" * 60)
    print("MEMORY MANAGEMENT - TÜM TESTLER")
    print("=" * 60)

    for name, test_func in tests:
        try:
            test_func()
        except Exception as e:
            print(f"\n❌ {name} FAILED: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 60)
    print("TÜM TESTLER TAMAMLANDI")
    print("=" * 60)


if __name__ == "__main__":
    # Her exercise'ı ayrı ayrı test edebilirsiniz:
    # test_memory_leak_detector()
    # test_smart_cache()
    # test_memory_profiler()
    # test_object_pool()
    # test_circular_reference_detector()
    # test_efficient_list()
    # test_weak_cache()
    # test_memory_monitor()

    # Veya hepsini birden:
    run_all_tests()
