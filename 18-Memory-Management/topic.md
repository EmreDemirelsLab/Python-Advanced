# Memory Management (Bellek Yönetimi)

## İçindekiler
1. [Garbage Collection (Çöp Toplama)](#garbage-collection)
2. [Reference Counting (Referans Sayımı)](#reference-counting)
3. [Memory Leaks Detection (Bellek Sızıntılarını Tespit)](#memory-leaks-detection)
4. [__slots__ Optimization](#slots-optimization)
5. [Weak References (Zayıf Referanslar)](#weak-references)
6. [Memory Profiling (Bellek Profilleme)](#memory-profiling)
7. [Object Internals (Nesne İçyapısı)](#object-internals)
8. [Memory Optimization Patterns](#memory-optimization-patterns)

---

## Garbage Collection (Çöp Toplama)

Python'da bellek yönetimi otomatiktir ve iki temel mekanizma kullanır: **Reference Counting** ve **Garbage Collection**. GC modülü, döngüsel referansları tespit edip temizler.

### Örnek 1: GC Modülü Temel Kullanım

```python
import gc
import sys

# GC istatistiklerini kontrol etme
def gc_statistics():
    """GC durumunu ve istatistiklerini gösterir"""
    print(f"GC Etkin mi: {gc.isenabled()}")
    print(f"GC Sayaçları: {gc.get_count()}")  # (gen0, gen1, gen2)
    print(f"GC Eşikleri: {gc.get_threshold()}")  # (threshold0, threshold1, threshold2)
    print(f"Toplanan Nesneler: {gc.collect()}")

gc_statistics()

# Output:
# GC Etkin mi: True
# GC Sayaçları: (421, 3, 2)
# GC Eşikleri: (700, 10, 10)
# Toplanan Nesneler: 0
```

**Açıklama**: Python'un garbage collector'ı üç jenerasyon kullanır. Gen0 en genç nesneleri içerir ve en sık taranır. Eşikler, her jenerasyonun ne zaman toplanacağını belirler.

### Örnek 2: Generational Garbage Collection

```python
import gc

class TrackedObject:
    """GC tarafından takip edilen nesne"""
    _instances = []

    def __init__(self, name):
        self.name = name
        TrackedObject._instances.append(self)
        print(f"Oluşturuldu: {name}")

    def __del__(self):
        print(f"Silindi: {self.name}")

# Gen0'da nesneler oluşturma
def create_gen0_objects():
    """Gen0'da nesneler oluşturur"""
    for i in range(5):
        obj = TrackedObject(f"Gen0-{i}")
        # Referans silinir, GC toplayacak

    print(f"\nGen0 count önce: {gc.get_count()[0]}")
    collected = gc.collect(0)  # Sadece Gen0'ı topla
    print(f"Gen0'dan toplanan: {collected}")
    print(f"Gen0 count sonra: {gc.get_count()[0]}")

create_gen0_objects()

# Jenerasyon istatistikleri
def generation_stats():
    """Her jenerasyondaki nesne sayısını gösterir"""
    for i in range(3):
        objects = gc.get_objects(generation=i)
        print(f"Gen{i}: {len(objects)} nesne")

generation_stats()
```

**Açıklama**: Python'un GC'si nesne yaşlarına göre 3 jenerasyona ayırır. Genç nesneler (Gen0) sık, yaşlı nesneler (Gen2) seyrek toplanır. Bu, "generational hypothesis"e dayanır.

### Örnek 3: Döngüsel Referans Tespiti

```python
import gc
import weakref

class Node:
    """Döngüsel referans oluşturabilecek node sınıfı"""
    def __init__(self, value):
        self.value = value
        self.next = None
        self.prev = None

    def __repr__(self):
        return f"Node({self.value})"

    def __del__(self):
        print(f"Node {self.value} silindi")

# Döngüsel referans oluşturma
def create_circular_reference():
    """İki nesne arasında döngüsel referans oluşturur"""
    node1 = Node(1)
    node2 = Node(2)

    # Döngüsel referans
    node1.next = node2
    node2.prev = node1

    print(f"Referans sayısı (node1): {sys.getrefcount(node1) - 1}")
    print(f"Referans sayısı (node2): {sys.getrefcount(node2) - 1}")

    # Fonksiyon bitince referanslar kaybolur ama döngü var
    return node1, node2

# GC olmadan
print("=== GC Devre Dışı ===")
gc.disable()
n1, n2 = create_circular_reference()
del n1, n2
print(f"Silme sonrası GC count: {gc.get_count()}")

# GC ile
print("\n=== GC Devrede ===")
gc.enable()
n1, n2 = create_circular_reference()
del n1, n2
collected = gc.collect()
print(f"GC toplanan: {collected} nesne")
```

**Açıklama**: Döngüsel referanslar (A→B, B→A) reference counting ile temizlenemez. GC modülü bu döngüleri tespit edip temizler. `gc.disable()` ile GC'yi devre dışı bırakabilirsiniz.

### Örnek 4: GC Callbacks ve Debugging

```python
import gc
import sys

class DebugObject:
    """GC debug için özel nesne"""
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return f"DebugObject({self.name})"

    def __del__(self):
        print(f"__del__ çağrıldı: {self.name}")

# GC debug flags
def enable_gc_debugging():
    """GC debug modunu aktif eder"""
    # DEBUG_STATS: İstatistikleri yazdır
    # DEBUG_LEAK: Sızıntıları tespit et
    # DEBUG_UNCOLLECTABLE: Toplanamayan nesneleri göster
    gc.set_debug(gc.DEBUG_STATS | gc.DEBUG_UNCOLLECTABLE)

    # Döngüsel referans oluştur
    a = DebugObject("A")
    b = DebugObject("B")
    a.ref = b
    b.ref = a

    del a, b

    print("\n=== GC Collect çağrılıyor ===")
    gc.collect()

    # Toplanamayan nesneleri kontrol et
    if gc.garbage:
        print(f"\nToplanamayan nesneler: {gc.garbage}")
    else:
        print("\nTüm nesneler temizlendi")

    gc.set_debug(0)  # Debug modunu kapat

enable_gc_debugging()

# GC callbacks
def gc_callback(phase, info):
    """GC her çalıştığında çağrılır"""
    print(f"GC Phase: {phase}, Info: {info}")

gc.callbacks.append(gc_callback)

# Test
objs = [DebugObject(f"Test-{i}") for i in range(3)]
del objs
gc.collect()
```

**Açıklama**: `gc.set_debug()` ile GC'nin davranışını izleyebilirsiniz. `gc.garbage` listesi toplanamayan nesneleri içerir (genellikle `__del__` metodlu döngüsel referanslar).

---

## Reference Counting (Referans Sayımı)

Python'da her nesnenin bir referans sayacı vardır. Sayaç 0'a düştüğünde nesne hemen silinir. Bu, bellek yönetiminin birincil mekanizmasıdır.

### Örnek 5: Reference Counting Detaylı

```python
import sys
import ctypes

class RefCounter:
    """Referans sayımını gösterir"""
    def __init__(self, name):
        self.name = name
        print(f"{name} oluşturuldu, refcount: {sys.getrefcount(self) - 1}")

    def __del__(self):
        print(f"{self.name} silindi")

def analyze_refcount():
    """Referans sayımını detaylı analiz eder"""
    obj = RefCounter("Test")
    print(f"1. Durum: {sys.getrefcount(obj) - 1}")  # -1: getrefcount'un kendi referansı

    # Liste içine koy
    lst = [obj]
    print(f"2. Liste'ye eklendi: {sys.getrefcount(obj) - 1}")

    # Dictionary'ye koy
    dct = {'obj': obj}
    print(f"3. Dict'e eklendi: {sys.getrefcount(obj) - 1}")

    # Fonksiyona gönder
    def use_obj(o):
        print(f"4. Fonksiyon içinde: {sys.getrefcount(o) - 1}")

    use_obj(obj)
    print(f"5. Fonksiyondan sonra: {sys.getrefcount(obj) - 1}")

    # Referansları temizle
    del lst
    print(f"6. Liste silindi: {sys.getrefcount(obj) - 1}")

    del dct
    print(f"7. Dict silindi: {sys.getrefcount(obj) - 1}")

    # Son referans
    print("8. obj siliniyor...")
    del obj

analyze_refcount()

# C-level referans sayısını okuma
def get_refcount_c_level(obj):
    """C seviyesinde gerçek referans sayısını döndürür"""
    # Python nesnesinin C struct'ındaki ob_refcnt alanını okur
    return ctypes.c_long.from_address(id(obj)).value

obj = [1, 2, 3]
print(f"\nC-level refcount: {get_refcount_c_level(obj)}")
print(f"sys.getrefcount: {sys.getrefcount(obj)}")
```

**Açıklama**: `sys.getrefcount()` nesneye kaç referans olduğunu gösterir. Her atama, liste ekleme, dict değeri yapma refcount'u artırır. Fonksiyon parametresi de geçici bir referans oluşturur.

### Örnek 6: Reference Counting ve Performance

```python
import sys
import time

def measure_refcount_overhead():
    """Referans sayımının performans etkisini ölçer"""

    # Test 1: Basit atama
    start = time.perf_counter()
    for _ in range(1_000_000):
        x = [1, 2, 3]
        del x
    elapsed1 = time.perf_counter() - start
    print(f"Basit atama/silme: {elapsed1:.4f}s")

    # Test 2: Çoklu referans
    start = time.perf_counter()
    for _ in range(1_000_000):
        x = [1, 2, 3]
        y = x  # Refcount +1
        z = x  # Refcount +1
        del x, y, z
    elapsed2 = time.perf_counter() - start
    print(f"Çoklu referans: {elapsed2:.4f}s")

    # Test 3: Container içinde
    start = time.perf_counter()
    for _ in range(1_000_000):
        x = [1, 2, 3]
        lst = [x] * 10  # 10 referans
        del lst, x
    elapsed3 = time.perf_counter() - start
    print(f"Container içinde: {elapsed3:.4f}s")

measure_refcount_overhead()

# Interning ve referans sayımı
def string_interning():
    """String interning'in referans sayımına etkisi"""
    # Küçük stringler ve integer'lar interned olur
    s1 = "hello"
    s2 = "hello"
    print(f"\nString interning:")
    print(f"s1 is s2: {s1 is s2}")
    print(f"id(s1) == id(s2): {id(s1) == id(s2)}")
    print(f"Refcount: {sys.getrefcount(s1) - 1}")

    # Integer interning (-5 to 256)
    i1 = 100
    i2 = 100
    print(f"\nInteger interning:")
    print(f"i1 is i2: {i1 is i2}")
    print(f"Refcount: {sys.getrefcount(i1) - 1}")

    # Büyük sayılar interned değil
    i3 = 1000
    i4 = 1000
    print(f"\nBüyük integer:")
    print(f"i3 is i4: {i3 is i4}")

string_interning()
```

**Açıklama**: Referans sayımı her atama/silme işleminde overhead yaratır. Python bazı değerleri (küçük int'ler, stringler) "intern" eder - bellekte tek kopya tutar.

---

## Memory Leaks Detection (Bellek Sızıntılarını Tespit)

Bellek sızıntıları, kullanılmayan nesnelerin referanslarının silinmemesi sonucu oluşur. Python'da genellikle döngüsel referanslar, global değişkenler veya cache'ler sorun yaratır.

### Örnek 7: tracemalloc ile Bellek Sızıntısı Tespiti

```python
import tracemalloc
import gc

class LeakyClass:
    """Bellek sızıntısına neden olabilecek sınıf"""
    _cache = []  # Global cache - potansiyel sızıntı kaynağı

    def __init__(self, data):
        self.data = data
        LeakyClass._cache.append(self)  # Cache'e ekleniyor

    def __repr__(self):
        return f"LeakyClass(data={len(self.data)})"

def detect_memory_leak():
    """Bellek sızıntısını tespit eder"""
    # tracemalloc'u başlat
    tracemalloc.start()

    # İlk snapshot
    snapshot1 = tracemalloc.take_snapshot()

    # Bellek kullanan işlem
    objects = []
    for i in range(1000):
        obj = LeakyClass([0] * 1000)  # Her biri ~8KB
        objects.append(obj)

    # objects listesini temizle ama cache'te hala referans var!
    del objects
    gc.collect()

    # İkinci snapshot
    snapshot2 = tracemalloc.take_snapshot()

    # Farkları analiz et
    top_stats = snapshot2.compare_to(snapshot1, 'lineno')

    print("[ Top 10 bellek artışı ]")
    for stat in top_stats[:10]:
        print(stat)

    # Bellek kullanımı
    current, peak = tracemalloc.get_traced_memory()
    print(f"\nMevcut bellek: {current / 1024 / 1024:.2f} MB")
    print(f"Peak bellek: {peak / 1024 / 1024:.2f} MB")

    # Cache'i temizle (sızıntıyı düzelt)
    print(f"\nCache boyutu: {len(LeakyClass._cache)}")
    LeakyClass._cache.clear()
    gc.collect()

    # Son durum
    snapshot3 = tracemalloc.take_snapshot()
    current, peak = tracemalloc.get_traced_memory()
    print(f"Cache temizlendi, yeni bellek: {current / 1024 / 1024:.2f} MB")

    tracemalloc.stop()

detect_memory_leak()
```

**Açıklama**: `tracemalloc` modülü bellek kullanımını satır bazında izler. `compare_to()` ile iki snapshot arasındaki farkı görebilirsiniz. Global cache'ler yaygın sızıntı kaynağıdır.

### Örnek 8: objgraph ile Referans Ağacı Analizi

```python
import gc
import sys

# objgraph yerine basit bir implementasyon
class MemoryLeakDetector:
    """Bellek sızıntılarını tespit eden araç"""

    @staticmethod
    def find_referrers(obj):
        """Bir nesneye referans veren nesneleri bulur"""
        referrers = gc.get_referrers(obj)
        return [r for r in referrers if not isinstance(r, type(sys._getframe()))]

    @staticmethod
    def find_referents(obj):
        """Bir nesnenin referans verdiği nesneleri bulur"""
        return gc.get_referents(obj)

    @staticmethod
    def show_refs(obj, max_depth=2, _depth=0, _seen=None):
        """Referans ağacını gösterir"""
        if _seen is None:
            _seen = set()

        if id(obj) in _seen or _depth > max_depth:
            return

        _seen.add(id(obj))
        indent = "  " * _depth
        print(f"{indent}{type(obj).__name__}: {id(obj)}")

        if _depth < max_depth:
            referents = gc.get_referents(obj)
            for ref in referents[:5]:  # İlk 5 referansı göster
                if not isinstance(ref, type):
                    MemoryLeakDetector.show_refs(ref, max_depth, _depth + 1, _seen)

# Döngüsel referans örneği
class Container:
    def __init__(self, name):
        self.name = name
        self.children = []

    def add_child(self, child):
        self.children.append(child)
        child.parent = self  # Geri referans - potansiyel sızıntı

def analyze_circular_reference():
    """Döngüsel referansı analiz eder"""
    parent = Container("Parent")
    child1 = Container("Child1")
    child2 = Container("Child2")

    parent.add_child(child1)
    parent.add_child(child2)

    print("=== Parent'a referans verenler ===")
    referrers = MemoryLeakDetector.find_referrers(parent)
    for i, ref in enumerate(referrers[:5], 1):
        print(f"{i}. {type(ref).__name__}")

    print("\n=== Child1'in referans ağacı ===")
    MemoryLeakDetector.show_refs(child1, max_depth=2)

    print(f"\n=== Referans sayıları ===")
    print(f"Parent: {sys.getrefcount(parent) - 1}")
    print(f"Child1: {sys.getrefcount(child1) - 1}")

    # Temizleme
    del parent, child1, child2
    collected = gc.collect()
    print(f"\nGC toplanan: {collected} nesne")

analyze_circular_reference()
```

**Açıklama**: `gc.get_referrers()` bir nesneye kimin referans verdiğini gösterir. Döngüsel referansları tespit etmek için referans ağacını takip edebilirsiniz.

### Örnek 9: Memory Leak Patterns ve Çözümleri

```python
import weakref
from typing import List, Dict

# PATTERN 1: Event Listener Leak
class BadEventEmitter:
    """Bellek sızıntısına neden olan event emitter"""
    def __init__(self):
        self.listeners: List = []

    def on(self, callback):
        self.listeners.append(callback)  # Strong reference - leak!

    def emit(self, *args):
        for callback in self.listeners:
            callback(*args)

class GoodEventEmitter:
    """Weak reference kullanan güvenli event emitter"""
    def __init__(self):
        self.listeners: List = []

    def on(self, callback):
        # Weak reference kullan
        self.listeners.append(weakref.ref(callback))

    def emit(self, *args):
        # Ölü referansları temizle
        self.listeners = [ref for ref in self.listeners if ref() is not None]
        for callback_ref in self.listeners:
            callback = callback_ref()
            if callback:
                callback(*args)

# PATTERN 2: Cache Leak
class BadCache:
    """Sınırsız büyüyen cache"""
    def __init__(self):
        self._cache: Dict = {}

    def get(self, key):
        if key not in self._cache:
            self._cache[key] = self._compute(key)
        return self._cache[key]

    def _compute(self, key):
        return [0] * 10000  # Büyük veri

class GoodCache:
    """LRU eviction stratejisi olan cache"""
    def __init__(self, max_size=100):
        self._cache: Dict = {}
        self._max_size = max_size
        self._access_order: List = []

    def get(self, key):
        if key not in self._cache:
            if len(self._cache) >= self._max_size:
                # En eski girdiyi çıkar
                oldest = self._access_order.pop(0)
                del self._cache[oldest]

            self._cache[key] = self._compute(key)
            self._access_order.append(key)
        else:
            # LRU güncelle
            self._access_order.remove(key)
            self._access_order.append(key)

        return self._cache[key]

    def _compute(self, key):
        return [0] * 10000

# PATTERN 3: Closure Leak
def create_bad_closure():
    """Closure içinde büyük veri tutan fonksiyon"""
    big_data = [0] * 1_000_000  # 8MB veri

    def processor():
        # big_data kullanılmasa bile closure'da tutulur
        return "processed"

    return processor

def create_good_closure():
    """Sadece gerekli veriyi tutan closure"""
    big_data = [0] * 1_000_000
    # Sadece gerekli kısmı al
    needed = len(big_data)
    del big_data  # Büyük veriyi serbest bırak

    def processor():
        return f"processed {needed} items"

    return processor

# Test
def test_patterns():
    """Memory leak pattern'lerini test eder"""
    import sys

    # Event emitter test
    print("=== Event Emitter Test ===")
    bad_emitter = BadEventEmitter()
    good_emitter = GoodEventEmitter()

    def handler():
        pass

    bad_emitter.on(handler)
    good_emitter.on(handler)

    print(f"Bad emitter listeners: {len(bad_emitter.listeners)}")
    print(f"Good emitter listeners: {len(good_emitter.listeners)}")

    del handler
    gc.collect()

    print(f"After del - Bad: {len(bad_emitter.listeners)}")
    good_emitter.emit()  # Cleanup çağrılır
    print(f"After del - Good: {len(good_emitter.listeners)}")

    # Closure test
    print("\n=== Closure Test ===")
    bad_func = create_bad_closure()
    good_func = create_good_closure()

    print(f"Bad closure size: {sys.getsizeof(bad_func)} bytes")
    print(f"Good closure size: {sys.getsizeof(good_func)} bytes")

test_patterns()
```

**Açıklama**: Yaygın bellek sızıntı pattern'leri: 1) Event listener'lar (weak ref kullan), 2) Sınırsız cache (eviction stratejisi), 3) Closure'larda gereksiz veri (sadece gerekeni tut).

---

## __slots__ Optimization

`__slots__` sınıf değişkenleri için `__dict__` yerine sabit boyutlu tuple kullanır. Bu, bellek kullanımını %50'ye kadar azaltabilir.

### Örnek 10: __slots__ Temel Kullanım

```python
import sys

class WithoutSlots:
    """Normal sınıf - __dict__ kullanır"""
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

class WithSlots:
    """__slots__ kullanan sınıf"""
    __slots__ = ['x', 'y', 'z']

    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

# Bellek karşılaştırması
obj1 = WithoutSlots(1, 2, 3)
obj2 = WithSlots(1, 2, 3)

print("=== Bellek Kullanımı ===")
print(f"Without __slots__: {sys.getsizeof(obj1)} bytes")
print(f"With __slots__: {sys.getsizeof(obj2)} bytes")
print(f"Tasarruf: {sys.getsizeof(obj1) - sys.getsizeof(obj2)} bytes")

# __dict__ kontrolü
print("\n=== __dict__ Kontrolü ===")
print(f"Without __slots__ has __dict__: {hasattr(obj1, '__dict__')}")
print(f"With __slots__ has __dict__: {hasattr(obj2, '__dict__')}")

if hasattr(obj1, '__dict__'):
    print(f"obj1.__dict__: {obj1.__dict__}")

# __slots__ ile dinamik attribute eklenemez
try:
    obj2.new_attr = 100
except AttributeError as e:
    print(f"\n__slots__ hatası: {e}")

# Çok sayıda nesne ile test
def compare_memory_usage(count=100_000):
    """Çok sayıda nesne ile bellek kullanımını karşılaştırır"""
    import gc
    import tracemalloc

    tracemalloc.start()

    # Without slots
    snapshot1 = tracemalloc.take_snapshot()
    objects1 = [WithoutSlots(i, i+1, i+2) for i in range(count)]
    snapshot2 = tracemalloc.take_snapshot()

    stats = snapshot2.compare_to(snapshot1, 'lineno')
    without_slots_memory = sum(stat.size_diff for stat in stats)

    del objects1
    gc.collect()

    # With slots
    snapshot3 = tracemalloc.take_snapshot()
    objects2 = [WithSlots(i, i+1, i+2) for i in range(count)]
    snapshot4 = tracemalloc.take_snapshot()

    stats = snapshot4.compare_to(snapshot3, 'lineno')
    with_slots_memory = sum(stat.size_diff for stat in stats)

    print(f"\n=== {count:,} Nesne ===")
    print(f"Without __slots__: {without_slots_memory / 1024 / 1024:.2f} MB")
    print(f"With __slots__: {with_slots_memory / 1024 / 1024:.2f} MB")
    print(f"Tasarruf: {(without_slots_memory - with_slots_memory) / 1024 / 1024:.2f} MB")

    tracemalloc.stop()

compare_memory_usage()
```

**Açıklama**: `__slots__` her instance için `__dict__` oluşturmayı engeller. Milyonlarca nesne için büyük bellek tasarrufu sağlar ama dinamik attribute ekleyemezsiniz.

### Örnek 11: __slots__ ve Inheritance

```python
import sys

class Base:
    """__slots__ ile base sınıf"""
    __slots__ = ['x', 'y']

    def __init__(self, x, y):
        self.x = x
        self.y = y

class DerivedWithSlots(Base):
    """__slots__ ile türetilmiş sınıf"""
    __slots__ = ['z']  # Sadece yeni attribute'ları tanımla

    def __init__(self, x, y, z):
        super().__init__(x, y)
        self.z = z

class DerivedWithoutSlots(Base):
    """__slots__ olmadan türetilmiş sınıf"""
    def __init__(self, x, y, w):
        super().__init__(x, y)
        self.w = w  # __dict__ oluşturulur

# Karşılaştırma
obj1 = Base(1, 2)
obj2 = DerivedWithSlots(1, 2, 3)
obj3 = DerivedWithoutSlots(1, 2, 4)

print("=== Inheritance ve __slots__ ===")
print(f"Base: {sys.getsizeof(obj1)} bytes - __dict__: {hasattr(obj1, '__dict__')}")
print(f"DerivedWithSlots: {sys.getsizeof(obj2)} bytes - __dict__: {hasattr(obj2, '__dict__')}")
print(f"DerivedWithoutSlots: {sys.getsizeof(obj3)} bytes - __dict__: {hasattr(obj3, '__dict__')}")

# __slots__ + __dict__ kombinasyonu
class Hybrid:
    """Hem __slots__ hem __dict__ kullanan sınıf"""
    __slots__ = ['x', 'y', '__dict__']  # __dict__ ekle

    def __init__(self, x, y):
        self.x = x
        self.y = y

obj4 = Hybrid(1, 2)
obj4.dynamic = 100  # __dict__ olduğu için çalışır

print(f"\nHybrid: {sys.getsizeof(obj4)} bytes")
print(f"Has __dict__: {hasattr(obj4, '__dict__')}")
print(f"obj4.__dict__: {obj4.__dict__}")

# __weakref__ için __slots__
class WeakRefable:
    """Weak reference destekleyen __slots__ sınıfı"""
    __slots__ = ['x', 'y', '__weakref__']

    def __init__(self, x, y):
        self.x = x
        self.y = y

import weakref

obj5 = WeakRefable(1, 2)
weak_ref = weakref.ref(obj5)
print(f"\nWeakref oluşturuldu: {weak_ref() is not None}")
```

**Açıklama**: Türetilmiş sınıflarda sadece yeni attribute'ları `__slots__`'a ekleyin. `__dict__` veya `__weakref__` gerekiyorsa bunları da `__slots__`'a ekleyebilirsiniz.

### Örnek 12: __slots__ Performance Test

```python
import time
import sys

class Point2D:
    __slots__ = ['x', 'y']

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def distance(self):
        return (self.x ** 2 + self.y ** 2) ** 0.5

class Point2DDict:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def distance(self):
        return (self.x ** 2 + self.y ** 2) ** 0.5

def benchmark_slots_vs_dict(count=1_000_000):
    """__slots__ ve __dict__ performansını karşılaştırır"""

    # Creation benchmark
    print(f"=== {count:,} nesne oluşturma ===")

    start = time.perf_counter()
    slots_objects = [Point2D(i, i+1) for i in range(count)]
    slots_time = time.perf_counter() - start
    print(f"__slots__: {slots_time:.4f}s")

    start = time.perf_counter()
    dict_objects = [Point2DDict(i, i+1) for i in range(count)]
    dict_time = time.perf_counter() - start
    print(f"__dict__: {dict_time:.4f}s")
    print(f"Speedup: {dict_time / slots_time:.2f}x")

    # Attribute access benchmark
    print(f"\n=== Attribute erişim ===")

    start = time.perf_counter()
    for obj in slots_objects:
        _ = obj.x + obj.y
    slots_access_time = time.perf_counter() - start
    print(f"__slots__: {slots_access_time:.4f}s")

    start = time.perf_counter()
    for obj in dict_objects:
        _ = obj.x + obj.y
    dict_access_time = time.perf_counter() - start
    print(f"__dict__: {dict_access_time:.4f}s")
    print(f"Speedup: {dict_access_time / slots_access_time:.2f}x")

    # Memory usage
    print(f"\n=== Bellek kullanımı ===")
    slots_memory = sys.getsizeof(slots_objects[0]) * count
    dict_memory = sys.getsizeof(dict_objects[0]) * count
    print(f"__slots__: {slots_memory / 1024 / 1024:.2f} MB")
    print(f"__dict__: {dict_memory / 1024 / 1024:.2f} MB")
    print(f"Tasarruf: {(dict_memory - slots_memory) / 1024 / 1024:.2f} MB")

benchmark_slots_vs_dict()
```

**Açıklama**: `__slots__` nesne oluşturmayı ve attribute erişimi hızlandırır. Özellikle milyonlarca küçük nesne için ideal (Point, Vector, Coordinate gibi).

---

## Weak References (Zayıf Referanslar)

Weak reference, nesneyi "canlı tutmayan" bir referanstır. Referans sayısına dahil olmaz ve GC nesneyi toplayabilir.

### Örnek 13: weakref Temel Kullanım

```python
import weakref
import gc

class ExpensiveObject:
    """Pahalı nesne"""
    def __init__(self, name):
        self.name = name
        self.data = [0] * 1_000_000  # ~8MB

    def __repr__(self):
        return f"ExpensiveObject({self.name})"

    def __del__(self):
        print(f"{self.name} silindi")

# Strong reference
print("=== Strong Reference ===")
obj1 = ExpensiveObject("Obj1")
ref1 = obj1  # Strong reference
print(f"Refcount: {sys.getrefcount(obj1) - 1}")
del obj1
print("obj1 silindi ama ref1 hala tutuyor")
print(f"ref1: {ref1}")
del ref1
gc.collect()

# Weak reference
print("\n=== Weak Reference ===")
obj2 = ExpensiveObject("Obj2")
weak_ref = weakref.ref(obj2)  # Weak reference
print(f"Refcount: {sys.getrefcount(obj2) - 1}")  # Weak ref sayılmaz
print(f"Weak ref geçerli: {weak_ref() is not None}")
print(f"Weak ref: {weak_ref()}")

del obj2
gc.collect()
print(f"obj2 silindi, weak ref: {weak_ref()}")  # None döner

# WeakValueDictionary
print("\n=== WeakValueDictionary ===")
cache = weakref.WeakValueDictionary()

obj3 = ExpensiveObject("Cached")
cache['key1'] = obj3
print(f"Cache'te: {'key1' in cache}")
print(f"Cache value: {cache.get('key1')}")

del obj3
gc.collect()
print(f"obj3 silindi, cache'te: {'key1' in cache}")
```

**Açıklama**: Weak reference ile cache, observer pattern gibi yapılarda bellek sızıntısını önleyebilirsiniz. Nesne başka yerden referans edilmediğinde otomatik temizlenir.

### Örnek 14: WeakSet ve WeakKeyDictionary

```python
import weakref
import gc

class Observer:
    """Observer pattern için gözlemci"""
    def __init__(self, name):
        self.name = name

    def update(self, message):
        print(f"{self.name} güncellendi: {message}")

    def __repr__(self):
        return f"Observer({self.name})"

class Subject:
    """WeakSet kullanan subject"""
    def __init__(self):
        self._observers = weakref.WeakSet()

    def attach(self, observer):
        """Observer ekle"""
        self._observers.add(observer)
        print(f"{observer} eklendi")

    def notify(self, message):
        """Tüm observer'ları bilgilendir"""
        print(f"\nBildiriliyor: {len(self._observers)} observer")
        for observer in self._observers:
            observer.update(message)

# Test
subject = Subject()

# Observer'lar oluştur
obs1 = Observer("Obs1")
obs2 = Observer("Obs2")
obs3 = Observer("Obs3")

subject.attach(obs1)
subject.attach(obs2)
subject.attach(obs3)

subject.notify("İlk mesaj")

# Bir observer'ı sil
print("\n=== obs2 siliniyor ===")
del obs2
gc.collect()

subject.notify("İkinci mesaj")

# WeakKeyDictionary örneği
print("\n=== WeakKeyDictionary ===")

class SessionManager:
    """Weak key dict kullanan session manager"""
    def __init__(self):
        self._sessions = weakref.WeakKeyDictionary()

    def set_session(self, user, data):
        """User için session oluştur"""
        self._sessions[user] = data
        print(f"Session oluşturuldu: {user}")

    def get_session(self, user):
        """User'ın session'ını getir"""
        return self._sessions.get(user)

    def active_sessions(self):
        """Aktif session sayısı"""
        return len(self._sessions)

class User:
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return f"User({self.name})"

manager = SessionManager()

user1 = User("Alice")
user2 = User("Bob")

manager.set_session(user1, {"token": "abc123"})
manager.set_session(user2, {"token": "def456"})

print(f"\nAktif sessions: {manager.active_sessions()}")
print(f"User1 session: {manager.get_session(user1)}")

# user1'i sil
del user1
gc.collect()

print(f"user1 silindi, aktif sessions: {manager.active_sessions()}")
```

**Açıklama**: `WeakSet` ve `WeakKeyDictionary` otomatik temizlik sağlar. Observer pattern'de observer'lar silindiğinde otomatik liste dışı kalır.

### Örnek 15: Weak Reference Callbacks

```python
import weakref
import gc

class Resource:
    """Kaynağı temsil eden sınıf"""
    def __init__(self, name):
        self.name = name
        print(f"Resource {name} oluşturuldu")

    def __del__(self):
        print(f"Resource {self.name} silindi")

def cleanup_callback(weak_ref):
    """Weak reference ölünce çağrılır"""
    print(f"Callback: Nesne toplandı (weak_ref: {weak_ref})")

# Callback ile weak reference
print("=== Weak Reference Callback ===")
resource = Resource("DB Connection")
weak_ref = weakref.ref(resource, cleanup_callback)

print(f"Resource var: {weak_ref() is not None}")
del resource
gc.collect()
print(f"Resource var: {weak_ref() is not None}")

# finalize ile cleanup
print("\n=== weakref.finalize ===")

class FileHandler:
    """Dosya işleyici"""
    def __init__(self, filename):
        self.filename = filename
        self.file = open(filename, 'w')
        print(f"{filename} açıldı")

        # finalize ile otomatik cleanup
        self._finalizer = weakref.finalize(self, self.close_file, self.file)

    @staticmethod
    def close_file(file):
        """Dosyayı kapat"""
        if not file.closed:
            file.close()
            print(f"Dosya kapatıldı: {file.name}")

    def __repr__(self):
        return f"FileHandler({self.filename})"

# Test
import tempfile
import os

temp_file = tempfile.mktemp(suffix='.txt')
handler = FileHandler(temp_file)

print(f"Handler oluşturuldu: {handler}")
print(f"File açık: {not handler.file.closed}")

del handler
gc.collect()

# Cleanup edildi mi kontrol et
print(f"Temp file var: {os.path.exists(temp_file)}")
if os.path.exists(temp_file):
    os.remove(temp_file)

# Proxy objects
print("\n=== weakref.proxy ===")

class Data:
    def __init__(self, value):
        self.value = value

    def get_value(self):
        return self.value

data = Data(100)
proxy = weakref.proxy(data)

print(f"Proxy değer: {proxy.get_value()}")
print(f"Proxy type: {type(proxy)}")

del data
gc.collect()

try:
    print(proxy.get_value())
except ReferenceError as e:
    print(f"Proxy hatası: {e}")
```

**Açıklama**: Weak reference callback'leri nesne toplanınca çağrılır. `weakref.finalize` ile otomatik cleanup yapabilirsiniz. `weakref.proxy` nesne gibi davranır ama weak referanstır.

---

## Memory Profiling (Bellek Profilleme)

Bellek profilleme, uygulamanızın hangi bölümlerinin çok bellek kullandığını tespit etmenizi sağlar.

### Örnek 16: memory_profiler ile Detaylı Profiling

```python
# memory_profiler yerine manuel profiling
import tracemalloc
import linecache
import os

def display_top(snapshot, key_type='lineno', limit=10):
    """En çok bellek kullanan satırları gösterir"""
    snapshot = snapshot.filter_traces((
        tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
        tracemalloc.Filter(False, "<unknown>"),
    ))
    top_stats = snapshot.statistics(key_type)

    print(f"Top {limit} lines")
    for index, stat in enumerate(top_stats[:limit], 1):
        frame = stat.traceback[0]
        filename = os.path.sep.join(frame.filename.split(os.path.sep)[-2:])
        print(f"#{index}: {filename}:{frame.lineno}: {stat.size / 1024:.1f} KiB")
        line = linecache.getline(frame.filename, frame.lineno).strip()
        if line:
            print(f"    {line}")

    other = top_stats[limit:]
    if other:
        size = sum(stat.size for stat in other)
        print(f"{len(other)} other: {size / 1024:.1f} KiB")

    total = sum(stat.size for stat in top_stats)
    print(f"Total allocated size: {total / 1024:.1f} KiB")

def memory_intensive_function():
    """Bellek yoğun fonksiyon"""
    # Liste oluştur
    big_list = [i for i in range(1_000_000)]

    # Dict oluştur
    big_dict = {i: str(i) * 10 for i in range(100_000)}

    # Nested yapı
    nested = [[j for j in range(100)] for i in range(10_000)]

    return len(big_list), len(big_dict), len(nested)

# Profiling
tracemalloc.start()

snapshot1 = tracemalloc.take_snapshot()
result = memory_intensive_function()
snapshot2 = tracemalloc.take_snapshot()

print("=== Memory Profile ===")
display_top(snapshot2, limit=5)

print("\n=== Compared to Baseline ===")
top_stats = snapshot2.compare_to(snapshot1, 'lineno')
for stat in top_stats[:5]:
    print(f"{stat}")

tracemalloc.stop()
```

**Açıklama**: `tracemalloc` her bellek allocation'ını izler. Snapshot'lar arasındaki farkı görerek hangi satırların bellek kullandığını tespit edebilirsiniz.

### Örnek 17: sys.getsizeof ve Deep Size

```python
import sys
from typing import Any

def get_size(obj, seen=None):
    """Nesnenin ve içeriğinin toplam boyutunu hesaplar"""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()

    obj_id = id(obj)
    if obj_id in seen:
        return 0

    # Döngüyü önlemek için işaretle
    seen.add(obj_id)

    # Recursive olarak içeriği hesapla
    if isinstance(obj, dict):
        size += sum(get_size(k, seen) + get_size(v, seen) for k, v in obj.items())
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        try:
            size += sum(get_size(item, seen) for item in obj)
        except TypeError:
            pass

    return size

# Test
def compare_sizes():
    """Farklı veri yapılarının boyutlarını karşılaştırır"""

    # Liste vs Tuple
    data_list = list(range(1000))
    data_tuple = tuple(range(1000))

    print("=== Liste vs Tuple ===")
    print(f"List: {sys.getsizeof(data_list)} bytes")
    print(f"Tuple: {sys.getsizeof(data_tuple)} bytes")
    print(f"Fark: {sys.getsizeof(data_list) - sys.getsizeof(data_tuple)} bytes")

    # Dict vs List of tuples
    data_dict = {i: i*2 for i in range(1000)}
    data_pairs = [(i, i*2) for i in range(1000)]

    print("\n=== Dict vs List of Tuples ===")
    print(f"Dict: {get_size(data_dict)} bytes")
    print(f"List of tuples: {get_size(data_pairs)} bytes")

    # Set vs List
    data_set = set(range(1000))
    data_list2 = list(range(1000))

    print("\n=== Set vs List ===")
    print(f"Set: {sys.getsizeof(data_set)} bytes")
    print(f"List: {sys.getsizeof(data_list2)} bytes")

    # String interning
    s1 = "hello" * 100
    s2 = "x" * 10000

    print("\n=== Strings ===")
    print(f"'hello' * 100: {sys.getsizeof(s1)} bytes")
    print(f"'x' * 10000: {sys.getsizeof(s2)} bytes")

    # Class with __slots__ vs without
    class WithSlots:
        __slots__ = ['a', 'b', 'c']
        def __init__(self):
            self.a = self.b = self.c = 1

    class WithoutSlots:
        def __init__(self):
            self.a = self.b = self.c = 1

    obj_slots = WithSlots()
    obj_dict = WithoutSlots()

    print("\n=== Class Instances ===")
    print(f"With __slots__: {get_size(obj_slots)} bytes")
    print(f"Without __slots__: {get_size(obj_dict)} bytes")

compare_sizes()

# Nested structures
def analyze_nested_structure():
    """İç içe yapıların bellek kullanımını analiz eder"""
    data = {
        'users': [
            {
                'id': i,
                'name': f'User{i}',
                'tags': [f'tag{j}' for j in range(10)],
                'metadata': {'created': '2024-01-01', 'active': True}
            }
            for i in range(100)
        ],
        'config': {'timeout': 30, 'retries': 3},
    }

    print("\n=== Nested Structure Analysis ===")
    print(f"Total size: {get_size(data) / 1024:.2f} KB")
    print(f"Users list: {get_size(data['users']) / 1024:.2f} KB")
    print(f"Single user: {get_size(data['users'][0])} bytes")
    print(f"Config: {get_size(data['config'])} bytes")

analyze_nested_structure()
```

**Açıklama**: `sys.getsizeof()` sadece nesnenin kendisini ölçer. İç içe yapılar için recursive hesaplama gerekir. Tuple < List, __slots__ < __dict__ şeklinde bellek verimliliği vardır.

---

## Object Internals (Nesne İçyapısı)

Python nesnelerinin C seviyesindeki yapısını anlamak, bellek optimizasyonu için önemlidir.

### Örnek 18: PyObject Structure

```python
import sys
import ctypes

class PyObject(ctypes.Structure):
    """Python nesnesinin C struct temsili"""
    _fields_ = [
        ('ob_refcnt', ctypes.c_ssize_t),
        ('ob_type', ctypes.c_void_p),
    ]

def get_object_info(obj):
    """Nesnenin C seviyesindeki bilgilerini gösterir"""
    py_obj = PyObject.from_address(id(obj))

    print(f"=== Object Info ===")
    print(f"Type: {type(obj).__name__}")
    print(f"ID: {id(obj)}")
    print(f"Size: {sys.getsizeof(obj)} bytes")
    print(f"RefCount (C): {py_obj.ob_refcnt}")
    print(f"RefCount (Python): {sys.getrefcount(obj) - 1}")
    print(f"Type pointer: {hex(py_obj.ob_type)}")

# Test
obj = [1, 2, 3]
get_object_info(obj)

# Object overhead
def calculate_overhead():
    """Farklı nesne tipleri için overhead hesaplar"""
    print("\n=== Object Overhead ===")

    # Empty containers
    empty_list = []
    empty_tuple = ()
    empty_dict = {}
    empty_set = set()

    print(f"Empty list: {sys.getsizeof(empty_list)} bytes")
    print(f"Empty tuple: {sys.getsizeof(empty_tuple)} bytes")
    print(f"Empty dict: {sys.getsizeof(empty_dict)} bytes")
    print(f"Empty set: {sys.getsizeof(empty_set)} bytes")

    # Single element
    one_list = [1]
    one_tuple = (1,)
    one_dict = {1: 1}
    one_set = {1}

    print(f"\nOne element list: {sys.getsizeof(one_list)} bytes")
    print(f"One element tuple: {sys.getsizeof(one_tuple)} bytes")
    print(f"One element dict: {sys.getsizeof(one_dict)} bytes")
    print(f"One element set: {sys.getsizeof(one_set)} bytes")

    # Per-element cost
    print(f"\n=== Per-element Cost ===")
    print(f"List: {sys.getsizeof(one_list) - sys.getsizeof(empty_list)} bytes/element")
    print(f"Dict: {sys.getsizeof(one_dict) - sys.getsizeof(empty_dict)} bytes/element")
    print(f"Set: {sys.getsizeof(one_set) - sys.getsizeof(empty_set)} bytes/element")

calculate_overhead()

# Integer object pool
def integer_caching():
    """Python'un integer caching mekanizması"""
    print("\n=== Integer Caching ===")

    # Small integers (-5 to 256) are cached
    a = 100
    b = 100
    print(f"a = 100, b = 100")
    print(f"a is b: {a is b}")
    print(f"id(a) == id(b): {id(a) == id(b)}")

    # Large integers are not cached
    c = 1000
    d = 1000
    print(f"\nc = 1000, d = 1000")
    print(f"c is d: {c is d}")
    print(f"id(c) == id(d): {id(c) == id(d)}")

    # But equal values
    print(f"c == d: {c == d}")

    # Integer size
    small_int = 1
    big_int = 2**100

    print(f"\n=== Integer Size ===")
    print(f"Small int (1): {sys.getsizeof(small_int)} bytes")
    print(f"Big int (2^100): {sys.getsizeof(big_int)} bytes")

integer_caching()
```

**Açıklama**: Her Python nesnesi en az `ob_refcnt` (referans sayısı) ve `ob_type` (tip pointer) içerir. Boş container'lar bile overhead taşır. Python küçük integer'ları cache'ler.

### Örnek 19: Memory Layout ve Alignment

```python
import sys
import array

def analyze_memory_layout():
    """Farklı veri tiplerinin bellek düzenini analiz eder"""

    print("=== Primitive Types ===")
    print(f"int: {sys.getsizeof(0)} bytes")
    print(f"float: {sys.getsizeof(0.0)} bytes")
    print(f"bool: {sys.getsizeof(True)} bytes")
    print(f"None: {sys.getsizeof(None)} bytes")

    print("\n=== String Types ===")
    print(f"Empty str: {sys.getsizeof('')} bytes")
    print(f"ASCII str (10 chars): {sys.getsizeof('a' * 10)} bytes")
    print(f"Unicode str (10 chars): {sys.getsizeof('😀' * 10)} bytes")
    print(f"Bytes (10): {sys.getsizeof(b'a' * 10)} bytes")

    print("\n=== Collections ===")
    # List growth
    for size in [0, 1, 5, 10, 50, 100]:
        lst = list(range(size))
        print(f"List[{size}]: {sys.getsizeof(lst)} bytes")

    print("\n=== Array Module (C arrays) ===")
    # array module kullanımı
    py_list = [i for i in range(1000)]
    c_array = array.array('i', range(1000))  # signed int

    print(f"Python list (1000 ints): {sys.getsizeof(py_list)} bytes")
    print(f"C array (1000 ints): {sys.getsizeof(c_array)} bytes")
    print(f"Tasarruf: {sys.getsizeof(py_list) - sys.getsizeof(c_array)} bytes")

    # Array types
    print("\n=== Array Types ===")
    for typecode, name in [('b', 'signed char'), ('h', 'short'), ('i', 'int'),
                            ('l', 'long'), ('f', 'float'), ('d', 'double')]:
        arr = array.array(typecode, range(100))
        print(f"{name} ({typecode}): {sys.getsizeof(arr)} bytes")

analyze_memory_layout()

# Dict size ve resize
def dict_resizing():
    """Dictionary'nin resize mekanizması"""
    print("\n=== Dictionary Resizing ===")
    d = {}
    prev_size = sys.getsizeof(d)

    for i in range(100):
        d[i] = i
        new_size = sys.getsizeof(d)
        if new_size != prev_size:
            print(f"After {i} items: {new_size} bytes (was {prev_size})")
            prev_size = new_size

dict_resizing()
```

**Açıklama**: Python listeleri dinamik array'dir ve büyüdükçe resize olur. `array` modülü C array'leri sunar (daha compact). Dict'ler hash table'dır ve load factor'e göre resize olur.

---

## Memory Optimization Patterns

Production ortamında kullanılabilecek bellek optimizasyonu pattern'leri ve best practice'ler.

### Örnek 20: Generator vs List (Lazy Evaluation)

```python
import sys
import time

def list_approach(n):
    """Liste döndüren yaklaşım"""
    return [i ** 2 for i in range(n)]

def generator_approach(n):
    """Generator döndüren yaklaşım"""
    return (i ** 2 for i in range(n))

def compare_memory_and_performance(n=1_000_000):
    """Generator ve liste yaklaşımlarını karşılaştırır"""

    # Memory comparison
    print(f"=== Memory Comparison (n={n:,}) ===")
    lst = list_approach(n)
    gen = generator_approach(n)

    print(f"List size: {sys.getsizeof(lst) / 1024 / 1024:.2f} MB")
    print(f"Generator size: {sys.getsizeof(gen)} bytes")

    # Performance comparison
    print(f"\n=== Performance Comparison ===")

    # List creation time
    start = time.perf_counter()
    lst = list_approach(n)
    list_time = time.perf_counter() - start
    print(f"List creation: {list_time:.4f}s")

    # Generator creation time (instant)
    start = time.perf_counter()
    gen = generator_approach(n)
    gen_time = time.perf_counter() - start
    print(f"Generator creation: {gen_time:.6f}s")

    # Iteration time
    start = time.perf_counter()
    for _ in lst:
        pass
    list_iter_time = time.perf_counter() - start
    print(f"List iteration: {list_iter_time:.4f}s")

    start = time.perf_counter()
    for _ in gen:
        pass
    gen_iter_time = time.perf_counter() - start
    print(f"Generator iteration: {gen_iter_time:.4f}s")

    del lst, gen

compare_memory_and_performance()

# Iterator chaining
def iterator_chaining_example():
    """Iterator chain ile bellek verimliliği"""
    import itertools

    print("\n=== Iterator Chaining ===")

    # Bad: Intermediate lists
    def bad_approach(n):
        data = list(range(n))
        filtered = [x for x in data if x % 2 == 0]
        squared = [x ** 2 for x in filtered]
        limited = squared[:10]
        return limited

    # Good: Generator chain
    def good_approach(n):
        data = range(n)
        filtered = (x for x in data if x % 2 == 0)
        squared = (x ** 2 for x in filtered)
        limited = itertools.islice(squared, 10)
        return list(limited)

    n = 1_000_000

    # Bad approach memory
    import tracemalloc
    tracemalloc.start()
    snapshot1 = tracemalloc.take_snapshot()
    result1 = bad_approach(n)
    snapshot2 = tracemalloc.take_snapshot()
    stats = snapshot2.compare_to(snapshot1, 'lineno')
    bad_memory = sum(s.size_diff for s in stats)

    tracemalloc.stop()
    tracemalloc.start()

    # Good approach memory
    snapshot3 = tracemalloc.take_snapshot()
    result2 = good_approach(n)
    snapshot4 = tracemalloc.take_snapshot()
    stats = snapshot4.compare_to(snapshot3, 'lineno')
    good_memory = sum(s.size_diff for s in stats)

    tracemalloc.stop()

    print(f"Bad approach: {bad_memory / 1024:.2f} KB")
    print(f"Good approach: {good_memory / 1024:.2f} KB")
    print(f"Tasarruf: {(bad_memory - good_memory) / 1024:.2f} KB")

iterator_chaining_example()
```

**Açıklama**: Generator'lar lazy evaluation yapar - değerleri ihtiyaç duyuldukça üretir. Büyük veri setleri için çok daha az bellek kullanır. Iterator chaining ile pipeline oluşturabilirsiniz.

Bu advanced memory management dökümanı, Python'da bellek yönetiminin tüm önemli yönlerini kapsamaktadır. Production ortamında bellek sızıntılarını tespit etme, optimizasyon yapma ve profiling konularında uzmanlaşmanızı sağlar.
