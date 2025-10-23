# Advanced Data Structures (İleri Seviye Veri Yapıları)

## İçindekiler
1. [Collections Modülü](#collections-modülü)
2. [Heapq - Heap Queue Algorithm](#heapq---heap-queue-algorithm)
3. [Bisect - Array Bisection Algorithm](#bisect---array-bisection-algorithm)
4. [Array Module](#array-module)
5. [Custom Data Structures](#custom-data-structures)
6. [Trie (Prefix Tree)](#trie-prefix-tree)
7. [Graph Implementations](#graph-implementations)
8. [LRU Cache Implementation](#lru-cache-implementation)
9. [Performance Comparisons](#performance-comparisons)

---

## Collections Modülü

Python'un `collections` modülü, standart veri yapılarının gelişmiş versiyonlarını sunar. Bu veri yapıları, özel kullanım durumları için optimize edilmiştir.

### 1. Deque (Double-Ended Queue)

**Deque**, her iki uçtan da O(1) kompleksitesinde ekleme ve çıkarma işlemlerine izin veren bir veri yapısıdır.

```python
from collections import deque
import time

# Deque oluşturma ve temel işlemler
dq = deque([1, 2, 3, 4, 5])

# Sağdan ve soldan ekleme - O(1)
dq.append(6)           # Sağa ekle
dq.appendleft(0)       # Sola ekle
print(f"Deque: {dq}")  # deque([0, 1, 2, 3, 4, 5, 6])

# Sağdan ve soldan çıkarma - O(1)
dq.pop()               # Sağdan çıkar
dq.popleft()          # Soldan çıkar
print(f"After pops: {dq}")

# Rotate işlemi - O(k) where k is the number of rotations
dq.rotate(2)          # Sağa kaydır
print(f"After rotate(2): {dq}")

dq.rotate(-1)         # Sola kaydır
print(f"After rotate(-1): {dq}")

# Performance comparison: deque vs list
def performance_test():
    # List ile sol tarafa ekleme
    start = time.time()
    lst = []
    for i in range(100000):
        lst.insert(0, i)  # O(n) - her seferinde kaydırma yapar
    list_time = time.time() - start

    # Deque ile sol tarafa ekleme
    start = time.time()
    dq = deque()
    for i in range(100000):
        dq.appendleft(i)  # O(1) - hızlı
    deque_time = time.time() - start

    print(f"\nList left insertion: {list_time:.4f}s")
    print(f"Deque left insertion: {deque_time:.4f}s")
    print(f"Deque is {list_time/deque_time:.2f}x faster!")

performance_test()
```

**Complexity Analysis:**
- append/appendleft: O(1)
- pop/popleft: O(1)
- Access by index: O(n)
- rotate: O(k)
- extend/extendleft: O(k)

### 2. Counter - Elemanları Sayma

**Counter**, hashable nesneleri saymak için özel bir dictionary alt sınıfıdır.

```python
from collections import Counter

# Temel kullanım
text = "python programming is awesome and python is powerful"
word_count = Counter(text.split())
print(f"Word frequencies: {word_count}")

# En yaygın elemanlar
print(f"Most common 3: {word_count.most_common(3)}")

# Matematiksel işlemler
c1 = Counter(['a', 'b', 'c', 'a', 'b', 'b'])
c2 = Counter(['a', 'b', 'd', 'd'])

print(f"\nUnion (|): {c1 | c2}")        # Maximum of each
print(f"Intersection (&): {c1 & c2}")   # Minimum of each
print(f"Addition (+): {c1 + c2}")        # Sum counts
print(f"Subtraction (-): {c1 - c2}")     # Subtract (keep positive)

# Real-world örnek: Anagram detection
def are_anagrams(str1, str2):
    """İki kelimenin anagram olup olmadığını kontrol eder - O(n)"""
    return Counter(str1.lower()) == Counter(str2.lower())

print(f"\nAre 'listen' and 'silent' anagrams? {are_anagrams('listen', 'silent')}")

# Find missing elements
def find_missing_elements(list1, list2):
    """list1'de olup list2'de olmayan elemanları bulur"""
    return list((Counter(list1) - Counter(list2)).elements())

print(f"Missing: {find_missing_elements([1,2,3,4,5], [2,4,5])}")
```

**Complexity Analysis:**
- Counting: O(n)
- most_common(): O(n log k) where k is the number of elements returned
- Mathematical operations: O(n)

### 3. defaultdict - Varsayılan Değerli Dictionary

**defaultdict**, eksik anahtarlar için otomatik olarak varsayılan değer üreten bir dictionary türüdür.

```python
from collections import defaultdict

# Liste ile defaultdict
graph = defaultdict(list)
graph['A'].append('B')
graph['A'].append('C')
graph['B'].append('D')
print(f"Graph: {dict(graph)}")

# int ile defaultdict - sayaç olarak
word_freq = defaultdict(int)
for word in ['apple', 'banana', 'apple', 'cherry', 'banana', 'apple']:
    word_freq[word] += 1
print(f"\nWord frequencies: {dict(word_freq)}")

# set ile defaultdict
groups = defaultdict(set)
students = [('Math', 'John'), ('Math', 'Alice'), ('Physics', 'John')]
for subject, student in students:
    groups[subject].add(student)
print(f"\nStudent groups: {dict(groups)}")

# Nested defaultdict - Tree structure
def tree():
    """Sınırsız derinlikte nested dictionary oluşturur"""
    return defaultdict(tree)

users = tree()
users['john']['age'] = 30
users['john']['address']['city'] = 'New York'
users['john']['address']['zip'] = '10001'
print(f"\nNested structure: {users}")

# Real-world örnek: Grouping items
def group_by_length(words):
    """Kelimeleri uzunluklarına göre gruplar - O(n)"""
    grouped = defaultdict(list)
    for word in words:
        grouped[len(word)].append(word)
    return dict(grouped)

words = ['cat', 'dog', 'elephant', 'ant', 'bear', 'lion']
print(f"\nGrouped by length: {group_by_length(words)}")
```

**Complexity Analysis:**
- Access/Insert: O(1) average case
- Memory: O(n) where n is number of keys

### 4. ChainMap - Birden Fazla Dictionary'yi Birleştirme

**ChainMap**, birden fazla dictionary'yi tek bir view'da birleştirir (kopyalamadan).

```python
from collections import ChainMap

# Temel kullanım - scope chain simulation
defaults = {'theme': 'light', 'language': 'en', 'notifications': True}
user_settings = {'theme': 'dark', 'language': 'tr'}
session_settings = {'notifications': False}

# İlk bulduğu değeri döner (soldan sağa)
settings = ChainMap(session_settings, user_settings, defaults)
print(f"Theme: {settings['theme']}")              # 'dark' from user_settings
print(f"Language: {settings['language']}")        # 'tr' from user_settings
print(f"Notifications: {settings['notifications']}")  # False from session_settings

# Yeni map ekleme
cli_settings = {'theme': 'blue'}
settings = settings.new_child(cli_settings)
print(f"\nWith CLI settings - Theme: {settings['theme']}")  # 'blue'

# Real-world örnek: Configuration hierarchy
class ConfigManager:
    """Hierarchical configuration management"""

    def __init__(self):
        self.system_defaults = {
            'debug': False,
            'log_level': 'INFO',
            'max_connections': 100
        }
        self.user_config = {}
        self.runtime_config = {}

    def get_config(self):
        """Priority: runtime > user > defaults"""
        return ChainMap(
            self.runtime_config,
            self.user_config,
            self.system_defaults
        )

    def set_user_config(self, **kwargs):
        self.user_config.update(kwargs)

    def set_runtime_config(self, **kwargs):
        self.runtime_config.update(kwargs)

config_manager = ConfigManager()
config_manager.set_user_config(debug=True, log_level='DEBUG')
config_manager.set_runtime_config(max_connections=200)

config = config_manager.get_config()
print(f"\nConfig - Debug: {config['debug']}, Log Level: {config['log_level']}")
print(f"Max Connections: {config['max_connections']}")
```

**Complexity Analysis:**
- Lookup: O(m) where m is number of maps
- Insert/Update: O(1) (affects only first map)
- Memory: O(1) (no copying)

### 5. OrderedDict - Sıralı Dictionary

**OrderedDict**, ekleme sırasını koruyan dictionary implementasyonudur. Python 3.7+ normal dict'ler de sırayı korur, ancak OrderedDict ek özellikler sunar.

```python
from collections import OrderedDict

# Temel kullanım
od = OrderedDict()
od['banana'] = 3
od['apple'] = 4
od['pear'] = 1
od['orange'] = 2

print(f"OrderedDict: {od}")

# move_to_end() - LRU cache için kritik
od.move_to_end('apple')  # En sona taşı
print(f"After moving 'apple' to end: {od}")

od.move_to_end('pear', last=False)  # En başa taşı
print(f"After moving 'pear' to start: {od}")

# popitem() - LIFO veya FIFO
last_item = od.popitem(last=True)   # Son elemanı çıkar (LIFO)
print(f"Popped last: {last_item}")

first_item = od.popitem(last=False) # İlk elemanı çıkar (FIFO)
print(f"Popped first: {first_item}")

# Real-world örnek: LRU Cache basit implementasyon
class SimpleLRUCache:
    """Basit LRU Cache implementasyonu - O(1) operations"""

    def __init__(self, capacity: int):
        self.cache = OrderedDict()
        self.capacity = capacity

    def get(self, key):
        """Değeri getir ve en sona taşı - O(1)"""
        if key not in self.cache:
            return -1
        self.cache.move_to_end(key)  # Recently used
        return self.cache[key]

    def put(self, key, value):
        """Değer ekle, kapasite doluysa en eskiyi çıkar - O(1)"""
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)  # Remove oldest

    def __repr__(self):
        return f"LRUCache({list(self.cache.items())})"

cache = SimpleLRUCache(3)
cache.put('a', 1)
cache.put('b', 2)
cache.put('c', 3)
print(f"\nCache after 3 puts: {cache}")

cache.get('a')  # 'a' becomes most recent
cache.put('d', 4)  # 'b' should be evicted
print(f"Cache after get('a') and put('d'): {cache}")
```

**Complexity Analysis:**
- Access: O(1)
- Insert: O(1)
- move_to_end: O(1)
- popitem: O(1)

---

## Heapq - Heap Queue Algorithm

**Heapq**, priority queue implementasyonu için kullanılan min-heap algoritmasıdır.

```python
import heapq

# Min heap oluşturma
heap = [5, 3, 7, 1, 9, 4, 6]
heapq.heapify(heap)  # O(n) - in-place heap oluşturur
print(f"Min heap: {heap}")  # [1, 3, 4, 5, 9, 7, 6]

# En küçük elemanı al - O(log n)
smallest = heapq.heappop(heap)
print(f"Popped smallest: {smallest}, Heap: {heap}")

# Eleman ekle - O(log n)
heapq.heappush(heap, 2)
print(f"After push(2): {heap}")

# Push ve pop birlikte - O(log n)
heapq.heappushpop(heap, 8)  # Push 8, then pop smallest
print(f"After heappushpop(8): {heap}")

# En küçük n elemanı bul - O(n log k)
numbers = [23, 1, 45, 12, 34, 56, 7, 89, 15]
smallest_3 = heapq.nsmallest(3, numbers)
largest_3 = heapq.nlargest(3, numbers)
print(f"\nSmallest 3: {smallest_3}")
print(f"Largest 3: {largest_3}")

# Max heap simulation (negatif değerlerle)
max_heap = []
for num in [3, 1, 4, 1, 5, 9, 2, 6]:
    heapq.heappush(max_heap, -num)  # Negatif olarak ekle

print(f"\nMax heap (negated): {max_heap}")
max_value = -heapq.heappop(max_heap)  # Negatifi geri çevir
print(f"Max value: {max_value}")

# Real-world örnek: Task scheduler with priority
class Task:
    def __init__(self, name, priority, deadline):
        self.name = name
        self.priority = priority
        self.deadline = deadline

    def __lt__(self, other):
        # Önce priority, sonra deadline'a göre sırala
        if self.priority != other.priority:
            return self.priority < other.priority
        return self.deadline < other.deadline

    def __repr__(self):
        return f"Task({self.name}, p={self.priority}, d={self.deadline})"

# Task scheduler
task_queue = []
heapq.heappush(task_queue, Task("Debug bug", 1, 10))
heapq.heappush(task_queue, Task("Write docs", 3, 15))
heapq.heappush(task_queue, Task("Fix security issue", 1, 5))
heapq.heappush(task_queue, Task("Code review", 2, 12))

print("\nTask execution order:")
while task_queue:
    task = heapq.heappop(task_queue)
    print(f"  Execute: {task}")

# Real-world örnek: K-way merge (birden fazla sorted list'i merge etme)
def merge_k_sorted_lists(lists):
    """K sorted list'i merge eder - O(n log k)"""
    heap = []
    result = []

    # Her listenin ilk elemanını heap'e ekle
    for i, lst in enumerate(lists):
        if lst:
            heapq.heappush(heap, (lst[0], i, 0))  # (value, list_idx, element_idx)

    # Heap'ten en küçüğü al ve sonraki elemanı ekle
    while heap:
        value, list_idx, element_idx = heapq.heappop(heap)
        result.append(value)

        # Bu listenin sonraki elemanı varsa ekle
        if element_idx + 1 < len(lists[list_idx]):
            next_value = lists[list_idx][element_idx + 1]
            heapq.heappush(heap, (next_value, list_idx, element_idx + 1))

    return result

sorted_lists = [[1, 4, 7], [2, 5, 8], [3, 6, 9]]
merged = merge_k_sorted_lists(sorted_lists)
print(f"\nMerged k sorted lists: {merged}")

# Median tracker - running median
class MedianFinder:
    """Stream'den gelen sayıların median'ını O(log n) ile bulur"""

    def __init__(self):
        self.small = []  # Max heap (lower half) - negated
        self.large = []  # Min heap (upper half)

    def add_num(self, num):
        """Sayı ekle - O(log n)"""
        # Her zaman önce small'a ekle
        heapq.heappush(self.small, -num)

        # small'ın max'ı large'ın min'inden büyükse taşı
        if self.small and self.large and (-self.small[0] > self.large[0]):
            val = -heapq.heappop(self.small)
            heapq.heappush(self.large, val)

        # Balance heap sizes (small <= large + 1)
        if len(self.small) > len(self.large) + 1:
            val = -heapq.heappop(self.small)
            heapq.heappush(self.large, val)
        if len(self.large) > len(self.small):
            val = heapq.heappop(self.large)
            heapq.heappush(self.small, -val)

    def find_median(self):
        """Median bul - O(1)"""
        if len(self.small) > len(self.large):
            return -self.small[0]
        return (-self.small[0] + self.large[0]) / 2.0

mf = MedianFinder()
for num in [1, 2, 3, 4, 5]:
    mf.add_num(num)
    print(f"Added {num}, Median: {mf.find_median()}")
```

**Complexity Analysis:**
- heapify: O(n)
- heappush: O(log n)
- heappop: O(log n)
- nsmallest/nlargest: O(n log k)
- heap[0] (peek): O(1)

---

## Bisect - Array Bisection Algorithm

**Bisect**, sorted list'lerde binary search ve insertion işlemleri için kullanılır.

```python
import bisect

# Sorted list
numbers = [1, 3, 5, 7, 9, 11, 13, 15]

# Insertion point bulma - O(log n)
pos = bisect.bisect_left(numbers, 6)   # 6'nın gitmesi gereken index
print(f"Position for 6: {pos}")  # 3

pos_right = bisect.bisect_right(numbers, 7)  # 7'den sonraki index
print(f"Position after 7: {pos_right}")  # 4

# Eleman ekleme (sorted'ı korur) - O(n) çünkü liste kaydırması gerekir
bisect.insort_left(numbers, 6)
print(f"After insort(6): {numbers}")

# Real-world örnek: Grade classification
def get_grade(score):
    """Puana göre not hesaplar - O(log n)"""
    breakpoints = [60, 70, 80, 90]
    grades = ['F', 'D', 'C', 'B', 'A']
    index = bisect.bisect(breakpoints, score)
    return grades[index]

scores = [55, 65, 75, 85, 95]
for score in scores:
    print(f"Score {score}: Grade {get_grade(score)}")

# Real-world örnek: Time-series data search
from datetime import datetime, timedelta

class TimeSeries:
    """Time-series data with efficient search - O(log n)"""

    def __init__(self):
        self.timestamps = []
        self.values = []

    def add(self, timestamp, value):
        """Yeni data point ekle - O(n)"""
        pos = bisect.bisect_left(self.timestamps, timestamp)
        self.timestamps.insert(pos, timestamp)
        self.values.insert(pos, value)

    def get_range(self, start_time, end_time):
        """Zaman aralığındaki değerleri getir - O(log n + k)"""
        start_idx = bisect.bisect_left(self.timestamps, start_time)
        end_idx = bisect.bisect_right(self.timestamps, end_time)
        return list(zip(self.timestamps[start_idx:end_idx],
                       self.values[start_idx:end_idx]))

    def get_closest(self, timestamp):
        """En yakın timestamp'i bul - O(log n)"""
        pos = bisect.bisect_left(self.timestamps, timestamp)
        if pos == 0:
            return self.timestamps[0], self.values[0]
        if pos == len(self.timestamps):
            return self.timestamps[-1], self.values[-1]

        # İki adaydan en yakınını seç
        before = self.timestamps[pos - 1]
        after = self.timestamps[pos]
        if timestamp - before < after - timestamp:
            return before, self.values[pos - 1]
        return after, self.values[pos]

# Test time series
ts = TimeSeries()
base_time = datetime(2024, 1, 1, 12, 0, 0)
for i in range(10):
    ts.add(base_time + timedelta(minutes=i*10), i * 100)

# Aralık sorgusu
start = base_time + timedelta(minutes=15)
end = base_time + timedelta(minutes=45)
print(f"\nData in range: {ts.get_range(start, end)}")

# En yakın değer
query_time = base_time + timedelta(minutes=23)
closest = ts.get_closest(query_time)
print(f"Closest to query: {closest}")

# Real-world örnek: Percentile calculation
def calculate_percentile(data, percentile):
    """Percentile hesaplar (sorted data için) - O(1)"""
    if not data:
        return None
    index = int(len(data) * percentile / 100)
    return data[min(index, len(data) - 1)]

sorted_data = sorted([23, 45, 12, 67, 89, 34, 56, 78, 90, 11])
print(f"\n50th percentile (median): {calculate_percentile(sorted_data, 50)}")
print(f"90th percentile: {calculate_percentile(sorted_data, 90)}")
print(f"95th percentile: {calculate_percentile(sorted_data, 95)}")
```

**Complexity Analysis:**
- bisect_left/right: O(log n)
- insort_left/right: O(n) due to list shifting
- Best use case: Frequently searched, infrequently modified lists

---

## Array Module

**Array**, tip-homojen veri için bellek-efektif depolama sağlar. List'lerden daha az bellek kullanır.

```python
import array
import sys

# Array oluşturma (type codes: 'i'=int, 'f'=float, 'd'=double, etc.)
int_array = array.array('i', [1, 2, 3, 4, 5])
float_array = array.array('d', [1.1, 2.2, 3.3, 4.4, 5.5])

print(f"Int array: {int_array}")
print(f"Float array: {float_array}")

# Memory comparison
list_obj = [1, 2, 3, 4, 5] * 1000
array_obj = array.array('i', [1, 2, 3, 4, 5] * 1000)

print(f"\nList size: {sys.getsizeof(list_obj)} bytes")
print(f"Array size: {sys.getsizeof(array_obj)} bytes")
print(f"Memory saved: {sys.getsizeof(list_obj) - sys.getsizeof(array_obj)} bytes")

# Array operations
int_array.append(6)
int_array.extend([7, 8, 9])
print(f"\nAfter append/extend: {int_array}")

# Slice operations
print(f"Slice [2:5]: {int_array[2:5]}")

# Buffer operations - C-compatible
buffer_view = memoryview(int_array)
print(f"Memory view: {buffer_view.tolist()}")

# Real-world örnek: Pixel data storage
class ImageBuffer:
    """Memory-efficient image pixel storage"""

    def __init__(self, width, height):
        self.width = width
        self.height = height
        # RGB values (0-255) için unsigned char array
        self.pixels = array.array('B', [0] * (width * height * 3))

    def set_pixel(self, x, y, r, g, b):
        """Pixel rengini ayarla - O(1)"""
        idx = (y * self.width + x) * 3
        self.pixels[idx] = r
        self.pixels[idx + 1] = g
        self.pixels[idx + 2] = b

    def get_pixel(self, x, y):
        """Pixel rengini getir - O(1)"""
        idx = (y * self.width + x) * 3
        return tuple(self.pixels[idx:idx+3])

    def fill(self, r, g, b):
        """Tüm image'ı bir renkle doldur - O(n)"""
        for i in range(0, len(self.pixels), 3):
            self.pixels[i] = r
            self.pixels[i + 1] = g
            self.pixels[i + 2] = b

# Test image buffer
img = ImageBuffer(100, 100)
img.set_pixel(50, 50, 255, 0, 0)  # Red pixel at center
print(f"\nPixel at (50,50): RGB{img.get_pixel(50, 50)}")
print(f"Image buffer size: {sys.getsizeof(img.pixels)} bytes")
```

**Complexity Analysis:**
- Access: O(1)
- Append: O(1) amortized
- Insert: O(n)
- Memory: ~1/4 of list for integers
- Best use case: Large homogeneous numeric datasets

---

## Custom Data Structures

### Linked List Implementation

```python
class Node:
    """Doubly linked list node"""
    def __init__(self, data):
        self.data = data
        self.next = None
        self.prev = None

class DoublyLinkedList:
    """Doubly linked list with O(1) head/tail operations"""

    def __init__(self):
        self.head = None
        self.tail = None
        self.size = 0

    def append(self, data):
        """Sona ekle - O(1)"""
        new_node = Node(data)
        if not self.head:
            self.head = self.tail = new_node
        else:
            new_node.prev = self.tail
            self.tail.next = new_node
            self.tail = new_node
        self.size += 1

    def prepend(self, data):
        """Başa ekle - O(1)"""
        new_node = Node(data)
        if not self.head:
            self.head = self.tail = new_node
        else:
            new_node.next = self.head
            self.head.prev = new_node
            self.head = new_node
        self.size += 1

    def delete(self, data):
        """İlk bulunan data'yı sil - O(n)"""
        current = self.head
        while current:
            if current.data == data:
                if current.prev:
                    current.prev.next = current.next
                else:
                    self.head = current.next

                if current.next:
                    current.next.prev = current.prev
                else:
                    self.tail = current.prev

                self.size -= 1
                return True
            current = current.next
        return False

    def find(self, data):
        """Data ara - O(n)"""
        current = self.head
        while current:
            if current.data == data:
                return True
            current = current.next
        return False

    def __iter__(self):
        current = self.head
        while current:
            yield current.data
            current = current.next

    def __len__(self):
        return self.size

    def __repr__(self):
        return ' <-> '.join(str(data) for data in self)

# Test linked list
dll = DoublyLinkedList()
dll.append(1)
dll.append(2)
dll.append(3)
dll.prepend(0)
print(f"Linked list: {dll}")
print(f"Find 2: {dll.find(2)}")
dll.delete(2)
print(f"After delete(2): {dll}")
```

---

## Trie (Prefix Tree)

**Trie**, string arama ve prefix matching için optimize edilmiş tree veri yapısıdır.

```python
class TrieNode:
    """Trie node with children dictionary"""
    def __init__(self):
        self.children = {}
        self.is_end_of_word = False
        self.word_count = 0  # Bu node'dan geçen kelime sayısı

class Trie:
    """Trie (Prefix Tree) implementation for efficient string operations"""

    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        """Kelime ekle - O(m) where m is word length"""
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
            node.word_count += 1
        node.is_end_of_word = True

    def search(self, word):
        """Kelime ara (exact match) - O(m)"""
        node = self._find_node(word)
        return node is not None and node.is_end_of_word

    def starts_with(self, prefix):
        """Prefix ile başlayan kelime var mı - O(m)"""
        return self._find_node(prefix) is not None

    def _find_node(self, prefix):
        """Prefix'e karşılık gelen node'u bul - O(m)"""
        node = self.root
        for char in prefix:
            if char not in node.children:
                return None
            node = node.children[char]
        return node

    def get_all_words_with_prefix(self, prefix):
        """Prefix ile başlayan tüm kelimeleri getir - O(n)"""
        node = self._find_node(prefix)
        if not node:
            return []

        words = []
        self._dfs(node, prefix, words)
        return words

    def _dfs(self, node, prefix, words):
        """DFS ile tüm kelimeleri topla"""
        if node.is_end_of_word:
            words.append(prefix)

        for char, child_node in node.children.items():
            self._dfs(child_node, prefix + char, words)

    def count_words_with_prefix(self, prefix):
        """Prefix ile başlayan kelime sayısı - O(m)"""
        node = self._find_node(prefix)
        return node.word_count if node else 0

    def delete(self, word):
        """Kelime sil - O(m)"""
        def _delete_helper(node, word, index):
            if index == len(word):
                if not node.is_end_of_word:
                    return False
                node.is_end_of_word = False
                return len(node.children) == 0

            char = word[index]
            if char not in node.children:
                return False

            should_delete_child = _delete_helper(node.children[char], word, index + 1)

            if should_delete_child:
                del node.children[char]
                return len(node.children) == 0 and not node.is_end_of_word

            return False

        _delete_helper(self.root, word, 0)

    def autocomplete(self, prefix, max_results=5):
        """Autocomplete suggestions - O(p + n)"""
        words = self.get_all_words_with_prefix(prefix)
        return words[:max_results]

# Test Trie
trie = Trie()
words = ["apple", "app", "application", "apply", "banana", "band", "bandana"]
for word in words:
    trie.insert(word)

print("Trie Operations:")
print(f"Search 'app': {trie.search('app')}")
print(f"Search 'appl': {trie.search('appl')}")
print(f"Starts with 'app': {trie.starts_with('app')}")
print(f"Words with prefix 'app': {trie.get_all_words_with_prefix('app')}")
print(f"Count with prefix 'ban': {trie.count_words_with_prefix('ban')}")
print(f"Autocomplete 'ba': {trie.autocomplete('ba', 3)}")

# Real-world örnek: Search engine autocomplete
class SearchEngine:
    """Search engine with autocomplete and ranking"""

    def __init__(self):
        self.trie = Trie()
        self.search_counts = {}  # Arama sayılarını takip et

    def index_query(self, query):
        """Search query'yi indexle"""
        query_lower = query.lower()
        self.trie.insert(query_lower)
        self.search_counts[query_lower] = self.search_counts.get(query_lower, 0) + 1

    def suggest(self, prefix, max_results=5):
        """Autocomplete suggestions (popularity'ye göre sıralı)"""
        prefix_lower = prefix.lower()
        suggestions = self.trie.get_all_words_with_prefix(prefix_lower)

        # Popularity'ye göre sırala
        suggestions.sort(key=lambda x: self.search_counts.get(x, 0), reverse=True)
        return suggestions[:max_results]

search_engine = SearchEngine()
queries = ["python tutorial", "python advanced", "python basics",
           "python tutorial", "python tips", "python tutorial"]
for query in queries:
    search_engine.index_query(query)

print(f"\nSearch suggestions for 'python': {search_engine.suggest('python', 3)}")
```

**Complexity Analysis:**
- Insert: O(m) where m is word length
- Search: O(m)
- Prefix search: O(m + n) where n is number of matching words
- Space: O(ALPHABET_SIZE * N * M) worst case
- Best use case: Autocomplete, spell checker, IP routing

---

## Graph Implementations

```python
from collections import defaultdict, deque

class Graph:
    """Graph implementation with adjacency list"""

    def __init__(self, directed=False):
        self.graph = defaultdict(list)
        self.directed = directed

    def add_edge(self, u, v, weight=1):
        """Edge ekle - O(1)"""
        self.graph[u].append((v, weight))
        if not self.directed:
            self.graph[v].append((u, weight))

    def bfs(self, start):
        """Breadth-First Search - O(V + E)"""
        visited = set()
        queue = deque([start])
        visited.add(start)
        result = []

        while queue:
            vertex = queue.popleft()
            result.append(vertex)

            for neighbor, _ in self.graph[vertex]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

        return result

    def dfs(self, start):
        """Depth-First Search - O(V + E)"""
        visited = set()
        result = []

        def dfs_helper(vertex):
            visited.add(vertex)
            result.append(vertex)

            for neighbor, _ in self.graph[vertex]:
                if neighbor not in visited:
                    dfs_helper(neighbor)

        dfs_helper(start)
        return result

    def shortest_path(self, start, end):
        """BFS ile en kısa path bulma (unweighted) - O(V + E)"""
        if start == end:
            return [start]

        visited = {start}
        queue = deque([(start, [start])])

        while queue:
            vertex, path = queue.popleft()

            for neighbor, _ in self.graph[vertex]:
                if neighbor == end:
                    return path + [neighbor]

                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))

        return None  # Path bulunamadı

    def dijkstra(self, start):
        """Dijkstra algoritması (weighted shortest path) - O((V + E) log V)"""
        import heapq

        distances = {vertex: float('inf') for vertex in self.graph}
        distances[start] = 0
        pq = [(0, start)]  # (distance, vertex)
        visited = set()

        while pq:
            current_dist, current = heapq.heappop(pq)

            if current in visited:
                continue

            visited.add(current)

            for neighbor, weight in self.graph[current]:
                distance = current_dist + weight

                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    heapq.heappush(pq, (distance, neighbor))

        return distances

    def has_cycle(self):
        """Cycle detection (undirected graph) - O(V + E)"""
        visited = set()

        def dfs_cycle(vertex, parent):
            visited.add(vertex)

            for neighbor, _ in self.graph[vertex]:
                if neighbor not in visited:
                    if dfs_cycle(neighbor, vertex):
                        return True
                elif neighbor != parent:
                    return True

            return False

        for vertex in self.graph:
            if vertex not in visited:
                if dfs_cycle(vertex, None):
                    return True

        return False

    def topological_sort(self):
        """Topological sort (DAG için) - O(V + E)"""
        if not self.directed:
            raise ValueError("Topological sort only for directed graphs")

        in_degree = defaultdict(int)
        for vertex in self.graph:
            for neighbor, _ in self.graph[vertex]:
                in_degree[neighbor] += 1

        queue = deque([v for v in self.graph if in_degree[v] == 0])
        result = []

        while queue:
            vertex = queue.popleft()
            result.append(vertex)

            for neighbor, _ in self.graph[vertex]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if len(result) != len(self.graph):
            raise ValueError("Graph has a cycle")

        return result

# Test Graph
g = Graph()
g.add_edge('A', 'B')
g.add_edge('A', 'C')
g.add_edge('B', 'D')
g.add_edge('C', 'D')
g.add_edge('D', 'E')

print("\nGraph Traversals:")
print(f"BFS from A: {g.bfs('A')}")
print(f"DFS from A: {g.dfs('A')}")
print(f"Shortest path A to E: {g.shortest_path('A', 'E')}")

# Weighted graph
wg = Graph(directed=True)
wg.add_edge('A', 'B', 4)
wg.add_edge('A', 'C', 2)
wg.add_edge('B', 'D', 3)
wg.add_edge('C', 'D', 1)
wg.add_edge('D', 'E', 2)

print(f"\nDijkstra from A: {wg.dijkstra('A')}")
```

**Complexity Analysis:**
- Add edge: O(1)
- BFS/DFS: O(V + E)
- Dijkstra: O((V + E) log V) with min-heap
- Topological sort: O(V + E)
- Space: O(V + E)

---

## LRU Cache Implementation

```python
from collections import OrderedDict

class LRUCache:
    """
    LRU (Least Recently Used) Cache implementation
    O(1) get ve put operations
    """

    def __init__(self, capacity: int):
        self.cache = OrderedDict()
        self.capacity = capacity

    def get(self, key: int) -> int:
        """Değeri getir ve recently used olarak işaretle - O(1)"""
        if key not in self.cache:
            return -1

        # Move to end (most recently used)
        self.cache.move_to_end(key)
        return self.cache[key]

    def put(self, key: int, value: int) -> None:
        """Değer ekle/güncelle - O(1)"""
        if key in self.cache:
            # Mevcut key'i güncelle ve most recent yap
            self.cache.move_to_end(key)

        self.cache[key] = value

        # Kapasite aşıldıysa en az kullanılanı çıkar
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)  # Remove least recently used

    def __repr__(self):
        return f"LRUCache({dict(self.cache)})"

# Doubly Linked List + HashMap ile custom implementation
class Node:
    def __init__(self, key, value):
        self.key = key
        self.value = value
        self.prev = None
        self.next = None

class LRUCacheCustom:
    """
    Custom LRU Cache with doubly linked list + hashmap
    More educational, shows underlying mechanism
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = {}  # key -> Node

        # Dummy head and tail for easier manipulation
        self.head = Node(0, 0)
        self.tail = Node(0, 0)
        self.head.next = self.tail
        self.tail.prev = self.head

    def _add_to_front(self, node):
        """Node'u head'in hemen arkasına ekle (most recent)"""
        node.prev = self.head
        node.next = self.head.next
        self.head.next.prev = node
        self.head.next = node

    def _remove_node(self, node):
        """Node'u linked list'ten çıkar"""
        prev_node = node.prev
        next_node = node.next
        prev_node.next = next_node
        next_node.prev = prev_node

    def get(self, key: int) -> int:
        """Değeri getir - O(1)"""
        if key not in self.cache:
            return -1

        node = self.cache[key]

        # Move to front (most recently used)
        self._remove_node(node)
        self._add_to_front(node)

        return node.value

    def put(self, key: int, value: int) -> None:
        """Değer ekle/güncelle - O(1)"""
        if key in self.cache:
            # Mevcut node'u güncelle
            node = self.cache[key]
            node.value = value
            self._remove_node(node)
            self._add_to_front(node)
        else:
            # Yeni node oluştur
            new_node = Node(key, value)
            self.cache[key] = new_node
            self._add_to_front(new_node)

            # Kapasite kontrolü
            if len(self.cache) > self.capacity:
                # Remove least recently used (tail'den önceki)
                lru = self.tail.prev
                self._remove_node(lru)
                del self.cache[lru.key]

    def __repr__(self):
        items = []
        current = self.head.next
        while current != self.tail:
            items.append(f"{current.key}:{current.value}")
            current = current.next
        return f"LRUCache([{', '.join(items)}])"

# Test LRU Cache
print("\nLRU Cache Test:")
cache = LRUCache(3)

cache.put(1, "A")
cache.put(2, "B")
cache.put(3, "C")
print(f"After 3 puts: {cache}")

cache.get(1)  # Make 1 most recent
print(f"After get(1): {cache}")

cache.put(4, "D")  # Should evict 2 (least recent)
print(f"After put(4, 'D'): {cache}")

print(f"Get(2): {cache.get(2)}")  # Should return -1

# Custom implementation test
print("\nCustom LRU Cache Test:")
custom_cache = LRUCacheCustom(3)
custom_cache.put(1, "A")
custom_cache.put(2, "B")
custom_cache.put(3, "C")
print(f"After 3 puts: {custom_cache}")

custom_cache.get(1)
print(f"After get(1): {custom_cache}")

custom_cache.put(4, "D")
print(f"After put(4, 'D'): {custom_cache}")
```

**Complexity Analysis:**
- get: O(1)
- put: O(1)
- Space: O(capacity)
- Best use case: Caching, memoization, database query cache

---

## Performance Comparisons

```python
import time
import sys
from collections import deque, defaultdict, Counter
import random

def time_function(func, *args, **kwargs):
    """Function execution time ölçer"""
    start = time.perf_counter()
    result = func(*args, **kwargs)
    end = time.perf_counter()
    return end - start, result

print("=" * 80)
print("PERFORMANCE COMPARISONS")
print("=" * 80)

# 1. List vs Deque - Left insertion
print("\n1. LEFT INSERTION: List vs Deque")
print("-" * 40)

n = 50000

def list_left_insert():
    lst = []
    for i in range(n):
        lst.insert(0, i)
    return lst

def deque_left_insert():
    dq = deque()
    for i in range(n):
        dq.appendleft(i)
    return dq

list_time, _ = time_function(list_left_insert)
deque_time, _ = time_function(deque_left_insert)

print(f"List:  {list_time:.4f}s - O(n) per insert")
print(f"Deque: {deque_time:.4f}s - O(1) per insert")
print(f"Speedup: {list_time/deque_time:.2f}x")

# 2. Dictionary vs defaultdict - Grouping
print("\n2. GROUPING OPERATIONS: dict vs defaultdict")
print("-" * 40)

data = [(random.randint(0, 100), i) for i in range(100000)]

def dict_grouping():
    groups = {}
    for key, value in data:
        if key not in groups:
            groups[key] = []
        groups[key].append(value)
    return groups

def defaultdict_grouping():
    groups = defaultdict(list)
    for key, value in data:
        groups[key].append(value)
    return groups

dict_time, _ = time_function(dict_grouping)
defaultdict_time, _ = time_function(defaultdict_grouping)

print(f"Dict:        {dict_time:.4f}s - Requires key checking")
print(f"Defaultdict: {defaultdict_time:.4f}s - No key checking")
print(f"Speedup: {dict_time/defaultdict_time:.2f}x")

# 3. Counting: Manual vs Counter
print("\n3. COUNTING: Manual vs Counter")
print("-" * 40)

words = ['apple', 'banana', 'apple', 'cherry'] * 10000

def manual_counting():
    counts = {}
    for word in words:
        counts[word] = counts.get(word, 0) + 1
    return counts

def counter_counting():
    return Counter(words)

manual_time, _ = time_function(manual_counting)
counter_time, _ = time_function(counter_counting)

print(f"Manual:  {manual_time:.4f}s")
print(f"Counter: {counter_time:.4f}s")
print(f"Speedup: {manual_time/counter_time:.2f}x")

# 4. Priority Queue: List vs Heapq
print("\n4. PRIORITY QUEUE: List sort vs Heapq")
print("-" * 40)

import heapq

tasks = [(random.randint(1, 100), f"task_{i}") for i in range(10000)]

def list_priority_queue():
    pq = []
    for priority, task in tasks:
        pq.append((priority, task))
        pq.sort()  # O(n log n) her seferinde!

    result = []
    while pq:
        result.append(pq.pop(0))
    return result

def heapq_priority_queue():
    pq = []
    for priority, task in tasks:
        heapq.heappush(pq, (priority, task))  # O(log n)

    result = []
    while pq:
        result.append(heapq.heappop(pq))  # O(log n)
    return result

list_pq_time, _ = time_function(list_priority_queue)
heapq_time, _ = time_function(heapq_priority_queue)

print(f"List sort: {list_pq_time:.4f}s - O(n²log n)")
print(f"Heapq:     {heapq_time:.4f}s - O(n log n)")
print(f"Speedup: {list_pq_time/heapq_time:.2f}x")

# 5. Search: Linear vs Binary (bisect)
print("\n5. SEARCH: Linear vs Binary Search (bisect)")
print("-" * 40)

import bisect

sorted_list = sorted(range(100000))
searches = [random.randint(0, 100000) for _ in range(1000)]

def linear_search():
    results = []
    for target in searches:
        results.append(target in sorted_list)
    return results

def binary_search():
    results = []
    for target in searches:
        idx = bisect.bisect_left(sorted_list, target)
        results.append(idx < len(sorted_list) and sorted_list[idx] == target)
    return results

linear_time, _ = time_function(linear_search)
binary_time, _ = time_function(binary_search)

print(f"Linear: {linear_time:.4f}s - O(n)")
print(f"Binary: {binary_time:.4f}s - O(log n)")
print(f"Speedup: {linear_time/binary_time:.2f}x")

# 6. Memory: List vs Array
print("\n6. MEMORY USAGE: List vs Array")
print("-" * 40)

import array

size = 100000

int_list = list(range(size))
int_array = array.array('i', range(size))

list_size = sys.getsizeof(int_list)
array_size = sys.getsizeof(int_array)

print(f"List:  {list_size:,} bytes")
print(f"Array: {array_size:,} bytes")
print(f"Memory saved: {list_size - array_size:,} bytes ({(1-array_size/list_size)*100:.1f}%)")

# 7. String Search: In operator vs Trie
print("\n7. STRING PREFIX SEARCH: List iteration vs Trie")
print("-" * 40)

words = ["apple", "application", "apply", "banana", "band"] * 1000

def list_prefix_search(prefix):
    return [word for word in words if word.startswith(prefix)]

# Trie from earlier
trie = Trie()
for word in words:
    trie.insert(word)

list_search_time, _ = time_function(list_prefix_search, "app")
trie_search_time, _ = time_function(trie.get_all_words_with_prefix, "app")

print(f"List iteration: {list_search_time:.6f}s - O(n*m)")
print(f"Trie search:    {trie_search_time:.6f}s - O(m+k)")
print(f"Speedup: {list_search_time/trie_search_time:.2f}x")

print("\n" + "=" * 80)
print("SUMMARY: Seçim Kriterleri")
print("=" * 80)
print("""
1. Deque: Her iki uçtan ekleme/çıkarma gerekiyorsa
2. Counter: Eleman sayma ve frequency operations
3. defaultdict: Gruplandırma ve nested structures
4. OrderedDict: Insertion order + move_to_end (LRU cache)
5. ChainMap: Layered configurations, scope simulation
6. heapq: Priority queue, top-k problems
7. bisect: Sorted list'lerde binary search
8. array: Homogeneous numeric data, memory efficiency
9. Trie: Prefix matching, autocomplete
10. Graph: Relationship modeling, shortest path
11. LRU Cache: Caching with automatic eviction
""")
```

---

## Özet: Data Structure Seçim Rehberi

| İhtiyaç | Data Structure | Complexity | Use Case |
|---------|---------------|------------|-----------|
| FIFO/LIFO işlemler | deque | O(1) | Queue, Stack |
| Eleman sayma | Counter | O(n) | Frequency analysis |
| Gruplandırma | defaultdict | O(1) | Grouping, nested dict |
| LRU Cache | OrderedDict | O(1) | Caching |
| Priority Queue | heapq | O(log n) | Task scheduling |
| Binary Search | bisect | O(log n) | Sorted data search |
| Memory efficiency | array | O(1) | Numeric data |
| Prefix search | Trie | O(m) | Autocomplete |
| Relationships | Graph | O(V+E) | Network analysis |

## Sonuç

Advanced data structures, kod performansını ve okunabilirliğini dramatik şekilde artırır. Doğru veri yapısını seçmek, O(n²) algoritmayı O(n log n) veya O(n)'e düşürebilir. Python'un collections modülü ve custom implementations, hemen hemen her problem için optimize çözüm sunar.

**Ana Prensipler:**
1. **Complexity analizi yapın** - Big O notation ile düşünün
2. **Built-in kullanın** - Python'un optimize edilmiş yapıları hızlıdır
3. **Trade-offs'u anlayın** - Memory vs Speed, Simplicity vs Performance
4. **Profile edin** - Varsayımlarınızı test edin, gerçek data ile ölçün
