# Generators ve Iterators (İleri Seviye)

## İçindekiler
1. [Iterator Protocol](#iterator-protocol)
2. [Generator Functions](#generator-functions)
3. [Generator Expressions](#generator-expressions)
4. [yield from Statement](#yield-from-statement)
5. [itertools Module](#itertools-module)
6. [Memory Efficiency](#memory-efficiency)
7. [Infinite Generators](#infinite-generators)
8. [Pipeline Patterns](#pipeline-patterns)
9. [Generator-based Coroutines](#generator-based-coroutines)

---

## Iterator Protocol

### Temel Kavramlar

**Iterator Protocol**, Python'da iteration (yineleme) mekanizmasının temelidir. Bir nesnenin iterable olması için `__iter__()` ve `__next__()` metodlarını implement etmesi gerekir.

```python
class Countdown:
    """Basit bir countdown iterator örneği"""
    def __init__(self, start):
        self.current = start

    def __iter__(self):
        return self

    def __next__(self):
        if self.current <= 0:
            raise StopIteration
        self.current -= 1
        return self.current + 1

# Kullanım
for num in Countdown(5):
    print(num)  # 5, 4, 3, 2, 1
```

### Örnek 1: Custom Range Iterator

```python
class CustomRange:
    """Gelişmiş range implementasyonu"""
    def __init__(self, start, end=None, step=1):
        if end is None:
            self.start = 0
            self.end = start
        else:
            self.start = start
            self.end = end
        self.step = step
        self.current = self.start

    def __iter__(self):
        self.current = self.start
        return self

    def __next__(self):
        if (self.step > 0 and self.current >= self.end) or \
           (self.step < 0 and self.current <= self.end):
            raise StopIteration

        value = self.current
        self.current += self.step
        return value

    def __len__(self):
        return max(0, (self.end - self.start + self.step - 1) // self.step)

    def __reversed__(self):
        return CustomRange(
            self.end - self.step,
            self.start - self.step,
            -self.step
        )

# Kullanım
print(list(CustomRange(10)))  # [0, 1, 2, ..., 9]
print(list(CustomRange(5, 10)))  # [5, 6, 7, 8, 9]
print(list(CustomRange(10, 0, -2)))  # [10, 8, 6, 4, 2]
```

### Örnek 2: Fibonacci Iterator

```python
class FibonacciIterator:
    """Fibonacci sayılarını üreten iterator"""
    def __init__(self, max_count=None):
        self.max_count = max_count
        self.count = 0
        self.a, self.b = 0, 1

    def __iter__(self):
        return self

    def __next__(self):
        if self.max_count is not None and self.count >= self.max_count:
            raise StopIteration

        self.count += 1
        result = self.a
        self.a, self.b = self.b, self.a + self.b
        return result

# Kullanım
fib = FibonacciIterator(10)
print(list(fib))  # [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]
```

### Örnek 3: Sentinel Iterator

```python
class SentinelIterator:
    """Belirli bir değer görülene kadar iterate eden iterator"""
    def __init__(self, callable_obj, sentinel):
        self.callable_obj = callable_obj
        self.sentinel = sentinel

    def __iter__(self):
        return self

    def __next__(self):
        value = self.callable_obj()
        if value == self.sentinel:
            raise StopIteration
        return value

# Kullanım: Dosyadan boş satır görülene kadar okuma
def read_lines_until_empty():
    with open('data.txt', 'r') as f:
        def read_line():
            return f.readline().strip()

        return SentinelIterator(read_line, '')
```

---

## Generator Functions

Generator'lar, `yield` anahtar kelimesi kullanarak değer üreten fonksiyonlardır. Her `yield` çağrısında durumu korunur ve bir sonraki çağrıda kaldığı yerden devam eder.

### Örnek 4: Basit Generator

```python
def simple_generator():
    """Basit bir generator fonksiyonu"""
    print("Start")
    yield 1
    print("Middle")
    yield 2
    print("End")
    yield 3

gen = simple_generator()
print(next(gen))  # Start, sonra 1
print(next(gen))  # Middle, sonra 2
print(next(gen))  # End, sonra 3
```

### Örnek 5: Fibonacci Generator

```python
def fibonacci(n):
    """İlk n Fibonacci sayısını üreten generator"""
    a, b = 0, 1
    count = 0
    while count < n:
        yield a
        a, b = b, a + b
        count += 1

# Kullanım
print(list(fibonacci(10)))  # [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]
```

### Örnek 6: File Reader Generator

```python
def read_large_file(file_path, chunk_size=1024):
    """Büyük dosyaları chunk chunk okuyan generator"""
    with open(file_path, 'r', encoding='utf-8') as file:
        while True:
            chunk = file.read(chunk_size)
            if not chunk:
                break
            yield chunk

# Kullanım
for chunk in read_large_file('large_file.txt'):
    process_chunk(chunk)  # Her chunk'ı işle
```

### Örnek 7: CSV Parser Generator

```python
def parse_csv(file_path, delimiter=',', skip_header=True):
    """CSV dosyasını satır satır parse eden generator"""
    with open(file_path, 'r', encoding='utf-8') as file:
        if skip_header:
            next(file)  # Header'ı atla

        for line in file:
            # Strip whitespace and split
            fields = [field.strip() for field in line.strip().split(delimiter)]
            yield fields

# Kullanım
for row in parse_csv('data.csv'):
    print(row)
```

### Örnek 8: Generator with send()

```python
def running_average():
    """send() metodu ile değer alan ve ortalama hesaplayan generator"""
    total = 0.0
    count = 0
    average = None

    while True:
        value = yield average
        if value is None:
            break
        total += value
        count += 1
        average = total / count

# Kullanım
avg_gen = running_average()
next(avg_gen)  # Generator'ı başlat
print(avg_gen.send(10))  # 10.0
print(avg_gen.send(20))  # 15.0
print(avg_gen.send(30))  # 20.0
```

---

## Generator Expressions

Generator expression'lar, list comprehension'ların generator versiyonudur. Parantez `()` kullanılır ve lazy evaluation yapar.

### Örnek 9: Basic Generator Expression

```python
# List comprehension vs Generator expression
list_comp = [x**2 for x in range(10)]  # Tüm değerleri hemen hesaplar
gen_exp = (x**2 for x in range(10))    # Lazy evaluation

print(type(list_comp))  # <class 'list'>
print(type(gen_exp))    # <class 'generator'>

# Memory kullanımı
import sys
print(sys.getsizeof(list_comp))  # ~200 bytes
print(sys.getsizeof(gen_exp))    # ~128 bytes
```

### Örnek 10: Chained Generator Expressions

```python
# Zincirleme generator expression'lar
numbers = range(1000000)
squares = (x**2 for x in numbers)
evens = (x for x in squares if x % 2 == 0)
sum_result = sum(evens)

# Tek satırda
result = sum(x**2 for x in range(1000000) if x**2 % 2 == 0)
```

### Örnek 11: Generator Expression in Functions

```python
def find_duplicates(items):
    """Tekrarlanan elemanları bulan generator"""
    seen = set()
    return (item for item in items
            if item in seen or seen.add(item) is None)

# Kullanım
data = [1, 2, 3, 2, 4, 3, 5]
duplicates = list(find_duplicates(data))
print(duplicates)  # [2, 3]
```

---

## yield from Statement

`yield from`, bir generator'ın başka bir generator'dan değer almasını sağlar. Delegation pattern için kullanılır.

### Örnek 12: Basic yield from

```python
def generator1():
    yield 1
    yield 2

def generator2():
    yield 3
    yield 4

def combined_generator():
    yield from generator1()
    yield from generator2()

print(list(combined_generator()))  # [1, 2, 3, 4]
```

### Örnek 13: Recursive Generator with yield from

```python
def flatten(nested_list):
    """İç içe listeleri düzleştiren recursive generator"""
    for item in nested_list:
        if isinstance(item, list):
            yield from flatten(item)
        else:
            yield item

# Kullanım
nested = [1, [2, 3, [4, 5]], 6, [7, [8, 9]]]
print(list(flatten(nested)))  # [1, 2, 3, 4, 5, 6, 7, 8, 9]
```

### Örnek 14: Tree Traversal with yield from

```python
class TreeNode:
    def __init__(self, value, left=None, right=None):
        self.value = value
        self.left = left
        self.right = right

    def inorder(self):
        """Inorder traversal generator"""
        if self.left:
            yield from self.left.inorder()
        yield self.value
        if self.right:
            yield from self.right.inorder()

# Binary tree oluştur
root = TreeNode(4,
    TreeNode(2, TreeNode(1), TreeNode(3)),
    TreeNode(6, TreeNode(5), TreeNode(7))
)

print(list(root.inorder()))  # [1, 2, 3, 4, 5, 6, 7]
```

---

## itertools Module

Python'un `itertools` modülü, verimli iteration için güçlü araçlar sunar.

### Örnek 15: itertools.chain

```python
import itertools

# Birden fazla iterable'ı birleştirme
list1 = [1, 2, 3]
list2 = [4, 5, 6]
list3 = [7, 8, 9]

chained = itertools.chain(list1, list2, list3)
print(list(chained))  # [1, 2, 3, 4, 5, 6, 7, 8, 9]

# chain.from_iterable - nested iterables için
nested = [[1, 2], [3, 4], [5, 6]]
flattened = itertools.chain.from_iterable(nested)
print(list(flattened))  # [1, 2, 3, 4, 5, 6]
```

### Örnek 16: itertools kombinasyonları

```python
import itertools

# Permutations - tüm permütasyonlar
print(list(itertools.permutations([1, 2, 3], 2)))
# [(1, 2), (1, 3), (2, 1), (2, 3), (3, 1), (3, 2)]

# Combinations - kombinasyonlar (sıra önemli değil)
print(list(itertools.combinations([1, 2, 3], 2)))
# [(1, 2), (1, 3), (2, 3)]

# Combinations with replacement
print(list(itertools.combinations_with_replacement([1, 2], 2)))
# [(1, 1), (1, 2), (2, 2)]

# Product - kartezyen çarpım
print(list(itertools.product([1, 2], ['a', 'b'])))
# [(1, 'a'), (1, 'b'), (2, 'a'), (2, 'b')]
```

### Örnek 17: itertools.groupby

```python
import itertools

# Ardışık eşit elemanları gruplama
data = [1, 1, 2, 2, 2, 3, 3, 1, 1]
for key, group in itertools.groupby(data):
    print(f"{key}: {list(group)}")
# 1: [1, 1]
# 2: [2, 2, 2]
# 3: [3, 3]
# 1: [1, 1]

# Key function ile gruplama
people = [
    {'name': 'Alice', 'age': 25},
    {'name': 'Bob', 'age': 25},
    {'name': 'Charlie', 'age': 30},
    {'name': 'David', 'age': 30},
]

# Yaşa göre grupla (önce sıralanmalı!)
sorted_people = sorted(people, key=lambda x: x['age'])
for age, group in itertools.groupby(sorted_people, key=lambda x: x['age']):
    print(f"Age {age}: {[p['name'] for p in group]}")
# Age 25: ['Alice', 'Bob']
# Age 30: ['Charlie', 'David']
```

### Örnek 18: itertools.islice ve takewhile

```python
import itertools

# islice - belirli aralıktaki elemanları al
infinite = itertools.count(0)  # 0, 1, 2, 3, ...
first_10 = itertools.islice(infinite, 10)
print(list(first_10))  # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# takewhile - koşul sağlandığı sürece al
numbers = [1, 3, 5, 7, 8, 10, 12, 13]
less_than_8 = itertools.takewhile(lambda x: x < 8, numbers)
print(list(less_than_8))  # [1, 3, 5, 7]

# dropwhile - koşul sağlandığı sürece atla
numbers = [1, 3, 5, 7, 8, 10, 12, 13]
from_8 = itertools.dropwhile(lambda x: x < 8, numbers)
print(list(from_8))  # [8, 10, 12, 13]
```

---

## Memory Efficiency

Generator'ların en büyük avantajı memory verimliliğidir. Değerler lazy olarak üretilir.

### Örnek 19: Memory Comparison

```python
import sys

# List - tüm değerleri bellekte tutar
def list_approach(n):
    return [x**2 for x in range(n)]

# Generator - lazy evaluation
def generator_approach(n):
    return (x**2 for x in range(n))

n = 1000000
list_result = list_approach(n)
gen_result = generator_approach(n)

print(f"List size: {sys.getsizeof(list_result):,} bytes")
print(f"Generator size: {sys.getsizeof(gen_result):,} bytes")
# List size: ~8,000,000 bytes
# Generator size: ~128 bytes
```

### Örnek 20: Large File Processing

```python
def process_large_log_file(file_path):
    """Büyük log dosyalarını memory-efficient şekilde işler"""
    def read_lines():
        with open(file_path, 'r') as file:
            for line in file:
                yield line.strip()

    def filter_errors(lines):
        for line in lines:
            if 'ERROR' in line:
                yield line

    def parse_error_details(lines):
        for line in lines:
            # Parse timestamp, error code, message
            parts = line.split('|')
            if len(parts) >= 3:
                yield {
                    'timestamp': parts[0].strip(),
                    'code': parts[1].strip(),
                    'message': parts[2].strip()
                }

    # Pipeline oluştur
    lines = read_lines()
    errors = filter_errors(lines)
    parsed = parse_error_details(errors)

    return parsed

# Kullanım - dosya tamamı bellekte tutulmaz!
for error in process_large_log_file('application.log'):
    print(error)
```

---

## Infinite Generators

Sonsuz generator'lar, teorik olarak sonsuz veri akışı oluşturur.

### Örnek 21: Infinite Counter

```python
def counter(start=0, step=1):
    """Sonsuz sayaç generator"""
    current = start
    while True:
        yield current
        current += step

# Kullanım - islice ile sınırlandır
import itertools
c = counter(10, 5)
first_5 = itertools.islice(c, 5)
print(list(first_5))  # [10, 15, 20, 25, 30]
```

### Örnek 22: Infinite Cycle

```python
import itertools

def traffic_light():
    """Sonsuz trafik ışığı simülasyonu"""
    colors = ['Red', 'Yellow', 'Green']
    return itertools.cycle(colors)

# Kullanım
lights = traffic_light()
for i, color in enumerate(lights):
    if i >= 10:
        break
    print(f"Light {i+1}: {color}")
# Red, Yellow, Green, Red, Yellow, Green, ...
```

### Örnek 23: Random Data Generator

```python
import random
import itertools

def random_walk(start=0):
    """Sonsuz random walk generator"""
    position = start
    while True:
        yield position
        position += random.choice([-1, 1])

# Kullanım
walk = random_walk()
path = list(itertools.islice(walk, 20))
print(path)  # [0, 1, 0, -1, -2, -1, ...]
```

---

## Pipeline Patterns

Generator'ları pipeline pattern ile birleştirerek güçlü data processing yapabiliriz.

### Örnek 24: Data Processing Pipeline

```python
def read_data(file_path):
    """Adım 1: Veriyi oku"""
    with open(file_path, 'r') as file:
        for line in file:
            yield line.strip()

def parse_json_lines(lines):
    """Adım 2: JSON parse et"""
    import json
    for line in lines:
        try:
            yield json.loads(line)
        except json.JSONDecodeError:
            continue  # Hatalı satırları atla

def filter_active_users(records):
    """Adım 3: Aktif kullanıcıları filtrele"""
    for record in records:
        if record.get('status') == 'active':
            yield record

def extract_emails(records):
    """Adım 4: Email adreslerini çıkar"""
    for record in records:
        if 'email' in record:
            yield record['email']

def deduplicate(items):
    """Adım 5: Tekrarları kaldır"""
    seen = set()
    for item in items:
        if item not in seen:
            seen.add(item)
            yield item

# Pipeline'ı kur
pipeline = deduplicate(
    extract_emails(
        filter_active_users(
            parse_json_lines(
                read_data('users.jsonl')
            )
        )
    )
)

# Her email'i işle (memory-efficient!)
for email in pipeline:
    send_newsletter(email)
```

### Örnek 25: ETL Pipeline

```python
import csv
from datetime import datetime
from decimal import Decimal

def extract_csv(file_path):
    """Extract: CSV'den veri oku"""
    with open(file_path, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            yield row

def transform_sales_data(records):
    """Transform: Veriyi dönüştür"""
    for record in records:
        try:
            yield {
                'date': datetime.strptime(record['date'], '%Y-%m-%d'),
                'product': record['product'].strip().upper(),
                'quantity': int(record['quantity']),
                'price': Decimal(record['price']),
                'total': Decimal(record['quantity']) * Decimal(record['price'])
            }
        except (ValueError, KeyError) as e:
            print(f"Skipping invalid record: {e}")
            continue

def filter_high_value(records, threshold=1000):
    """Filter: Yüksek değerli satışları filtrele"""
    for record in records:
        if record['total'] >= threshold:
            yield record

def aggregate_by_product(records):
    """Aggregate: Ürüne göre topla"""
    from collections import defaultdict
    aggregates = defaultdict(lambda: {'quantity': 0, 'total': Decimal(0)})

    for record in records:
        product = record['product']
        aggregates[product]['quantity'] += record['quantity']
        aggregates[product]['total'] += record['total']

    for product, data in aggregates.items():
        yield {
            'product': product,
            'total_quantity': data['quantity'],
            'total_revenue': data['total']
        }

# ETL Pipeline
pipeline = aggregate_by_product(
    filter_high_value(
        transform_sales_data(
            extract_csv('sales.csv')
        ),
        threshold=500
    )
)

for result in pipeline:
    print(f"{result['product']}: {result['total_revenue']}")
```

---

## Generator-based Coroutines

Generator'lar, `send()` metodu ile coroutine olarak da kullanılabilir (Python 3.4 öncesi async pattern).

### Örnek 26: Simple Coroutine

```python
def coroutine_grep(pattern):
    """Pattern matching coroutine"""
    print(f"Searching for: {pattern}")
    try:
        while True:
            line = yield  # Veri al
            if pattern in line:
                print(f"Found: {line}")
    except GeneratorExit:
        print("Coroutine closing")

# Kullanım
grep = coroutine_grep("Python")
next(grep)  # Coroutine'i başlat (prime)

grep.send("Java is great")
grep.send("Python is awesome")  # Found: Python is awesome
grep.send("C++ is fast")
grep.close()  # Coroutine'i kapat
```

### Örnek 27: Coroutine Pipeline

```python
def producer(target):
    """Veri üreten coroutine"""
    while True:
        data = yield
        target.send(data)

def filter_coroutine(pattern, target):
    """Filtreleme yapan coroutine"""
    while True:
        data = yield
        if pattern in data:
            target.send(data)

def printer():
    """Yazdıran coroutine"""
    while True:
        data = yield
        print(f"Output: {data}")

# Pipeline kur
p = printer()
f = filter_coroutine("important", p)
prod = producer(f)

# Prime all coroutines
next(p)
next(f)
next(prod)

# Veri gönder
prod.send("regular message")
prod.send("important message")  # Output: important message
prod.send("another regular")
prod.send("very important data")  # Output: very important data
```

### Örnek 28: Stateful Coroutine

```python
def moving_average(window_size):
    """Hareketli ortalama hesaplayan coroutine"""
    values = []
    while True:
        value = yield
        values.append(value)
        if len(values) > window_size:
            values.pop(0)

        average = sum(values) / len(values)
        print(f"Values: {values}, Average: {average:.2f}")

# Kullanım
ma = moving_average(3)
next(ma)  # Prime

ma.send(10)  # Average: 10.00
ma.send(20)  # Average: 15.00
ma.send(30)  # Average: 20.00
ma.send(40)  # Average: 30.00 (son 3 değer)
```

---

## Advanced Patterns

### Örnek 29: Generator Decorator

```python
from functools import wraps

def coroutine(func):
    """Coroutine'leri otomatik prime eden decorator"""
    @wraps(func)
    def primer(*args, **kwargs):
        gen = func(*args, **kwargs)
        next(gen)
        return gen
    return primer

@coroutine
def grep(pattern):
    print(f"Looking for {pattern}")
    while True:
        line = yield
        if pattern in line:
            print(f"Match: {line}")

# Otomatik prime edilir
g = grep("error")
g.send("info: all good")
g.send("error: something wrong")  # Match: error: something wrong
```

### Örnek 30: Generator State Machine

```python
def traffic_light_fsm():
    """Finite State Machine olarak trafik ışığı"""
    print("Starting at RED")
    while True:
        # RED state
        action = yield "RED"
        if action == "next":
            print("Switching to YELLOW")

        # YELLOW state
        action = yield "YELLOW"
        if action == "next":
            print("Switching to GREEN")

        # GREEN state
        action = yield "GREEN"
        if action == "next":
            print("Switching to RED")

# Kullanım
light = traffic_light_fsm()
print(next(light))  # RED
print(light.send("next"))  # YELLOW
print(light.send("next"))  # GREEN
print(light.send("next"))  # RED
```

---

## Best Practices

### 1. Generator İsimlendirme
```python
# İyi - generator olduğu açık
def iter_users():
    pass

def generate_primes():
    pass

def read_lines():
    pass

# Kötü - function gibi görünüyor
def get_users():  # List döndürüyor gibi
    pass
```

### 2. Exception Handling
```python
def safe_generator(items):
    """Exception'ları düzgün handle eden generator"""
    try:
        for item in items:
            try:
                yield process(item)
            except ValueError as e:
                print(f"Skipping invalid item: {e}")
                continue
    finally:
        # Cleanup kodu
        print("Generator closing, cleaning up resources")
```

### 3. Memory Profiling
```python
import tracemalloc

# Memory kullanımını ölç
tracemalloc.start()

# List approach
snapshot1 = tracemalloc.take_snapshot()
data_list = [x**2 for x in range(1000000)]
snapshot2 = tracemalloc.take_snapshot()
list_memory = sum(stat.size for stat in snapshot2.statistics('lineno'))

# Generator approach
snapshot3 = tracemalloc.take_snapshot()
data_gen = (x**2 for x in range(1000000))
snapshot4 = tracemalloc.take_snapshot()
gen_memory = sum(stat.size for stat in snapshot4.statistics('lineno'))

print(f"List: {list_memory:,} bytes")
print(f"Generator: {gen_memory:,} bytes")
```

---

## Özet

### Iterator Protocol
- `__iter__()` ve `__next__()` metodları
- `StopIteration` exception
- Custom iteration davranışı

### Generator Functions
- `yield` anahtar kelimesi
- Lazy evaluation
- State preservation
- Memory efficiency

### Generator Expressions
- Compact syntax
- List comprehension alternatifi
- Chaining desteği

### yield from
- Generator delegation
- Recursive generators
- Cleaner kod

### itertools Module
- chain, combinations, permutations
- groupby, islice, takewhile
- Infinite iterators (count, cycle, repeat)

### Memory Efficiency
- Lazy evaluation
- Stream processing
- Large file handling

### Pipeline Patterns
- Modüler data processing
- Composable generators
- ETL workflows

### Generator Coroutines
- `send()` metodu
- Two-way communication
- State machines

Generator ve Iterator'lar, Python'da memory-efficient, elegant ve ölçeklenebilir kod yazmanın temelidir!
