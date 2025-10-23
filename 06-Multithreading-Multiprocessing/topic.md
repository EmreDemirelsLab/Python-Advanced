# Multithreading & Multiprocessing (İleri Seviye)

## İçindekiler
1. [Threading Modülü](#threading-modülü)
2. [Global Interpreter Lock (GIL)](#global-interpreter-lock-gil)
3. [Thread Safety ve Senkronizasyon](#thread-safety-ve-senkronizasyon)
4. [Multiprocessing](#multiprocessing)
5. [Concurrent.futures](#concurrentfutures)
6. [Shared Memory ve IPC](#shared-memory-ve-ipc)
7. [Best Practices](#best-practices)

---

## Threading Modülü

### Temel Thread Kullanımı

Threading, aynı process içinde birden fazla iş parçacığı çalıştırmayı sağlar. Python'da `threading` modülü kullanılır.

```python
import threading
import time

# Basit thread örneği
def worker(name, delay):
    """Thread'in çalıştıracağı fonksiyon"""
    print(f"Thread {name} başladı")
    time.sleep(delay)
    print(f"Thread {name} tamamlandı")

# Thread oluşturma ve başlatma
threads = []
for i in range(5):
    t = threading.Thread(target=worker, args=(f"T{i}", i * 0.5))
    threads.append(t)
    t.start()

# Tüm thread'lerin bitmesini bekle
for t in threads:
    t.join()

print("Tüm thread'ler tamamlandı")
```

### Thread Sınıfından Türetme

```python
import threading
import time
from typing import Any

class DownloadThread(threading.Thread):
    """Dosya indirme işlemini simüle eden thread sınıfı"""

    def __init__(self, url: str, thread_id: int):
        super().__init__()
        self.url = url
        self.thread_id = thread_id
        self.result = None
        self._stop_event = threading.Event()

    def run(self):
        """Thread'in ana işlevi"""
        print(f"[Thread-{self.thread_id}] İndirme başladı: {self.url}")

        for i in range(5):
            if self._stop_event.is_set():
                print(f"[Thread-{self.thread_id}] Durduruldu")
                return

            time.sleep(0.5)
            print(f"[Thread-{self.thread_id}] İlerleme: %{(i+1)*20}")

        self.result = f"Data from {self.url}"
        print(f"[Thread-{self.thread_id}] Tamamlandı")

    def stop(self):
        """Thread'i durdur"""
        self._stop_event.set()

# Kullanım
downloaders = [
    DownloadThread(f"http://example.com/file{i}.zip", i)
    for i in range(3)
]

for d in downloaders:
    d.start()

for d in downloaders:
    d.join()
    print(f"Sonuç: {d.result}")
```

### Thread-Local Storage

Her thread'in kendi veri alanı - thread'ler arası veri karışmasını önler.

```python
import threading
import random

# Thread-local data
thread_local = threading.local()

def process_data(thread_id):
    """Her thread kendi verisiyle çalışır"""
    # Her thread için farklı değer
    thread_local.value = random.randint(1, 100)
    thread_local.name = f"Thread-{thread_id}"

    print(f"{thread_local.name} başladı, değer: {thread_local.value}")

    # Simulate processing
    total = sum(range(thread_local.value))

    print(f"{thread_local.name} sonuç: {total}")

threads = [threading.Thread(target=process_data, args=(i,)) for i in range(5)]

for t in threads:
    t.start()

for t in threads:
    t.join()
```

---

## Global Interpreter Lock (GIL)

### GIL Nedir?

GIL, CPython'da aynı anda sadece bir thread'in Python bytecode çalıştırmasına izin veren bir mutex'tir. Bu, CPU-bound işlemlerde threading'in performans artışı sağlamaması anlamına gelir.

```python
import threading
import time

def cpu_bound_task(n):
    """CPU-yoğun işlem - GIL'den etkilenir"""
    total = 0
    for i in range(n):
        total += i ** 2
    return total

def io_bound_task(n):
    """I/O-yoğun işlem - GIL'den daha az etkilenir"""
    time.sleep(n)
    return f"Completed after {n} seconds"

# CPU-bound: Threading YAVAŞ
start = time.time()
threads = []
for _ in range(4):
    t = threading.Thread(target=cpu_bound_task, args=(10_000_000,))
    threads.append(t)
    t.start()

for t in threads:
    t.join()

print(f"CPU-bound threading: {time.time() - start:.2f} saniye")

# CPU-bound: Sıralı (karşılaştırma için)
start = time.time()
for _ in range(4):
    cpu_bound_task(10_000_000)
print(f"CPU-bound sıralı: {time.time() - start:.2f} saniye")

# I/O-bound: Threading HIZLI
start = time.time()
threads = []
for _ in range(4):
    t = threading.Thread(target=io_bound_task, args=(1,))
    threads.append(t)
    t.start()

for t in threads:
    t.join()

print(f"I/O-bound threading: {time.time() - start:.2f} saniye")
```

### GIL'i Aşmanın Yolları

```python
import multiprocessing
import numpy as np
import time

def heavy_computation(data):
    """Ağır hesaplama - NumPy GIL'i serbest bırakır"""
    result = np.sum(data ** 2)
    return result

# 1. NumPy kullan (C extension GIL'i serbest bırakır)
start = time.time()
data = np.random.random(10_000_000)
result = heavy_computation(data)
print(f"NumPy ile: {time.time() - start:.3f} saniye")

# 2. Multiprocessing kullan (ayrı process = ayrı GIL)
def process_chunk(chunk):
    return np.sum(chunk ** 2)

start = time.time()
chunks = np.array_split(data, 4)

with multiprocessing.Pool(4) as pool:
    results = pool.map(process_chunk, chunks)
    total = sum(results)

print(f"Multiprocessing ile: {time.time() - start:.3f} saniye")
```

---

## Thread Safety ve Senkronizasyon

### Lock (Mutex)

Thread'lerin paylaşılan kaynaklara senkronize erişimi için temel mekanizma.

```python
import threading

class BankAccount:
    """Thread-safe banka hesabı"""

    def __init__(self, balance=0):
        self.balance = balance
        self._lock = threading.Lock()

    def deposit(self, amount):
        """Para yatırma - thread-safe"""
        with self._lock:  # Context manager ile otomatik acquire/release
            current = self.balance
            # Simulate processing delay
            import time
            time.sleep(0.001)
            self.balance = current + amount

    def withdraw(self, amount):
        """Para çekme - thread-safe"""
        with self._lock:
            if self.balance >= amount:
                current = self.balance
                import time
                time.sleep(0.001)
                self.balance = current - amount
                return True
            return False

    def get_balance(self):
        """Bakiye sorgulama - thread-safe"""
        with self._lock:
            return self.balance

# Test
account = BankAccount(1000)

def make_transactions(account, num_transactions):
    for _ in range(num_transactions):
        account.deposit(100)
        account.withdraw(50)

threads = [
    threading.Thread(target=make_transactions, args=(account, 100))
    for _ in range(10)
]

for t in threads:
    t.start()

for t in threads:
    t.join()

print(f"Final balance: {account.get_balance()}")  # 1000 + (10 * 100 * 50) = 6000
```

### RLock (Reentrant Lock)

Aynı thread tarafından birden fazla kez acquire edilebilen lock.

```python
import threading

class TreeNode:
    """Thread-safe ağaç düğümü - RLock gerektirir"""

    _lock = threading.RLock()  # Class-level reentrant lock

    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None

    def insert(self, value):
        """Değer ekleme - recursive, bu yüzden RLock gerekli"""
        with TreeNode._lock:
            if value < self.value:
                if self.left is None:
                    self.left = TreeNode(value)
                else:
                    self.left.insert(value)  # Aynı lock'u tekrar acquire eder
            else:
                if self.right is None:
                    self.right = TreeNode(value)
                else:
                    self.right.insert(value)

    def search(self, value):
        """Değer arama - recursive"""
        with TreeNode._lock:
            if value == self.value:
                return True
            elif value < self.value and self.left:
                return self.left.search(value)
            elif value > self.value and self.right:
                return self.right.search(value)
            return False

# Test
root = TreeNode(50)

def insert_values(root, values):
    for v in values:
        root.insert(v)

import random
threads = []
for _ in range(5):
    values = [random.randint(1, 100) for _ in range(20)]
    t = threading.Thread(target=insert_values, args=(root, values))
    threads.append(t)
    t.start()

for t in threads:
    t.join()

print(f"50 var mı? {root.search(50)}")
```

### Semaphore

Belirli sayıda thread'in aynı anda bir kaynağa erişmesine izin verir.

```python
import threading
import time
import random

class ConnectionPool:
    """Thread-safe bağlantı havuzu - Semaphore ile"""

    def __init__(self, max_connections=3):
        self.semaphore = threading.Semaphore(max_connections)
        self.active_connections = 0
        self._lock = threading.Lock()

    def acquire_connection(self, client_id):
        """Bağlantı al"""
        print(f"[{client_id}] Bağlantı bekleniyor...")
        self.semaphore.acquire()

        with self._lock:
            self.active_connections += 1
            print(f"[{client_id}] Bağlantı alındı! Aktif: {self.active_connections}")

        # Simulate using connection
        time.sleep(random.uniform(1, 3))

        with self._lock:
            self.active_connections -= 1
            print(f"[{client_id}] Bağlantı bırakıldı. Aktif: {self.active_connections}")

        self.semaphore.release()

# Test - 10 client, sadece 3 aynı anda bağlanabilir
pool = ConnectionPool(max_connections=3)

threads = [
    threading.Thread(target=pool.acquire_connection, args=(f"Client-{i}",))
    for i in range(10)
]

for t in threads:
    t.start()

for t in threads:
    t.join()
```

### Event

Thread'ler arası sinyal gönderme mekanizması.

```python
import threading
import time

class DataProcessor:
    """Event ile senkronize veri işleyici"""

    def __init__(self):
        self.data_ready = threading.Event()
        self.processing_done = threading.Event()
        self.data = None

    def produce_data(self):
        """Veri üretici"""
        print("[Producer] Veri hazırlanıyor...")
        time.sleep(2)

        self.data = list(range(1, 11))
        print(f"[Producer] Veri hazır: {self.data}")

        self.data_ready.set()  # Signal: data hazır

        # İşlemin bitmesini bekle
        self.processing_done.wait()
        print("[Producer] İşlem tamamlandı, temizlik yapılıyor")

    def consume_data(self):
        """Veri tüketici"""
        print("[Consumer] Veri bekleniyor...")

        self.data_ready.wait()  # Veri hazır olana kadar bekle

        print(f"[Consumer] Veri alındı: {self.data}")
        result = sum(self.data)
        print(f"[Consumer] Toplam: {result}")

        self.processing_done.set()  # Signal: işlem bitti

# Test
processor = DataProcessor()

producer = threading.Thread(target=processor.produce_data)
consumer = threading.Thread(target=processor.consume_data)

producer.start()
consumer.start()

producer.join()
consumer.join()
```

### Condition Variable

Daha karmaşık senkronizasyon senaryoları için.

```python
import threading
import time
import random

class BoundedQueue:
    """Thread-safe bounded queue - Condition variable ile"""

    def __init__(self, max_size=5):
        self.queue = []
        self.max_size = max_size
        self.condition = threading.Condition()

    def put(self, item):
        """Kuyruğa ekle - dolu ise bekle"""
        with self.condition:
            while len(self.queue) >= self.max_size:
                print(f"[Producer] Kuyruk dolu, bekleniyor...")
                self.condition.wait()  # Bekle

            self.queue.append(item)
            print(f"[Producer] Eklendi: {item}, Kuyruk: {len(self.queue)}")
            self.condition.notify()  # Consumer'ı uyandır

    def get(self):
        """Kuyruktan al - boş ise bekle"""
        with self.condition:
            while len(self.queue) == 0:
                print(f"[Consumer] Kuyruk boş, bekleniyor...")
                self.condition.wait()  # Bekle

            item = self.queue.pop(0)
            print(f"[Consumer] Alındı: {item}, Kuyruk: {len(self.queue)}")
            self.condition.notify()  # Producer'ı uyandır
            return item

# Test
queue = BoundedQueue(max_size=3)

def producer(queue, num_items):
    for i in range(num_items):
        time.sleep(random.uniform(0.1, 0.5))
        queue.put(f"Item-{i}")

def consumer(queue, num_items):
    for _ in range(num_items):
        time.sleep(random.uniform(0.2, 0.8))
        item = queue.get()

producer_thread = threading.Thread(target=producer, args=(queue, 10))
consumer_thread = threading.Thread(target=consumer, args=(queue, 10))

producer_thread.start()
consumer_thread.start()

producer_thread.join()
consumer_thread.join()
```

### Queue - Thread-Safe Kuyruk

Python'un built-in thread-safe queue implementasyonu.

```python
import threading
import queue
import time
import random

class WorkerPool:
    """Queue ile thread havuzu"""

    def __init__(self, num_workers=3):
        self.task_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.workers = []

        # Worker thread'leri oluştur
        for i in range(num_workers):
            worker = threading.Thread(target=self._worker, args=(i,))
            worker.daemon = True  # Ana program bitince otomatik kapan
            worker.start()
            self.workers.append(worker)

    def _worker(self, worker_id):
        """Worker thread'in çalıştıracağı fonksiyon"""
        while True:
            try:
                # Task al (timeout ile)
                task = self.task_queue.get(timeout=1)

                if task is None:  # Poison pill - worker'ı durdur
                    break

                # Task'ı işle
                task_id, data = task
                print(f"[Worker-{worker_id}] Task-{task_id} işleniyor...")

                # Simulate processing
                time.sleep(random.uniform(0.5, 2))
                result = data * 2

                # Sonucu kaydet
                self.result_queue.put((task_id, result))
                print(f"[Worker-{worker_id}] Task-{task_id} tamamlandı")

                # Task tamamlandı sinyali
                self.task_queue.task_done()

            except queue.Empty:
                continue

    def submit(self, task_id, data):
        """Task gönder"""
        self.task_queue.put((task_id, data))

    def wait_completion(self):
        """Tüm task'ların bitmesini bekle"""
        self.task_queue.join()

    def shutdown(self):
        """Worker'ları kapat"""
        for _ in self.workers:
            self.task_queue.put(None)  # Poison pill

        for worker in self.workers:
            worker.join()

    def get_results(self):
        """Sonuçları al"""
        results = []
        while not self.result_queue.empty():
            results.append(self.result_queue.get())
        return results

# Test
pool = WorkerPool(num_workers=3)

# Task'ları gönder
for i in range(10):
    pool.submit(i, random.randint(1, 100))

# Tamamlanmasını bekle
pool.wait_completion()

# Sonuçları al
results = pool.get_results()
print(f"\nToplam {len(results)} sonuç alındı")
for task_id, result in sorted(results):
    print(f"Task-{task_id}: {result}")

# Kapat
pool.shutdown()
```

---

## Multiprocessing

### Process Oluşturma

Her process ayrı Python interpreter ve bellek alanı kullanır - GIL'den etkilenmez.

```python
import multiprocessing
import os
import time

def worker(num, name):
    """Process worker fonksiyonu"""
    print(f"[{name}] PID: {os.getpid()}, Parent PID: {os.getppid()}")
    time.sleep(num)
    print(f"[{name}] İşlem tamamlandı")
    return num ** 2

if __name__ == '__main__':
    print(f"Ana process PID: {os.getpid()}")

    # Process oluşturma ve başlatma
    processes = []
    for i in range(4):
        p = multiprocessing.Process(
            target=worker,
            args=(i * 0.5, f"Process-{i}")
        )
        processes.append(p)
        p.start()

    # Tüm process'lerin bitmesini bekle
    for p in processes:
        p.join()

    print("Tüm process'ler tamamlandı")
```

### Process Pool

Process'leri yönetmek ve yeniden kullanmak için.

```python
import multiprocessing
import time

def cpu_intensive_task(n):
    """CPU-yoğun işlem"""
    result = 0
    for i in range(n):
        result += i ** 2
    return result

if __name__ == '__main__':
    # Process havuzu ile paralel işlem
    start = time.time()

    with multiprocessing.Pool(processes=4) as pool:
        # map - tüm sonuçları bekle
        results = pool.map(cpu_intensive_task, [10_000_000] * 8)

    print(f"Pool ile: {time.time() - start:.2f} saniye")
    print(f"Sonuçlar: {len(results)} adet")

    # Karşılaştırma - sıralı işlem
    start = time.time()
    results = [cpu_intensive_task(10_000_000) for _ in range(8)]
    print(f"Sıralı: {time.time() - start:.2f} saniye")
```

### Process Pool - İleri Teknikler

```python
import multiprocessing
import time

def process_item(item):
    """Her bir öğeyi işle"""
    time.sleep(0.5)
    return item * 2, item ** 2

if __name__ == '__main__':
    items = list(range(20))

    with multiprocessing.Pool(processes=4) as pool:
        # 1. map - basit, tüm sonuçları bekle
        print("map kullanımı:")
        results = pool.map(process_item, items)
        print(f"Sonuçlar: {results[:5]}...")

        # 2. imap - lazy iterator, bellek verimli
        print("\nimap kullanımı:")
        for result in pool.imap(process_item, items):
            print(f"Sonuç alındı: {result}")

        # 3. imap_unordered - sırasız ama daha hızlı
        print("\nimap_unordered kullanımı:")
        for result in pool.imap_unordered(process_item, items):
            print(f"Sonuç alındı: {result}")

        # 4. apply_async - asenkron, non-blocking
        print("\napply_async kullanımı:")
        async_results = [pool.apply_async(process_item, (i,)) for i in items]

        # Sonuçları al (gerektiğinde)
        for async_result in async_results:
            result = async_result.get(timeout=5)  # Timeout ile
            print(f"Async sonuç: {result}")

        # 5. starmap - multiple arguments
        print("\nstarmap kullanımı:")
        def multi_arg_func(a, b, c):
            return a + b * c

        args_list = [(i, i+1, i+2) for i in range(10)]
        results = pool.starmap(multi_arg_func, args_list)
        print(f"Starmap sonuçlar: {results}")
```

### Process İletişimi - Queue

```python
import multiprocessing
import time
import random

def producer(queue, num_items):
    """Veri üretici process"""
    for i in range(num_items):
        item = random.randint(1, 100)
        queue.put(item)
        print(f"[Producer] Üretildi: {item}")
        time.sleep(random.uniform(0.1, 0.5))

    queue.put(None)  # Poison pill
    print("[Producer] Tamamlandı")

def consumer(queue, consumer_id):
    """Veri tüketici process"""
    total = 0
    while True:
        item = queue.get()

        if item is None:
            queue.put(None)  # Diğer consumer'lar için poison pill'i geri koy
            break

        total += item
        print(f"[Consumer-{consumer_id}] İşlendi: {item}, Toplam: {total}")
        time.sleep(random.uniform(0.1, 0.3))

    print(f"[Consumer-{consumer_id}] Tamamlandı, Toplam: {total}")
    return total

if __name__ == '__main__':
    # Process-safe queue
    queue = multiprocessing.Queue()

    # Producer process
    prod = multiprocessing.Process(target=producer, args=(queue, 20))

    # Consumer processes
    consumers = [
        multiprocessing.Process(target=consumer, args=(queue, i))
        for i in range(3)
    ]

    # Başlat
    prod.start()
    for c in consumers:
        c.start()

    # Bekle
    prod.join()
    for c in consumers:
        c.join()

    print("Tüm process'ler tamamlandı")
```

### Process İletişimi - Pipe

İki process arası çift yönlü iletişim.

```python
import multiprocessing
import time

def worker(conn, worker_id):
    """Worker process - pipe ile iletişim"""
    print(f"[Worker-{worker_id}] Başladı")

    while True:
        # Ana process'ten mesaj al
        msg = conn.recv()

        if msg == "STOP":
            print(f"[Worker-{worker_id}] Durduruluyor")
            break

        # İşle
        result = msg * 2
        print(f"[Worker-{worker_id}] İşlendi: {msg} -> {result}")

        # Sonucu gönder
        conn.send(result)

    conn.close()

if __name__ == '__main__':
    # Pipe oluştur
    parent_conn, child_conn = multiprocessing.Pipe()

    # Worker process başlat
    worker_proc = multiprocessing.Process(target=worker, args=(child_conn, 1))
    worker_proc.start()

    # Worker'a görevler gönder
    for i in range(5):
        parent_conn.send(i * 10)
        result = parent_conn.recv()
        print(f"[Main] Sonuç alındı: {result}")
        time.sleep(0.5)

    # Worker'ı durdur
    parent_conn.send("STOP")
    worker_proc.join()

    print("İşlem tamamlandı")
```

---

## Concurrent.futures

Modern, yüksek seviyeli paralel programlama API'si.

### ThreadPoolExecutor

```python
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import requests

def download_url(url):
    """URL'den içerik indir"""
    print(f"İndiriliyor: {url}")
    try:
        response = requests.get(url, timeout=5)
        return url, len(response.content), "OK"
    except Exception as e:
        return url, 0, str(e)

# ThreadPoolExecutor ile paralel download
urls = [
    'https://www.python.org',
    'https://www.github.com',
    'https://www.stackoverflow.com',
    'https://www.reddit.com',
    'https://www.wikipedia.org',
]

print("ThreadPoolExecutor ile:")
start = time.time()

with ThreadPoolExecutor(max_workers=5) as executor:
    # submit ile görevleri gönder
    future_to_url = {executor.submit(download_url, url): url for url in urls}

    # Sonuçları al (tamamlandıkça)
    for future in as_completed(future_to_url):
        url = future_to_url[future]
        try:
            result_url, size, status = future.result()
            print(f"✓ {result_url}: {size} bytes, Status: {status}")
        except Exception as e:
            print(f"✗ {url}: Hata - {e}")

print(f"Süre: {time.time() - start:.2f} saniye")
```

### ProcessPoolExecutor

```python
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import time

def cpu_bound_task(n):
    """CPU-yoğun işlem"""
    print(f"İşlem başladı: n={n}")
    result = sum(i * i for i in range(n))
    print(f"İşlem bitti: n={n}")
    return n, result

if __name__ == '__main__':
    tasks = [10_000_000, 5_000_000, 15_000_000, 8_000_000]

    print("ProcessPoolExecutor ile:")
    start = time.time()

    with ProcessPoolExecutor(max_workers=4) as executor:
        # Submit tüm task'ları
        futures = [executor.submit(cpu_bound_task, n) for n in tasks]

        # İlk tamamlanan sonucu al
        done, not_done = wait(futures, return_when=FIRST_COMPLETED)

        print(f"\nİlk tamamlanan: {len(done)} adet")
        for future in done:
            n, result = future.result()
            print(f"n={n}, result={result}")

        # Kalanları bekle
        print(f"\nKalan {len(not_done)} task bekleniyor...")
        for future in not_done:
            n, result = future.result()
            print(f"n={n}, result={result}")

    print(f"\nToplam süre: {time.time() - start:.2f} saniye")
```

### Executor - map() Kullanımı

```python
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import time

def task(n):
    """Basit task"""
    time.sleep(0.5)
    return n * n

if __name__ == '__main__':
    numbers = list(range(20))

    # ThreadPoolExecutor ile map
    print("ThreadPoolExecutor:")
    start = time.time()

    with ThreadPoolExecutor(max_workers=5) as executor:
        results = executor.map(task, numbers)
        print(f"Sonuçlar: {list(results)}")

    print(f"Süre: {time.time() - start:.2f} saniye")

    # ProcessPoolExecutor ile map
    print("\nProcessPoolExecutor:")
    start = time.time()

    with ProcessPoolExecutor(max_workers=4) as executor:
        results = executor.map(task, numbers)
        print(f"Sonuçlar: {list(results)}")

    print(f"Süre: {time.time() - start:.2f} saniye")
```

### Future Callbacks

```python
from concurrent.futures import ThreadPoolExecutor
import time

def task(n):
    """Task fonksiyonu"""
    time.sleep(n)
    if n == 3:
        raise ValueError(f"n={n} kabul edilemez!")
    return n * n

def task_done_callback(future):
    """Task tamamlandığında çağrılır"""
    try:
        result = future.result()
        print(f"✓ Callback: Sonuç = {result}")
    except Exception as e:
        print(f"✗ Callback: Hata = {e}")

# Executor ile callback kullanımı
with ThreadPoolExecutor(max_workers=3) as executor:
    for n in [1, 2, 3, 4]:
        future = executor.submit(task, n)
        future.add_done_callback(task_done_callback)

print("Tüm task'lar gönderildi, callbacks çalışıyor...")
time.sleep(5)
```

---

## Shared Memory ve IPC

### Shared Value ve Array

```python
import multiprocessing
import time

def increment_counter(counter, lock, num_iterations):
    """Counter'ı artır - shared memory ile"""
    for _ in range(num_iterations):
        with lock:
            counter.value += 1

def fill_array(arr, lock, start, end):
    """Array'i doldur - shared memory ile"""
    for i in range(start, end):
        with lock:
            arr[i] = i * i

if __name__ == '__main__':
    # Shared value
    counter = multiprocessing.Value('i', 0)  # 'i' = integer
    lock = multiprocessing.Lock()

    processes = [
        multiprocessing.Process(target=increment_counter, args=(counter, lock, 10000))
        for _ in range(10)
    ]

    for p in processes:
        p.start()

    for p in processes:
        p.join()

    print(f"Counter: {counter.value}")  # 100000

    # Shared array
    shared_array = multiprocessing.Array('i', 100)  # 100 integer'lık array

    # Array'i paralel doldur
    chunk_size = 25
    processes = [
        multiprocessing.Process(
            target=fill_array,
            args=(shared_array, lock, i*chunk_size, (i+1)*chunk_size)
        )
        for i in range(4)
    ]

    for p in processes:
        p.start()

    for p in processes:
        p.join()

    print(f"Array ilk 10 eleman: {shared_array[:10]}")
```

### Manager - Karmaşık Veri Yapıları

```python
import multiprocessing
import time

def add_to_dict(shared_dict, lock, key, value):
    """Shared dictionary'e ekle"""
    with lock:
        shared_dict[key] = value
        print(f"Eklendi: {key} = {value}")

def add_to_list(shared_list, lock, items):
    """Shared list'e ekle"""
    with lock:
        shared_list.extend(items)
        print(f"Liste boyutu: {len(shared_list)}")

if __name__ == '__main__':
    # Manager ile shared data structures
    with multiprocessing.Manager() as manager:
        # Shared dictionary
        shared_dict = manager.dict()

        # Shared list
        shared_list = manager.list()

        # Lock
        lock = manager.Lock()

        # Dictionary'e paralel yazma
        dict_processes = [
            multiprocessing.Process(
                target=add_to_dict,
                args=(shared_dict, lock, f"key_{i}", i*10)
            )
            for i in range(10)
        ]

        for p in dict_processes:
            p.start()

        for p in dict_processes:
            p.join()

        print(f"\nShared dict: {dict(shared_dict)}")

        # List'e paralel yazma
        list_processes = [
            multiprocessing.Process(
                target=add_to_list,
                args=(shared_list, lock, list(range(i*10, (i+1)*10)))
            )
            for i in range(5)
        ]

        for p in list_processes:
            p.start()

        for p in list_processes:
            p.join()

        print(f"\nShared list: {len(shared_list)} eleman")
        print(f"İlk 10: {shared_list[:10]}")
```

### Namespace - Shared Objects

```python
import multiprocessing
import time

def worker(namespace, worker_id):
    """Namespace ile çalışan worker"""
    namespace.counter += 1
    namespace.workers.append(worker_id)
    time.sleep(0.5)
    namespace.status = f"Worker-{worker_id} completed"
    print(f"[Worker-{worker_id}] Counter: {namespace.counter}")

if __name__ == '__main__':
    with multiprocessing.Manager() as manager:
        # Namespace oluştur
        namespace = manager.Namespace()
        namespace.counter = 0
        namespace.workers = manager.list()
        namespace.status = "Idle"

        # Worker processes
        processes = [
            multiprocessing.Process(target=worker, args=(namespace, i))
            for i in range(5)
        ]

        for p in processes:
            p.start()

        for p in processes:
            p.join()

        print(f"\nFinal counter: {namespace.counter}")
        print(f"Workers: {list(namespace.workers)}")
        print(f"Status: {namespace.status}")
```

---

## Best Practices

### 1. Doğru Araç Seçimi

```python
import time
import threading
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

def io_bound_task():
    """I/O-bound: Network, disk, database"""
    time.sleep(1)
    return "IO result"

def cpu_bound_task():
    """CPU-bound: Hesaplama, veri işleme"""
    return sum(i*i for i in range(1_000_000))

if __name__ == '__main__':
    # I/O-bound: Threading kullan
    print("I/O-bound task - Threading:")
    start = time.time()
    with ThreadPoolExecutor(max_workers=10) as executor:
        results = list(executor.map(lambda x: io_bound_task(), range(10)))
    print(f"Süre: {time.time() - start:.2f}s")  # ~1s (paralel)

    # CPU-bound: Multiprocessing kullan
    print("\nCPU-bound task - Multiprocessing:")
    start = time.time()
    with ProcessPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(lambda x: cpu_bound_task(), range(4)))
    print(f"Süre: {time.time() - start:.2f}s")  # GIL'den etkilenmez
```

### 2. Resource Management

```python
import threading
import contextlib

class ResourcePool:
    """Thread-safe kaynak havuzu"""

    def __init__(self, create_resource, max_resources=5):
        self._create = create_resource
        self._pool = []
        self._in_use = set()
        self._lock = threading.Lock()
        self._available = threading.Semaphore(max_resources)

    @contextlib.contextmanager
    def acquire(self):
        """Kaynak al - context manager ile otomatik release"""
        self._available.acquire()

        with self._lock:
            if self._pool:
                resource = self._pool.pop()
            else:
                resource = self._create()

            self._in_use.add(id(resource))

        try:
            yield resource
        finally:
            with self._lock:
                self._in_use.remove(id(resource))
                self._pool.append(resource)

            self._available.release()

    def stats(self):
        """İstatistikler"""
        with self._lock:
            return {
                'available': len(self._pool),
                'in_use': len(self._in_use),
                'total': len(self._pool) + len(self._in_use)
            }

# Kullanım
def create_db_connection():
    """Simüle edilmiş DB bağlantısı"""
    import random
    return f"Connection-{random.randint(1000, 9999)}"

pool = ResourcePool(create_db_connection, max_resources=3)

def worker(pool, worker_id):
    with pool.acquire() as conn:
        print(f"[Worker-{worker_id}] Bağlantı alındı: {conn}")
        time.sleep(1)
        print(f"[Worker-{worker_id}] Bağlantı bırakıldı")

threads = [threading.Thread(target=worker, args=(pool, i)) for i in range(10)]
for t in threads:
    t.start()
for t in threads:
    t.join()

print(f"Pool stats: {pool.stats()}")
```

### 3. Deadlock Önleme

```python
import threading
import time

# YANLIŞ - Deadlock riski var!
lock_a = threading.Lock()
lock_b = threading.Lock()

def bad_worker_1():
    with lock_a:
        time.sleep(0.1)
        with lock_b:  # Worker 2 lock_b'yi tutuyorsa deadlock!
            print("Worker 1")

def bad_worker_2():
    with lock_b:
        time.sleep(0.1)
        with lock_a:  # Worker 1 lock_a'yı tutuyorsa deadlock!
            print("Worker 2")

# DOĞRU - Lock sıralaması tutarlı
def good_worker_1():
    with lock_a:  # Her zaman önce lock_a
        with lock_b:  # Sonra lock_b
            print("Worker 1")

def good_worker_2():
    with lock_a:  # Her zaman önce lock_a
        with lock_b:  # Sonra lock_b
            print("Worker 2")

# Alternatif: timeout ile lock alma
def safe_worker():
    while True:
        acquired_a = lock_a.acquire(timeout=1)
        if not acquired_a:
            continue

        try:
            acquired_b = lock_b.acquire(timeout=1)
            if not acquired_b:
                continue

            try:
                print("Safe worker")
                break
            finally:
                lock_b.release()
        finally:
            lock_a.release()
```

### 4. Error Handling

```python
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

def risky_task(n):
    """Hata verebilecek task"""
    time.sleep(0.5)

    if n % 3 == 0:
        raise ValueError(f"n={n} ile çalışamam!")

    return n * n

# Hataları yakalama ve loglama
results = []
errors = []

with ThreadPoolExecutor(max_workers=5) as executor:
    future_to_n = {executor.submit(risky_task, n): n for n in range(10)}

    for future in as_completed(future_to_n):
        n = future_to_n[future]

        try:
            result = future.result()
            results.append((n, result))
            print(f"✓ n={n}, result={result}")

        except Exception as e:
            errors.append((n, str(e)))
            print(f"✗ n={n}, error={e}")

print(f"\nBaşarılı: {len(results)}, Hatalı: {len(errors)}")
```

### 5. Graceful Shutdown

```python
import threading
import time
import signal
import sys

class GracefulWorker:
    """Düzgün kapanabilen worker"""

    def __init__(self):
        self.shutdown_event = threading.Event()
        self.threads = []

    def worker(self, worker_id):
        """Worker thread"""
        print(f"[Worker-{worker_id}] Başladı")

        while not self.shutdown_event.is_set():
            # İşlem yap (küçük parçalar halinde)
            print(f"[Worker-{worker_id}] Çalışıyor...")
            time.sleep(1)

        print(f"[Worker-{worker_id}] Kapanıyor...")

    def start(self, num_workers=3):
        """Worker'ları başlat"""
        for i in range(num_workers):
            t = threading.Thread(target=self.worker, args=(i,))
            t.start()
            self.threads.append(t)

    def shutdown(self):
        """Tüm worker'ları kapat"""
        print("\nKapatma sinyali alındı...")
        self.shutdown_event.set()

        # Tüm thread'lerin bitmesini bekle
        for t in self.threads:
            t.join(timeout=5)

        print("Tüm worker'lar kapandı")

# Kullanım
worker_manager = GracefulWorker()

def signal_handler(sig, frame):
    """SIGINT (Ctrl+C) handler"""
    worker_manager.shutdown()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

worker_manager.start(num_workers=3)
print("Worker'lar çalışıyor... (Ctrl+C ile durdur)")

# Ana thread'i beklet
for t in worker_manager.threads:
    t.join()
```

---

## Özet ve Karşılaştırma

### Threading vs Multiprocessing

| Özellik | Threading | Multiprocessing |
|---------|-----------|-----------------|
| **Bellek** | Paylaşılan bellek | Ayrı bellek alanları |
| **GIL** | Var (tek Python bytecode) | Yok (ayrı interpreter) |
| **I/O-bound** | Mükemmel | Gereksiz overhead |
| **CPU-bound** | Kötü (GIL) | Mükemmel |
| **İletişim** | Kolay (shared memory) | Zor (Queue, Pipe) |
| **Overhead** | Düşük | Yüksek (process spawn) |
| **Debugging** | Kolay | Zor |

### Senkronizasyon Mekanizmaları

| Mekanizma | Kullanım Senaryosu |
|-----------|-------------------|
| **Lock** | Basit mutual exclusion |
| **RLock** | Recursive lock ihtiyacı |
| **Semaphore** | Sınırlı kaynak erişimi |
| **Event** | Basit sinyal gönderme |
| **Condition** | Karmaşık senkronizasyon |
| **Queue** | Producer-consumer pattern |

### Best Practices Özet

1. **I/O-bound** → Threading veya AsyncIO
2. **CPU-bound** → Multiprocessing
3. **Karışık** → ProcessPoolExecutor + ThreadPoolExecutor
4. **Basit paralellik** → concurrent.futures kullan
5. **Karmaşık kontrol** → threading/multiprocessing modüllerini kullan
6. **Her zaman** → Context manager kullan (with statement)
7. **Her zaman** → Hata yakalama ekle
8. **Her zaman** → Graceful shutdown düşün
9. **Deadlock'tan kaçın** → Lock sıralaması tutarlı olsun
10. **Test et** → Race condition'ları bulmak zor, çok test et!
