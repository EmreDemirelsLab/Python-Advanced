# Async Programming (Asenkron Programlama)

## İçindekiler
1. [Async/Await Temelleri](#asyncawait-temelleri)
2. [Asyncio Modülü](#asyncio-modülü)
3. [Coroutines (Eş Yordamlar)](#coroutines)
4. [Event Loop (Olay Döngüsü)](#event-loop)
5. [Tasks ve Futures](#tasks-ve-futures)
6. [Async Context Managers](#async-context-managers)
7. [Async Iterators ve Generators](#async-iterators-ve-generators)
8. [Concurrency Patterns](#concurrency-patterns)
9. [Error Handling](#error-handling)
10. [Production Patterns](#production-patterns)

---

## Async/Await Temelleri

**Asenkron programlama**, I/O bağlı işlemlerde (network, disk, database) programın diğer görevleri yerine getirirken beklemesine olanak tanır. Python'da `async`/`await` anahtar kelimeleri ile modern asenkron kod yazılır.

### Örnek 1: Basit Async Function

```python
import asyncio
import time

# Senkron fonksiyon - bloklar
def sync_task(name: str, delay: int) -> str:
    print(f"[{time.strftime('%H:%M:%S')}] {name} başladı")
    time.sleep(delay)  # CPU'yu bloklar
    print(f"[{time.strftime('%H:%M:%S')}] {name} bitti")
    return f"{name} tamamlandı"

# Asenkron fonksiyon - bloklamaz
async def async_task(name: str, delay: int) -> str:
    print(f"[{time.strftime('%H:%M:%S')}] {name} başladı")
    await asyncio.sleep(delay)  # Başka görevlere izin verir
    print(f"[{time.strftime('%H:%M:%S')}] {name} bitti")
    return f"{name} tamamlandı"

# Senkron çalıştırma - 6 saniye sürer
def run_sync():
    start = time.time()
    sync_task("Task-1", 2)
    sync_task("Task-2", 2)
    sync_task("Task-3", 2)
    print(f"Toplam süre: {time.time() - start:.2f} saniye")

# Asenkron çalıştırma - 2 saniye sürer
async def run_async():
    start = time.time()
    await asyncio.gather(
        async_task("Task-1", 2),
        async_task("Task-2", 2),
        async_task("Task-3", 2)
    )
    print(f"Toplam süre: {time.time() - start:.2f} saniye")

if __name__ == "__main__":
    print("=== Senkron Çalıştırma ===")
    run_sync()

    print("\n=== Asenkron Çalıştırma ===")
    asyncio.run(run_async())
```

### Örnek 2: Async/Await Kuralları

```python
import asyncio
from typing import List, Any

# async def ile tanımlanan fonksiyonlar coroutine döner
async def fetch_data(user_id: int) -> dict:
    """Veritabanından veri çeker (simüle edilmiş)"""
    await asyncio.sleep(0.5)  # await sadece async fonksiyonlarda kullanılır
    return {"id": user_id, "name": f"User-{user_id}"}

# Normal fonksiyondan async fonksiyon çağrılamaz
def cannot_call_async():
    # data = await fetch_data(1)  # SyntaxError!
    # Doğru yol: asyncio.run() kullan
    data = asyncio.run(fetch_data(1))
    return data

# Async fonksiyondan normal fonksiyon çağrılabilir
async def can_call_sync():
    result = len("test")  # Normal fonksiyon - await gereksiz
    data = await fetch_data(1)  # Async fonksiyon - await gerekli
    return result, data

# Async fonksiyondan async fonksiyon çağırma
async def call_async_from_async():
    # await kullanmalısınız yoksa coroutine nesnesi döner
    data = await fetch_data(1)  # Doğru

    # coro = fetch_data(2)  # Yanlış! Coroutine çalışmaz
    # print(coro)  # <coroutine object fetch_data at 0x...>

    return data

# Birden fazla async işlemi paralel çalıştırma
async def fetch_multiple_users(user_ids: List[int]) -> List[dict]:
    """Birden fazla kullanıcıyı paralel olarak çeker"""
    # Sıralı çalıştırma (yavaş)
    # results = []
    # for uid in user_ids:
    #     results.append(await fetch_data(uid))  # 0.5 * n saniye

    # Paralel çalıştırma (hızlı)
    tasks = [fetch_data(uid) for uid in user_ids]
    results = await asyncio.gather(*tasks)  # 0.5 saniye (hepsi paralel)
    return results

async def main():
    # Tek kullanıcı çekme
    user = await fetch_data(1)
    print(f"Tek kullanıcı: {user}")

    # Çoklu kullanıcı çekme
    users = await fetch_multiple_users([1, 2, 3, 4, 5])
    print(f"Çoklu kullanıcı: {len(users)} kullanıcı çekildi")

    # Normal fonksiyon çağırma
    sync_result = cannot_call_async()
    print(f"Sync'ten async: {sync_result}")

if __name__ == "__main__":
    asyncio.run(main())
```

---

## Asyncio Modülü

**asyncio** modülü, Python'da asenkron programlama için temel altyapıyı sağlar. Event loop, task yönetimi, I/O işlemleri gibi özellikleri içerir.

### Örnek 3: Asyncio Temel Fonksiyonlar

```python
import asyncio
import aiohttp
import aiofiles
from typing import List, Dict
import json

# asyncio.run() - En üst seviye coroutine'i çalıştırır
async def top_level_example():
    """Ana async fonksiyon - program giriş noktası"""
    print("Program başladı")
    await asyncio.sleep(1)
    print("Program bitti")

# asyncio.create_task() - Coroutine'i task olarak planlar
async def background_task(name: str, count: int):
    """Arka planda çalışan görev"""
    for i in range(count):
        print(f"{name}: {i+1}/{count}")
        await asyncio.sleep(0.5)
    return f"{name} tamamlandı"

async def task_example():
    """Task oluşturma ve yönetme"""
    # Task'ları oluştur (hemen çalışmaya başlar)
    task1 = asyncio.create_task(background_task("Task-1", 3))
    task2 = asyncio.create_task(background_task("Task-2", 2))

    # Task'ların bitmesini bekle
    result1 = await task1
    result2 = await task2

    print(f"Sonuçlar: {result1}, {result2}")

# asyncio.gather() - Birden fazla coroutine'i paralel çalıştırır
async def gather_example():
    """Gather ile paralel çalıştırma"""
    # Tüm task'lar paralel çalışır
    results = await asyncio.gather(
        background_task("A", 2),
        background_task("B", 2),
        background_task("C", 2),
        return_exceptions=True  # Exception'ları yakalama
    )
    print(f"Gather sonuçları: {results}")

# asyncio.wait() - Task'ları bekler ve kontrol sağlar
async def wait_example():
    """Wait ile gelişmiş task kontrolü"""
    tasks = [
        asyncio.create_task(background_task(f"Task-{i}", i))
        for i in range(1, 4)
    ]

    # İlk tamamlananı bekle
    done, pending = await asyncio.wait(
        tasks,
        return_when=asyncio.FIRST_COMPLETED
    )

    print(f"İlk tamamlanan: {len(done)} task")
    print(f"Devam eden: {len(pending)} task")

    # Kalan task'ları iptal et
    for task in pending:
        task.cancel()

    # İptal edilenleri bekle
    await asyncio.gather(*pending, return_exceptions=True)

# asyncio.wait_for() - Timeout ile bekleme
async def slow_operation():
    """Yavaş işlem"""
    await asyncio.sleep(5)
    return "Tamamlandı"

async def timeout_example():
    """Timeout örneği"""
    try:
        # Maksimum 2 saniye bekle
        result = await asyncio.wait_for(slow_operation(), timeout=2.0)
        print(result)
    except asyncio.TimeoutError:
        print("İşlem zaman aşımına uğradı!")

# asyncio.as_completed() - Tamamlandıkça işle
async def as_completed_example():
    """Tamamlanan task'ları anında işle"""
    tasks = [
        background_task(f"Task-{i}", i)
        for i in range(1, 5)
    ]

    # Task'lar tamamlandıkça işlenir
    for coro in asyncio.as_completed(tasks):
        result = await coro
        print(f"Tamamlandı: {result}")

# Gerçek dünya örneği: Paralel HTTP istekleri
async def fetch_url(session: aiohttp.ClientSession, url: str) -> Dict:
    """URL'den veri çeker"""
    try:
        async with session.get(url, timeout=10) as response:
            return {
                "url": url,
                "status": response.status,
                "size": len(await response.text())
            }
    except Exception as e:
        return {"url": url, "error": str(e)}

async def parallel_http_requests():
    """Paralel HTTP istekleri"""
    urls = [
        "https://jsonplaceholder.typicode.com/posts/1",
        "https://jsonplaceholder.typicode.com/posts/2",
        "https://jsonplaceholder.typicode.com/posts/3",
        "https://jsonplaceholder.typicode.com/posts/4",
        "https://jsonplaceholder.typicode.com/posts/5",
    ]

    async with aiohttp.ClientSession() as session:
        # Tüm istekleri paralel gönder
        tasks = [fetch_url(session, url) for url in urls]
        results = await asyncio.gather(*tasks)

        for result in results:
            if "error" in result:
                print(f"❌ {result['url']}: {result['error']}")
            else:
                print(f"✓ {result['url']}: {result['status']} ({result['size']} bytes)")

async def main():
    print("=== Top Level Example ===")
    await top_level_example()

    print("\n=== Task Example ===")
    await task_example()

    print("\n=== Gather Example ===")
    await gather_example()

    print("\n=== Wait Example ===")
    await wait_example()

    print("\n=== Timeout Example ===")
    await timeout_example()

    print("\n=== As Completed Example ===")
    await as_completed_example()

    print("\n=== Parallel HTTP Requests ===")
    await parallel_http_requests()

if __name__ == "__main__":
    asyncio.run(main())
```

---

## Coroutines (Eş Yordamlar)

**Coroutine'ler**, duraklayabilen ve devam ettirilebilen fonksiyonlardır. `async def` ile tanımlanır ve `await` ifadesiyle duraklatılır.

### Örnek 4: Coroutine Yaşam Döngüsü

```python
import asyncio
import inspect
from typing import Coroutine, Any

# Coroutine tanımlama
async def simple_coroutine(name: str) -> str:
    """Basit bir coroutine"""
    print(f"{name} başladı")
    await asyncio.sleep(1)
    print(f"{name} bitti")
    return f"{name} tamamlandı"

# Coroutine durumlarını inceleyelim
def inspect_coroutine():
    """Coroutine nesnesini incele"""
    # Coroutine oluştur (henüz çalışmadı)
    coro = simple_coroutine("Test")

    print(f"Tip: {type(coro)}")
    print(f"Coroutine mu?: {inspect.iscoroutine(coro)}")
    print(f"Durum: {inspect.getcoroutinestate(coro)}")

    # Coroutine'i çalıştır
    result = asyncio.run(coro)
    print(f"Sonuç: {result}")

# Nested coroutines
async def fetch_user(user_id: int) -> dict:
    """Kullanıcı bilgisini çeker"""
    await asyncio.sleep(0.5)
    return {"id": user_id, "name": f"User-{user_id}"}

async def fetch_user_posts(user_id: int) -> list:
    """Kullanıcının gönderilerini çeker"""
    await asyncio.sleep(0.5)
    return [
        {"id": 1, "title": "Post 1"},
        {"id": 2, "title": "Post 2"}
    ]

async def fetch_user_with_posts(user_id: int) -> dict:
    """Kullanıcı ve gönderilerini birlikte çeker"""
    # Sıralı çalıştırma
    # user = await fetch_user(user_id)
    # posts = await fetch_user_posts(user_id)

    # Paralel çalıştırma (daha hızlı)
    user, posts = await asyncio.gather(
        fetch_user(user_id),
        fetch_user_posts(user_id)
    )

    return {**user, "posts": posts}

# Coroutine zincirleme
async def step1():
    """İlk adım"""
    print("Adım 1 çalışıyor")
    await asyncio.sleep(0.5)
    return "Adım 1 tamamlandı"

async def step2(previous_result: str):
    """İkinci adım"""
    print(f"Adım 2 çalışıyor (önceki: {previous_result})")
    await asyncio.sleep(0.5)
    return "Adım 2 tamamlandı"

async def step3(previous_result: str):
    """Üçüncü adım"""
    print(f"Adım 3 çalışıyor (önceki: {previous_result})")
    await asyncio.sleep(0.5)
    return "Adım 3 tamamlandı"

async def chained_coroutines():
    """Zincirleme coroutine'ler"""
    result1 = await step1()
    result2 = await step2(result1)
    result3 = await step3(result2)
    return result3

# Coroutine generator pattern
async def async_range(start: int, stop: int, step: int = 1):
    """Async range generator"""
    current = start
    while current < stop:
        await asyncio.sleep(0.1)  # Her değer üretiminde bekle
        yield current
        current += step

async def use_async_generator():
    """Async generator kullanımı"""
    print("Async range kullanımı:")
    async for num in async_range(0, 5):
        print(f"Değer: {num}")

async def main():
    print("=== Coroutine İnceleme ===")
    inspect_coroutine()

    print("\n=== Nested Coroutines ===")
    user_data = await fetch_user_with_posts(1)
    print(f"Kullanıcı verisi: {user_data}")

    print("\n=== Chained Coroutines ===")
    final_result = await chained_coroutines()
    print(f"Final sonuç: {final_result}")

    print("\n=== Async Generator ===")
    await use_async_generator()

if __name__ == "__main__":
    asyncio.run(main())
```

### Örnek 5: Coroutine İletişimi ve Koordinasyon

```python
import asyncio
from typing import List
import random

# Producer-Consumer Pattern
async def producer(queue: asyncio.Queue, producer_id: int):
    """Veri üreten coroutine"""
    for i in range(5):
        item = f"P{producer_id}-Item{i}"
        await asyncio.sleep(random.uniform(0.1, 0.5))
        await queue.put(item)
        print(f"Producer {producer_id} üretti: {item}")

    # İş bittiğinde None gönder
    await queue.put(None)

async def consumer(queue: asyncio.Queue, consumer_id: int):
    """Veri tüketen coroutine"""
    while True:
        item = await queue.get()

        # None gelirse bitir
        if item is None:
            await queue.put(None)  # Diğer consumer'lar için
            break

        # İşlemi simüle et
        await asyncio.sleep(random.uniform(0.1, 0.3))
        print(f"Consumer {consumer_id} işledi: {item}")
        queue.task_done()

async def producer_consumer_pattern():
    """Producer-Consumer pattern örneği"""
    queue = asyncio.Queue(maxsize=10)

    # Producer ve consumer'ları başlat
    producers = [
        asyncio.create_task(producer(queue, i))
        for i in range(2)
    ]

    consumers = [
        asyncio.create_task(consumer(queue, i))
        for i in range(3)
    ]

    # Producer'ların bitmesini bekle
    await asyncio.gather(*producers)

    # Queue'nun boşalmasını bekle
    await queue.join()

    # Consumer'ları durdur
    await queue.put(None)
    await asyncio.gather(*consumers)

    print("Producer-Consumer işlemi tamamlandı")

async def main():
    await producer_consumer_pattern()

if __name__ == "__main__":
    asyncio.run(main())
```

---

## Event Loop (Olay Döngüsü)

**Event loop**, asenkron işlemlerin çalıştırılmasını yöneten temel mekanizmadır. Task'ları planlar, I/O işlemlerini izler ve callback'leri çağırır.

### Örnek 6: Event Loop Yönetimi

```python
import asyncio
import signal
import sys
from typing import Optional

# Event loop'a erişim
async def get_loop_info():
    """Event loop bilgilerini göster"""
    loop = asyncio.get_running_loop()

    print(f"Loop tipi: {type(loop)}")
    print(f"Running: {loop.is_running()}")
    print(f"Closed: {loop.is_closed()}")
    print(f"Debug mode: {loop.get_debug()}")

# Callback ile çalışma
def callback_function(future: asyncio.Future):
    """Callback fonksiyonu"""
    print(f"Callback çağrıldı! Sonuç: {future.result()}")

async def callback_example():
    """Callback ile çalışma örneği"""
    loop = asyncio.get_running_loop()

    # Future oluştur
    future = loop.create_future()

    # Callback ekle
    future.add_done_callback(callback_function)

    # 1 saniye sonra sonucu set et
    loop.call_later(1, future.set_result, "Tamamlandı!")

    # Future'ı bekle
    result = await future
    return result

# Call soon, call later, call at
async def scheduling_example():
    """Task zamanlama örneği"""
    loop = asyncio.get_running_loop()

    def scheduled_callback(name: str):
        print(f"[{asyncio.get_event_loop().time():.2f}] {name} çağrıldı")

    # Hemen çağır (sonraki iterasyonda)
    loop.call_soon(scheduled_callback, "call_soon")

    # 1 saniye sonra çağır
    loop.call_later(1, scheduled_callback, "call_later (1s)")

    # 2 saniye sonra çağır
    loop.call_later(2, scheduled_callback, "call_later (2s)")

    # Belirli bir zamanda çağır
    loop.call_at(loop.time() + 3, scheduled_callback, "call_at (3s)")

    # Callback'lerin çalışması için bekle
    await asyncio.sleep(4)

# Custom event loop policy
class CustomEventLoopPolicy(asyncio.DefaultEventLoopPolicy):
    """Özel event loop policy"""

    def new_event_loop(self):
        loop = super().new_event_loop()
        print("Yeni event loop oluşturuldu")
        return loop

# Signal handling
async def signal_handler_example():
    """Signal handling örneği"""
    loop = asyncio.get_running_loop()

    def handle_signal(sig):
        print(f"\nSignal alındı: {sig.name}")
        print("Temiz kapanış yapılıyor...")
        # Loop'u durdur
        loop.stop()

    # SIGINT (Ctrl+C) ve SIGTERM için handler ekle
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, handle_signal, sig)

    print("Ctrl+C ile programı durdurun...")

    try:
        # Sürekli çalışan bir task
        while True:
            await asyncio.sleep(1)
            print("Çalışıyor...")
    except asyncio.CancelledError:
        print("Task iptal edildi")

# Threading ile event loop
import threading
import time

def run_async_in_thread(coro):
    """Ayrı thread'de async fonksiyon çalıştır"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()

async def async_task_in_thread():
    """Thread'de çalışacak async task"""
    print(f"Async task çalışıyor (thread: {threading.current_thread().name})")
    await asyncio.sleep(1)
    return "Thread'den sonuç"

def threading_example():
    """Threading ile event loop örneği"""
    # Ana thread
    print(f"Ana thread: {threading.current_thread().name}")

    # Ayrı thread'de async task çalıştır
    thread = threading.Thread(
        target=lambda: print(run_async_in_thread(async_task_in_thread())),
        name="AsyncThread"
    )
    thread.start()
    thread.join()

# Event loop performans izleme
async def performance_monitoring():
    """Event loop performansını izle"""
    loop = asyncio.get_running_loop()

    # Debug mode'u aç
    loop.set_debug(True)

    async def slow_task():
        """Yavaş task (uyarı üretir)"""
        await asyncio.sleep(0.2)  # 100ms'den uzun task'lar uyarı üretir

    # Yavaş task'ı çalıştır
    await slow_task()

async def main():
    print("=== Event Loop Info ===")
    await get_loop_info()

    print("\n=== Callback Example ===")
    result = await callback_example()
    print(f"Sonuç: {result}")

    print("\n=== Scheduling Example ===")
    await scheduling_example()

    print("\n=== Threading Example ===")
    threading_example()

    # Signal handler örneğini manuel olarak test edin
    # print("\n=== Signal Handler Example ===")
    # await signal_handler_example()

if __name__ == "__main__":
    asyncio.run(main())
```

---

## Tasks ve Futures

**Task'lar**, coroutine'leri event loop'ta çalışan nesneler haline getirir. **Future'lar** ise henüz tamamlanmamış bir işlemin sonucunu temsil eder.

### Örnek 7: Task Yönetimi

```python
import asyncio
from typing import List, Optional, Set
import time

# Task oluşturma ve yönetme
async def monitored_task(name: str, duration: float) -> str:
    """İzlenebilir task"""
    print(f"[{time.strftime('%H:%M:%S')}] {name} başladı")

    try:
        await asyncio.sleep(duration)
        result = f"{name} başarıyla tamamlandı ({duration}s)"
        print(f"[{time.strftime('%H:%M:%S')}] {result}")
        return result
    except asyncio.CancelledError:
        print(f"[{time.strftime('%H:%M:%S')}] {name} iptal edildi")
        raise

async def task_management_example():
    """Task yönetimi örneği"""
    # Task oluştur
    task1 = asyncio.create_task(monitored_task("Task-1", 2), name="task-1")
    task2 = asyncio.create_task(monitored_task("Task-2", 3), name="task-2")
    task3 = asyncio.create_task(monitored_task("Task-3", 1), name="task-3")

    # Task bilgilerini göster
    print(f"Task-1 name: {task1.get_name()}")
    print(f"Task-1 done: {task1.done()}")
    print(f"Task-1 cancelled: {task1.cancelled()}")

    # Task'ların tamamlanmasını bekle
    results = await asyncio.gather(task1, task2, task3)
    print(f"Tüm task'lar tamamlandı: {results}")

# Task iptal etme
async def cancellable_task(name: str):
    """İptal edilebilir task"""
    try:
        for i in range(10):
            print(f"{name}: adım {i+1}")
            await asyncio.sleep(0.5)
        return f"{name} tamamlandı"
    except asyncio.CancelledError:
        print(f"{name} iptal edildi!")
        # Cleanup işlemleri
        await asyncio.sleep(0.1)
        print(f"{name} cleanup tamamlandı")
        raise  # CancelledError'ı yeniden fırlat

async def cancellation_example():
    """Task iptali örneği"""
    task = asyncio.create_task(cancellable_task("Cancellable"))

    # 2 saniye bekle
    await asyncio.sleep(2)

    # Task'ı iptal et
    task.cancel()

    try:
        await task
    except asyncio.CancelledError:
        print("Task başarıyla iptal edildi")

# Task grubu yönetimi
class TaskGroup:
    """Task grubu yöneticisi"""

    def __init__(self, name: str):
        self.name = name
        self.tasks: Set[asyncio.Task] = set()

    def create_task(self, coro) -> asyncio.Task:
        """Yeni task oluştur ve gruba ekle"""
        task = asyncio.create_task(coro)
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)
        return task

    async def wait_all(self, timeout: Optional[float] = None):
        """Tüm task'ların bitmesini bekle"""
        if not self.tasks:
            return []

        try:
            return await asyncio.wait_for(
                asyncio.gather(*self.tasks),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            print(f"{self.name}: Timeout! Kalan task'lar iptal ediliyor...")
            for task in self.tasks:
                task.cancel()
            raise

    def cancel_all(self):
        """Tüm task'ları iptal et"""
        for task in self.tasks:
            task.cancel()

    def __len__(self):
        return len(self.tasks)

async def task_group_example():
    """Task grubu örneği"""
    group = TaskGroup("MyGroup")

    # Grup'a task'lar ekle
    group.create_task(monitored_task("A", 1))
    group.create_task(monitored_task("B", 2))
    group.create_task(monitored_task("C", 3))

    print(f"Grup'ta {len(group)} task var")

    # Tüm task'ların bitmesini bekle
    results = await group.wait_all(timeout=5)
    print(f"Tüm task'lar tamamlandı: {len(results)} sonuç")

# Task sonuçlarını işleme
async def task_results_example():
    """Task sonuçlarını işleme"""
    # Task'ları oluştur
    tasks = [
        asyncio.create_task(monitored_task(f"Task-{i}", i * 0.5))
        for i in range(1, 6)
    ]

    # Tamamlandıkça işle
    for coro in asyncio.as_completed(tasks):
        result = await coro
        print(f"Sonuç alındı: {result}")

# Future örneği
async def future_example():
    """Future kullanımı"""
    loop = asyncio.get_running_loop()

    # Future oluştur
    future = loop.create_future()

    async def set_future_result():
        """Future'a sonuç set et"""
        await asyncio.sleep(1)
        future.set_result("Future sonucu!")

    # Future'a sonuç set edecek task oluştur
    asyncio.create_task(set_future_result())

    # Future'ı bekle
    print("Future bekleniyor...")
    result = await future
    print(f"Future sonucu: {result}")

# Task exception handling
async def failing_task(name: str):
    """Hata veren task"""
    await asyncio.sleep(0.5)
    raise ValueError(f"{name} bir hata verdi!")

async def exception_handling_example():
    """Task exception handling"""
    tasks = [
        asyncio.create_task(monitored_task("Success-1", 1)),
        asyncio.create_task(failing_task("Failure")),
        asyncio.create_task(monitored_task("Success-2", 1)),
    ]

    # return_exceptions=True ile hataları yakala
    results = await asyncio.gather(*tasks, return_exceptions=True)

    for i, result in enumerate(results):
        if isinstance(result, Exception):
            print(f"Task {i} hata verdi: {result}")
        else:
            print(f"Task {i} başarılı: {result}")

async def main():
    print("=== Task Management ===")
    await task_management_example()

    print("\n=== Task Cancellation ===")
    await cancellation_example()

    print("\n=== Task Group ===")
    await task_group_example()

    print("\n=== Task Results ===")
    await task_results_example()

    print("\n=== Future Example ===")
    await future_example()

    print("\n=== Exception Handling ===")
    await exception_handling_example()

if __name__ == "__main__":
    asyncio.run(main())
```

---

## Async Context Managers

**Async context manager'lar**, `async with` ifadesiyle kullanılan ve asenkron setup/cleanup işlemleri yapan nesnelerdir. `__aenter__` ve `__aexit__` metotlarını implemente ederler.

### Örnek 8: Async Context Manager

```python
import asyncio
import aiohttp
import aiofiles
from typing import Optional
import time

# Basit async context manager
class AsyncTimer:
    """Async işlemlerin süresini ölçen context manager"""

    def __init__(self, name: str):
        self.name = name
        self.start_time: Optional[float] = None

    async def __aenter__(self):
        """Context manager başlangıcı"""
        self.start_time = time.time()
        print(f"[{self.name}] Başladı")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager bitişi"""
        elapsed = time.time() - self.start_time
        print(f"[{self.name}] Bitti ({elapsed:.2f}s)")

        # Exception'ı suppress etme
        return False

async def timer_example():
    """AsyncTimer kullanımı"""
    async with AsyncTimer("İşlem-1"):
        await asyncio.sleep(1)
        print("İşlem çalışıyor...")

# Database connection context manager
class AsyncDatabaseConnection:
    """Asenkron veritabanı bağlantısı (simüle edilmiş)"""

    def __init__(self, host: str, database: str):
        self.host = host
        self.database = database
        self.connection = None

    async def __aenter__(self):
        """Bağlantıyı aç"""
        print(f"Bağlantı açılıyor: {self.host}/{self.database}")
        await asyncio.sleep(0.5)  # Bağlantı simülasyonu
        self.connection = f"Connection<{self.host}/{self.database}>"
        print("Bağlantı açıldı")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Bağlantıyı kapat"""
        print("Bağlantı kapatılıyor...")
        await asyncio.sleep(0.2)  # Kapatma simülasyonu
        self.connection = None
        print("Bağlantı kapatıldı")

        if exc_type:
            print(f"Hata ile kapatıldı: {exc_type.__name__}")

        return False

    async def query(self, sql: str) -> list:
        """SQL sorgusu çalıştır"""
        if not self.connection:
            raise RuntimeError("Bağlantı açık değil")

        print(f"Sorgu çalıştırılıyor: {sql}")
        await asyncio.sleep(0.3)
        return [{"id": 1, "name": "Test"}]

async def database_example():
    """Database context manager kullanımı"""
    async with AsyncDatabaseConnection("localhost", "mydb") as db:
        results = await db.query("SELECT * FROM users")
        print(f"Sonuçlar: {results}")

# File operations context manager
class AsyncFileManager:
    """Async dosya işlemleri manager"""

    def __init__(self, filename: str, mode: str = 'r'):
        self.filename = filename
        self.mode = mode
        self.file = None

    async def __aenter__(self):
        """Dosyayı aç"""
        print(f"Dosya açılıyor: {self.filename}")
        self.file = await aiofiles.open(self.filename, self.mode)
        return self.file

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Dosyayı kapat"""
        print(f"Dosya kapatılıyor: {self.filename}")
        if self.file:
            await self.file.close()
        return False

async def file_example():
    """Async file işlemleri"""
    # Dosyaya yaz
    async with AsyncFileManager("/tmp/test_async.txt", "w") as f:
        await f.write("Async file operations\n")
        await f.write("Line 2\n")

    # Dosyadan oku
    async with AsyncFileManager("/tmp/test_async.txt", "r") as f:
        content = await f.read()
        print(f"Dosya içeriği:\n{content}")

# HTTP session context manager
class AsyncHTTPSession:
    """HTTP session manager"""

    def __init__(self, base_url: str, timeout: int = 10):
        self.base_url = base_url
        self.timeout = timeout
        self.session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        """Session oluştur"""
        print("HTTP session oluşturuluyor...")
        self.session = aiohttp.ClientSession(
            base_url=self.base_url,
            timeout=aiohttp.ClientTimeout(total=self.timeout)
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Session'ı kapat"""
        print("HTTP session kapatılıyor...")
        if self.session:
            await self.session.close()
        return False

    async def get(self, path: str) -> dict:
        """GET isteği"""
        if not self.session:
            raise RuntimeError("Session açık değil")

        async with self.session.get(path) as response:
            return {
                "status": response.status,
                "data": await response.json()
            }

async def http_session_example():
    """HTTP session kullanımı"""
    async with AsyncHTTPSession("https://jsonplaceholder.typicode.com") as session:
        result = await session.get("/posts/1")
        print(f"Status: {result['status']}")
        print(f"Data: {result['data']}")

# Resource pool context manager
class AsyncResourcePool:
    """Async kaynak havuzu"""

    def __init__(self, name: str, size: int):
        self.name = name
        self.size = size
        self.semaphore = asyncio.Semaphore(size)
        self.active_count = 0

    async def __aenter__(self):
        """Kaynak al"""
        await self.semaphore.acquire()
        self.active_count += 1
        print(f"[{self.name}] Kaynak alındı (aktif: {self.active_count}/{self.size})")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Kaynağı serbest bırak"""
        self.active_count -= 1
        self.semaphore.release()
        print(f"[{self.name}] Kaynak serbest bırakıldı (aktif: {self.active_count}/{self.size})")
        return False

async def resource_pool_example():
    """Resource pool kullanımı"""
    pool = AsyncResourcePool("DBPool", 2)

    async def use_resource(worker_id: int):
        """Kaynak kullan"""
        async with pool:
            print(f"Worker {worker_id} çalışıyor")
            await asyncio.sleep(1)
            print(f"Worker {worker_id} bitti")

    # 5 worker oluştur ama sadece 2'si aynı anda çalışabilir
    await asyncio.gather(*[use_resource(i) for i in range(5)])

# contextlib.asynccontextmanager decorator
from contextlib import asynccontextmanager

@asynccontextmanager
async def async_resource(name: str):
    """Decorator ile async context manager"""
    print(f"[{name}] Kaynak alınıyor")
    await asyncio.sleep(0.1)

    try:
        yield name
    finally:
        print(f"[{name}] Kaynak temizleniyor")
        await asyncio.sleep(0.1)

async def decorator_example():
    """Decorator ile context manager"""
    async with async_resource("MyResource") as resource:
        print(f"Kaynak kullanılıyor: {resource}")
        await asyncio.sleep(0.5)

async def main():
    print("=== Timer Example ===")
    await timer_example()

    print("\n=== Database Example ===")
    await database_example()

    print("\n=== File Example ===")
    await file_example()

    print("\n=== HTTP Session Example ===")
    await http_session_example()

    print("\n=== Resource Pool Example ===")
    await resource_pool_example()

    print("\n=== Decorator Example ===")
    await decorator_example()

if __name__ == "__main__":
    asyncio.run(main())
```

---

## Async Iterators ve Generators

**Async iterator'lar** ve **generator'lar**, asenkron olarak veri akışı sağlar. `__aiter__` ve `__anext__` metotlarıyla implement edilir veya `async def` ile `yield` kullanılır.

### Örnek 9: Async Iterators

```python
import asyncio
from typing import AsyncIterator, List
import aiohttp

# Async generator
async def async_counter(start: int, end: int, delay: float = 0.5):
    """Async sayaç generator"""
    for i in range(start, end):
        await asyncio.sleep(delay)
        yield i

async def async_generator_example():
    """Async generator kullanımı"""
    print("Async counter:")
    async for num in async_counter(1, 6, 0.3):
        print(f"Sayı: {num}")

# Async iterator class
class AsyncRange:
    """Async range iterator"""

    def __init__(self, start: int, end: int, step: int = 1):
        self.current = start
        self.end = end
        self.step = step

    def __aiter__(self):
        """Iterator döndür"""
        return self

    async def __anext__(self):
        """Sonraki değeri döndür"""
        if self.current >= self.end:
            raise StopAsyncIteration

        await asyncio.sleep(0.1)
        value = self.current
        self.current += self.step
        return value

async def async_iterator_example():
    """Async iterator kullanımı"""
    print("Async range:")
    async for num in AsyncRange(0, 10, 2):
        print(f"Değer: {num}")

# Async file reader
class AsyncFileReader:
    """Satır satır async dosya okuyucu"""

    def __init__(self, filename: str):
        self.filename = filename
        self.file = None

    async def __aenter__(self):
        """Context manager entry"""
        import aiofiles
        self.file = await aiofiles.open(self.filename, 'r')
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        if self.file:
            await self.file.close()
        return False

    def __aiter__(self):
        """Iterator döndür"""
        return self

    async def __anext__(self):
        """Sonraki satırı oku"""
        if not self.file:
            raise StopAsyncIteration

        line = await self.file.readline()
        if not line:
            raise StopAsyncIteration

        return line.strip()

async def file_reader_example():
    """Async file reader kullanımı"""
    # Test dosyası oluştur
    import aiofiles
    test_file = "/tmp/async_test.txt"

    async with aiofiles.open(test_file, 'w') as f:
        for i in range(5):
            await f.write(f"Line {i+1}\n")

    # Dosyayı oku
    print("Dosya içeriği:")
    async with AsyncFileReader(test_file) as reader:
        async for line in reader:
            print(f"  {line}")

# Async data fetcher
class AsyncDataFetcher:
    """API'den async veri çeken iterator"""

    def __init__(self, base_url: str, start_id: int, end_id: int):
        self.base_url = base_url
        self.current_id = start_id
        self.end_id = end_id
        self.session = None

    async def __aenter__(self):
        """Session oluştur"""
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Session'ı kapat"""
        if self.session:
            await self.session.close()
        return False

    def __aiter__(self):
        return self

    async def __anext__(self):
        """Sonraki veriyi çek"""
        if self.current_id > self.end_id:
            raise StopAsyncIteration

        if not self.session:
            raise RuntimeError("Session açık değil")

        url = f"{self.base_url}/{self.current_id}"
        async with self.session.get(url) as response:
            data = await response.json()
            self.current_id += 1
            return data

async def data_fetcher_example():
    """Async data fetcher kullanımı"""
    print("API verilerini çekme:")
    async with AsyncDataFetcher(
        "https://jsonplaceholder.typicode.com/posts",
        1, 3
    ) as fetcher:
        async for data in fetcher:
            print(f"  Post {data['id']}: {data['title'][:50]}...")

# Async stream processor
async def async_stream_processor(data_stream: AsyncIterator[int]) -> AsyncIterator[int]:
    """Async stream işleyici"""
    async for item in data_stream:
        # İşleme yap
        await asyncio.sleep(0.1)
        processed = item * 2
        yield processed

async def stream_processor_example():
    """Stream processor kullanımı"""
    print("Stream processing:")

    # Veri akışı oluştur
    stream = async_counter(1, 6, 0.1)

    # İşle
    processed_stream = async_stream_processor(stream)

    # Sonuçları al
    async for result in processed_stream:
        print(f"İşlenmiş: {result}")

# Async comprehension
async def async_comprehension_example():
    """Async comprehension kullanımı"""
    # Async list comprehension
    squares = [i**2 async for i in async_counter(1, 6, 0.1)]
    print(f"Kareler: {squares}")

    # Async list comprehension with condition
    evens = [i async for i in async_counter(1, 11, 0.05) if i % 2 == 0]
    print(f"Çift sayılar: {evens}")

    # Async dict comprehension
    square_dict = {i: i**2 async for i in async_counter(1, 6, 0.05)}
    print(f"Kare dictionary: {square_dict}")

# Async generator with send
async def async_generator_with_send():
    """Send destekleyen async generator"""
    value = 0
    while True:
        received = yield value
        if received is not None:
            value = received
        else:
            value += 1
        await asyncio.sleep(0.1)

async def generator_send_example():
    """Generator send kullanımı"""
    print("Generator send:")
    gen = async_generator_with_send()

    print(await gen.asend(None))  # İlk değer: 0
    print(await gen.asend(None))  # 1
    print(await gen.asend(10))    # 10 gönder
    print(await gen.asend(None))  # 11

    await gen.aclose()

# Async batch processor
async def async_batch_generator(items: List, batch_size: int):
    """Batch'ler halinde veri döndüren generator"""
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        await asyncio.sleep(0.1)  # Batch işleme simülasyonu
        yield batch

async def batch_processor_example():
    """Batch processor kullanımı"""
    print("Batch processing:")
    items = list(range(1, 21))

    async for batch in async_batch_generator(items, 5):
        print(f"Batch: {batch}")

async def main():
    print("=== Async Generator ===")
    await async_generator_example()

    print("\n=== Async Iterator ===")
    await async_iterator_example()

    print("\n=== File Reader ===")
    await file_reader_example()

    print("\n=== Data Fetcher ===")
    await data_fetcher_example()

    print("\n=== Stream Processor ===")
    await stream_processor_example()

    print("\n=== Async Comprehension ===")
    await async_comprehension_example()

    print("\n=== Generator Send ===")
    await generator_send_example()

    print("\n=== Batch Processor ===")
    await batch_processor_example()

if __name__ == "__main__":
    asyncio.run(main())
```

---

## Concurrency Patterns

**Concurrency pattern'leri**, asenkron programlamada yaygın kullanılan yapılardır. Paralel işlem, rate limiting, circuit breaker gibi kalıpları içerir.

### Örnek 10: Rate Limiting

```python
import asyncio
import time
from typing import Callable, Any
from collections import deque

class RateLimiter:
    """Rate limiter (istek hızı sınırlayıcı)"""

    def __init__(self, max_calls: int, time_window: float):
        """
        Args:
            max_calls: Zaman penceresi içinde maksimum çağrı sayısı
            time_window: Zaman penceresi (saniye)
        """
        self.max_calls = max_calls
        self.time_window = time_window
        self.calls = deque()
        self.lock = asyncio.Lock()

    async def __aenter__(self):
        """Rate limiter ile korunan bölgeye gir"""
        async with self.lock:
            now = time.time()

            # Eski çağrıları temizle
            while self.calls and self.calls[0] < now - self.time_window:
                self.calls.popleft()

            # Limit aşılmış mı kontrol et
            if len(self.calls) >= self.max_calls:
                # Bekleme süresi hesapla
                sleep_time = self.calls[0] + self.time_window - now
                print(f"Rate limit! {sleep_time:.2f}s bekleniyor...")
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

async def rate_limiter_example():
    """Rate limiter kullanımı"""
    # Saniyede maksimum 3 çağrı
    limiter = RateLimiter(max_calls=3, time_window=1.0)

    async def api_call(call_id: int):
        """API çağrısı simülasyonu"""
        async with limiter:
            print(f"[{time.strftime('%H:%M:%S')}] API call {call_id}")
            await asyncio.sleep(0.1)

    # 10 çağrı yap (rate limit devreye girer)
    await asyncio.gather(*[api_call(i) for i in range(10)])

async def main():
    await rate_limiter_example()

if __name__ == "__main__":
    asyncio.run(main())
```

### Örnek 11: Circuit Breaker Pattern

```python
import asyncio
import time
from enum import Enum
from typing import Callable, Any

class CircuitState(Enum):
    """Circuit breaker durumları"""
    CLOSED = "closed"      # Normal çalışma
    OPEN = "open"          # Hata durumu, istekleri engelle
    HALF_OPEN = "half_open"  # Test durumu

class CircuitBreaker:
    """Circuit breaker pattern implementasyonu"""

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
        self.last_failure_time = None
        self.state = CircuitState.CLOSED

    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """Korumalı fonksiyon çağrısı"""
        # Circuit açık mı kontrol et
        if self.state == CircuitState.OPEN:
            # Recovery timeout doldu mu?
            if time.time() - self.last_failure_time >= self.recovery_timeout:
                print("Circuit HALF_OPEN durumuna geçti (test)")
                self.state = CircuitState.HALF_OPEN
            else:
                raise Exception("Circuit OPEN! İstekler engelleniyor.")

        try:
            # Fonksiyonu çağır
            result = await func(*args, **kwargs)

            # Başarılı - circuit'i sıfırla
            if self.state == CircuitState.HALF_OPEN:
                print("Circuit CLOSED durumuna döndü (başarılı test)")
                self.state = CircuitState.CLOSED
                self.failure_count = 0

            return result

        except self.expected_exception as e:
            # Hata - sayacı artır
            self.failure_count += 1
            self.last_failure_time = time.time()

            print(f"Hata #{self.failure_count}: {e}")

            # Threshold aşıldı mı?
            if self.failure_count >= self.failure_threshold:
                print(f"Circuit OPEN oldu! (threshold: {self.failure_threshold})")
                self.state = CircuitState.OPEN

            raise

async def circuit_breaker_example():
    """Circuit breaker kullanımı"""
    circuit = CircuitBreaker(
        failure_threshold=3,
        recovery_timeout=5.0
    )

    # Hata veren servis simülasyonu
    failure_mode = True

    async def unreliable_service():
        """Güvenilmez servis"""
        if failure_mode:
            raise Exception("Servis hatası!")
        return "Başarılı"

    # Birden fazla çağrı yap
    for i in range(10):
        try:
            result = await circuit.call(unreliable_service)
            print(f"Çağrı {i+1}: {result}")
        except Exception as e:
            print(f"Çağrı {i+1} başarısız: {e}")

        await asyncio.sleep(1)

        # 5. çağrıdan sonra servisi düzelt
        if i == 5:
            failure_mode = False
            print("\n>>> Servis düzeltildi <<<\n")

async def main():
    await circuit_breaker_example()

if __name__ == "__main__":
    asyncio.run(main())
```

### Örnek 12: Worker Pool Pattern

```python
import asyncio
from typing import Callable, Any, List
import random

class WorkerPool:
    """Worker pool pattern - Sınırlı sayıda worker ile görevleri işle"""

    def __init__(self, num_workers: int):
        self.num_workers = num_workers
        self.queue = asyncio.Queue()
        self.workers = []

    async def worker(self, worker_id: int):
        """Worker coroutine"""
        while True:
            # Kuyruktan görev al
            task = await self.queue.get()

            # Poison pill kontrolü (durdurma sinyali)
            if task is None:
                self.queue.task_done()
                break

            func, args, kwargs = task

            try:
                print(f"Worker-{worker_id} görevi işliyor...")
                result = await func(*args, **kwargs)
                print(f"Worker-{worker_id} tamamladı: {result}")
            except Exception as e:
                print(f"Worker-{worker_id} hata aldı: {e}")
            finally:
                self.queue.task_done()

    async def start(self):
        """Worker'ları başlat"""
        self.workers = [
            asyncio.create_task(self.worker(i))
            for i in range(self.num_workers)
        ]

    async def submit(self, func: Callable, *args, **kwargs):
        """Görevi kuyruğa ekle"""
        await self.queue.put((func, args, kwargs))

    async def join(self):
        """Tüm görevlerin bitmesini bekle"""
        await self.queue.join()

    async def stop(self):
        """Worker'ları durdur"""
        # Her worker için poison pill gönder
        for _ in range(self.num_workers):
            await self.queue.put(None)

        # Worker'ların bitmesini bekle
        await asyncio.gather(*self.workers)

async def process_item(item_id: int) -> str:
    """İşlem simülasyonu"""
    delay = random.uniform(0.5, 2.0)
    await asyncio.sleep(delay)
    return f"Item-{item_id} işlendi ({delay:.2f}s)"

async def worker_pool_example():
    """Worker pool kullanımı"""
    # 3 worker'lı pool oluştur
    pool = WorkerPool(num_workers=3)

    # Worker'ları başlat
    await pool.start()

    # 10 görev gönder
    for i in range(10):
        await pool.submit(process_item, i)

    # Tüm görevlerin bitmesini bekle
    await pool.join()

    # Pool'u durdur
    await pool.stop()

    print("Tüm görevler tamamlandı!")

async def main():
    await worker_pool_example()

if __name__ == "__main__":
    asyncio.run(main())
```

### Örnek 13: Semaphore Pattern (Kaynak Sınırlama)

```python
import asyncio
import random

async def semaphore_example():
    """Semaphore ile kaynak sınırlama"""
    # Maksimum 3 eşzamanlı işlem
    semaphore = asyncio.Semaphore(3)

    async def limited_resource(worker_id: int):
        """Sınırlı kaynağa erişen işlem"""
        print(f"Worker-{worker_id} kaynak için bekliyor...")

        async with semaphore:
            print(f"Worker-{worker_id} kaynağı aldı!")
            delay = random.uniform(1, 3)
            await asyncio.sleep(delay)
            print(f"Worker-{worker_id} kaynağı serbest bıraktı ({delay:.2f}s)")

        return f"Worker-{worker_id} tamamlandı"

    # 10 worker oluştur ama sadece 3'ü aynı anda çalışabilir
    results = await asyncio.gather(*[
        limited_resource(i) for i in range(10)
    ])

    print(f"\nTüm işlemler tamamlandı: {len(results)} sonuç")

async def main():
    await semaphore_example()

if __name__ == "__main__":
    asyncio.run(main())
```

---

## Error Handling

Asenkron kodda **hata yönetimi**, senkron koddan farklı yaklaşımlar gerektirir. Task'lar, gather, wait gibi yapılarda hataların nasıl yakalanacağını bilmek önemlidir.

### Örnek 14: Async Exception Handling

```python
import asyncio
from typing import List
import traceback

# Basit exception handling
async def failing_task(task_id: int):
    """Hata veren task"""
    await asyncio.sleep(0.5)
    if task_id % 2 == 0:
        raise ValueError(f"Task-{task_id} başarısız!")
    return f"Task-{task_id} başarılı"

async def basic_exception_handling():
    """Temel exception handling"""
    try:
        result = await failing_task(2)
        print(result)
    except ValueError as e:
        print(f"Hata yakalandı: {e}")

# Gather ile exception handling
async def gather_exception_handling():
    """Gather ile exception handling"""
    # return_exceptions=False (default): İlk exception fırlatılır
    print("=== return_exceptions=False ===")
    try:
        results = await asyncio.gather(
            failing_task(1),
            failing_task(2),
            failing_task(3),
            return_exceptions=False
        )
        print(f"Sonuçlar: {results}")
    except ValueError as e:
        print(f"Gather exception: {e}")

    # return_exceptions=True: Exception'lar sonuç listesinde döner
    print("\n=== return_exceptions=True ===")
    results = await asyncio.gather(
        failing_task(1),
        failing_task(2),
        failing_task(3),
        return_exceptions=True
    )

    for i, result in enumerate(results):
        if isinstance(result, Exception):
            print(f"Task {i} hata: {result}")
        else:
            print(f"Task {i} başarılı: {result}")

# Task exception handling
async def task_exception_handling():
    """Task'larda exception handling"""
    # Task oluştur
    task = asyncio.create_task(failing_task(2))

    # Task'ı beklemeden önce exception oluşur
    await asyncio.sleep(1)

    # Task done kontrolü
    if task.done():
        try:
            result = task.result()
            print(f"Task sonucu: {result}")
        except ValueError as e:
            print(f"Task exception: {e}")

# Exception logging
async def logged_task(task_id: int):
    """Loglanan task"""
    try:
        await asyncio.sleep(0.5)
        if task_id % 2 == 0:
            raise ValueError(f"Task-{task_id} hatası")
        return f"Task-{task_id} başarılı"
    except Exception as e:
        print(f"[ERROR] Task-{task_id}: {e}")
        print(f"[TRACE] {traceback.format_exc()}")
        raise

async def exception_logging():
    """Exception logging örneği"""
    results = await asyncio.gather(
        logged_task(1),
        logged_task(2),
        logged_task(3),
        return_exceptions=True
    )

# Retry pattern
async def retry_async(func, max_retries: int = 3, delay: float = 1.0):
    """Async retry pattern"""
    last_exception = None

    for attempt in range(max_retries):
        try:
            return await func()
        except Exception as e:
            last_exception = e
            print(f"Deneme {attempt + 1}/{max_retries} başarısız: {e}")

            if attempt < max_retries - 1:
                await asyncio.sleep(delay)

    # Tüm denemeler başarısız
    raise last_exception

async def unreliable_operation():
    """Güvenilmez işlem"""
    import random
    await asyncio.sleep(0.5)
    if random.random() < 0.7:  # %70 başarısızlık
        raise ConnectionError("Bağlantı hatası!")
    return "Başarılı!"

async def retry_example():
    """Retry pattern kullanımı"""
    try:
        result = await retry_async(unreliable_operation, max_retries=5, delay=0.5)
        print(f"Sonuç: {result}")
    except Exception as e:
        print(f"Tüm denemeler başarısız: {e}")

# Exception propagation
async def level3():
    """En içteki seviye"""
    await asyncio.sleep(0.1)
    raise RuntimeError("Level 3 hatası!")

async def level2():
    """Orta seviye"""
    await asyncio.sleep(0.1)
    await level3()

async def level1():
    """En dıştaki seviye"""
    await asyncio.sleep(0.1)
    await level2()

async def exception_propagation():
    """Exception propagation örneği"""
    try:
        await level1()
    except RuntimeError as e:
        print(f"Exception yakalandı: {e}")
        print(f"Traceback:\n{traceback.format_exc()}")

# Cleanup on exception
class AsyncResource:
    """Exception durumunda cleanup yapan kaynak"""

    def __init__(self, name: str):
        self.name = name
        self.acquired = False

    async def acquire(self):
        """Kaynağı al"""
        print(f"[{self.name}] Kaynak alınıyor...")
        await asyncio.sleep(0.5)
        self.acquired = True
        print(f"[{self.name}] Kaynak alındı")

    async def release(self):
        """Kaynağı serbest bırak"""
        if self.acquired:
            print(f"[{self.name}] Kaynak serbest bırakılıyor...")
            await asyncio.sleep(0.2)
            self.acquired = False
            print(f"[{self.name}] Kaynak serbest bırakıldı")

    async def __aenter__(self):
        await self.acquire()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.release()
        if exc_type:
            print(f"[{self.name}] Exception ile çıkış: {exc_type.__name__}")
        return False

async def cleanup_example():
    """Cleanup on exception örneği"""
    try:
        async with AsyncResource("DB Connection") as resource:
            print("İşlem yapılıyor...")
            await asyncio.sleep(0.5)
            raise ValueError("İşlem hatası!")
    except ValueError as e:
        print(f"Hata yakalandı: {e}")

async def main():
    print("=== Basic Exception Handling ===")
    await basic_exception_handling()

    print("\n=== Gather Exception Handling ===")
    await gather_exception_handling()

    print("\n=== Task Exception Handling ===")
    await task_exception_handling()

    print("\n=== Exception Logging ===")
    await exception_logging()

    print("\n=== Retry Pattern ===")
    await retry_example()

    print("\n=== Exception Propagation ===")
    await exception_propagation()

    print("\n=== Cleanup Example ===")
    await cleanup_example()

if __name__ == "__main__":
    asyncio.run(main())
```

---

## Production Patterns

Production ortamında kullanılan **gelişmiş async pattern'ler**: timeout, graceful shutdown, health check, connection pooling.

### Örnek 15: Production-Ready Async Application

```python
import asyncio
import signal
import sys
from typing import Optional, List
from dataclasses import dataclass
from datetime import datetime
import logging

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class AppConfig:
    """Uygulama konfigürasyonu"""
    worker_count: int = 4
    shutdown_timeout: float = 30.0
    health_check_interval: float = 5.0
    task_timeout: float = 10.0

class AsyncApplication:
    """Production-ready async application"""

    def __init__(self, config: AppConfig):
        self.config = config
        self.workers: List[asyncio.Task] = []
        self.running = False
        self.shutdown_event = asyncio.Event()

    async def worker(self, worker_id: int):
        """Worker coroutine"""
        logger.info(f"Worker-{worker_id} başlatıldı")

        try:
            while not self.shutdown_event.is_set():
                # İş yap
                await asyncio.sleep(1)
                logger.debug(f"Worker-{worker_id} çalışıyor...")

        except asyncio.CancelledError:
            logger.info(f"Worker-{worker_id} iptal edildi")
            raise
        finally:
            logger.info(f"Worker-{worker_id} durdu")

    async def health_check(self):
        """Sağlık kontrolü"""
        logger.info("Health check başlatıldı")

        while not self.shutdown_event.is_set():
            try:
                # Sağlık kontrolü yap
                active_workers = sum(1 for w in self.workers if not w.done())
                logger.info(f"Health check: {active_workers}/{len(self.workers)} worker aktif")

                # Belirli aralıklarla kontrol et
                await asyncio.sleep(self.config.health_check_interval)

            except asyncio.CancelledError:
                logger.info("Health check iptal edildi")
                raise

    async def start(self):
        """Uygulamayı başlat"""
        logger.info("Uygulama başlatılıyor...")
        self.running = True

        # Worker'ları başlat
        self.workers = [
            asyncio.create_task(self.worker(i))
            for i in range(self.config.worker_count)
        ]

        # Health check'i başlat
        health_check_task = asyncio.create_task(self.health_check())

        # Signal handler'ları ayarla
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, self.signal_handler, sig)

        logger.info("Uygulama başlatıldı")

        # Shutdown sinyalini bekle
        await self.shutdown_event.wait()

        # Graceful shutdown
        await self.shutdown(health_check_task)

    def signal_handler(self, sig):
        """Signal handler"""
        logger.info(f"Signal alındı: {sig.name}")
        self.shutdown_event.set()

    async def shutdown(self, health_check_task: asyncio.Task):
        """Graceful shutdown"""
        logger.info("Graceful shutdown başlatılıyor...")

        # Health check'i iptal et
        health_check_task.cancel()
        try:
            await health_check_task
        except asyncio.CancelledError:
            pass

        # Worker'ları iptal et
        for worker in self.workers:
            worker.cancel()

        # Worker'ların bitmesini bekle (timeout ile)
        try:
            await asyncio.wait_for(
                asyncio.gather(*self.workers, return_exceptions=True),
                timeout=self.config.shutdown_timeout
            )
            logger.info("Tüm worker'lar düzgün şekilde durdu")
        except asyncio.TimeoutError:
            logger.error("Shutdown timeout! Bazı worker'lar durdurulamadı")

        self.running = False
        logger.info("Uygulama durdu")

async def production_app_example():
    """Production app kullanımı"""
    config = AppConfig(
        worker_count=3,
        shutdown_timeout=10.0,
        health_check_interval=2.0
    )

    app = AsyncApplication(config)
    await app.start()

# Connection pooling
class AsyncConnectionPool:
    """Async connection pool"""

    def __init__(self, max_connections: int):
        self.max_connections = max_connections
        self.pool = asyncio.Queue(maxsize=max_connections)
        self.created_connections = 0

    async def create_connection(self):
        """Yeni bağlantı oluştur"""
        await asyncio.sleep(0.5)  # Bağlantı simülasyonu
        self.created_connections += 1
        return f"Connection-{self.created_connections}"

    async def get_connection(self):
        """Pool'dan bağlantı al"""
        try:
            # Pool'da hazır bağlantı var mı?
            return self.pool.get_nowait()
        except asyncio.QueueEmpty:
            # Yeni bağlantı oluştur
            if self.created_connections < self.max_connections:
                return await self.create_connection()
            else:
                # Pool'dan bağlantı beklemelisin
                return await self.pool.get()

    async def release_connection(self, connection):
        """Bağlantıyı pool'a geri ver"""
        await self.pool.put(connection)

    async def close(self):
        """Pool'u kapat"""
        while not self.pool.empty():
            await self.pool.get()
        logger.info("Connection pool kapatıldı")

async def connection_pool_example():
    """Connection pool kullanımı"""
    pool = AsyncConnectionPool(max_connections=3)

    async def use_connection(worker_id: int):
        """Bağlantı kullan"""
        conn = await pool.get_connection()
        logger.info(f"Worker-{worker_id} bağlantı aldı: {conn}")

        await asyncio.sleep(1)

        await pool.release_connection(conn)
        logger.info(f"Worker-{worker_id} bağlantıyı geri verdi: {conn}")

    # 6 worker ama sadece 3 bağlantı
    await asyncio.gather(*[use_connection(i) for i in range(6)])

    await pool.close()

async def main():
    # Connection pool örneğini çalıştır
    print("=== Connection Pool Example ===")
    await connection_pool_example()

    # Production app için Ctrl+C ile durdurun
    # print("\n=== Production App (Ctrl+C ile durdurun) ===")
    # await production_app_example()

if __name__ == "__main__":
    asyncio.run(main())
```

---

## Özet

Bu döküman, Python'da **async programming**'in ileri seviye konularını kapsar:

1. **Async/Await**: Modern asenkron kod yazımı
2. **Asyncio Modülü**: Event loop, task yönetimi, concurrency
3. **Coroutines**: Eş yordamlar ve yaşam döngüsü
4. **Event Loop**: Olay döngüsü yönetimi
5. **Tasks ve Futures**: Asenkron görev yönetimi
6. **Async Context Managers**: Asenkron kaynak yönetimi
7. **Async Iterators**: Asenkron veri akışları
8. **Concurrency Patterns**: Rate limiting, circuit breaker, worker pool
9. **Error Handling**: Asenkron hata yönetimi
10. **Production Patterns**: Production-ready async uygulamalar

Asenkron programlama, I/O-bound işlemlerde yüksek performans sağlar ve modern Python uygulamalarının temel taşıdır.
