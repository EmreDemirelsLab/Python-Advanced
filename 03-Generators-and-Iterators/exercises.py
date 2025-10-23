"""
GENERATORS VE ITERATORS - İLERI SEVİYE EGZERSİZLER
=================================================

Bu dosya, production-ready generator ve iterator pattern'lerini içerir.
Her soru, gerçek dünya senaryolarına dayanır.

Seviye: Medium - Expert
Konular:
- Custom iterators
- Generator functions
- Generator expressions
- yield from
- itertools module
- Memory-efficient processing
- Infinite generators
- Pipeline patterns
- Coroutines
"""

import itertools
import csv
import json
from typing import Iterator, Iterable, Any, TypeVar, Callable
from collections import deque
from datetime import datetime, timedelta
import random

# ============================================================================
# SORU 1: Custom Sliding Window Iterator
# ============================================================================
"""
Bir iterable üzerinde sliding window (kayar pencere) oluşturan bir iterator
class'ı yazın. Bu, time series analysis, moving averages ve pattern matching
için kullanılır.

Gereksinimler:
- Window boyutu parametresi
- Step parametresi (pencere kaç adım kayacak)
- Fill value (eksik değerler için)

Örnek:
    list(SlidingWindow([1,2,3,4,5], window_size=3, step=1))
    → [[1,2,3], [2,3,4], [3,4,5]]
"""

# TODO: SlidingWindow class'ını implement edin


# ÇÖZÜM:
class SlidingWindow:
    """
    Sliding window iterator - time series analysis için optimize edilmiş

    Production kullanımı:
    - Moving averages hesaplama
    - Pattern detection
    - Signal processing
    - N-gram generation
    """

    def __init__(self, iterable: Iterable, window_size: int,
                 step: int = 1, fill_value=None):
        """
        Args:
            iterable: Kaynak veri
            window_size: Pencere boyutu
            step: Her adımda kaç eleman atlayacak
            fill_value: Eksik değerler için dolgu
        """
        if window_size < 1:
            raise ValueError("Window size must be at least 1")
        if step < 1:
            raise ValueError("Step must be at least 1")

        self.iterable = iterable
        self.window_size = window_size
        self.step = step
        self.fill_value = fill_value

    def __iter__(self):
        """Iterator protocol implementation"""
        iterator = iter(self.iterable)
        window = deque(maxlen=self.window_size)

        # İlk pencereyi doldur
        for _ in range(self.window_size):
            try:
                window.append(next(iterator))
            except StopIteration:
                # Eksik değerleri fill_value ile doldur
                window.append(self.fill_value)

        yield list(window)

        # Kalan elemanları işle
        try:
            while True:
                # Step kadar eleman atla ve ekle
                for _ in range(self.step):
                    window.append(next(iterator))
                yield list(window)
        except StopIteration:
            pass

    def __repr__(self):
        return f"SlidingWindow(window_size={self.window_size}, step={self.step})"


# Test
if __name__ == "__main__":
    # Basit kullanım
    data = [1, 2, 3, 4, 5, 6, 7]
    windows = SlidingWindow(data, window_size=3, step=1)
    print("Sliding windows:", list(windows))

    # Moving average hesaplama
    prices = [100, 102, 98, 105, 103, 107, 106]
    moving_avg = [sum(w)/len(w) for w in SlidingWindow(prices, window_size=3)]
    print("Moving averages:", moving_avg)


# ============================================================================
# SORU 2: Memory-Efficient Log File Analyzer
# ============================================================================
"""
Çok büyük log dosyalarını (GB seviyesinde) memory-efficient şekilde analiz
eden bir generator pipeline yazın.

Gereksinimler:
- Chunk chunk okuma (tüm dosya bellekte tutulmamalı)
- Error seviyesine göre filtreleme (ERROR, WARNING, INFO)
- Timestamp parsing
- Error pattern matching
- Özet istatistikler

Log formatı: "2024-01-15 10:30:45 | ERROR | Database connection failed"
"""

# TODO: Log analyzer generator'larını implement edin


# ÇÖZÜM:
def read_log_file_chunks(file_path: str, chunk_size: int = 8192) -> Iterator[str]:
    """
    Dosyayı chunk chunk okur - memory efficient

    Args:
        file_path: Log dosyası yolu
        chunk_size: Her chunk'ın boyutu (bytes)

    Yields:
        Her satır
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        buffer = ""
        while True:
            chunk = file.read(chunk_size)
            if not chunk:
                if buffer:
                    yield buffer
                break

            buffer += chunk
            lines = buffer.split('\n')
            buffer = lines[-1]  # Son incomplete satırı sakla

            for line in lines[:-1]:
                if line.strip():
                    yield line.strip()


def parse_log_line(line: str) -> dict:
    """
    Log satırını parse eder

    Format: "2024-01-15 10:30:45 | ERROR | Database connection failed"
    """
    try:
        parts = line.split('|')
        if len(parts) >= 3:
            return {
                'timestamp': datetime.strptime(parts[0].strip(), '%Y-%m-%d %H:%M:%S'),
                'level': parts[1].strip(),
                'message': parts[2].strip(),
                'raw': line
            }
    except (ValueError, IndexError):
        pass

    return None


def filter_by_level(logs: Iterator[dict], *levels: str) -> Iterator[dict]:
    """Belirtilen seviyelerdeki logları filtreler"""
    levels_set = set(level.upper() for level in levels)
    for log in logs:
        if log and log.get('level') in levels_set:
            yield log


def filter_by_pattern(logs: Iterator[dict], pattern: str) -> Iterator[dict]:
    """Message içinde pattern geçenleri filtreler"""
    pattern_lower = pattern.lower()
    for log in logs:
        if log and pattern_lower in log.get('message', '').lower():
            yield log


def filter_by_time_range(logs: Iterator[dict],
                        start: datetime, end: datetime) -> Iterator[dict]:
    """Belirli zaman aralığındaki logları filtreler"""
    for log in logs:
        if log and start <= log.get('timestamp', datetime.min) <= end:
            yield log


def aggregate_log_stats(logs: Iterator[dict]) -> dict:
    """
    Log istatistiklerini toplar

    Returns:
        İstatistik dictionary
    """
    stats = {
        'total': 0,
        'by_level': {},
        'by_hour': {},
        'error_messages': []
    }

    for log in logs:
        if not log:
            continue

        stats['total'] += 1

        # Level stats
        level = log.get('level', 'UNKNOWN')
        stats['by_level'][level] = stats['by_level'].get(level, 0) + 1

        # Hour stats
        timestamp = log.get('timestamp')
        if timestamp:
            hour = timestamp.strftime('%H:00')
            stats['by_hour'][hour] = stats['by_hour'].get(hour, 0) + 1

        # Error messages topla
        if level == 'ERROR':
            stats['error_messages'].append(log.get('message', ''))

    return stats


# Complete pipeline örneği
def analyze_logs(file_path: str, level: str = None, pattern: str = None) -> dict:
    """
    Complete log analysis pipeline

    Example:
        stats = analyze_logs('app.log', level='ERROR', pattern='database')
    """
    # Pipeline'ı kur
    lines = read_log_file_chunks(file_path)
    parsed = (parse_log_line(line) for line in lines)

    # Filtreleri uygula
    filtered = parsed
    if level:
        filtered = filter_by_level(filtered, level)
    if pattern:
        filtered = filter_by_pattern(filtered, pattern)

    # Aggregate
    return aggregate_log_stats(filtered)


# ============================================================================
# SORU 3: Infinite Prime Number Generator
# ============================================================================
"""
Sonsuz asal sayı üreten bir generator yazın. Sieve of Eratosthenes
algoritmasını kullanarak memory-efficient olmalı.

Gereksinimler:
- Sonsuz asal sayı üretimi
- Memory-efficient (tüm asalları bellekte tutmamalı)
- Optimization için cache
"""

# TODO: Prime generator'ı implement edin


# ÇÖZÜM:
def infinite_primes() -> Iterator[int]:
    """
    Sonsuz asal sayı generator - Sieve of Eratosthenes

    Memory efficient: Sadece gerekli asalları tutar

    Yields:
        Sıradaki asal sayı

    Example:
        primes = infinite_primes()
        first_10 = list(itertools.islice(primes, 10))
        # [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
    """
    yield 2  # İlk asal

    # Bilinen asalları sakla (optimizasyon için)
    known_primes = [2]
    candidate = 3

    while True:
        is_prime = True

        # Sadece sqrt(candidate)'a kadar olan asalları kontrol et
        sqrt_candidate = int(candidate ** 0.5) + 1

        for prime in known_primes:
            if prime > sqrt_candidate:
                break
            if candidate % prime == 0:
                is_prime = False
                break

        if is_prime:
            known_primes.append(candidate)
            yield candidate

        candidate += 2  # Tek sayıları kontrol et


def primes_up_to(n: int) -> Iterator[int]:
    """
    N'e kadar olan asal sayıları üretir

    Args:
        n: Üst limit

    Yields:
        n'e kadar olan asallar
    """
    if n < 2:
        return

    # Sieve of Eratosthenes - klasik implementasyon
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False

    for i in range(2, int(n**0.5) + 1):
        if sieve[i]:
            # i'nin katlarını işaretle
            for j in range(i*i, n + 1, i):
                sieve[j] = False

    # Asal olanları yield et
    for i in range(2, n + 1):
        if sieve[i]:
            yield i


def prime_factorization(n: int) -> Iterator[tuple[int, int]]:
    """
    Bir sayının asal çarpanlarına ayırır

    Args:
        n: Çarpanlarına ayrılacak sayı

    Yields:
        (asal, üs) tuple'ları

    Example:
        list(prime_factorization(60))
        # [(2, 2), (3, 1), (5, 1)]  → 60 = 2² × 3 × 5
    """
    if n < 2:
        return

    # 2'ye bölünme
    count = 0
    while n % 2 == 0:
        count += 1
        n //= 2
    if count > 0:
        yield (2, count)

    # Tek sayılara bölünme
    divisor = 3
    while divisor * divisor <= n:
        count = 0
        while n % divisor == 0:
            count += 1
            n //= divisor
        if count > 0:
            yield (divisor, count)
        divisor += 2

    # Kalan sayı asal ise
    if n > 1:
        yield (n, 1)


# Test
if __name__ == "__main__":
    # İlk 20 asal
    primes = infinite_primes()
    first_20 = list(itertools.islice(primes, 20))
    print("First 20 primes:", first_20)

    # 100'e kadar asallar
    print("Primes up to 100:", list(primes_up_to(100)))

    # Asal çarpanlara ayırma
    print("Prime factors of 60:", list(prime_factorization(60)))


# ============================================================================
# SORU 4: Data Stream Processor with Buffering
# ============================================================================
"""
Real-time data stream'lerini işleyen bir generator pipeline yazın.
Buffering, batching ve windowing desteği olmalı.

Gereksinimler:
- Configurable buffer size
- Time-based ve count-based batching
- Sliding ve tumbling window desteği
- Backpressure handling
"""

# TODO: Stream processor'ları implement edin


# ÇÖZÜM:
def batch_by_count(stream: Iterator, batch_size: int) -> Iterator[list]:
    """
    Stream'i count-based batch'lere böler

    Args:
        stream: Giriş stream
        batch_size: Her batch'teki eleman sayısı

    Yields:
        batch_size elemanlı listeler

    Example:
        stream = range(10)
        batches = batch_by_count(stream, 3)
        # [[0,1,2], [3,4,5], [6,7,8], [9]]
    """
    batch = []
    for item in stream:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []

    # Son batch'i gönder (incomplete olabilir)
    if batch:
        yield batch


def batch_by_time(stream: Iterator,
                  time_window: float,
                  get_timestamp: Callable = None) -> Iterator[list]:
    """
    Stream'i time-based batch'lere böler

    Args:
        stream: Giriş stream
        time_window: Zaman penceresi (saniye)
        get_timestamp: Element'ten timestamp çıkaran function

    Yields:
        Time window içindeki elemanlar
    """
    import time

    if get_timestamp is None:
        get_timestamp = lambda x: time.time()

    batch = []
    window_start = None

    for item in stream:
        current_time = get_timestamp(item)

        if window_start is None:
            window_start = current_time

        # Window doldu mu?
        if current_time - window_start >= time_window:
            if batch:
                yield batch
            batch = [item]
            window_start = current_time
        else:
            batch.append(item)

    # Son batch
    if batch:
        yield batch


def tumbling_window(stream: Iterator, window_size: int) -> Iterator[list]:
    """
    Tumbling window (non-overlapping)

    Her window bir sonraki window ile overlap etmez.

    Example:
        stream = range(10)
        windows = tumbling_window(stream, 3)
        # [[0,1,2], [3,4,5], [6,7,8], [9]]
    """
    window = []
    for item in stream:
        window.append(item)
        if len(window) >= window_size:
            yield window
            window = []

    if window:
        yield window


def sliding_window_generator(stream: Iterator,
                            window_size: int,
                            slide_by: int = 1) -> Iterator[list]:
    """
    Sliding window (overlapping)

    Args:
        stream: Giriş stream
        window_size: Pencere boyutu
        slide_by: Kaç eleman kayacak

    Example:
        stream = range(5)
        windows = sliding_window_generator(stream, 3, 1)
        # [[0,1,2], [1,2,3], [2,3,4]]
    """
    window = deque(maxlen=window_size)

    # İlk pencereyi doldur
    for _ in range(window_size):
        try:
            window.append(next(stream))
        except StopIteration:
            if window:
                yield list(window)
            return

    yield list(window)

    # Kalan elemanları işle
    count = 0
    for item in stream:
        window.append(item)
        count += 1
        if count >= slide_by:
            yield list(window)
            count = 0


def buffered_stream(stream: Iterator,
                   buffer_size: int,
                   process_func: Callable = None) -> Iterator:
    """
    Buffered stream processing - backpressure handling

    Args:
        stream: Giriş stream
        buffer_size: Buffer boyutu
        process_func: Her batch'e uygulanacak fonksiyon

    Yields:
        İşlenmiş elemanlar
    """
    buffer = deque(maxlen=buffer_size)

    for item in stream:
        buffer.append(item)

        # Buffer doldu mu?
        if len(buffer) >= buffer_size:
            batch = list(buffer)
            buffer.clear()

            # Process batch
            if process_func:
                result = process_func(batch)
                if result:
                    yield from result
            else:
                yield from batch

    # Kalan buffer
    if buffer:
        if process_func:
            result = process_func(list(buffer))
            if result:
                yield from result
        else:
            yield from buffer


# Pipeline örneği
def process_sensor_data(data_stream: Iterator[dict]) -> Iterator[dict]:
    """
    Sensor data processing pipeline

    Example:
        sensors = generate_sensor_data()
        processed = process_sensor_data(sensors)
    """
    # 1. Batch by count
    batched = batch_by_count(data_stream, batch_size=10)

    # 2. Her batch'in ortalamasını al
    def calculate_avg(batch):
        avg_value = sum(item['value'] for item in batch) / len(batch)
        return [{
            'timestamp': batch[-1]['timestamp'],
            'avg_value': avg_value,
            'sample_count': len(batch)
        }]

    # 3. Process ve yield
    processed = buffered_stream(batched, buffer_size=5, process_func=calculate_avg)

    return processed


# Test helper
def generate_sensor_data(count: int = 100) -> Iterator[dict]:
    """Test için sensor data generator"""
    import time
    for i in range(count):
        yield {
            'sensor_id': 'TEMP_01',
            'timestamp': time.time(),
            'value': random.uniform(20.0, 25.0)
        }


# ============================================================================
# SORU 5: CSV to JSON Converter Pipeline
# ============================================================================
"""
Büyük CSV dosyalarını JSON'a dönüştüren memory-efficient bir pipeline yazın.

Gereksinimler:
- Streaming processing (tüm dosya bellekte tutulmamalı)
- Schema validation
- Data transformation
- Error handling ve logging
- Progress tracking
"""

# TODO: CSV to JSON converter'ı implement edin


# ÇÖZÜM:
def read_csv_stream(file_path: str,
                   delimiter: str = ',',
                   skip_header: bool = False) -> Iterator[list]:
    """
    CSV dosyasını stream olarak okur

    Args:
        file_path: CSV dosya yolu
        delimiter: Ayırıcı karakter
        skip_header: İlk satırı atla

    Yields:
        Her satır (list olarak)
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        if skip_header:
            next(file)

        for line_num, line in enumerate(file, start=1):
            try:
                fields = [field.strip() for field in line.strip().split(delimiter)]
                yield fields
            except Exception as e:
                print(f"Error at line {line_num}: {e}")
                continue


def csv_to_dict(rows: Iterator[list],
               headers: list) -> Iterator[dict]:
    """
    CSV satırlarını dictionary'ye dönüştürür

    Args:
        rows: CSV satırları
        headers: Sütun isimleri

    Yields:
        Her satır dictionary olarak
    """
    for row in rows:
        if len(row) != len(headers):
            print(f"Warning: Row length mismatch. Expected {len(headers)}, got {len(row)}")
            continue

        yield dict(zip(headers, row))


def validate_schema(records: Iterator[dict],
                   schema: dict) -> Iterator[dict]:
    """
    Record'ları schema'ya göre validate eder

    Args:
        records: Giriş records
        schema: Validation schema
            {
                'field_name': {
                    'type': str/int/float,
                    'required': bool,
                    'min': value,
                    'max': value
                }
            }

    Yields:
        Validate edilmiş records
    """
    for record in records:
        valid = True

        for field_name, rules in schema.items():
            value = record.get(field_name)

            # Required check
            if rules.get('required', False) and value is None:
                print(f"Validation error: {field_name} is required")
                valid = False
                break

            if value is not None:
                # Type check ve conversion
                expected_type = rules.get('type')
                if expected_type:
                    try:
                        record[field_name] = expected_type(value)
                    except (ValueError, TypeError):
                        print(f"Validation error: {field_name} must be {expected_type.__name__}")
                        valid = False
                        break

                # Min/Max check
                if 'min' in rules and record[field_name] < rules['min']:
                    print(f"Validation error: {field_name} below minimum")
                    valid = False
                    break

                if 'max' in rules and record[field_name] > rules['max']:
                    print(f"Validation error: {field_name} above maximum")
                    valid = False
                    break

        if valid:
            yield record


def transform_fields(records: Iterator[dict],
                    transformations: dict) -> Iterator[dict]:
    """
    Field'lara transformation uygular

    Args:
        records: Giriş records
        transformations: Field transformations
            {
                'field_name': transformation_function
            }

    Yields:
        Transform edilmiş records
    """
    for record in records:
        for field_name, transform_func in transformations.items():
            if field_name in record:
                try:
                    record[field_name] = transform_func(record[field_name])
                except Exception as e:
                    print(f"Transformation error on {field_name}: {e}")

        yield record


def write_json_stream(records: Iterator[dict],
                     output_path: str,
                     indent: int = 2) -> int:
    """
    Records'ları JSON dosyasına yazar (streaming)

    Args:
        records: Yazılacak records
        output_path: Çıktı dosyası
        indent: JSON indentation

    Returns:
        Yazılan record sayısı
    """
    count = 0

    with open(output_path, 'w', encoding='utf-8') as file:
        file.write('[\n')

        first = True
        for record in records:
            if not first:
                file.write(',\n')
            first = False

            json_str = json.dumps(record, indent=indent, ensure_ascii=False)
            # İlk satırı indent et
            indented = '\n'.join('  ' + line for line in json_str.split('\n'))
            file.write(indented)
            count += 1

        file.write('\n]')

    return count


# Complete pipeline
def csv_to_json_pipeline(csv_path: str,
                        json_path: str,
                        headers: list,
                        schema: dict = None,
                        transformations: dict = None) -> dict:
    """
    Complete CSV to JSON conversion pipeline

    Args:
        csv_path: Input CSV file
        json_path: Output JSON file
        headers: CSV column names
        schema: Validation schema (optional)
        transformations: Field transformations (optional)

    Returns:
        Statistics dictionary

    Example:
        headers = ['id', 'name', 'age', 'email']
        schema = {
            'id': {'type': int, 'required': True},
            'age': {'type': int, 'min': 0, 'max': 150}
        }
        transformations = {
            'name': str.upper,
            'email': str.lower
        }

        stats = csv_to_json_pipeline(
            'users.csv',
            'users.json',
            headers,
            schema,
            transformations
        )
    """
    import time
    start_time = time.time()

    # Pipeline'ı kur
    rows = read_csv_stream(csv_path, skip_header=True)
    dicts = csv_to_dict(rows, headers)

    # Validation
    if schema:
        dicts = validate_schema(dicts, schema)

    # Transformation
    if transformations:
        dicts = transform_fields(dicts, transformations)

    # Write
    count = write_json_stream(dicts, json_path)

    end_time = time.time()

    return {
        'records_processed': count,
        'time_elapsed': end_time - start_time,
        'records_per_second': count / (end_time - start_time)
    }


# ============================================================================
# SORU 6: Recursive Directory Tree Generator
# ============================================================================
"""
Bir dizin ağacını traverse eden ve dosya bilgilerini yield eden bir generator
yazın. yield from kullanarak recursive traversal yapın.

Gereksinimler:
- Recursive directory traversal
- File metadata (size, modified time, etc.)
- Filter desteği (extension, size, etc.)
- Depth limiting
"""

# TODO: Directory tree generator'ı implement edin


# ÇÖZÜM:
import os
from pathlib import Path
from typing import Optional

def walk_directory(path: str,
                  max_depth: Optional[int] = None,
                  include_dirs: bool = True,
                  filter_func: Optional[Callable] = None,
                  current_depth: int = 0) -> Iterator[dict]:
    """
    Recursive directory traversal generator

    Args:
        path: Başlangıç dizini
        max_depth: Maksimum depth (None = sınırsız)
        include_dirs: Dizinleri de yield et
        filter_func: Filtreleme fonksiyonu
        current_depth: Mevcut depth (internal)

    Yields:
        File/directory bilgileri
        {
            'path': str,
            'name': str,
            'type': 'file' | 'dir',
            'size': int,
            'modified': datetime,
            'depth': int
        }
    """
    if max_depth is not None and current_depth > max_depth:
        return

    try:
        path_obj = Path(path)

        for item in path_obj.iterdir():
            # File bilgilerini topla
            stat = item.stat()
            info = {
                'path': str(item.absolute()),
                'name': item.name,
                'type': 'dir' if item.is_dir() else 'file',
                'size': stat.st_size,
                'modified': datetime.fromtimestamp(stat.st_mtime),
                'depth': current_depth
            }

            # Filter uygula
            if filter_func and not filter_func(info):
                continue

            # Directory ise
            if item.is_dir():
                if include_dirs:
                    yield info

                # Recursive descent
                yield from walk_directory(
                    str(item),
                    max_depth,
                    include_dirs,
                    filter_func,
                    current_depth + 1
                )
            else:
                # Regular file
                yield info

    except PermissionError:
        print(f"Permission denied: {path}")
    except Exception as e:
        print(f"Error accessing {path}: {e}")


def find_files_by_extension(root: str, *extensions: str) -> Iterator[dict]:
    """
    Belirli extension'lara sahip dosyaları bulur

    Example:
        python_files = find_files_by_extension('/project', '.py', '.pyx')
    """
    def filter_extension(info):
        if info['type'] == 'dir':
            return True
        return any(info['name'].endswith(ext) for ext in extensions)

    yield from walk_directory(root, filter_func=filter_extension, include_dirs=False)


def find_large_files(root: str, min_size_mb: float = 10) -> Iterator[dict]:
    """
    Belirli boyuttan büyük dosyaları bulur

    Args:
        root: Kök dizin
        min_size_mb: Minimum dosya boyutu (MB)

    Yields:
        Büyük dosya bilgileri
    """
    min_size_bytes = min_size_mb * 1024 * 1024

    def filter_size(info):
        return info['type'] == 'file' and info['size'] >= min_size_bytes

    files = walk_directory(root, filter_func=filter_size, include_dirs=False)

    # Boyuta göre sırala (generator'dan listeye çevirip tekrar generator yap)
    for file_info in sorted(files, key=lambda x: x['size'], reverse=True):
        yield file_info


def directory_tree_string(path: str, max_depth: int = 3) -> Iterator[str]:
    """
    Dizin ağacını string representation olarak üretir

    Example:
        for line in directory_tree_string('/project'):
            print(line)
    """
    def format_size(size_bytes):
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024:
                return f"{size_bytes:.1f}{unit}"
            size_bytes /= 1024
        return f"{size_bytes:.1f}TB"

    for info in walk_directory(path, max_depth=max_depth):
        indent = "  " * info['depth']
        icon = "📁" if info['type'] == 'dir' else "📄"
        size = format_size(info['size']) if info['type'] == 'file' else ""

        yield f"{indent}{icon} {info['name']} {size}"


# ============================================================================
# SORU 7: Merge Sorted Iterables
# ============================================================================
"""
Birden fazla sıralı iterable'ı birleştirip sıralı tek bir iterable dönen
bir generator yazın. heapq.merge benzeri ama custom comparison desteği ile.

Gereksinimler:
- Multiple sorted iterables
- Custom key function
- Reverse sorting
- Memory efficient (tüm elemanlar bellekte tutulmamalı)
"""

# TODO: Merge sorted generator'ı implement edin


# ÇÖZÜM:
import heapq
from typing import TypeVar, List

T = TypeVar('T')

def merge_sorted(*iterables: Iterable[T],
                key: Optional[Callable[[T], any]] = None,
                reverse: bool = False) -> Iterator[T]:
    """
    Birden fazla sıralı iterable'ı merge eder

    Args:
        *iterables: Sıralı iterable'lar
        key: Comparison key function
        reverse: Reverse sorting

    Yields:
        Merge edilmiş sıralı elemanlar

    Example:
        list1 = [1, 4, 7, 10]
        list2 = [2, 5, 8]
        list3 = [3, 6, 9, 11]

        merged = merge_sorted(list1, list2, list3)
        # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    """
    if key is None:
        key = lambda x: x

    # Her iterable için (key_value, counter, value, iterator) tuple'ı
    heap = []
    counter = 0  # Eşit elemanlar için stable sorting

    # Her iterable'ı heap'e ekle
    for iterable in iterables:
        iterator = iter(iterable)
        try:
            value = next(iterator)
            key_value = key(value)

            if reverse:
                # Reverse için key'i negatif yap (sayılar için)
                # Genel çözüm için custom comparator gerekir
                heap_item = (-key_value, counter, value, iterator, key)
            else:
                heap_item = (key_value, counter, value, iterator, key)

            heapq.heappush(heap, heap_item)
            counter += 1
        except StopIteration:
            continue

    # Heap'ten çek ve yeni elemanları ekle
    while heap:
        if reverse:
            neg_key_value, _, value, iterator, key_func = heapq.heappop(heap)
        else:
            key_value, _, value, iterator, key_func = heapq.heappop(heap)

        yield value

        # Sonraki elemanı al
        try:
            next_value = next(iterator)
            next_key = key_func(next_value)

            if reverse:
                heap_item = (-next_key, counter, next_value, iterator, key_func)
            else:
                heap_item = (next_key, counter, next_value, iterator, key_func)

            heapq.heappush(heap, heap_item)
            counter += 1
        except StopIteration:
            continue


def merge_sorted_files(*file_paths: str,
                      key: Optional[Callable] = None,
                      output_path: Optional[str] = None) -> Iterator[str]:
    """
    Birden fazla sıralı dosyayı merge eder

    Memory efficient: Dosyaları satır satır okur

    Args:
        *file_paths: Sıralı dosya yolları
        key: Line comparison key
        output_path: Çıktı dosyası (optional)

    Yields:
        Merge edilmiş satırlar
    """
    def file_generator(path):
        with open(path, 'r') as f:
            for line in f:
                yield line.strip()

    # Her dosya için generator oluştur
    generators = [file_generator(path) for path in file_paths]

    # Merge
    merged = merge_sorted(*generators, key=key)

    # Output file varsa yaz
    if output_path:
        with open(output_path, 'w') as out:
            for line in merged:
                out.write(line + '\n')
                yield line
    else:
        yield from merged


# Test
if __name__ == "__main__":
    # Basit merge
    list1 = [1, 4, 7, 10]
    list2 = [2, 5, 8]
    list3 = [3, 6, 9, 11]

    merged = merge_sorted(list1, list2, list3)
    print("Merged:", list(merged))

    # Custom key ile merge
    users1 = [{'name': 'Alice', 'age': 25}, {'name': 'Charlie', 'age': 30}]
    users2 = [{'name': 'Bob', 'age': 27}, {'name': 'David', 'age': 35}]

    merged_users = merge_sorted(users1, users2, key=lambda x: x['age'])
    print("Merged users:", list(merged_users))


# ============================================================================
# SORU 8: Coroutine-based Pipeline
# ============================================================================
"""
send() metodu kullanan coroutine-based bir data processing pipeline yazın.

Gereksinimler:
- Coroutine decorator
- Multi-stage pipeline
- Error handling
- Pipeline statistics
"""

# TODO: Coroutine pipeline'ı implement edin


# ÇÖZÜM:
from functools import wraps

def coroutine(func):
    """
    Coroutine'i otomatik prime eden decorator

    Coroutine'ler kullanılmadan önce next() ile prime edilmelidir.
    Bu decorator bunu otomatik yapar.
    """
    @wraps(func)
    def primer(*args, **kwargs):
        gen = func(*args, **kwargs)
        next(gen)  # Prime the coroutine
        return gen
    return primer


@coroutine
def pipeline_source(target):
    """
    Pipeline'ın başlangıcı - veri alır ve iletir

    Args:
        target: Sonraki coroutine
    """
    try:
        while True:
            data = yield
            target.send(data)
    except GeneratorExit:
        target.close()


@coroutine
def filter_coroutine(predicate, target):
    """
    Filtreleme coroutine

    Args:
        predicate: Filtreleme fonksiyonu
        target: Sonraki coroutine
    """
    try:
        while True:
            data = yield
            if predicate(data):
                target.send(data)
    except GeneratorExit:
        target.close()


@coroutine
def map_coroutine(transform, target):
    """
    Transformation coroutine

    Args:
        transform: Dönüştürme fonksiyonu
        target: Sonraki coroutine
    """
    try:
        while True:
            data = yield
            transformed = transform(data)
            target.send(transformed)
    except GeneratorExit:
        target.close()


@coroutine
def broadcast(*targets):
    """
    Veriyi birden fazla target'a gönderir

    Args:
        *targets: Target coroutine'ler
    """
    try:
        while True:
            data = yield
            for target in targets:
                target.send(data)
    except GeneratorExit:
        for target in targets:
            target.close()


@coroutine
def accumulator(result_list):
    """
    Veriyi bir listeye toplar

    Args:
        result_list: Sonuçların ekleneceği liste
    """
    try:
        while True:
            data = yield
            result_list.append(data)
    except GeneratorExit:
        pass


@coroutine
def statistics_coroutine():
    """
    İstatistik toplayan coroutine

    Returns:
        İstatistikler (close() çağrıldığında)
    """
    stats = {
        'count': 0,
        'sum': 0,
        'min': float('inf'),
        'max': float('-inf'),
        'values': []
    }

    try:
        while True:
            value = yield
            stats['count'] += 1
            stats['sum'] += value
            stats['min'] = min(stats['min'], value)
            stats['max'] = max(stats['max'], value)
            stats['values'].append(value)
    except GeneratorExit:
        if stats['count'] > 0:
            stats['avg'] = stats['sum'] / stats['count']
        return stats


@coroutine
def error_handler(target, error_callback=None):
    """
    Error handling coroutine

    Args:
        target: Sonraki coroutine
        error_callback: Hata durumunda çağrılacak fonksiyon
    """
    try:
        while True:
            data = yield
            try:
                target.send(data)
            except Exception as e:
                if error_callback:
                    error_callback(data, e)
                else:
                    print(f"Error processing {data}: {e}")
    except GeneratorExit:
        target.close()


# Pipeline builder
class CoroutinePipeline:
    """
    Coroutine pipeline builder - fluent interface

    Example:
        results = []
        pipeline = (CoroutinePipeline()
            .filter(lambda x: x > 0)
            .map(lambda x: x * 2)
            .collect(results))

        for value in [-1, 2, -3, 4]:
            pipeline.send(value)

        pipeline.close()
        print(results)  # [4, 8]
    """

    def __init__(self):
        self.pipeline = None

    def filter(self, predicate):
        """Filtreleme ekle"""
        if self.pipeline is None:
            result_list = []
            self.pipeline = filter_coroutine(predicate, accumulator(result_list))
            self._result = result_list
        else:
            self.pipeline = filter_coroutine(predicate, self.pipeline)
        return self

    def map(self, transform):
        """Transformation ekle"""
        if self.pipeline is None:
            result_list = []
            self.pipeline = map_coroutine(transform, accumulator(result_list))
            self._result = result_list
        else:
            self.pipeline = map_coroutine(transform, self.pipeline)
        return self

    def collect(self, result_list):
        """Sonuçları topla"""
        if self.pipeline is None:
            self.pipeline = accumulator(result_list)
        else:
            self.pipeline = accumulator(result_list)
        self._result = result_list
        return self

    def send(self, data):
        """Pipeline'a veri gönder"""
        if self.pipeline:
            self.pipeline.send(data)

    def close(self):
        """Pipeline'ı kapat"""
        if self.pipeline:
            self.pipeline.close()

    def get_result(self):
        """Sonuçları al"""
        return getattr(self, '_result', None)


# Test
if __name__ == "__main__":
    # Basit coroutine pipeline
    results = []
    sink = accumulator(results)
    multiply = map_coroutine(lambda x: x * 2, sink)
    positive_filter = filter_coroutine(lambda x: x > 0, multiply)

    # Veri gönder
    for value in [-1, 2, -3, 4, 5]:
        positive_filter.send(value)

    positive_filter.close()
    print("Results:", results)  # [4, 8, 10]

    # Pipeline builder kullanımı
    results2 = []
    pipeline = (CoroutinePipeline()
        .filter(lambda x: x % 2 == 0)
        .map(lambda x: x ** 2)
        .collect(results2))

    for value in range(10):
        pipeline.send(value)

    pipeline.close()
    print("Pipeline results:", results2)  # [0, 4, 16, 36, 64]


# ============================================================================
# SORU 9: Lazy Evaluation with Caching
# ============================================================================
"""
Lazy evaluation ve caching destekli bir generator wrapper yazın.
Expensive computations için kullanılır.

Gereksinimler:
- Lazy evaluation
- Result caching
- Cache eviction policy (LRU)
- Memory limits
"""

# TODO: Cached generator'ı implement edin


# ÇÖZÜM:
from collections import OrderedDict
from typing import Hashable

class CachedGenerator:
    """
    Caching destekli lazy generator wrapper

    Example:
        @CachedGenerator(max_cache_size=100)
        def expensive_computation(n):
            for i in range(n):
                yield compute_expensive(i)
    """

    def __init__(self, max_cache_size: int = 128):
        """
        Args:
            max_cache_size: Maksimum cache boyutu (eleman sayısı)
        """
        self.max_cache_size = max_cache_size
        self.cache = OrderedDict()  # LRU cache için

    def __call__(self, func):
        """Decorator implementation"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Cache key oluştur
            cache_key = self._make_key(args, kwargs)

            # Cache'de var mı?
            if cache_key in self.cache:
                # LRU: En sona taşı
                self.cache.move_to_end(cache_key)
                # Cached değerleri yield et
                yield from self.cache[cache_key]
            else:
                # Yeni hesapla ve cache'le
                results = []
                for value in func(*args, **kwargs):
                    results.append(value)
                    yield value

                # Cache'e ekle
                self.cache[cache_key] = results

                # Cache overflow kontrolü (LRU eviction)
                if len(self.cache) > self.max_cache_size:
                    self.cache.popitem(last=False)  # İlk (en eski) elemanı çıkar

        return wrapper

    def _make_key(self, args, kwargs) -> tuple:
        """Cache key oluştur"""
        key_parts = [args]
        if kwargs:
            key_parts.append(tuple(sorted(kwargs.items())))
        return tuple(key_parts)

    def clear_cache(self):
        """Cache'i temizle"""
        self.cache.clear()

    def cache_info(self) -> dict:
        """Cache bilgilerini döner"""
        return {
            'size': len(self.cache),
            'max_size': self.max_cache_size,
            'keys': list(self.cache.keys())
        }


class LazySequence:
    """
    Lazy evaluation sequence - değerler sadece erişildiğinde hesaplanır

    Example:
        seq = LazySequence(lambda i: i**2, length=1000000)
        print(seq[100])  # Sadece 100. eleman hesaplanır
        print(list(seq[:10]))  # İlk 10 eleman hesaplanır
    """

    def __init__(self, generator_func: Callable[[int], any], length: int):
        """
        Args:
            generator_func: Index alan ve değer dönen fonksiyon
            length: Sequence uzunluğu
        """
        self.generator_func = generator_func
        self.length = length
        self._cache = {}

    def __len__(self):
        return self.length

    def __getitem__(self, key):
        if isinstance(key, slice):
            # Slice desteği
            start, stop, step = key.indices(self.length)
            return [self._get_value(i) for i in range(start, stop, step)]
        elif isinstance(key, int):
            # Negatif index desteği
            if key < 0:
                key += self.length

            if key < 0 or key >= self.length:
                raise IndexError("Index out of range")

            return self._get_value(key)
        else:
            raise TypeError("Indices must be integers or slices")

    def _get_value(self, index: int):
        """Değeri cache'den al veya hesapla"""
        if index not in self._cache:
            self._cache[index] = self.generator_func(index)
        return self._cache[index]

    def __iter__(self):
        """Iterator desteği"""
        for i in range(self.length):
            yield self._get_value(i)

    def cache_info(self) -> dict:
        """Cache bilgileri"""
        return {
            'cached_items': len(self._cache),
            'total_items': self.length,
            'cache_ratio': len(self._cache) / self.length if self.length > 0 else 0
        }


# Test
if __name__ == "__main__":
    # Cached generator kullanımı
    cache_gen = CachedGenerator(max_cache_size=3)

    @cache_gen
    def fibonacci_generator(n):
        """Fibonacci sayıları üret"""
        a, b = 0, 1
        for _ in range(n):
            yield a
            a, b = b, a + b

    # İlk çağrı - hesaplanır
    print("First call:", list(fibonacci_generator(10)))

    # İkinci çağrı - cache'den gelir
    print("Second call (cached):", list(fibonacci_generator(10)))

    # Cache bilgisi
    print("Cache info:", cache_gen.cache_info())

    # Lazy sequence kullanımı
    lazy_seq = LazySequence(lambda i: i**2, length=1000000)

    print("Item 100:", lazy_seq[100])  # Sadece 100. eleman hesaplanır
    print("First 5:", lazy_seq[:5])  # İlk 5 eleman hesaplanır
    print("Cache info:", lazy_seq.cache_info())


# ============================================================================
# SORU 10: Generator-based State Machine
# ============================================================================
"""
Generator kullanarak bir state machine (durum makinesi) implement edin.
send() ile state transition'ları kontrol edin.

Gereksinimler:
- Multiple states
- State transitions
- Event handling
- State history
"""

# TODO: State machine'i implement edin


# ÇÖZÜM:
from enum import Enum, auto
from dataclasses import dataclass
from typing import Optional, Dict, Set
import time


class State(Enum):
    """Örnek state'ler"""
    IDLE = auto()
    RUNNING = auto()
    PAUSED = auto()
    STOPPED = auto()
    ERROR = auto()


@dataclass
class Event:
    """State machine event"""
    name: str
    data: Optional[dict] = None
    timestamp: float = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()


@dataclass
class StateTransition:
    """State geçişi kaydı"""
    from_state: State
    to_state: State
    event: Event
    timestamp: float


class StateMachine:
    """
    Generator-based state machine

    Example:
        sm = StateMachine(initial_state=State.IDLE)
        sm.add_transition(State.IDLE, 'start', State.RUNNING)
        sm.add_transition(State.RUNNING, 'pause', State.PAUSED)
        sm.add_transition(State.PAUSED, 'resume', State.RUNNING)
        sm.add_transition(State.RUNNING, 'stop', State.STOPPED)

        machine = sm.run()
        next(machine)  # Prime

        machine.send(Event('start'))  # IDLE -> RUNNING
        machine.send(Event('pause'))  # RUNNING -> PAUSED
        machine.send(Event('resume'))  # PAUSED -> RUNNING
        machine.send(Event('stop'))  # RUNNING -> STOPPED
    """

    def __init__(self, initial_state: State):
        """
        Args:
            initial_state: Başlangıç state'i
        """
        self.current_state = initial_state
        self.transitions: Dict[tuple, State] = {}  # (from_state, event) -> to_state
        self.history: list[StateTransition] = []
        self.state_handlers: Dict[State, Callable] = {}
        self.transition_handlers: Dict[tuple, Callable] = {}

    def add_transition(self, from_state: State, event_name: str, to_state: State):
        """
        State geçişi tanımla

        Args:
            from_state: Kaynak state
            event_name: Event ismi
            to_state: Hedef state
        """
        self.transitions[(from_state, event_name)] = to_state

    def add_state_handler(self, state: State, handler: Callable):
        """
        State'e girildiğinde çağrılacak handler

        Args:
            state: State
            handler: Handler fonksiyonu
        """
        self.state_handlers[state] = handler

    def add_transition_handler(self, from_state: State, to_state: State,
                              handler: Callable):
        """
        State geçişinde çağrılacak handler

        Args:
            from_state: Kaynak state
            to_state: Hedef state
            handler: Handler fonksiyonu
        """
        self.transition_handlers[(from_state, to_state)] = handler

    def run(self):
        """
        State machine coroutine

        Yields:
            Current state
        """
        print(f"State machine started in state: {self.current_state.name}")

        try:
            while True:
                # Event bekle
                event = yield self.current_state

                if event is None:
                    continue

                # Geçiş var mı?
                transition_key = (self.current_state, event.name)
                if transition_key not in self.transitions:
                    print(f"Invalid transition: {self.current_state.name} -> {event.name}")
                    continue

                # Yeni state
                new_state = self.transitions[transition_key]

                print(f"Transition: {self.current_state.name} -> {new_state.name} "
                      f"(event: {event.name})")

                # Transition handler çağır
                handler_key = (self.current_state, new_state)
                if handler_key in self.transition_handlers:
                    self.transition_handlers[handler_key](event)

                # History'ye ekle
                transition = StateTransition(
                    from_state=self.current_state,
                    to_state=new_state,
                    event=event,
                    timestamp=time.time()
                )
                self.history.append(transition)

                # State değiştir
                old_state = self.current_state
                self.current_state = new_state

                # State handler çağır
                if new_state in self.state_handlers:
                    self.state_handlers[new_state](old_state, event)

        except GeneratorExit:
            print(f"State machine stopped in state: {self.current_state.name}")

    def get_history(self) -> list[StateTransition]:
        """State geçiş geçmişini döner"""
        return self.history.copy()

    def get_current_state(self) -> State:
        """Mevcut state'i döner"""
        return self.current_state


# Örnek: Medya player state machine
def create_media_player_fsm() -> StateMachine:
    """
    Medya player state machine

    States: IDLE, PLAYING, PAUSED, STOPPED
    Events: play, pause, stop, resume
    """
    sm = StateMachine(initial_state=State.IDLE)

    # Transitions
    sm.add_transition(State.IDLE, 'play', State.RUNNING)
    sm.add_transition(State.RUNNING, 'pause', State.PAUSED)
    sm.add_transition(State.PAUSED, 'resume', State.RUNNING)
    sm.add_transition(State.RUNNING, 'stop', State.STOPPED)
    sm.add_transition(State.PAUSED, 'stop', State.STOPPED)
    sm.add_transition(State.STOPPED, 'play', State.RUNNING)

    # State handlers
    def on_playing(old_state, event):
        print(f"  ▶️  Now playing... (from {old_state.name})")

    def on_paused(old_state, event):
        print(f"  ⏸️  Paused")

    def on_stopped(old_state, event):
        print(f"  ⏹️  Stopped")

    sm.add_state_handler(State.RUNNING, on_playing)
    sm.add_state_handler(State.PAUSED, on_paused)
    sm.add_state_handler(State.STOPPED, on_stopped)

    return sm


# Test
if __name__ == "__main__":
    # Medya player state machine test
    player = create_media_player_fsm()
    machine = player.run()
    next(machine)  # Prime

    # Simülasyon
    machine.send(Event('play'))
    time.sleep(0.1)
    machine.send(Event('pause'))
    time.sleep(0.1)
    machine.send(Event('resume'))
    time.sleep(0.1)
    machine.send(Event('stop'))

    # History
    print("\nState transition history:")
    for trans in player.get_history():
        print(f"  {trans.from_state.name} -> {trans.to_state.name} "
              f"via '{trans.event.name}'")


# ============================================================================
# SORU 11-15: Daha Fazla Advanced Patterns
# ============================================================================

# SORU 11: Parallel Generator Processing
"""
Birden fazla generator'ı paralel olarak işleyen bir sistem yazın.
"""

def parallel_generators(*generators, max_workers: int = 4):
    """
    Birden fazla generator'ı concurrent olarak çalıştırır

    Args:
        *generators: Generator fonksiyonları
        max_workers: Maksimum worker sayısı

    Yields:
        (generator_index, value) tuple'ları
    """
    from concurrent.futures import ThreadPoolExecutor
    from queue import Queue
    import threading

    result_queue = Queue()
    done_event = threading.Event()

    def worker(gen_index, generator):
        """Her generator için worker thread"""
        try:
            for value in generator:
                result_queue.put((gen_index, value))
        except Exception as e:
            result_queue.put((gen_index, Exception(f"Error in generator {gen_index}: {e}")))
        finally:
            result_queue.put((gen_index, None))  # Sentinel

    # Thread pool başlat
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Her generator için worker başlat
        futures = []
        for i, gen in enumerate(generators):
            future = executor.submit(worker, i, gen)
            futures.append(future)

        # Sonuçları yield et
        completed = 0
        while completed < len(generators):
            gen_index, value = result_queue.get()

            if value is None:
                # Generator tamamlandı
                completed += 1
            elif isinstance(value, Exception):
                # Hata oluştu
                print(f"Error: {value}")
            else:
                # Normal değer
                yield (gen_index, value)


# SORU 12: Windowed Aggregation
"""
Time-series data için windowed aggregation yapan generator.
"""

def windowed_aggregation(data_stream: Iterator[dict],
                        window_seconds: float,
                        aggregation_func: Callable,
                        timestamp_key: str = 'timestamp'):
    """
    Zaman pencerelerine göre aggregation yapar

    Args:
        data_stream: Timestamp içeren data stream
        window_seconds: Pencere boyutu (saniye)
        aggregation_func: Aggregation fonksiyonu
        timestamp_key: Timestamp field ismi

    Yields:
        Aggregate edilmiş değerler

    Example:
        data = [
            {'timestamp': 1.0, 'value': 10},
            {'timestamp': 1.5, 'value': 20},
            {'timestamp': 2.0, 'value': 30},
            {'timestamp': 3.0, 'value': 40},
        ]

        windows = windowed_aggregation(
            iter(data),
            window_seconds=1.0,
            aggregation_func=lambda items: sum(x['value'] for x in items)
        )
    """
    current_window = []
    window_start = None

    for item in data_stream:
        timestamp = item.get(timestamp_key)

        if window_start is None:
            window_start = timestamp

        # Yeni pencere mi?
        if timestamp - window_start >= window_seconds:
            # Mevcut pencereyi aggregate et
            if current_window:
                result = {
                    'window_start': window_start,
                    'window_end': window_start + window_seconds,
                    'count': len(current_window),
                    'value': aggregation_func(current_window)
                }
                yield result

            # Yeni pencere başlat
            current_window = [item]
            window_start = timestamp
        else:
            current_window.append(item)

    # Son pencere
    if current_window:
        result = {
            'window_start': window_start,
            'window_end': window_start + window_seconds,
            'count': len(current_window),
            'value': aggregation_func(current_window)
        }
        yield result


# SORU 13: Rate Limiter Generator
"""
API rate limiting için generator.
"""

def rate_limiter(requests: Iterator,
                max_per_second: float,
                burst_size: int = 1):
    """
    Token bucket algoritması ile rate limiting

    Args:
        requests: İstek stream
        max_per_second: Saniyedeki maksimum istek
        burst_size: Burst boyutu

    Yields:
        Rate-limited requests

    Example:
        requests = range(100)
        limited = rate_limiter(requests, max_per_second=10)

        for req in limited:
            make_api_call(req)  # Saniyede max 10 istek
    """
    import time

    tokens = burst_size
    last_update = time.time()
    token_rate = max_per_second

    for request in requests:
        now = time.time()

        # Token'ları yenile
        time_passed = now - last_update
        tokens = min(burst_size, tokens + time_passed * token_rate)
        last_update = now

        # Token var mı?
        if tokens < 1:
            # Bekle
            wait_time = (1 - tokens) / token_rate
            time.sleep(wait_time)
            tokens = 0
            last_update = time.time()

        # Bir token kullan
        tokens -= 1

        yield request


# SORU 14: Dependency-aware Pipeline
"""
Bağımlılıkları olan task'ları sırayla işleyen generator.
"""

def dependency_pipeline(tasks: dict[str, dict]) -> Iterator[tuple[str, any]]:
    """
    Task bağımlılıklarına göre sıralı execution

    Args:
        tasks: Task dictionary
            {
                'task_name': {
                    'func': callable,
                    'depends_on': [task_names],
                    'args': tuple,
                    'kwargs': dict
                }
            }

    Yields:
        (task_name, result) tuple'ları

    Example:
        tasks = {
            'fetch_data': {
                'func': fetch_from_api,
                'depends_on': [],
                'args': ()
            },
            'process_data': {
                'func': process,
                'depends_on': ['fetch_data'],
                'args': ()
            },
            'save_data': {
                'func': save_to_db,
                'depends_on': ['process_data'],
                'args': ()
            }
        }

        for task_name, result in dependency_pipeline(tasks):
            print(f"{task_name} completed: {result}")
    """
    completed = {}
    remaining = set(tasks.keys())

    while remaining:
        # Çalıştırılabilir task bul
        ready = None
        for task_name in remaining:
            task = tasks[task_name]
            dependencies = task.get('depends_on', [])

            # Tüm bağımlılıklar tamamlandı mı?
            if all(dep in completed for dep in dependencies):
                ready = task_name
                break

        if ready is None:
            raise ValueError("Circular dependency detected or missing task!")

        # Task'ı çalıştır
        task = tasks[ready]
        func = task['func']
        args = task.get('args', ())
        kwargs = task.get('kwargs', {})

        # Bağımlılık sonuçlarını inject et
        dep_results = {dep: completed[dep] for dep in task.get('depends_on', [])}
        if dep_results:
            kwargs['dependencies'] = dep_results

        result = func(*args, **kwargs)
        completed[ready] = result
        remaining.remove(ready)

        yield (ready, result)


# SORU 15: Memory-mapped File Generator
"""
Çok büyük dosyaları memory-mapped şekilde okuyan generator.
"""

def mmap_file_reader(file_path: str,
                    chunk_size: int = 1024*1024) -> Iterator[bytes]:
    """
    Memory-mapped file okuma - çok büyük dosyalar için

    Args:
        file_path: Dosya yolu
        chunk_size: Chunk boyutu (bytes)

    Yields:
        Dosya chunk'ları (bytes)

    Example:
        # 100GB dosya oku (tamamı bellekte tutulmaz)
        for chunk in mmap_file_reader('huge_file.bin'):
            process_chunk(chunk)
    """
    import mmap

    with open(file_path, 'r+b') as f:
        # Memory-map file
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mmapped_file:
            file_size = len(mmapped_file)

            # Chunk chunk oku
            for offset in range(0, file_size, chunk_size):
                end = min(offset + chunk_size, file_size)
                chunk = mmapped_file[offset:end]
                yield chunk


def mmap_line_reader(file_path: str) -> Iterator[str]:
    """
    Memory-mapped satır satır okuma

    Args:
        file_path: Dosya yolu

    Yields:
        Her satır
    """
    import mmap

    with open(file_path, 'r+b') as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mmapped_file:
            # Satır buffer
            line_buffer = b''

            for i in range(len(mmapped_file)):
                byte = mmapped_file[i:i+1]

                if byte == b'\n':
                    # Satır tamamlandı
                    if line_buffer:
                        yield line_buffer.decode('utf-8')
                        line_buffer = b''
                else:
                    line_buffer += byte

            # Son satır
            if line_buffer:
                yield line_buffer.decode('utf-8')


"""
BONUS: Generator Performance Tips
==================================

1. Generator Expression vs List Comprehension
   - Generator: (x**2 for x in range(1000000))  # ~128 bytes
   - List: [x**2 for x in range(1000000)]      # ~8MB

2. Early Termination
   - itertools.islice kullan
   - Koşullu break ile

3. Chaining Generators
   - itertools.chain ile
   - yield from ile

4. Memory Profiling
   - tracemalloc kullan
   - memory_profiler kullan

5. Performance Profiling
   - cProfile kullan
   - line_profiler kullan

6. Pipeline Optimization
   - Filter'ları başa al (erken eleme)
   - Expensive operations'ı sona al
   - Parallel processing için ThreadPoolExecutor

7. Infinite Generators
   - Always use with islice or takewhile
   - Timeout mekanizması ekle

8. Error Handling
   - try-except inside generator
   - Generator wrapper ile
   - Error logging ekle

Bu pattern'ler production-ready, memory-efficient ve scalable kod
yazmak için essential'dır!
"""
