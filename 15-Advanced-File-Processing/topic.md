# Advanced File Processing (İleri Seviye Dosya İşlemleri)

## İçindekiler
1. [Binary File Operations](#binary-file-operations)
2. [Memory-Mapped Files (mmap)](#memory-mapped-files)
3. [Advanced Serialization](#advanced-serialization)
4. [Data Formats](#data-formats)
5. [Compression & Archives](#compression-archives)
6. [Stream Processing](#stream-processing)
7. [Large File Handling](#large-file-handling)
8. [Production Patterns](#production-patterns)

---

## Binary File Operations

### Temel Binary İşlemler
Binary dosyalar, veriyi byte seviyesinde saklar ve daha verimli okuma/yazma işlemleri sunar.

```python
import struct
import os
from pathlib import Path

# Binary dosya yazma - struct kullanımı
def write_binary_data(filename, records):
    """
    Structured binary data yazma
    Format: int(4), float(4), string(20)
    """
    with open(filename, 'wb') as f:
        for record in records:
            # Struct pack: verileri binary'ye dönüştür
            packed = struct.pack('if20s',
                                record['id'],
                                record['value'],
                                record['name'].encode('utf-8'))
            f.write(packed)

# Binary dosya okuma
def read_binary_data(filename):
    """Structured binary data okuma"""
    records = []
    record_size = struct.calcsize('if20s')

    with open(filename, 'rb') as f:
        while True:
            data = f.read(record_size)
            if not data:
                break

            # Unpack: binary'den veri tipine dönüştür
            unpacked = struct.unpack('if20s', data)
            records.append({
                'id': unpacked[0],
                'value': unpacked[1],
                'name': unpacked[2].decode('utf-8').rstrip('\x00')
            })

    return records

# Kullanım
data = [
    {'id': 1, 'value': 99.5, 'name': 'Product A'},
    {'id': 2, 'value': 150.75, 'name': 'Product B'}
]
write_binary_data('data.bin', data)
print(read_binary_data('data.bin'))
```

### Custom Binary Format
```python
import struct
from datetime import datetime
from enum import IntEnum

class RecordType(IntEnum):
    """Binary record tipleri"""
    HEADER = 0x01
    DATA = 0x02
    FOOTER = 0x03

class BinaryFileWriter:
    """
    Custom binary format writer
    Production-ready binary file handler
    """
    def __init__(self, filename, version=1):
        self.filename = filename
        self.version = version
        self.file = None
        self.record_count = 0

    def __enter__(self):
        self.file = open(self.filename, 'wb')
        self._write_header()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._write_footer()
        self.file.close()

    def _write_header(self):
        """File header yazma"""
        # Magic number (4 bytes) + Version (2 bytes) + Timestamp (8 bytes)
        magic = b'PYAF'  # Python Advanced File
        timestamp = int(datetime.now().timestamp())
        header = struct.pack('4sHQ', magic, self.version, timestamp)

        self.file.write(struct.pack('B', RecordType.HEADER))
        self.file.write(struct.pack('I', len(header)))
        self.file.write(header)

    def write_record(self, data_type, data):
        """Data record yazma"""
        # Type (1) + Length (4) + Data (variable)
        data_bytes = data if isinstance(data, bytes) else str(data).encode('utf-8')

        self.file.write(struct.pack('B', RecordType.DATA))
        self.file.write(struct.pack('I', len(data_bytes)))
        self.file.write(data_bytes)

        self.record_count += 1

    def _write_footer(self):
        """File footer yazma"""
        # Record count + CRC32 checksum
        import zlib
        self.file.seek(0)
        content = self.file.read()
        checksum = zlib.crc32(content)

        self.file.seek(0, 2)  # EOF
        footer = struct.pack('II', self.record_count, checksum)

        self.file.write(struct.pack('B', RecordType.FOOTER))
        self.file.write(struct.pack('I', len(footer)))
        self.file.write(footer)

# Kullanım
with BinaryFileWriter('custom.bin') as writer:
    writer.write_record('string', 'Hello World')
    writer.write_record('number', '12345')
    writer.write_record('data', b'\x00\x01\x02\x03')
```

---

## Memory-Mapped Files (mmap)

### Temel mmap Kullanımı
Memory-mapped files, büyük dosyaları RAM'de gibi işlememizi sağlar.

```python
import mmap
import os

def mmap_read_example(filename):
    """
    Memory-mapped file okuma
    Büyük dosyalar için çok verimli
    """
    with open(filename, 'r+b') as f:
        # Dosyayı memory'ye map et
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mmapped:
            # Byte array gibi kullan
            print(f"File size: {len(mmapped)} bytes")

            # İlk 100 byte oku
            print(mmapped[:100])

            # String arama (çok hızlı)
            position = mmapped.find(b'search_term')
            if position != -1:
                print(f"Found at position: {position}")

            # Belirli bir pozisyondan oku
            mmapped.seek(1000)
            data = mmapped.read(100)
            print(data)

def mmap_write_example(filename):
    """Memory-mapped file yazma"""
    # Dosya boyutu belirle (örnek: 1MB)
    size = 1024 * 1024

    with open(filename, 'wb') as f:
        f.write(b'\x00' * size)  # Dosyayı oluştur

    with open(filename, 'r+b') as f:
        with mmap.mmap(f.fileno(), size) as mmapped:
            # Random access yazma
            mmapped[0:13] = b'Hello, World!'
            mmapped[100:116] = b'Memory Mapping!'

            # Seek ve write
            mmapped.seek(500)
            mmapped.write(b'Fast writing')

            # Flush to disk
            mmapped.flush()

# Kullanım
# mmap_read_example('large_file.bin')
# mmap_write_example('mmap_test.bin')
```

### Advanced mmap: Shared Memory
```python
import mmap
import struct
import time
from multiprocessing import Process

def mmap_shared_memory():
    """
    Process'ler arası veri paylaşımı
    Shared memory kullanımı
    """
    # Anonymous memory mapping (shared between processes)
    shared_mem = mmap.mmap(-1, 1024)  # 1KB shared memory

    def writer_process(shared_mem):
        """Writer process"""
        for i in range(10):
            # Counter yaz
            shared_mem.seek(0)
            shared_mem.write(struct.pack('I', i))
            shared_mem.flush()
            time.sleep(0.5)

    def reader_process(shared_mem):
        """Reader process"""
        for _ in range(10):
            shared_mem.seek(0)
            data = shared_mem.read(4)
            if data:
                counter = struct.unpack('I', data)[0]
                print(f"Read counter: {counter}")
            time.sleep(0.5)

    # Processes oluştur
    writer = Process(target=writer_process, args=(shared_mem,))
    reader = Process(target=reader_process, args=(shared_mem,))

    writer.start()
    reader.start()

    writer.join()
    reader.join()

    shared_mem.close()

# mmap_shared_memory()
```

### mmap ile Large File Processing
```python
import mmap
import re
from typing import Iterator, Tuple

class MemoryMappedProcessor:
    """
    Memory-mapped file processor
    Büyük dosyalar için optimize edilmiş
    """
    def __init__(self, filename, chunk_size=8192):
        self.filename = filename
        self.chunk_size = chunk_size

    def find_pattern(self, pattern: bytes) -> Iterator[int]:
        """Dosyada pattern ara (tüm pozisyonlar)"""
        with open(self.filename, 'rb') as f:
            with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mmapped:
                pos = 0
                while True:
                    pos = mmapped.find(pattern, pos)
                    if pos == -1:
                        break
                    yield pos
                    pos += 1

    def count_occurrences(self, pattern: bytes) -> int:
        """Pattern sayısını hesapla"""
        return sum(1 for _ in self.find_pattern(pattern))

    def replace_pattern(self, old_pattern: bytes, new_pattern: bytes,
                       output_file: str):
        """
        Pattern değiştirme (yeni dosyaya)
        Aynı boyutta olmalı
        """
        if len(old_pattern) != len(new_pattern):
            raise ValueError("Patterns must be same length for in-place")

        with open(self.filename, 'rb') as f_in:
            with mmap.mmap(f_in.fileno(), 0, access=mmap.ACCESS_READ) as mmapped:
                with open(output_file, 'wb') as f_out:
                    pos = 0
                    last_pos = 0

                    for pos in self.find_pattern(old_pattern):
                        # Önceki kısmı yaz
                        f_out.write(mmapped[last_pos:pos])
                        # Yeni pattern'i yaz
                        f_out.write(new_pattern)
                        last_pos = pos + len(old_pattern)

                    # Kalan kısmı yaz
                    f_out.write(mmapped[last_pos:])

    def extract_lines_matching(self, pattern: re.Pattern) -> Iterator[str]:
        """Regex ile eşleşen satırları al"""
        with open(self.filename, 'rb') as f:
            with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mmapped:
                for line in iter(mmapped.readline, b''):
                    if pattern.search(line):
                        yield line.decode('utf-8', errors='ignore')

# Kullanım
# processor = MemoryMappedProcessor('large_log.txt')
# count = processor.count_occurrences(b'ERROR')
# print(f"Error count: {count}")
```

---

## Advanced Serialization

### Pickle Advanced
```python
import pickle
import pickletools
from typing import Any
import io

class CustomPickler:
    """
    Advanced pickle kullanımı
    Güvenlik ve optimizasyon
    """

    @staticmethod
    def save_optimized(obj: Any, filename: str, protocol: int = pickle.HIGHEST_PROTOCOL):
        """Optimize edilmiş pickle save"""
        with open(filename, 'wb') as f:
            pickle.dump(obj, f, protocol=protocol)

    @staticmethod
    def load_safe(filename: str, allowed_types: set = None):
        """
        Güvenli pickle load
        Sadece belirli tiplere izin ver
        """
        class RestrictedUnpickler(pickle.Unpickler):
            def find_class(self, module, name):
                if allowed_types and f"{module}.{name}" not in allowed_types:
                    raise pickle.UnpicklingError(
                        f"Global '{module}.{name}' is forbidden"
                    )
                return super().find_class(module, name)

        with open(filename, 'rb') as f:
            return RestrictedUnpickler(f).load()

    @staticmethod
    def optimize_pickle(input_file: str, output_file: str):
        """
        Pickle dosyasını optimize et
        pickletools kullanarak
        """
        with open(input_file, 'rb') as f_in:
            with open(output_file, 'wb') as f_out:
                pickletools.optimize(f_in, f_out)

    @staticmethod
    def analyze_pickle(filename: str):
        """Pickle içeriğini analiz et"""
        with open(filename, 'rb') as f:
            pickletools.dis(f)

# Kullanım
data = {
    'users': [{'id': 1, 'name': 'Ali'}, {'id': 2, 'name': 'Ayşe'}],
    'config': {'timeout': 30, 'retries': 3}
}

CustomPickler.save_optimized(data, 'data.pkl')
loaded = CustomPickler.load_safe('data.pkl', {'builtins.dict', 'builtins.list'})
print(loaded)
```

### JSON Advanced
```python
import json
from datetime import datetime, date
from decimal import Decimal
from pathlib import Path
from typing import Any
import dataclasses

class AdvancedJSONEncoder(json.JSONEncoder):
    """
    Gelişmiş JSON encoder
    Custom tipleri serialize et
    """
    def default(self, obj):
        # Datetime handling
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()

        # Decimal handling
        if isinstance(obj, Decimal):
            return float(obj)

        # Path handling
        if isinstance(obj, Path):
            return str(obj)

        # Dataclass handling
        if dataclasses.is_dataclass(obj):
            return dataclasses.asdict(obj)

        # Set handling
        if isinstance(obj, set):
            return list(obj)

        # Bytes handling
        if isinstance(obj, bytes):
            return obj.decode('utf-8', errors='ignore')

        return super().default(obj)

class AdvancedJSONDecoder(json.JSONDecoder):
    """Gelişmiş JSON decoder"""
    def __init__(self, *args, **kwargs):
        super().__init__(object_hook=self.object_hook, *args, **kwargs)

    def object_hook(self, obj):
        """Custom object reconstruction"""
        # ISO datetime strings to datetime
        for key, value in obj.items():
            if isinstance(value, str):
                # Try parsing as datetime
                try:
                    obj[key] = datetime.fromisoformat(value)
                except (ValueError, AttributeError):
                    pass

        return obj

class StreamingJSONWriter:
    """
    Streaming JSON writer
    Büyük JSON dosyaları için
    """
    def __init__(self, filename):
        self.filename = filename
        self.file = None
        self.first_item = True

    def __enter__(self):
        self.file = open(self.filename, 'w')
        self.file.write('[\n')
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.file.write('\n]')
        self.file.close()

    def write_item(self, item: dict):
        """Tek bir item yaz"""
        if not self.first_item:
            self.file.write(',\n')

        json.dump(item, self.file, indent=2, cls=AdvancedJSONEncoder)
        self.first_item = False

# Kullanım
@dataclasses.dataclass
class User:
    id: int
    name: str
    created_at: datetime
    balance: Decimal

users = [
    User(1, 'Ali', datetime.now(), Decimal('100.50')),
    User(2, 'Ayşe', datetime.now(), Decimal('250.75'))
]

# Streaming write
with StreamingJSONWriter('users.json') as writer:
    for user in users:
        writer.write_item(user)

# Advanced encode/decode
data = {
    'timestamp': datetime.now(),
    'path': Path('/home/user'),
    'amount': Decimal('99.99'),
    'tags': {'python', 'json'}
}

json_str = json.dumps(data, cls=AdvancedJSONEncoder, indent=2)
print(json_str)
```

### YAML Processing
```python
"""
YAML: Human-friendly data serialization
pip install pyyaml
"""
import yaml
from typing import Any, Dict
from pathlib import Path

class YAMLProcessor:
    """YAML dosya işlemleri"""

    @staticmethod
    def load_yaml(filename: str) -> Dict[str, Any]:
        """YAML dosyası yükle"""
        with open(filename, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    @staticmethod
    def save_yaml(data: Dict[str, Any], filename: str,
                  sort_keys: bool = False):
        """YAML dosyası kaydet"""
        with open(filename, 'w', encoding='utf-8') as f:
            yaml.dump(data, f,
                     default_flow_style=False,
                     allow_unicode=True,
                     sort_keys=sort_keys,
                     indent=2)

    @staticmethod
    def load_multiple_documents(filename: str):
        """
        Çoklu YAML dökümanları yükle
        (--- ile ayrılmış)
        """
        with open(filename, 'r', encoding='utf-8') as f:
            return list(yaml.safe_load_all(f))

    @staticmethod
    def merge_yaml_files(*filenames: str) -> Dict[str, Any]:
        """Birden fazla YAML dosyasını birleştir"""
        merged = {}
        for filename in filenames:
            data = YAMLProcessor.load_yaml(filename)
            merged.update(data)
        return merged

# Kullanım örneği
config = {
    'database': {
        'host': 'localhost',
        'port': 5432,
        'credentials': {
            'username': 'admin',
            'password': 'secret'
        }
    },
    'logging': {
        'level': 'INFO',
        'handlers': ['console', 'file']
    }
}

# YAMLProcessor.save_yaml(config, 'config.yaml')
# loaded = YAMLProcessor.load_yaml('config.yaml')
```

### MessagePack
```python
"""
MessagePack: Binary serialization (faster than JSON)
pip install msgpack
"""
import msgpack
from datetime import datetime
from typing import Any

class MessagePackProcessor:
    """MessagePack serialization"""

    @staticmethod
    def pack(data: Any) -> bytes:
        """Python object -> MessagePack binary"""
        return msgpack.packb(data, use_bin_type=True)

    @staticmethod
    def unpack(data: bytes) -> Any:
        """MessagePack binary -> Python object"""
        return msgpack.unpackb(data, raw=False)

    @staticmethod
    def pack_file(data: Any, filename: str):
        """Dosyaya MessagePack yaz"""
        with open(filename, 'wb') as f:
            msgpack.pack(data, f, use_bin_type=True)

    @staticmethod
    def unpack_file(filename: str) -> Any:
        """Dosyadan MessagePack oku"""
        with open(filename, 'rb') as f:
            return msgpack.unpack(f, raw=False)

    @staticmethod
    def stream_pack(items, filename: str):
        """Streaming pack (büyük veri setleri)"""
        with open(filename, 'wb') as f:
            packer = msgpack.Packer(use_bin_type=True)
            for item in items:
                f.write(packer.pack(item))

    @staticmethod
    def stream_unpack(filename: str):
        """Streaming unpack"""
        with open(filename, 'rb') as f:
            unpacker = msgpack.Unpacker(f, raw=False)
            for item in unpacker:
                yield item

# Performans karşılaştırması
import json
import time

data = {
    'users': [{'id': i, 'name': f'User{i}', 'active': True}
              for i in range(10000)]
}

# JSON
start = time.time()
json_data = json.dumps(data).encode('utf-8')
json_time = time.time() - start
print(f"JSON: {len(json_data)} bytes, {json_time:.4f}s")

# MessagePack
start = time.time()
msgpack_data = MessagePackProcessor.pack(data)
msgpack_time = time.time() - start
print(f"MessagePack: {len(msgpack_data)} bytes, {msgpack_time:.4f}s")
print(f"Size reduction: {(1 - len(msgpack_data)/len(json_data))*100:.1f}%")
```

---

## Data Formats

### XML Processing
```python
import xml.etree.ElementTree as ET
from xml.dom import minidom
from typing import Dict, Any, List

class XMLProcessor:
    """Advanced XML processing"""

    @staticmethod
    def dict_to_xml(data: Dict[str, Any], root_name: str = 'root') -> str:
        """Dictionary'den XML oluştur"""
        def build_element(parent, data):
            if isinstance(data, dict):
                for key, value in data.items():
                    child = ET.SubElement(parent, key)
                    build_element(child, value)
            elif isinstance(data, list):
                for item in data:
                    build_element(parent, item)
            else:
                parent.text = str(data)

        root = ET.Element(root_name)
        build_element(root, data)

        # Pretty print
        rough_string = ET.tostring(root, encoding='unicode')
        reparsed = minidom.parseString(rough_string)
        return reparsed.toprettyxml(indent="  ")

    @staticmethod
    def xml_to_dict(xml_string: str) -> Dict[str, Any]:
        """XML'den dictionary oluştur"""
        def parse_element(element):
            result = {}

            # Attributes
            if element.attrib:
                result['@attributes'] = element.attrib

            # Children
            children = list(element)
            if children:
                child_dict = {}
                for child in children:
                    child_data = parse_element(child)
                    if child.tag in child_dict:
                        # Multiple children with same tag
                        if not isinstance(child_dict[child.tag], list):
                            child_dict[child.tag] = [child_dict[child.tag]]
                        child_dict[child.tag].append(child_data)
                    else:
                        child_dict[child.tag] = child_data
                result.update(child_dict)

            # Text content
            if element.text and element.text.strip():
                if result:
                    result['#text'] = element.text.strip()
                else:
                    return element.text.strip()

            return result if result else None

        root = ET.fromstring(xml_string)
        return {root.tag: parse_element(root)}

    @staticmethod
    def process_large_xml(filename: str, tag: str):
        """
        Büyük XML dosyalarını streaming ile işle
        Memory-efficient
        """
        context = ET.iterparse(filename, events=('start', 'end'))
        context = iter(context)

        event, root = next(context)

        for event, elem in context:
            if event == 'end' and elem.tag == tag:
                # Process element
                yield XMLProcessor._element_to_dict(elem)

                # Clear element to free memory
                root.clear()

    @staticmethod
    def _element_to_dict(elem) -> Dict[str, Any]:
        """Element'i dict'e çevir"""
        result = {'tag': elem.tag}
        if elem.attrib:
            result['attributes'] = elem.attrib
        if elem.text and elem.text.strip():
            result['text'] = elem.text.strip()
        return result

# Kullanım
data = {
    'users': {
        'user': [
            {'id': '1', 'name': 'Ali', 'email': 'ali@example.com'},
            {'id': '2', 'name': 'Ayşe', 'email': 'ayse@example.com'}
        ]
    }
}

xml_string = XMLProcessor.dict_to_xml(data, 'database')
print(xml_string)

parsed = XMLProcessor.xml_to_dict(xml_string)
print(parsed)
```

### Parquet Files
```python
"""
Parquet: Columnar storage format (very efficient for big data)
pip install pyarrow pandas
"""
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
from typing import Dict, List, Any

class ParquetProcessor:
    """Parquet dosya işlemleri"""

    @staticmethod
    def write_parquet(data: List[Dict[str, Any]], filename: str,
                     compression: str = 'snappy'):
        """
        Parquet dosyası yaz
        Compression: 'snappy', 'gzip', 'brotli', 'lz4', 'zstd'
        """
        # DataFrame'e çevir
        df = pd.DataFrame(data)

        # Parquet'e yaz
        df.to_parquet(filename,
                     engine='pyarrow',
                     compression=compression,
                     index=False)

    @staticmethod
    def read_parquet(filename: str, columns: List[str] = None) -> pd.DataFrame:
        """
        Parquet dosyası oku
        columns: Sadece belirli sütunları oku (performans)
        """
        return pd.read_parquet(filename,
                              engine='pyarrow',
                              columns=columns)

    @staticmethod
    def write_partitioned_parquet(df: pd.DataFrame, base_path: str,
                                  partition_cols: List[str]):
        """
        Partitioned Parquet (büyük veri setleri)
        Örnek: year=2024/month=01/data.parquet
        """
        df.to_parquet(base_path,
                     engine='pyarrow',
                     partition_cols=partition_cols,
                     compression='snappy')

    @staticmethod
    def get_parquet_metadata(filename: str) -> Dict[str, Any]:
        """Parquet metadata oku"""
        parquet_file = pq.ParquetFile(filename)
        metadata = parquet_file.metadata

        return {
            'num_rows': metadata.num_rows,
            'num_columns': metadata.num_columns,
            'num_row_groups': metadata.num_row_groups,
            'created_by': metadata.created_by,
            'serialized_size': metadata.serialized_size
        }

    @staticmethod
    def append_to_parquet(new_data: List[Dict], filename: str):
        """Parquet dosyasına veri ekle"""
        # Mevcut veriyi oku
        try:
            existing_df = pd.read_parquet(filename)
            new_df = pd.DataFrame(new_data)
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        except FileNotFoundError:
            combined_df = pd.DataFrame(new_data)

        # Yeniden yaz
        combined_df.to_parquet(filename, engine='pyarrow', index=False)

# Kullanım
data = [
    {'id': 1, 'name': 'Ali', 'age': 30, 'city': 'Istanbul', 'salary': 50000},
    {'id': 2, 'name': 'Ayşe', 'age': 25, 'city': 'Ankara', 'salary': 45000},
    {'id': 3, 'name': 'Mehmet', 'age': 35, 'city': 'Izmir', 'salary': 55000}
]

# ParquetProcessor.write_parquet(data, 'employees.parquet', compression='snappy')
# df = ParquetProcessor.read_parquet('employees.parquet', columns=['name', 'salary'])
# metadata = ParquetProcessor.get_parquet_metadata('employees.parquet')
# print(metadata)
```

### HDF5 Files
```python
"""
HDF5: Hierarchical Data Format (scientific/numerical data)
pip install h5py numpy
"""
import h5py
import numpy as np
from typing import Dict, Any

class HDF5Processor:
    """HDF5 dosya işlemleri"""

    @staticmethod
    def write_hdf5(filename: str, datasets: Dict[str, np.ndarray],
                   compression: str = 'gzip'):
        """
        HDF5 dosyası yaz
        datasets: {name: numpy_array}
        """
        with h5py.File(filename, 'w') as f:
            for name, data in datasets.items():
                f.create_dataset(name, data=data, compression=compression)

    @staticmethod
    def read_hdf5(filename: str, dataset_name: str = None) -> Dict[str, np.ndarray]:
        """HDF5 dosyası oku"""
        result = {}
        with h5py.File(filename, 'r') as f:
            if dataset_name:
                # Belirli bir dataset oku
                result[dataset_name] = f[dataset_name][:]
            else:
                # Tüm datasets'leri oku
                for key in f.keys():
                    result[key] = f[key][:]
        return result

    @staticmethod
    def append_to_hdf5(filename: str, dataset_name: str,
                       new_data: np.ndarray):
        """HDF5'e veri ekle (resizable dataset)"""
        with h5py.File(filename, 'a') as f:
            if dataset_name in f:
                # Resize and append
                dataset = f[dataset_name]
                old_size = dataset.shape[0]
                new_size = old_size + new_data.shape[0]
                dataset.resize(new_size, axis=0)
                dataset[old_size:] = new_data
            else:
                # Create new resizable dataset
                maxshape = (None,) + new_data.shape[1:]
                f.create_dataset(dataset_name,
                               data=new_data,
                               maxshape=maxshape,
                               chunks=True)

    @staticmethod
    def create_hierarchical_hdf5(filename: str):
        """
        Hierarchical structure oluştur
        /group1/dataset1
        /group1/group2/dataset2
        """
        with h5py.File(filename, 'w') as f:
            # Group oluştur
            group1 = f.create_group('experiments')
            group2 = group1.create_group('exp001')

            # Datasets ekle
            group2.create_dataset('data', data=np.random.rand(100, 10))
            group2.create_dataset('labels', data=np.random.randint(0, 5, 100))

            # Attributes ekle
            group2.attrs['description'] = 'First experiment'
            group2.attrs['date'] = '2024-01-01'

# Kullanım
datasets = {
    'temperature': np.random.rand(1000, 50),  # 1000 timesteps, 50 sensors
    'pressure': np.random.rand(1000, 50),
    'humidity': np.random.rand(1000, 50)
}

# HDF5Processor.write_hdf5('sensor_data.h5', datasets, compression='gzip')
# data = HDF5Processor.read_hdf5('sensor_data.h5', 'temperature')
# print(data['temperature'].shape)
```

---

## Compression & Archives

### Compression Utilities
```python
import gzip
import bz2
import lzma
import zlib
from pathlib import Path
from typing import Union

class CompressionManager:
    """Çoklu compression format desteği"""

    ALGORITHMS = {
        'gzip': (gzip.open, '.gz'),
        'bz2': (bz2.open, '.bz2'),
        'lzma': (lzma.open, '.xz'),
    }

    @classmethod
    def compress_file(cls, input_file: str, algorithm: str = 'gzip',
                     compression_level: int = 9) -> str:
        """
        Dosyayı sıkıştır
        algorithm: 'gzip', 'bz2', 'lzma'
        compression_level: 1-9 (9 = max compression)
        """
        if algorithm not in cls.ALGORITHMS:
            raise ValueError(f"Unknown algorithm: {algorithm}")

        opener, ext = cls.ALGORITHMS[algorithm]
        output_file = f"{input_file}{ext}"

        with open(input_file, 'rb') as f_in:
            with opener(output_file, 'wb', compresslevel=compression_level) as f_out:
                # Chunk'lar halinde oku ve yaz (büyük dosyalar için)
                while chunk := f_in.read(1024 * 1024):  # 1MB chunks
                    f_out.write(chunk)

        return output_file

    @classmethod
    def decompress_file(cls, input_file: str, output_file: str = None) -> str:
        """Dosyayı açıkla"""
        # Uzantıdan algorithm'ı belirle
        path = Path(input_file)
        algorithm = None

        for alg, (opener, ext) in cls.ALGORITHMS.items():
            if path.suffix == ext:
                algorithm = alg
                break

        if not algorithm:
            raise ValueError(f"Cannot determine compression type: {input_file}")

        if not output_file:
            output_file = str(path.with_suffix(''))

        opener, _ = cls.ALGORITHMS[algorithm]

        with opener(input_file, 'rb') as f_in:
            with open(output_file, 'wb') as f_out:
                while chunk := f_in.read(1024 * 1024):
                    f_out.write(chunk)

        return output_file

    @staticmethod
    def compress_string(data: Union[str, bytes], algorithm: str = 'gzip') -> bytes:
        """String/bytes sıkıştır"""
        if isinstance(data, str):
            data = data.encode('utf-8')

        if algorithm == 'gzip':
            return gzip.compress(data)
        elif algorithm == 'bz2':
            return bz2.compress(data)
        elif algorithm == 'lzma':
            return lzma.compress(data)
        elif algorithm == 'zlib':
            return zlib.compress(data)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

    @staticmethod
    def decompress_string(data: bytes, algorithm: str = 'gzip') -> bytes:
        """Sıkıştırılmış string/bytes aç"""
        if algorithm == 'gzip':
            return gzip.decompress(data)
        elif algorithm == 'bz2':
            return bz2.decompress(data)
        elif algorithm == 'lzma':
            return lzma.decompress(data)
        elif algorithm == 'zlib':
            return zlib.decompress(data)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

    @classmethod
    def compare_algorithms(cls, data: Union[str, bytes]) -> dict:
        """Farklı algoritmaları karşılaştır"""
        if isinstance(data, str):
            data = data.encode('utf-8')

        original_size = len(data)
        results = {'original_size': original_size}

        for algorithm in ['gzip', 'bz2', 'lzma', 'zlib']:
            compressed = cls.compress_string(data, algorithm)
            results[algorithm] = {
                'compressed_size': len(compressed),
                'ratio': len(compressed) / original_size,
                'reduction': f"{(1 - len(compressed)/original_size) * 100:.1f}%"
            }

        return results

# Kullanım
# Dosya sıkıştırma
# compressed = CompressionManager.compress_file('large_file.txt', 'gzip', 9)
# decompressed = CompressionManager.decompress_file(compressed)

# String sıkıştırma
text = "Lorem ipsum " * 1000
results = CompressionManager.compare_algorithms(text)
for algo, stats in results.items():
    if algo != 'original_size':
        print(f"{algo}: {stats['reduction']} reduction")
```

### Archive Handling (tar, zip)
```python
import tarfile
import zipfile
from pathlib import Path
from typing import List, Union
import os

class ArchiveManager:
    """TAR ve ZIP archive yönetimi"""

    @staticmethod
    def create_tar(output_file: str, files: List[str],
                   compression: str = 'gz'):
        """
        TAR archive oluştur
        compression: '', 'gz', 'bz2', 'xz'
        """
        mode = f'w:{compression}' if compression else 'w'

        with tarfile.open(output_file, mode) as tar:
            for file in files:
                tar.add(file, arcname=Path(file).name)

    @staticmethod
    def extract_tar(archive_file: str, extract_path: str = '.'):
        """TAR archive'i çıkart"""
        with tarfile.open(archive_file, 'r:*') as tar:
            # Güvenlik: path traversal kontrolü
            for member in tar.getmembers():
                if member.name.startswith('/') or '..' in member.name:
                    raise ValueError(f"Unsafe path: {member.name}")

            tar.extractall(extract_path)

    @staticmethod
    def create_zip(output_file: str, files: List[str],
                   compression: int = zipfile.ZIP_DEFLATED):
        """
        ZIP archive oluştur
        compression: ZIP_STORED, ZIP_DEFLATED, ZIP_BZIP2, ZIP_LZMA
        """
        with zipfile.ZipFile(output_file, 'w', compression=compression) as zipf:
            for file in files:
                zipf.write(file, arcname=Path(file).name)

    @staticmethod
    def extract_zip(archive_file: str, extract_path: str = '.',
                   members: List[str] = None):
        """ZIP archive'i çıkart"""
        with zipfile.ZipFile(archive_file, 'r') as zipf:
            # Güvenlik kontrolü
            for name in zipf.namelist():
                if name.startswith('/') or '..' in name:
                    raise ValueError(f"Unsafe path: {name}")

            if members:
                zipf.extractall(extract_path, members=members)
            else:
                zipf.extractall(extract_path)

    @staticmethod
    def list_archive(archive_file: str) -> List[dict]:
        """Archive içeriğini listele"""
        results = []

        if archive_file.endswith(('.tar', '.tar.gz', '.tar.bz2', '.tar.xz')):
            with tarfile.open(archive_file, 'r:*') as tar:
                for member in tar.getmembers():
                    results.append({
                        'name': member.name,
                        'size': member.size,
                        'type': 'file' if member.isfile() else 'dir',
                        'mtime': member.mtime
                    })

        elif archive_file.endswith('.zip'):
            with zipfile.ZipFile(archive_file, 'r') as zipf:
                for info in zipf.infolist():
                    results.append({
                        'name': info.filename,
                        'size': info.file_size,
                        'compressed_size': info.compress_size,
                        'type': 'dir' if info.is_dir() else 'file'
                    })

        return results

    @staticmethod
    def add_to_archive(archive_file: str, new_files: List[str]):
        """Mevcut archive'e dosya ekle"""
        if archive_file.endswith('.zip'):
            with zipfile.ZipFile(archive_file, 'a') as zipf:
                for file in new_files:
                    zipf.write(file, arcname=Path(file).name)
        else:
            # TAR için: okuma + yazma gerekiyor
            import tempfile
            import shutil

            with tempfile.TemporaryDirectory() as tmpdir:
                # Extract
                ArchiveManager.extract_tar(archive_file, tmpdir)

                # Add new files
                for file in new_files:
                    shutil.copy(file, tmpdir)

                # Recreate archive
                all_files = [os.path.join(tmpdir, f) for f in os.listdir(tmpdir)]

                # Determine compression
                if archive_file.endswith('.tar.gz'):
                    comp = 'gz'
                elif archive_file.endswith('.tar.bz2'):
                    comp = 'bz2'
                elif archive_file.endswith('.tar.xz'):
                    comp = 'xz'
                else:
                    comp = ''

                ArchiveManager.create_tar(archive_file, all_files, comp)

# Kullanım
# files = ['file1.txt', 'file2.txt', 'file3.txt']
# ArchiveManager.create_tar('backup.tar.gz', files, 'gz')
# ArchiveManager.create_zip('backup.zip', files)
#
# contents = ArchiveManager.list_archive('backup.tar.gz')
# for item in contents:
#     print(f"{item['name']}: {item['size']} bytes")
```

---

## Stream Processing

### Stream Processing Class
```python
from typing import Iterator, Callable, Any
import itertools

class StreamProcessor:
    """
    Lazy evaluation ile stream processing
    Büyük veri setleri için memory-efficient
    """

    def __init__(self, source: Iterator):
        self.source = source

    def map(self, func: Callable) -> 'StreamProcessor':
        """Her element'e function uygula"""
        self.source = map(func, self.source)
        return self

    def filter(self, predicate: Callable) -> 'StreamProcessor':
        """Condition'a uyan elementleri filtrele"""
        self.source = filter(predicate, self.source)
        return self

    def take(self, n: int) -> 'StreamProcessor':
        """İlk n elementi al"""
        self.source = itertools.islice(self.source, n)
        return self

    def skip(self, n: int) -> 'StreamProcessor':
        """İlk n elementi atla"""
        self.source = itertools.islice(self.source, n, None)
        return self

    def batch(self, size: int) -> 'StreamProcessor':
        """Elementleri batch'lere böl"""
        def batch_iterator():
            iterator = iter(self.source)
            while True:
                batch = list(itertools.islice(iterator, size))
                if not batch:
                    break
                yield batch

        self.source = batch_iterator()
        return self

    def chunk_by(self, key_func: Callable) -> 'StreamProcessor':
        """Aynı key'e sahip elementleri grupla"""
        self.source = (list(group) for key, group in
                      itertools.groupby(self.source, key_func))
        return self

    def flatten(self) -> 'StreamProcessor':
        """Nested iterables'ı düzleştir"""
        self.source = itertools.chain.from_iterable(self.source)
        return self

    def distinct(self) -> 'StreamProcessor':
        """Duplicate'leri kaldır (order preserved)"""
        def distinct_iterator():
            seen = set()
            for item in self.source:
                # Hashable check
                try:
                    if item not in seen:
                        seen.add(item)
                        yield item
                except TypeError:
                    # Unhashable type
                    yield item

        self.source = distinct_iterator()
        return self

    def peek(self, func: Callable) -> 'StreamProcessor':
        """Her element için side-effect (debug için)"""
        def peek_iterator():
            for item in self.source:
                func(item)
                yield item

        self.source = peek_iterator()
        return self

    def collect(self) -> list:
        """Stream'i list'e çevir (terminal operation)"""
        return list(self.source)

    def reduce(self, func: Callable, initial: Any = None) -> Any:
        """Reduce operation (terminal)"""
        if initial is None:
            return functools.reduce(func, self.source)
        return functools.reduce(func, self.source, initial)

    def count(self) -> int:
        """Element sayısı (terminal)"""
        return sum(1 for _ in self.source)

    def first(self, default=None):
        """İlk elementi al (terminal)"""
        return next(self.source, default)

    def any(self, predicate: Callable = None) -> bool:
        """Herhangi bir element condition'ı sağlıyor mu?"""
        if predicate:
            return any(predicate(item) for item in self.source)
        return any(self.source)

    def all(self, predicate: Callable = None) -> bool:
        """Tüm elementler condition'ı sağlıyor mu?"""
        if predicate:
            return all(predicate(item) for item in self.source)
        return all(self.source)

# Kullanım örnekleri
import functools

# Örnek 1: Log dosyası processing
def process_logs():
    """Log satırlarını streaming ile işle"""
    def read_log_lines(filename):
        with open(filename, 'r') as f:
            for line in f:
                yield line.strip()

    # Stream pipeline
    result = (StreamProcessor(read_log_lines('app.log'))
             .filter(lambda line: 'ERROR' in line)
             .map(lambda line: line.split('|'))
             .filter(lambda parts: len(parts) >= 3)
             .map(lambda parts: {
                 'timestamp': parts[0],
                 'level': parts[1],
                 'message': parts[2]
             })
             .take(100)
             .collect())

    return result

# Örnek 2: Numeric processing
numbers = range(1, 1000000)
result = (StreamProcessor(iter(numbers))
         .filter(lambda x: x % 2 == 0)  # Even numbers
         .map(lambda x: x ** 2)  # Square
         .take(10)  # First 10
         .collect())

print(result)  # [4, 16, 36, 64, 100, 144, 196, 256, 324, 400]

# Örnek 3: Batching
data = range(1, 21)
batches = (StreamProcessor(iter(data))
          .batch(5)
          .collect())

print(batches)  # [[1,2,3,4,5], [6,7,8,9,10], ...]
```

### File Stream Processing
```python
from typing import Iterator, Callable
import csv
import json

class FileStreamProcessor:
    """Dosya streaming işlemleri"""

    @staticmethod
    def stream_lines(filename: str, encoding: str = 'utf-8',
                    skip_empty: bool = True) -> Iterator[str]:
        """Satır satır dosya okuma (lazy)"""
        with open(filename, 'r', encoding=encoding) as f:
            for line in f:
                line = line.rstrip('\n\r')
                if skip_empty and not line:
                    continue
                yield line

    @staticmethod
    def stream_csv(filename: str, **csv_kwargs) -> Iterator[dict]:
        """CSV streaming (DictReader)"""
        with open(filename, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, **csv_kwargs)
            for row in reader:
                yield row

    @staticmethod
    def stream_json_array(filename: str) -> Iterator[dict]:
        """
        JSON array streaming
        [{"id": 1}, {"id": 2}, ...]
        """
        import ijson  # pip install ijson

        with open(filename, 'rb') as f:
            parser = ijson.items(f, 'item')
            for item in parser:
                yield item

    @staticmethod
    def stream_binary_chunks(filename: str, chunk_size: int = 8192) -> Iterator[bytes]:
        """Binary dosya chunk'lar halinde oku"""
        with open(filename, 'rb') as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                yield chunk

    @staticmethod
    def transform_file(input_file: str, output_file: str,
                      transformer: Callable[[str], str],
                      encoding: str = 'utf-8'):
        """
        Dosyayı transform et (streaming)
        Memory-efficient
        """
        with open(input_file, 'r', encoding=encoding) as f_in:
            with open(output_file, 'w', encoding=encoding) as f_out:
                for line in f_in:
                    transformed = transformer(line)
                    f_out.write(transformed)

    @staticmethod
    def merge_sorted_files(*filenames: str, output_file: str,
                          key_func: Callable = None):
        """
        Sorted dosyaları merge et
        External merge sort pattern
        """
        import heapq

        # Her dosya için iterator aç
        iterators = []
        for filename in filenames:
            iterator = FileStreamProcessor.stream_lines(filename)
            try:
                first_line = next(iterator)
                if key_func:
                    iterators.append((key_func(first_line), first_line, iterator, filename))
                else:
                    iterators.append((first_line, first_line, iterator, filename))
            except StopIteration:
                pass

        # Min heap oluştur
        heapq.heapify(iterators)

        # Merge
        with open(output_file, 'w') as f_out:
            while iterators:
                if key_func:
                    key, line, iterator, filename = heapq.heappop(iterators)
                else:
                    line, _, iterator, filename = heapq.heappop(iterators)

                f_out.write(line + '\n')

                # Sonraki satırı al
                try:
                    next_line = next(iterator)
                    if key_func:
                        heapq.heappush(iterators, (key_func(next_line), next_line, iterator, filename))
                    else:
                        heapq.heappush(iterators, (next_line, next_line, iterator, filename))
                except StopIteration:
                    pass

# Kullanım
# CSV stream processing
# for row in FileStreamProcessor.stream_csv('users.csv'):
#     if int(row['age']) > 30:
#         print(row['name'])

# File transformation
# FileStreamProcessor.transform_file(
#     'input.txt',
#     'output.txt',
#     lambda line: line.upper()
# )
```

---

## Large File Handling

### Large File Processor
```python
import os
from pathlib import Path
from typing import Iterator, Callable, Any
import hashlib
import tempfile
import shutil

class LargeFileProcessor:
    """
    Büyük dosya işleme utilities
    Memory-efficient patterns
    """

    @staticmethod
    def get_file_size(filename: str) -> int:
        """Dosya boyutunu al"""
        return os.path.getsize(filename)

    @staticmethod
    def get_line_count(filename: str, encoding: str = 'utf-8') -> int:
        """Satır sayısını hızlıca hesapla"""
        count = 0
        with open(filename, 'rb') as f:
            # Binary mode daha hızlı
            for _ in f:
                count += 1
        return count

    @staticmethod
    def split_file(filename: str, num_parts: int = None,
                  chunk_size: int = None, output_dir: str = None) -> list:
        """
        Dosyayı parçalara böl
        num_parts VEYA chunk_size (bytes) belirt
        """
        if not output_dir:
            output_dir = Path(filename).parent

        file_size = LargeFileProcessor.get_file_size(filename)
        base_name = Path(filename).stem

        if num_parts:
            chunk_size = file_size // num_parts
        elif not chunk_size:
            raise ValueError("num_parts veya chunk_size belirtilmeli")

        output_files = []
        part_num = 0

        with open(filename, 'rb') as f_in:
            while True:
                chunk = f_in.read(chunk_size)
                if not chunk:
                    break

                output_file = os.path.join(output_dir, f"{base_name}.part{part_num:04d}")
                with open(output_file, 'wb') as f_out:
                    f_out.write(chunk)

                output_files.append(output_file)
                part_num += 1

        return output_files

    @staticmethod
    def merge_files(input_files: list, output_file: str):
        """Dosyaları birleştir"""
        with open(output_file, 'wb') as f_out:
            for input_file in input_files:
                with open(input_file, 'rb') as f_in:
                    shutil.copyfileobj(f_in, f_out, length=1024*1024)  # 1MB buffer

    @staticmethod
    def compute_hash(filename: str, algorithm: str = 'sha256',
                    chunk_size: int = 8192) -> str:
        """
        Dosya hash'i hesapla (streaming)
        algorithm: 'md5', 'sha1', 'sha256', 'sha512'
        """
        hasher = hashlib.new(algorithm)

        with open(filename, 'rb') as f:
            while chunk := f.read(chunk_size):
                hasher.update(chunk)

        return hasher.hexdigest()

    @staticmethod
    def deduplicate_lines(input_file: str, output_file: str,
                         case_sensitive: bool = True,
                         keep_order: bool = True):
        """
        Duplicate satırları kaldır
        Memory-efficient (external sorting kullanır)
        """
        if keep_order:
            # Order-preserving deduplication
            seen = set()
            with open(input_file, 'r', encoding='utf-8') as f_in:
                with open(output_file, 'w', encoding='utf-8') as f_out:
                    for line in f_in:
                        key = line if case_sensitive else line.lower()
                        if key not in seen:
                            seen.add(key)
                            f_out.write(line)
        else:
            # Sorted deduplication (daha memory-efficient)
            with tempfile.TemporaryDirectory() as tmpdir:
                sorted_file = os.path.join(tmpdir, 'sorted.txt')

                # Sort file
                LargeFileProcessor.external_sort(input_file, sorted_file,
                                                case_sensitive=case_sensitive)

                # Remove duplicates
                with open(sorted_file, 'r', encoding='utf-8') as f_in:
                    with open(output_file, 'w', encoding='utf-8') as f_out:
                        prev_line = None
                        for line in f_in:
                            key = line if case_sensitive else line.lower()
                            if key != prev_line:
                                f_out.write(line)
                                prev_line = key

    @staticmethod
    def external_sort(input_file: str, output_file: str,
                     chunk_size: int = 100_000,
                     case_sensitive: bool = True):
        """
        External sorting (büyük dosyalar için)
        Chunk'lara böl, her chunk'ı sort et, merge et
        """
        import tempfile

        # Temporary directory
        with tempfile.TemporaryDirectory() as tmpdir:
            # Phase 1: Sort chunks
            chunk_files = []
            chunk_num = 0

            with open(input_file, 'r', encoding='utf-8') as f:
                while True:
                    lines = list(itertools.islice(f, chunk_size))
                    if not lines:
                        break

                    # Sort chunk
                    if case_sensitive:
                        lines.sort()
                    else:
                        lines.sort(key=str.lower)

                    # Save chunk
                    chunk_file = os.path.join(tmpdir, f'chunk_{chunk_num:04d}.txt')
                    with open(chunk_file, 'w', encoding='utf-8') as chunk_f:
                        chunk_f.writelines(lines)

                    chunk_files.append(chunk_file)
                    chunk_num += 1

            # Phase 2: Merge sorted chunks
            key_func = None if case_sensitive else str.lower
            FileStreamProcessor.merge_sorted_files(*chunk_files,
                                                  output_file=output_file,
                                                  key_func=key_func)

    @staticmethod
    def tail_file(filename: str, num_lines: int = 10) -> list:
        """
        Dosyanın son N satırını al (tail komutu)
        Büyük dosyalar için optimize edilmiş
        """
        with open(filename, 'rb') as f:
            # Dosya sonuna git
            f.seek(0, 2)
            file_size = f.tell()

            # Geriye doğru oku
            block_size = 1024
            blocks = []
            lines_found = 0
            position = file_size

            while position > 0 and lines_found < num_lines:
                # Block size ayarla
                read_size = min(block_size, position)
                position -= read_size

                # Oku
                f.seek(position)
                block = f.read(read_size)
                blocks.append(block)

                # Satır sayısını hesapla
                lines_found = sum(block.count(b'\n') for block in blocks)

            # Tüm blocks'ları birleştir
            data = b''.join(reversed(blocks))
            lines = data.decode('utf-8', errors='ignore').splitlines()

            # Son N satırı döndür
            return lines[-num_lines:]

import itertools

# Kullanım örnekleri
# File splitting
# parts = LargeFileProcessor.split_file('large_file.txt', num_parts=10)
# LargeFileProcessor.merge_files(parts, 'merged.txt')

# Hash computation
# file_hash = LargeFileProcessor.compute_hash('file.txt', 'sha256')
# print(f"SHA256: {file_hash}")

# Deduplication
# LargeFileProcessor.deduplicate_lines('input.txt', 'output.txt')

# External sort
# LargeFileProcessor.external_sort('unsorted.txt', 'sorted.txt', chunk_size=100000)

# Tail
# last_lines = LargeFileProcessor.tail_file('log.txt', 20)
# for line in last_lines:
#     print(line)
```

---

## Production Patterns

### File Processing Pipeline
```python
from dataclasses import dataclass
from typing import Callable, Any, List
from enum import Enum
import logging
from datetime import datetime
import traceback

class ProcessingStatus(Enum):
    """Processing durumu"""
    PENDING = "pending"
    PROCESSING = "processing"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"

@dataclass
class ProcessingResult:
    """Processing sonucu"""
    status: ProcessingStatus
    input_file: str
    output_file: str = None
    records_processed: int = 0
    errors: List[str] = None
    start_time: datetime = None
    end_time: datetime = None
    metadata: dict = None

    @property
    def duration(self):
        """İşlem süresi"""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None

class FileProcessingPipeline:
    """
    Production-ready file processing pipeline
    Error handling, logging, monitoring
    """

    def __init__(self, name: str = "Pipeline"):
        self.name = name
        self.steps = []
        self.logger = logging.getLogger(f"Pipeline.{name}")
        self.results = []

    def add_step(self, name: str, processor: Callable,
                 skip_on_error: bool = False):
        """Pipeline step ekle"""
        self.steps.append({
            'name': name,
            'processor': processor,
            'skip_on_error': skip_on_error
        })
        return self

    def process_file(self, input_file: str, output_file: str = None,
                    **kwargs) -> ProcessingResult:
        """Tek dosya işle"""
        result = ProcessingResult(
            status=ProcessingStatus.PENDING,
            input_file=input_file,
            output_file=output_file,
            start_time=datetime.now(),
            errors=[],
            metadata={}
        )

        try:
            self.logger.info(f"Processing started: {input_file}")
            result.status = ProcessingStatus.PROCESSING

            # Her step'i çalıştır
            current_file = input_file

            for step in self.steps:
                step_name = step['name']
                processor = step['processor']
                skip_on_error = step['skip_on_error']

                self.logger.debug(f"Running step: {step_name}")

                try:
                    # Step'i çalıştır
                    step_result = processor(current_file, **kwargs)

                    # Sonucu metadata'ya ekle
                    result.metadata[step_name] = step_result

                    # Eğer step yeni dosya döndürdüyse, onu kullan
                    if isinstance(step_result, str) and os.path.exists(step_result):
                        current_file = step_result

                except Exception as e:
                    error_msg = f"Step '{step_name}' failed: {str(e)}"
                    self.logger.error(error_msg)
                    self.logger.debug(traceback.format_exc())
                    result.errors.append(error_msg)

                    if not skip_on_error:
                        raise

            # Son dosya output olarak kaydet
            if current_file != input_file:
                result.output_file = current_file

            result.status = ProcessingStatus.SUCCESS
            self.logger.info(f"Processing completed: {input_file}")

        except Exception as e:
            result.status = ProcessingStatus.FAILED
            result.errors.append(str(e))
            self.logger.error(f"Processing failed: {input_file} - {str(e)}")

        finally:
            result.end_time = datetime.now()
            self.results.append(result)

        return result

    def process_batch(self, input_files: List[str],
                     output_dir: str = None,
                     parallel: bool = False,
                     max_workers: int = 4) -> List[ProcessingResult]:
        """Birden fazla dosya işle"""
        if parallel:
            from concurrent.futures import ThreadPoolExecutor, as_completed

            results = []
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all tasks
                future_to_file = {
                    executor.submit(self.process_file, file,
                                  self._get_output_path(file, output_dir)): file
                    for file in input_files
                }

                # Collect results
                for future in as_completed(future_to_file):
                    file = future_to_file[future]
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        self.logger.error(f"Batch processing error for {file}: {e}")

            return results
        else:
            # Sequential processing
            return [
                self.process_file(file, self._get_output_path(file, output_dir))
                for file in input_files
            ]

    def _get_output_path(self, input_file: str, output_dir: str) -> str:
        """Output path oluştur"""
        if not output_dir:
            return None

        filename = Path(input_file).name
        return os.path.join(output_dir, filename)

    def get_summary(self) -> dict:
        """Pipeline özeti"""
        total = len(self.results)
        success = sum(1 for r in self.results if r.status == ProcessingStatus.SUCCESS)
        failed = sum(1 for r in self.results if r.status == ProcessingStatus.FAILED)

        return {
            'total_files': total,
            'successful': success,
            'failed': failed,
            'success_rate': f"{(success/total*100 if total > 0 else 0):.1f}%",
            'total_records': sum(r.records_processed for r in self.results),
            'avg_duration': sum(r.duration or 0 for r in self.results) / total if total > 0 else 0
        }

# Kullanım örneği
def validate_step(filename, **kwargs):
    """Validation step"""
    # Check file exists and not empty
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File not found: {filename}")

    if os.path.getsize(filename) == 0:
        raise ValueError(f"File is empty: {filename}")

    return {'valid': True}

def transform_step(filename, **kwargs):
    """Transformation step"""
    output_file = filename + '.transformed'

    # Transform logic
    with open(filename, 'r') as f_in:
        with open(output_file, 'w') as f_out:
            for line in f_in:
                f_out.write(line.upper())

    return output_file

def compress_step(filename, **kwargs):
    """Compression step"""
    output_file = filename + '.gz'
    CompressionManager.compress_file(filename, 'gzip')
    return output_file

# Pipeline oluştur
pipeline = (FileProcessingPipeline("ETL-Pipeline")
           .add_step("validate", validate_step)
           .add_step("transform", transform_step)
           .add_step("compress", compress_step, skip_on_error=True))

# Tek dosya
# result = pipeline.process_file('input.txt')
# print(f"Status: {result.status}, Duration: {result.duration}s")

# Batch processing
# results = pipeline.process_batch(['file1.txt', 'file2.txt', 'file3.txt'],
#                                  output_dir='./output',
#                                  parallel=True)
# summary = pipeline.get_summary()
# print(summary)
```

### Error Handling & Recovery
```python
import os
import shutil
import tempfile
from contextlib import contextmanager
from typing import Callable
import logging

class SafeFileProcessor:
    """
    Güvenli dosya işleme
    Atomic operations, rollback, backup
    """

    @staticmethod
    @contextmanager
    def atomic_write(filename: str, mode: str = 'w',
                    encoding: str = 'utf-8', **kwargs):
        """
        Atomic file write
        Başarısız olursa orijinal dosya korunur
        """
        # Temporary file oluştur
        tmpdir = os.path.dirname(filename) or '.'
        tmpfd, tmpname = tempfile.mkstemp(dir=tmpdir, prefix='.tmp_')

        try:
            with os.fdopen(tmpfd, mode, encoding=encoding, **kwargs) as f:
                yield f

            # Başarılı: temp dosyayı asıl dosyaya taşı
            shutil.move(tmpname, filename)

        except Exception:
            # Hata: temp dosyayı sil
            try:
                os.remove(tmpname)
            except OSError:
                pass
            raise

    @staticmethod
    @contextmanager
    def backup_file(filename: str, backup_suffix: str = '.backup'):
        """
        Dosya backup'ı ile işlem
        Hata olursa restore eder
        """
        backup_name = filename + backup_suffix

        # Backup oluştur
        if os.path.exists(filename):
            shutil.copy2(filename, backup_name)
            backup_created = True
        else:
            backup_created = False

        try:
            yield filename

            # Başarılı: backup'ı sil
            if backup_created:
                os.remove(backup_name)

        except Exception:
            # Hata: restore et
            if backup_created:
                shutil.move(backup_name, filename)
            raise

    @staticmethod
    def retry_operation(operation: Callable, max_retries: int = 3,
                       retry_delay: float = 1.0,
                       exceptions: tuple = (IOError, OSError)):
        """
        İşlemi retry et
        Geçici hatalar için
        """
        import time

        last_exception = None

        for attempt in range(max_retries):
            try:
                return operation()
            except exceptions as e:
                last_exception = e
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (attempt + 1))
                    logging.warning(f"Retry {attempt + 1}/{max_retries}: {e}")

        # Tüm retry'lar başarısız
        raise last_exception

    @staticmethod
    def safe_delete(filename: str, trash: bool = True):
        """
        Güvenli dosya silme
        trash=True: dosyayı trash'e taşı
        """
        if not os.path.exists(filename):
            return

        if trash:
            # Trash directory oluştur
            trash_dir = os.path.join(os.path.dirname(filename) or '.', '.trash')
            os.makedirs(trash_dir, exist_ok=True)

            # Unique name oluştur
            from datetime import datetime
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            basename = os.path.basename(filename)
            trash_name = os.path.join(trash_dir, f"{timestamp}_{basename}")

            # Taşı
            shutil.move(filename, trash_name)
        else:
            # Direkt sil
            os.remove(filename)

# Kullanım
# Atomic write
try:
    with SafeFileProcessor.atomic_write('important.txt') as f:
        f.write('Critical data\n')
        f.write('More critical data\n')
        # raise Exception("Something went wrong")  # Test: file won't be created
except Exception as e:
    print(f"Error: {e}")

# Backup & restore
try:
    with SafeFileProcessor.backup_file('data.txt'):
        # Modify file
        with open('data.txt', 'w') as f:
            f.write('New data\n')
            # raise Exception("Error!")  # Test: file will be restored
except Exception as e:
    print(f"Error: {e}, file restored")

# Retry
def unreliable_operation():
    import random
    if random.random() < 0.7:  # 70% fail rate
        raise IOError("Network error")
    return "Success!"

# result = SafeFileProcessor.retry_operation(unreliable_operation, max_retries=5)
# print(result)
```

## Özet

Bu dokümanda Advanced File Processing konularını detaylıca inceledik:

1. **Binary Operations**: struct, custom formats, efficient binary I/O
2. **Memory-Mapped Files**: mmap, shared memory, fast access
3. **Serialization**: pickle, JSON, YAML, MessagePack
4. **Data Formats**: XML, Parquet, HDF5
5. **Compression**: gzip, bz2, lzma, tar, zip
6. **Stream Processing**: Lazy evaluation, pipelines, memory-efficient
7. **Large Files**: splitting, merging, external sort, deduplication
8. **Production Patterns**: pipelines, error handling, atomic operations

Production sistemlerde dosya işlemleri kritik öneme sahiptir. Memory-efficient patterns ve proper error handling kullanmak çok önemlidir.
