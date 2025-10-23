# 15 - Advanced File Processing

## İçerik

Bu bölüm, Python'da ileri seviye dosya işleme tekniklerini kapsar.

### Dosyalar

1. **topic.md** (2,136 satır, 64KB)
   - İleri seviye dosya işleme kavramları
   - 15-20 advanced kod örneği
   - Türkçe açıklamalar
   - Production patterns

2. **exercises.py** (1,666 satır, 50KB)
   - 10 advanced alıştırma (Medium to Expert)
   - Her biri çözümlü
   - Gerçek dünya senaryoları
   - Test fonksiyonları

## Kapsanan Konular

### 1. Binary File Operations
- struct module kullanımı
- Custom binary formats
- Efficient binary I/O

### 2. Memory-Mapped Files (mmap)
- Large file handling
- Shared memory
- Fast pattern matching

### 3. Advanced Serialization
- Pickle (advanced & safe)
- JSON (custom encoders/decoders)
- YAML processing
- MessagePack

### 4. Data Formats
- XML processing
- Parquet files (columnar storage)
- HDF5 files (scientific data)

### 5. Compression & Archives
- gzip, bz2, lzma compression
- tar, zip archive handling
- Compression comparison

### 6. Stream Processing
- Lazy evaluation
- Pipeline pattern
- Memory-efficient processing
- Batch operations

### 7. Large File Handling
- File splitting/merging
- External sorting
- Deduplication
- Hash computation
- Tail operations

### 8. Production Patterns
- File processing pipelines
- Error handling & recovery
- Atomic operations
- Backup & restore
- Retry mechanisms

## Alıştırmalar

1. **Binary Log Parser** (Medium)
   - Binary format parsing
   - Struct usage
   - Log analysis

2. **Memory-Mapped Search** (Medium)
   - mmap kullanımı
   - Pattern matching
   - Line extraction

3. **Multi-Format Converter** (Hard)
   - CSV, JSON, Pickle, MessagePack
   - Format detection
   - Streaming conversion

4. **Large File Deduplicator** (Hard)
   - External sorting
   - Memory-efficient dedup
   - Order-preserving options

5. **Archive Manager** (Medium)
   - tar.gz, zip handling
   - Incremental backups
   - Metadata tracking

6. **Stream Pipeline** (Hard)
   - Lazy evaluation
   - Method chaining
   - Map, filter, batch operations

7. **Log Analyzer** (Hard)
   - Log parsing
   - Pattern matching
   - Statistics & reporting

8. **ETL Pipeline** (Expert)
   - Extract, Transform, Load
   - Multiple data sources
   - Error handling
   - Metrics tracking

9. **Checksum & Integrity** (Medium)
   - Hash computation
   - Manifest creation
   - Verification

10. **Data Migration** (Hard)
    - Schema mapping
    - Type conversion
    - Validation

## Kullanım

### Tests Çalıştırma

```bash
cd "/Users/mac/Emre Demirel/GitHub/Python-Advanced/15-Advanced-File-Processing"
python3 exercises.py
```

### Örnekleri İnceleme

```bash
# topic.md dosyasını okuyun - 20 advanced örnek içerir
# Her örnek production-ready patterns gösterir
```

## Önemli Notlar

- **Memory Efficiency**: Büyük dosyalar için streaming kullanın
- **Error Handling**: Production'da robust error handling şart
- **Atomic Operations**: File corruption'dan kaçının
- **Compression**: Doğru compression algoritması seçin
- **Format Selection**: Use case'e göre format seçin:
  - CSV: Simple tabular data
  - JSON: Hierarchical, human-readable
  - Parquet: Big data, analytics
  - HDF5: Scientific, numerical data
  - MessagePack: Binary, fast serialization

## Gereksinimler

Bazı örnekler için ek kütüphaneler:

```bash
pip install pyarrow pandas  # Parquet support
pip install h5py numpy      # HDF5 support
pip install pyyaml          # YAML support
pip install msgpack         # MessagePack support
pip install ijson           # Streaming JSON
```

## Production Tips

1. **Always use context managers** (`with` statements)
2. **Stream large files** (don't load into memory)
3. **Use appropriate buffer sizes** (8KB-1MB typical)
4. **Implement retry logic** for I/O operations
5. **Validate data** before processing
6. **Create backups** before modifying files
7. **Use atomic writes** for critical data
8. **Monitor memory usage** for large files
9. **Log errors** comprehensively
10. **Test with large files** in development

## Kaynaklar

- Python struct module documentation
- Python mmap module documentation
- Parquet format specification
- HDF5 documentation
- MessagePack specification
