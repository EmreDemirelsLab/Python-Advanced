# Python Advanced - İleri Seviye Python Eğitimi

Hoş geldiniz! Bu repo, Python programlama dilinin ileri seviye konularını kapsamlı bir şekilde öğrenmek için hazırlanmış profesyonel bir eğitim kaynağıdır.

## 🎯 Hedef Kitle

Bu repo **Python temellerini bilen** ve **ileri seviye konulara geçmek isteyen** geliştiriciler için hazırlanmıştır.

**Ön Koşullar:**
- Python temelleri (değişkenler, veri tipleri, operatörler)
- Kontrol yapıları (if/else, loops)
- Fonksiyonlar ve temel OOP
- Liste, tuple, dictionary, set kullanımı

> 💡 **Not:** Temel konular için [Python Fundamentals](https://github.com/EmreDemirelsLab/Python-Fundemantals) reposuna bakabilirsiniz.

---

## 📚 İçindekiler

### İleri Seviye Konular

1. **[Advanced Object-Oriented Programming](./01-Advanced-OOP/)** - İleri Seviye Nesne Yönelimli Programlama
   - Abstract Classes (ABC modülü)
   - Multiple Inheritance & MRO
   - Metaclasses
   - Descriptors
   - Magic Methods derinlemesine
   - Dataclasses

2. **[Decorators and Closures](./02-Decorators-and-Closures/)** - Dekoratörler ve Closure'lar
   - Function Decorators
   - Class Decorators
   - Decorator Chaining
   - functools modülü
   - Parametreli Decorators
   - Closure kavramı

3. **[Generators and Iterators](./03-Generators-and-Iterators/)** - Generatörler ve Iteratörler
   - Iterator Protocol
   - Generator Functions (yield)
   - Generator Expressions
   - itertools modülü
   - yield from
   - Async Generators

4. **[Context Managers](./04-Context-Managers/)** - Context Manager'lar
   - with Statement
   - contextlib modülü
   - Custom Context Managers
   - __enter__ ve __exit__
   - contextmanager decorator
   - Resource Management

5. **[Async Programming](./05-Async-Programming/)** - Asenkron Programlama
   - Async/Await
   - asyncio modülü
   - Coroutines
   - Tasks ve Event Loop
   - Async Context Managers
   - concurrent.futures

6. **[Multithreading and Multiprocessing](./06-Multithreading-Multiprocessing/)** - Çoklu İş Parçacığı
   - Threading vs Multiprocessing
   - GIL (Global Interpreter Lock)
   - Thread Safety
   - Locks, Semaphores, Events
   - Process Pools
   - Queue & Communication

7. **[Advanced Data Structures](./07-Advanced-Data-Structures/)** - İleri Veri Yapıları
   - collections modülü
   - deque, Counter, defaultdict
   - heapq (Priority Queues)
   - bisect (Binary Search)
   - Custom Data Structures
   - Memory Efficiency

8. **[Regular Expressions Advanced](./08-Regex-Advanced/)** - İleri Regex
   - Complex Patterns
   - Lookahead & Lookbehind
   - Named Groups
   - Performance Optimization
   - Text Processing
   - Web Scraping Patterns

9. **[Testing and Debugging](./09-Testing-and-Debugging/)** - Test ve Hata Ayıklama
   - unittest modülü
   - pytest Framework
   - Mocking & Fixtures
   - Test Coverage
   - pdb Debugger
   - Profiling & Performance

10. **[Type Hints and Static Type Checking](./10-Type-Hints/)** - Tip İpuçları
    - Type Annotations
    - typing modülü
    - Generics
    - Protocol Classes
    - mypy ile Type Checking
    - TypedDict

11. **[Design Patterns](./11-Design-Patterns/)** - Tasarım Desenleri
    - Creational Patterns
    - Structural Patterns
    - Behavioral Patterns
    - Singleton, Factory, Observer
    - Dependency Injection
    - Real-world Applications

12. **[Database Programming](./12-Database-Programming/)** - Veritabanı Programlama
    - SQLite Advanced
    - SQLAlchemy ORM
    - Connection Pooling
    - Transactions & ACID
    - MongoDB ile Python
    - Database Migrations

13. **[Web Development Fundamentals](./13-Web-Development/)** - Web Geliştirme
    - HTTP Protocol
    - REST API Concepts
    - Flask Framework
    - FastAPI Introduction
    - Authentication & Authorization
    - API Best Practices

14. **[Web Scraping](./14-Web-Scraping/)** - Web Kazıma
    - requests modülü
    - BeautifulSoup4
    - Selenium Automation
    - API Rate Limiting
    - Error Handling
    - Ethics & Legal

15. **[Advanced File and Data Processing](./15-Advanced-File-Processing/)** - Dosya İşleme
    - Binary Files
    - Memory-Mapped Files
    - Serialization Advanced
    - Data Formats (JSON, YAML, XML)
    - Compression
    - Stream Processing

16. **[Package Management](./16-Package-Management/)** - Paket Yönetimi
    - pip Advanced
    - Virtual Environments
    - Poetry
    - Package Creation
    - PyPI Publishing
    - Dependency Management

17. **[Performance Optimization](./17-Performance-Optimization/)** - Performans
    - Profiling Tools
    - Memory Profiling
    - Algorithm Optimization
    - Caching Strategies
    - Lazy Evaluation
    - NumPy Performance

18. **[Memory Management](./18-Memory-Management/)** - Bellek Yönetimi
    - Garbage Collection
    - Reference Counting
    - Memory Leaks
    - __slots__
    - Weak References
    - Memory Optimization

19. **[Networking and Sockets](./19-Networking/)** - Ağ Programlama
    - Socket Programming
    - TCP/UDP Protocols
    - Client-Server Architecture
    - HTTP Requests
    - WebSockets
    - Network Security

20. **[Advanced Python Internals](./20-Python-Internals/)** - Python İç Yapısı
    - Metaclasses Deep Dive
    - Bytecode
    - Import System
    - sys & os Advanced
    - Signal Handling
    - C Extensions

---

## 🎯 Nasıl Kullanılır?

### Öğrenme Yolu

1. **Temel Bilgi Kontrolü**: Python temellerine hakimseniz devam edin
2. **Sıralı İlerleme**: Konuları sırayla takip edin (bazı konular birbirine bağlıdır)
3. **Her Konu İçin**:
   - `topic.md` dosyasını okuyun (detaylı açıklamalar ve örnekler)
   - Örnekleri kendi ortamınızda çalıştırın
   - `exercises.py` dosyasındaki alıştırmaları yapın
   - Çözümleri inceleyin ve karşılaştırın

4. **Pratik Yapın**: Her konunun sonundaki challenge problemlerini mutlaka çözün

### Dosya Yapısı

Her konu klasörü şu yapıdadır:

```
01-Advanced-OOP/
├── topic.md          # Detaylı konu anlatımı + örnekler
└── exercises.py      # Alıştırmalar + Çözümler
```

---

## 💡 İpuçları

- **Sabırlı Olun**: İleri seviye konular zaman alır
- **Pratik Yapın**: Sadece okumak yeterli değil, mutlaka kod yazın
- **Hata Mesajlarını Okuyun**: En iyi öğretmen hatalarınızdır
- **Araştırın**: Anlamadığınız kavramları farklı kaynaklardan araştırın
- **Projeler Yapın**: Öğrendiklerinizi gerçek projelerde kullanın
- **Toplulukla Etkileşim**: Stack Overflow, Reddit, GitHub'da aktif olun

---

## 🚀 Başlamadan Önce

### Gereksinimler

```bash
# Python 3.10+ önerilir
python --version

# Sanal ortam oluşturun
python -m venv venv

# Aktif edin
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# Gerekli paketleri yükleyin (her konu için farklı olabilir)
pip install -r requirements.txt
```

---

## 📖 Ek Kaynaklar

### Resmi Dokümantasyon
- [Python Official Documentation](https://docs.python.org/3/)
- [Python Enhancement Proposals (PEPs)](https://www.python.org/dev/peps/)
- [Python Package Index (PyPI)](https://pypi.org/)

### Önerilen Kitaplar
- "Fluent Python" - Luciano Ramalho
- "Python Cookbook" - David Beazley & Brian K. Jones
- "Effective Python" - Brett Slatkin
- "Python in a Nutshell" - Alex Martelli

### Online Kaynaklar
- [Real Python](https://realpython.com/) - Advanced tutorials
- [Python.org Advanced Tutorial](https://docs.python.org/3/tutorial/)
- [PyCon Talks](https://www.youtube.com/c/PyConUS) - Conference videos
- [Planet Python](https://planetpython.org/) - Blog aggregator

### Topluluklar
- [r/Python](https://www.reddit.com/r/Python/) - Reddit community
- [Python Discord](https://discord.gg/python) - Discord server
- [Stack Overflow](https://stackoverflow.com/questions/tagged/python) - Q&A
- [Python Forum](https://python-forum.io/) - Discussion forum

---

## 🎓 Öğrenme Yol Haritası

```
Temel Seviye (Fundamentals) ✅
    ↓
İleri OOP & Decorators (1-2) 📍
    ↓
Generators & Context Managers (3-4)
    ↓
Async & Concurrency (5-6)
    ↓
Advanced Data & Testing (7-9)
    ↓
Type Hints & Patterns (10-11)
    ↓
Database & Web (12-14)
    ↓
Files & Packages (15-16)
    ↓
Performance & Memory (17-18)
    ↓
Networking & Internals (19-20)
    ↓
Python Expert! 🎉
```

---

## 🤝 Katkıda Bulunma

Bu repo sürekli gelişmektedir. Katkıda bulunmak isterseniz:
- Hata bulursanız [issue](https://github.com/EmreDemirelsLab/Python-Advanced/issues) açabilirsiniz
- Yeni örnekler veya alıştırmalar eklemek için pull request gönderebilirsiniz
- İyileştirme önerilerinizi paylaşabilirsiniz
- Yazım hatalarını düzeltebilirsiniz

---

## 📊 İlerleme Takibi

Öğrendiğiniz konuları işaretleyerek ilerlemenizi takip edebilirsiniz:

- [ ] 01 - Advanced OOP
- [ ] 02 - Decorators and Closures
- [ ] 03 - Generators and Iterators
- [ ] 04 - Context Managers
- [ ] 05 - Async Programming
- [ ] 06 - Multithreading & Multiprocessing
- [ ] 07 - Advanced Data Structures
- [ ] 08 - Regular Expressions Advanced
- [ ] 09 - Testing and Debugging
- [ ] 10 - Type Hints
- [ ] 11 - Design Patterns
- [ ] 12 - Database Programming
- [ ] 13 - Web Development
- [ ] 14 - Web Scraping
- [ ] 15 - Advanced File Processing
- [ ] 16 - Package Management
- [ ] 17 - Performance Optimization
- [ ] 18 - Memory Management
- [ ] 19 - Networking
- [ ] 20 - Python Internals

---

## 📝 Lisans

Bu proje eğitim amaçlıdır ve herkes tarafından özgürce kullanılabilir.

---

## 🌟 Teşekkürler

Bu repo'yu yararlı bulduysanız ⭐ vermeyi unutmayın!

**İyi öğrenmeler ve mutlu kodlamalar! 🚀**

*Son güncelleme: 2025*

---

## 📞 İletişim

Sorularınız veya önerileriniz için:
- GitHub Issues: [Python-Advanced/issues](https://github.com/EmreDemirelsLab/Python-Advanced/issues)
- Temel konular için: [Python-Fundamentals](https://github.com/EmreDemirelsLab/Python-Fundemantals)
