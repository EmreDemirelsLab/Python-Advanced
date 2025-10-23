# Testing ve Debugging (Test ve Hata Ayıklama)

## İçindekiler
1. [unittest Framework](#unittest-framework)
2. [pytest Framework](#pytest-framework)
3. [Test Fixtures](#test-fixtures)
4. [Mocking ve Patching](#mocking-ve-patching)
5. [Parametrized Tests](#parametrized-tests)
6. [Test Coverage](#test-coverage)
7. [pdb Debugger](#pdb-debugger)
8. [Profiling](#profiling)
9. [Memory Profiling](#memory-profiling)
10. [CI/CD Basics](#cicd-basics)

---

## unittest Framework

### Temel unittest Kullanımı

unittest, Python'un standart kütüphanesinde bulunan test framework'üdür.

```python
import unittest
from typing import List

class Calculator:
    """Hesap makinesi sınıfı"""

    @staticmethod
    def add(a: float, b: float) -> float:
        return a + b

    @staticmethod
    def divide(a: float, b: float) -> float:
        if b == 0:
            raise ValueError("Sıfıra bölme hatası")
        return a / b

    @staticmethod
    def factorial(n: int) -> int:
        if n < 0:
            raise ValueError("Negatif sayının faktöriyeli alınamaz")
        if n == 0 or n == 1:
            return 1
        return n * Calculator.factorial(n - 1)

class TestCalculator(unittest.TestCase):
    """Calculator sınıfı için test case'leri"""

    def setUp(self):
        """Her test metodundan önce çalışır"""
        self.calc = Calculator()
        print("Test başlatılıyor...")

    def tearDown(self):
        """Her test metodundan sonra çalışır"""
        print("Test tamamlandı...")

    def test_add(self):
        """Toplama işlemini test et"""
        self.assertEqual(self.calc.add(2, 3), 5)
        self.assertEqual(self.calc.add(-1, 1), 0)
        self.assertEqual(self.calc.add(0.1, 0.2), 0.3)

    def test_divide(self):
        """Bölme işlemini test et"""
        self.assertEqual(self.calc.divide(10, 2), 5)
        self.assertAlmostEqual(self.calc.divide(1, 3), 0.333, places=2)

        # Exception kontrolü
        with self.assertRaises(ValueError):
            self.calc.divide(10, 0)

    def test_factorial(self):
        """Faktöriyel hesaplamasını test et"""
        self.assertEqual(self.calc.factorial(5), 120)
        self.assertEqual(self.calc.factorial(0), 1)

        with self.assertRaises(ValueError):
            self.calc.factorial(-5)

    @unittest.skip("Geçici olarak atlanıyor")
    def test_skipped(self):
        """Bu test atlanacak"""
        pass

    @unittest.skipIf(True, "Koşul sağlandığı için atlanıyor")
    def test_skip_if(self):
        """Koşullu atlama"""
        pass

if __name__ == '__main__':
    unittest.main(verbosity=2)
```

### Test Assertions

```python
import unittest

class TestAssertions(unittest.TestCase):
    """Çeşitli assertion örnekleri"""

    def test_equality_assertions(self):
        """Eşitlik kontrolü"""
        self.assertEqual(1 + 1, 2)
        self.assertNotEqual(1, 2)
        self.assertTrue(True)
        self.assertFalse(False)
        self.assertIs(None, None)
        self.assertIsNot([], [])

    def test_numeric_assertions(self):
        """Sayısal kontroller"""
        self.assertGreater(5, 3)
        self.assertGreaterEqual(5, 5)
        self.assertLess(3, 5)
        self.assertLessEqual(3, 3)
        self.assertAlmostEqual(0.1 + 0.2, 0.3, places=7)

    def test_container_assertions(self):
        """Koleksiyon kontrolleri"""
        self.assertIn(1, [1, 2, 3])
        self.assertNotIn(4, [1, 2, 3])
        self.assertListEqual([1, 2], [1, 2])
        self.assertDictEqual({'a': 1}, {'a': 1})
        self.assertSetEqual({1, 2}, {2, 1})

    def test_type_assertions(self):
        """Tip kontrolleri"""
        self.assertIsInstance("hello", str)
        self.assertIsInstance([], list)
        self.assertNotIsInstance("hello", int)

    def test_exception_assertions(self):
        """Exception kontrolleri"""
        with self.assertRaises(ZeroDivisionError):
            1 / 0

        with self.assertRaises(ValueError) as context:
            int("invalid")

        self.assertIn("invalid literal", str(context.exception))
```

---

## pytest Framework

pytest, modern ve kullanıcı dostu bir test framework'üdür. unittest'e göre daha az boilerplate kod gerektirir.

### Temel pytest Kullanımı

```python
# test_basic_pytest.py
import pytest

def add(a, b):
    return a + b

def divide(a, b):
    if b == 0:
        raise ValueError("Sıfıra bölme hatası")
    return a / b

# Test fonksiyonları "test_" ile başlamalı
def test_add():
    """Basit toplama testi"""
    assert add(2, 3) == 5
    assert add(-1, 1) == 0
    assert add(0, 0) == 0

def test_divide():
    """Bölme testi"""
    assert divide(10, 2) == 5
    assert divide(9, 3) == 3

    # Exception testi
    with pytest.raises(ValueError):
        divide(10, 0)

    # Exception mesajı kontrolü
    with pytest.raises(ValueError, match="Sıfıra bölme"):
        divide(5, 0)

def test_approximate():
    """Yaklaşık değer testi"""
    assert 0.1 + 0.2 == pytest.approx(0.3)
    assert divide(1, 3) == pytest.approx(0.333, abs=0.001)

# pytest markers
@pytest.mark.slow
def test_slow_operation():
    """Yavaş çalışan test"""
    import time
    time.sleep(0.1)
    assert True

@pytest.mark.skip(reason="Henüz implement edilmedi")
def test_not_implemented():
    """Atlanacak test"""
    pass

@pytest.mark.skipif(True, reason="Koşul sağlandı")
def test_conditional_skip():
    """Koşullu atlama"""
    pass

@pytest.mark.xfail(reason="Bilinen bug")
def test_expected_failure():
    """Başarısız olması beklenen test"""
    assert False
```

---

## Test Fixtures

Fixtures, test verilerini ve kaynakları yönetmek için kullanılır.

### pytest Fixtures

```python
# conftest.py veya test dosyası içinde
import pytest
from typing import List, Dict
import tempfile
import json
import sqlite3

# Basit fixture
@pytest.fixture
def sample_data():
    """Basit veri fixture'ı"""
    return [1, 2, 3, 4, 5]

# Setup/Teardown ile fixture
@pytest.fixture
def database_connection():
    """Veritabanı bağlantısı fixture'ı"""
    # Setup
    conn = sqlite3.connect(':memory:')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE users (
            id INTEGER PRIMARY KEY,
            name TEXT,
            email TEXT
        )
    ''')
    conn.commit()

    # Test'e bağlantıyı ver
    yield conn

    # Teardown
    conn.close()

# Scope'lu fixture
@pytest.fixture(scope="module")
def expensive_resource():
    """Module başına bir kez oluşturulan kaynak"""
    print("\n[SETUP] Expensive resource oluşturuluyor...")
    resource = {"data": "expensive to create"}
    yield resource
    print("\n[TEARDOWN] Expensive resource temizleniyor...")

# Parametrized fixture
@pytest.fixture(params=[1, 2, 3])
def number(request):
    """Parametreli fixture"""
    return request.param

# Temp file fixture
@pytest.fixture
def temp_json_file():
    """Geçici JSON dosyası"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        data = {"users": [{"name": "Ali", "age": 30}]}
        json.dump(data, f)
        temp_path = f.name

    yield temp_path

    # Cleanup
    import os
    os.unlink(temp_path)

# Test fonksiyonlarında fixture kullanımı
def test_with_fixture(sample_data):
    """Fixture kullanan test"""
    assert len(sample_data) == 5
    assert sum(sample_data) == 15

def test_database(database_connection):
    """Veritabanı testi"""
    cursor = database_connection.cursor()
    cursor.execute("INSERT INTO users (name, email) VALUES (?, ?)",
                   ("Ahmet", "ahmet@example.com"))
    database_connection.commit()

    cursor.execute("SELECT * FROM users WHERE name = ?", ("Ahmet",))
    result = cursor.fetchone()

    assert result is not None
    assert result[1] == "Ahmet"

def test_parametrized_fixture(number):
    """Parametreli fixture testi"""
    assert number in [1, 2, 3]
    assert number > 0

def test_temp_file(temp_json_file):
    """Geçici dosya testi"""
    with open(temp_json_file, 'r') as f:
        data = json.load(f)

    assert "users" in data
    assert len(data["users"]) == 1
```

### Fixture Composition

```python
import pytest

@pytest.fixture
def user_data():
    """Kullanıcı verisi"""
    return {"name": "Mehmet", "email": "mehmet@example.com"}

@pytest.fixture
def authenticated_user(user_data):
    """Kimliği doğrulanmış kullanıcı (user_data'yı kullanır)"""
    return {
        **user_data,
        "token": "abc123",
        "authenticated": True
    }

@pytest.fixture
def admin_user(authenticated_user):
    """Admin kullanıcı (authenticated_user'ı kullanır)"""
    return {
        **authenticated_user,
        "role": "admin",
        "permissions": ["read", "write", "delete"]
    }

def test_user_data(user_data):
    """Basit kullanıcı testi"""
    assert user_data["name"] == "Mehmet"

def test_authenticated_user(authenticated_user):
    """Kimlik doğrulamalı kullanıcı testi"""
    assert authenticated_user["authenticated"] is True
    assert "token" in authenticated_user

def test_admin_user(admin_user):
    """Admin kullanıcı testi"""
    assert admin_user["role"] == "admin"
    assert "delete" in admin_user["permissions"]
```

---

## Mocking ve Patching

Mocking, dış bağımlılıkları (API, veritabanı, dosya sistemi) simüle etmek için kullanılır.

### unittest.mock Kullanımı

```python
from unittest.mock import Mock, MagicMock, patch, call
import requests
from typing import Dict, List

class UserService:
    """Kullanıcı servisi"""

    def __init__(self, api_url: str):
        self.api_url = api_url

    def get_user(self, user_id: int) -> Dict:
        """API'den kullanıcı bilgisi al"""
        response = requests.get(f"{self.api_url}/users/{user_id}")
        response.raise_for_status()
        return response.json()

    def create_user(self, user_data: Dict) -> Dict:
        """Yeni kullanıcı oluştur"""
        response = requests.post(f"{self.api_url}/users", json=user_data)
        response.raise_for_status()
        return response.json()

    def get_user_posts(self, user_id: int) -> List[Dict]:
        """Kullanıcının gönderilerini al"""
        user = self.get_user(user_id)
        response = requests.get(f"{self.api_url}/users/{user_id}/posts")
        response.raise_for_status()
        return response.json()

# Test dosyası
def test_get_user_with_mock():
    """Mock kullanarak API çağrısını test et"""
    # Mock response oluştur
    mock_response = Mock()
    mock_response.json.return_value = {
        "id": 1,
        "name": "Ali Veli",
        "email": "ali@example.com"
    }
    mock_response.raise_for_status.return_value = None

    # requests.get'i patch et
    with patch('requests.get', return_value=mock_response) as mock_get:
        service = UserService("http://api.example.com")
        user = service.get_user(1)

        # Assertions
        assert user["name"] == "Ali Veli"
        assert user["id"] == 1

        # Mock'un doğru çağrıldığını kontrol et
        mock_get.assert_called_once_with("http://api.example.com/users/1")

def test_create_user_with_mock():
    """POST isteğini mock'la"""
    user_data = {"name": "Ayşe", "email": "ayse@example.com"}

    mock_response = Mock()
    mock_response.json.return_value = {**user_data, "id": 2}
    mock_response.raise_for_status.return_value = None

    with patch('requests.post', return_value=mock_response) as mock_post:
        service = UserService("http://api.example.com")
        result = service.create_user(user_data)

        assert result["id"] == 2
        assert result["name"] == "Ayşe"

        mock_post.assert_called_once_with(
            "http://api.example.com/users",
            json=user_data
        )

def test_multiple_calls():
    """Birden fazla API çağrısını test et"""
    mock_user_response = Mock()
    mock_user_response.json.return_value = {"id": 1, "name": "Ali"}
    mock_user_response.raise_for_status.return_value = None

    mock_posts_response = Mock()
    mock_posts_response.json.return_value = [
        {"id": 1, "title": "Post 1"},
        {"id": 2, "title": "Post 2"}
    ]
    mock_posts_response.raise_for_status.return_value = None

    with patch('requests.get') as mock_get:
        # İlk çağrı user, ikinci çağrı posts dönsün
        mock_get.side_effect = [mock_user_response, mock_posts_response]

        service = UserService("http://api.example.com")
        posts = service.get_user_posts(1)

        assert len(posts) == 2
        assert posts[0]["title"] == "Post 1"
        assert mock_get.call_count == 2
```

### pytest-mock ile Mocking

```python
import pytest
from unittest.mock import Mock, MagicMock

class EmailService:
    """Email gönderme servisi"""

    def send_email(self, to: str, subject: str, body: str) -> bool:
        """Email gönder (gerçek implementasyon)"""
        # Gerçek email gönderme kodu burada olurdu
        raise NotImplementedError("Gerçek email servisi")

class NotificationService:
    """Bildirim servisi"""

    def __init__(self, email_service: EmailService):
        self.email_service = email_service

    def notify_user(self, user_email: str, message: str) -> bool:
        """Kullanıcıya bildirim gönder"""
        try:
            return self.email_service.send_email(
                to=user_email,
                subject="Bildirim",
                body=message
            )
        except Exception as e:
            print(f"Email gönderilemedi: {e}")
            return False

# pytest-mock kullanımı
def test_notification_success(mocker):
    """Başarılı bildirim testi"""
    # EmailService'i mock'la
    mock_email_service = mocker.Mock(spec=EmailService)
    mock_email_service.send_email.return_value = True

    notification_service = NotificationService(mock_email_service)
    result = notification_service.notify_user("user@example.com", "Test mesajı")

    assert result is True
    mock_email_service.send_email.assert_called_once_with(
        to="user@example.com",
        subject="Bildirim",
        body="Test mesajı"
    )

def test_notification_failure(mocker):
    """Başarısız bildirim testi"""
    mock_email_service = mocker.Mock(spec=EmailService)
    mock_email_service.send_email.side_effect = Exception("SMTP hatası")

    notification_service = NotificationService(mock_email_service)
    result = notification_service.notify_user("user@example.com", "Test")

    assert result is False

# Attribute ve method mocking
def test_class_mocking(mocker):
    """Sınıf attribute ve metodlarını mock'la"""
    mock_obj = mocker.Mock()

    # Attribute ayarla
    mock_obj.name = "Test Object"
    mock_obj.value = 42

    # Method return değeri ayarla
    mock_obj.get_data.return_value = {"key": "value"}
    mock_obj.process.return_value = True

    # Kullanım
    assert mock_obj.name == "Test Object"
    assert mock_obj.get_data() == {"key": "value"}
    assert mock_obj.process()

    # Call kontrolü
    mock_obj.process.assert_called_once()
```

### MagicMock ve Özel Metodlar

```python
from unittest.mock import MagicMock
import pytest

def test_magic_methods():
    """MagicMock ile özel metodları test et"""
    mock = MagicMock()

    # __getitem__ mock
    mock.__getitem__.return_value = "mocked value"
    assert mock["any_key"] == "mocked value"

    # __len__ mock
    mock.__len__.return_value = 10
    assert len(mock) == 10

    # __iter__ mock
    mock.__iter__.return_value = iter([1, 2, 3])
    assert list(mock) == [1, 2, 3]

    # __contains__ mock
    mock.__contains__.return_value = True
    assert "anything" in mock

    # Context manager mock
    mock.__enter__.return_value = "context value"
    with mock as m:
        assert m == "context value"
    mock.__exit__.assert_called_once()

class FileProcessor:
    """Dosya işleme sınıfı"""

    def process_file(self, filepath: str) -> Dict:
        """Dosyayı işle"""
        with open(filepath, 'r') as f:
            content = f.read()
        return {"lines": len(content.split('\n')), "chars": len(content)}

def test_file_processor(mocker):
    """Dosya işlemlerini mock'la"""
    mock_open = mocker.mock_open(read_data="Line 1\nLine 2\nLine 3")
    mocker.patch('builtins.open', mock_open)

    processor = FileProcessor()
    result = processor.process_file("dummy.txt")

    assert result["lines"] == 3
    mock_open.assert_called_once_with("dummy.txt", 'r')
```

---

## Parametrized Tests

Parametrized testler, aynı test mantığını farklı verilerle çalıştırmayı sağlar.

### pytest.mark.parametrize

```python
import pytest

def is_palindrome(s: str) -> bool:
    """Bir string'in palindrome olup olmadığını kontrol et"""
    s = s.lower().replace(" ", "")
    return s == s[::-1]

# Basit parametrize
@pytest.mark.parametrize("word,expected", [
    ("radar", True),
    ("python", False),
    ("level", True),
    ("kayak", True),
    ("hello", False),
])
def test_palindrome(word, expected):
    """Palindrome kontrolü testi"""
    assert is_palindrome(word) == expected

# Çoklu parametreler
@pytest.mark.parametrize("a,b,expected", [
    (1, 1, 2),
    (2, 3, 5),
    (-1, 1, 0),
    (0, 0, 0),
    (100, 200, 300),
])
def test_addition(a, b, expected):
    """Toplama testi"""
    assert a + b == expected

# İç içe parametrize
@pytest.mark.parametrize("base", [2, 3, 5])
@pytest.mark.parametrize("exponent", [2, 3, 4])
def test_power(base, exponent):
    """Üs alma testi"""
    result = base ** exponent
    assert result > 0
    assert result == pow(base, exponent)

# Parametrize ile ID'ler
@pytest.mark.parametrize("test_input,expected", [
    ("valid@email.com", True),
    ("invalid.email", False),
    ("test@test.co", True),
    ("@invalid.com", False),
], ids=["valid_email", "no_at_sign", "short_domain", "starts_with_at"])
def test_email_validation(test_input, expected):
    """Email validasyon testi"""
    import re
    pattern = r'^[\w\.-]+@[\w\.-]+\.\w+$'
    assert bool(re.match(pattern, test_input)) == expected

# Karmaşık veri yapıları ile parametrize
@pytest.mark.parametrize("user_data", [
    {"name": "Ali", "age": 25, "role": "developer"},
    {"name": "Ayşe", "age": 30, "role": "designer"},
    {"name": "Mehmet", "age": 35, "role": "manager"},
])
def test_user_processing(user_data):
    """Kullanıcı verisi işleme testi"""
    assert "name" in user_data
    assert user_data["age"] > 0
    assert len(user_data["name"]) > 0
```

### Parametrize Fixtures

```python
import pytest

@pytest.fixture(params=[
    {"db": "postgresql", "host": "localhost", "port": 5432},
    {"db": "mysql", "host": "localhost", "port": 3306},
    {"db": "mongodb", "host": "localhost", "port": 27017},
])
def database_config(request):
    """Farklı veritabanı konfigürasyonları"""
    return request.param

def test_database_connection(database_config):
    """Her veritabanı için bağlantı testi"""
    assert "db" in database_config
    assert "host" in database_config
    assert database_config["port"] > 0
    print(f"\nTesting {database_config['db']} on port {database_config['port']}")

# pytest.param ile özel parametreler
@pytest.mark.parametrize("value,expected", [
    pytest.param(1, 1, id="single"),
    pytest.param(2, 4, id="double"),
    pytest.param(5, 25, id="five"),
    pytest.param(-1, 1, id="negative", marks=pytest.mark.xfail),
])
def test_square(value, expected):
    """Karesi alma testi"""
    assert value ** 2 == expected
```

---

## Test Coverage

Test coverage, kodun ne kadarının testlerle kapsandığını ölçer.

### pytest-cov Kullanımı

```python
# calculator.py
class Calculator:
    """Gelişmiş hesap makinesi"""

    def add(self, a: float, b: float) -> float:
        """Toplama"""
        return a + b

    def subtract(self, a: float, b: float) -> float:
        """Çıkarma"""
        return a - b

    def multiply(self, a: float, b: float) -> float:
        """Çarpma"""
        return a * b

    def divide(self, a: float, b: float) -> float:
        """Bölme"""
        if b == 0:
            raise ValueError("Sıfıra bölme hatası")
        return a / b

    def power(self, base: float, exp: float) -> float:
        """Üs alma"""
        return base ** exp

    def sqrt(self, n: float) -> float:
        """Karekök"""
        if n < 0:
            raise ValueError("Negatif sayının karekökü alınamaz")
        return n ** 0.5

# test_calculator_coverage.py
import pytest
from calculator import Calculator

@pytest.fixture
def calc():
    return Calculator()

def test_add(calc):
    assert calc.add(2, 3) == 5

def test_subtract(calc):
    assert calc.subtract(5, 3) == 2

def test_multiply(calc):
    assert calc.multiply(4, 5) == 20

def test_divide(calc):
    assert calc.divide(10, 2) == 5
    with pytest.raises(ValueError):
        calc.divide(10, 0)

def test_power(calc):
    assert calc.power(2, 3) == 8

def test_sqrt(calc):
    assert calc.sqrt(9) == 3
    with pytest.raises(ValueError):
        calc.sqrt(-1)

# Coverage çalıştırma:
# pytest --cov=calculator --cov-report=html --cov-report=term
# Bu komut HTML rapor ve terminal raporu oluşturur
```

---

## pdb Debugger

pdb, Python'un built-in debugger'ıdır.

### pdb Temel Kullanımı

```python
import pdb

def complex_calculation(numbers: list) -> dict:
    """Karmaşık hesaplama fonksiyonu"""
    total = 0
    squares = []

    for num in numbers:
        # Debugger'ı başlat
        # pdb.set_trace()  # Python < 3.7
        breakpoint()  # Python >= 3.7 (önerilen)

        square = num ** 2
        squares.append(square)
        total += square

    average = total / len(numbers) if numbers else 0

    return {
        "total": total,
        "average": average,
        "squares": squares,
        "count": len(numbers)
    }

# pdb komutları:
# n (next): Sonraki satıra geç
# s (step): Fonksiyonun içine gir
# c (continue): Devam et
# l (list): Kodu göster
# p <variable>: Değişken değerini yazdır
# pp <variable>: Pretty print
# w (where): Stack trace göster
# b <line>: Breakpoint koy
# q (quit): Çık

def buggy_function(x: int, y: int) -> int:
    """Hatalı fonksiyon örneği"""
    result = x + y

    if result > 10:
        # Buraya breakpoint koy
        breakpoint()
        result = result * 2

    return result

# Post-mortem debugging
def failing_function():
    """Hata veren fonksiyon"""
    try:
        x = 1 / 0
    except Exception:
        import pdb
        pdb.post_mortem()  # Exception noktasında debug başlat
```

### pdb ile pytest

```python
import pytest

def divide_list(numbers: list, divisor: int) -> list:
    """Liste elemanlarını böl"""
    result = []
    for num in numbers:
        # pytest --pdb ile çalıştırınca hata noktasında durur
        result.append(num / divisor)
    return result

def test_divide_list():
    """Bölme testi"""
    numbers = [10, 20, 30]
    result = divide_list(numbers, 2)
    assert result == [5, 10, 15]

    # Bu test başarısız olur ve --pdb ile debug edebilirsiniz
    # pytest --pdb test_file.py

# pytest.set_trace() kullanımı
def test_with_breakpoint():
    """Breakpoint ile test"""
    x = 10
    y = 20

    # Test ortasında debug başlat
    pytest.set_trace()

    result = x + y
    assert result == 30
```

---

## Profiling

Profiling, kodun performansını ölçmek için kullanılır.

### cProfile Kullanımı

```python
import cProfile
import pstats
from io import StringIO
from typing import List

def fibonacci(n: int) -> int:
    """Fibonacci sayısı hesapla (yavaş versiyon)"""
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

def fibonacci_optimized(n: int, memo: dict = None) -> int:
    """Fibonacci sayısı hesapla (optimize edilmiş)"""
    if memo is None:
        memo = {}

    if n in memo:
        return memo[n]

    if n <= 1:
        return n

    memo[n] = fibonacci_optimized(n - 1, memo) + fibonacci_optimized(n - 2, memo)
    return memo[n]

def profile_function():
    """Fonksiyonu profile et"""
    # Yavaş versiyon
    result1 = fibonacci(30)

    # Hızlı versiyon
    result2 = fibonacci_optimized(30)

    return result1, result2

if __name__ == '__main__':
    # cProfile ile profiling
    profiler = cProfile.Profile()
    profiler.enable()

    result = profile_function()

    profiler.disable()

    # Sonuçları göster
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(10)  # İlk 10 satırı göster

    # Dosyaya kaydet
    stats.dump_stats('profile_results.prof')

# Decorator ile profiling
def profile(func):
    """Profiling decorator'ı"""
    def wrapper(*args, **kwargs):
        profiler = cProfile.Profile()
        profiler.enable()

        result = func(*args, **kwargs)

        profiler.disable()
        stats = pstats.Stats(profiler)
        stats.sort_stats('cumulative')
        stats.print_stats()

        return result
    return wrapper

@profile
def slow_function():
    """Yavaş fonksiyon"""
    total = 0
    for i in range(1000000):
        total += i
    return total
```

### line_profiler Kullanımı

```python
# line_profiler ile satır satır profiling
# Kurulum: pip install line-profiler

from typing import List

# @profile decorator'ı kullan (line_profiler tarafından sağlanır)
def process_data(data: List[int]) -> dict:
    """Veriyi işle"""
    # Her satırın ne kadar sürdüğünü ölç
    total = sum(data)
    average = total / len(data)

    squares = [x ** 2 for x in data]
    square_sum = sum(squares)

    filtered = [x for x in data if x > average]

    return {
        "total": total,
        "average": average,
        "square_sum": square_sum,
        "above_average": filtered
    }

# Çalıştırma:
# kernprof -l -v script.py
# Bu komut her satırın çalışma süresini gösterir
```

### timeit Kullanımı

```python
import timeit
from typing import List

def list_comprehension_method(n: int) -> List[int]:
    """List comprehension ile"""
    return [i ** 2 for i in range(n)]

def loop_method(n: int) -> List[int]:
    """Loop ile"""
    result = []
    for i in range(n):
        result.append(i ** 2)
    return result

def map_method(n: int) -> List[int]:
    """Map ile"""
    return list(map(lambda x: x ** 2, range(n)))

# Performans karşılaştırması
if __name__ == '__main__':
    n = 10000
    iterations = 1000

    # List comprehension
    time1 = timeit.timeit(
        lambda: list_comprehension_method(n),
        number=iterations
    )

    # Loop
    time2 = timeit.timeit(
        lambda: loop_method(n),
        number=iterations
    )

    # Map
    time3 = timeit.timeit(
        lambda: map_method(n),
        number=iterations
    )

    print(f"List comprehension: {time1:.4f} saniye")
    print(f"Loop: {time2:.4f} saniye")
    print(f"Map: {time3:.4f} saniye")
```

---

## Memory Profiling

Memory profiling, bellek kullanımını analiz etmek için kullanılır.

### memory_profiler Kullanımı

```python
# Kurulum: pip install memory-profiler
from memory_profiler import profile
from typing import List

@profile
def memory_intensive_function():
    """Bellek yoğun fonksiyon"""
    # Büyük liste oluştur
    big_list = [i for i in range(1000000)]

    # Liste kopyası
    big_list_copy = big_list.copy()

    # Dictionary oluştur
    big_dict = {i: i ** 2 for i in range(100000)}

    # İşlemler
    filtered = [x for x in big_list if x % 2 == 0]

    return len(filtered)

@profile
def memory_efficient_function():
    """Bellek verimli fonksiyon"""
    # Generator kullan
    total = sum(i for i in range(1000000) if i % 2 == 0)
    return total

# Çalıştırma:
# python -m memory_profiler script.py

# tracemalloc ile bellek takibi (Python 3.4+)
import tracemalloc

def analyze_memory():
    """Bellek kullanımını analiz et"""
    # Takibi başlat
    tracemalloc.start()

    # İlk snapshot
    snapshot1 = tracemalloc.take_snapshot()

    # Bellek yoğun işlem
    big_list = [i ** 2 for i in range(100000)]
    big_dict = {i: str(i) for i in range(50000)}

    # İkinci snapshot
    snapshot2 = tracemalloc.take_snapshot()

    # Farkları göster
    top_stats = snapshot2.compare_to(snapshot1, 'lineno')

    print("[ Top 10 bellek kullanımı ]")
    for stat in top_stats[:10]:
        print(stat)

    # Takibi durdur
    tracemalloc.stop()

if __name__ == '__main__':
    analyze_memory()
```

---

## CI/CD Basics

Continuous Integration / Continuous Deployment temel kavramları.

### GitHub Actions ile pytest

```yaml
# .github/workflows/tests.yml
name: Tests

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main, develop ]

jobs:
  test:
    runs-on: ubuntu-latest

    strategy:
      matrix:
        python-version: [3.8, 3.9, '3.10', '3.11']

    steps:
    - uses: actions/checkout@v3

    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install pytest pytest-cov
        if [ -f requirements.txt ]; then pip install -r requirements.txt; fi

    - name: Run tests with coverage
      run: |
        pytest --cov=. --cov-report=xml --cov-report=term

    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
        fail_ci_if_error: true
```

### pytest.ini Konfigürasyonu

```ini
# pytest.ini
[pytest]
# Test discovery patterns
python_files = test_*.py *_test.py
python_classes = Test*
python_functions = test_*

# Test paths
testpaths = tests

# Options
addopts =
    -v
    --strict-markers
    --cov=src
    --cov-report=html
    --cov-report=term-missing
    --cov-fail-under=80

# Markers
markers =
    slow: marks tests as slow (deselect with '-m "not slow"')
    integration: marks tests as integration tests
    unit: marks tests as unit tests
    smoke: marks tests as smoke tests

# Coverage options
[coverage:run]
source = src
omit =
    */tests/*
    */test_*.py

[coverage:report]
exclude_lines =
    pragma: no cover
    def __repr__
    raise AssertionError
    raise NotImplementedError
    if __name__ == .__main__.:
```

### tox ile Multi-Environment Testing

```ini
# tox.ini
[tox]
envlist = py38,py39,py310,py311,lint

[testenv]
deps =
    pytest
    pytest-cov
    pytest-mock
commands =
    pytest tests/ --cov=src --cov-report=term-missing

[testenv:lint]
deps =
    flake8
    black
    mypy
commands =
    flake8 src tests
    black --check src tests
    mypy src
```

### Pre-commit Hooks

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.4.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files

  - repo: https://github.com/psf/black
    rev: 23.3.0
    hooks:
      - id: black

  - repo: https://github.com/PyCQA/flake8
    rev: 6.0.0
    hooks:
      - id: flake8

  - repo: local
    hooks:
      - id: pytest
        name: pytest
        entry: pytest
        language: system
        pass_filenames: false
        always_run: true
```

---

## Test Best Practices

### AAA Pattern (Arrange-Act-Assert)

```python
import pytest
from typing import List, Dict

class ShoppingCart:
    """Alışveriş sepeti"""

    def __init__(self):
        self.items: List[Dict] = []

    def add_item(self, name: str, price: float, quantity: int = 1):
        """Ürün ekle"""
        self.items.append({
            "name": name,
            "price": price,
            "quantity": quantity
        })

    def get_total(self) -> float:
        """Toplam tutarı hesapla"""
        return sum(item["price"] * item["quantity"] for item in self.items)

    def apply_discount(self, percentage: float):
        """İndirim uygula"""
        for item in self.items:
            item["price"] *= (1 - percentage / 100)

def test_shopping_cart_total():
    """Alışveriş sepeti toplam testi (AAA pattern)"""
    # Arrange (Hazırlık)
    cart = ShoppingCart()
    cart.add_item("Elma", 5.0, 3)
    cart.add_item("Armut", 7.0, 2)

    # Act (Eylem)
    total = cart.get_total()

    # Assert (Doğrulama)
    assert total == 29.0  # (5*3) + (7*2)

def test_shopping_cart_discount():
    """İndirim testi (AAA pattern)"""
    # Arrange
    cart = ShoppingCart()
    cart.add_item("Laptop", 1000.0, 1)

    # Act
    cart.apply_discount(10)  # %10 indirim
    total = cart.get_total()

    # Assert
    assert total == 900.0
```

### Test Isolation

```python
import pytest

# Her test bağımsız olmalı
class Database:
    """Basit veritabanı simülasyonu"""

    def __init__(self):
        self.data = {}

    def insert(self, key: str, value: any):
        self.data[key] = value

    def get(self, key: str):
        return self.data.get(key)

    def clear(self):
        self.data.clear()

@pytest.fixture
def clean_db():
    """Her test için temiz veritabanı"""
    db = Database()
    yield db
    db.clear()  # Teardown

def test_insert(clean_db):
    """Insert testi - diğer testlerden bağımsız"""
    clean_db.insert("user1", {"name": "Ali"})
    assert clean_db.get("user1") == {"name": "Ali"}

def test_multiple_inserts(clean_db):
    """Multiple insert testi - temiz başlar"""
    clean_db.insert("user1", {"name": "Ali"})
    clean_db.insert("user2", {"name": "Ayşe"})
    assert len(clean_db.data) == 2
```

---

## Özet

Testing ve Debugging konusunda ele aldığımız ana başlıklar:

1. **unittest**: Python'un built-in test framework'ü
2. **pytest**: Modern ve kullanıcı dostu test framework'ü
3. **Fixtures**: Test verilerini ve kaynakları yönetme
4. **Mocking**: Dış bağımlılıkları simüle etme
5. **Parametrized Tests**: Aynı testi farklı verilerle çalıştırma
6. **Coverage**: Test kapsamını ölçme
7. **pdb**: Python debugger kullanımı
8. **Profiling**: Performans analizi (cProfile, line_profiler)
9. **Memory Profiling**: Bellek kullanımı analizi
10. **CI/CD**: Sürekli entegrasyon ve deployment temelleri

Bu teknikler, production-ready ve güvenilir Python uygulamaları geliştirmek için kritik öneme sahiptir.
