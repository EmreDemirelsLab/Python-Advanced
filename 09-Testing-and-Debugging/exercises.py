"""
Testing ve Debugging - İleri Seviye Alıştırmalar

Bu modül, modern test teknikleri ve debugging pratikleri üzerine
gerçek dünya senaryolarını içeren egzersizler sunar.

Her alıştırma şunları içerir:
- TODO: Görev tanımı
- Çözüm: Profesyonel implementasyon
- Test cases
"""

import pytest
from unittest.mock import Mock, MagicMock, patch, call
from typing import List, Dict, Optional, Callable
import json
import time
from datetime import datetime, timedelta
import sqlite3
from dataclasses import dataclass
import requests


# ============================================================================
# EXERCISE 1: API Test Suite (unittest.mock kullanımı)
# Zorluk: Orta
# Konu: REST API testing, mocking external calls
# ============================================================================

"""
TODO: Bir hava durumu API'si için test suite oluşturun.

Gereksinimler:
1. WeatherAPI sınıfı:
   - get_current_weather(city: str) -> dict metodu
   - get_forecast(city, days) -> list metodu
   - API çağrıları requests kütüphanesi ile yapılmalı

2. Test suite:
   - API başarılı yanıt testleri
   - API hata durumu testleri (404, 500, timeout)
   - Rate limiting testi
   - Cache mekanizması testi

İpucu: @patch decorator ve Mock objeler kullanın
"""

# ÇÖZÜM:

class WeatherAPI:
    """Hava durumu API client'ı"""

    def __init__(self, api_key: str, base_url: str = "https://api.weather.com"):
        self.api_key = api_key
        self.base_url = base_url
        self.cache = {}
        self.request_count = 0
        self.rate_limit = 100

    def get_current_weather(self, city: str) -> Dict:
        """
        Şehir için güncel hava durumunu getir

        Args:
            city: Şehir adı

        Returns:
            Hava durumu bilgisi içeren dict

        Raises:
            ValueError: Rate limit aşıldıysa
            requests.HTTPError: API hatası durumunda
        """
        if self.request_count >= self.rate_limit:
            raise ValueError("Rate limit exceeded")

        # Cache kontrolü
        cache_key = f"current_{city}"
        if cache_key in self.cache:
            cached_data, timestamp = self.cache[cache_key]
            if time.time() - timestamp < 300:  # 5 dakika cache
                return cached_data

        self.request_count += 1

        url = f"{self.base_url}/current"
        params = {"city": city, "api_key": self.api_key}

        response = requests.get(url, params=params, timeout=5)
        response.raise_for_status()

        data = response.json()
        self.cache[cache_key] = (data, time.time())

        return data

    def get_forecast(self, city: str, days: int = 3) -> List[Dict]:
        """
        Şehir için hava durumu tahmini getir

        Args:
            city: Şehir adı
            days: Kaç günlük tahmin (1-7)

        Returns:
            Günlük tahminleri içeren liste
        """
        if not 1 <= days <= 7:
            raise ValueError("Days must be between 1 and 7")

        if self.request_count >= self.rate_limit:
            raise ValueError("Rate limit exceeded")

        self.request_count += 1

        url = f"{self.base_url}/forecast"
        params = {"city": city, "days": days, "api_key": self.api_key}

        response = requests.get(url, params=params, timeout=5)
        response.raise_for_status()

        return response.json()["forecast"]


class TestWeatherAPI:
    """WeatherAPI test suite'i"""

    @pytest.fixture
    def api_client(self):
        """Test için API client"""
        return WeatherAPI(api_key="test_key_123")

    @patch('requests.get')
    def test_get_current_weather_success(self, mock_get, api_client):
        """Başarılı hava durumu sorgulama testi"""
        # Mock response hazırla
        mock_response = Mock()
        mock_response.json.return_value = {
            "city": "Istanbul",
            "temperature": 20,
            "conditions": "Sunny",
            "humidity": 65
        }
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        # Test et
        result = api_client.get_current_weather("Istanbul")

        # Doğrula
        assert result["city"] == "Istanbul"
        assert result["temperature"] == 20
        assert result["conditions"] == "Sunny"

        # API'nin doğru parametrelerle çağrıldığını kontrol et
        mock_get.assert_called_once()
        call_args = mock_get.call_args
        assert call_args[1]["params"]["city"] == "Istanbul"
        assert call_args[1]["params"]["api_key"] == "test_key_123"

    @patch('requests.get')
    def test_get_current_weather_not_found(self, mock_get, api_client):
        """Şehir bulunamadı hatası testi"""
        # 404 hatası simüle et
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = requests.HTTPError("404 Not Found")
        mock_get.return_value = mock_response

        # Hata fırlatılmasını bekle
        with pytest.raises(requests.HTTPError):
            api_client.get_current_weather("NonexistentCity")

    @patch('requests.get')
    def test_cache_mechanism(self, mock_get, api_client):
        """Cache mekanizması testi"""
        # Mock response
        mock_response = Mock()
        mock_response.json.return_value = {"city": "Ankara", "temperature": 18}
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        # İlk çağrı - API'ye gider
        result1 = api_client.get_current_weather("Ankara")

        # İkinci çağrı - cache'den gelir
        result2 = api_client.get_current_weather("Ankara")

        # API'nin sadece bir kez çağrıldığını doğrula
        assert mock_get.call_count == 1
        assert result1 == result2

    def test_rate_limiting(self, api_client):
        """Rate limiting testi"""
        # Rate limit'i düşür
        api_client.rate_limit = 3

        with patch('requests.get') as mock_get:
            mock_response = Mock()
            mock_response.json.return_value = {"city": "Izmir", "temperature": 22}
            mock_response.raise_for_status.return_value = None
            mock_get.return_value = mock_response

            # 3 başarılı çağrı
            for _ in range(3):
                api_client.get_current_weather("Izmir")

            # 4. çağrı hata vermeli
            with pytest.raises(ValueError, match="Rate limit exceeded"):
                api_client.get_current_weather("Izmir")

    @patch('requests.get')
    def test_get_forecast_success(self, mock_get, api_client):
        """Tahmin getirme testi"""
        mock_response = Mock()
        mock_response.json.return_value = {
            "forecast": [
                {"day": "Monday", "temp": 20},
                {"day": "Tuesday", "temp": 22},
                {"day": "Wednesday", "temp": 19}
            ]
        }
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        result = api_client.get_forecast("Istanbul", days=3)

        assert len(result) == 3
        assert result[0]["day"] == "Monday"

    def test_get_forecast_invalid_days(self, api_client):
        """Geçersiz gün sayısı testi"""
        with pytest.raises(ValueError, match="Days must be between 1 and 7"):
            api_client.get_forecast("Istanbul", days=10)


# ============================================================================
# EXERCISE 2: Database Testing with Fixtures
# Zorluk: Orta
# Konu: SQLite fixtures, test isolation, transaction rollback
# ============================================================================

"""
TODO: Bir kullanıcı yönetim sistemi için veritabanı testleri yazın.

Gereksinimler:
1. UserRepository sınıfı:
   - create_user(name, email) -> int
   - get_user(user_id) -> dict
   - update_user(user_id, data) -> bool
   - delete_user(user_id) -> bool
   - find_by_email(email) -> dict

2. Fixtures:
   - In-memory SQLite veritabanı
   - Test verisi için factory fixture
   - Her test sonrası cleanup

3. Test her çalıştığında temiz veritabanı kullanmalı
"""

# ÇÖZÜM:

class UserRepository:
    """Kullanıcı veritabanı işlemleri"""

    def __init__(self, db_connection):
        self.conn = db_connection
        self._create_table()

    def _create_table(self):
        """Users tablosunu oluştur"""
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT UNIQUE NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        self.conn.commit()

    def create_user(self, name: str, email: str) -> int:
        """Yeni kullanıcı oluştur"""
        cursor = self.conn.execute(
            'INSERT INTO users (name, email) VALUES (?, ?)',
            (name, email)
        )
        self.conn.commit()
        return cursor.lastrowid

    def get_user(self, user_id: int) -> Optional[Dict]:
        """ID ile kullanıcı getir"""
        cursor = self.conn.execute(
            'SELECT id, name, email, created_at FROM users WHERE id = ?',
            (user_id,)
        )
        row = cursor.fetchone()

        if row:
            return {
                "id": row[0],
                "name": row[1],
                "email": row[2],
                "created_at": row[3]
            }
        return None

    def update_user(self, user_id: int, data: Dict) -> bool:
        """Kullanıcı bilgilerini güncelle"""
        fields = []
        values = []

        if "name" in data:
            fields.append("name = ?")
            values.append(data["name"])

        if "email" in data:
            fields.append("email = ?")
            values.append(data["email"])

        if not fields:
            return False

        values.append(user_id)
        query = f"UPDATE users SET {', '.join(fields)} WHERE id = ?"

        cursor = self.conn.execute(query, values)
        self.conn.commit()

        return cursor.rowcount > 0

    def delete_user(self, user_id: int) -> bool:
        """Kullanıcıyı sil"""
        cursor = self.conn.execute('DELETE FROM users WHERE id = ?', (user_id,))
        self.conn.commit()
        return cursor.rowcount > 0

    def find_by_email(self, email: str) -> Optional[Dict]:
        """Email ile kullanıcı bul"""
        cursor = self.conn.execute(
            'SELECT id, name, email, created_at FROM users WHERE email = ?',
            (email,)
        )
        row = cursor.fetchone()

        if row:
            return {
                "id": row[0],
                "name": row[1],
                "email": row[2],
                "created_at": row[3]
            }
        return None


class TestUserRepository:
    """UserRepository test suite'i"""

    @pytest.fixture
    def db_connection(self):
        """Her test için temiz in-memory veritabanı"""
        conn = sqlite3.connect(':memory:')
        yield conn
        conn.close()

    @pytest.fixture
    def user_repo(self, db_connection):
        """UserRepository instance"""
        return UserRepository(db_connection)

    @pytest.fixture
    def sample_user(self, user_repo):
        """Test için örnek kullanıcı"""
        user_id = user_repo.create_user("Ali Yılmaz", "ali@example.com")
        return user_id

    def test_create_user(self, user_repo):
        """Kullanıcı oluşturma testi"""
        user_id = user_repo.create_user("Ayşe Demir", "ayse@example.com")

        assert user_id > 0

        # Kullanıcının oluşturulduğunu doğrula
        user = user_repo.get_user(user_id)
        assert user is not None
        assert user["name"] == "Ayşe Demir"
        assert user["email"] == "ayse@example.com"

    def test_create_duplicate_email(self, user_repo, sample_user):
        """Aynı email ile ikinci kullanıcı oluşturma hatası"""
        with pytest.raises(sqlite3.IntegrityError):
            user_repo.create_user("Başka Biri", "ali@example.com")

    def test_get_user(self, user_repo, sample_user):
        """Kullanıcı getirme testi"""
        user = user_repo.get_user(sample_user)

        assert user is not None
        assert user["id"] == sample_user
        assert user["name"] == "Ali Yılmaz"
        assert user["email"] == "ali@example.com"
        assert "created_at" in user

    def test_get_nonexistent_user(self, user_repo):
        """Var olmayan kullanıcı testi"""
        user = user_repo.get_user(99999)
        assert user is None

    def test_update_user(self, user_repo, sample_user):
        """Kullanıcı güncelleme testi"""
        success = user_repo.update_user(sample_user, {
            "name": "Ali Veli Yılmaz",
            "email": "ali.yilmaz@example.com"
        })

        assert success is True

        # Güncellendiğini doğrula
        user = user_repo.get_user(sample_user)
        assert user["name"] == "Ali Veli Yılmaz"
        assert user["email"] == "ali.yilmaz@example.com"

    def test_update_user_partial(self, user_repo, sample_user):
        """Kısmi güncelleme testi"""
        success = user_repo.update_user(sample_user, {"name": "Yeni İsim"})

        assert success is True

        user = user_repo.get_user(sample_user)
        assert user["name"] == "Yeni İsim"
        assert user["email"] == "ali@example.com"  # Email değişmedi

    def test_delete_user(self, user_repo, sample_user):
        """Kullanıcı silme testi"""
        success = user_repo.delete_user(sample_user)

        assert success is True

        # Silindiğini doğrula
        user = user_repo.get_user(sample_user)
        assert user is None

    def test_delete_nonexistent_user(self, user_repo):
        """Var olmayan kullanıcı silme testi"""
        success = user_repo.delete_user(99999)
        assert success is False

    def test_find_by_email(self, user_repo, sample_user):
        """Email ile kullanıcı bulma testi"""
        user = user_repo.find_by_email("ali@example.com")

        assert user is not None
        assert user["id"] == sample_user
        assert user["name"] == "Ali Yılmaz"

    def test_find_by_email_not_found(self, user_repo):
        """Email ile bulunamayan kullanıcı testi"""
        user = user_repo.find_by_email("notfound@example.com")
        assert user is None

    def test_multiple_users(self, user_repo):
        """Çoklu kullanıcı işlemleri testi"""
        # 3 kullanıcı oluştur
        id1 = user_repo.create_user("User 1", "user1@example.com")
        id2 = user_repo.create_user("User 2", "user2@example.com")
        id3 = user_repo.create_user("User 3", "user3@example.com")

        # Hepsini doğrula
        assert user_repo.get_user(id1)["name"] == "User 1"
        assert user_repo.get_user(id2)["name"] == "User 2"
        assert user_repo.get_user(id3)["name"] == "User 3"

        # Birini sil
        user_repo.delete_user(id2)

        # Kontrol et
        assert user_repo.get_user(id1) is not None
        assert user_repo.get_user(id2) is None
        assert user_repo.get_user(id3) is not None


# ============================================================================
# EXERCISE 3: Parametrized Testing
# Zorluk: Orta
# Konu: pytest.mark.parametrize, test data combinations
# ============================================================================

"""
TODO: Form validasyon fonksiyonları için parametrize testler yazın.

Gereksinimler:
1. Validasyon fonksiyonları:
   - validate_email(email) -> tuple[bool, str]
   - validate_password(password) -> tuple[bool, str]
   - validate_phone(phone) -> tuple[bool, str]

2. Her validasyon için 10+ test case
3. pytest.parametrize kullanarak DRY prensibine uygun testler
"""

# ÇÖZÜM:

import re
from typing import Tuple

class FormValidator:
    """Form validasyon yardımcıları"""

    @staticmethod
    def validate_email(email: str) -> Tuple[bool, str]:
        """
        Email validasyonu

        Returns:
            (is_valid, error_message) tuple
        """
        if not email:
            return False, "Email boş olamaz"

        if len(email) > 254:
            return False, "Email çok uzun"

        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'

        if not re.match(pattern, email):
            return False, "Geçersiz email formatı"

        return True, ""

    @staticmethod
    def validate_password(password: str) -> Tuple[bool, str]:
        """
        Şifre validasyonu

        Kurallar:
        - En az 8 karakter
        - En az bir büyük harf
        - En az bir küçük harf
        - En az bir rakam
        - En az bir özel karakter
        """
        if not password:
            return False, "Şifre boş olamaz"

        if len(password) < 8:
            return False, "Şifre en az 8 karakter olmalı"

        if not re.search(r'[A-Z]', password):
            return False, "Şifre en az bir büyük harf içermeli"

        if not re.search(r'[a-z]', password):
            return False, "Şifre en az bir küçük harf içermeli"

        if not re.search(r'\d', password):
            return False, "Şifre en az bir rakam içermeli"

        if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
            return False, "Şifre en az bir özel karakter içermeli"

        return True, ""

    @staticmethod
    def validate_phone(phone: str) -> Tuple[bool, str]:
        """
        Telefon numarası validasyonu (Türkiye formatı)

        Kabul edilen formatlar:
        - 05551234567
        - +905551234567
        - 0 555 123 45 67
        - (555) 123 45 67
        """
        if not phone:
            return False, "Telefon numarası boş olamaz"

        # Boşlukları ve özel karakterleri temizle
        cleaned = re.sub(r'[\s\-\(\)]', '', phone)

        # +90 ile başlıyorsa kaldır
        if cleaned.startswith('+90'):
            cleaned = '0' + cleaned[3:]

        # Türkiye formatı kontrolü (0 ile başlayan 11 haneli)
        if not re.match(r'^0[5][0-9]{9}$', cleaned):
            return False, "Geçersiz telefon numarası formatı"

        return True, ""


class TestFormValidator:
    """Form validator test suite'i"""

    @pytest.mark.parametrize("email,expected_valid,error_contains", [
        # Geçerli emailler
        ("user@example.com", True, ""),
        ("test.user@example.com", True, ""),
        ("user+tag@example.co.uk", True, ""),
        ("user_name@example-domain.com", True, ""),

        # Geçersiz emailler
        ("", False, "boş"),
        ("invalid", False, "Geçersiz"),
        ("@example.com", False, "Geçersiz"),
        ("user@", False, "Geçersiz"),
        ("user @example.com", False, "Geçersiz"),
        ("user@example", False, "Geçersiz"),
        ("user@@example.com", False, "Geçersiz"),
        ("a" * 250 + "@example.com", False, "uzun"),
    ])
    def test_email_validation(self, email, expected_valid, error_contains):
        """Email validasyon testleri"""
        is_valid, error = FormValidator.validate_email(email)

        assert is_valid == expected_valid

        if not expected_valid:
            assert error_contains.lower() in error.lower()

    @pytest.mark.parametrize("password,expected_valid,error_contains", [
        # Geçerli şifreler
        ("Password123!", True, ""),
        ("MyP@ssw0rd", True, ""),
        ("Str0ng!Pass", True, ""),
        ("C0mpl3x@Pwd", True, ""),

        # Geçersiz şifreler - boş
        ("", False, "boş"),

        # Geçersiz şifreler - çok kısa
        ("Pass1!", False, "8 karakter"),
        ("Aa1!", False, "8 karakter"),

        # Geçersiz şifreler - büyük harf yok
        ("password123!", False, "büyük harf"),

        # Geçersiz şifreler - küçük harf yok
        ("PASSWORD123!", False, "küçük harf"),

        # Geçersiz şifreler - rakam yok
        ("Password!", False, "rakam"),

        # Geçersiz şifreler - özel karakter yok
        ("Password123", False, "özel karakter"),

        # Kombinasyon eksiklikleri
        ("password", False, ""),  # Birden fazla kural ihlali
        ("12345678", False, ""),  # Sadece rakam
    ])
    def test_password_validation(self, password, expected_valid, error_contains):
        """Şifre validasyon testleri"""
        is_valid, error = FormValidator.validate_password(password)

        assert is_valid == expected_valid

        if not expected_valid and error_contains:
            assert error_contains.lower() in error.lower()

    @pytest.mark.parametrize("phone,expected_valid", [
        # Geçerli formatlar
        ("05551234567", True),
        ("+905551234567", True),
        ("0 555 123 45 67", True),
        ("0555 123 45 67", True),
        ("(555) 123 45 67", True),
        ("0-555-123-45-67", True),

        # Farklı operatörler
        ("05321234567", True),
        ("05421234567", True),
        ("05051234567", True),

        # Geçersiz formatlar
        ("", False),
        ("1234567890", False),  # 0 ile başlamıyor
        ("05551234", False),  # Çok kısa
        ("055512345678", False),  # Çok uzun
        ("15551234567", False),  # 0 ile başlamıyor
        ("02121234567", False),  # 5 ile devam etmiyor
        ("abc5551234567", False),  # Harf içeriyor
    ])
    def test_phone_validation(self, phone, expected_valid):
        """Telefon numarası validasyon testleri"""
        is_valid, error = FormValidator.validate_phone(phone)

        assert is_valid == expected_valid

        if not expected_valid:
            assert len(error) > 0


# ============================================================================
# EXERCISE 4: TDD - Test Driven Development
# Zorluk: İleri
# Konu: TDD cycle, red-green-refactor
# ============================================================================

"""
TODO: TDD yaklaşımı ile bir Stack veri yapısı implement edin.

TDD Döngüsü:
1. RED: Test yaz (başarısız olsun)
2. GREEN: Minimum kod yaz (test geçsin)
3. REFACTOR: Kodu iyileştir

Stack operasyonları:
- push(item): Eleman ekle
- pop(): Son elemanı çıkar ve döndür
- peek(): Son elemana bak (çıkarma)
- is_empty(): Boş mu kontrol et
- size(): Eleman sayısı
- clear(): Temizle

Önce tüm testleri yazın, sonra implementasyonu yapın!
"""

# ÇÖZÜM:

# Önce testler (RED phase)
class TestStack:
    """Stack veri yapısı testleri - TDD"""

    @pytest.fixture
    def stack(self):
        """Her test için yeni stack"""
        return Stack()

    def test_new_stack_is_empty(self, stack):
        """Yeni stack boş olmalı"""
        assert stack.is_empty() is True
        assert stack.size() == 0

    def test_push_adds_item(self, stack):
        """Push ile eleman ekleme"""
        stack.push(1)
        assert stack.is_empty() is False
        assert stack.size() == 1

    def test_push_multiple_items(self, stack):
        """Birden fazla eleman ekleme"""
        stack.push(1)
        stack.push(2)
        stack.push(3)
        assert stack.size() == 3

    def test_pop_removes_and_returns_last_item(self, stack):
        """Pop son elemanı çıkarıp döndürmeli"""
        stack.push(1)
        stack.push(2)

        item = stack.pop()

        assert item == 2
        assert stack.size() == 1

    def test_pop_from_empty_stack_raises_error(self, stack):
        """Boş stack'ten pop hata vermeli"""
        with pytest.raises(IndexError, match="Stack is empty"):
            stack.pop()

    def test_peek_returns_last_item_without_removing(self, stack):
        """Peek son elemanı döndürmeli ama çıkarmamalı"""
        stack.push(1)
        stack.push(2)

        item = stack.peek()

        assert item == 2
        assert stack.size() == 2  # Çıkarılmadı

    def test_peek_on_empty_stack_raises_error(self, stack):
        """Boş stack'te peek hata vermeli"""
        with pytest.raises(IndexError, match="Stack is empty"):
            stack.peek()

    def test_clear_removes_all_items(self, stack):
        """Clear tüm elemanları temizlemeli"""
        stack.push(1)
        stack.push(2)
        stack.push(3)

        stack.clear()

        assert stack.is_empty() is True
        assert stack.size() == 0

    def test_stack_follows_lifo_order(self, stack):
        """LIFO (Last In First Out) sıralaması"""
        items = [1, 2, 3, 4, 5]

        for item in items:
            stack.push(item)

        # Ters sırada çıkmalı
        for i in range(len(items) - 1, -1, -1):
            assert stack.pop() == items[i]

    @pytest.mark.parametrize("items", [
        [1, 2, 3],
        ["a", "b", "c"],
        [{"key": "value"}, [1, 2], "string"],
        [None, 0, False, ""]
    ])
    def test_stack_with_different_types(self, stack, items):
        """Farklı veri tipleriyle çalışma"""
        for item in items:
            stack.push(item)

        for i in range(len(items) - 1, -1, -1):
            assert stack.pop() == items[i]


# Şimdi implementasyon (GREEN phase)
class Stack:
    """LIFO (Last In First Out) Stack veri yapısı"""

    def __init__(self):
        """Yeni stack oluştur"""
        self._items = []

    def push(self, item):
        """
        Stack'e eleman ekle

        Args:
            item: Eklenecek eleman
        """
        self._items.append(item)

    def pop(self):
        """
        Stack'ten son elemanı çıkar ve döndür

        Returns:
            Son eleman

        Raises:
            IndexError: Stack boşsa
        """
        if self.is_empty():
            raise IndexError("Stack is empty")
        return self._items.pop()

    def peek(self):
        """
        Son elemana bak (çıkarma)

        Returns:
            Son eleman

        Raises:
            IndexError: Stack boşsa
        """
        if self.is_empty():
            raise IndexError("Stack is empty")
        return self._items[-1]

    def is_empty(self) -> bool:
        """
        Stack boş mu kontrol et

        Returns:
            True eğer boşsa
        """
        return len(self._items) == 0

    def size(self) -> int:
        """
        Stack'teki eleman sayısı

        Returns:
            Eleman sayısı
        """
        return len(self._items)

    def clear(self):
        """Tüm elemanları temizle"""
        self._items.clear()

    def __repr__(self):
        """String representation"""
        return f"Stack({self._items})"


# ============================================================================
# EXERCISE 5: Performance Testing
# Zorluk: İleri
# Konu: pytest-benchmark, performance regression testing
# ============================================================================

"""
TODO: Algoritma performans testleri yazın.

Gereksinimler:
1. Farklı sorting algoritmalarını test edin
2. Performans karşılaştırması yapın
3. pytest-benchmark veya timeit kullanın
4. Big O analizi yapacak testler yazın
"""

# ÇÖZÜM:

def bubble_sort(arr: List[int]) -> List[int]:
    """Bubble sort - O(n²)"""
    arr = arr.copy()
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr


def quick_sort(arr: List[int]) -> List[int]:
    """Quick sort - O(n log n) average"""
    if len(arr) <= 1:
        return arr

    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]

    return quick_sort(left) + middle + quick_sort(right)


def merge_sort(arr: List[int]) -> List[int]:
    """Merge sort - O(n log n)"""
    if len(arr) <= 1:
        return arr

    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])

    return merge(left, right)


def merge(left: List[int], right: List[int]) -> List[int]:
    """İki sıralı listeyi birleştir"""
    result = []
    i = j = 0

    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1

    result.extend(left[i:])
    result.extend(right[j:])

    return result


class TestSortingPerformance:
    """Sorting algoritmaları performans testleri"""

    @pytest.fixture
    def small_data(self):
        """Küçük veri seti"""
        return [5, 2, 8, 1, 9, 3, 7, 4, 6]

    @pytest.fixture
    def medium_data(self):
        """Orta veri seti"""
        import random
        return random.sample(range(1000), 100)

    @pytest.fixture
    def large_data(self):
        """Büyük veri seti"""
        import random
        return random.sample(range(10000), 1000)

    # Doğruluk testleri
    @pytest.mark.parametrize("sort_func", [bubble_sort, quick_sort, merge_sort])
    def test_sorting_correctness(self, sort_func, small_data):
        """Sorting algoritması doğruluğu"""
        result = sort_func(small_data)
        assert result == sorted(small_data)

    @pytest.mark.parametrize("sort_func", [bubble_sort, quick_sort, merge_sort])
    def test_sorting_with_duplicates(self, sort_func):
        """Tekrarlı elemanlarla sorting"""
        data = [5, 2, 8, 2, 9, 5, 7, 8, 5]
        result = sort_func(data)
        assert result == sorted(data)

    @pytest.mark.parametrize("sort_func", [bubble_sort, quick_sort, merge_sort])
    def test_sorting_empty_list(self, sort_func):
        """Boş liste sorting"""
        result = sort_func([])
        assert result == []

    @pytest.mark.parametrize("sort_func", [bubble_sort, quick_sort, merge_sort])
    def test_sorting_single_element(self, sort_func):
        """Tek elemanlı liste sorting"""
        result = sort_func([42])
        assert result == [42]

    # Performans testleri (manuel timing)
    def test_performance_comparison(self, medium_data):
        """Algoritma performans karşılaştırması"""
        import time

        algorithms = {
            "Bubble Sort": bubble_sort,
            "Quick Sort": quick_sort,
            "Merge Sort": merge_sort,
            "Python sorted": lambda x: sorted(x)
        }

        results = {}

        for name, func in algorithms.items():
            start = time.perf_counter()
            func(medium_data)
            end = time.perf_counter()
            results[name] = end - start

        # Quick sort ve merge sort bubble'dan hızlı olmalı
        assert results["Quick Sort"] < results["Bubble Sort"]
        assert results["Merge Sort"] < results["Bubble Sort"]

        print("\nPerformans Sonuçları:")
        for name, duration in sorted(results.items(), key=lambda x: x[1]):
            print(f"  {name}: {duration:.6f} saniye")

    # Big O analizi için test
    @pytest.mark.parametrize("size", [10, 50, 100, 200])
    def test_bubble_sort_complexity(self, size):
        """Bubble sort O(n²) kompleksitesi"""
        import random
        import time

        data = random.sample(range(size * 10), size)

        start = time.perf_counter()
        bubble_sort(data)
        duration = time.perf_counter() - start

        # Sonraki testlerle karşılaştırma için kaydet
        print(f"\nSize {size}: {duration:.6f} saniye")

        # n² kompleksite bekleriz, yani size 2x olunca time ~4x olmalı
        # (Bu basit bir gösterimdir, gerçek O(n²) analizi daha karmaşık)


# ============================================================================
# EXERCISE 6: Integration Testing
# Zorluk: İleri
# Konu: Multiple component integration, end-to-end testing
# ============================================================================

"""
TODO: Bir blog sistemi için integration testleri yazın.

Sistem bileşenleri:
1. Database (SQLite)
2. UserService
3. PostService
4. CommentService

Test senaryoları:
- Kullanıcı oluştur -> Post yaz -> Yorum ekle
- Post sil -> Yorumlar da silinmeli
- User sil -> Post'ları ve yorumları da silinmeli
"""

# ÇÖZÜM:

@dataclass
class User:
    id: Optional[int]
    username: str
    email: str


@dataclass
class Post:
    id: Optional[int]
    user_id: int
    title: str
    content: str
    created_at: Optional[str] = None


@dataclass
class Comment:
    id: Optional[int]
    post_id: int
    user_id: int
    content: str
    created_at: Optional[str] = None


class BlogDatabase:
    """Blog veritabanı yönetimi"""

    def __init__(self, connection):
        self.conn = connection
        self._create_tables()

    def _create_tables(self):
        """Tabloları oluştur"""
        self.conn.executescript('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL
            );

            CREATE TABLE IF NOT EXISTS posts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                title TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS comments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                post_id INTEGER NOT NULL,
                user_id INTEGER NOT NULL,
                content TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (post_id) REFERENCES posts(id) ON DELETE CASCADE,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            );
        ''')
        self.conn.commit()


class UserService:
    """Kullanıcı servisi"""

    def __init__(self, db: BlogDatabase):
        self.db = db

    def create_user(self, username: str, email: str) -> int:
        """Kullanıcı oluştur"""
        cursor = self.db.conn.execute(
            'INSERT INTO users (username, email) VALUES (?, ?)',
            (username, email)
        )
        self.db.conn.commit()
        return cursor.lastrowid

    def get_user(self, user_id: int) -> Optional[User]:
        """Kullanıcı getir"""
        cursor = self.db.conn.execute(
            'SELECT id, username, email FROM users WHERE id = ?',
            (user_id,)
        )
        row = cursor.fetchone()

        if row:
            return User(id=row[0], username=row[1], email=row[2])
        return None

    def delete_user(self, user_id: int):
        """Kullanıcı sil (cascade ile post ve commentler de silinir)"""
        self.db.conn.execute('DELETE FROM users WHERE id = ?', (user_id,))
        self.db.conn.commit()


class PostService:
    """Post servisi"""

    def __init__(self, db: BlogDatabase):
        self.db = db

    def create_post(self, user_id: int, title: str, content: str) -> int:
        """Post oluştur"""
        cursor = self.db.conn.execute(
            'INSERT INTO posts (user_id, title, content) VALUES (?, ?, ?)',
            (user_id, title, content)
        )
        self.db.conn.commit()
        return cursor.lastrowid

    def get_post(self, post_id: int) -> Optional[Post]:
        """Post getir"""
        cursor = self.db.conn.execute(
            'SELECT id, user_id, title, content, created_at FROM posts WHERE id = ?',
            (post_id,)
        )
        row = cursor.fetchone()

        if row:
            return Post(
                id=row[0],
                user_id=row[1],
                title=row[2],
                content=row[3],
                created_at=row[4]
            )
        return None

    def get_user_posts(self, user_id: int) -> List[Post]:
        """Kullanıcının postlarını getir"""
        cursor = self.db.conn.execute(
            'SELECT id, user_id, title, content, created_at FROM posts WHERE user_id = ?',
            (user_id,)
        )

        posts = []
        for row in cursor.fetchall():
            posts.append(Post(
                id=row[0],
                user_id=row[1],
                title=row[2],
                content=row[3],
                created_at=row[4]
            ))

        return posts

    def delete_post(self, post_id: int):
        """Post sil (cascade ile commentler de silinir)"""
        self.db.conn.execute('DELETE FROM posts WHERE id = ?', (post_id,))
        self.db.conn.commit()


class CommentService:
    """Yorum servisi"""

    def __init__(self, db: BlogDatabase):
        self.db = db

    def create_comment(self, post_id: int, user_id: int, content: str) -> int:
        """Yorum oluştur"""
        cursor = self.db.conn.execute(
            'INSERT INTO comments (post_id, user_id, content) VALUES (?, ?, ?)',
            (post_id, user_id, content)
        )
        self.db.conn.commit()
        return cursor.lastrowid

    def get_post_comments(self, post_id: int) -> List[Comment]:
        """Post yorumlarını getir"""
        cursor = self.db.conn.execute(
            'SELECT id, post_id, user_id, content, created_at FROM comments WHERE post_id = ?',
            (post_id,)
        )

        comments = []
        for row in cursor.fetchall():
            comments.append(Comment(
                id=row[0],
                post_id=row[1],
                user_id=row[2],
                content=row[3],
                created_at=row[4]
            ))

        return comments


class TestBlogSystemIntegration:
    """Blog sistemi integration testleri"""

    @pytest.fixture
    def blog_system(self):
        """Tüm blog sistemi setup"""
        conn = sqlite3.connect(':memory:')
        # Foreign key constraint'leri aktif et
        conn.execute('PRAGMA foreign_keys = ON')

        db = BlogDatabase(conn)
        user_service = UserService(db)
        post_service = PostService(db)
        comment_service = CommentService(db)

        yield {
            'db': db,
            'users': user_service,
            'posts': post_service,
            'comments': comment_service
        }

        conn.close()

    def test_create_user_post_comment_workflow(self, blog_system):
        """Tam workflow testi: user -> post -> comment"""
        users = blog_system['users']
        posts = blog_system['posts']
        comments = blog_system['comments']

        # 1. Kullanıcı oluştur
        user_id = users.create_user("ahmet", "ahmet@example.com")
        user = users.get_user(user_id)
        assert user is not None
        assert user.username == "ahmet"

        # 2. Post oluştur
        post_id = posts.create_post(
            user_id,
            "İlk Post",
            "Bu benim ilk blog postum!"
        )
        post = posts.get_post(post_id)
        assert post is not None
        assert post.title == "İlk Post"
        assert post.user_id == user_id

        # 3. Yorum ekle
        comment_id = comments.create_comment(
            post_id,
            user_id,
            "Harika bir post!"
        )

        post_comments = comments.get_post_comments(post_id)
        assert len(post_comments) == 1
        assert post_comments[0].content == "Harika bir post!"

    def test_delete_post_cascades_to_comments(self, blog_system):
        """Post silinince yorumlar da silinmeli"""
        users = blog_system['users']
        posts = blog_system['posts']
        comments = blog_system['comments']

        # Setup
        user_id = users.create_user("mehmet", "mehmet@example.com")
        post_id = posts.create_post(user_id, "Test Post", "Content")

        comment_id1 = comments.create_comment(post_id, user_id, "Yorum 1")
        comment_id2 = comments.create_comment(post_id, user_id, "Yorum 2")

        # Yorumları doğrula
        assert len(comments.get_post_comments(post_id)) == 2

        # Post'u sil
        posts.delete_post(post_id)

        # Yorumlar da silinmeli
        assert len(comments.get_post_comments(post_id)) == 0
        assert posts.get_post(post_id) is None

    def test_delete_user_cascades_to_posts_and_comments(self, blog_system):
        """User silinince post ve yorumlar da silinmeli"""
        users = blog_system['users']
        posts = blog_system['posts']
        comments = blog_system['comments']

        # İki kullanıcı oluştur
        user1_id = users.create_user("user1", "user1@example.com")
        user2_id = users.create_user("user2", "user2@example.com")

        # User1'in postu
        post_id = posts.create_post(user1_id, "User1 Post", "Content")

        # Her iki kullanıcı da yorum yapsın
        comments.create_comment(post_id, user1_id, "User1 yorumu")
        comments.create_comment(post_id, user2_id, "User2 yorumu")

        # User1'i sil
        users.delete_user(user1_id)

        # User1'in postu silinmeli
        assert posts.get_post(post_id) is None

        # User1 yokken tüm yorumlar da gitmiş olmalı (post cascade ile)
        assert len(comments.get_post_comments(post_id)) == 0

        # User2 hala var olmalı
        assert users.get_user(user2_id) is not None

    def test_multiple_users_multiple_posts(self, blog_system):
        """Çoklu kullanıcı ve post senaryosu"""
        users = blog_system['users']
        posts = blog_system['posts']
        comments = blog_system['comments']

        # 3 kullanıcı oluştur
        user_ids = []
        for i in range(3):
            user_id = users.create_user(f"user{i}", f"user{i}@example.com")
            user_ids.append(user_id)

        # Her kullanıcı 2 post yazsın
        for user_id in user_ids:
            for i in range(2):
                posts.create_post(user_id, f"Post {i}", f"Content {i}")

        # Her kullanıcının 2 postu olmalı
        for user_id in user_ids:
            user_posts = posts.get_user_posts(user_id)
            assert len(user_posts) == 2

        # Bir kullanıcıyı sil
        users.delete_user(user_ids[0])

        # Silinen kullanıcının postları gitmiş olmalı
        assert len(posts.get_user_posts(user_ids[0])) == 0

        # Diğer kullanıcıların postları durmalı
        assert len(posts.get_user_posts(user_ids[1])) == 2
        assert len(posts.get_user_posts(user_ids[2])) == 2


# ============================================================================
# EXERCISE 7: Custom pytest Fixtures ve Plugins
# Zorluk: İleri
# Konu: Advanced fixtures, fixture factories, custom markers
# ============================================================================

"""
TODO: Özelleştirilmiş pytest fixtures ve markers oluşturun.

Gereksinimler:
1. Factory fixture (dinamik test verisi)
2. Autouse fixture (her test için otomatik)
3. Custom marker (@pytest.mark.api, @pytest.mark.slow)
4. Fixture parametrization
5. Fixture scopes (function, class, module, session)
"""

# ÇÖZÜM:

# conftest.py içeriği (fixture definitions)

@pytest.fixture(scope="session")
def session_config():
    """Session boyunca kullanılacak config"""
    print("\n[SESSION SETUP] Config yükleniyor...")
    config = {
        "api_url": "https://api.example.com",
        "timeout": 30,
        "retry_count": 3
    }
    yield config
    print("\n[SESSION TEARDOWN] Config temizleniyor...")


@pytest.fixture(scope="module")
def database_connection():
    """Module seviyesinde shared database connection"""
    print("\n[MODULE SETUP] Database bağlantısı açılıyor...")
    conn = sqlite3.connect(':memory:')

    # Tablo oluştur
    conn.execute('''
        CREATE TABLE test_data (
            id INTEGER PRIMARY KEY,
            value TEXT
        )
    ''')
    conn.commit()

    yield conn

    print("\n[MODULE TEARDOWN] Database bağlantısı kapatılıyor...")
    conn.close()


@pytest.fixture
def user_factory():
    """User oluşturma factory fixture"""
    created_users = []

    def _create_user(username: str = None, email: str = None):
        """Dinamik user oluştur"""
        if username is None:
            username = f"user_{len(created_users)}"
        if email is None:
            email = f"{username}@example.com"

        user = {"id": len(created_users) + 1, "username": username, "email": email}
        created_users.append(user)
        return user

    yield _create_user

    # Cleanup
    print(f"\n[TEARDOWN] {len(created_users)} user temizlendi")


@pytest.fixture(autouse=True)
def test_lifecycle():
    """Her test için otomatik çalışan fixture"""
    print("\n[TEST START]")
    start_time = time.time()

    yield

    duration = time.time() - start_time
    print(f"\n[TEST END] Süre: {duration:.3f}s")


@pytest.fixture(params=["sqlite", "postgresql", "mysql"])
def database_type(request):
    """Farklı database tipleri için parametrized fixture"""
    return request.param


# Custom marker definitions
def pytest_configure(config):
    """Custom marker'ları kaydet"""
    config.addinivalue_line(
        "markers", "api: mark test as API integration test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "smoke: mark test as smoke test"
    )


class TestCustomFixtures:
    """Custom fixture kullanım örnekleri"""

    def test_user_factory(self, user_factory):
        """Factory fixture kullanımı"""
        # Varsayılan user
        user1 = user_factory()
        assert user1["username"] == "user_0"

        # Özel user
        user2 = user_factory("ahmet", "ahmet@test.com")
        assert user2["username"] == "ahmet"
        assert user2["email"] == "ahmet@test.com"

        # Başka user
        user3 = user_factory("mehmet")
        assert user3["username"] == "mehmet"

    def test_session_config(self, session_config):
        """Session fixture kullanımı"""
        assert "api_url" in session_config
        assert session_config["timeout"] == 30

    def test_database_type(self, database_type):
        """Parametrized fixture kullanımı"""
        # Bu test her database_type için çalışır
        print(f"\nTesting with {database_type}")
        assert database_type in ["sqlite", "postgresql", "mysql"]

    @pytest.mark.api
    @pytest.mark.slow
    def test_with_custom_markers(self):
        """Custom marker kullanımı"""
        # Bu test API ve slow olarak işaretli
        # pytest -m "api" ile sadece API testleri çalıştırılabilir
        # pytest -m "not slow" ile yavaş testler atlanabilir
        time.sleep(0.1)
        assert True

    @pytest.mark.smoke
    def test_smoke_test(self):
        """Smoke test - hızlı temel kontrol"""
        assert 1 + 1 == 2


# Fixture composition örneği
@pytest.fixture
def authenticated_user(user_factory):
    """Kimliği doğrulanmış kullanıcı (user_factory'yi kullanır)"""
    user = user_factory("auth_user")
    user["token"] = "abc123xyz"
    user["authenticated"] = True
    return user


def test_authenticated_user_fixture(authenticated_user):
    """Composed fixture kullanımı"""
    assert authenticated_user["authenticated"] is True
    assert "token" in authenticated_user


# ============================================================================
# ALIŞTIRSINI ÇALIŞTIRMA
# ============================================================================

if __name__ == "__main__":
    """
    Testleri çalıştırmak için:

    # Tüm testler
    pytest exercises.py -v

    # Sadece bir test sınıfı
    pytest exercises.py::TestWeatherAPI -v

    # Sadece bir test
    pytest exercises.py::TestWeatherAPI::test_get_current_weather_success -v

    # Coverage ile
    pytest exercises.py --cov --cov-report=html

    # Slow testleri atla
    pytest exercises.py -m "not slow" -v

    # Sadece API testleri
    pytest exercises.py -m "api" -v

    # Verbose + print output
    pytest exercises.py -v -s

    # Başarısız testlerde debug
    pytest exercises.py --pdb

    # Parallel çalıştırma (pytest-xdist ile)
    pytest exercises.py -n auto
    """
    pytest.main([__file__, "-v", "--tb=short"])
