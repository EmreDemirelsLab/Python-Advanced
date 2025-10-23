# İleri Düzey Nesne Yönelimli Programlama (Advanced OOP)

## İçindekiler
1. [Abstract Base Classes (ABC)](#abstract-base-classes-abc)
2. [Multiple Inheritance ve MRO](#multiple-inheritance-ve-mro)
3. [Metaclasses](#metaclasses)
4. [Descriptors](#descriptors)
5. [Advanced Magic Methods](#advanced-magic-methods)
6. [Dataclasses](#dataclasses)
7. [__slots__ Optimizasyonu](#slots-optimizasyonu)
8. [Property Decorators Advanced](#property-decorators-advanced)

---

## Abstract Base Classes (ABC)

Abstract Base Classes (ABC), Python'da interface benzeri yapılar oluşturmak ve alt sınıfların belirli metodları implement etmesini zorlamak için kullanılır. Production-ready kodlarda tip güvenliği ve API kontratları için kritik öneme sahiptir.

### Örnek 1: Temel ABC Kullanımı

```python
from abc import ABC, abstractmethod
from typing import List, Dict

class DataProcessor(ABC):
    """
    Veri işleme için soyut temel sınıf.
    Tüm alt sınıflar belirtilen metodları implement etmelidir.
    """

    @abstractmethod
    def load_data(self, source: str) -> List[Dict]:
        """Veriyi kaynaktan yükle"""
        pass

    @abstractmethod
    def process_data(self, data: List[Dict]) -> List[Dict]:
        """Veriyi işle"""
        pass

    @abstractmethod
    def save_data(self, data: List[Dict], destination: str) -> bool:
        """İşlenmiş veriyi kaydet"""
        pass

    # Template method pattern - concrete implementation
    def execute_pipeline(self, source: str, destination: str) -> bool:
        """Tüm işlem hattını çalıştır"""
        data = self.load_data(source)
        processed = self.process_data(data)
        return self.save_data(processed, destination)


class CSVProcessor(DataProcessor):
    """CSV dosyaları için concrete implementation"""

    def load_data(self, source: str) -> List[Dict]:
        import csv
        with open(source, 'r') as f:
            return list(csv.DictReader(f))

    def process_data(self, data: List[Dict]) -> List[Dict]:
        # Örnek: tüm sayısal değerleri 2 ile çarp
        return [{k: float(v) * 2 if v.isdigit() else v
                 for k, v in row.items()} for row in data]

    def save_data(self, data: List[Dict], destination: str) -> bool:
        import csv
        if not data:
            return False
        with open(destination, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=data[0].keys())
            writer.writeheader()
            writer.writerows(data)
        return True
```

### Örnek 2: Abstract Properties ve Class Methods

```python
from abc import ABC, abstractmethod
from typing import ClassVar, Optional

class DatabaseConnection(ABC):
    """
    Veritabanı bağlantısı için soyut sınıf.
    Abstract property ve class method kullanımı.
    """

    _instances: ClassVar[Dict[str, 'DatabaseConnection']] = {}

    @property
    @abstractmethod
    def connection_string(self) -> str:
        """Bağlantı string'i döndür"""
        pass

    @property
    @abstractmethod
    def driver_name(self) -> str:
        """Sürücü adını döndür"""
        pass

    @abstractmethod
    def connect(self) -> bool:
        """Bağlantı kur"""
        pass

    @abstractmethod
    def disconnect(self) -> bool:
        """Bağlantıyı kapat"""
        pass

    @classmethod
    def get_instance(cls, connection_id: str) -> Optional['DatabaseConnection']:
        """Singleton pattern ile instance döndür"""
        return cls._instances.get(connection_id)


class PostgreSQLConnection(DatabaseConnection):
    """PostgreSQL için concrete implementation"""

    def __init__(self, host: str, port: int, database: str, user: str, password: str):
        self.host = host
        self.port = port
        self.database = database
        self.user = user
        self.password = password
        self._connection = None
        self._connection_id = f"{host}:{port}/{database}"
        DatabaseConnection._instances[self._connection_id] = self

    @property
    def connection_string(self) -> str:
        return f"postgresql://{self.user}:****@{self.host}:{self.port}/{self.database}"

    @property
    def driver_name(self) -> str:
        return "psycopg2"

    def connect(self) -> bool:
        # Gerçek implementasyonda psycopg2 kullanılır
        print(f"Connecting to {self.connection_string}")
        self._connection = f"Mock connection to {self.database}"
        return True

    def disconnect(self) -> bool:
        if self._connection:
            print(f"Disconnecting from {self.database}")
            self._connection = None
            return True
        return False
```

### Örnek 3: ABC ile Plugin Sistemi

```python
from abc import ABC, abstractmethod
from typing import Any, Dict, List
import inspect

class PluginRegistry:
    """Plugin'leri otomatik olarak kaydeden registry"""
    _plugins: Dict[str, type] = {}

    @classmethod
    def register(cls, plugin_class: type) -> type:
        """Plugin sınıfını kaydet"""
        cls._plugins[plugin_class.__name__] = plugin_class
        return plugin_class

    @classmethod
    def get_plugin(cls, name: str) -> Optional[type]:
        """İsme göre plugin getir"""
        return cls._plugins.get(name)

    @classmethod
    def list_plugins(cls) -> List[str]:
        """Tüm plugin isimlerini listele"""
        return list(cls._plugins.keys())


class Plugin(ABC):
    """Tüm plugin'ler için base class"""

    def __init_subclass__(cls, **kwargs):
        """Alt sınıf oluşturulduğunda otomatik kaydet"""
        super().__init_subclass__(**kwargs)
        if not inspect.isabstract(cls):
            PluginRegistry.register(cls)

    @abstractmethod
    def execute(self, *args, **kwargs) -> Any:
        """Plugin'in ana işlevini çalıştır"""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Plugin adı"""
        pass

    @property
    @abstractmethod
    def version(self) -> str:
        """Plugin versiyonu"""
        pass


class ImageProcessorPlugin(Plugin):
    """Resim işleme plugin'i"""

    @property
    def name(self) -> str:
        return "ImageProcessor"

    @property
    def version(self) -> str:
        return "1.0.0"

    def execute(self, image_path: str, operation: str) -> str:
        return f"Processing {image_path} with {operation}"


class DataValidatorPlugin(Plugin):
    """Veri validasyonu plugin'i"""

    @property
    def name(self) -> str:
        return "DataValidator"

    @property
    def version(self) -> str:
        return "2.1.0"

    def execute(self, data: Dict, schema: Dict) -> bool:
        # Basit validasyon örneği
        return all(key in data for key in schema.get('required', []))
```

---

## Multiple Inheritance ve MRO

Python'da multiple inheritance, bir sınıfın birden fazla parent sınıftan türemesine olanak tanır. Method Resolution Order (MRO), Python'un hangi sırayla metodları arayacağını belirleyen C3 linearization algoritmasıdır.

### Örnek 4: Multiple Inheritance ve MRO

```python
class LoggerMixin:
    """Loglama yetenekleri ekleyen mixin"""

    def log(self, message: str, level: str = "INFO") -> None:
        print(f"[{level}] {self.__class__.__name__}: {message}")

    def log_method_call(self, method_name: str, *args, **kwargs) -> None:
        self.log(f"Calling {method_name} with args={args}, kwargs={kwargs}")


class SerializableMixin:
    """Serileştirme yetenekleri ekleyen mixin"""

    def to_dict(self) -> Dict[str, Any]:
        """Nesneyi dictionary'ye çevir"""
        return {k: v for k, v in self.__dict__.items()
                if not k.startswith('_')}

    def from_dict(self, data: Dict[str, Any]) -> 'SerializableMixin':
        """Dictionary'den nesne oluştur"""
        for key, value in data.items():
            setattr(self, key, value)
        return self


class CacheableMixin:
    """Önbellekleme yetenekleri ekleyen mixin"""
    _cache: ClassVar[Dict[str, Any]] = {}

    def cache_set(self, key: str, value: Any, ttl: int = 300) -> None:
        """Cache'e değer ekle"""
        import time
        self._cache[key] = {'value': value, 'expires': time.time() + ttl}

    def cache_get(self, key: str) -> Optional[Any]:
        """Cache'den değer al"""
        import time
        if key in self._cache:
            item = self._cache[key]
            if time.time() < item['expires']:
                return item['value']
            else:
                del self._cache[key]
        return None


class APIClient(LoggerMixin, SerializableMixin, CacheableMixin):
    """
    Multiple inheritance kullanarak zenginleştirilmiş API client.
    MRO: APIClient -> LoggerMixin -> SerializableMixin -> CacheableMixin -> object
    """

    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url
        self.api_key = api_key
        self.log(f"Initialized with base_url: {base_url}")

    def get(self, endpoint: str, use_cache: bool = True) -> Dict:
        """API GET request"""
        cache_key = f"{self.base_url}{endpoint}"

        if use_cache:
            cached = self.cache_get(cache_key)
            if cached:
                self.log(f"Cache hit for {endpoint}")
                return cached

        self.log_method_call("get", endpoint)
        # Gerçek API çağrısı simülasyonu
        result = {"endpoint": endpoint, "data": "mock_data"}
        self.cache_set(cache_key, result)
        return result


# MRO'yu kontrol et
print(APIClient.__mro__)
# (<class 'APIClient'>, <class 'LoggerMixin'>, <class 'SerializableMixin'>,
#  <class 'CacheableMixin'>, <class 'object'>)
```

### Örnek 5: Diamond Problem ve super()

```python
class Base:
    """Temel sınıf"""

    def __init__(self):
        print("Base.__init__")
        self.base_value = "base"

    def method(self):
        print("Base.method")


class Left(Base):
    """Sol dal"""

    def __init__(self):
        print("Left.__init__")
        super().__init__()
        self.left_value = "left"

    def method(self):
        print("Left.method")
        super().method()


class Right(Base):
    """Sağ dal"""

    def __init__(self):
        print("Right.__init__")
        super().__init__()
        self.right_value = "right"

    def method(self):
        print("Right.method")
        super().method()


class Diamond(Left, Right):
    """
    Diamond inheritance pattern.
    MRO: Diamond -> Left -> Right -> Base -> object
    super() sayesinde Base.__init__ sadece bir kez çağrılır.
    """

    def __init__(self):
        print("Diamond.__init__")
        super().__init__()
        self.diamond_value = "diamond"

    def method(self):
        print("Diamond.method")
        super().method()


# Test
d = Diamond()
# Output:
# Diamond.__init__
# Left.__init__
# Right.__init__
# Base.__init__

d.method()
# Output:
# Diamond.method
# Left.method
# Right.method
# Base.method
```

### Örnek 6: Cooperative Multiple Inheritance

```python
from typing import Protocol
import functools

class Validator:
    """Validasyon base class"""

    def validate(self, value: Any) -> bool:
        return True


class TypeValidator(Validator):
    """Tip validasyonu"""

    def __init__(self, expected_type: type, **kwargs):
        self.expected_type = expected_type
        super().__init__(**kwargs)

    def validate(self, value: Any) -> bool:
        if not isinstance(value, self.expected_type):
            raise TypeError(f"Expected {self.expected_type}, got {type(value)}")
        return super().validate(value)


class RangeValidator(Validator):
    """Aralık validasyonu"""

    def __init__(self, min_val: float = None, max_val: float = None, **kwargs):
        self.min_val = min_val
        self.max_val = max_val
        super().__init__(**kwargs)

    def validate(self, value: Any) -> bool:
        if self.min_val is not None and value < self.min_val:
            raise ValueError(f"Value {value} is less than minimum {self.min_val}")
        if self.max_val is not None and value > self.max_val:
            raise ValueError(f"Value {value} is greater than maximum {self.max_val}")
        return super().validate(value)


class LengthValidator(Validator):
    """Uzunluk validasyonu"""

    def __init__(self, min_length: int = None, max_length: int = None, **kwargs):
        self.min_length = min_length
        self.max_length = max_length
        super().__init__(**kwargs)

    def validate(self, value: Any) -> bool:
        length = len(value)
        if self.min_length is not None and length < self.min_length:
            raise ValueError(f"Length {length} is less than minimum {self.min_length}")
        if self.max_length is not None and length > self.max_length:
            raise ValueError(f"Length {length} is greater than maximum {self.max_length}")
        return super().validate(value)


class StringValidator(TypeValidator, LengthValidator, RangeValidator):
    """String için composite validator"""

    def __init__(self, min_length: int = None, max_length: int = None):
        super().__init__(
            expected_type=str,
            min_length=min_length,
            max_length=max_length
        )


# Kullanım
validator = StringValidator(min_length=3, max_length=10)
print(validator.validate("hello"))  # True
# print(validator.validate("ab"))  # ValueError: Length 2 is less than minimum 3
```

---

## Metaclasses

Metaclass'lar, sınıfların sınıflarıdır. Sınıf oluşturma sürecini kontrol eder ve customize ederler. Framework ve library geliştirmede güçlü bir araçtır.

### Örnek 7: Temel Metaclass

```python
class SingletonMeta(type):
    """
    Singleton pattern implementasyonu için metaclass.
    Her sınıf için tek bir instance oluşturulmasını garanti eder.
    """
    _instances: Dict[type, object] = {}
    _lock = __import__('threading').Lock()

    def __call__(cls, *args, **kwargs):
        """
        Sınıf çağrıldığında (__call__) devreye girer.
        Thread-safe singleton implementasyonu.
        """
        if cls not in cls._instances:
            with cls._lock:
                if cls not in cls._instances:
                    instance = super().__call__(*args, **kwargs)
                    cls._instances[cls] = instance
        return cls._instances[cls]


class DatabasePool(metaclass=SingletonMeta):
    """Singleton pattern kullanan database connection pool"""

    def __init__(self, max_connections: int = 10):
        print(f"Creating DatabasePool with {max_connections} connections")
        self.max_connections = max_connections
        self.connections = []

    def get_connection(self):
        return f"Connection from pool of {self.max_connections}"


# Test
pool1 = DatabasePool(max_connections=5)
pool2 = DatabasePool(max_connections=20)  # Parametreler ignore edilir
print(pool1 is pool2)  # True - aynı instance
```

### Örnek 8: Attribute Validation Metaclass

```python
class AttributeValidationMeta(type):
    """
    Sınıf attribute'larını otomatik validate eden metaclass.
    Type hints kullanarak runtime type checking yapar.
    """

    def __new__(mcs, name, bases, namespace, **kwargs):
        """Yeni sınıf oluşturulurken çağrılır"""

        # __annotations__ varsa, her attribute için property oluştur
        if '__annotations__' in namespace:
            annotations = namespace['__annotations__']

            for attr_name, attr_type in annotations.items():
                # Private attribute adı
                private_name = f'_{attr_name}'

                # Getter
                def make_getter(private_name):
                    def getter(self):
                        return getattr(self, private_name, None)
                    return getter

                # Setter with validation
                def make_setter(private_name, expected_type):
                    def setter(self, value):
                        if not isinstance(value, expected_type):
                            raise TypeError(
                                f"{attr_name} must be {expected_type.__name__}, "
                                f"got {type(value).__name__}"
                            )
                        setattr(self, private_name, value)
                    return setter

                # Property oluştur ve namespace'e ekle
                namespace[attr_name] = property(
                    make_getter(private_name),
                    make_setter(private_name, attr_type)
                )

        return super().__new__(mcs, name, bases, namespace)


class User(metaclass=AttributeValidationMeta):
    """Type-checked attributes ile user sınıfı"""
    name: str
    age: int
    email: str

    def __init__(self, name: str, age: int, email: str):
        self.name = name
        self.age = age
        self.email = email


# Test
user = User("John", 30, "john@example.com")
print(user.name)  # John
# user.age = "thirty"  # TypeError: age must be int, got str
```

### Örnek 9: ORM-style Metaclass

```python
from typing import Any, Dict, List

class Field:
    """Database field descriptor"""

    def __init__(self, field_type: type, primary_key: bool = False,
                 nullable: bool = True, default: Any = None):
        self.field_type = field_type
        self.primary_key = primary_key
        self.nullable = nullable
        self.default = default
        self.name = None

    def __set_name__(self, owner, name):
        self.name = name

    def __get__(self, instance, owner):
        if instance is None:
            return self
        return instance.__dict__.get(self.name, self.default)

    def __set__(self, instance, value):
        if value is None and not self.nullable:
            raise ValueError(f"{self.name} cannot be None")
        if value is not None and not isinstance(value, self.field_type):
            raise TypeError(f"{self.name} must be {self.field_type.__name__}")
        instance.__dict__[self.name] = value


class ModelMeta(type):
    """
    ORM-style model metaclass.
    Field'ları otomatik olarak toplar ve metadata oluşturur.
    """

    def __new__(mcs, name, bases, namespace, **kwargs):
        # Field'ları topla
        fields = {}
        for key, value in namespace.items():
            if isinstance(value, Field):
                fields[key] = value

        # Metadata ekle
        namespace['_fields'] = fields
        namespace['_table_name'] = kwargs.get('table_name', name.lower())

        cls = super().__new__(mcs, name, bases, namespace)
        return cls

    def __init__(cls, name, bases, namespace, **kwargs):
        super().__init__(name, bases, namespace)


class Model(metaclass=ModelMeta):
    """Base model class"""

    def __init__(self, **kwargs):
        for field_name in self._fields:
            value = kwargs.get(field_name)
            setattr(self, field_name, value)

    def save(self):
        """Simulated save operation"""
        values = {name: getattr(self, name) for name in self._fields}
        print(f"INSERT INTO {self._table_name} {values}")
        return self

    @classmethod
    def get_schema(cls) -> Dict[str, Field]:
        """Return field schema"""
        return cls._fields.copy()


class Product(Model, table_name='products'):
    """Product model örneği"""
    id = Field(int, primary_key=True)
    name = Field(str, nullable=False)
    price = Field(float, nullable=False)
    description = Field(str, default="")


# Kullanım
product = Product(id=1, name="Laptop", price=999.99, description="Gaming laptop")
product.save()
print(Product.get_schema())
```

### Örnek 10: Method Registry Metaclass

```python
import inspect
from typing import Callable

class MethodRegistryMeta(type):
    """
    Belirli decorator'lara sahip metodları otomatik kaydeden metaclass.
    API endpoint'leri, event handler'lar gibi senaryolar için kullanılır.
    """

    def __new__(mcs, name, bases, namespace, **kwargs):
        cls = super().__new__(mcs, name, bases, namespace)

        # Tüm metodları tara
        routes = {}
        for attr_name, attr_value in namespace.items():
            if inspect.isfunction(attr_value):
                # Eğer metodun _route attribute'u varsa kaydet
                if hasattr(attr_value, '_route'):
                    route_info = attr_value._route
                    routes[route_info['path']] = {
                        'method': attr_value,
                        'http_method': route_info.get('http_method', 'GET'),
                        'name': attr_name
                    }

        cls._routes = routes
        return cls


def route(path: str, method: str = 'GET'):
    """Decorator to mark methods as routes"""
    def decorator(func: Callable) -> Callable:
        func._route = {'path': path, 'http_method': method}
        return func
    return decorator


class APIController(metaclass=MethodRegistryMeta):
    """Base controller with automatic route registration"""

    @classmethod
    def get_routes(cls) -> Dict:
        """Return all registered routes"""
        return cls._routes


class UserController(APIController):
    """User management API"""

    @route('/users', method='GET')
    def list_users(self):
        return {"users": ["user1", "user2"]}

    @route('/users/<id>', method='GET')
    def get_user(self, user_id: int):
        return {"user": f"User {user_id}"}

    @route('/users', method='POST')
    def create_user(self, data: Dict):
        return {"created": data}

    def helper_method(self):
        """This won't be registered as a route"""
        pass


# Test
print(UserController.get_routes())
# {'/users': {...}, '/users/<id>': {...}}
```

---

## Descriptors

Descriptor'lar, attribute access'i kontrol etmek için kullanılan protocol'dür. `__get__`, `__set__`, ve `__delete__` metodlarını implement ederek çalışırlar.

### Örnek 11: Data ve Non-Data Descriptors

```python
class DataDescriptor:
    """
    Data descriptor: hem __get__ hem __set__ implement eder.
    Instance __dict__ üzerinde önceliğe sahiptir.
    """

    def __init__(self, name: str = None):
        self.name = name

    def __set_name__(self, owner, name):
        self.name = name

    def __get__(self, instance, owner):
        if instance is None:
            return self
        return instance.__dict__.get(f'_{self.name}', None)

    def __set__(self, instance, value):
        print(f"Setting {self.name} to {value}")
        instance.__dict__[f'_{self.name}'] = value

    def __delete__(self, instance):
        print(f"Deleting {self.name}")
        del instance.__dict__[f'_{self.name}']


class NonDataDescriptor:
    """
    Non-data descriptor: sadece __get__ implement eder.
    Instance __dict__ üzerinde önceliği yoktur.
    """

    def __init__(self, func):
        self.func = func
        self.name = func.__name__

    def __get__(self, instance, owner):
        if instance is None:
            return self
        print(f"Accessing {self.name}")
        return self.func(instance)


class Example:
    # Data descriptor
    managed_attr = DataDescriptor()

    # Non-data descriptor (method gibi)
    @NonDataDescriptor
    def computed(self):
        return "computed value"

    def __init__(self):
        self.managed_attr = "initial"


# Test
obj = Example()
print(obj.managed_attr)  # Descriptor __get__ çağrılır
obj.managed_attr = "new"  # Descriptor __set__ çağrılır

# Non-data descriptor instance __dict__ ile override edilebilir
print(obj.computed)  # Descriptor __get__ çağrılır
obj.__dict__['computed'] = "overridden"
print(obj.computed)  # "overridden" döner, descriptor atlanır
```

### Örnek 12: Typed Property Descriptor

```python
from typing import Type, Any, Optional
import re

class TypedProperty:
    """
    Type checking ve validation yapan descriptor.
    Production-ready property implementasyonu.
    """

    def __init__(self,
                 expected_type: Type,
                 min_value: Any = None,
                 max_value: Any = None,
                 pattern: Optional[str] = None,
                 choices: Optional[List] = None):
        self.expected_type = expected_type
        self.min_value = min_value
        self.max_value = max_value
        self.pattern = re.compile(pattern) if pattern else None
        self.choices = choices
        self.name = None

    def __set_name__(self, owner, name):
        self.name = name

    def __get__(self, instance, owner):
        if instance is None:
            return self
        return instance.__dict__.get(self.name)

    def __set__(self, instance, value):
        # Type check
        if not isinstance(value, self.expected_type):
            raise TypeError(
                f"{self.name} must be {self.expected_type.__name__}, "
                f"got {type(value).__name__}"
            )

        # Range validation
        if self.min_value is not None and value < self.min_value:
            raise ValueError(f"{self.name} must be >= {self.min_value}")
        if self.max_value is not None and value > self.max_value:
            raise ValueError(f"{self.name} must be <= {self.max_value}")

        # Pattern validation (for strings)
        if self.pattern and isinstance(value, str):
            if not self.pattern.match(value):
                raise ValueError(f"{self.name} must match pattern {self.pattern.pattern}")

        # Choices validation
        if self.choices and value not in self.choices:
            raise ValueError(f"{self.name} must be one of {self.choices}")

        instance.__dict__[self.name] = value

    def __delete__(self, instance):
        raise AttributeError(f"Cannot delete {self.name}")


class Product:
    """TypedProperty kullanarak validated product class"""

    name = TypedProperty(str, min_value=1, max_value=100)
    price = TypedProperty(float, min_value=0.0)
    sku = TypedProperty(str, pattern=r'^[A-Z]{3}-\d{4}$')
    category = TypedProperty(str, choices=['Electronics', 'Clothing', 'Food'])
    quantity = TypedProperty(int, min_value=0)

    def __init__(self, name: str, price: float, sku: str, category: str, quantity: int):
        self.name = name
        self.price = price
        self.sku = sku
        self.category = category
        self.quantity = quantity


# Test
product = Product(
    name="Laptop",
    price=999.99,
    sku="LAP-1234",
    category="Electronics",
    quantity=10
)

# product.price = -100  # ValueError: price must be >= 0.0
# product.sku = "invalid"  # ValueError: sku must match pattern...
```

### Örnek 13: Lazy Property Descriptor

```python
import time
from functools import wraps

class LazyProperty:
    """
    Lazy evaluation descriptor.
    İlk erişimde hesaplanır ve cache'lenir.
    """

    def __init__(self, func):
        self.func = func
        self.name = func.__name__
        self.__doc__ = func.__doc__

    def __get__(self, instance, owner):
        if instance is None:
            return self

        # Cache key
        cache_key = f'_lazy_{self.name}'

        # Eğer cache'de yoksa hesapla
        if cache_key not in instance.__dict__:
            print(f"Computing {self.name}...")
            value = self.func(instance)
            instance.__dict__[cache_key] = value
        else:
            print(f"Using cached {self.name}...")

        return instance.__dict__[cache_key]

    def __set__(self, instance, value):
        raise AttributeError(f"Cannot set {self.name}")


class DataAnalyzer:
    """Expensive hesaplamalar için lazy property örneği"""

    def __init__(self, data: List[float]):
        self.data = data

    @LazyProperty
    def mean(self) -> float:
        """Ortalama hesapla (expensive operation)"""
        time.sleep(1)  # Simulated expensive computation
        return sum(self.data) / len(self.data)

    @LazyProperty
    def variance(self) -> float:
        """Varyans hesapla (expensive operation)"""
        time.sleep(1)  # Simulated expensive computation
        mean = self.mean
        return sum((x - mean) ** 2 for x in self.data) / len(self.data)

    @LazyProperty
    def std_dev(self) -> float:
        """Standart sapma hesapla"""
        return self.variance ** 0.5


# Test
analyzer = DataAnalyzer([1, 2, 3, 4, 5])
print(analyzer.mean)  # Computing mean... (1 saniye)
print(analyzer.mean)  # Using cached mean... (anında)
print(analyzer.variance)  # Computing variance... (1 saniye, mean cache'den)
```

### Örnek 14: Validation Chain Descriptor

```python
from typing import Callable, List

class Validator:
    """Base validator class"""

    def __call__(self, value: Any) -> None:
        """Raise exception if validation fails"""
        pass


class RangeValidator(Validator):
    """Range validation"""

    def __init__(self, min_val=None, max_val=None):
        self.min_val = min_val
        self.max_val = max_val

    def __call__(self, value: Any) -> None:
        if self.min_val is not None and value < self.min_val:
            raise ValueError(f"Value must be >= {self.min_val}")
        if self.max_val is not None and value > self.max_val:
            raise ValueError(f"Value must be <= {self.max_val}")


class PatternValidator(Validator):
    """Regex pattern validation"""

    def __init__(self, pattern: str):
        self.pattern = re.compile(pattern)

    def __call__(self, value: Any) -> None:
        if not self.pattern.match(str(value)):
            raise ValueError(f"Value must match pattern {self.pattern.pattern}")


class CustomValidator(Validator):
    """Custom function validator"""

    def __init__(self, func: Callable[[Any], bool], message: str):
        self.func = func
        self.message = message

    def __call__(self, value: Any) -> None:
        if not self.func(value):
            raise ValueError(self.message)


class ValidatedProperty:
    """
    Validator chain kullanarak property descriptor.
    Multiple validation rules apply edilebilir.
    """

    def __init__(self, *validators: Validator):
        self.validators = validators
        self.name = None

    def __set_name__(self, owner, name):
        self.name = name

    def __get__(self, instance, owner):
        if instance is None:
            return self
        return instance.__dict__.get(self.name)

    def __set__(self, instance, value):
        # Tüm validator'ları çalıştır
        for validator in self.validators:
            validator(value)
        instance.__dict__[self.name] = value


class User:
    """Multiple validation kullanarak user class"""

    age = ValidatedProperty(
        RangeValidator(min_val=0, max_val=150)
    )

    email = ValidatedProperty(
        PatternValidator(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
    )

    username = ValidatedProperty(
        PatternValidator(r'^[a-zA-Z0-9_]{3,20}$'),
        CustomValidator(
            lambda x: not x.startswith('_'),
            "Username cannot start with underscore"
        )
    )

    def __init__(self, age: int, email: str, username: str):
        self.age = age
        self.email = email
        self.username = username


# Test
user = User(age=25, email="john@example.com", username="john_doe")
# user.age = 200  # ValueError: Value must be <= 150
# user.email = "invalid"  # ValueError: Value must match pattern...
```

---

## Advanced Magic Methods

Magic method'lar (dunder methods), Python'da operator overloading ve özel davranışlar implement etmek için kullanılır.

### Örnek 15: Context Manager Protocol

```python
import time
from typing import Optional
import traceback

class DatabaseTransaction:
    """
    Context manager protocol ile transaction yönetimi.
    __enter__ ve __exit__ magic methods kullanır.
    """

    def __init__(self, connection, isolation_level: str = 'READ_COMMITTED'):
        self.connection = connection
        self.isolation_level = isolation_level
        self.transaction = None

    def __enter__(self):
        """Context'e girildiğinde transaction başlat"""
        print(f"Starting transaction with isolation level: {self.isolation_level}")
        self.transaction = f"Transaction-{time.time()}"
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context'ten çıkılırken commit veya rollback yap"""
        if exc_type is None:
            # No exception, commit
            print("Committing transaction")
            return False
        else:
            # Exception occurred, rollback
            print(f"Rolling back transaction due to {exc_type.__name__}: {exc_val}")
            print("Traceback:")
            traceback.print_tb(exc_tb)
            return False  # False = exception'ı propagate et


class DatabaseConnection:
    """Database connection with transaction support"""

    def transaction(self, isolation_level: str = 'READ_COMMITTED'):
        """Transaction context manager döndür"""
        return DatabaseTransaction(self, isolation_level)

    def execute(self, query: str):
        """Execute SQL query"""
        print(f"Executing: {query}")


# Kullanım
db = DatabaseConnection()

# Başarılı transaction
with db.transaction() as txn:
    db.execute("INSERT INTO users VALUES (1, 'John')")
    db.execute("INSERT INTO orders VALUES (1, 100)")

# Hatalı transaction (rollback)
try:
    with db.transaction() as txn:
        db.execute("INSERT INTO users VALUES (2, 'Jane')")
        raise ValueError("Something went wrong")
        db.execute("INSERT INTO orders VALUES (2, 200)")
except ValueError:
    pass
```

### Örnek 16: Callable Objects ve __call__

```python
from typing import Callable, Any
import time

class RateLimiter:
    """
    __call__ kullanarak rate limiting implementasyonu.
    Function decorator olarak kullanılabilir.
    """

    def __init__(self, max_calls: int, period: float):
        self.max_calls = max_calls
        self.period = period
        self.calls = []

    def __call__(self, func: Callable) -> Callable:
        """Decorator olarak kullanım"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            now = time.time()

            # Eski çağrıları temizle
            self.calls = [call_time for call_time in self.calls
                         if now - call_time < self.period]

            # Rate limit kontrolü
            if len(self.calls) >= self.max_calls:
                sleep_time = self.period - (now - self.calls[0])
                print(f"Rate limit reached. Sleeping for {sleep_time:.2f}s")
                time.sleep(sleep_time)
                self.calls.pop(0)

            self.calls.append(now)
            return func(*args, **kwargs)

        return wrapper


class Memoize:
    """
    __call__ ile memoization implementasyonu.
    Expensive fonksiyonları cache'ler.
    """

    def __init__(self, func: Callable):
        self.func = func
        self.cache = {}
        self.hits = 0
        self.misses = 0

    def __call__(self, *args, **kwargs):
        # Kwargs'ı hashable yap
        key = (args, tuple(sorted(kwargs.items())))

        if key in self.cache:
            self.hits += 1
            print(f"Cache hit! (hits: {self.hits}, misses: {self.misses})")
            return self.cache[key]

        self.misses += 1
        print(f"Cache miss! (hits: {self.hits}, misses: {self.misses})")
        result = self.func(*args, **kwargs)
        self.cache[key] = result
        return result

    def clear_cache(self):
        """Cache'i temizle"""
        self.cache.clear()
        self.hits = 0
        self.misses = 0


# Kullanım
@RateLimiter(max_calls=3, period=5.0)
@Memoize
def expensive_api_call(endpoint: str) -> Dict:
    """Simulated expensive API call"""
    time.sleep(0.5)
    return {"endpoint": endpoint, "data": "response"}


# Test
for i in range(5):
    result = expensive_api_call("/api/users")
```

### Örnek 17: Comparison ve Hashing Magic Methods

```python
from functools import total_ordering
from typing import Any

@total_ordering
class Version:
    """
    Semantic versioning comparison.
    __eq__, __lt__ ve @total_ordering ile tüm karşılaştırmalar.
    """

    def __init__(self, version_string: str):
        parts = version_string.split('.')
        self.major = int(parts[0])
        self.minor = int(parts[1]) if len(parts) > 1 else 0
        self.patch = int(parts[2]) if len(parts) > 2 else 0

    def __eq__(self, other):
        if not isinstance(other, Version):
            return NotImplemented
        return (self.major, self.minor, self.patch) == \
               (other.major, other.minor, other.patch)

    def __lt__(self, other):
        if not isinstance(other, Version):
            return NotImplemented
        return (self.major, self.minor, self.patch) < \
               (other.major, other.minor, other.patch)

    def __hash__(self):
        """Hash implementasyonu (set, dict key için gerekli)"""
        return hash((self.major, self.minor, self.patch))

    def __repr__(self):
        return f"Version('{self.major}.{self.minor}.{self.patch}')"

    def __str__(self):
        return f"{self.major}.{self.minor}.{self.patch}"


class Point:
    """
    2D point with all comparison and arithmetic magic methods.
    """

    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

    # Arithmetic operators
    def __add__(self, other):
        if isinstance(other, Point):
            return Point(self.x + other.x, self.y + other.y)
        return NotImplemented

    def __sub__(self, other):
        if isinstance(other, Point):
            return Point(self.x - other.x, self.y - other.y)
        return NotImplemented

    def __mul__(self, scalar):
        if isinstance(scalar, (int, float)):
            return Point(self.x * scalar, self.y * scalar)
        return NotImplemented

    def __rmul__(self, scalar):
        return self.__mul__(scalar)

    def __truediv__(self, scalar):
        if isinstance(scalar, (int, float)):
            return Point(self.x / scalar, self.y / scalar)
        return NotImplemented

    # Unary operators
    def __neg__(self):
        return Point(-self.x, -self.y)

    def __abs__(self):
        return (self.x ** 2 + self.y ** 2) ** 0.5

    # Comparison
    def __eq__(self, other):
        if isinstance(other, Point):
            return self.x == other.x and self.y == other.y
        return False

    def __hash__(self):
        return hash((self.x, self.y))

    def __repr__(self):
        return f"Point({self.x}, {self.y})"


# Test
v1 = Version("1.2.3")
v2 = Version("1.2.4")
v3 = Version("2.0.0")

print(v1 < v2)  # True
print(v2 < v3)  # True
print(sorted([v3, v1, v2]))  # [Version('1.2.3'), Version('1.2.4'), Version('2.0.0')]

p1 = Point(1, 2)
p2 = Point(3, 4)
print(p1 + p2)  # Point(4, 6)
print(p1 * 2)   # Point(2, 4)
print(abs(p1))  # 2.23606797749979
```

### Örnek 18: Container Magic Methods

```python
from typing import Iterator, Any
from collections.abc import MutableSequence

class CircularBuffer(MutableSequence):
    """
    Container magic methods ile circular buffer implementasyonu.
    __getitem__, __setitem__, __len__, __iter__, etc.
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = [None] * capacity
        self.head = 0
        self.tail = 0
        self.size = 0

    def __len__(self) -> int:
        """len(buffer)"""
        return self.size

    def __getitem__(self, index: int) -> Any:
        """buffer[index]"""
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]

        if not -len(self) <= index < len(self):
            raise IndexError("Index out of range")

        if index < 0:
            index += len(self)

        actual_index = (self.head + index) % self.capacity
        return self.buffer[actual_index]

    def __setitem__(self, index: int, value: Any) -> None:
        """buffer[index] = value"""
        if isinstance(index, slice):
            raise NotImplementedError("Slice assignment not supported")

        if not -len(self) <= index < len(self):
            raise IndexError("Index out of range")

        if index < 0:
            index += len(self)

        actual_index = (self.head + index) % self.capacity
        self.buffer[actual_index] = value

    def __delitem__(self, index: int) -> None:
        """del buffer[index]"""
        raise NotImplementedError("Deletion not supported in circular buffer")

    def __iter__(self) -> Iterator:
        """for item in buffer"""
        for i in range(len(self)):
            yield self[i]

    def __contains__(self, value: Any) -> bool:
        """value in buffer"""
        return any(item == value for item in self)

    def __reversed__(self):
        """reversed(buffer)"""
        for i in range(len(self) - 1, -1, -1):
            yield self[i]

    def insert(self, index: int, value: Any) -> None:
        """MutableSequence requirement"""
        raise NotImplementedError("Insert not supported in circular buffer")

    def append(self, value: Any) -> None:
        """Add item to buffer"""
        if self.size < self.capacity:
            self.buffer[self.tail] = value
            self.tail = (self.tail + 1) % self.capacity
            self.size += 1
        else:
            # Overwrite oldest item
            self.buffer[self.tail] = value
            self.tail = (self.tail + 1) % self.capacity
            self.head = (self.head + 1) % self.capacity

    def __repr__(self):
        return f"CircularBuffer({list(self)})"


# Test
buffer = CircularBuffer(3)
buffer.append(1)
buffer.append(2)
buffer.append(3)
print(buffer)  # CircularBuffer([1, 2, 3])

buffer.append(4)  # Overwrites 1
print(buffer)  # CircularBuffer([2, 3, 4])

print(buffer[0])  # 2
print(2 in buffer)  # True
print(list(reversed(buffer)))  # [4, 3, 2]
```

---

## Dataclasses

Dataclass'lar, Python 3.7+ ile gelen ve boilerplate code'u azaltan bir özelliktir. `__init__`, `__repr__`, `__eq__` gibi metodları otomatik oluşturur.

### Örnek 19: Advanced Dataclass Features

```python
from dataclasses import dataclass, field, asdict, astuple, replace
from typing import List, Optional, ClassVar
from datetime import datetime
import json

@dataclass(order=True, frozen=False)
class Product:
    """
    Advanced dataclass örneği.
    order=True: comparison operators oluşturur
    frozen=False: mutable (frozen=True immutable yapar)
    """

    # sort_index için kullanılacak (comparison'da)
    sort_index: float = field(init=False, repr=False)

    name: str
    price: float
    category: str
    stock: int = 0
    tags: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)

    # Class variable (instance'a ait değil)
    _registry: ClassVar[Dict[str, 'Product']] = {}

    def __post_init__(self):
        """Dataclass oluşturulduktan sonra çağrılır"""
        # Sort için price kullan
        self.sort_index = self.price

        # Registry'ye ekle
        Product._registry[self.name] = self

        # Validation
        if self.price < 0:
            raise ValueError("Price cannot be negative")
        if self.stock < 0:
            raise ValueError("Stock cannot be negative")

    def apply_discount(self, percentage: float) -> 'Product':
        """İndirim uygula ve yeni Product döndür"""
        return replace(self, price=self.price * (1 - percentage / 100))

    def to_json(self) -> str:
        """JSON'a çevir"""
        data = asdict(self)
        data['created_at'] = data['created_at'].isoformat()
        return json.dumps(data, indent=2)

    @classmethod
    def from_dict(cls, data: Dict) -> 'Product':
        """Dict'ten Product oluştur"""
        if 'created_at' in data and isinstance(data['created_at'], str):
            data['created_at'] = datetime.fromisoformat(data['created_at'])
        return cls(**data)


# Test
p1 = Product(name="Laptop", price=1000, category="Electronics", tags=["tech", "portable"])
p2 = Product(name="Mouse", price=50, category="Electronics", tags=["tech"])

# Comparison (order=True sayesinde)
print(p1 > p2)  # True (price'a göre)

# asdict, astuple
print(asdict(p1))
print(astuple(p1))

# replace (immutable update)
discounted = p1.apply_discount(10)
print(f"Original: ${p1.price}, Discounted: ${discounted.price}")

# JSON serialization
print(p1.to_json())
```

### Örnek 20: Dataclass Inheritance ve Field Customization

```python
from dataclasses import dataclass, field, InitVar
from typing import Optional, Any
from abc import ABC, abstractmethod

@dataclass
class Entity(ABC):
    """Base entity class"""
    id: Optional[int] = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    @abstractmethod
    def validate(self) -> bool:
        """Validation logic"""
        pass


@dataclass
class User(Entity):
    """
    User entity with complex field customization.
    InitVar: sadece __init__'te kullanılan, instance attribute olmayan parametre
    """

    # Normal fields
    username: str
    email: str

    # Field with metadata
    age: int = field(metadata={"min": 0, "max": 150})

    # Field with custom compare and repr
    password_hash: str = field(repr=False, compare=False)

    # InitVar: sadece __post_init__'te kullanılır
    password: InitVar[Optional[str]] = None

    # Field with factory function
    preferences: Dict[str, Any] = field(default_factory=dict)

    # Excluded from __init__
    login_count: int = field(default=0, init=False)

    def __post_init__(self, password: Optional[str]):
        """Post-initialization processing"""
        if password:
            # Simulated password hashing
            self.password_hash = f"hashed_{password}"

        # Update timestamp
        self.updated_at = datetime.now()

    def validate(self) -> bool:
        """Validate user data"""
        metadata = self.__dataclass_fields__['age'].metadata
        if not metadata['min'] <= self.age <= metadata['max']:
            return False

        if '@' not in self.email:
            return False

        return True

    def increment_login(self):
        """Increment login counter"""
        self.login_count += 1
        self.updated_at = datetime.now()


@dataclass
class Admin(User):
    """Admin user with additional fields"""
    permissions: List[str] = field(default_factory=lambda: ["read", "write"])
    is_superuser: bool = False

    def validate(self) -> bool:
        """Extended validation"""
        if not super().validate():
            return False

        # Admin-specific validation
        if not self.permissions:
            return False

        return True


# Test
user = User(
    username="john_doe",
    email="john@example.com",
    age=30,
    password="secret123"
)

print(user)  # password_hash görünmez (repr=False)
print(user.validate())  # True
user.increment_login()
print(user.login_count)  # 1

admin = Admin(
    username="admin",
    email="admin@example.com",
    age=35,
    password="admin123",
    is_superuser=True
)
print(admin.permissions)  # ["read", "write"]
```

### Örnek 21: Frozen Dataclass ve Post-Init Processing

```python
from dataclasses import dataclass, field
from typing import Tuple, FrozenSet
import hashlib

@dataclass(frozen=True)
class ImmutableConfig:
    """
    Frozen dataclass: tamamen immutable.
    Hash hesaplanabilir, dict key olarak kullanılabilir.
    """

    host: str
    port: int
    use_ssl: bool = True
    timeout: float = 30.0

    # Frozen olduğu için __post_init__ içinde normal assignment çalışmaz
    # object.__setattr__ kullanılmalı
    connection_string: str = field(init=False)
    config_hash: str = field(init=False)

    def __post_init__(self):
        """Computed fields for frozen dataclass"""
        protocol = "https" if self.use_ssl else "http"
        conn_str = f"{protocol}://{self.host}:{self.port}"

        # Frozen dataclass'ta değer atama
        object.__setattr__(self, 'connection_string', conn_str)

        # Hash hesapla
        hash_input = f"{self.host}:{self.port}:{self.use_ssl}:{self.timeout}"
        config_hash = hashlib.md5(hash_input.encode()).hexdigest()
        object.__setattr__(self, 'config_hash', config_hash)


@dataclass(frozen=True)
class Point3D:
    """3D point (immutable)"""
    x: float
    y: float
    z: float

    distance: float = field(init=False)

    def __post_init__(self):
        """Calculate distance from origin"""
        dist = (self.x ** 2 + self.y ** 2 + self.z ** 2) ** 0.5
        object.__setattr__(self, 'distance', dist)

    def translate(self, dx: float, dy: float, dz: float) -> 'Point3D':
        """Return new translated point (immutable)"""
        return Point3D(self.x + dx, self.y + dy, self.z + dz)

    def scale(self, factor: float) -> 'Point3D':
        """Return new scaled point"""
        return Point3D(self.x * factor, self.y * factor, self.z * factor)


# Test
config = ImmutableConfig(host="api.example.com", port=443)
print(config.connection_string)  # https://api.example.com:443
print(config.config_hash)

# config.port = 8080  # FrozenInstanceError

# Frozen dataclass hash'lenebilir
configs = {
    config: "production",
    ImmutableConfig(host="localhost", port=8000, use_ssl=False): "development"
}

point = Point3D(3, 4, 5)
print(point.distance)  # 7.071...
new_point = point.translate(1, 1, 1)
print(new_point)  # Point3D(x=4, y=5, z=6, distance=8.774...)
```

---

## __slots__ Optimizasyonu

`__slots__`, instance attribute'larını sınırlayarak memory kullanımını optimize eder ve attribute access'i hızlandırır. `__dict__` yerine fixed-size array kullanır.

### Örnek 22: __slots__ Kullanımı ve Performans

```python
import sys
from typing import ClassVar

class WithoutSlots:
    """Normal class (with __dict__)"""

    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y


class WithSlots:
    """
    __slots__ kullanan class.
    Memory efficient, faster attribute access.
    """
    __slots__ = ('x', 'y')

    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y


# Memory comparison
obj_without = WithoutSlots(1, 2)
obj_with = WithSlots(1, 2)

print(f"Without __slots__: {sys.getsizeof(obj_without)} bytes")
print(f"With __slots__: {sys.getsizeof(obj_with)} bytes")

# __dict__ var mı?
print(f"Has __dict__: {hasattr(obj_without, '__dict__')}")  # True
print(f"Has __dict__: {hasattr(obj_with, '__dict__')}")  # False

# Dynamic attribute
obj_without.z = 3  # OK
# obj_with.z = 3  # AttributeError: 'WithSlots' object has no attribute 'z'


class OptimizedDataPoint:
    """
    Production-ready __slots__ kullanımı.
    Büyük veri yapıları için memory optimization.
    """
    __slots__ = ('_x', '_y', '_z', '_timestamp', '__weakref__')

    # Class variable (slots'ta değil)
    _instances_count: ClassVar[int] = 0

    def __init__(self, x: float, y: float, z: float):
        self._x = x
        self._y = y
        self._z = z
        self._timestamp = __import__('time').time()
        OptimizedDataPoint._instances_count += 1

    @property
    def x(self) -> float:
        return self._x

    @property
    def y(self) -> float:
        return self._y

    @property
    def z(self) -> float:
        return self._z

    @property
    def timestamp(self) -> float:
        return self._timestamp

    def __repr__(self):
        return f"DataPoint({self._x}, {self._y}, {self._z})"


# Performans testi
import time

def benchmark(cls, n=100000):
    """Benchmark class instantiation"""
    start = time.time()
    objects = [cls(i, i*2) for i in range(n)]
    end = time.time()

    # Memory usage
    memory = sum(sys.getsizeof(obj) for obj in objects)

    return end - start, memory / (1024 * 1024)  # time, MB


time_without, mem_without = benchmark(WithoutSlots)
time_with, mem_with = benchmark(WithSlots)

print(f"\nBenchmark (100k instances):")
print(f"Without __slots__: {time_without:.3f}s, {mem_without:.2f}MB")
print(f"With __slots__: {time_with:.3f}s, {mem_with:.2f}MB")
print(f"Speedup: {time_without/time_with:.2f}x")
print(f"Memory saved: {(1 - mem_with/mem_without)*100:.1f}%")
```

### Örnek 23: __slots__ ile Inheritance

```python
class Base:
    """Base class with __slots__"""
    __slots__ = ('base_attr',)

    def __init__(self, base_attr):
        self.base_attr = base_attr


class Derived(Base):
    """
    Derived class must declare its own __slots__.
    Sadece kendi attribute'larını ekler.
    """
    __slots__ = ('derived_attr',)

    def __init__(self, base_attr, derived_attr):
        super().__init__(base_attr)
        self.derived_attr = derived_attr


class MultipleInheritance:
    """
    Multiple inheritance ile __slots__ tricky olabilir.
    Sadece bir parent'ın non-empty __slots__'u olabilir.
    """
    __slots__ = ('multi_attr',)


# Test
derived = Derived("base", "derived")
print(derived.base_attr, derived.derived_attr)
# derived.new_attr = "test"  # AttributeError


class SlotsMixin:
    """Mixin with __slots__ = ()"""
    __slots__ = ()  # Empty slots - allows multiple inheritance

    def mixin_method(self):
        return "mixin"


class CombinedWithSlots(SlotsMixin):
    """Combining mixin with slots"""
    __slots__ = ('attr1', 'attr2')

    def __init__(self, attr1, attr2):
        self.attr1 = attr1
        self.attr2 = attr2


combined = CombinedWithSlots(1, 2)
print(combined.mixin_method())  # "mixin"
```

---

## Property Decorators Advanced

Property decorator'ları, attribute access'i kontrol etmek ve computed properties oluşturmak için kullanılır.

### Örnek 24: Advanced Property Patterns

```python
class LazyProperty:
    """
    Custom lazy property descriptor.
    İlk erişimde hesaplanır, sonra cache'lenir.
    """

    def __init__(self, func):
        self.func = func
        self.name = func.__name__

    def __get__(self, instance, owner):
        if instance is None:
            return self

        # Cache key
        cache_key = f'_lazy_{self.name}'

        # Eğer cache'de yoksa hesapla ve cache'le
        if not hasattr(instance, cache_key):
            value = self.func(instance)
            setattr(instance, cache_key, value)

        return getattr(instance, cache_key)


class CachedProperty:
    """
    TTL'li cached property.
    Belirli süre sonra yeniden hesaplanır.
    """

    def __init__(self, ttl=300):
        self.ttl = ttl
        self.func = None

    def __call__(self, func):
        self.func = func
        return self

    def __get__(self, instance, owner):
        if instance is None:
            return self

        import time

        cache_key = f'_cache_{self.func.__name__}'
        time_key = f'_time_{self.func.__name__}'

        # Check if cached and not expired
        if hasattr(instance, cache_key):
            cached_time = getattr(instance, time_key, 0)
            if time.time() - cached_time < self.ttl:
                return getattr(instance, cache_key)

        # Compute and cache
        value = self.func(instance)
        setattr(instance, cache_key, value)
        setattr(instance, time_key, time.time())

        return value


class Temperature:
    """
    Temperature sınıfı - property pattern'leri gösteriyor.
    """

    def __init__(self, celsius: float):
        self._celsius = celsius

    @property
    def celsius(self) -> float:
        """Celsius getter"""
        return self._celsius

    @celsius.setter
    def celsius(self, value: float) -> None:
        """Celsius setter with validation"""
        if value < -273.15:
            raise ValueError("Temperature below absolute zero!")
        self._celsius = value

    @property
    def fahrenheit(self) -> float:
        """Fahrenheit (computed property)"""
        return self._celsius * 9/5 + 32

    @fahrenheit.setter
    def fahrenheit(self, value: float) -> None:
        """Fahrenheit setter"""
        self.celsius = (value - 32) * 5/9

    @property
    def kelvin(self) -> float:
        """Kelvin (computed property)"""
        return self._celsius + 273.15

    @kelvin.setter
    def kelvin(self, value: float) -> None:
        """Kelvin setter"""
        self.celsius = value - 273.15


class DataProcessor:
    """Advanced property patterns"""

    def __init__(self, data: List[float]):
        self._data = data
        self._sorted_cache = None

    @property
    def data(self) -> List[float]:
        """Data getter"""
        return self._data

    @data.setter
    def data(self, value: List[float]) -> None:
        """Data setter - invalidate cache"""
        self._data = value
        self._sorted_cache = None  # Invalidate cache

    @LazyProperty
    def mean(self) -> float:
        """Lazy computed mean"""
        print("Computing mean...")
        return sum(self._data) / len(self._data)

    @CachedProperty(ttl=10)
    def variance(self) -> float:
        """Cached variance (10s TTL)"""
        print("Computing variance...")
        mean = self.mean
        return sum((x - mean) ** 2 for x in self._data) / len(self._data)

    @property
    def std_dev(self) -> float:
        """Standard deviation (always computed)"""
        return self.variance ** 0.5


# Test
temp = Temperature(25)
print(temp.celsius)  # 25
print(temp.fahrenheit)  # 77.0
temp.fahrenheit = 32
print(temp.celsius)  # 0.0

processor = DataProcessor([1, 2, 3, 4, 5])
print(processor.mean)  # Computing mean... 3.0
print(processor.mean)  # 3.0 (cached)
print(processor.variance)  # Computing variance... 2.0
```

### Örnek 25: Property Factory Pattern

```python
from typing import Callable, Any

def validated_property(
    validator: Callable[[Any], bool],
    error_message: str = "Invalid value"
):
    """
    Property factory - validator ile property oluşturur.
    """
    def decorator(func):
        name = func.__name__
        private_name = f'_{name}'

        def getter(self):
            return getattr(self, private_name, None)

        def setter(self, value):
            if not validator(value):
                raise ValueError(f"{name}: {error_message}")
            setattr(self, private_name, value)

        return property(getter, setter)

    return decorator


def typed_property(expected_type: type):
    """Type-checked property factory"""
    def decorator(func):
        name = func.__name__
        private_name = f'_{name}'

        def getter(self):
            return getattr(self, private_name, None)

        def setter(self, value):
            if not isinstance(value, expected_type):
                raise TypeError(
                    f"{name} must be {expected_type.__name__}, "
                    f"got {type(value).__name__}"
                )
            setattr(self, private_name, value)

        return property(getter, setter)

    return decorator


class Account:
    """Property factory patterns kullanarak account"""

    @typed_property(str)
    def username(self):
        """Type-checked username"""
        pass

    @validated_property(
        lambda x: isinstance(x, str) and '@' in x,
        "Must be a valid email"
    )
    def email(self):
        """Validated email"""
        pass

    @validated_property(
        lambda x: isinstance(x, (int, float)) and x >= 0,
        "Balance cannot be negative"
    )
    def balance(self):
        """Validated balance"""
        pass

    def __init__(self, username: str, email: str, balance: float = 0.0):
        self.username = username
        self.email = email
        self.balance = balance

    def deposit(self, amount: float):
        """Deposit money"""
        if amount <= 0:
            raise ValueError("Deposit amount must be positive")
        self.balance += amount

    def withdraw(self, amount: float):
        """Withdraw money"""
        if amount <= 0:
            raise ValueError("Withdrawal amount must be positive")
        if amount > self.balance:
            raise ValueError("Insufficient funds")
        self.balance -= amount


# Test
account = Account("john_doe", "john@example.com", 1000.0)
account.deposit(500)
print(account.balance)  # 1500.0

# account.balance = -100  # ValueError: Balance cannot be negative
# account.email = "invalid"  # ValueError: Must be a valid email
```

---

## Özet

Bu doküman, Python'da ileri düzey OOP konularını kapsamlı şekilde ele aldı:

1. **Abstract Base Classes**: Interface benzeri yapılar ve contract enforcement
2. **Multiple Inheritance ve MRO**: C3 linearization ve diamond problem çözümü
3. **Metaclasses**: Sınıf oluşturma sürecinin kontrolü ve customization
4. **Descriptors**: Attribute access kontrolü ve reusable property logic
5. **Advanced Magic Methods**: Operator overloading ve special behaviors
6. **Dataclasses**: Boilerplate reduction ve modern Python patterns
7. **__slots__**: Memory optimization ve performance improvements
8. **Property Decorators**: Computed properties ve advanced patterns

Her konu, production-ready örneklerle desteklendi. Bu pattern'ler, büyük ölçekli Python projelerinde, framework geliştirmede ve performans-kritik uygulamalarda yaygın olarak kullanılır.
