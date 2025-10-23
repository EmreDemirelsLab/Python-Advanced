# Design Patterns (Tasarım Kalıpları)

## İçindekiler
1. [Design Patterns Nedir?](#design-patterns-nedir)
2. [SOLID Prensipleri](#solid-prensipleri)
3. [Creational Patterns (Yaratımsal Kalıplar)](#creational-patterns)
4. [Structural Patterns (Yapısal Kalıplar)](#structural-patterns)
5. [Behavioral Patterns (Davranışsal Kalıplar)](#behavioral-patterns)
6. [Pythonic Patterns](#pythonic-patterns)

---

## Design Patterns Nedir?

**Design Patterns**, yazılım geliştirmede karşılaşılan yaygın problemlere yönelik test edilmiş, yeniden kullanılabilir çözümlerdir. Gang of Four (GoF) tarafından popülerleştirilen bu kalıplar, kodun bakımını kolaylaştırır ve genişletilebilirliği artırır.

**Avantajları:**
- Kanıtlanmış çözümler
- Geliştirici iletişimini kolaylaştırır
- Kod tekrarını azaltır
- Bakımı kolaylaştırır
- Esneklik ve genişletilebilirlik sağlar

---

## SOLID Prensipleri

SOLID, nesne yönelimli programlamada kod kalitesini artıran beş temel prensipten oluşur.

### 1. Single Responsibility Principle (SRP)
Bir sınıf sadece bir sorumluluğa sahip olmalıdır.

```python
# Kötü: Birden fazla sorumluluk
class User:
    def __init__(self, name):
        self.name = name

    def get_user(self):
        pass

    def save_to_db(self):
        # Veritabanı işlemi
        pass

    def send_email(self):
        # Email gönderme işlemi
        pass

# İyi: Her sınıf tek bir sorumluluğa sahip
class User:
    def __init__(self, name):
        self.name = name

class UserRepository:
    def save(self, user):
        # Veritabanı işlemi
        pass

class EmailService:
    def send_email(self, user, message):
        # Email gönderme işlemi
        pass
```

### 2. Open/Closed Principle (OCP)
Sınıflar genişlemeye açık, değişikliğe kapalı olmalıdır.

```python
from abc import ABC, abstractmethod

# Kötü: Her yeni ödeme türü için mevcut kodu değiştiriyoruz
class PaymentProcessor:
    def process(self, payment_type):
        if payment_type == "credit_card":
            # Kredi kartı işlemi
            pass
        elif payment_type == "paypal":
            # PayPal işlemi
            pass

# İyi: Yeni ödeme türleri için sadece yeni sınıf ekliyoruz
class PaymentMethod(ABC):
    @abstractmethod
    def process(self):
        pass

class CreditCardPayment(PaymentMethod):
    def process(self):
        print("Processing credit card payment")

class PayPalPayment(PaymentMethod):
    def process(self):
        print("Processing PayPal payment")

class CryptoPayment(PaymentMethod):
    def process(self):
        print("Processing crypto payment")

class PaymentProcessor:
    def process_payment(self, payment_method: PaymentMethod):
        payment_method.process()
```

### 3. Liskov Substitution Principle (LSP)
Alt sınıflar, üst sınıfların yerine kullanılabilmelidir.

```python
from abc import ABC, abstractmethod

# Kötü: Rectangle yerine Square kullanıldığında davranış bozuluyor
class Rectangle:
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def set_width(self, width):
        self.width = width

    def set_height(self, height):
        self.height = height

    def area(self):
        return self.width * self.height

class Square(Rectangle):
    def set_width(self, width):
        self.width = width
        self.height = width

    def set_height(self, height):
        self.width = height
        self.height = height

# İyi: Doğru soyutlama
class Shape(ABC):
    @abstractmethod
    def area(self):
        pass

class Rectangle(Shape):
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def area(self):
        return self.width * self.height

class Square(Shape):
    def __init__(self, side):
        self.side = side

    def area(self):
        return self.side ** 2
```

### 4. Interface Segregation Principle (ISP)
Sınıflar, kullanmadıkları metotları içeren interface'lere bağımlı olmamalıdır.

```python
from abc import ABC, abstractmethod

# Kötü: Tüm cihazlar tüm metotları uygulamak zorunda
class Device(ABC):
    @abstractmethod
    def print(self):
        pass

    @abstractmethod
    def scan(self):
        pass

    @abstractmethod
    def fax(self):
        pass

# İyi: Arayüzler küçük ve spesifik
class Printer(ABC):
    @abstractmethod
    def print(self):
        pass

class Scanner(ABC):
    @abstractmethod
    def scan(self):
        pass

class Fax(ABC):
    @abstractmethod
    def fax(self):
        pass

class SimplePrinter(Printer):
    def print(self):
        print("Printing...")

class MultiFunctionDevice(Printer, Scanner, Fax):
    def print(self):
        print("Printing...")

    def scan(self):
        print("Scanning...")

    def fax(self):
        print("Faxing...")
```

### 5. Dependency Inversion Principle (DIP)
Üst seviye modüller, alt seviye modüllere bağımlı olmamalıdır. Her ikisi de soyutlamalara bağımlı olmalıdır.

```python
from abc import ABC, abstractmethod

# Kötü: Üst seviye sınıf, alt seviye sınıfa doğrudan bağımlı
class MySQLDatabase:
    def connect(self):
        print("Connected to MySQL")

class UserService:
    def __init__(self):
        self.db = MySQLDatabase()  # Sıkı bağımlılık

# İyi: Her ikisi de soyutlamaya bağımlı
class Database(ABC):
    @abstractmethod
    def connect(self):
        pass

class MySQLDatabase(Database):
    def connect(self):
        print("Connected to MySQL")

class PostgreSQLDatabase(Database):
    def connect(self):
        print("Connected to PostgreSQL")

class UserService:
    def __init__(self, database: Database):
        self.db = database  # Gevşek bağımlılık

    def get_users(self):
        self.db.connect()
        return []

# Kullanım
mysql_db = MySQLDatabase()
user_service = UserService(mysql_db)

postgres_db = PostgreSQLDatabase()
user_service2 = UserService(postgres_db)
```

---

## Creational Patterns (Yaratımsal Kalıplar)

### 1. Singleton Pattern

**Amaç:** Bir sınıfın sadece bir örneğinin oluşturulmasını garanti eder.

**Kullanım Alanları:**
- Veritabanı bağlantıları
- Logger sınıfları
- Konfigürasyon yönetimi
- Cache yönetimi

```python
# Temel Singleton Implementation
class Singleton:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance

# Test
s1 = Singleton()
s2 = Singleton()
print(s1 is s2)  # True
```

**Thread-Safe Singleton:**

```python
import threading

class ThreadSafeSingleton:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            with cls._lock:
                # Double-checked locking
                if not cls._instance:
                    cls._instance = super().__new__(cls)
        return cls._instance

# Pythonic Singleton (Metaclass ile)
class SingletonMeta(type):
    _instances = {}
    _lock = threading.Lock()

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            with cls._lock:
                if cls not in cls._instances:
                    cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]

class DatabaseConnection(metaclass=SingletonMeta):
    def __init__(self):
        self.connection = None

    def connect(self, host, port):
        if not self.connection:
            self.connection = f"Connected to {host}:{port}"
            print(self.connection)
        return self.connection

# Test
db1 = DatabaseConnection()
db1.connect("localhost", 5432)

db2 = DatabaseConnection()
print(db1 is db2)  # True
```

**Real-World: Logger Singleton**

```python
import logging
from datetime import datetime

class Logger(metaclass=SingletonMeta):
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)

        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)

        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        ch.setFormatter(formatter)

        self.logger.addHandler(ch)

    def debug(self, message):
        self.logger.debug(message)

    def info(self, message):
        self.logger.info(message)

    def warning(self, message):
        self.logger.warning(message)

    def error(self, message):
        self.logger.error(message)

# Kullanım
logger1 = Logger()
logger1.info("Application started")

logger2 = Logger()
logger2.error("An error occurred")

print(logger1 is logger2)  # True
```

### 2. Factory Pattern

**Amaç:** Nesne oluşturma mantığını gizler ve alt sınıfların hangi sınıfın örneğinin oluşturulacağına karar vermesini sağlar.

**Simple Factory:**

```python
from abc import ABC, abstractmethod
from enum import Enum

class VehicleType(Enum):
    CAR = "car"
    TRUCK = "truck"
    MOTORCYCLE = "motorcycle"

class Vehicle(ABC):
    @abstractmethod
    def drive(self):
        pass

    @abstractmethod
    def get_specs(self):
        pass

class Car(Vehicle):
    def drive(self):
        return "Driving a car"

    def get_specs(self):
        return {"wheels": 4, "type": "Car", "capacity": 5}

class Truck(Vehicle):
    def drive(self):
        return "Driving a truck"

    def get_specs(self):
        return {"wheels": 6, "type": "Truck", "capacity": 2}

class Motorcycle(Vehicle):
    def drive(self):
        return "Riding a motorcycle"

    def get_specs(self):
        return {"wheels": 2, "type": "Motorcycle", "capacity": 2}

class VehicleFactory:
    @staticmethod
    def create_vehicle(vehicle_type: VehicleType) -> Vehicle:
        vehicles = {
            VehicleType.CAR: Car,
            VehicleType.TRUCK: Truck,
            VehicleType.MOTORCYCLE: Motorcycle
        }

        vehicle_class = vehicles.get(vehicle_type)
        if not vehicle_class:
            raise ValueError(f"Unknown vehicle type: {vehicle_type}")

        return vehicle_class()

# Kullanım
factory = VehicleFactory()
car = factory.create_vehicle(VehicleType.CAR)
print(car.drive())  # Driving a car
print(car.get_specs())  # {'wheels': 4, 'type': 'Car', 'capacity': 5}
```

**Factory Method Pattern:**

```python
from abc import ABC, abstractmethod

class Document(ABC):
    @abstractmethod
    def create(self):
        pass

    @abstractmethod
    def save(self):
        pass

class PDFDocument(Document):
    def create(self):
        return "PDF document created"

    def save(self):
        return "PDF document saved"

class WordDocument(Document):
    def create(self):
        return "Word document created"

    def save(self):
        return "Word document saved"

class ExcelDocument(Document):
    def create(self):
        return "Excel document created"

    def save(self):
        return "Excel document saved"

class DocumentCreator(ABC):
    @abstractmethod
    def factory_method(self) -> Document:
        pass

    def create_document(self):
        document = self.factory_method()
        print(document.create())
        print(document.save())
        return document

class PDFCreator(DocumentCreator):
    def factory_method(self) -> Document:
        return PDFDocument()

class WordCreator(DocumentCreator):
    def factory_method(self) -> Document:
        return WordDocument()

class ExcelCreator(DocumentCreator):
    def factory_method(self) -> Document:
        return ExcelDocument()

# Kullanım
pdf_creator = PDFCreator()
pdf_creator.create_document()

word_creator = WordCreator()
word_creator.create_document()
```

**Real-World: Database Connection Factory**

```python
from abc import ABC, abstractmethod
from typing import Dict, Any

class DatabaseConnection(ABC):
    @abstractmethod
    def connect(self) -> str:
        pass

    @abstractmethod
    def execute(self, query: str) -> Any:
        pass

    @abstractmethod
    def close(self) -> None:
        pass

class MySQLConnection(DatabaseConnection):
    def __init__(self, host: str, port: int, username: str, password: str):
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.connection = None

    def connect(self) -> str:
        self.connection = f"MySQL://{self.username}@{self.host}:{self.port}"
        return f"Connected to {self.connection}"

    def execute(self, query: str) -> Any:
        return f"Executing MySQL query: {query}"

    def close(self) -> None:
        self.connection = None
        print("MySQL connection closed")

class PostgreSQLConnection(DatabaseConnection):
    def __init__(self, host: str, port: int, username: str, password: str):
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.connection = None

    def connect(self) -> str:
        self.connection = f"PostgreSQL://{self.username}@{self.host}:{self.port}"
        return f"Connected to {self.connection}"

    def execute(self, query: str) -> Any:
        return f"Executing PostgreSQL query: {query}"

    def close(self) -> None:
        self.connection = None
        print("PostgreSQL connection closed")

class MongoDBConnection(DatabaseConnection):
    def __init__(self, host: str, port: int, username: str, password: str):
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.connection = None

    def connect(self) -> str:
        self.connection = f"MongoDB://{self.username}@{self.host}:{self.port}"
        return f"Connected to {self.connection}"

    def execute(self, query: str) -> Any:
        return f"Executing MongoDB query: {query}"

    def close(self) -> None:
        self.connection = None
        print("MongoDB connection closed")

class DatabaseFactory:
    @staticmethod
    def create_connection(db_type: str, config: Dict[str, Any]) -> DatabaseConnection:
        databases = {
            "mysql": MySQLConnection,
            "postgresql": PostgreSQLConnection,
            "mongodb": MongoDBConnection
        }

        db_class = databases.get(db_type.lower())
        if not db_class:
            raise ValueError(f"Unsupported database type: {db_type}")

        return db_class(**config)

# Kullanım
config = {
    "host": "localhost",
    "port": 5432,
    "username": "admin",
    "password": "secret"
}

db = DatabaseFactory.create_connection("postgresql", config)
print(db.connect())
print(db.execute("SELECT * FROM users"))
db.close()
```

### 3. Builder Pattern

**Amaç:** Karmaşık nesnelerin adım adım oluşturulmasını sağlar. Aynı yapım sürecini kullanarak farklı temsiller oluşturabilir.

**Klasik Builder Pattern:**

```python
from typing import Optional, List

class Computer:
    def __init__(self):
        self.cpu: Optional[str] = None
        self.ram: Optional[str] = None
        self.storage: Optional[str] = None
        self.gpu: Optional[str] = None
        self.motherboard: Optional[str] = None
        self.power_supply: Optional[str] = None
        self.case: Optional[str] = None
        self.cooling: Optional[str] = None

    def __str__(self):
        specs = []
        specs.append(f"CPU: {self.cpu}")
        specs.append(f"RAM: {self.ram}")
        specs.append(f"Storage: {self.storage}")
        specs.append(f"GPU: {self.gpu}")
        specs.append(f"Motherboard: {self.motherboard}")
        specs.append(f"Power Supply: {self.power_supply}")
        specs.append(f"Case: {self.case}")
        specs.append(f"Cooling: {self.cooling}")
        return "\n".join(specs)

class ComputerBuilder:
    def __init__(self):
        self.computer = Computer()

    def set_cpu(self, cpu: str):
        self.computer.cpu = cpu
        return self

    def set_ram(self, ram: str):
        self.computer.ram = ram
        return self

    def set_storage(self, storage: str):
        self.computer.storage = storage
        return self

    def set_gpu(self, gpu: str):
        self.computer.gpu = gpu
        return self

    def set_motherboard(self, motherboard: str):
        self.computer.motherboard = motherboard
        return self

    def set_power_supply(self, power_supply: str):
        self.computer.power_supply = power_supply
        return self

    def set_case(self, case: str):
        self.computer.case = case
        return self

    def set_cooling(self, cooling: str):
        self.computer.cooling = cooling
        return self

    def build(self) -> Computer:
        return self.computer

# Kullanım - Method Chaining
gaming_pc = (ComputerBuilder()
    .set_cpu("Intel i9-13900K")
    .set_ram("32GB DDR5")
    .set_storage("2TB NVMe SSD")
    .set_gpu("NVIDIA RTX 4090")
    .set_motherboard("ASUS ROG Maximus")
    .set_power_supply("1000W 80+ Gold")
    .set_case("Lian Li O11 Dynamic")
    .set_cooling("Custom Water Cooling")
    .build())

print("Gaming PC:")
print(gaming_pc)
print()

office_pc = (ComputerBuilder()
    .set_cpu("Intel i5-12400")
    .set_ram("16GB DDR4")
    .set_storage("512GB SSD")
    .set_motherboard("MSI B660M")
    .set_power_supply("500W 80+ Bronze")
    .set_case("Fractal Design Define")
    .build())

print("Office PC:")
print(office_pc)
```

**Director Pattern ile Builder:**

```python
class ComputerDirector:
    def __init__(self, builder: ComputerBuilder):
        self.builder = builder

    def build_gaming_pc(self) -> Computer:
        return (self.builder
            .set_cpu("AMD Ryzen 9 7950X")
            .set_ram("64GB DDR5")
            .set_storage("4TB NVMe SSD")
            .set_gpu("NVIDIA RTX 4090")
            .set_motherboard("ASUS ROG Crosshair")
            .set_power_supply("1200W 80+ Platinum")
            .set_case("Corsair 5000D")
            .set_cooling("Custom Water Cooling")
            .build())

    def build_workstation(self) -> Computer:
        return (self.builder
            .set_cpu("AMD Threadripper 5995WX")
            .set_ram("256GB DDR4 ECC")
            .set_storage("8TB NVMe SSD RAID")
            .set_gpu("NVIDIA RTX A6000")
            .set_motherboard("ASUS Pro WS WRX80E-SAGE")
            .set_power_supply("1600W 80+ Titanium")
            .set_case("Fractal Design Define 7 XL")
            .set_cooling("Custom Water Cooling")
            .build())

    def build_budget_pc(self) -> Computer:
        return (self.builder
            .set_cpu("AMD Ryzen 5 5600")
            .set_ram("16GB DDR4")
            .set_storage("512GB SSD")
            .set_gpu("AMD RX 6600")
            .set_motherboard("MSI B550M")
            .set_power_supply("550W 80+ Bronze")
            .set_case("NZXT H510")
            .build())

# Kullanım
director = ComputerDirector(ComputerBuilder())

gaming_pc = director.build_gaming_pc()
print("Gaming PC:")
print(gaming_pc)
```

**Real-World: SQL Query Builder**

```python
from typing import List, Optional, Dict, Any

class SQLQuery:
    def __init__(self):
        self.select_fields: List[str] = []
        self.from_table: Optional[str] = None
        self.joins: List[Dict[str, str]] = []
        self.where_conditions: List[str] = []
        self.group_by_fields: List[str] = []
        self.having_conditions: List[str] = []
        self.order_by_fields: List[Dict[str, str]] = []
        self.limit_value: Optional[int] = None
        self.offset_value: Optional[int] = None

    def __str__(self):
        query_parts = []

        # SELECT
        if self.select_fields:
            query_parts.append(f"SELECT {', '.join(self.select_fields)}")
        else:
            query_parts.append("SELECT *")

        # FROM
        if self.from_table:
            query_parts.append(f"FROM {self.from_table}")

        # JOINS
        for join in self.joins:
            query_parts.append(f"{join['type']} {join['table']} ON {join['condition']}")

        # WHERE
        if self.where_conditions:
            query_parts.append(f"WHERE {' AND '.join(self.where_conditions)}")

        # GROUP BY
        if self.group_by_fields:
            query_parts.append(f"GROUP BY {', '.join(self.group_by_fields)}")

        # HAVING
        if self.having_conditions:
            query_parts.append(f"HAVING {' AND '.join(self.having_conditions)}")

        # ORDER BY
        if self.order_by_fields:
            order_clauses = [f"{field['name']} {field['direction']}"
                           for field in self.order_by_fields]
            query_parts.append(f"ORDER BY {', '.join(order_clauses)}")

        # LIMIT
        if self.limit_value:
            query_parts.append(f"LIMIT {self.limit_value}")

        # OFFSET
        if self.offset_value:
            query_parts.append(f"OFFSET {self.offset_value}")

        return "\n".join(query_parts)

class SQLQueryBuilder:
    def __init__(self):
        self.query = SQLQuery()

    def select(self, *fields: str):
        self.query.select_fields.extend(fields)
        return self

    def from_table(self, table: str):
        self.query.from_table = table
        return self

    def inner_join(self, table: str, condition: str):
        self.query.joins.append({
            "type": "INNER JOIN",
            "table": table,
            "condition": condition
        })
        return self

    def left_join(self, table: str, condition: str):
        self.query.joins.append({
            "type": "LEFT JOIN",
            "table": table,
            "condition": condition
        })
        return self

    def where(self, condition: str):
        self.query.where_conditions.append(condition)
        return self

    def group_by(self, *fields: str):
        self.query.group_by_fields.extend(fields)
        return self

    def having(self, condition: str):
        self.query.having_conditions.append(condition)
        return self

    def order_by(self, field: str, direction: str = "ASC"):
        self.query.order_by_fields.append({
            "name": field,
            "direction": direction
        })
        return self

    def limit(self, limit: int):
        self.query.limit_value = limit
        return self

    def offset(self, offset: int):
        self.query.offset_value = offset
        return self

    def build(self) -> SQLQuery:
        return self.query

    def reset(self):
        self.query = SQLQuery()
        return self

# Kullanım
query = (SQLQueryBuilder()
    .select("users.id", "users.name", "COUNT(orders.id) as order_count")
    .from_table("users")
    .left_join("orders", "users.id = orders.user_id")
    .where("users.active = 1")
    .where("users.created_at > '2023-01-01'")
    .group_by("users.id", "users.name")
    .having("COUNT(orders.id) > 5")
    .order_by("order_count", "DESC")
    .limit(10)
    .build())

print(query)
```

### 4. Prototype Pattern

**Amaç:** Mevcut nesneleri kopyalayarak yeni nesneler oluşturur. Nesne oluşturma maliyeti yüksek olduğunda kullanılır.

```python
import copy
from typing import List, Dict, Any

class Prototype:
    def clone(self):
        """Shallow copy döndürür"""
        return copy.copy(self)

    def deep_clone(self):
        """Deep copy döndürür"""
        return copy.deepcopy(self)

class Document(Prototype):
    def __init__(self, title: str, content: str, metadata: Dict[str, Any]):
        self.title = title
        self.content = content
        self.metadata = metadata
        self.created_at = None

    def __str__(self):
        return f"Document(title='{self.title}', content='{self.content[:30]}...', metadata={self.metadata})"

# Kullanım
original_doc = Document(
    title="Original Document",
    content="This is the original content of the document.",
    metadata={"author": "John Doe", "version": 1, "tags": ["python", "design-patterns"]}
)

# Shallow copy
shallow_copy = original_doc.clone()
shallow_copy.title = "Shallow Copy Document"
shallow_copy.metadata["version"] = 2  # Bu orijinali de etkiler (shallow copy)

print("Original:", original_doc)
print("Shallow Copy:", shallow_copy)
print()

# Deep copy
deep_copy = original_doc.deep_clone()
deep_copy.title = "Deep Copy Document"
deep_copy.metadata["version"] = 3  # Bu orijinali etkilemez (deep copy)
deep_copy.metadata["tags"].append("prototype")

print("Original:", original_doc)
print("Deep Copy:", deep_copy)
```

**Real-World: Game Character Prototype**

```python
import copy
from dataclasses import dataclass, field
from typing import List, Dict

@dataclass
class Stats:
    health: int = 100
    mana: int = 100
    strength: int = 10
    agility: int = 10
    intelligence: int = 10

@dataclass
class Equipment:
    weapon: str = "None"
    armor: str = "None"
    accessories: List[str] = field(default_factory=list)

class Character(Prototype):
    def __init__(self, name: str, char_class: str, stats: Stats, equipment: Equipment):
        self.name = name
        self.char_class = char_class
        self.stats = stats
        self.equipment = equipment
        self.skills: List[str] = []
        self.level: int = 1

    def __str__(self):
        return (f"Character(name='{self.name}', class='{self.char_class}', "
                f"level={self.level}, stats={self.stats}, equipment={self.equipment})")

# Prototip karakterler oluştur
warrior_prototype = Character(
    name="Warrior",
    char_class="Warrior",
    stats=Stats(health=150, mana=50, strength=20, agility=10, intelligence=5),
    equipment=Equipment(weapon="Sword", armor="Heavy Armor", accessories=["Shield"])
)
warrior_prototype.skills = ["Slash", "Block", "War Cry"]

mage_prototype = Character(
    name="Mage",
    char_class="Mage",
    stats=Stats(health=80, mana=200, strength=5, agility=8, intelligence=25),
    equipment=Equipment(weapon="Staff", armor="Robe", accessories=["Magic Ring"])
)
mage_prototype.skills = ["Fireball", "Ice Blast", "Teleport"]

rogue_prototype = Character(
    name="Rogue",
    char_class="Rogue",
    stats=Stats(health=100, mana=80, strength=12, agility=25, intelligence=10),
    equipment=Equipment(weapon="Daggers", armor="Light Armor", accessories=["Cloak"])
)
rogue_prototype.skills = ["Backstab", "Stealth", "Poison"]

# Prototype Registry
class CharacterRegistry:
    def __init__(self):
        self._prototypes: Dict[str, Character] = {}

    def register(self, key: str, prototype: Character):
        self._prototypes[key] = prototype

    def unregister(self, key: str):
        del self._prototypes[key]

    def create(self, key: str, name: str) -> Character:
        prototype = self._prototypes.get(key)
        if not prototype:
            raise ValueError(f"Prototype not found: {key}")

        # Deep clone yaparak yeni karakter oluştur
        character = prototype.deep_clone()
        character.name = name
        return character

# Registry'yi doldur
registry = CharacterRegistry()
registry.register("warrior", warrior_prototype)
registry.register("mage", mage_prototype)
registry.register("rogue", rogue_prototype)

# Yeni karakterler oluştur
player1 = registry.create("warrior", "Conan")
player1.level = 5
player1.stats.strength = 25

player2 = registry.create("mage", "Gandalf")
player2.level = 10
player2.stats.intelligence = 30

player3 = registry.create("rogue", "Ezio")
player3.level = 7
player3.equipment.accessories.append("Lockpick")

print(player1)
print(player2)
print(player3)
print("\nOriginal prototypes remain unchanged:")
print(warrior_prototype)
```

---

## Structural Patterns (Yapısal Kalıplar)

### 5. Adapter Pattern

**Amaç:** Uyumsuz arayüzleri birlikte çalışabilir hale getirir. Mevcut bir sınıfın arayüzünü, istemcinin beklediği arayüze dönüştürür.

```python
from abc import ABC, abstractmethod

# Target Interface - İstemcinin beklediği arayüz
class MediaPlayer(ABC):
    @abstractmethod
    def play(self, filename: str):
        pass

# Adaptee - Uyarlanması gereken sınıf
class VLCPlayer:
    def play_vlc(self, filename: str):
        print(f"Playing {filename} with VLC Player")

class WindowsMediaPlayer:
    def play_wmp(self, filename: str):
        print(f"Playing {filename} with Windows Media Player")

# Adapter
class VLCAdapter(MediaPlayer):
    def __init__(self, vlc_player: VLCPlayer):
        self.vlc_player = vlc_player

    def play(self, filename: str):
        self.vlc_player.play_vlc(filename)

class WindowsMediaAdapter(MediaPlayer):
    def __init__(self, wmp: WindowsMediaPlayer):
        self.wmp = wmp

    def play(self, filename: str):
        self.wmp.play_wmp(filename)

# Client
class AudioPlayer:
    def __init__(self, player: MediaPlayer):
        self.player = player

    def play_audio(self, filename: str):
        self.player.play(filename)

# Kullanım
vlc = VLCPlayer()
vlc_adapter = VLCAdapter(vlc)
audio_player1 = AudioPlayer(vlc_adapter)
audio_player1.play_audio("song.mp3")

wmp = WindowsMediaPlayer()
wmp_adapter = WindowsMediaAdapter(wmp)
audio_player2 = AudioPlayer(wmp_adapter)
audio_player2.play_audio("song.mp3")
```

**Real-World: Payment Gateway Adapter**

```python
from abc import ABC, abstractmethod
from typing import Dict, Any

# Target Interface
class PaymentGateway(ABC):
    @abstractmethod
    def process_payment(self, amount: float, currency: str, customer_data: Dict[str, Any]) -> bool:
        pass

    @abstractmethod
    def refund_payment(self, transaction_id: str, amount: float) -> bool:
        pass

# Adaptee 1 - Stripe API
class StripeAPI:
    def charge(self, amount_cents: int, currency: str, token: str, metadata: Dict):
        print(f"Stripe: Charging {amount_cents} cents in {currency}")
        return {"id": "stripe_12345", "status": "succeeded"}

    def create_refund(self, charge_id: str, amount_cents: int):
        print(f"Stripe: Refunding {amount_cents} cents for charge {charge_id}")
        return {"id": "refund_12345", "status": "succeeded"}

# Adaptee 2 - PayPal API
class PayPalAPI:
    def execute_payment(self, total: str, currency_code: str, payer_info: Dict):
        print(f"PayPal: Executing payment of {total} {currency_code}")
        return {"paymentId": "paypal_67890", "state": "approved"}

    def refund_sale(self, sale_id: str, refund_amount: Dict):
        print(f"PayPal: Refunding sale {sale_id}")
        return {"refundId": "refund_67890", "state": "completed"}

# Adapter 1 - Stripe
class StripeAdapter(PaymentGateway):
    def __init__(self):
        self.stripe = StripeAPI()

    def process_payment(self, amount: float, currency: str, customer_data: Dict[str, Any]) -> bool:
        amount_cents = int(amount * 100)
        token = customer_data.get("token", "tok_visa")
        metadata = {
            "customer_email": customer_data.get("email"),
            "customer_name": customer_data.get("name")
        }

        result = self.stripe.charge(amount_cents, currency, token, metadata)
        return result["status"] == "succeeded"

    def refund_payment(self, transaction_id: str, amount: float) -> bool:
        amount_cents = int(amount * 100)
        result = self.stripe.create_refund(transaction_id, amount_cents)
        return result["status"] == "succeeded"

# Adapter 2 - PayPal
class PayPalAdapter(PaymentGateway):
    def __init__(self):
        self.paypal = PayPalAPI()

    def process_payment(self, amount: float, currency: str, customer_data: Dict[str, Any]) -> bool:
        payer_info = {
            "email": customer_data.get("email"),
            "first_name": customer_data.get("name", "").split()[0],
            "last_name": " ".join(customer_data.get("name", "").split()[1:])
        }

        result = self.paypal.execute_payment(str(amount), currency, payer_info)
        return result["state"] == "approved"

    def refund_payment(self, transaction_id: str, amount: float) -> bool:
        refund_amount = {"total": str(amount), "currency": "USD"}
        result = self.paypal.refund_sale(transaction_id, refund_amount)
        return result["state"] == "completed"

# Client - E-commerce System
class PaymentProcessor:
    def __init__(self, gateway: PaymentGateway):
        self.gateway = gateway

    def charge_customer(self, amount: float, currency: str, customer_data: Dict[str, Any]):
        print(f"\nProcessing payment of {amount} {currency}")
        success = self.gateway.process_payment(amount, currency, customer_data)
        if success:
            print("Payment successful!")
        else:
            print("Payment failed!")
        return success

    def process_refund(self, transaction_id: str, amount: float):
        print(f"\nProcessing refund of {amount}")
        success = self.gateway.refund_payment(transaction_id, amount)
        if success:
            print("Refund successful!")
        else:
            print("Refund failed!")
        return success

# Kullanım
customer = {
    "name": "John Doe",
    "email": "john@example.com",
    "token": "tok_visa"
}

# Stripe ile ödeme
stripe_gateway = StripeAdapter()
processor = PaymentProcessor(stripe_gateway)
processor.charge_customer(99.99, "USD", customer)

# PayPal ile ödeme
paypal_gateway = PayPalAdapter()
processor = PaymentProcessor(paypal_gateway)
processor.charge_customer(149.99, "USD", customer)
```

### 6. Decorator Pattern

**Amaç:** Nesnelere dinamik olarak yeni sorumluluklar ekler. Alt sınıf oluşturmaya alternatif esnek bir yöntemdir.

```python
from abc import ABC, abstractmethod

# Component
class Coffee(ABC):
    @abstractmethod
    def cost(self) -> float:
        pass

    @abstractmethod
    def description(self) -> str:
        pass

# Concrete Component
class SimpleCoffee(Coffee):
    def cost(self) -> float:
        return 5.0

    def description(self) -> str:
        return "Simple Coffee"

# Decorator Base
class CoffeeDecorator(Coffee):
    def __init__(self, coffee: Coffee):
        self._coffee = coffee

    def cost(self) -> float:
        return self._coffee.cost()

    def description(self) -> str:
        return self._coffee.description()

# Concrete Decorators
class Milk(CoffeeDecorator):
    def cost(self) -> float:
        return self._coffee.cost() + 1.5

    def description(self) -> str:
        return self._coffee.description() + ", Milk"

class Sugar(CoffeeDecorator):
    def cost(self) -> float:
        return self._coffee.cost() + 0.5

    def description(self) -> str:
        return self._coffee.description() + ", Sugar"

class WhippedCream(CoffeeDecorator):
    def cost(self) -> float:
        return self._coffee.cost() + 2.0

    def description(self) -> str:
        return self._coffee.description() + ", Whipped Cream"

class Caramel(CoffeeDecorator):
    def cost(self) -> float:
        return self._coffee.cost() + 1.8

    def description(self) -> str:
        return self._coffee.description() + ", Caramel"

# Kullanım
coffee = SimpleCoffee()
print(f"{coffee.description()}: ${coffee.cost()}")

# Süt ekle
coffee = Milk(coffee)
print(f"{coffee.description()}: ${coffee.cost()}")

# Şeker ekle
coffee = Sugar(coffee)
print(f"{coffee.description()}: ${coffee.cost()}")

# Krema ekle
coffee = WhippedCream(coffee)
print(f"{coffee.description()}: ${coffee.cost()}")

# Yeni kahve - Tek seferde tüm dekoratörleri ekle
fancy_coffee = WhippedCream(Caramel(Milk(SimpleCoffee())))
print(f"\n{fancy_coffee.description()}: ${fancy_coffee.cost()}")
```

**Python Decorators ile Function Decorator:**

```python
import time
import functools
from typing import Callable, Any

def timer(func: Callable) -> Callable:
    """Fonksiyonun çalışma süresini ölçer"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} took {end - start:.4f} seconds")
        return result
    return wrapper

def cache(func: Callable) -> Callable:
    """Fonksiyon sonuçlarını önbelleğe alır"""
    cached_results = {}

    @functools.wraps(func)
    def wrapper(*args):
        if args in cached_results:
            print(f"Returning cached result for {args}")
            return cached_results[args]

        result = func(*args)
        cached_results[args] = result
        return result
    return wrapper

def logger(func: Callable) -> Callable:
    """Fonksiyon çağrılarını loglar"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        print(f"Calling {func.__name__} with args={args}, kwargs={kwargs}")
        result = func(*args, **kwargs)
        print(f"{func.__name__} returned {result}")
        return result
    return wrapper

# Kullanım - Çoklu decorator
@timer
@cache
@logger
def fibonacci(n: int) -> int:
    if n < 2:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

print(fibonacci(10))
print(fibonacci(10))  # Cached result
```

**Real-World: Text Processing Pipeline**

```python
from abc import ABC, abstractmethod
from typing import List

class TextProcessor(ABC):
    @abstractmethod
    def process(self, text: str) -> str:
        pass

class PlainText(TextProcessor):
    def process(self, text: str) -> str:
        return text

class TextDecorator(TextProcessor):
    def __init__(self, processor: TextProcessor):
        self._processor = processor

    def process(self, text: str) -> str:
        return self._processor.process(text)

class UpperCaseDecorator(TextDecorator):
    def process(self, text: str) -> str:
        return self._processor.process(text).upper()

class TrimDecorator(TextDecorator):
    def process(self, text: str) -> str:
        return self._processor.process(text).strip()

class RemoveSpacesDecorator(TextDecorator):
    def process(self, text: str) -> str:
        text = self._processor.process(text)
        return " ".join(text.split())

class HTMLTagDecorator(TextDecorator):
    def __init__(self, processor: TextProcessor, tag: str):
        super().__init__(processor)
        self.tag = tag

    def process(self, text: str) -> str:
        text = self._processor.process(text)
        return f"<{self.tag}>{text}</{self.tag}>"

class MarkdownBoldDecorator(TextDecorator):
    def process(self, text: str) -> str:
        return f"**{self._processor.process(text)}**"

class MarkdownItalicDecorator(TextDecorator):
    def process(self, text: str) -> str:
        return f"*{self._processor.process(text)}*"

# Kullanım
text = "  hello world  with   extra   spaces  "

# Pipeline 1: Temizle ve büyük harfe çevir
processor1 = UpperCaseDecorator(RemoveSpacesDecorator(TrimDecorator(PlainText())))
print("Pipeline 1:", processor1.process(text))

# Pipeline 2: HTML formatla
processor2 = HTMLTagDecorator(
    HTMLTagDecorator(
        TrimDecorator(PlainText()),
        "strong"
    ),
    "p"
)
print("Pipeline 2:", processor2.process(text))

# Pipeline 3: Markdown formatla
processor3 = MarkdownBoldDecorator(
    MarkdownItalicDecorator(
        RemoveSpacesDecorator(
            TrimDecorator(PlainText())
        )
    )
)
print("Pipeline 3:", processor3.process(text))
```

### 7. Facade Pattern

**Amaç:** Karmaşık bir alt sisteme basit bir arayüz sağlar. Alt sistemin kullanımını kolaylaştırır.

```python
# Karmaşık alt sistem sınıfları
class CPU:
    def freeze(self):
        print("CPU: Freezing...")

    def jump(self, position):
        print(f"CPU: Jumping to position {position}")

    def execute(self):
        print("CPU: Executing...")

class Memory:
    def load(self, position, data):
        print(f"Memory: Loading data '{data}' to position {position}")

class HardDrive:
    def read(self, lba, size):
        print(f"HardDrive: Reading {size} bytes from LBA {lba}")
        return f"Data from sector {lba}"

class GPU:
    def render(self):
        print("GPU: Rendering graphics...")

# Facade
class ComputerFacade:
    def __init__(self):
        self.cpu = CPU()
        self.memory = Memory()
        self.hard_drive = HardDrive()
        self.gpu = GPU()

    def start(self):
        """Bilgisayarı başlatmak için basit bir arayüz"""
        print("Starting computer...")
        self.cpu.freeze()
        boot_data = self.hard_drive.read(0, 1024)
        self.memory.load(0, boot_data)
        self.cpu.jump(0)
        self.cpu.execute()
        self.gpu.render()
        print("Computer started successfully!\n")

# Kullanım - Karmaşık işlemler basit hale geldi
computer = ComputerFacade()
computer.start()
```

**Real-World: E-commerce Order Facade**

```python
from typing import List, Dict, Any
from dataclasses import dataclass
from datetime import datetime

@dataclass
class Product:
    id: int
    name: str
    price: float
    stock: int

@dataclass
class Customer:
    id: int
    name: str
    email: str
    address: str

# Alt sistem sınıfları
class InventorySystem:
    def __init__(self):
        self.products = {
            1: Product(1, "Laptop", 999.99, 10),
            2: Product(2, "Mouse", 29.99, 50),
            3: Product(3, "Keyboard", 79.99, 30)
        }

    def check_availability(self, product_id: int, quantity: int) -> bool:
        product = self.products.get(product_id)
        if product and product.stock >= quantity:
            print(f"Inventory: {product.name} is available (stock: {product.stock})")
            return True
        print(f"Inventory: Product {product_id} is not available")
        return False

    def reserve_items(self, product_id: int, quantity: int) -> bool:
        product = self.products.get(product_id)
        if product and product.stock >= quantity:
            product.stock -= quantity
            print(f"Inventory: Reserved {quantity}x {product.name}")
            return True
        return False

    def get_price(self, product_id: int) -> float:
        product = self.products.get(product_id)
        return product.price if product else 0.0

class PaymentSystem:
    def validate_card(self, card_number: str) -> bool:
        print(f"Payment: Validating card {card_number[-4:]}")
        return len(card_number) == 16

    def charge(self, card_number: str, amount: float) -> str:
        print(f"Payment: Charging ${amount:.2f} to card {card_number[-4:]}")
        return f"TXN_{datetime.now().timestamp()}"

    def refund(self, transaction_id: str, amount: float) -> bool:
        print(f"Payment: Refunding ${amount:.2f} for transaction {transaction_id}")
        return True

class ShippingSystem:
    def calculate_shipping(self, address: str, weight: float) -> float:
        print(f"Shipping: Calculating shipping to {address}")
        return 10.0 if weight < 5 else 20.0

    def create_shipment(self, address: str, items: List[Dict]) -> str:
        print(f"Shipping: Creating shipment to {address}")
        return f"SHIP_{datetime.now().timestamp()}"

    def track_shipment(self, tracking_number: str) -> str:
        return f"Shipment {tracking_number} is in transit"

class NotificationSystem:
    def send_email(self, email: str, subject: str, body: str):
        print(f"Notification: Sending email to {email}")
        print(f"  Subject: {subject}")
        print(f"  Body: {body[:50]}...")

    def send_sms(self, phone: str, message: str):
        print(f"Notification: Sending SMS to {phone}: {message}")

class OrderDatabase:
    def __init__(self):
        self.orders = {}
        self.order_counter = 1000

    def save_order(self, order_data: Dict[str, Any]) -> int:
        order_id = self.order_counter
        self.order_counter += 1
        self.orders[order_id] = order_data
        print(f"Database: Order #{order_id} saved")
        return order_id

    def get_order(self, order_id: int) -> Dict[str, Any]:
        return self.orders.get(order_id, {})

# Facade - Tüm alt sistemleri koordine eder
class OrderFacade:
    def __init__(self):
        self.inventory = InventorySystem()
        self.payment = PaymentSystem()
        self.shipping = ShippingSystem()
        self.notification = NotificationSystem()
        self.database = OrderDatabase()

    def place_order(self, customer: Customer, items: List[Dict[str, Any]],
                   card_number: str) -> Dict[str, Any]:
        """
        Sipariş verme işlemini basitleştirir
        items: [{"product_id": 1, "quantity": 2}, ...]
        """
        print(f"\n{'='*60}")
        print(f"Processing order for {customer.name}")
        print(f"{'='*60}\n")

        # 1. Stok kontrolü
        for item in items:
            if not self.inventory.check_availability(item["product_id"], item["quantity"]):
                return {"success": False, "error": "Item not available"}

        # 2. Toplam tutarı hesapla
        total_amount = sum(
            self.inventory.get_price(item["product_id"]) * item["quantity"]
            for item in items
        )
        shipping_cost = self.shipping.calculate_shipping(customer.address, 2.0)
        total_amount += shipping_cost

        print(f"\nOrder Summary:")
        print(f"  Subtotal: ${total_amount - shipping_cost:.2f}")
        print(f"  Shipping: ${shipping_cost:.2f}")
        print(f"  Total: ${total_amount:.2f}\n")

        # 3. Ödeme işlemi
        if not self.payment.validate_card(card_number):
            return {"success": False, "error": "Invalid card"}

        transaction_id = self.payment.charge(card_number, total_amount)

        # 4. Stok rezervasyonu
        for item in items:
            self.inventory.reserve_items(item["product_id"], item["quantity"])

        # 5. Kargo oluştur
        tracking_number = self.shipping.create_shipment(customer.address, items)

        # 6. Siparişi kaydet
        order_data = {
            "customer": customer,
            "items": items,
            "total": total_amount,
            "transaction_id": transaction_id,
            "tracking_number": tracking_number,
            "status": "confirmed",
            "created_at": datetime.now()
        }
        order_id = self.database.save_order(order_data)

        # 7. Bildirim gönder
        self.notification.send_email(
            customer.email,
            f"Order #{order_id} Confirmed",
            f"Your order has been confirmed. Total: ${total_amount:.2f}\n"
            f"Tracking number: {tracking_number}"
        )

        print(f"\n{'='*60}")
        print(f"Order #{order_id} completed successfully!")
        print(f"{'='*60}\n")

        return {
            "success": True,
            "order_id": order_id,
            "transaction_id": transaction_id,
            "tracking_number": tracking_number,
            "total": total_amount
        }

    def cancel_order(self, order_id: int) -> bool:
        """Sipariş iptal etme işlemini basitleştirir"""
        print(f"\nCancelling order #{order_id}...")

        order = self.database.get_order(order_id)
        if not order:
            return False

        # Ödeme iadesi
        self.payment.refund(order["transaction_id"], order["total"])

        # Bildirim gönder
        self.notification.send_email(
            order["customer"].email,
            f"Order #{order_id} Cancelled",
            "Your order has been cancelled and refunded."
        )

        print(f"Order #{order_id} cancelled successfully!\n")
        return True

# Kullanım
customer = Customer(1, "John Doe", "john@example.com", "123 Main St, NYC")

items = [
    {"product_id": 1, "quantity": 1},
    {"product_id": 2, "quantity": 2}
]

# Facade sayesinde tüm karmaşık işlemler tek bir metodla yapılıyor
order_facade = OrderFacade()
result = order_facade.place_order(customer, items, "1234567890123456")

if result["success"]:
    print(f"Order ID: {result['order_id']}")
    print(f"Tracking: {result['tracking_number']}")
```

### 8. Proxy Pattern

**Amaç:** Bir nesneye erişimi kontrol etmek için vekil (proxy) nesne kullanır. Asıl nesneye erişimi kontrol eder, geciktirir veya optimize eder.

**Proxy Türleri:**
1. **Virtual Proxy:** Ağır nesnelerin lazy initialization'ı
2. **Protection Proxy:** Erişim kontrolü
3. **Remote Proxy:** Uzak nesnelere erişim
4. **Cache Proxy:** Sonuçları önbelleğe alma

```python
from abc import ABC, abstractmethod
import time

# Subject Interface
class Image(ABC):
    @abstractmethod
    def display(self):
        pass

# Real Subject - Ağır nesne
class RealImage(Image):
    def __init__(self, filename: str):
        self.filename = filename
        self._load_from_disk()

    def _load_from_disk(self):
        print(f"Loading image from disk: {self.filename}")
        time.sleep(2)  # Simülasyon: Disk'ten yükleme zamanı

    def display(self):
        print(f"Displaying image: {self.filename}")

# Virtual Proxy - Lazy loading
class ImageProxy(Image):
    def __init__(self, filename: str):
        self.filename = filename
        self._real_image = None

    def display(self):
        if self._real_image is None:
            self._real_image = RealImage(self.filename)
        self._real_image.display()

# Kullanım
print("Creating proxy...")
image = ImageProxy("photo.jpg")

print("\nFirst display (loads from disk):")
image.display()

print("\nSecond display (uses cached):")
image.display()
```

**Protection Proxy - Erişim Kontrolü:**

```python
from abc import ABC, abstractmethod
from typing import Optional

class BankAccount(ABC):
    @abstractmethod
    def deposit(self, amount: float) -> bool:
        pass

    @abstractmethod
    def withdraw(self, amount: float) -> bool:
        pass

    @abstractmethod
    def get_balance(self) -> float:
        pass

class RealBankAccount(BankAccount):
    def __init__(self, account_number: str, initial_balance: float = 0):
        self.account_number = account_number
        self._balance = initial_balance

    def deposit(self, amount: float) -> bool:
        self._balance += amount
        print(f"Deposited ${amount:.2f}. New balance: ${self._balance:.2f}")
        return True

    def withdraw(self, amount: float) -> bool:
        if self._balance >= amount:
            self._balance -= amount
            print(f"Withdrew ${amount:.2f}. New balance: ${self._balance:.2f}")
            return True
        print("Insufficient funds")
        return False

    def get_balance(self) -> float:
        return self._balance

class BankAccountProxy(BankAccount):
    def __init__(self, account: RealBankAccount, password: str):
        self._account = account
        self._password = password
        self._authenticated = False

    def authenticate(self, password: str) -> bool:
        self._authenticated = (password == self._password)
        if self._authenticated:
            print("Authentication successful")
        else:
            print("Authentication failed")
        return self._authenticated

    def deposit(self, amount: float) -> bool:
        if not self._authenticated:
            print("Access denied: Please authenticate first")
            return False
        return self._account.deposit(amount)

    def withdraw(self, amount: float) -> bool:
        if not self._authenticated:
            print("Access denied: Please authenticate first")
            return False

        # Ek kontrol: Büyük miktarlar için onay gerektir
        if amount > 10000:
            print("Large withdrawal requires additional authorization")
            return False

        return self._account.withdraw(amount)

    def get_balance(self) -> float:
        if not self._authenticated:
            print("Access denied: Please authenticate first")
            return 0.0
        return self._account.get_balance()

# Kullanım
account = RealBankAccount("123456789", 5000)
proxy = BankAccountProxy(account, "secret123")

# Kimlik doğrulama olmadan erişim
proxy.deposit(1000)
proxy.get_balance()

# Kimlik doğrulama ile erişim
proxy.authenticate("secret123")
proxy.deposit(1000)
print(f"Balance: ${proxy.get_balance():.2f}")
proxy.withdraw(500)
proxy.withdraw(15000)  # Çok büyük çekim
```

**Real-World: API Rate Limiter Proxy**

```python
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from collections import deque
from typing import Dict, Any

class APIClient(ABC):
    @abstractmethod
    def make_request(self, endpoint: str, params: Dict[str, Any]) -> Dict[str, Any]:
        pass

class RealAPIClient(APIClient):
    def make_request(self, endpoint: str, params: Dict[str, Any]) -> Dict[str, Any]:
        print(f"Making API request to {endpoint}")
        # Simülasyon: Gerçek API çağrısı
        return {
            "status": "success",
            "data": f"Response from {endpoint}",
            "timestamp": datetime.now().isoformat()
        }

class RateLimiterProxy(APIClient):
    def __init__(self, api_client: APIClient, max_requests: int = 10,
                 time_window: int = 60):
        """
        api_client: Gerçek API client
        max_requests: Zaman penceresi içinde izin verilen maksimum istek sayısı
        time_window: Zaman penceresi (saniye)
        """
        self._api_client = api_client
        self._max_requests = max_requests
        self._time_window = time_window
        self._requests: deque = deque()
        self._cache: Dict[str, Dict] = {}

    def _is_rate_limited(self) -> bool:
        """Rate limit kontrolü"""
        now = datetime.now()
        cutoff_time = now - timedelta(seconds=self._time_window)

        # Eski istekleri temizle
        while self._requests and self._requests[0] < cutoff_time:
            self._requests.popleft()

        return len(self._requests) >= self._max_requests

    def _get_cache_key(self, endpoint: str, params: Dict[str, Any]) -> str:
        """Cache anahtarı oluştur"""
        params_str = str(sorted(params.items()))
        return f"{endpoint}:{params_str}"

    def make_request(self, endpoint: str, params: Dict[str, Any]) -> Dict[str, Any]:
        # Cache kontrolü
        cache_key = self._get_cache_key(endpoint, params)
        if cache_key in self._cache:
            cached_data = self._cache[cache_key]
            cache_age = (datetime.now() - cached_data["cached_at"]).seconds

            if cache_age < 300:  # 5 dakika cache
                print(f"Returning cached response (age: {cache_age}s)")
                return cached_data["response"]

        # Rate limit kontrolü
        if self._is_rate_limited():
            wait_time = self._time_window - (
                datetime.now() - self._requests[0]
            ).seconds
            return {
                "status": "error",
                "message": f"Rate limit exceeded. Try again in {wait_time} seconds",
                "requests_remaining": 0
            }

        # İsteği yap
        self._requests.append(datetime.now())
        response = self._api_client.make_request(endpoint, params)

        # Sonucu önbelleğe al
        self._cache[cache_key] = {
            "response": response,
            "cached_at": datetime.now()
        }

        # Metadata ekle
        response["rate_limit"] = {
            "requests_made": len(self._requests),
            "requests_remaining": self._max_requests - len(self._requests),
            "reset_time": (self._requests[0] + timedelta(seconds=self._time_window)).isoformat()
        }

        return response

# Kullanım
api = RealAPIClient()
proxy = RateLimiterProxy(api, max_requests=5, time_window=10)

# Normal istekler
for i in range(7):
    print(f"\n--- Request {i+1} ---")
    result = proxy.make_request("/api/users", {"page": 1})
    print(f"Status: {result['status']}")
    if "rate_limit" in result:
        print(f"Requests remaining: {result['rate_limit']['requests_remaining']}")
    time.sleep(0.5)

# Cache test
print("\n--- Cached Request ---")
result = proxy.make_request("/api/users", {"page": 1})
```

---

## Behavioral Patterns (Davranışsal Kalıplar)

### 9. Observer Pattern

**Amaç:** Bir nesnedeki değişiklikleri, ona bağımlı tüm nesnelere otomatik olarak bildirir. Publish-Subscribe modeli olarak da bilinir.

```python
from abc import ABC, abstractmethod
from typing import List

# Observer Interface
class Observer(ABC):
    @abstractmethod
    def update(self, subject):
        pass

# Subject (Observable)
class Subject:
    def __init__(self):
        self._observers: List[Observer] = []
        self._state = None

    def attach(self, observer: Observer):
        if observer not in self._observers:
            self._observers.append(observer)
            print(f"Attached observer: {observer.__class__.__name__}")

    def detach(self, observer: Observer):
        if observer in self._observers:
            self._observers.remove(observer)
            print(f"Detached observer: {observer.__class__.__name__}")

    def notify(self):
        print(f"Notifying {len(self._observers)} observers...")
        for observer in self._observers:
            observer.update(self)

    def set_state(self, state):
        print(f"\nSubject: State changed to {state}")
        self._state = state
        self.notify()

    def get_state(self):
        return self._state

# Concrete Observers
class BinaryObserver(Observer):
    def update(self, subject):
        state = subject.get_state()
        print(f"BinaryObserver: {bin(state) if isinstance(state, int) else 'N/A'}")

class HexObserver(Observer):
    def update(self, subject):
        state = subject.get_state()
        print(f"HexObserver: {hex(state) if isinstance(state, int) else 'N/A'}")

class OctalObserver(Observer):
    def update(self, subject):
        state = subject.get_state()
        print(f"OctalObserver: {oct(state) if isinstance(state, int) else 'N/A'}")

# Kullanım
subject = Subject()

binary_obs = BinaryObserver()
hex_obs = HexObserver()
octal_obs = OctalObserver()

subject.attach(binary_obs)
subject.attach(hex_obs)
subject.attach(octal_obs)

subject.set_state(15)
subject.set_state(10)

subject.detach(hex_obs)
subject.set_state(20)
```

**Real-World: Stock Market Monitoring System**

```python
from abc import ABC, abstractmethod
from typing import List, Dict
from dataclasses import dataclass
from datetime import datetime

@dataclass
class Stock:
    symbol: str
    price: float
    change: float
    timestamp: datetime

class StockObserver(ABC):
    @abstractmethod
    def update(self, stock: Stock):
        pass

class StockMarket:
    def __init__(self):
        self._observers: Dict[str, List[StockObserver]] = {}
        self._stocks: Dict[str, Stock] = {}

    def subscribe(self, symbol: str, observer: StockObserver):
        if symbol not in self._observers:
            self._observers[symbol] = []

        if observer not in self._observers[symbol]:
            self._observers[symbol].append(observer)
            print(f"{observer.__class__.__name__} subscribed to {symbol}")

    def unsubscribe(self, symbol: str, observer: StockObserver):
        if symbol in self._observers and observer in self._observers[symbol]:
            self._observers[symbol].remove(observer)
            print(f"{observer.__class__.__name__} unsubscribed from {symbol}")

    def update_stock(self, symbol: str, price: float):
        old_price = self._stocks[symbol].price if symbol in self._stocks else price
        change = ((price - old_price) / old_price * 100) if old_price else 0

        stock = Stock(symbol, price, change, datetime.now())
        self._stocks[symbol] = stock

        print(f"\n{'='*60}")
        print(f"Stock Update: {symbol} = ${price:.2f} ({change:+.2f}%)")
        print(f"{'='*60}")

        self._notify(symbol, stock)

    def _notify(self, symbol: str, stock: Stock):
        if symbol in self._observers:
            for observer in self._observers[symbol]:
                observer.update(stock)

# Concrete Observers
class PriceAlertObserver(StockObserver):
    def __init__(self, name: str, threshold_low: float, threshold_high: float):
        self.name = name
        self.threshold_low = threshold_low
        self.threshold_high = threshold_high

    def update(self, stock: Stock):
        if stock.price < self.threshold_low:
            print(f"  [ALERT - {self.name}] {stock.symbol} is below ${self.threshold_low}!")
        elif stock.price > self.threshold_high:
            print(f"  [ALERT - {self.name}] {stock.symbol} is above ${self.threshold_high}!")

class PortfolioObserver(StockObserver):
    def __init__(self, name: str, shares: int):
        self.name = name
        self.shares = shares
        self.total_value = 0

    def update(self, stock: Stock):
        self.total_value = stock.price * self.shares
        profit_loss = (stock.change / 100) * self.total_value
        print(f"  [PORTFOLIO - {self.name}] Value: ${self.total_value:.2f} "
              f"(P/L: ${profit_loss:+.2f})")

class LoggerObserver(StockObserver):
    def __init__(self):
        self.logs = []

    def update(self, stock: Stock):
        log_entry = (f"[{stock.timestamp.strftime('%Y-%m-%d %H:%M:%S')}] "
                    f"{stock.symbol}: ${stock.price:.2f} ({stock.change:+.2f}%)")
        self.logs.append(log_entry)
        print(f"  [LOG] {log_entry}")

class TradingBotObserver(StockObserver):
    def __init__(self, name: str):
        self.name = name

    def update(self, stock: Stock):
        if stock.change < -5:
            print(f"  [BOT - {self.name}] 🤖 Auto-buying {stock.symbol} - Price dropped {stock.change:.2f}%")
        elif stock.change > 5:
            print(f"  [BOT - {self.name}] 🤖 Auto-selling {stock.symbol} - Price increased {stock.change:.2f}%")

# Kullanım
market = StockMarket()

# Gözlemcileri oluştur
alert1 = PriceAlertObserver("Alert 1", threshold_low=140, threshold_high=160)
portfolio1 = PortfolioObserver("John's Portfolio", shares=100)
logger = LoggerObserver()
bot = TradingBotObserver("AlgoBot-1")

# Abonelikleri ayarla
market.subscribe("AAPL", alert1)
market.subscribe("AAPL", portfolio1)
market.subscribe("AAPL", logger)
market.subscribe("AAPL", bot)

# Hisse senedi güncellemeleri
market.update_stock("AAPL", 150.00)
market.update_stock("AAPL", 142.50)  # Alert tetiklenecek
market.update_stock("AAPL", 135.00)  # Bot alım yapacak
market.update_stock("AAPL", 158.00)  # Bot satış yapacak

# Farklı hisse senedi için
alert2 = PriceAlertObserver("Alert 2", threshold_low=2800, threshold_high=3200)
market.subscribe("GOOGL", alert2)
market.subscribe("GOOGL", logger)

market.update_stock("GOOGL", 3000.00)
market.update_stock("GOOGL", 3250.00)
```

### 10. Strategy Pattern

**Amaç:** Bir algoritma ailesini tanımlar, her birini ayrı sınıflara koyar ve birbirinin yerine kullanılabilir hale getirir.

```python
from abc import ABC, abstractmethod
from typing import List

# Strategy Interface
class SortStrategy(ABC):
    @abstractmethod
    def sort(self, data: List[int]) -> List[int]:
        pass

# Concrete Strategies
class BubbleSortStrategy(SortStrategy):
    def sort(self, data: List[int]) -> List[int]:
        print("Sorting using Bubble Sort")
        arr = data.copy()
        n = len(arr)
        for i in range(n):
            for j in range(0, n-i-1):
                if arr[j] > arr[j+1]:
                    arr[j], arr[j+1] = arr[j+1], arr[j]
        return arr

class QuickSortStrategy(SortStrategy):
    def sort(self, data: List[int]) -> List[int]:
        print("Sorting using Quick Sort")
        arr = data.copy()
        self._quick_sort(arr, 0, len(arr)-1)
        return arr

    def _quick_sort(self, arr, low, high):
        if low < high:
            pi = self._partition(arr, low, high)
            self._quick_sort(arr, low, pi-1)
            self._quick_sort(arr, pi+1, high)

    def _partition(self, arr, low, high):
        pivot = arr[high]
        i = low - 1
        for j in range(low, high):
            if arr[j] <= pivot:
                i += 1
                arr[i], arr[j] = arr[j], arr[i]
        arr[i+1], arr[high] = arr[high], arr[i+1]
        return i + 1

class MergeSortStrategy(SortStrategy):
    def sort(self, data: List[int]) -> List[int]:
        print("Sorting using Merge Sort")
        arr = data.copy()
        self._merge_sort(arr, 0, len(arr)-1)
        return arr

    def _merge_sort(self, arr, left, right):
        if left < right:
            mid = (left + right) // 2
            self._merge_sort(arr, left, mid)
            self._merge_sort(arr, mid+1, right)
            self._merge(arr, left, mid, right)

    def _merge(self, arr, left, mid, right):
        L = arr[left:mid+1]
        R = arr[mid+1:right+1]
        i = j = 0
        k = left

        while i < len(L) and j < len(R):
            if L[i] <= R[j]:
                arr[k] = L[i]
                i += 1
            else:
                arr[k] = R[j]
                j += 1
            k += 1

        while i < len(L):
            arr[k] = L[i]
            i += 1
            k += 1

        while j < len(R):
            arr[k] = R[j]
            j += 1
            k += 1

# Context
class Sorter:
    def __init__(self, strategy: SortStrategy):
        self._strategy = strategy

    def set_strategy(self, strategy: SortStrategy):
        self._strategy = strategy

    def sort(self, data: List[int]) -> List[int]:
        return self._strategy.sort(data)

# Kullanım
data = [64, 34, 25, 12, 22, 11, 90, 88, 45, 50, 30, 17]

sorter = Sorter(BubbleSortStrategy())
print(f"Original: {data}")
print(f"Sorted: {sorter.sort(data)}\n")

sorter.set_strategy(QuickSortStrategy())
print(f"Sorted: {sorter.sort(data)}\n")

sorter.set_strategy(MergeSortStrategy())
print(f"Sorted: {sorter.sort(data)}")
```

**Real-World: Payment Processing System**

```python
from abc import ABC, abstractmethod
from typing import Dict, Any
from dataclasses import dataclass
from datetime import datetime

@dataclass
class PaymentDetails:
    amount: float
    currency: str
    customer_id: str
    description: str

@dataclass
class PaymentResult:
    success: bool
    transaction_id: str
    message: str
    timestamp: datetime
    details: Dict[str, Any]

# Strategy Interface
class PaymentStrategy(ABC):
    @abstractmethod
    def process_payment(self, details: PaymentDetails) -> PaymentResult:
        pass

    @abstractmethod
    def validate(self, details: PaymentDetails) -> bool:
        pass

# Concrete Strategies
class CreditCardStrategy(PaymentStrategy):
    def __init__(self, card_number: str, cvv: str, expiry: str):
        self.card_number = card_number
        self.cvv = cvv
        self.expiry = expiry

    def validate(self, details: PaymentDetails) -> bool:
        print("Validating credit card...")
        # Basit validasyon simülasyonu
        if len(self.card_number) != 16 or len(self.cvv) != 3:
            return False
        return True

    def process_payment(self, details: PaymentDetails) -> PaymentResult:
        print(f"\n{'='*60}")
        print("Processing Credit Card Payment")
        print(f"{'='*60}")

        if not self.validate(details):
            return PaymentResult(
                success=False,
                transaction_id="",
                message="Invalid card details",
                timestamp=datetime.now(),
                details={}
            )

        # Ödeme işleme simülasyonu
        print(f"Charging ${details.amount} to card ending in {self.card_number[-4:]}")
        print(f"Description: {details.description}")

        return PaymentResult(
            success=True,
            transaction_id=f"CC_{datetime.now().timestamp()}",
            message="Payment successful",
            timestamp=datetime.now(),
            details={
                "method": "credit_card",
                "card_last4": self.card_number[-4:],
                "amount": details.amount,
                "currency": details.currency
            }
        )

class PayPalStrategy(PaymentStrategy):
    def __init__(self, email: str, password: str):
        self.email = email
        self.password = password

    def validate(self, details: PaymentDetails) -> bool:
        print("Validating PayPal account...")
        # Validasyon simülasyonu
        return "@" in self.email

    def process_payment(self, details: PaymentDetails) -> PaymentResult:
        print(f"\n{'='*60}")
        print("Processing PayPal Payment")
        print(f"{'='*60}")

        if not self.validate(details):
            return PaymentResult(
                success=False,
                transaction_id="",
                message="Invalid PayPal credentials",
                timestamp=datetime.now(),
                details={}
            )

        print(f"Processing ${details.amount} via PayPal")
        print(f"PayPal account: {self.email}")
        print(f"Description: {details.description}")

        return PaymentResult(
            success=True,
            transaction_id=f"PP_{datetime.now().timestamp()}",
            message="Payment successful",
            timestamp=datetime.now(),
            details={
                "method": "paypal",
                "email": self.email,
                "amount": details.amount,
                "currency": details.currency
            }
        )

class CryptoStrategy(PaymentStrategy):
    def __init__(self, wallet_address: str, crypto_type: str = "BTC"):
        self.wallet_address = wallet_address
        self.crypto_type = crypto_type

    def validate(self, details: PaymentDetails) -> bool:
        print(f"Validating {self.crypto_type} wallet...")
        # Wallet adresi validasyonu
        return len(self.wallet_address) > 20

    def process_payment(self, details: PaymentDetails) -> PaymentResult:
        print(f"\n{'='*60}")
        print(f"Processing {self.crypto_type} Payment")
        print(f"{'='*60}")

        if not self.validate(details):
            return PaymentResult(
                success=False,
                transaction_id="",
                message="Invalid wallet address",
                timestamp=datetime.now(),
                details={}
            )

        # Kripto kuru hesaplama (simülasyon)
        crypto_rates = {"BTC": 45000, "ETH": 3000, "USDT": 1}
        crypto_amount = details.amount / crypto_rates.get(self.crypto_type, 1)

        print(f"Processing ${details.amount} ({crypto_amount:.6f} {self.crypto_type})")
        print(f"Wallet: {self.wallet_address[:8]}...{self.wallet_address[-8:]}")
        print(f"Description: {details.description}")

        return PaymentResult(
            success=True,
            transaction_id=f"CRYPTO_{datetime.now().timestamp()}",
            message="Payment successful",
            timestamp=datetime.now(),
            details={
                "method": "cryptocurrency",
                "crypto_type": self.crypto_type,
                "crypto_amount": crypto_amount,
                "wallet": self.wallet_address,
                "amount": details.amount,
                "currency": details.currency
            }
        )

class BankTransferStrategy(PaymentStrategy):
    def __init__(self, iban: str, bank_name: str):
        self.iban = iban
        self.bank_name = bank_name

    def validate(self, details: PaymentDetails) -> bool:
        print("Validating bank account...")
        return len(self.iban) > 15

    def process_payment(self, details: PaymentDetails) -> PaymentResult:
        print(f"\n{'='*60}")
        print("Processing Bank Transfer")
        print(f"{'='*60}")

        if not self.validate(details):
            return PaymentResult(
                success=False,
                transaction_id="",
                message="Invalid IBAN",
                timestamp=datetime.now(),
                details={}
            )

        print(f"Initiating bank transfer of ${details.amount}")
        print(f"Bank: {self.bank_name}")
        print(f"IBAN: {self.iban[:4]}...{self.iban[-4:]}")
        print(f"Description: {details.description}")
        print("Note: Transfer will be processed in 1-3 business days")

        return PaymentResult(
            success=True,
            transaction_id=f"BANK_{datetime.now().timestamp()}",
            message="Transfer initiated successfully",
            timestamp=datetime.now(),
            details={
                "method": "bank_transfer",
                "bank": self.bank_name,
                "iban": self.iban,
                "amount": details.amount,
                "currency": details.currency,
                "processing_time": "1-3 business days"
            }
        )

# Context
class PaymentProcessor:
    def __init__(self, strategy: PaymentStrategy):
        self._strategy = strategy

    def set_strategy(self, strategy: PaymentStrategy):
        self._strategy = strategy
        print(f"\nPayment method changed to: {strategy.__class__.__name__}")

    def process(self, details: PaymentDetails) -> PaymentResult:
        result = self._strategy.process_payment(details)

        print(f"\n{'='*60}")
        if result.success:
            print(f"✓ Transaction ID: {result.transaction_id}")
            print(f"✓ {result.message}")
        else:
            print(f"✗ {result.message}")
        print(f"{'='*60}\n")

        return result

# Kullanım
payment_details = PaymentDetails(
    amount=150.00,
    currency="USD",
    customer_id="CUST_12345",
    description="Premium subscription - 1 year"
)

# Kredi kartı ile ödeme
cc_strategy = CreditCardStrategy("1234567890123456", "123", "12/25")
processor = PaymentProcessor(cc_strategy)
processor.process(payment_details)

# PayPal ile ödeme
paypal_strategy = PayPalStrategy("user@example.com", "password123")
processor.set_strategy(paypal_strategy)
processor.process(payment_details)

# Kripto ile ödeme
crypto_strategy = CryptoStrategy("1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa", "BTC")
processor.set_strategy(crypto_strategy)
processor.process(payment_details)

# Banka transferi ile ödeme
bank_strategy = BankTransferStrategy("TR330006100519786457841326", "Ziraat Bank")
processor.set_strategy(bank_strategy)
processor.process(payment_details)
```

Bu dosyanın devamını bir sonraki mesajda göndereceğim (karakter sınırı nedeniyle).

---

## Pythonic Patterns

Python'un kendine özgü özellikleri kullanılarak yazılan design pattern'ler:

### Context Managers

```python
from contextlib import contextmanager
import time

# __enter__ ve __exit__ ile
class Timer:
    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.elapsed = self.end - self.start
        print(f"Elapsed time: {self.elapsed:.4f} seconds")

# Kullanım
with Timer():
    time.sleep(1)
    print("Working...")

# Generator-based context manager
@contextmanager
def timer():
    start = time.time()
    yield
    end = time.time()
    print(f"Elapsed time: {end - start:.4f} seconds")

with timer():
    time.sleep(0.5)
    print("Working...")
```

### Descriptors

```python
class ValidatedAttribute:
    def __init__(self, min_value=None, max_value=None):
        self.min_value = min_value
        self.max_value = max_value
        self.data = {}

    def __get__(self, instance, owner):
        if instance is None:
            return self
        return self.data.get(id(instance))

    def __set__(self, instance, value):
        if self.min_value is not None and value < self.min_value:
            raise ValueError(f"Value must be >= {self.min_value}")
        if self.max_value is not None and value > self.max_value:
            raise ValueError(f"Value must be <= {self.max_value}")
        self.data[id(instance)] = value

class Product:
    price = ValidatedAttribute(min_value=0)
    quantity = ValidatedAttribute(min_value=0, max_value=1000)

    def __init__(self, name, price, quantity):
        self.name = name
        self.price = price
        self.quantity = quantity

product = Product("Laptop", 1500, 10)
# product.price = -100  # ValueError
```

---

## Behavioral Patterns (Devam)

### 11. Command Pattern

**Amaç:** İstekleri nesneler olarak kapsüller, böylece farklı isteklerle parametrize edebilir, kuyruğa alabilir veya geri alabilirsiniz.

```python
from abc import ABC, abstractmethod
from typing import List

# Command Interface
class Command(ABC):
    @abstractmethod
    def execute(self):
        pass

    @abstractmethod
    def undo(self):
        pass

# Receiver - Gerçek işi yapan sınıf
class TextEditor:
    def __init__(self):
        self.text = ""

    def write(self, text: str):
        self.text += text

    def delete(self, length: int):
        self.text = self.text[:-length]

    def get_text(self):
        return self.text

# Concrete Commands
class WriteCommand(Command):
    def __init__(self, editor: TextEditor, text: str):
        self.editor = editor
        self.text = text

    def execute(self):
        self.editor.write(self.text)

    def undo(self):
        self.editor.delete(len(self.text))

class DeleteCommand(Command):
    def __init__(self, editor: TextEditor, length: int):
        self.editor = editor
        self.length = length
        self.deleted_text = ""

    def execute(self):
        text = self.editor.get_text()
        self.deleted_text = text[-self.length:]
        self.editor.delete(self.length)

    def undo(self):
        self.editor.write(self.deleted_text)

# Invoker
class CommandHistory:
    def __init__(self):
        self.history: List[Command] = []
        self.current_index = -1

    def execute(self, command: Command):
        # Yeni komut eklendiğinde, redo history'sini temizle
        self.history = self.history[:self.current_index + 1]
        command.execute()
        self.history.append(command)
        self.current_index += 1

    def undo(self):
        if self.current_index >= 0:
            command = self.history[self.current_index]
            command.undo()
            self.current_index -= 1
            return True
        return False

    def redo(self):
        if self.current_index < len(self.history) - 1:
            self.current_index += 1
            command = self.history[self.current_index]
            command.execute()
            return True
        return False

# Kullanım
editor = TextEditor()
history = CommandHistory()

# Komutları çalıştır
history.execute(WriteCommand(editor, "Hello "))
print(f"Text: '{editor.get_text()}'")  # Hello

history.execute(WriteCommand(editor, "World"))
print(f"Text: '{editor.get_text()}'")  # Hello World

history.execute(WriteCommand(editor, "!"))
print(f"Text: '{editor.get_text()}'")  # Hello World!

# Undo
print("\nUndo:")
history.undo()
print(f"Text: '{editor.get_text()}'")  # Hello World

history.undo()
print(f"Text: '{editor.get_text()}'")  # Hello

# Redo
print("\nRedo:")
history.redo()
print(f"Text: '{editor.get_text()}'")  # Hello World
```

**Real-World: Smart Home Automation**

```python
from abc import ABC, abstractmethod
from typing import List, Dict
from datetime import datetime

# Receivers
class Light:
    def __init__(self, location: str):
        self.location = location
        self.is_on = False
        self.brightness = 0

    def turn_on(self):
        self.is_on = True
        self.brightness = 100
        print(f"{self.location} light is ON")

    def turn_off(self):
        self.is_on = False
        self.brightness = 0
        print(f"{self.location} light is OFF")

    def dim(self, level: int):
        self.brightness = level
        print(f"{self.location} light dimmed to {level}%")

class Thermostat:
    def __init__(self):
        self.temperature = 20

    def set_temperature(self, temp: int):
        self.temperature = temp
        print(f"Thermostat set to {temp}°C")

class SecuritySystem:
    def __init__(self):
        self.armed = False

    def arm(self):
        self.armed = True
        print("Security system ARMED")

    def disarm(self):
        self.armed = False
        print("Security system DISARMED")

# Commands
class Command(ABC):
    @abstractmethod
    def execute(self):
        pass

    @abstractmethod
    def undo(self):
        pass

class LightOnCommand(Command):
    def __init__(self, light: Light):
        self.light = light

    def execute(self):
        self.light.turn_on()

    def undo(self):
        self.light.turn_off()

class LightOffCommand(Command):
    def __init__(self, light: Light):
        self.light = light

    def execute(self):
        self.light.turn_off()

    def undo(self):
        self.light.turn_on()

class ThermostatCommand(Command):
    def __init__(self, thermostat: Thermostat, temperature: int):
        self.thermostat = thermostat
        self.temperature = temperature
        self.previous_temperature = None

    def execute(self):
        self.previous_temperature = self.thermostat.temperature
        self.thermostat.set_temperature(self.temperature)

    def undo(self):
        self.thermostat.set_temperature(self.previous_temperature)

class MacroCommand(Command):
    """Birden fazla komutu birlikte çalıştırır"""
    def __init__(self, commands: List[Command]):
        self.commands = commands

    def execute(self):
        for command in self.commands:
            command.execute()

    def undo(self):
        for command in reversed(self.commands):
            command.undo()

# Remote Control
class RemoteControl:
    def __init__(self):
        self.commands: Dict[str, Command] = {}
        self.history: List[Command] = []

    def set_command(self, button: str, command: Command):
        self.commands[button] = command

    def press_button(self, button: str):
        if button in self.commands:
            command = self.commands[button]
            command.execute()
            self.history.append(command)
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Executed: {button}")

    def press_undo(self):
        if self.history:
            command = self.history.pop()
            command.undo()
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Undo last command")

# Kullanım
living_room_light = Light("Living Room")
bedroom_light = Light("Bedroom")
thermostat = Thermostat()
security = SecuritySystem()

remote = RemoteControl()

# Tek komutlar
remote.set_command("living_room_on", LightOnCommand(living_room_light))
remote.set_command("living_room_off", LightOffCommand(living_room_light))
remote.set_command("bedroom_on", LightOnCommand(bedroom_light))
remote.set_command("heat", ThermostatCommand(thermostat, 24))

# Macro komut - "Leaving Home" senaryosu
leaving_home = MacroCommand([
    LightOffCommand(living_room_light),
    LightOffCommand(bedroom_light),
    ThermostatCommand(thermostat, 18)
])
remote.set_command("leaving_home", leaving_home)

# Komutları çalıştır
print("=== Smart Home Control ===\n")
remote.press_button("living_room_on")
remote.press_button("heat")
print("\nLeaving home...")
remote.press_button("leaving_home")
print("\nOops, came back! Undo...")
remote.press_undo()
```

### 12. State Pattern

**Amaç:** Bir nesnenin iç durumu değiştiğinde davranışını değiştirebilmesini sağlar.

```python
from abc import ABC, abstractmethod

# State Interface
class State(ABC):
    @abstractmethod
    def insert_coin(self, machine):
        pass

    @abstractmethod
    def eject_coin(self, machine):
        pass

    @abstractmethod
    def select_product(self, machine):
        pass

    @abstractmethod
    def dispense(self, machine):
        pass

# Concrete States
class NoCoinState(State):
    def insert_coin(self, machine):
        print("Coin inserted")
        machine.set_state(machine.has_coin_state)

    def eject_coin(self, machine):
        print("No coin to eject")

    def select_product(self, machine):
        print("Insert coin first")

    def dispense(self, machine):
        print("Insert coin first")

class HasCoinState(State):
    def insert_coin(self, machine):
        print("Coin already inserted")

    def eject_coin(self, machine):
        print("Coin ejected")
        machine.set_state(machine.no_coin_state)

    def select_product(self, machine):
        print("Product selected")
        machine.set_state(machine.dispensing_state)

    def dispense(self, machine):
        print("Select product first")

class DispensingState(State):
    def insert_coin(self, machine):
        print("Please wait, dispensing product")

    def eject_coin(self, machine):
        print("Cannot eject coin, product is being dispensed")

    def select_product(self, machine):
        print("Already dispensing")

    def dispense(self, machine):
        print("Dispensing product...")
        machine.set_state(machine.no_coin_state)

class OutOfStockState(State):
    def insert_coin(self, machine):
        print("Machine is out of stock, cannot accept coin")

    def eject_coin(self, machine):
        print("No coin to eject")

    def select_product(self, machine):
        print("Machine is out of stock")

    def dispense(self, machine):
        print("Machine is out of stock")

# Context
class VendingMachine:
    def __init__(self):
        self.no_coin_state = NoCoinState()
        self.has_coin_state = HasCoinState()
        self.dispensing_state = DispensingState()
        self.out_of_stock_state = OutOfStockState()

        self.state = self.no_coin_state
        self.stock = 10

    def set_state(self, state: State):
        self.state = state

    def insert_coin(self):
        self.state.insert_coin(self)

    def eject_coin(self):
        self.state.eject_coin(self)

    def select_product(self):
        self.state.select_product(self)
        if self.state == self.dispensing_state:
            self.state.dispense(self)
            self.stock -= 1
            if self.stock == 0:
                print("Machine is now out of stock!")
                self.set_state(self.out_of_stock_state)

# Kullanım
machine = VendingMachine()

print("=== Vending Machine Demo ===\n")
machine.select_product()  # No coin
machine.insert_coin()
machine.insert_coin()  # Already inserted
machine.select_product()  # Dispenses
print(f"Stock remaining: {machine.stock}\n")

machine.insert_coin()
machine.eject_coin()
```

**Real-World: Order Processing System**

```python
from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum

class OrderStatus(Enum):
    PENDING = "pending"
    CONFIRMED = "confirmed"
    PROCESSING = "processing"
    SHIPPED = "shipped"
    DELIVERED = "delivered"
    CANCELLED = "cancelled"

class OrderState(ABC):
    @abstractmethod
    def confirm(self, order):
        pass

    @abstractmethod
    def process(self, order):
        pass

    @abstractmethod
    def ship(self, order):
        pass

    @abstractmethod
    def deliver(self, order):
        pass

    @abstractmethod
    def cancel(self, order):
        pass

class PendingState(OrderState):
    def confirm(self, order):
        print(f"Order {order.order_id} confirmed")
        order.status = OrderStatus.CONFIRMED
        order.set_state(ConfirmedState())

    def process(self, order):
        print("Cannot process: Order must be confirmed first")

    def ship(self, order):
        print("Cannot ship: Order is pending")

    def deliver(self, order):
        print("Cannot deliver: Order is pending")

    def cancel(self, order):
        print(f"Order {order.order_id} cancelled")
        order.status = OrderStatus.CANCELLED
        order.set_state(CancelledState())

class ConfirmedState(OrderState):
    def confirm(self, order):
        print("Order is already confirmed")

    def process(self, order):
        print(f"Processing order {order.order_id}")
        order.status = OrderStatus.PROCESSING
        order.set_state(ProcessingState())

    def ship(self, order):
        print("Cannot ship: Order must be processed first")

    def deliver(self, order):
        print("Cannot deliver: Order must be shipped first")

    def cancel(self, order):
        print(f"Order {order.order_id} cancelled")
        order.status = OrderStatus.CANCELLED
        order.set_state(CancelledState())

class ProcessingState(OrderState):
    def confirm(self, order):
        print("Order is already confirmed")

    def process(self, order):
        print("Order is already being processed")

    def ship(self, order):
        print(f"Order {order.order_id} shipped")
        order.status = OrderStatus.SHIPPED
        order.set_state(ShippedState())

    def deliver(self, order):
        print("Cannot deliver: Order must be shipped first")

    def cancel(self, order):
        print("Cannot cancel: Order is being processed")

class ShippedState(OrderState):
    def confirm(self, order):
        print("Order is already confirmed and shipped")

    def process(self, order):
        print("Order is already processed")

    def ship(self, order):
        print("Order is already shipped")

    def deliver(self, order):
        print(f"Order {order.order_id} delivered")
        order.status = OrderStatus.DELIVERED
        order.set_state(DeliveredState())

    def cancel(self, order):
        print("Cannot cancel: Order has been shipped")

class DeliveredState(OrderState):
    def confirm(self, order):
        print("Order is already delivered")

    def process(self, order):
        print("Order is already delivered")

    def ship(self, order):
        print("Order is already delivered")

    def deliver(self, order):
        print("Order is already delivered")

    def cancel(self, order):
        print("Cannot cancel: Order has been delivered")

class CancelledState(OrderState):
    def confirm(self, order):
        print("Cannot confirm: Order is cancelled")

    def process(self, order):
        print("Cannot process: Order is cancelled")

    def ship(self, order):
        print("Cannot ship: Order is cancelled")

    def deliver(self, order):
        print("Cannot deliver: Order is cancelled")

    def cancel(self, order):
        print("Order is already cancelled")

class Order:
    def __init__(self, order_id: str):
        self.order_id = order_id
        self.status = OrderStatus.PENDING
        self.state = PendingState()
        self.created_at = datetime.now()

    def set_state(self, state: OrderState):
        self.state = state

    def confirm(self):
        self.state.confirm(self)

    def process(self):
        self.state.process(self)

    def ship(self):
        self.state.ship(self)

    def deliver(self):
        self.state.deliver(self)

    def cancel(self):
        self.state.cancel(self)

    def get_status(self):
        return f"Order {self.order_id}: {self.status.value.upper()}"

# Kullanım
print("=== Order Processing System ===\n")

order = Order("ORD-12345")
print(order.get_status())

print("\n--- Normal Flow ---")
order.confirm()
print(order.get_status())

order.process()
print(order.get_status())

order.ship()
print(order.get_status())

order.deliver()
print(order.get_status())

print("\n--- Invalid Operations ---")
order.cancel()  # Cannot cancel delivered order

# Yeni sipariş - İptal senaryosu
print("\n--- Cancellation Flow ---")
order2 = Order("ORD-67890")
order2.confirm()
order2.cancel()
print(order2.get_status())
order2.process()  # Cannot process cancelled order
```

### 13. Template Method Pattern

**Amaç:** Bir algoritmanın iskeletini tanımlar, bazı adımların alt sınıflar tarafından override edilmesine izin verir.

```python
from abc import ABC, abstractmethod

class DataMiner(ABC):
    """Template Method Pattern"""

    def mine(self, path: str):
        """Template method - algoritmanın iskeleti"""
        file = self.open_file(path)
        raw_data = self.extract_data(file)
        data = self.parse_data(raw_data)
        analysis = self.analyze_data(data)
        self.send_report(analysis)
        self.close_file(file)

    @abstractmethod
    def open_file(self, path: str):
        pass

    @abstractmethod
    def extract_data(self, file):
        pass

    @abstractmethod
    def parse_data(self, raw_data):
        pass

    def analyze_data(self, data):
        """Default implementation - override edilebilir"""
        print("Analyzing data...")
        return {"status": "analyzed", "data": data}

    def send_report(self, analysis):
        """Default implementation"""
        print(f"Report sent: {analysis}")

    @abstractmethod
    def close_file(self, file):
        pass

class CSVDataMiner(DataMiner):
    def open_file(self, path: str):
        print(f"Opening CSV file: {path}")
        return f"CSV_FILE:{path}"

    def extract_data(self, file):
        print("Extracting data from CSV...")
        return "csv,data,rows"

    def parse_data(self, raw_data):
        print("Parsing CSV data...")
        return raw_data.split(',')

    def close_file(self, file):
        print("Closing CSV file")

class JSONDataMiner(DataMiner):
    def open_file(self, path: str):
        print(f"Opening JSON file: {path}")
        return f"JSON_FILE:{path}"

    def extract_data(self, file):
        print("Extracting data from JSON...")
        return '{"key": "value"}'

    def parse_data(self, raw_data):
        print("Parsing JSON data...")
        return eval(raw_data)  # Simplified

    def analyze_data(self, data):
        """Override default analysis"""
        print("Custom JSON analysis...")
        return {"status": "json_analyzed", "data": data}

    def close_file(self, file):
        print("Closing JSON file")

# Kullanım
print("=== CSV Mining ===")
csv_miner = CSVDataMiner()
csv_miner.mine("data.csv")

print("\n=== JSON Mining ===")
json_miner = JSONDataMiner()
json_miner.mine("data.json")
```

---

## Pythonic Patterns (Devam)

### Properties ve Getters/Setters

```python
class Person:
    def __init__(self, name: str, age: int):
        self._name = name
        self._age = age

    @property
    def name(self):
        """Getter for name"""
        return self._name

    @name.setter
    def name(self, value: str):
        """Setter for name"""
        if not value:
            raise ValueError("Name cannot be empty")
        self._name = value

    @property
    def age(self):
        """Getter for age"""
        return self._age

    @age.setter
    def age(self, value: int):
        """Setter for age with validation"""
        if not 0 <= value <= 150:
            raise ValueError("Age must be between 0 and 150")
        self._age = value

    @property
    def is_adult(self):
        """Computed property"""
        return self._age >= 18

# Kullanım
person = Person("John", 25)
print(person.name)  # John
print(person.age)   # 25
print(person.is_adult)  # True

person.age = 30  # Setter kullanıyor
# person.age = 200  # ValueError
```

### Borg (Monostate) Pattern

**Pythonic Singleton alternatifi:**

```python
class Borg:
    """Tüm örnekler aynı state'i paylaşır"""
    _shared_state = {}

    def __init__(self):
        self.__dict__ = self._shared_state

class DatabaseConnection(Borg):
    def __init__(self):
        super().__init__()
        if not hasattr(self, 'connection'):
            self.connection = None
            self.host = None

    def connect(self, host: str):
        self.connection = f"Connected to {host}"
        self.host = host

# Kullanım
db1 = DatabaseConnection()
db1.connect("localhost")

db2 = DatabaseConnection()
print(db2.connection)  # Connected to localhost
print(db2.host)  # localhost

# Farklı objeler ama aynı state
print(db1 is db2)  # False
print(db1.__dict__ is db2.__dict__)  # True
```

### Mixin Pattern

```python
class JSONMixin:
    """JSON serialization mixin"""
    def to_json(self):
        import json
        return json.dumps(self.__dict__)

    @classmethod
    def from_json(cls, json_str):
        import json
        data = json.loads(json_str)
        return cls(**data)

class LoggingMixin:
    """Logging mixin"""
    def log(self, message):
        print(f"[{self.__class__.__name__}] {message}")

class User(JSONMixin, LoggingMixin):
    def __init__(self, name: str, email: str):
        self.name = name
        self.email = email

    def save(self):
        self.log(f"Saving user {self.name}")
        # Save logic here

# Kullanım
user = User("John Doe", "john@example.com")
user.save()
json_str = user.to_json()
print(json_str)

user2 = User.from_json(json_str)
print(user2.name)
```

### Registry Pattern

```python
class PluginRegistry:
    """Plugin kayıt sistemi"""
    _registry = {}

    @classmethod
    def register(cls, name):
        def decorator(plugin_class):
            cls._registry[name] = plugin_class
            return plugin_class
        return decorator

    @classmethod
    def get(cls, name):
        return cls._registry.get(name)

    @classmethod
    def get_all(cls):
        return cls._registry

# Plugin'leri kaydet
@PluginRegistry.register("mysql")
class MySQLPlugin:
    def connect(self):
        return "MySQL connected"

@PluginRegistry.register("postgres")
class PostgresPlugin:
    def connect(self):
        return "PostgreSQL connected"

# Kullanım
mysql = PluginRegistry.get("mysql")()
print(mysql.connect())

print("\nAll plugins:")
for name, plugin_class in PluginRegistry.get_all().items():
    print(f"  {name}: {plugin_class.__name__}")
```

---

## Best Practices

### 1. KISS (Keep It Simple, Stupid)
```python
# Kötü: Gereksiz karmaşık
class ComplexCalculator:
    def add(self, a, b):
        result = self._perform_addition(a, b)
        return self._validate_result(result)

    def _perform_addition(self, a, b):
        return a + b

    def _validate_result(self, result):
        return result

# İyi: Basit ve anlaşılır
def add(a, b):
    return a + b
```

### 2. DRY (Don't Repeat Yourself)
```python
# Kötü: Tekrar eden kod
def get_user_by_id(user_id):
    connection = create_db_connection()
    cursor = connection.cursor()
    cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
    result = cursor.fetchone()
    connection.close()
    return result

def get_user_by_email(email):
    connection = create_db_connection()
    cursor = connection.cursor()
    cursor.execute("SELECT * FROM users WHERE email = ?", (email,))
    result = cursor.fetchone()
    connection.close()
    return result

# İyi: Ortak fonksiyon
def execute_query(query, params):
    connection = create_db_connection()
    cursor = connection.cursor()
    cursor.execute(query, params)
    result = cursor.fetchone()
    connection.close()
    return result

def get_user_by_id(user_id):
    return execute_query("SELECT * FROM users WHERE id = ?", (user_id,))

def get_user_by_email(email):
    return execute_query("SELECT * FROM users WHERE email = ?", (email,))
```

### 3. YAGNI (You Aren't Gonna Need It)
```python
# Kötü: Gelecekte belki lazım olur diye eklenen özellikler
class User:
    def __init__(self, name):
        self.name = name
        self.future_feature_1 = None
        self.future_feature_2 = None
        self.maybe_needed = None

# İyi: Sadece şu an gereken özellikler
class User:
    def __init__(self, name):
        self.name = name
```

---

## Özet

### Ne Zaman Hangi Pattern?

**Creational Patterns:**
- **Singleton:** Tek örnek gerektiğinde (DB connection, logger)
- **Factory:** Nesne türü runtime'da belirlendiğinde
- **Builder:** Karmaşık nesneler adım adım oluşturulacaksa
- **Prototype:** Nesne oluşturma maliyeti yüksekse

**Structural Patterns:**
- **Adapter:** Uyumsuz arayüzleri birleştirmek için
- **Decorator:** Dinamik olarak sorumluluk eklemek için
- **Facade:** Karmaşık sistemi basitleştirmek için
- **Proxy:** Erişim kontrolü veya lazy loading için

**Behavioral Patterns:**
- **Observer:** Event-driven sistemler için
- **Strategy:** Algoritma değişimi gerektiğinde
- **Command:** İşlemleri nesneler olarak saklamak için
- **State:** Durum bazlı davranış değişimi için
- **Template Method:** Algoritma iskeleti sabitken adımlar değişiyorsa

### Design Pattern Anti-Patterns

**1. Overengineering:** Her şey için pattern kullanmak
**2. God Object:** Tek bir sınıfta çok fazla sorumluluk
**3. Lava Flow:** Kullanılmayan kod parçaları
**4. Golden Hammer:** Her probleme aynı çözümü uygulamak

---

## Sonuç

Design patterns, yazılım geliştirmede karşılaşılan yaygın problemlere kanıtlanmış çözümlerdir. Ancak:

- **Pattern'ler araç, amaç değildir**
- **Her probleme pattern uygulamaya çalışmayın**
- **Kodun basitliğini koruyun**
- **Python'un Pythonic özelliklerini kullanın**
- **İhtiyaç yokken pattern eklemeyin (YAGNI)**

Design patterns'i öğrenmek önemlidir, ancak ne zaman kullanmayacağınızı bilmek de o kadar önemlidir!
