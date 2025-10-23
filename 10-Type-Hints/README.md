# Type Hints ve Static Type Checking

Modern Python 3.10+ type hints ve static type checking için ileri seviye materyal.

## Dosyalar

### 📚 topic.md (1092 satır)
Kapsamlı Type Hints rehberi:
- ✅ Modern Python 3.10+ syntax (| operatörü, built-in generics)
- ✅ Typing module (Optional, Union, Literal, Final, TypedDict)
- ✅ Generics ve Type Variables (TypeVar, Generic, ParamSpec)
- ✅ Protocol ve Structural Subtyping
- ✅ Callable types ve Function Overloading (@overload)
- ✅ NewType ve Type Aliases
- ✅ Mypy configuration ve best practices
- ✅ Type Guards ve Type Narrowing
- ✅ Production patterns (API typing, DI, Event systems)
- ✅ 100+ kod örneği
- ✅ Türkçe açıklamalar

### 💻 exercises.py (1782 satır)
12 ileri seviye alıştırma:

**MEDIUM (3 alıştırma):**
1. Generic Repository Pattern - CRUD with generics
2. TypedDict API Responses - Type-safe REST API
3. Protocol-Based Plugins - Duck typing with protocols

**HARD (4 alıştırma):**
4. Type-Safe Decorator Factory - ParamSpec, retry/cache
5. Generic State Machine - Enum-based FSM
6. Function Overloading - @overload patterns
7. Type Guards ve Narrowing - TypeGuard, runtime validation

**EXPERT (5 alıştırma):**
8. Async Generic Repository - Async + Generics
9. Type-Safe Event System - Generic event bus
10. Builder Pattern with Types - Fluent interface
11. Dependency Injection Container - Protocol-based DI
12. Advanced Type Validation - Runtime type checking

## Özellikler

✨ **Modern Python 3.10+**
- `|` union operatörü (int | str yerine Union[int, str])
- Built-in generics (list[str] yerine List[str])
- Pattern matching support
- TypeAlias ve Self type

🎯 **Production Ready**
- Real-world patterns
- API type safety
- Generic collections
- Protocol-based design
- Async type hints

🔍 **Mypy Integration**
- Type narrowing
- Type guards
- Strict mode configuration
- Runtime type checking

## Kullanım

### Topic'i İncele
```bash
# Markdown görüntüleyici ile
mdcat topic.md

# Veya doğrudan oku
cat topic.md
```

### Alıştırmaları Çalıştır
```bash
# Tüm testleri çalıştır
python exercises.py

# Type checking yap
mypy exercises.py --strict

# Belirli bir alıştırmayı test et
python -c "from exercises import test_repository; test_repository()"
```

## Temel Kavramlar

### 1. Generic Types
```python
from typing import TypeVar, Generic

T = TypeVar('T')

class Stack(Generic[T]):
    def push(self, item: T) -> None: ...
    def pop(self) -> T: ...
```

### 2. Protocol (Structural Subtyping)
```python
from typing import Protocol

class Drawable(Protocol):
    def draw(self) -> str: ...
```

### 3. TypedDict
```python
from typing import TypedDict, NotRequired

class User(TypedDict):
    id: int
    name: str
    email: NotRequired[str]  # Optional
```

### 4. Overload
```python
from typing import overload, Literal

@overload
def fetch(source: Literal["db"]) -> dict: ...

@overload
def fetch(source: Literal["api"]) -> list: ...
```

## Mypy Configuration

```ini
[mypy]
python_version = 3.10
warn_return_any = True
disallow_untyped_defs = True
strict = True
```

## Best Practices

1. ✅ Progressive typing - Kademeli ekle
2. ✅ Protocol over inheritance - Yapısal typing
3. ✅ Generic types - Yeniden kullanılabilir
4. ✅ Type narrowing - isinstance, TypeGuard
5. ✅ Mypy in CI/CD - Otomatik kontrol

## Kaynaklar

- [Python Typing Docs](https://docs.python.org/3/library/typing.html)
- [Mypy Documentation](https://mypy.readthedocs.io/)
- [PEP 484 - Type Hints](https://www.python.org/dev/peps/pep-0484/)
- [PEP 544 - Protocols](https://www.python.org/dev/peps/pep-0544/)
- [PEP 585 - Type Hinting Generics](https://www.python.org/dev/peps/pep-0585/)
