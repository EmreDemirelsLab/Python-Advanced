"""
Python Internals - İleri Seviye Egzersizler
Her egzersiz TODO kısmında tamamlanmalı, sonra çözümle karşılaştırılmalıdır.
"""

# ============================================================================
# EXERCISE 1: Singleton Metaclass with Thread Safety
# Difficulty: Medium
# Topics: Metaclasses, Threading, Design Patterns
# ============================================================================

"""
Egzersiz 1: Thread-Safe Singleton Metaclass

Thread-safe bir Singleton metaclass yazın:
- Her class için tek bir instance'a izin vermeli
- Thread-safe olmalı (threading.Lock kullanın)
- Instance creation logging ekleyin
- __init__ metodunun sadece ilk seferde çalışmasını sağlayın
"""

import threading
from typing import Any, Dict


# TODO: SingletonMeta metaclass'ını implement edin
class SingletonMeta(type):
    """
    Thread-safe Singleton metaclass.

    Attributes:
        _instances: Class instance'larını saklar
        _lock: Thread synchronization için lock
    """
    pass

# SOLUTION:
class SingletonMetaSolution(type):
    """Thread-safe Singleton metaclass"""
    _instances: Dict[type, Any] = {}
    _lock: threading.Lock = threading.Lock()

    def __call__(cls, *args, **kwargs):
        # Double-checked locking pattern
        if cls not in cls._instances:
            with cls._lock:
                if cls not in cls._instances:
                    print(f"Creating new instance of {cls.__name__}")
                    instance = super().__call__(*args, **kwargs)
                    cls._instances[cls] = instance
                    # İlk initialization için flag
                    instance._initialized = True
                else:
                    print(f"Returning existing instance of {cls.__name__}")
        else:
            print(f"Returning existing instance of {cls.__name__}")

        return cls._instances[cls]


def test_singleton_metaclass():
    """Test SingletonMeta"""
    print("\n=== Testing Singleton Metaclass ===\n")

    class Database(metaclass=SingletonMetaSolution):
        def __init__(self, connection_string):
            if hasattr(self, '_initialized'):
                return  # Skip re-initialization
            self.connection_string = connection_string
            print(f"Initializing Database: {connection_string}")

    # Test
    db1 = Database("postgresql://localhost/db1")
    db2 = Database("postgresql://localhost/db2")

    print(f"\nSame instance: {db1 is db2}")
    print(f"Connection string: {db1.connection_string}")

    # Thread safety test
    instances = []

    def create_db():
        db = Database("test_connection")
        instances.append(db)

    threads = [threading.Thread(target=create_db) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    print(f"\nAll instances are same: {all(inst is instances[0] for inst in instances)}")


# ============================================================================
# EXERCISE 2: Auto-Property Metaclass
# Difficulty: Medium-Hard
# Topics: Metaclasses, Descriptors, Type Hints
# ============================================================================

"""
Egzersiz 2: Auto-Property Metaclass

Type hint'lere göre otomatik property oluşturan bir metaclass yazın:
- __annotations__'ı kullanarak typed attributes bulun
- Her typed attribute için property oluşturun
- Type validation ekleyin
- Getter ve setter metodları otomatik oluşturun
- Private attribute'lara dönüştürün (_name)
"""

# TODO: AutoPropertyMeta metaclass'ını implement edin
class AutoPropertyMeta(type):
    """
    Type hint'lere göre otomatik property oluşturan metaclass.

    Example:
        class Person(metaclass=AutoPropertyMeta):
            name: str
            age: int
    """
    pass

# SOLUTION:
class AutoPropertyMetaSolution(type):
    """Auto-property metaclass with type validation"""

    def __new__(mcs, name, bases, namespace):
        # Annotations'ı al
        annotations = namespace.get('__annotations__', {})

        for attr_name, attr_type in annotations.items():
            if attr_name.startswith('_'):
                continue  # Private attributes'u skip et

            private_name = f'_{attr_name}'

            # Property oluştur
            def make_property(attr_name, attr_type):
                def getter(self):
                    return getattr(self, f'_{attr_name}', None)

                def setter(self, value):
                    # Type validation
                    if not isinstance(value, attr_type):
                        raise TypeError(
                            f"{attr_name} must be {attr_type.__name__}, "
                            f"got {type(value).__name__}"
                        )
                    setattr(self, f'_{attr_name}', value)

                def deleter(self):
                    delattr(self, f'_{attr_name}')

                return property(getter, setter, deleter,
                              f"Property for {attr_name} ({attr_type.__name__})")

            # Property'yi namespace'e ekle
            namespace[attr_name] = make_property(attr_name, attr_type)

        return super().__new__(mcs, name, bases, namespace)


def test_auto_property_metaclass():
    """Test AutoPropertyMeta"""
    print("\n=== Testing Auto-Property Metaclass ===\n")

    class Person(metaclass=AutoPropertyMetaSolution):
        name: str
        age: int
        email: str

        def __init__(self, name, age, email):
            self.name = name
            self.age = age
            self.email = email

    # Test
    person = Person("Alice", 30, "alice@example.com")
    print(f"Person: {person.name}, {person.age}, {person.email}")

    # Type validation
    try:
        person.age = "invalid"
    except TypeError as e:
        print(f"\nValidation error: {e}")

    # Property info
    print(f"\nProperty docs:")
    for attr in ['name', 'age', 'email']:
        prop = getattr(Person, attr)
        print(f"  {attr}: {prop.__doc__}")


# ============================================================================
# EXERCISE 3: Bytecode Analyzer
# Difficulty: Hard
# Topics: Bytecode, dis module, Code Objects
# ============================================================================

"""
Egzersiz 3: Bytecode Analyzer

Fonksiyon bytecode'unu analiz eden bir class yazın:
- Bytecode instruction'larını parse edin
- Loop count, function call count hesaplayın
- Constant ve variable kullanımını analiz edin
- Complexity score hesaplayın
- Optimization önerileri verin
"""

import dis
import types
from collections import Counter


# TODO: BytecodeAnalyzer class'ını implement edin
class BytecodeAnalyzer:
    """
    Fonksiyon bytecode'unu analiz eder.

    Attributes:
        func: Analiz edilecek fonksiyon
        instructions: Bytecode instruction'ları
    """

    def __init__(self, func):
        self.func = func
        self.instructions = None

    def analyze(self) -> dict:
        """Bytecode'u analiz et ve report döndür"""
        pass

# SOLUTION:
class BytecodeAnalyzerSolution:
    """Bytecode analyzer implementation"""

    def __init__(self, func):
        self.func = func
        self.instructions = list(dis.get_instructions(func))

    def analyze(self) -> dict:
        """Comprehensive bytecode analysis"""
        return {
            'instruction_count': len(self.instructions),
            'loop_count': self._count_loops(),
            'call_count': self._count_calls(),
            'load_count': self._count_loads(),
            'store_count': self._count_stores(),
            'constants': self._get_constants(),
            'variables': self._get_variables(),
            'complexity_score': self._calculate_complexity(),
            'optimizations': self._suggest_optimizations()
        }

    def _count_loops(self) -> int:
        """Loop instruction sayısı"""
        loop_ops = {'FOR_ITER', 'JUMP_BACKWARD', 'GET_ITER'}
        return sum(1 for inst in self.instructions if inst.opname in loop_ops)

    def _count_calls(self) -> int:
        """Function call sayısı"""
        call_ops = {'CALL_FUNCTION', 'CALL_METHOD', 'CALL'}
        return sum(1 for inst in self.instructions if inst.opname in call_ops)

    def _count_loads(self) -> int:
        """Load operation sayısı"""
        return sum(1 for inst in self.instructions if 'LOAD' in inst.opname)

    def _count_stores(self) -> int:
        """Store operation sayısı"""
        return sum(1 for inst in self.instructions if 'STORE' in inst.opname)

    def _get_constants(self) -> list:
        """Kullanılan constant'lar"""
        return list(self.func.__code__.co_consts)

    def _get_variables(self) -> list:
        """Kullanılan variable'lar"""
        return list(self.func.__code__.co_varnames)

    def _calculate_complexity(self) -> int:
        """Complexity score (basit metrik)"""
        return (
            len(self.instructions) +
            self._count_loops() * 5 +
            self._count_calls() * 2
        )

    def _suggest_optimizations(self) -> list:
        """Optimization önerileri"""
        suggestions = []

        # Constant folding check
        load_const_count = sum(1 for inst in self.instructions
                              if inst.opname == 'LOAD_CONST')
        if load_const_count > 5:
            suggestions.append("Consider using constant folding")

        # Loop optimization
        if self._count_loops() > 2:
            suggestions.append("Multiple loops detected - consider vectorization")

        # Call optimization
        if self._count_calls() > 10:
            suggestions.append("Many function calls - consider inlining")

        return suggestions or ["No obvious optimizations"]


def test_bytecode_analyzer():
    """Test BytecodeAnalyzer"""
    print("\n=== Testing Bytecode Analyzer ===\n")

    def sample_function(n):
        """Sample function to analyze"""
        result = []
        for i in range(n):
            if i % 2 == 0:
                result.append(i * 2)
        return sum(result)

    analyzer = BytecodeAnalyzerSolution(sample_function)
    report = analyzer.analyze()

    print(f"Function: {sample_function.__name__}")
    print(f"\nAnalysis Report:")
    for key, value in report.items():
        print(f"  {key}: {value}")


# ============================================================================
# EXERCISE 4: Custom Import Hook
# Difficulty: Hard
# Topics: Import System, importlib, Hooks
# ============================================================================

"""
Egzersiz 4: Encrypted Module Importer

Encrypt edilmiş Python dosyalarını import eden bir sistem yazın:
- Custom MetaPathFinder ve Loader implement edin
- .pye (encrypted) uzantılı dosyaları okuyun
- Simple XOR encryption/decryption kullanın
- Module caching ekleyin
- Import monitoring yapın
"""

import sys
import importlib.abc
import importlib.machinery


# TODO: EncryptedImporter class'ını implement edin
class EncryptedImporter:
    """
    Encrypted (.pye) dosyalarını import eder.

    Encryption: Basit XOR cipher
    """
    pass

# SOLUTION:
class EncryptedImporterSolution(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Encrypted module importer"""

    def __init__(self, key: int = 42):
        self.key = key
        self.import_log = []

    def find_spec(self, fullname, path, target=None):
        """Encrypted module'ü bul"""
        print(f"Looking for encrypted module: {fullname}")

        # .pye dosyası için spec oluştur (gerçek dosya yoksa None döner)
        if fullname.startswith('encrypted_'):
            return importlib.machinery.ModuleSpec(
                fullname,
                self,
                origin=f"{fullname}.pye"
            )
        return None

    def create_module(self, spec):
        """Module object oluştur"""
        return None  # Default module creation

    def exec_module(self, module):
        """Module'ü execute et"""
        print(f"Loading encrypted module: {module.__name__}")

        # Simulated encrypted content
        encrypted_code = self._get_encrypted_content(module.__name__)
        decrypted_code = self._decrypt(encrypted_code)

        # Execute decrypted code
        exec(decrypted_code, module.__dict__)

        # Log import
        self.import_log.append(module.__name__)
        module.__doc__ = f"Encrypted module: {module.__name__}"

    def _encrypt(self, data: str) -> bytes:
        """Simple XOR encryption"""
        return bytes([ord(c) ^ self.key for c in data])

    def _decrypt(self, data: bytes) -> str:
        """Simple XOR decryption"""
        return ''.join([chr(b ^ self.key) for b in data])

    def _get_encrypted_content(self, module_name: str) -> bytes:
        """Simulated encrypted content"""
        # Real implementation would read from .pye file
        code = f"""
# Encrypted module: {module_name}
ENCRYPTED_VALUE = 12345
def encrypted_function(x):
    return x * 2
"""
        return self._encrypt(code)


def test_encrypted_importer():
    """Test EncryptedImporter"""
    print("\n=== Testing Encrypted Importer ===\n")

    # Install importer
    importer = EncryptedImporterSolution(key=42)
    sys.meta_path.insert(0, importer)

    try:
        # Import encrypted module
        import encrypted_test

        print(f"\nModule imported: {encrypted_test.__name__}")
        print(f"ENCRYPTED_VALUE: {encrypted_test.ENCRYPTED_VALUE}")
        print(f"encrypted_function(5): {encrypted_test.encrypted_function(5)}")
        print(f"\nImport log: {importer.import_log}")
    except Exception as e:
        print(f"Import error: {e}")
    finally:
        # Cleanup
        sys.meta_path.remove(importer)


# ============================================================================
# EXERCISE 5: Frame Inspector ve Debugger
# Difficulty: Hard
# Topics: Frame Objects, Debugging, Introspection
# ============================================================================

"""
Egzersiz 5: Advanced Frame Inspector

Call stack'i analiz eden bir debugger yazın:
- Frame inspection utilities
- Local ve global variable tracking
- Call stack visualization
- Execution time tracking
- Variable change detection
"""

import sys
import time
from typing import Optional, Dict, List


# TODO: FrameInspector class'ını implement edin
class FrameInspector:
    """
    Frame inspection ve debugging utilities.

    Features:
        - Stack trace analysis
        - Variable tracking
        - Execution timing
    """
    pass

# SOLUTION:
class FrameInspectorSolution:
    """Advanced frame inspection and debugging"""

    def __init__(self):
        self.call_log: List[Dict] = []
        self.execution_times: Dict[str, float] = {}

    def get_call_stack(self, depth: Optional[int] = None) -> List[Dict]:
        """Get detailed call stack information"""
        stack = []
        frame = sys._getframe(1)  # Skip this method

        while frame is not None:
            if depth is not None and len(stack) >= depth:
                break

            code = frame.f_code
            stack.append({
                'function': code.co_name,
                'filename': code.co_filename,
                'lineno': frame.f_lineno,
                'locals': dict(frame.f_locals),
                'arg_count': code.co_argcount,
                'var_names': code.co_varnames
            })

            frame = frame.f_back

        return stack

    def print_stack(self, depth: Optional[int] = None):
        """Print formatted call stack"""
        stack = self.get_call_stack(depth)

        print("\n=== Call Stack ===")
        for i, frame_info in enumerate(stack):
            print(f"\nFrame {i}: {frame_info['function']}")
            print(f"  Location: {frame_info['filename']}:{frame_info['lineno']}")
            print(f"  Arguments: {frame_info['arg_count']}")
            print(f"  Local variables: {list(frame_info['locals'].keys())}")

    def trace_calls(self, func):
        """Decorator to trace function calls"""
        def wrapper(*args, **kwargs):
            func_name = func.__name__
            start_time = time.time()

            # Log call
            self.call_log.append({
                'function': func_name,
                'args': args,
                'kwargs': kwargs,
                'timestamp': start_time
            })

            print(f"\n>>> Calling {func_name}")
            print(f"    Args: {args}, Kwargs: {kwargs}")

            try:
                result = func(*args, **kwargs)
                elapsed = time.time() - start_time

                # Track execution time
                if func_name not in self.execution_times:
                    self.execution_times[func_name] = []
                self.execution_times[func_name].append(elapsed)

                print(f"<<< {func_name} completed in {elapsed:.6f}s")
                print(f"    Result: {result}")

                return result
            except Exception as e:
                print(f"!!! {func_name} raised {type(e).__name__}: {e}")
                raise

        return wrapper

    def get_statistics(self) -> Dict:
        """Get execution statistics"""
        stats = {}

        for func_name, times in self.execution_times.items():
            stats[func_name] = {
                'call_count': len(times),
                'total_time': sum(times),
                'avg_time': sum(times) / len(times),
                'min_time': min(times),
                'max_time': max(times)
            }

        return stats

    def compare_frames(self, frame1, frame2) -> Dict:
        """Compare two frames for variable changes"""
        locals1 = frame1.f_locals
        locals2 = frame2.f_locals

        changes = {
            'added': set(locals2.keys()) - set(locals1.keys()),
            'removed': set(locals1.keys()) - set(locals2.keys()),
            'modified': {}
        }

        for key in set(locals1.keys()) & set(locals2.keys()):
            if locals1[key] != locals2[key]:
                changes['modified'][key] = {
                    'old': locals1[key],
                    'new': locals2[key]
                }

        return changes


def test_frame_inspector():
    """Test FrameInspector"""
    print("\n=== Testing Frame Inspector ===\n")

    inspector = FrameInspectorSolution()

    @inspector.trace_calls
    def calculate(x, y):
        result = x + y
        return result * 2

    @inspector.trace_calls
    def process_data(data):
        total = sum(data)
        avg = total / len(data)
        return {'total': total, 'avg': avg}

    # Test calls
    calculate(10, 5)
    process_data([1, 2, 3, 4, 5])
    calculate(20, 30)

    # Statistics
    print("\n=== Execution Statistics ===")
    stats = inspector.get_statistics()
    for func_name, func_stats in stats.items():
        print(f"\n{func_name}:")
        for key, value in func_stats.items():
            print(f"  {key}: {value}")

    # Stack trace
    def nested_call():
        inspector.print_stack(depth=5)

    def outer():
        nested_call()

    outer()


# ============================================================================
# EXERCISE 6: Custom Code Object Manipulator
# Difficulty: Expert
# Topics: Code Objects, Bytecode, Dynamic Code Generation
# ============================================================================

"""
Egzersiz 6: Code Object Transformer

Code object'leri manipüle eden bir transformer yazın:
- Constant değerleri değiştirin
- Variable name'leri rename edin
- New code object oluşturun
- Function behavior'ını değiştirin
- Optimization uygulayın
"""

import types
import dis


# TODO: CodeTransformer class'ını implement edin
class CodeTransformer:
    """
    Code object transformation ve manipulation.

    Capabilities:
        - Constant replacement
        - Variable renaming
        - Code optimization
    """
    pass

# SOLUTION:
class CodeTransformerSolution:
    """Code object transformer and optimizer"""

    def __init__(self, func):
        self.original_func = func
        self.code = func.__code__

    def replace_constants(self, replacements: Dict) -> types.FunctionType:
        """Replace constants in code object"""
        # Yeni constant tuple oluştur
        new_consts = []
        for const in self.code.co_consts:
            if const in replacements:
                new_consts.append(replacements[const])
            else:
                new_consts.append(const)

        # Yeni code object
        new_code = self.code.replace(co_consts=tuple(new_consts))

        # Yeni function
        return types.FunctionType(
            new_code,
            self.original_func.__globals__,
            self.original_func.__name__ + '_modified',
            self.original_func.__defaults__,
            self.original_func.__closure__
        )

    def rename_variables(self, renames: Dict[str, str]) -> types.FunctionType:
        """Rename variables in code object"""
        # Yeni variable names
        new_varnames = []
        for varname in self.code.co_varnames:
            if varname in renames:
                new_varnames.append(renames[varname])
            else:
                new_varnames.append(varname)

        # Yeni code object
        new_code = self.code.replace(co_varnames=tuple(new_varnames))

        return types.FunctionType(
            new_code,
            self.original_func.__globals__,
            self.original_func.__name__ + '_renamed'
        )

    def optimize_constants(self) -> types.FunctionType:
        """Optimize constant expressions"""
        # Bu örnek için constant folding simulation
        # Gerçek implementation bytecode manipulation gerektirir

        print(f"Analyzing {self.original_func.__name__} for optimizations...")

        instructions = list(dis.get_instructions(self.original_func))
        load_const_count = sum(1 for inst in instructions
                              if inst.opname == 'LOAD_CONST')

        print(f"Found {load_const_count} LOAD_CONST operations")

        # Bu noktada gerçek bytecode optimization yapılabilir
        return self.original_func

    def create_wrapper(self, pre_hook=None, post_hook=None) -> types.FunctionType:
        """Create function wrapper with hooks"""
        def wrapper(*args, **kwargs):
            if pre_hook:
                pre_hook(args, kwargs)

            result = self.original_func(*args, **kwargs)

            if post_hook:
                post_hook(result)

            return result

        return wrapper

    def analyze_code(self) -> Dict:
        """Analyze code object properties"""
        return {
            'name': self.code.co_name,
            'argcount': self.code.co_argcount,
            'nlocals': self.code.co_nlocals,
            'stacksize': self.code.co_stacksize,
            'flags': self.code.co_flags,
            'constants': self.code.co_consts,
            'names': self.code.co_names,
            'varnames': self.code.co_varnames,
            'freevars': self.code.co_freevars,
            'cellvars': self.code.co_cellvars,
            'filename': self.code.co_filename,
            'firstlineno': self.code.co_firstlineno,
            'bytecode_size': len(self.code.co_code)
        }


def test_code_transformer():
    """Test CodeTransformer"""
    print("\n=== Testing Code Transformer ===\n")

    def original_function(x):
        y = 10
        z = 20
        return x + y + z

    transformer = CodeTransformerSolution(original_function)

    # Analyze
    print("Original Function Analysis:")
    analysis = transformer.analyze_code()
    for key, value in analysis.items():
        print(f"  {key}: {value}")

    # Replace constants
    print("\n--- Constant Replacement ---")
    modified = transformer.replace_constants({10: 100, 20: 200})

    print(f"Original: {original_function(5)}")
    print(f"Modified: {modified(5)}")

    # Rename variables
    print("\n--- Variable Renaming ---")
    renamed = transformer.rename_variables({'x': 'input_value', 'y': 'offset'})
    print(f"Renamed vars: {renamed.__code__.co_varnames}")

    # Wrapper
    print("\n--- Function Wrapper ---")

    def pre_hook(args, kwargs):
        print(f"  Pre-hook: args={args}, kwargs={kwargs}")

    def post_hook(result):
        print(f"  Post-hook: result={result}")

    wrapped = transformer.create_wrapper(pre_hook, post_hook)
    result = wrapped(5)


# ============================================================================
# EXERCISE 7: Metaclass Framework
# Difficulty: Expert
# Topics: Metaclasses, ORM-style Framework, Descriptors
# ============================================================================

"""
Egzersiz 7: Simple ORM Metaclass

ORM-style bir framework için metaclass yazın:
- Field tanımlamaları (StringField, IntField, etc.)
- Automatic __init__ generation
- Validation
- SQL query generation (basit)
- Object to dict conversion
"""

from typing import Any, Type


# TODO: Field classes ve ModelMeta metaclass'ını implement edin
class Field:
    """Base field class"""
    pass

class StringField(Field):
    """String field"""
    pass

class IntField(Field):
    """Integer field"""
    pass

class ModelMeta(type):
    """ORM-style model metaclass"""
    pass

# SOLUTION:
class FieldSolution:
    """Base field descriptor"""

    def __init__(self, field_type, required=True, default=None):
        self.field_type = field_type
        self.required = required
        self.default = default
        self.name = None

    def __set_name__(self, owner, name):
        self.name = name

    def __get__(self, instance, owner):
        if instance is None:
            return self
        return instance.__dict__.get(self.name, self.default)

    def __set__(self, instance, value):
        # Validation
        if value is None:
            if self.required:
                raise ValueError(f"{self.name} is required")
            instance.__dict__[self.name] = None
            return

        if not isinstance(value, self.field_type):
            raise TypeError(
                f"{self.name} must be {self.field_type.__name__}, "
                f"got {type(value).__name__}"
            )

        instance.__dict__[self.name] = value


class StringFieldSolution(FieldSolution):
    """String field with max length"""

    def __init__(self, max_length=255, **kwargs):
        super().__init__(str, **kwargs)
        self.max_length = max_length

    def __set__(self, instance, value):
        super().__set__(instance, value)

        if value and len(value) > self.max_length:
            raise ValueError(
                f"{self.name} exceeds max length {self.max_length}"
            )


class IntFieldSolution(FieldSolution):
    """Integer field with min/max"""

    def __init__(self, min_value=None, max_value=None, **kwargs):
        super().__init__(int, **kwargs)
        self.min_value = min_value
        self.max_value = max_value

    def __set__(self, instance, value):
        super().__set__(instance, value)

        if value is not None:
            if self.min_value is not None and value < self.min_value:
                raise ValueError(f"{self.name} must be >= {self.min_value}")
            if self.max_value is not None and value > self.max_value:
                raise ValueError(f"{self.name} must be <= {self.max_value}")


class ModelMetaSolution(type):
    """ORM-style model metaclass"""

    def __new__(mcs, name, bases, namespace):
        # Field'leri topla
        fields = {}
        for key, value in namespace.items():
            if isinstance(value, FieldSolution):
                fields[key] = value

        # Store fields
        namespace['_fields'] = fields

        # Generate __init__
        def __init__(self, **kwargs):
            for field_name, field in self._fields.items():
                value = kwargs.get(field_name, field.default)
                setattr(self, field_name, value)

        namespace['__init__'] = __init__

        # to_dict method
        def to_dict(self):
            return {
                field_name: getattr(self, field_name)
                for field_name in self._fields
            }

        namespace['to_dict'] = to_dict

        # from_dict classmethod
        @classmethod
        def from_dict(cls, data):
            return cls(**data)

        namespace['from_dict'] = from_dict

        # Simple SQL generation
        def to_sql_insert(self):
            table_name = self.__class__.__name__.lower()
            columns = ', '.join(self._fields.keys())
            values = ', '.join(f"'{v}'" if isinstance(v, str) else str(v)
                             for v in self.to_dict().values())
            return f"INSERT INTO {table_name} ({columns}) VALUES ({values})"

        namespace['to_sql_insert'] = to_sql_insert

        return super().__new__(mcs, name, bases, namespace)


def test_orm_metaclass():
    """Test ORM Metaclass"""
    print("\n=== Testing ORM Metaclass ===\n")

    class User(metaclass=ModelMetaSolution):
        name = StringFieldSolution(max_length=100)
        age = IntFieldSolution(min_value=0, max_value=150)
        email = StringFieldSolution(max_length=255)

    # Create instance
    user = User(name="Alice", age=30, email="alice@example.com")

    print(f"User created: {user.name}, {user.age}, {user.email}")
    print(f"User dict: {user.to_dict()}")
    print(f"SQL INSERT: {user.to_sql_insert()}")

    # Validation test
    print("\n--- Validation Tests ---")

    try:
        user.age = "invalid"
    except TypeError as e:
        print(f"Type error: {e}")

    try:
        user.age = 200
    except ValueError as e:
        print(f"Value error: {e}")

    try:
        user.name = "x" * 150
    except ValueError as e:
        print(f"Length error: {e}")

    # from_dict
    print("\n--- From Dict ---")
    data = {'name': 'Bob', 'age': 25, 'email': 'bob@example.com'}
    user2 = User.from_dict(data)
    print(f"User from dict: {user2.to_dict()}")


# ============================================================================
# EXERCISE 8: Import Monitoring System
# Difficulty: Medium-Hard
# Topics: Import Hooks, sys.meta_path, Performance Monitoring
# ============================================================================

"""
Egzersiz 8: Import Performance Monitor

Import işlemlerini monitor eden bir sistem yazın:
- Her import'un süresini ölçün
- Import dependency tree oluşturun
- Slow import'ları tespit edin
- Import statistics raporu
- Cache hit/miss tracking
"""

import sys
import time
import importlib
from collections import defaultdict


# TODO: ImportMonitor class'ını implement edin
class ImportMonitor:
    """
    Import performance monitoring system.

    Tracks:
        - Import timing
        - Dependency tree
        - Cache hits
    """
    pass

# SOLUTION:
class ImportMonitorSolution:
    """Comprehensive import monitoring"""

    def __init__(self):
        self.import_times = {}
        self.import_tree = defaultdict(list)
        self.cache_hits = 0
        self.cache_misses = 0
        self.current_stack = []
        self.original_import = None

    def __enter__(self):
        """Start monitoring"""
        self.original_import = __builtins__.__import__
        __builtins__.__import__ = self._custom_import
        return self

    def __exit__(self, *args):
        """Stop monitoring"""
        __builtins__.__import__ = self.original_import

    def _custom_import(self, name, *args, **kwargs):
        """Custom import wrapper"""
        # Parent tracking
        parent = self.current_stack[-1] if self.current_stack else None

        # Check if already imported
        if name in sys.modules:
            self.cache_hits += 1
            if parent:
                self.import_tree[parent].append(f"{name} (cached)")
            return self.original_import(name, *args, **kwargs)

        # New import
        self.cache_misses += 1
        self.current_stack.append(name)

        # Time the import
        start_time = time.time()

        try:
            module = self.original_import(name, *args, **kwargs)
            elapsed = time.time() - start_time

            # Record timing
            self.import_times[name] = elapsed

            # Record dependency
            if parent:
                self.import_tree[parent].append(name)

            return module

        finally:
            self.current_stack.pop()

    def get_statistics(self) -> dict:
        """Get import statistics"""
        if not self.import_times:
            return {}

        times = list(self.import_times.values())

        return {
            'total_imports': len(self.import_times),
            'total_time': sum(times),
            'avg_time': sum(times) / len(times),
            'slowest': max(self.import_times.items(), key=lambda x: x[1]),
            'fastest': min(self.import_times.items(), key=lambda x: x[1]),
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': self.cache_hits / (self.cache_hits + self.cache_misses)
                       if (self.cache_hits + self.cache_misses) > 0 else 0
        }

    def print_report(self):
        """Print detailed report"""
        print("\n=== Import Performance Report ===\n")

        stats = self.get_statistics()

        if not stats:
            print("No imports tracked")
            return

        print("Overall Statistics:")
        print(f"  Total imports: {stats['total_imports']}")
        print(f"  Total time: {stats['total_time']:.6f}s")
        print(f"  Average time: {stats['avg_time']:.6f}s")
        print(f"  Cache hits: {stats['cache_hits']}")
        print(f"  Cache misses: {stats['cache_misses']}")
        print(f"  Hit rate: {stats['hit_rate']:.2%}")

        print(f"\nSlowest import: {stats['slowest'][0]} ({stats['slowest'][1]:.6f}s)")
        print(f"Fastest import: {stats['fastest'][0]} ({stats['fastest'][1]:.6f}s)")

        print("\nAll imports (sorted by time):")
        for module, elapsed in sorted(self.import_times.items(),
                                      key=lambda x: x[1], reverse=True):
            print(f"  {module}: {elapsed:.6f}s")

        if self.import_tree:
            print("\nDependency tree:")
            for parent, children in self.import_tree.items():
                print(f"  {parent}:")
                for child in children:
                    print(f"    - {child}")


def test_import_monitor():
    """Test ImportMonitor"""
    print("\n=== Testing Import Monitor ===\n")

    with ImportMonitorSolution() as monitor:
        # Import some modules
        import json
        import collections
        import itertools
        import functools

        # Re-import (cache hit)
        import json

    monitor.print_report()


# ============================================================================
# EXERCISE 9: Dynamic Class Generator
# Difficulty: Hard
# Topics: Metaclasses, Dynamic Type Creation, Code Generation
# ============================================================================

"""
Egzersiz 9: Dynamic Class Factory

Runtime'da class oluşturan bir factory yazın:
- Schema'dan class oluşturun
- Method generation
- Property generation
- Inheritance support
- Type hints ekleme
"""

from typing import Dict, Any, List, Callable


# TODO: ClassFactory'yi implement edin
class ClassFactory:
    """
    Dinamik olarak class oluşturur.

    Features:
        - Schema-based generation
        - Method injection
        - Dynamic inheritance
    """
    pass

# SOLUTION:
class ClassFactorySolution:
    """Dynamic class generation factory"""

    @staticmethod
    def create_class(
        name: str,
        attributes: Dict[str, Any] = None,
        methods: Dict[str, Callable] = None,
        bases: tuple = (),
        metaclass: type = type
    ) -> type:
        """
        Dinamik olarak class oluştur

        Args:
            name: Class name
            attributes: Class attributes
            methods: Methods to add
            bases: Base classes
            metaclass: Metaclass to use
        """
        namespace = {}

        # Attributes ekle
        if attributes:
            namespace.update(attributes)

        # Methods ekle
        if methods:
            namespace.update(methods)

        # Class oluştur
        return metaclass(name, bases or (object,), namespace)

    @staticmethod
    def from_schema(schema: Dict[str, Any]) -> type:
        """
        Schema'dan class oluştur

        Schema format:
        {
            'name': 'ClassName',
            'attributes': {'attr1': value1, ...},
            'methods': {'method1': func1, ...},
            'properties': ['prop1', 'prop2', ...],
            'bases': (BaseClass1, ...)
        }
        """
        name = schema['name']
        attributes = schema.get('attributes', {})
        methods = schema.get('methods', {})
        bases = schema.get('bases', ())
        properties = schema.get('properties', [])

        namespace = {}
        namespace.update(attributes)
        namespace.update(methods)

        # Generate __init__
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

        namespace['__init__'] = __init__

        # Generate properties
        for prop_name in properties:
            def make_property(name):
                def getter(self):
                    return getattr(self, f'_{name}', None)

                def setter(self, value):
                    setattr(self, f'_{name}', value)

                return property(getter, setter)

            namespace[prop_name] = make_property(prop_name)

        # Generate __repr__
        def __repr__(self):
            attrs = ', '.join(f"{k}={v!r}"
                            for k, v in self.__dict__.items()
                            if not k.startswith('_'))
            return f"{self.__class__.__name__}({attrs})"

        namespace['__repr__'] = __repr__

        return type(name, bases or (object,), namespace)

    @staticmethod
    def add_methods(cls: type, methods: Dict[str, Callable]):
        """Mevcut class'a method ekle"""
        for method_name, method in methods.items():
            setattr(cls, method_name, method)

    @staticmethod
    def create_dataclass(name: str, fields: List[tuple]) -> type:
        """
        Dataclass-style class oluştur

        Args:
            name: Class name
            fields: List of (field_name, field_type, default) tuples
        """
        namespace = {}
        annotations = {}
        defaults = {}

        # Fields'i parse et
        for field_info in fields:
            field_name = field_info[0]
            field_type = field_info[1] if len(field_info) > 1 else Any
            default = field_info[2] if len(field_info) > 2 else None

            annotations[field_name] = field_type
            if default is not None:
                defaults[field_name] = default

        namespace['__annotations__'] = annotations

        # __init__ generation
        def __init__(self, **kwargs):
            for field_name in annotations:
                value = kwargs.get(field_name, defaults.get(field_name))
                setattr(self, field_name, value)

        namespace['__init__'] = __init__

        # __repr__
        def __repr__(self):
            fields_str = ', '.join(f"{k}={getattr(self, k)!r}"
                                  for k in annotations)
            return f"{self.__class__.__name__}({fields_str})"

        namespace['__repr__'] = __repr__

        # __eq__
        def __eq__(self, other):
            if not isinstance(other, self.__class__):
                return False
            return all(getattr(self, k) == getattr(other, k)
                      for k in annotations)

        namespace['__eq__'] = __eq__

        return type(name, (object,), namespace)


def test_class_factory():
    """Test ClassFactory"""
    print("\n=== Testing Class Factory ===\n")

    factory = ClassFactorySolution()

    # Simple class creation
    print("--- Simple Class ---")
    MyClass = factory.create_class(
        'MyClass',
        attributes={'class_var': 42},
        methods={
            'greet': lambda self: f"Hello from {self.__class__.__name__}",
            'double': lambda self, x: x * 2
        }
    )

    obj = MyClass()
    print(f"Class: {obj.__class__.__name__}")
    print(f"Class var: {obj.class_var}")
    print(f"Greet: {obj.greet()}")
    print(f"Double(5): {obj.double(5)}")

    # Schema-based creation
    print("\n--- Schema-based Class ---")
    schema = {
        'name': 'Person',
        'attributes': {'species': 'human'},
        'properties': ['name', 'age'],
        'methods': {
            'introduce': lambda self: f"I'm {self.name}, {self.age} years old"
        }
    }

    Person = factory.from_schema(schema)
    person = Person(name='Alice', age=30)
    print(f"Person: {person}")
    print(f"Introduce: {person.introduce()}")

    # Dataclass-style
    print("\n--- Dataclass-style ---")
    Point = factory.create_dataclass(
        'Point',
        [('x', int, 0), ('y', int, 0)]
    )

    p1 = Point(x=10, y=20)
    p2 = Point(x=10, y=20)
    p3 = Point(x=5, y=15)

    print(f"Point 1: {p1}")
    print(f"Point 2: {p2}")
    print(f"p1 == p2: {p1 == p2}")
    print(f"p1 == p3: {p1 == p3}")


# ============================================================================
# EXERCISE 10: Advanced Signal Handler
# Difficulty: Medium
# Topics: Signal Handling, Context Managers, Graceful Shutdown
# ============================================================================

"""
Egzersiz 10: Graceful Shutdown Manager

Graceful shutdown için signal handler yazın:
- Multiple signal handling (SIGINT, SIGTERM)
- Cleanup callbacks
- Timeout for cleanup
- Shutdown hooks
- Status reporting
"""

import signal
import sys
import time
import atexit
from typing import Callable, List


# TODO: ShutdownManager'ı implement edin
class ShutdownManager:
    """
    Graceful shutdown management.

    Features:
        - Signal handling
        - Cleanup callbacks
        - Timeout management
    """
    pass

# SOLUTION:
class ShutdownManagerSolution:
    """Advanced graceful shutdown manager"""

    def __init__(self, timeout: int = 10):
        self.timeout = timeout
        self.cleanup_callbacks: List[Callable] = []
        self.shutdown_requested = False
        self.force_shutdown = False
        self.original_handlers = {}

        # Register atexit
        atexit.register(self._atexit_handler)

    def register_cleanup(self, callback: Callable, name: str = None):
        """Cleanup callback kaydet"""
        self.cleanup_callbacks.append({
            'callback': callback,
            'name': name or callback.__name__
        })

    def setup_handlers(self):
        """Signal handler'ları kur"""
        signals_to_handle = []

        # SIGINT (Ctrl+C)
        if hasattr(signal, 'SIGINT'):
            signals_to_handle.append(signal.SIGINT)

        # SIGTERM
        if hasattr(signal, 'SIGTERM'):
            signals_to_handle.append(signal.SIGTERM)

        for sig in signals_to_handle:
            self.original_handlers[sig] = signal.signal(sig, self._signal_handler)

        print(f"Shutdown handlers registered for signals: {signals_to_handle}")

    def _signal_handler(self, signum, frame):
        """Signal handler"""
        signal_name = signal.Signals(signum).name

        if self.shutdown_requested:
            print(f"\n!!! Forced shutdown requested ({signal_name})!")
            self.force_shutdown = True
            self._cleanup(force=True)
            sys.exit(1)

        print(f"\n>>> Shutdown requested ({signal_name})")
        print(f">>> Press Ctrl+C again to force quit")
        self.shutdown_requested = True

    def _cleanup(self, force: bool = False):
        """Execute cleanup callbacks"""
        print("\n>>> Starting cleanup process...")

        start_time = time.time()

        for i, callback_info in enumerate(self.cleanup_callbacks, 1):
            callback = callback_info['callback']
            name = callback_info['name']

            # Timeout check
            elapsed = time.time() - start_time
            if not force and elapsed > self.timeout:
                print(f"!!! Cleanup timeout reached ({self.timeout}s)")
                break

            try:
                print(f">>> Cleanup {i}/{len(self.cleanup_callbacks)}: {name}")
                callback()
            except Exception as e:
                print(f"!!! Cleanup error in {name}: {e}")

        total_time = time.time() - start_time
        print(f">>> Cleanup completed in {total_time:.2f}s")

    def _atexit_handler(self):
        """atexit handler"""
        if not self.shutdown_requested:
            print("\n>>> Process exiting (atexit)")
            self._cleanup()

    def wait_for_shutdown(self):
        """Shutdown signal'i bekle"""
        print(">>> Running (Press Ctrl+C to shutdown)...")

        try:
            while not self.shutdown_requested:
                time.sleep(0.1)
        except KeyboardInterrupt:
            # Handler zaten çalıştı
            pass

        # Cleanup
        self._cleanup()

    def run(self, main_loop: Callable):
        """Main loop'u shutdown management ile çalıştır"""
        self.setup_handlers()

        try:
            main_loop()
        except KeyboardInterrupt:
            pass
        finally:
            if not self.force_shutdown:
                self._cleanup()


def test_shutdown_manager():
    """Test ShutdownManager"""
    print("\n=== Testing Shutdown Manager ===\n")

    manager = ShutdownManagerSolution(timeout=5)

    # Cleanup callbacks
    def cleanup_database():
        print("  - Closing database connections...")
        time.sleep(0.5)

    def cleanup_cache():
        print("  - Flushing cache...")
        time.sleep(0.3)

    def cleanup_files():
        print("  - Closing open files...")
        time.sleep(0.2)

    manager.register_cleanup(cleanup_database)
    manager.register_cleanup(cleanup_cache)
    manager.register_cleanup(cleanup_files)

    # Setup handlers
    manager.setup_handlers()

    print("Shutdown manager ready")
    print("(Test çalıştırmak için Ctrl+C kullanın - güvenli test için comment'li)")

    # Gerçek kullanım (dikkatli!)
    # def main_loop():
    #     counter = 0
    #     while not manager.shutdown_requested:
    #         counter += 1
    #         print(f"Working... {counter}", end='\r')
    #         time.sleep(0.5)
    #
    # manager.run(main_loop)


# ============================================================================
# TEST RUNNER
# ============================================================================

def run_all_tests():
    """Tüm testleri çalıştır"""
    tests = [
        ("Singleton Metaclass", test_singleton_metaclass),
        ("Auto-Property Metaclass", test_auto_property_metaclass),
        ("Bytecode Analyzer", test_bytecode_analyzer),
        ("Encrypted Importer", test_encrypted_importer),
        ("Frame Inspector", test_frame_inspector),
        ("Code Transformer", test_code_transformer),
        ("ORM Metaclass", test_orm_metaclass),
        ("Import Monitor", test_import_monitor),
        ("Class Factory", test_class_factory),
        ("Shutdown Manager", test_shutdown_manager),
    ]

    print("=" * 70)
    print("PYTHON INTERNALS - İLERİ SEVİYE EGZERSİZLER")
    print("=" * 70)

    for test_name, test_func in tests:
        print(f"\n{'=' * 70}")
        print(f"TEST: {test_name}")
        print(f"{'=' * 70}")

        try:
            test_func()
        except Exception as e:
            print(f"\n!!! Test failed: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'=' * 70}")
    print("TÜM TESTLER TAMAMLANDI")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    run_all_tests()
