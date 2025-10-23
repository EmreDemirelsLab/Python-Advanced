# Python Internals (İleri Seviye)

## İçindekiler
1. [Metaclasses Deep Dive](#metaclasses-deep-dive)
2. [Bytecode ve Disassembly](#bytecode-ve-disassembly)
3. [Import System](#import-system)
4. [sys ve os Modülleri](#sys-ve-os-modülleri)
5. [Code Objects](#code-objects)
6. [Frame Objects](#frame-objects)
7. [Signal Handling](#signal-handling)
8. [Platform-Specific Code](#platform-specific-code)
9. [C Extensions Basics](#c-extensions-basics)
10. [Python Implementation Details](#python-implementation-details)

---

## Metaclasses Deep Dive

### Temel Kavramlar
Metaclass'lar Python'da class'ların class'larıdır. Her class aslında bir metaclass'ın instance'ıdır ve varsayılan olarak `type` metaclass'ını kullanır.

### Örnek 1: Basit Metaclass
```python
# Metaclass oluşturma ve kullanma
class Meta(type):
    """Custom metaclass - class oluşturma sürecini kontrol eder"""

    def __new__(mcs, name, bases, namespace, **kwargs):
        print(f"Creating class: {name}")
        print(f"Bases: {bases}")
        print(f"Namespace keys: {list(namespace.keys())}")

        # Class oluşturulmadan önce müdahale
        namespace['created_by_meta'] = True
        namespace['class_id'] = id(namespace)

        return super().__new__(mcs, name, bases, namespace)

    def __init__(cls, name, bases, namespace, **kwargs):
        print(f"Initializing class: {name}")
        super().__init__(name, bases, namespace)

    def __call__(cls, *args, **kwargs):
        """Instance oluşturma sürecini kontrol eder"""
        print(f"Creating instance of {cls.__name__}")
        instance = super().__call__(*args, **kwargs)
        print(f"Instance created: {instance}")
        return instance

class MyClass(metaclass=Meta):
    def __init__(self, value):
        self.value = value

    def method(self):
        return self.value * 2

# Test
obj = MyClass(10)
print(f"Has created_by_meta: {hasattr(MyClass, 'created_by_meta')}")
print(f"Class ID: {MyClass.class_id}")
```

### Örnek 2: Singleton Metaclass
```python
# Singleton pattern metaclass ile
class SingletonMeta(type):
    """Her class için sadece bir instance'a izin verir"""
    _instances = {}
    _lock = __import__('threading').Lock()

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            with cls._lock:
                # Double-checked locking
                if cls not in cls._instances:
                    instance = super().__call__(*args, **kwargs)
                    cls._instances[cls] = instance
        return cls._instances[cls]

class Database(metaclass=SingletonMeta):
    def __init__(self, connection_string):
        self.connection_string = connection_string
        print(f"Database initialized with: {connection_string}")

# Test
db1 = Database("postgresql://localhost/db1")
db2 = Database("postgresql://localhost/db2")  # Aynı instance
print(f"Same instance: {db1 is db2}")  # True
print(f"Connection: {db1.connection_string}")
```

### Örnek 3: Validation Metaclass
```python
# Attribute validation için metaclass
class ValidationMeta(type):
    """Class attribute'larını validate eden metaclass"""

    def __new__(mcs, name, bases, namespace):
        # __annotations__ kontrolü
        annotations = namespace.get('__annotations__', {})

        # Type checker metodları ekle
        for attr_name, attr_type in annotations.items():
            # Her typed attribute için setter oluştur
            private_name = f'_{attr_name}'

            def make_property(attr_name, attr_type):
                def getter(self):
                    return getattr(self, f'_{attr_name}', None)

                def setter(self, value):
                    if not isinstance(value, attr_type):
                        raise TypeError(
                            f"{attr_name} must be {attr_type.__name__}, "
                            f"got {type(value).__name__}"
                        )
                    setattr(self, f'_{attr_name}', value)

                return property(getter, setter)

            if attr_name in namespace:
                # Property olarak değiştir
                namespace[attr_name] = make_property(attr_name, attr_type)

        return super().__new__(mcs, name, bases, namespace)

class Person(metaclass=ValidationMeta):
    name: str
    age: int

    def __init__(self, name, age):
        self.name = name
        self.age = age

# Test
person = Person("Alice", 30)
print(f"Person: {person.name}, {person.age}")

try:
    person.age = "invalid"  # TypeError
except TypeError as e:
    print(f"Validation error: {e}")
```

---

## Bytecode ve Disassembly

### Bytecode Analizi
Python kodu önce bytecode'a derlenir, sonra CPython VM tarafından çalıştırılır.

### Örnek 4: Bytecode Disassembly
```python
import dis
import types

# Basit fonksiyon disassembly
def simple_function(x, y):
    """Basit bir matematik fonksiyonu"""
    result = x + y
    return result * 2

print("=== Simple Function Bytecode ===")
dis.dis(simple_function)

# Daha karmaşık fonksiyon
def complex_function(n):
    """List comprehension ve conditionals"""
    result = [i * 2 for i in range(n) if i % 2 == 0]
    return sum(result)

print("\n=== Complex Function Bytecode ===")
dis.dis(complex_function)

# Class method disassembly
class Calculator:
    def add(self, x, y):
        return x + y

print("\n=== Method Bytecode ===")
dis.dis(Calculator.add)
```

### Örnek 5: Bytecode Manipülasyonu
```python
import dis
import types

def original_function():
    """Original fonksiyon"""
    x = 10
    y = 20
    return x + y

print("=== Original Function ===")
dis.dis(original_function)

# Bytecode bilgilerini al
code = original_function.__code__

print(f"\nCode object attributes:")
print(f"  co_argcount: {code.co_argcount}")
print(f"  co_nlocals: {code.co_nlocals}")
print(f"  co_varnames: {code.co_varnames}")
print(f"  co_consts: {code.co_consts}")
print(f"  co_code (first 20 bytes): {code.co_code[:20]}")

# Yeni bir code object oluştur
new_code = code.replace(co_consts=(None, 100, 200))

new_function = types.FunctionType(
    new_code,
    original_function.__globals__,
    'modified_function'
)

print(f"\nOriginal result: {original_function()}")
print(f"Modified result: {new_function()}")
```

### Örnek 6: Bytecode Optimization Analizi
```python
import dis

# Constant folding örneği
def with_constant_folding():
    # Python bu hesaplamaları compile-time'da yapar
    x = 24 * 60 * 60  # saniye cinsinden bir gün
    return x

def without_constant_folding():
    # Bu runtime'da hesaplanır
    hours = 24
    minutes = 60
    seconds = 60
    return hours * minutes * seconds

print("=== With Constant Folding ===")
dis.dis(with_constant_folding)

print("\n=== Without Constant Folding ===")
dis.dis(without_constant_folding)

# Peephole optimization
def peephole_example():
    # Python bu tür optimizasyonları yapar
    x = None is None  # True'ya optimize edilir
    y = 1 + 2 + 3  # 6'ya optimize edilir
    return x, y

print("\n=== Peephole Optimization ===")
dis.dis(peephole_example)
```

---

## Import System

### Import Hooks ve Custom Importers

### Örnek 7: Custom Import Hook
```python
import sys
import importlib.abc
import importlib.machinery

class CustomFinder(importlib.abc.MetaPathFinder):
    """Custom module finder"""

    def find_spec(self, fullname, path, target=None):
        print(f"CustomFinder.find_spec called for: {fullname}")

        # Sadece 'custom_' ile başlayan modüller için
        if fullname.startswith('custom_'):
            return importlib.machinery.ModuleSpec(
                fullname,
                CustomLoader(),
                origin='custom'
            )
        return None

class CustomLoader(importlib.abc.Loader):
    """Custom module loader"""

    def create_module(self, spec):
        """Module object oluştur"""
        print(f"CustomLoader.create_module: {spec.name}")
        return None  # Default module oluşturma mekanizmasını kullan

    def exec_module(self, module):
        """Module'ü execute et"""
        print(f"CustomLoader.exec_module: {module.__name__}")

        # Module içeriğini dinamik olarak oluştur
        module.CUSTOM_VALUE = 42
        module.custom_function = lambda x: x * 2

        # Module attributes
        module.__doc__ = "Dynamically created module"
        module.__package__ = ""

# Custom finder'ı sys.meta_path'e ekle
sys.meta_path.insert(0, CustomFinder())

# Şimdi custom module'ü import edebiliriz
try:
    import custom_module
    print(f"\nCustom module imported!")
    print(f"CUSTOM_VALUE: {custom_module.CUSTOM_VALUE}")
    print(f"custom_function(5): {custom_module.custom_function(5)}")
except Exception as e:
    print(f"Import error: {e}")
```

### Örnek 8: JSON Module Importer
```python
import sys
import json
import importlib.abc
import importlib.machinery
from pathlib import Path

class JSONImporter(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """JSON dosyalarını Python modülü olarak import eder"""

    def __init__(self, path):
        self.path = Path(path)

    def find_spec(self, fullname, path, target=None):
        """JSON dosyası var mı kontrol et"""
        if path is None:
            path = [self.path]

        for search_path in path:
            json_file = Path(search_path) / f"{fullname}.json"
            if json_file.exists():
                return importlib.machinery.ModuleSpec(
                    fullname,
                    self,
                    origin=str(json_file)
                )
        return None

    def create_module(self, spec):
        return None  # Default module creation

    def exec_module(self, module):
        """JSON dosyasını oku ve module attribute'ları olarak ekle"""
        json_file = Path(module.__spec__.origin)

        with open(json_file, 'r') as f:
            data = json.load(f)

        # JSON data'yı module attributes olarak ekle
        for key, value in data.items():
            setattr(module, key, value)

        module.__file__ = str(json_file)
        module.__doc__ = f"Module loaded from {json_file}"

# Örnek kullanım (JSON dosyası olsaydı)
print("JSON Importer example - would work with actual JSON files")
```

### Örnek 9: Import Monitoring
```python
import sys
import importlib
import time
from contextlib import contextmanager

class ImportMonitor:
    """Import işlemlerini monitor eden context manager"""

    def __init__(self):
        self.imports = []
        self.original_import = None

    def custom_import(self, name, *args, **kwargs):
        """Import wrapper - timing ve logging"""
        start_time = time.time()

        # Original import'u çağır
        module = self.original_import(name, *args, **kwargs)

        # Timing bilgisi kaydet
        elapsed = time.time() - start_time
        self.imports.append({
            'name': name,
            'time': elapsed,
            'size': sys.getsizeof(module)
        })

        print(f"Imported {name} in {elapsed:.6f}s")
        return module

    def __enter__(self):
        # Import hook'u değiştir
        self.original_import = __builtins__.__import__
        __builtins__.__import__ = self.custom_import
        return self

    def __exit__(self, *args):
        # Original import'u geri yükle
        __builtins__.__import__ = self.original_import

    def report(self):
        """Import raporu"""
        print("\n=== Import Report ===")
        total_time = sum(imp['time'] for imp in self.imports)
        print(f"Total imports: {len(self.imports)}")
        print(f"Total time: {total_time:.6f}s")
        print("\nSlowest imports:")
        for imp in sorted(self.imports, key=lambda x: x['time'], reverse=True)[:5]:
            print(f"  {imp['name']}: {imp['time']:.6f}s")

# Test
with ImportMonitor() as monitor:
    import json
    import collections
    import itertools

monitor.report()
```

---

## sys ve os Modülleri

### Örnek 10: sys Modülü İleri Kullanım
```python
import sys
import platform

def analyze_python_environment():
    """Python çalışma ortamını detaylı analiz"""

    print("=== Python Environment Analysis ===\n")

    # Python version info
    print(f"Python Version: {sys.version}")
    print(f"Version Info: {sys.version_info}")
    print(f"Implementation: {sys.implementation.name}")
    print(f"Implementation Version: {sys.implementation.version}")

    # Platform info
    print(f"\nPlatform: {sys.platform}")
    print(f"Architecture: {platform.architecture()}")
    print(f"Machine: {platform.machine()}")

    # Memory management
    print(f"\nMax Int: {sys.maxsize}")
    print(f"Max Unicode: {sys.maxunicode}")
    print(f"Recursion Limit: {sys.getrecursionlimit()}")

    # Reference counting
    import gc
    print(f"\nGarbage Collection: {gc.isenabled()}")
    print(f"GC Counts: {gc.get_count()}")
    print(f"GC Threshold: {gc.get_threshold()}")

    # Module info
    print(f"\nBuilt-in Modules: {len(sys.builtin_module_names)}")
    print(f"Path Entries: {len(sys.path)}")
    print(f"\nPython Path (first 3):")
    for path in sys.path[:3]:
        print(f"  {path}")

    # Standard streams
    print(f"\nStdin: {sys.stdin}")
    print(f"Stdout: {sys.stdout}")
    print(f"Stderr: {sys.stderr}")

    # Exception handling
    exc_info = sys.exc_info()
    print(f"\nCurrent Exception: {exc_info[0]}")

analyze_python_environment()

# Reference counting example
import sys

class RefCountExample:
    def __init__(self, name):
        self.name = name

obj = RefCountExample("test")
print(f"\nReference count: {sys.getrefcount(obj)}")  # En az 2 (obj + getrefcount parametresi)

obj2 = obj  # Yeni referans
print(f"After assignment: {sys.getrefcount(obj)}")

del obj2
print(f"After deletion: {sys.getrefcount(obj)}")
```

### Örnek 11: Custom Exception Hook
```python
import sys
import traceback
import logging
from datetime import datetime

# Logging setup
logging.basicConfig(
    level=logging.ERROR,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class CustomExceptionHandler:
    """Custom exception handler with logging ve analysis"""

    def __init__(self):
        self.exception_count = {}
        self.original_excepthook = sys.excepthook

    def __call__(self, exc_type, exc_value, exc_traceback):
        """Custom exception handler"""
        # Exception counting
        exc_name = exc_type.__name__
        self.exception_count[exc_name] = self.exception_count.get(exc_name, 0) + 1

        # Log exception
        logging.error(f"Uncaught exception: {exc_name}")
        logging.error(f"Message: {exc_value}")
        logging.error(f"Occurred at: {datetime.now()}")

        # Detailed traceback
        tb_lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
        logging.error("Traceback:\n" + "".join(tb_lines))

        # Exception statistics
        logging.info(f"Exception count: {self.exception_count}")

        # Original handler'ı çağır (console output için)
        self.original_excepthook(exc_type, exc_value, exc_traceback)

    def get_statistics(self):
        """Exception istatistikleri"""
        return self.exception_count.copy()

# Custom handler'ı aktif et
handler = CustomExceptionHandler()
sys.excepthook = handler

# Test
try:
    def problematic_function():
        x = 1 / 0  # ZeroDivisionError

    problematic_function()
except:
    pass  # Handler catch edecek

print(f"\nException Statistics: {handler.get_statistics()}")
```

---

## Code Objects

### Örnek 12: Code Object Introspection
```python
import types
import inspect

def analyze_code_object(func):
    """Code object'i detaylı analiz et"""
    code = func.__code__

    print(f"=== Code Object Analysis: {func.__name__} ===\n")

    # Basic info
    print("Basic Information:")
    print(f"  Filename: {code.co_filename}")
    print(f"  First Line: {code.co_firstlineno}")
    print(f"  Name: {code.co_name}")

    # Arguments
    print(f"\nArgument Information:")
    print(f"  Arg Count: {code.co_argcount}")
    print(f"  Keyword-only Args: {code.co_kwonlyargcount}")
    print(f"  Positional-only Args: {code.co_posonlyargcount}")
    print(f"  Var Names: {code.co_varnames}")

    # Variables
    print(f"\nVariable Information:")
    print(f"  Local Variables: {code.co_nlocals}")
    print(f"  Free Variables: {code.co_freevars}")
    print(f"  Cell Variables: {code.co_cellvars}")

    # Constants and names
    print(f"\nConstants: {code.co_consts}")
    print(f"Names: {code.co_names}")

    # Stack and flags
    print(f"\nExecution Information:")
    print(f"  Stack Size: {code.co_stacksize}")
    print(f"  Flags: {code.co_flags}")
    print(f"  - Is Generator: {code.co_flags & inspect.CO_GENERATOR != 0}")
    print(f"  - Is Coroutine: {code.co_flags & inspect.CO_COROUTINE != 0}")
    print(f"  - Uses *args: {code.co_flags & inspect.CO_VARARGS != 0}")
    print(f"  - Uses **kwargs: {code.co_flags & inspect.CO_VARKEYWORDS != 0}")

    # Bytecode
    print(f"\nBytecode (first 30 bytes): {code.co_code[:30]}")

# Test fonksiyonları
def simple_function(a, b):
    c = a + b
    return c * 2

def complex_function(x, *args, **kwargs):
    result = [i * 2 for i in range(x)]
    return sum(result)

def generator_function(n):
    for i in range(n):
        yield i * 2

analyze_code_object(simple_function)
print("\n" + "="*60 + "\n")
analyze_code_object(complex_function)
print("\n" + "="*60 + "\n")
analyze_code_object(generator_function)
```

### Örnek 13: Dynamic Code Generation
```python
import types

class CodeGenerator:
    """Dinamik olarak fonksiyon oluşturucu"""

    @staticmethod
    def create_adder(n):
        """n değerini ekleyen bir fonksiyon oluştur"""
        def template(x):
            return x + n

        # Yeni code object oluştur
        code = template.__code__

        # Function oluştur
        new_func = types.FunctionType(
            code,
            {'n': n},  # Globals
            f'add_{n}',  # Name
            (n,),  # Defaults - kullanılmıyor ama örnek için
        )

        return new_func

    @staticmethod
    def create_multiplier(factor):
        """factor ile çarpan bir fonksiyon oluştur"""
        code_str = f"""
def multiply(x):
    return x * {factor}
"""
        # Compile code
        compiled = compile(code_str, '<dynamic>', 'exec')

        # Namespace
        namespace = {}
        exec(compiled, namespace)

        return namespace['multiply']

    @staticmethod
    def create_validator(type_check):
        """Belirli bir tip kontrolü yapan validator oluştur"""
        def validator(value):
            if not isinstance(value, type_check):
                raise TypeError(
                    f"Expected {type_check.__name__}, "
                    f"got {type(value).__name__}"
                )
            return True

        validator.__name__ = f'validate_{type_check.__name__}'
        return validator

# Test
print("=== Dynamic Code Generation ===\n")

# Adder fonksiyonları
add_5 = CodeGenerator.create_adder(5)
add_10 = CodeGenerator.create_adder(10)

print(f"add_5(3) = {add_5(3)}")
print(f"add_10(3) = {add_10(3)}")

# Multiplier fonksiyonları
multiply_by_3 = CodeGenerator.create_multiplier(3)
multiply_by_7 = CodeGenerator.create_multiplier(7)

print(f"\nmultiply_by_3(4) = {multiply_by_3(4)}")
print(f"multiply_by_7(4) = {multiply_by_7(4)}")

# Validators
validate_int = CodeGenerator.create_validator(int)
validate_str = CodeGenerator.create_validator(str)

print(f"\nvalidate_int(42): {validate_int(42)}")
try:
    validate_int("not an int")
except TypeError as e:
    print(f"validate_int('not an int'): {e}")
```

---

## Frame Objects

### Örnek 14: Stack Frame Inspection
```python
import sys
import inspect

def inspect_frame():
    """Mevcut frame'i incele"""
    frame = sys._getframe()

    print("=== Frame Inspection ===\n")
    print(f"Function Name: {frame.f_code.co_name}")
    print(f"Filename: {frame.f_code.co_filename}")
    print(f"Line Number: {frame.f_lineno}")
    print(f"Local Variables: {frame.f_locals}")
    print(f"Global Variables (count): {len(frame.f_globals)}")

    return frame

def nested_function_call():
    """Nested call stack'i göster"""
    x = 10
    y = 20

    def inner():
        z = 30
        return analyze_call_stack()

    return inner()

def analyze_call_stack():
    """Call stack'i analiz et"""
    print("\n=== Call Stack Analysis ===\n")

    frame = sys._getframe()
    depth = 0

    while frame is not None:
        code = frame.f_code
        print(f"Frame {depth}:")
        print(f"  Function: {code.co_name}")
        print(f"  File: {code.co_filename}:{frame.f_lineno}")
        print(f"  Locals: {list(frame.f_locals.keys())}")

        depth += 1
        frame = frame.f_back

    return depth

# Test
inspect_frame()
stack_depth = nested_function_call()
print(f"\nTotal stack depth: {stack_depth}")

# inspect modülü ile daha kolay
print("\n=== Using inspect module ===\n")

def show_caller_info():
    """Caller bilgisini göster"""
    # Bir üstteki frame
    caller_frame = inspect.currentframe().f_back

    info = inspect.getframeinfo(caller_frame)
    print(f"Called from:")
    print(f"  File: {info.filename}")
    print(f"  Function: {info.function}")
    print(f"  Line: {info.lineno}")
    print(f"  Code: {info.code_context[0].strip() if info.code_context else 'N/A'}")

def caller_function():
    print("Calling show_caller_info...")
    show_caller_info()

caller_function()
```

### Örnek 15: Frame Manipulation ve Debugging
```python
import sys
import inspect
from functools import wraps

class FrameDebugger:
    """Frame-based debugging utilities"""

    @staticmethod
    def trace_decorator(func):
        """Fonksiyon çağrılarını trace eden decorator"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            frame = sys._getframe()
            caller_frame = frame.f_back

            print(f"\n>>> Entering {func.__name__}")
            print(f"    Called from: {caller_frame.f_code.co_name}")
            print(f"    Args: {args}")
            print(f"    Kwargs: {kwargs}")

            try:
                result = func(*args, **kwargs)
                print(f"<<< Exiting {func.__name__}")
                print(f"    Result: {result}")
                return result
            except Exception as e:
                print(f"!!! Exception in {func.__name__}: {e}")
                raise

        return wrapper

    @staticmethod
    def get_local_vars(depth=1):
        """Belirli depth'teki local variables"""
        frame = sys._getframe(depth)
        return frame.f_locals.copy()

    @staticmethod
    def modify_caller_local(var_name, value):
        """Caller'ın local variable'ını değiştir (dikkatli kullan!)"""
        frame = sys._getframe(1)
        frame.f_locals[var_name] = value
        # Not: Bu her zaman çalışmayabilir çünkü locals() read-only olabilir

    @staticmethod
    def print_execution_context():
        """Tam execution context'i yazdır"""
        frame = sys._getframe(1)

        print("\n=== Execution Context ===")
        print(f"Function: {frame.f_code.co_name}")
        print(f"Line: {frame.f_lineno}")
        print("\nLocal Variables:")
        for name, value in frame.f_locals.items():
            print(f"  {name} = {value!r}")

        print("\nCall Stack:")
        for i, frame_info in enumerate(inspect.stack()[1:6]):
            print(f"  {i}: {frame_info.function} at {frame_info.filename}:{frame_info.lineno}")

# Test
@FrameDebugger.trace_decorator
def calculate(x, y, operation='add'):
    """Trace edilmiş fonksiyon"""
    FrameDebugger.print_execution_context()

    if operation == 'add':
        return x + y
    elif operation == 'multiply':
        return x * y
    return 0

# Test çağrısı
result = calculate(10, 5, operation='multiply')
print(f"\nFinal result: {result}")

# Local vars alma
def outer():
    x = 100
    y = 200

    def inner():
        z = 300
        # Outer'ın local variables'larını al
        outer_locals = FrameDebugger.get_local_vars(depth=2)
        print(f"\nOuter's locals from inner: {outer_locals}")

    inner()

outer()
```

---

## Signal Handling

### Örnek 16: Advanced Signal Handling
```python
import signal
import sys
import time
import os

class SignalHandler:
    """Advanced signal handling"""

    def __init__(self):
        self.shutdown_requested = False
        self.signal_count = {}

    def register_handlers(self):
        """Signal handler'ları kaydet"""
        # SIGINT (Ctrl+C) handler
        signal.signal(signal.SIGINT, self.handle_interrupt)

        # SIGTERM handler (Unix)
        if hasattr(signal, 'SIGTERM'):
            signal.signal(signal.SIGTERM, self.handle_terminate)

        # Alarm signal (Unix)
        if hasattr(signal, 'SIGALRM'):
            signal.signal(signal.SIGALRM, self.handle_alarm)

        print("Signal handlers registered")

    def handle_interrupt(self, signum, frame):
        """SIGINT (Ctrl+C) handler"""
        self.signal_count['SIGINT'] = self.signal_count.get('SIGINT', 0) + 1

        print(f"\n>>> Received SIGINT (count: {self.signal_count['SIGINT']})")
        print(f">>> Frame: {frame.f_code.co_name} at line {frame.f_lineno}")

        if self.signal_count['SIGINT'] >= 2:
            print(">>> Forced shutdown!")
            sys.exit(1)
        else:
            print(">>> Press Ctrl+C again to force quit")
            self.shutdown_requested = True

    def handle_terminate(self, signum, frame):
        """SIGTERM handler"""
        print("\n>>> Received SIGTERM, cleaning up...")
        self.cleanup()
        sys.exit(0)

    def handle_alarm(self, signum, frame):
        """SIGALRM handler"""
        print(f"\n>>> Alarm! Current function: {frame.f_code.co_name}")

    def cleanup(self):
        """Cleanup operations"""
        print(">>> Performing cleanup operations...")
        # Cleanup kodu buraya
        print(">>> Cleanup complete")

    def run(self):
        """Ana loop"""
        print("Running... (Press Ctrl+C to stop)")
        counter = 0

        while not self.shutdown_requested:
            counter += 1
            print(f"Working... {counter}", end='\r')
            time.sleep(0.5)

        print("\n>>> Graceful shutdown initiated")
        self.cleanup()

# Test (dikkatli - gerçek signal handler)
# handler = SignalHandler()
# handler.register_handlers()
# handler.run()

print("Signal handling example ready (commented out for safety)")
```

### Örnek 17: Timeout Decorator with Signals
```python
import signal
import functools
import time

class TimeoutError(Exception):
    """Timeout exception"""
    pass

def timeout(seconds):
    """Timeout decorator using SIGALRM (Unix only)"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Signal handler
            def timeout_handler(signum, frame):
                raise TimeoutError(f"Function '{func.__name__}' timed out after {seconds}s")

            # Platform check
            if not hasattr(signal, 'SIGALRM'):
                print(f"Warning: SIGALRM not available on this platform")
                return func(*args, **kwargs)

            # Eski handler'ı kaydet
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(seconds)

            try:
                result = func(*args, **kwargs)
            finally:
                # Alarm'ı iptal et ve eski handler'ı geri yükle
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)

            return result

        return wrapper
    return decorator

# Test fonksiyonları
@timeout(2)
def quick_function():
    """Hızlı tamamlanan fonksiyon"""
    time.sleep(1)
    return "Completed quickly"

@timeout(2)
def slow_function():
    """Yavaş fonksiyon - timeout'a uğrar"""
    time.sleep(5)
    return "Completed slowly"

# Test
print("Testing timeout decorator...\n")

try:
    result = quick_function()
    print(f"Quick function: {result}")
except TimeoutError as e:
    print(f"Timeout: {e}")

try:
    result = slow_function()
    print(f"Slow function: {result}")
except TimeoutError as e:
    print(f"Timeout: {e}")
```

---

## Platform-Specific Code

### Örnek 18: Cross-Platform Code
```python
import sys
import platform
import os
from pathlib import Path

class PlatformUtils:
    """Platform-specific utilities"""

    @staticmethod
    def get_platform_info():
        """Detaylı platform bilgisi"""
        return {
            'system': platform.system(),
            'release': platform.release(),
            'version': platform.version(),
            'machine': platform.machine(),
            'processor': platform.processor(),
            'python_version': platform.python_version(),
            'python_implementation': platform.python_implementation(),
        }

    @staticmethod
    def get_path_separator():
        """Platform-specific path separator"""
        return os.sep

    @staticmethod
    def get_config_dir():
        """Platform-specific config directory"""
        if sys.platform == 'win32':
            return Path(os.environ['APPDATA'])
        elif sys.platform == 'darwin':
            return Path.home() / 'Library' / 'Application Support'
        else:  # Linux/Unix
            return Path.home() / '.config'

    @staticmethod
    def get_temp_dir():
        """Platform-specific temp directory"""
        import tempfile
        return Path(tempfile.gettempdir())

    @staticmethod
    def clear_screen():
        """Platform-specific screen clear"""
        os.system('cls' if os.name == 'nt' else 'clear')

    @staticmethod
    def open_file_explorer(path):
        """Platform-specific file explorer"""
        path = str(path)

        if sys.platform == 'win32':
            os.startfile(path)
        elif sys.platform == 'darwin':
            os.system(f'open "{path}"')
        else:  # Linux
            os.system(f'xdg-open "{path}"')

    @staticmethod
    def get_cpu_count():
        """Platform-specific CPU count"""
        return os.cpu_count() or 1

    @staticmethod
    def is_64bit():
        """64-bit platform check"""
        return sys.maxsize > 2**32

# Test
print("=== Platform Information ===\n")

info = PlatformUtils.get_platform_info()
for key, value in info.items():
    print(f"{key}: {value}")

print(f"\nPath Separator: {PlatformUtils.get_path_separator()}")
print(f"Config Directory: {PlatformUtils.get_config_dir()}")
print(f"Temp Directory: {PlatformUtils.get_temp_dir()}")
print(f"CPU Count: {PlatformUtils.get_cpu_count()}")
print(f"Is 64-bit: {PlatformUtils.is_64bit()}")

# Platform-specific code örneği
if sys.platform == 'win32':
    print("\nRunning on Windows")
    # Windows-specific code
elif sys.platform == 'darwin':
    print("\nRunning on macOS")
    # macOS-specific code
else:
    print("\nRunning on Linux/Unix")
    # Linux-specific code
```

---

## C Extensions Basics

### Örnek 19: ctypes ile C Fonksiyon Kullanımı
```python
import ctypes
import sys
import platform

class CExtensionExample:
    """ctypes ile C extension örneği"""

    @staticmethod
    def use_c_stdlib():
        """C standard library fonksiyonları"""
        # Platform-specific library loading
        if sys.platform == 'win32':
            libc = ctypes.CDLL('msvcrt')
        else:
            libc = ctypes.CDLL('libc.so.6' if sys.platform.startswith('linux')
                              else 'libc.dylib')

        # printf kullanımı
        printf = libc.printf
        printf.argtypes = [ctypes.c_char_p]
        printf.restype = ctypes.c_int

        # C printf çağrısı
        result = printf(b"Hello from C printf!\n")
        print(f"printf returned: {result}")

        # strlen kullanımı
        strlen = libc.strlen
        strlen.argtypes = [ctypes.c_char_p]
        strlen.restype = ctypes.c_size_t

        test_string = b"Python and C together"
        length = strlen(test_string)
        print(f"String length (via C strlen): {length}")

        return libc

    @staticmethod
    def create_c_struct():
        """C struct tanımlama"""
        class Point(ctypes.Structure):
            _fields_ = [
                ('x', ctypes.c_int),
                ('y', ctypes.c_int)
            ]

        class Rectangle(ctypes.Structure):
            _fields_ = [
                ('top_left', Point),
                ('bottom_right', Point),
                ('color', ctypes.c_char * 20)
            ]

        # Struct kullanımı
        rect = Rectangle()
        rect.top_left.x = 0
        rect.top_left.y = 0
        rect.bottom_right.x = 100
        rect.bottom_right.y = 50
        rect.color = b"red"

        print(f"\nRectangle struct:")
        print(f"  Top-left: ({rect.top_left.x}, {rect.top_left.y})")
        print(f"  Bottom-right: ({rect.bottom_right.x}, {rect.bottom_right.y})")
        print(f"  Color: {rect.color.decode()}")
        print(f"  Size: {ctypes.sizeof(rect)} bytes")

        return rect

    @staticmethod
    def pointer_example():
        """C pointer kullanımı"""
        # Integer pointer
        value = ctypes.c_int(42)
        ptr = ctypes.pointer(value)

        print(f"\nPointer example:")
        print(f"  Value: {value.value}")
        print(f"  Pointer contents: {ptr.contents.value}")
        print(f"  Pointer address: {ctypes.addressof(value)}")

        # Pointer üzerinden değiştir
        ptr.contents = ctypes.c_int(100)
        print(f"  Modified value: {value.value}")

        # Array pointer
        arr = (ctypes.c_int * 5)(1, 2, 3, 4, 5)
        print(f"\n  Array: {[arr[i] for i in range(5)]}")
        print(f"  Array address: {ctypes.addressof(arr)}")

# Test
print("=== C Extensions with ctypes ===\n")

example = CExtensionExample()

# C stdlib
libc = example.use_c_stdlib()

# Struct
rect = example.create_c_struct()

# Pointers
example.pointer_example()
```

### Örnek 20: Python C API Simulation
```python
import sys
import ctypes

class PythonCAPI:
    """Python C API'ye benzer düşük seviye operasyonlar"""

    @staticmethod
    def get_refcount(obj):
        """Object reference count"""
        return sys.getrefcount(obj) - 1  # -1 for getrefcount's temporary reference

    @staticmethod
    def get_object_id(obj):
        """Object ID (memory address)"""
        return id(obj)

    @staticmethod
    def get_object_size(obj):
        """Object size in bytes"""
        return sys.getsizeof(obj)

    @staticmethod
    def get_type_info(obj):
        """Object type information"""
        obj_type = type(obj)
        return {
            'type': obj_type.__name__,
            'module': obj_type.__module__,
            'mro': [c.__name__ for c in obj_type.__mro__],
            'dict': hasattr(obj, '__dict__'),
            'slots': hasattr(obj_type, '__slots__'),
        }

    @staticmethod
    def analyze_object(obj):
        """Nesneyi detaylı analiz et"""
        print(f"\n=== Object Analysis ===")
        print(f"Object: {obj!r}")
        print(f"Type: {type(obj).__name__}")
        print(f"ID: {PythonCAPI.get_object_id(obj)}")
        print(f"Size: {PythonCAPI.get_object_size(obj)} bytes")
        print(f"RefCount: {PythonCAPI.get_refcount(obj)}")

        type_info = PythonCAPI.get_type_info(obj)
        print(f"Type Info:")
        for key, value in type_info.items():
            print(f"  {key}: {value}")

        # Attributes
        if hasattr(obj, '__dict__'):
            print(f"Instance Dict: {obj.__dict__}")

    @staticmethod
    def compare_objects(obj1, obj2):
        """İki nesneyi karşılaştır"""
        print(f"\n=== Object Comparison ===")
        print(f"Same object (is): {obj1 is obj2}")
        print(f"Equal (==): {obj1 == obj2}")
        print(f"Same ID: {id(obj1) == id(obj2)}")
        print(f"Same type: {type(obj1) is type(obj2)}")

        if obj1 is not obj2:
            print(f"ID difference: {abs(id(obj1) - id(obj2))}")

# Test
api = PythonCAPI()

# Simple objects
api.analyze_object(42)
api.analyze_object("Hello")
api.analyze_object([1, 2, 3])

# Custom class
class CustomClass:
    def __init__(self, value):
        self.value = value

obj = CustomClass(100)
api.analyze_object(obj)

# Object comparison
a = [1, 2, 3]
b = a
c = [1, 2, 3]

api.compare_objects(a, b)
api.compare_objects(a, c)

# Small integer caching
x = 256
y = 256
api.compare_objects(x, y)  # Same object (cached)

x = 257
y = 257
api.compare_objects(x, y)  # Different objects (not cached)
```

---

## Python Implementation Details

### İleri Seviye Özellikler

Python'un internal implementation detayları, performans optimizasyonları ve memory management konuları.

### Key Concepts:
- **Object internals**: Her Python nesnesi PyObject struct'ı
- **Memory management**: Reference counting + generational garbage collection
- **Small integer caching**: -5 ile 256 arası integer'lar cached
- **String interning**: Bazı string'ler memory'de tek kez saklanır
- **List over-allocation**: List'ler büyürken extra space ayırır
- **Dictionary implementation**: Hash table + open addressing
- **GIL (Global Interpreter Lock)**: Thread synchronization
- **Bytecode caching**: .pyc dosyaları compiled bytecode içerir

Bu advanced konuları anlamak, Python'u daha verimli kullanmanıza ve debug etmenize yardımcı olur.

---

## Özet

Python Internals konuları:

1. **Metaclasses**: Class oluşturma sürecini kontrol
2. **Bytecode**: Python kodu bytecode'a derlenir
3. **Import System**: Module loading mekanizması
4. **sys/os**: Sistem seviyesi operasyonlar
5. **Code Objects**: Compiled function representation
6. **Frame Objects**: Runtime execution context
7. **Signals**: Process signal handling
8. **Platform Code**: Cross-platform development
9. **C Extensions**: C ile Python entegrasyonu
10. **Implementation**: CPython internal detaylar

Python'un internal çalışma prensiplerini anlamak, daha iyi kod yazmanıza ve problemleri daha kolay çözmenize yardımcı olur.
