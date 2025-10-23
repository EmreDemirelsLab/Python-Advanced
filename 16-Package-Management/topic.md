# Package Management (Paket Yönetimi)

## İçindekiler
1. [Pip Advanced](#pip-advanced)
2. [Virtual Environments](#virtual-environments)
3. [Poetry - Modern Dependency Management](#poetry)
4. [Package Creation](#package-creation)
5. [Publishing to PyPI](#publishing-to-pypi)
6. [Semantic Versioning](#semantic-versioning)
7. [Dependency Management](#dependency-management)
8. [Production Best Practices](#production-best-practices)

## Pip Advanced

### Requirements.txt ve Constraints

**Requirements.txt** bağımlılıkları belirtir, **constraints.txt** ise versiyonları sınırlar.

```python
# requirements.txt örneği
"""
# Temel bağımlılıklar
requests>=2.28.0,<3.0.0
flask==2.3.0
numpy~=1.24.0  # Compatible release (>=1.24.0, <1.25.0)

# Ekstra özelliklerle
sqlalchemy[postgresql]>=2.0.0

# Git repository'den
git+https://github.com/user/repo.git@v1.0.0#egg=package-name

# Lokal paket
-e ./local-package

# Diğer requirements dosyalarını dahil etme
-r requirements-dev.txt

# Constraints dosyası kullanımı
-c constraints.txt
"""
```

```python
# constraints.txt örneği
"""
# Constraints sadece versiyon sınırlamaları belirtir
# Otomatik olarak yüklenmez, sadece yüklendiğinde sınırlar
certifi==2023.7.22
charset-normalizer==3.2.0
idna==3.4
urllib3>=1.26.0,<2.0.0
"""
```

### Pip Configuration

```python
# pip.conf / pip.ini yapılandırması
"""
# Global config: ~/.config/pip/pip.conf (Linux/macOS)
# Global config: %APPDATA%\pip\pip.ini (Windows)

[global]
timeout = 60
index-url = https://pypi.org/simple
extra-index-url = https://private-pypi.company.com/simple
trusted-host = private-pypi.company.com

[install]
no-cache-dir = true
compile = true

[list]
format = columns

[freeze]
timeout = 10
"""
```

### Advanced Pip Commands

```bash
# Hash verification (güvenlik için)
pip install --require-hashes -r requirements.txt

# Requirements file with hashes
"""
requests==2.28.0 \
    --hash=sha256:64299f4909223da747622c030b781c0d7811e359c37124b4bd368fb8c6518baa

flask==2.3.0 \
    --hash=sha256:58107ed83e1c5b6f79721c57a2b9f12aa5296c9a79c1e0c98a2b8c7c5c9c5c5c
"""

# Dependency resolution strategies
pip install --upgrade-strategy eager package-name  # Tüm bağımlılıkları güncelle
pip install --upgrade-strategy only-if-needed package-name  # Sadece gerekirse güncelle

# Compile from source
pip install --no-binary :all: numpy

# Install without dependencies
pip install --no-deps package-name

# Dry run (ne yükleneceğini göster)
pip install --dry-run package-name

# Download only (yükleme yapmadan indir)
pip download -d ./packages -r requirements.txt
```

## Virtual Environments

### venv (Python 3.3+)

```python
# venv oluşturma ve kullanma
"""
# Oluşturma
python -m venv myenv
python -m venv --system-site-packages myenv  # Sistem paketlerine erişim
python -m venv --copies myenv  # Symlink yerine dosya kopyala

# Aktivasyon
# Linux/macOS:
source myenv/bin/activate

# Windows:
myenv\Scripts\activate.bat  # CMD
myenv\Scripts\Activate.ps1  # PowerShell

# Deaktivasyon
deactivate
"""

# Programatik venv kullanımı
import venv
import os
from pathlib import Path

class ExtendedEnvBuilder(venv.EnvBuilder):
    """Özelleştirilmiş environment builder"""

    def __init__(self, *args, **kwargs):
        self.packages = kwargs.pop('packages', [])
        super().__init__(*args, **kwargs)

    def post_setup(self, context):
        """Environment oluşturulduktan sonra paketleri yükle"""
        if self.packages:
            self.install_packages(context)

    def install_packages(self, context):
        """Paketleri pip ile yükle"""
        import subprocess

        pip_exe = os.path.join(context.bin_path, 'pip')

        for package in self.packages:
            print(f"Installing {package}...")
            subprocess.check_call([pip_exe, 'install', package])

# Kullanım
builder = ExtendedEnvBuilder(
    with_pip=True,
    packages=['requests', 'flask', 'pytest']
)
builder.create('./my_project_env')
```

### virtualenv (Daha gelişmiş)

```python
# virtualenv özellikleri
"""
# Kurulum
pip install virtualenv

# Farklı Python versiyonları ile
virtualenv -p python3.9 myenv
virtualenv --python=/usr/bin/python3.10 myenv

# Template environment (hızlı klonlama)
virtualenv --always-copy template_env
virtualenv --relocatable template_env
cp -r template_env project_env

# Seed packages ile
virtualenv --seeder pip myenv

# System packages ile
virtualenv --system-site-packages myenv
"""

# virtualenv API kullanımı
from virtualenv import cli_run

def create_advanced_venv(path, python_version=None, packages=None):
    """Gelişmiş virtual environment oluşturma"""
    args = [str(path)]

    if python_version:
        args.extend(['--python', python_version])

    # Environment oluştur
    cli_run(args)

    # Paketleri yükle
    if packages:
        import subprocess
        pip_path = Path(path) / 'bin' / 'pip'
        if pip_path.exists():
            subprocess.check_call([str(pip_path), 'install'] + packages)

# Kullanım
create_advanced_venv(
    './my_env',
    python_version='3.10',
    packages=['django', 'celery', 'redis']
)
```

### pyenv (Python Version Management)

```bash
# pyenv kurulumu ve kullanımı
"""
# Kurulum (Linux/macOS)
curl https://pyenv.run | bash

# Python versiyonlarını listele
pyenv install --list

# Python versiyonu yükle
pyenv install 3.11.0
pyenv install 3.10.8

# Global Python versiyonu
pyenv global 3.11.0

# Local (proje bazlı) Python versiyonu
cd my_project
pyenv local 3.10.8  # .python-version dosyası oluşturur

# Shell için geçici versiyon
pyenv shell 3.9.0

# Yüklü versiyonları göster
pyenv versions

# pyenv-virtualenv plugin
pyenv virtualenv 3.11.0 my-project-env
pyenv activate my-project-env
pyenv deactivate
"""
```

## Poetry - Modern Dependency Management

### Poetry Kurulum ve Başlangıç

```bash
# Poetry kurulumu
curl -sSL https://install.python-poetry.org | python3 -

# Yeni proje oluşturma
poetry new my-package
poetry new --src my-package  # src layout ile

# Mevcut projeye ekle
cd existing-project
poetry init

# Bağımlılık ekleme
poetry add requests
poetry add --group dev pytest
poetry add "flask>=2.0.0,<3.0.0"
poetry add django@^4.0  # Semantic versioning

# Bağımlılık kaldırma
poetry remove requests

# Güncelleme
poetry update
poetry update requests  # Sadece requests'i güncelle

# Kilit dosyasını güncelle (paketleri yükleme)
poetry lock --no-update

# Paketleri yükle
poetry install
poetry install --no-dev  # Sadece production bağımlılıkları
poetry install --extras "mysql pgsql"  # Ekstra bağımlılıklarla
```

### pyproject.toml - Modern Configuration

```toml
# pyproject.toml örneği
[tool.poetry]
name = "my-awesome-package"
version = "0.1.0"
description = "An awesome Python package"
authors = ["Your Name <you@example.com>"]
license = "MIT"
readme = "README.md"
homepage = "https://github.com/user/my-awesome-package"
repository = "https://github.com/user/my-awesome-package"
documentation = "https://my-awesome-package.readthedocs.io"
keywords = ["api", "client", "awesome"]
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
]
packages = [{include = "my_package"}]

[tool.poetry.dependencies]
python = "^3.9"
requests = "^2.28.0"
pydantic = "^2.0.0"
click = "^8.1.0"

# Optional dependencies (extras)
psycopg2-binary = {version = "^2.9.0", optional = true}
pymysql = {version = "^1.0.0", optional = true}
redis = {version = "^4.5.0", optional = true}

# Platform specific
pywin32 = {version = "^305", platform = "win32"}

# Python version specific
importlib-metadata = {version = "^6.0", python = "<3.10"}

[tool.poetry.group.dev.dependencies]
pytest = "^7.3.0"
pytest-cov = "^4.1.0"
black = "^23.3.0"
mypy = "^1.3.0"
ruff = "^0.0.270"
pre-commit = "^3.3.0"

[tool.poetry.group.docs.dependencies]
sphinx = "^6.2.0"
sphinx-rtd-theme = "^1.2.0"

[tool.poetry.extras]
postgresql = ["psycopg2-binary"]
mysql = ["pymysql"]
cache = ["redis"]
all = ["psycopg2-binary", "pymysql", "redis"]

[tool.poetry.scripts]
my-cli = "my_package.cli:main"
my-admin = "my_package.admin:run"

[tool.poetry.plugins."my_package.plugins"]
plugin1 = "my_package.plugins.plugin1:Plugin1"

# Build system
[build-system]
requires = ["poetry-core>=1.0.0"]
build-backend = "poetry.core.masonry.api"

# Black configuration
[tool.black]
line-length = 100
target-version = ['py39', 'py310', 'py311']
include = '\.pyi?$'
extend-exclude = '''
/(
  # directories
  \.eggs
  | \.git
  | \.hg
  | \.mypy_cache
  | \.tox
  | \.venv
  | build
  | dist
)/
'''

# MyPy configuration
[tool.mypy]
python_version = "3.9"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true

# Pytest configuration
[tool.pytest.ini_options]
minversion = "7.0"
addopts = "-ra -q --strict-markers --cov=my_package"
testpaths = ["tests"]
markers = [
    "slow: marks tests as slow",
    "integration: marks tests as integration tests",
]

# Coverage configuration
[tool.coverage.run]
source = ["my_package"]
omit = ["*/tests/*", "*/test_*.py"]

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "raise AssertionError",
    "raise NotImplementedError",
    "if __name__ == .__main__.:",
    "if TYPE_CHECKING:",
]
```

### Poetry Advanced Usage

```python
# Poetry ile dependency resolution
"""
Poetry, bağımlılıkları çözmek için SAT solver kullanır:
1. Tüm bağımlılıkları topla
2. Version constraints'leri çöz
3. Conflict varsa rapor et
4. poetry.lock dosyası oluştur
"""

# Poetry plugin sistemi
"""
# Poetry plugin oluşturma
from poetry.plugins.application_plugin import ApplicationPlugin
from poetry.console.commands.command import Command

class MyPlugin(ApplicationPlugin):
    def activate(self, application):
        application.command_loader.register_factory(
            "my-command", MyCommand
        )

class MyCommand(Command):
    name = "my-command"
    description = "My custom command"

    def handle(self):
        self.line("Hello from my plugin!")
"""

# Poetry scripts kullanımı
"""
# pyproject.toml içinde:
[tool.poetry.scripts]
serve = "my_package.server:run"
migrate = "my_package.db:migrate"

# Komut satırından:
poetry run serve
poetry run migrate
"""
```

## Package Creation

### Modern Package Structure

```
my-package/
├── pyproject.toml
├── README.md
├── LICENSE
├── .gitignore
├── src/
│   └── my_package/
│       ├── __init__.py
│       ├── __main__.py
│       ├── core/
│       │   ├── __init__.py
│       │   └── module.py
│       ├── utils/
│       │   ├── __init__.py
│       │   └── helpers.py
│       └── py.typed
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   └── test_module.py
├── docs/
│   ├── conf.py
│   └── index.rst
└── examples/
    └── basic_usage.py
```

### Package __init__.py

```python
# src/my_package/__init__.py
"""
My Awesome Package

A comprehensive package for doing awesome things.
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "you@example.com"
__license__ = "MIT"

# Public API
from my_package.core.module import (
    AwesomeClass,
    awesome_function,
)
from my_package.utils.helpers import (
    helper_function,
)

# __all__ ile public API'yi kontrol et
__all__ = [
    # Core
    "AwesomeClass",
    "awesome_function",
    # Utils
    "helper_function",
    # Metadata
    "__version__",
    "__author__",
]

# Lazy imports (performance için)
def __getattr__(name):
    """Lazy import for heavy modules"""
    if name == "HeavyClass":
        from my_package.heavy import HeavyClass
        return HeavyClass
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

# Deprecation warnings
import warnings

def deprecated_function():
    """Eski fonksiyon - kullanmayın"""
    warnings.warn(
        "deprecated_function is deprecated, use new_function instead",
        DeprecationWarning,
        stacklevel=2
    )
    return new_function()
```

### setup.py (Legacy, backward compatibility için)

```python
# setup.py
from setuptools import setup, find_packages
from pathlib import Path

# README.md'yi oku
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

# Requirements'ı oku
def read_requirements(filename):
    """Requirements dosyasını oku"""
    with open(filename) as f:
        return [
            line.strip()
            for line in f
            if line.strip() and not line.startswith('#')
        ]

setup(
    name="my-awesome-package",
    version="0.1.0",
    author="Your Name",
    author_email="you@example.com",
    description="An awesome Python package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/user/my-awesome-package",
    project_urls={
        "Bug Tracker": "https://github.com/user/my-awesome-package/issues",
        "Documentation": "https://my-awesome-package.readthedocs.io",
        "Source Code": "https://github.com/user/my-awesome-package",
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=read_requirements("requirements.txt"),
    extras_require={
        "dev": read_requirements("requirements-dev.txt"),
        "docs": ["sphinx>=4.0", "sphinx-rtd-theme>=1.0"],
        "test": ["pytest>=7.0", "pytest-cov>=4.0"],
    },
    entry_points={
        "console_scripts": [
            "my-cli=my_package.cli:main",
        ],
        "my_package.plugins": [
            "plugin1=my_package.plugins:Plugin1",
        ],
    },
    include_package_data=True,
    package_data={
        "my_package": ["data/*.json", "templates/*.html"],
    },
    zip_safe=False,
)
```

### setup.cfg (Declarative configuration)

```ini
# setup.cfg
[metadata]
name = my-awesome-package
version = attr: my_package.__version__
author = Your Name
author_email = you@example.com
description = An awesome Python package
long_description = file: README.md
long_description_content_type = text/markdown
url = https://github.com/user/my-awesome-package
project_urls =
    Bug Tracker = https://github.com/user/my-awesome-package/issues
    Documentation = https://my-awesome-package.readthedocs.io
classifiers =
    Development Status :: 4 - Beta
    Intended Audience :: Developers
    License :: OSI Approved :: MIT License
    Programming Language :: Python :: 3
    Programming Language :: Python :: 3.9
    Programming Language :: Python :: 3.10
    Programming Language :: Python :: 3.11

[options]
packages = find:
package_dir =
    = src
python_requires = >=3.9
install_requires =
    requests>=2.28.0
    pydantic>=2.0.0
include_package_data = True
zip_safe = False

[options.packages.find]
where = src

[options.extras_require]
dev =
    pytest>=7.0
    black>=23.0
    mypy>=1.0
docs =
    sphinx>=4.0
    sphinx-rtd-theme>=1.0

[options.entry_points]
console_scripts =
    my-cli = my_package.cli:main

[options.package_data]
my_package = data/*.json, templates/*.html

# Tool configurations
[flake8]
max-line-length = 100
exclude = .git,__pycache__,build,dist
ignore = E203,W503

[mypy]
python_version = 3.9
warn_return_any = True
warn_unused_configs = True
```

### MANIFEST.in (Include non-Python files)

```
# MANIFEST.in
include README.md
include LICENSE
include requirements.txt
include requirements-dev.txt

recursive-include src/my_package/data *
recursive-include src/my_package/templates *
recursive-include docs *.rst *.py

exclude .gitignore
exclude .pre-commit-config.yaml
recursive-exclude * __pycache__
recursive-exclude * *.py[co]
```

## Publishing to PyPI

### Build ve Upload Process

```bash
# 1. Build distributions
poetry build
# veya
python -m build

# 2. Distributions'ı kontrol et
twine check dist/*

# 3. TestPyPI'a yükle (test için)
poetry config repositories.testpypi https://test.pypi.org/legacy/
poetry publish -r testpypi
# veya
twine upload --repository testpypi dist/*

# 4. TestPyPI'dan yükleyerek test et
pip install --index-url https://test.pypi.org/simple/ my-package

# 5. Gerçek PyPI'a yükle
poetry publish
# veya
twine upload dist/*

# 6. API token kullanımı (güvenli)
poetry config pypi-token.pypi <your-token>
# veya .pypirc dosyası:
"""
[pypi]
username = __token__
password = <your-token>

[testpypi]
username = __token__
password = <your-test-token>
"""
```

### Automated Publishing (GitHub Actions)

```yaml
# .github/workflows/publish.yml
name: Publish to PyPI

on:
  release:
    types: [published]

jobs:
  publish:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v3

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'

    - name: Install Poetry
      uses: snok/install-poetry@v1
      with:
        virtualenvs-create: true
        virtualenvs-in-project: true

    - name: Load cached venv
      id: cached-poetry-dependencies
      uses: actions/cache@v3
      with:
        path: .venv
        key: venv-${{ runner.os }}-${{ hashFiles('**/poetry.lock') }}

    - name: Install dependencies
      if: steps.cached-poetry-dependencies.outputs.cache-hit != 'true'
      run: poetry install --no-interaction --no-root

    - name: Install project
      run: poetry install --no-interaction

    - name: Run tests
      run: poetry run pytest

    - name: Build package
      run: poetry build

    - name: Publish to PyPI
      env:
        POETRY_PYPI_TOKEN_PYPI: ${{ secrets.PYPI_TOKEN }}
      run: poetry publish
```

## Semantic Versioning

### Version Numbering Scheme

```
MAJOR.MINOR.PATCH[-PRERELEASE][+BUILDMETADATA]

Örnekler:
1.0.0         - İlk stabil release
1.0.1         - Patch release (bug fixes)
1.1.0         - Minor release (yeni özellikler, backward compatible)
2.0.0         - Major release (breaking changes)
1.0.0-alpha.1 - Pre-release
1.0.0-beta.2  - Beta release
1.0.0-rc.1    - Release candidate
1.0.0+20230615  - Build metadata

Version Constraints:
^1.2.3   ->  >=1.2.3, <2.0.0  (Compatible release)
~1.2.3   ->  >=1.2.3, <1.3.0  (Patch updates only)
>=1.2.3  ->  >=1.2.3           (Minimum version)
==1.2.3  ->  ==1.2.3           (Exact version)
!=1.2.3  ->  !=1.2.3           (Exclude version)
1.2.*    ->  >=1.2.0, <1.3.0  (Wildcard)
```

### Version Management in Code

```python
# Versioning best practices
import re
from pathlib import Path
from typing import Tuple

class Version:
    """Semantic versioning implementation"""

    VERSION_PATTERN = re.compile(
        r'^(?P<major>0|[1-9]\d*)\.'
        r'(?P<minor>0|[1-9]\d*)\.'
        r'(?P<patch>0|[1-9]\d*)'
        r'(?:-(?P<prerelease>(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*)'
        r'(?:\.(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*))*))?'
        r'(?:\+(?P<buildmetadata>[0-9a-zA-Z-]+(?:\.[0-9a-zA-Z-]+)*))?$'
    )

    def __init__(self, version_string: str):
        match = self.VERSION_PATTERN.match(version_string)
        if not match:
            raise ValueError(f"Invalid version: {version_string}")

        self.major = int(match.group('major'))
        self.minor = int(match.group('minor'))
        self.patch = int(match.group('patch'))
        self.prerelease = match.group('prerelease')
        self.buildmetadata = match.group('buildmetadata')

    def __str__(self):
        version = f"{self.major}.{self.minor}.{self.patch}"
        if self.prerelease:
            version += f"-{self.prerelease}"
        if self.buildmetadata:
            version += f"+{self.buildmetadata}"
        return version

    def __repr__(self):
        return f"Version('{self}')"

    def __eq__(self, other):
        if not isinstance(other, Version):
            other = Version(str(other))
        return (
            self.major == other.major and
            self.minor == other.minor and
            self.patch == other.patch and
            self.prerelease == other.prerelease
        )

    def __lt__(self, other):
        if not isinstance(other, Version):
            other = Version(str(other))

        # Compare major.minor.patch
        if (self.major, self.minor, self.patch) != (other.major, other.minor, other.patch):
            return (self.major, self.minor, self.patch) < (other.major, other.minor, other.patch)

        # Pre-release versions are lower than release versions
        if self.prerelease and not other.prerelease:
            return True
        if not self.prerelease and other.prerelease:
            return False
        if self.prerelease and other.prerelease:
            return self.prerelease < other.prerelease

        return False

    def __le__(self, other):
        return self == other or self < other

    def __gt__(self, other):
        return not self <= other

    def __ge__(self, other):
        return not self < other

    def bump_major(self) -> 'Version':
        """Major version artır (breaking changes)"""
        return Version(f"{self.major + 1}.0.0")

    def bump_minor(self) -> 'Version':
        """Minor version artır (yeni özellikler)"""
        return Version(f"{self.major}.{self.minor + 1}.0")

    def bump_patch(self) -> 'Version':
        """Patch version artır (bug fixes)"""
        return Version(f"{self.major}.{self.minor}.{self.patch + 1}")

    def is_prerelease(self) -> bool:
        """Pre-release mi?"""
        return self.prerelease is not None

# Kullanım
v1 = Version("1.2.3")
v2 = Version("1.2.4")
v3 = Version("2.0.0-alpha.1")

print(v1 < v2)  # True
print(v3.is_prerelease())  # True
print(v1.bump_minor())  # 1.3.0
```

## Dependency Management

### Dependency Resolution

```python
# Dependency graph ve conflict resolution
from typing import Dict, List, Set, Optional
from dataclasses import dataclass
from collections import defaultdict, deque

@dataclass
class PackageVersion:
    """Paket versiyonu"""
    name: str
    version: str
    dependencies: Dict[str, str]  # {package_name: version_constraint}

    def __hash__(self):
        return hash((self.name, self.version))

    def __eq__(self, other):
        return self.name == other.name and self.version == other.version

class DependencyResolver:
    """Dependency conflict çözücü"""

    def __init__(self):
        self.package_registry: Dict[str, List[PackageVersion]] = defaultdict(list)

    def register_package(self, package: PackageVersion):
        """Paketi registry'e ekle"""
        self.package_registry[package.name].append(package)

    def resolve_dependencies(
        self,
        requirements: Dict[str, str]
    ) -> Optional[Dict[str, PackageVersion]]:
        """
        Bağımlılıkları çöz

        Returns:
            Çözüm bulunduysa {package_name: PackageVersion} dict
            Çözüm bulunamadıysa None
        """
        # Backtracking ile dependency resolution
        solution = {}

        def is_compatible(version: str, constraint: str) -> bool:
            """Version constraint'i kontrol et"""
            # Basitleştirilmiş version check
            if constraint.startswith('^'):
                # Compatible release: ^1.2.3 -> >=1.2.3, <2.0.0
                base = constraint[1:]
                v = Version(version)
                b = Version(base)
                return v >= b and v.major == b.major
            elif constraint.startswith('~'):
                # Patch updates: ~1.2.3 -> >=1.2.3, <1.3.0
                base = constraint[1:]
                v = Version(version)
                b = Version(base)
                return (v >= b and
                       v.major == b.major and
                       v.minor == b.minor)
            elif constraint.startswith('>='):
                return Version(version) >= Version(constraint[2:])
            elif constraint.startswith('=='):
                return version == constraint[2:]
            else:
                return True

        def backtrack(remaining_requirements: Dict[str, str]) -> bool:
            """Backtracking ile çözüm ara"""
            if not remaining_requirements:
                return True  # Tüm requirements çözüldü

            # Bir requirement seç
            package_name = next(iter(remaining_requirements))
            constraint = remaining_requirements[package_name]

            # Bu package için uygun versiyonları dene
            for package_version in self.package_registry.get(package_name, []):
                if not is_compatible(package_version.version, constraint):
                    continue

                # Bu versiyonu dene
                solution[package_name] = package_version

                # Yeni requirements ekle
                new_requirements = remaining_requirements.copy()
                del new_requirements[package_name]

                # Bu package'ın dependencies'ini ekle
                for dep_name, dep_constraint in package_version.dependencies.items():
                    if dep_name in solution:
                        # Conflict kontrolü
                        if not is_compatible(
                            solution[dep_name].version,
                            dep_constraint
                        ):
                            continue  # Conflict var, bu versiyonu atla
                    else:
                        new_requirements[dep_name] = dep_constraint

                # Recursive olarak devam et
                if backtrack(new_requirements):
                    return True

                # Bu versiyon çalışmadı, geri al
                del solution[package_name]

            return False  # Hiçbir versiyon çalışmadı

        if backtrack(requirements):
            return solution
        return None

# Kullanım örneği
resolver = DependencyResolver()

# Paketleri kaydet
resolver.register_package(PackageVersion(
    "requests", "2.28.0",
    {"urllib3": "^1.26.0", "certifi": ">=2022.0.0"}
))
resolver.register_package(PackageVersion(
    "requests", "2.29.0",
    {"urllib3": "^1.26.0", "certifi": ">=2023.0.0"}
))
resolver.register_package(PackageVersion(
    "urllib3", "1.26.15",
    {}
))
resolver.register_package(PackageVersion(
    "certifi", "2023.5.7",
    {}
))

# Dependencies'i çöz
requirements = {
    "requests": "^2.28.0",
}

solution = resolver.resolve_dependencies(requirements)
if solution:
    print("Çözüm bulundu:")
    for name, pkg in solution.items():
        print(f"  {name}=={pkg.version}")
else:
    print("Çözüm bulunamadı (conflict var)")
```

### Lock Files

```python
# Lock file yönetimi
import json
import hashlib
from typing import Dict, Any
from pathlib import Path
from datetime import datetime

class LockFile:
    """Lock file yöneticisi (poetry.lock benzeri)"""

    def __init__(self, lock_file_path: Path):
        self.lock_file_path = lock_file_path
        self.metadata = {
            "lock-version": "2.0",
            "python-versions": "^3.9",
            "content-hash": "",
        }
        self.packages = []

    def add_package(
        self,
        name: str,
        version: str,
        dependencies: Dict[str, str],
        source: str = "pypi",
        **extras
    ):
        """Lock file'a paket ekle"""
        package_info = {
            "name": name,
            "version": version,
            "description": extras.get("description", ""),
            "category": extras.get("category", "main"),
            "optional": extras.get("optional", False),
            "python-versions": extras.get("python-versions", "*"),
            "dependencies": dependencies,
            "source": {
                "type": source,
                "url": extras.get("url", "https://pypi.org/simple"),
                "reference": extras.get("reference", ""),
            },
        }

        # Hash ekle
        package_hash = self._compute_package_hash(package_info)
        package_info["hash"] = package_hash

        self.packages.append(package_info)

    def _compute_package_hash(self, package_info: Dict) -> str:
        """Paket için hash hesapla"""
        # Deterministic serialization
        content = json.dumps(package_info, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()

    def compute_content_hash(self, requirements: Dict[str, str]) -> str:
        """Tüm requirements için hash hesapla"""
        content = json.dumps(requirements, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()

    def save(self):
        """Lock file'ı kaydet"""
        # Content hash hesapla
        all_requirements = {}
        for pkg in self.packages:
            all_requirements[pkg["name"]] = pkg["version"]

        self.metadata["content-hash"] = self.compute_content_hash(all_requirements)
        self.metadata["generated-at"] = datetime.utcnow().isoformat()

        lock_data = {
            "metadata": self.metadata,
            "package": self.packages,
        }

        with open(self.lock_file_path, 'w') as f:
            # TOML yerine JSON kullanıyoruz (örnek için)
            json.dump(lock_data, f, indent=2)

    @classmethod
    def load(cls, lock_file_path: Path) -> 'LockFile':
        """Lock file'ı yükle"""
        lock_file = cls(lock_file_path)

        if lock_file_path.exists():
            with open(lock_file_path) as f:
                lock_data = json.load(f)

            lock_file.metadata = lock_data.get("metadata", {})
            lock_file.packages = lock_data.get("package", [])

        return lock_file

    def verify_integrity(self, requirements: Dict[str, str]) -> bool:
        """Lock file integrity kontrolü"""
        expected_hash = self.compute_content_hash(requirements)
        return self.metadata.get("content-hash") == expected_hash

# Kullanım
lock_file = LockFile(Path("package.lock"))
lock_file.add_package(
    "requests",
    "2.28.0",
    {"urllib3": "^1.26.0", "certifi": ">=2022.0.0"},
    description="HTTP library",
    category="main",
)
lock_file.save()

# Verify
loaded_lock = LockFile.load(Path("package.lock"))
is_valid = loaded_lock.verify_integrity({"requests": "^2.28.0"})
print(f"Lock file valid: {is_valid}")
```

## Production Best Practices

### 1. Reproducible Builds

```python
# requirements.txt ile reproducible builds
"""
# requirements.txt (pinned versions)
requests==2.28.0
flask==2.3.0
sqlalchemy==2.0.0

# requirements.in (unpinned)
requests>=2.28.0
flask>=2.3.0
sqlalchemy>=2.0.0

# pip-compile kullanımı (pip-tools)
pip-compile requirements.in -o requirements.txt

# Hash verification için
pip-compile --generate-hashes requirements.in
"""

# Docker ile reproducible environment
"""
FROM python:3.11-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application
COPY . .

# Non-root user
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

CMD ["python", "-m", "my_package"]
"""
```

### 2. Multi-stage Builds

```dockerfile
# Multi-stage Docker build
# Stage 1: Builder
FROM python:3.11-slim as builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Stage 2: Runtime
FROM python:3.11-slim

WORKDIR /app

# Copy only necessary files from builder
COPY --from=builder /root/.local /root/.local
COPY . .

# Make sure scripts in .local are usable
ENV PATH=/root/.local/bin:$PATH

# Run as non-root
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

CMD ["python", "-m", "my_package"]
```

### 3. Environment Management

```python
# .env file management
from pathlib import Path
from typing import Dict, Any
import os

class EnvManager:
    """Environment variable yönetimi"""

    def __init__(self, env_file: Path = Path(".env")):
        self.env_file = env_file
        self.variables: Dict[str, str] = {}

    def load(self):
        """Load environment variables from file"""
        if not self.env_file.exists():
            return

        with open(self.env_file) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue

                if '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")

                    self.variables[key] = value
                    os.environ[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        """Get environment variable"""
        return self.variables.get(key, os.environ.get(key, default))

    def require(self, *keys: str):
        """Require environment variables"""
        missing = [key for key in keys if key not in self.variables]
        if missing:
            raise EnvironmentError(
                f"Missing required environment variables: {', '.join(missing)}"
            )

    def get_typed(self, key: str, type_: type, default: Any = None) -> Any:
        """Get typed environment variable"""
        value = self.get(key)
        if value is None:
            return default

        try:
            if type_ == bool:
                return value.lower() in ('true', '1', 'yes', 'on')
            return type_(value)
        except (ValueError, TypeError):
            return default

# config.py
env = EnvManager()
env.load()

# Required variables
env.require('DATABASE_URL', 'SECRET_KEY')

# Configuration
DATABASE_URL = env.get('DATABASE_URL')
SECRET_KEY = env.get('SECRET_KEY')
DEBUG = env.get_typed('DEBUG', bool, False)
PORT = env.get_typed('PORT', int, 8000)
```

### 4. Package Security

```python
# Security scanning ve vulnerability check
"""
# Safety - bağımlılıklarda güvenlik açıkları
pip install safety
safety check
safety check --json
safety check --file requirements.txt

# Bandit - kod güvenlik analizi
pip install bandit
bandit -r my_package/
bandit -r my_package/ -f json -o security-report.json

# pip-audit - dependency auditing
pip install pip-audit
pip-audit
pip-audit --fix  # Otomatik fix
"""

# Automated security scanning (CI/CD)
"""
# .github/workflows/security.yml
name: Security Scan

on: [push, pull_request]

jobs:
  security:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install safety bandit pip-audit
        if [ -f requirements.txt ]; then pip install -r requirements.txt; fi

    - name: Run Safety
      run: safety check --json

    - name: Run Bandit
      run: bandit -r src/ -f json -o bandit-report.json

    - name: Run pip-audit
      run: pip-audit
"""
```

### 5. Performance Optimization

```python
# Lazy imports ve optimizasyon
import sys
from types import ModuleType
from typing import Any

class LazyLoader(ModuleType):
    """Lazy module loader"""

    def __init__(self, module_name: str):
        super().__init__(module_name)
        self.__module_name = module_name
        self.__loaded = False

    def __getattr__(self, name: str) -> Any:
        if not self.__loaded:
            # Gerçek modülü yükle
            import importlib
            module = importlib.import_module(self.__module_name)

            # Cache et
            self.__dict__.update(module.__dict__)
            self.__loaded = True

        return getattr(self, name)

# Kullanım
# my_package/__init__.py
def __getattr__(name):
    if name == "heavy_module":
        return LazyLoader("my_package.heavy_module")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

# Import optimization
"""
# Kötü
from my_package import heavy_module  # Her zaman yüklenir

# İyi
if TYPE_CHECKING:
    from my_package import heavy_module  # Sadece type checking için
else:
    heavy_module = None  # Runtime'da lazy load
"""
```

## Özet

Package Management modern Python development'ın temelidir:

1. **pip advanced**: requirements.txt, constraints, hash verification
2. **Virtual environments**: venv, virtualenv, pyenv ile izolasyon
3. **Poetry**: Modern dependency management ve packaging
4. **pyproject.toml**: Standardize edilmiş configuration
5. **Package creation**: setup.py, setup.cfg, modern best practices
6. **PyPI publishing**: Build, test, publish workflow
7. **Semantic versioning**: MAJOR.MINOR.PATCH convention
8. **Dependency resolution**: Conflict detection ve çözümü
9. **Lock files**: Reproducible builds
10. **Production practices**: Security, performance, Docker

Modern Python projelerinde Poetry ve pyproject.toml kullanımı önerilir!
