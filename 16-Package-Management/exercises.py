"""
PACKAGE MANAGEMENT - ADVANCED EXERCISES
======================================

Bu dosya, paket yönetimi konusunda ileri seviye alıştırmalar içerir.
Her alıştırma gerçek dünya senaryolarını kapsar.

Konular:
- Poetry ile proje yönetimi
- Dependency resolution
- Version management
- Package creation ve publishing
- Lock file yönetimi
- Security ve optimization

Her alıştırmanın zorluğu ve konusu belirtilmiştir.
"""

from typing import Dict, List, Set, Optional, Any, Tuple
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime
import json
import hashlib
import re
from collections import defaultdict
from enum import Enum


# ============================================================================
# EXERCISE 1: Poetry Project Manager
# Zorluk: Medium
# Konu: Poetry pyproject.toml yönetimi ve bağımlılık ekleme
# ============================================================================

class DependencyGroup(Enum):
    """Bağımlılık grupları"""
    MAIN = "main"
    DEV = "dev"
    DOCS = "docs"
    TEST = "test"


@dataclass
class Dependency:
    """Paket bağımlılığı"""
    name: str
    version: str
    optional: bool = False
    extras: List[str] = field(default_factory=list)
    python: Optional[str] = None
    platform: Optional[str] = None
    markers: Optional[str] = None


class PoetryProjectManager:
    """
    Poetry projesi yöneticisi

    TODO: Bu sınıfı implement edin
    - __init__: Proje bilgilerini al
    - add_dependency: Bağımlılık ekle
    - remove_dependency: Bağımlılık kaldır
    - get_dependencies: Grup bazında bağımlılıkları getir
    - generate_pyproject_toml: pyproject.toml içeriği oluştur
    - add_script: Console script ekle
    - add_extra: Extra bağımlılık grubu ekle
    """

    def __init__(
        self,
        name: str,
        version: str,
        description: str,
        authors: List[str],
        python_version: str = "^3.9"
    ):
        # TODO: Initialize edilecek
        pass

    def add_dependency(
        self,
        dependency: Dependency,
        group: DependencyGroup = DependencyGroup.MAIN
    ):
        """Bağımlılık ekle"""
        # TODO: Implement
        pass

    def remove_dependency(self, name: str, group: DependencyGroup = DependencyGroup.MAIN):
        """Bağımlılık kaldır"""
        # TODO: Implement
        pass

    def get_dependencies(self, group: DependencyGroup) -> List[Dependency]:
        """Grup bazında bağımlılıkları getir"""
        # TODO: Implement
        pass

    def add_script(self, name: str, entry_point: str):
        """Console script ekle"""
        # TODO: Implement
        pass

    def add_extra(self, name: str, dependencies: List[str]):
        """Extra bağımlılık grubu ekle"""
        # TODO: Implement
        pass

    def generate_pyproject_toml(self) -> str:
        """pyproject.toml içeriği oluştur"""
        # TODO: Implement - TOML formatında string döndür
        pass


# SOLUTION
class PoetryProjectManagerSolution:
    """Poetry projesi yöneticisi - Çözüm"""

    def __init__(
        self,
        name: str,
        version: str,
        description: str,
        authors: List[str],
        python_version: str = "^3.9"
    ):
        self.name = name
        self.version = version
        self.description = description
        self.authors = authors
        self.python_version = python_version

        # Bağımlılıklar grup bazında
        self.dependencies: Dict[DependencyGroup, Dict[str, Dependency]] = {
            group: {} for group in DependencyGroup
        }

        # Scripts ve extras
        self.scripts: Dict[str, str] = {}
        self.extras: Dict[str, List[str]] = {}

    def add_dependency(
        self,
        dependency: Dependency,
        group: DependencyGroup = DependencyGroup.MAIN
    ):
        """Bağımlılık ekle"""
        self.dependencies[group][dependency.name] = dependency

    def remove_dependency(self, name: str, group: DependencyGroup = DependencyGroup.MAIN):
        """Bağımlılık kaldır"""
        if name in self.dependencies[group]:
            del self.dependencies[group][name]

    def get_dependencies(self, group: DependencyGroup) -> List[Dependency]:
        """Grup bazında bağımlılıkları getir"""
        return list(self.dependencies[group].values())

    def add_script(self, name: str, entry_point: str):
        """Console script ekle"""
        self.scripts[name] = entry_point

    def add_extra(self, name: str, dependencies: List[str]):
        """Extra bağımlılık grubu ekle"""
        self.extras[name] = dependencies

    def _format_dependency(self, dep: Dependency) -> str:
        """Bağımlılığı TOML formatında string'e çevir"""
        if not any([dep.optional, dep.extras, dep.python, dep.platform, dep.markers]):
            # Basit format: package = "version"
            return f'"{dep.version}"'

        # Karmaşık format: package = {version = "...", ...}
        parts = [f'version = "{dep.version}"']

        if dep.optional:
            parts.append("optional = true")

        if dep.extras:
            extras_str = ", ".join(f'"{e}"' for e in dep.extras)
            parts.append(f"extras = [{extras_str}]")

        if dep.python:
            parts.append(f'python = "{dep.python}"')

        if dep.platform:
            parts.append(f'platform = "{dep.platform}"')

        if dep.markers:
            parts.append(f'markers = "{dep.markers}"')

        return "{" + ", ".join(parts) + "}"

    def generate_pyproject_toml(self) -> str:
        """pyproject.toml içeriği oluştur"""
        lines = ["[tool.poetry]"]
        lines.append(f'name = "{self.name}"')
        lines.append(f'version = "{self.version}"')
        lines.append(f'description = "{self.description}"')

        # Authors
        authors_str = ", ".join(f'"{a}"' for a in self.authors)
        lines.append(f"authors = [{authors_str}]")

        lines.append("")

        # Main dependencies
        lines.append("[tool.poetry.dependencies]")
        lines.append(f'python = "{self.python_version}"')

        for name, dep in sorted(self.dependencies[DependencyGroup.MAIN].items()):
            lines.append(f'{name} = {self._format_dependency(dep)}')

        # Dev dependencies
        if self.dependencies[DependencyGroup.DEV]:
            lines.append("")
            lines.append("[tool.poetry.group.dev.dependencies]")
            for name, dep in sorted(self.dependencies[DependencyGroup.DEV].items()):
                lines.append(f'{name} = {self._format_dependency(dep)}')

        # Test dependencies
        if self.dependencies[DependencyGroup.TEST]:
            lines.append("")
            lines.append("[tool.poetry.group.test.dependencies]")
            for name, dep in sorted(self.dependencies[DependencyGroup.TEST].items()):
                lines.append(f'{name} = {self._format_dependency(dep)}')

        # Docs dependencies
        if self.dependencies[DependencyGroup.DOCS]:
            lines.append("")
            lines.append("[tool.poetry.group.docs.dependencies]")
            for name, dep in sorted(self.dependencies[DependencyGroup.DOCS].items()):
                lines.append(f'{name} = {self._format_dependency(dep)}')

        # Extras
        if self.extras:
            lines.append("")
            lines.append("[tool.poetry.extras]")
            for name, deps in sorted(self.extras.items()):
                deps_str = ", ".join(f'"{d}"' for d in deps)
                lines.append(f'{name} = [{deps_str}]')

        # Scripts
        if self.scripts:
            lines.append("")
            lines.append("[tool.poetry.scripts]")
            for name, entry_point in sorted(self.scripts.items()):
                lines.append(f'{name} = "{entry_point}"')

        # Build system
        lines.append("")
        lines.append("[build-system]")
        lines.append('requires = ["poetry-core>=1.0.0"]')
        lines.append('build-backend = "poetry.core.masonry.api"')

        return "\n".join(lines)


# ============================================================================
# EXERCISE 2: Semantic Version Parser and Comparator
# Zorluk: Medium
# Konu: Semantic versioning implementation
# ============================================================================

class SemanticVersion:
    """
    Semantic versioning implementation

    TODO: Bu sınıfı implement edin
    - __init__: Version string'i parse et
    - Comparison operators: __eq__, __lt__, __le__, __gt__, __ge__
    - bump_major, bump_minor, bump_patch: Version artırma
    - is_compatible: Version uyumluluğu kontrolü
    - satisfies: Version constraint kontrolü
    """

    VERSION_PATTERN = re.compile(
        r'^(?P<major>0|[1-9]\d*)\.'
        r'(?P<minor>0|[1-9]\d*)\.'
        r'(?P<patch>0|[1-9]\d*)'
        r'(?:-(?P<prerelease>(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*)'
        r'(?:\.(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*))*))?'
        r'(?:\+(?P<buildmetadata>[0-9a-zA-Z-]+(?:\.[0-9a-zA-Z-]+)*))?$'
    )

    def __init__(self, version_string: str):
        # TODO: Parse version string
        pass

    def __str__(self) -> str:
        # TODO: Implement
        pass

    def __eq__(self, other) -> bool:
        # TODO: Implement
        pass

    def __lt__(self, other) -> bool:
        # TODO: Implement
        pass

    def bump_major(self) -> 'SemanticVersion':
        # TODO: Implement
        pass

    def bump_minor(self) -> 'SemanticVersion':
        # TODO: Implement
        pass

    def bump_patch(self) -> 'SemanticVersion':
        # TODO: Implement
        pass

    def satisfies(self, constraint: str) -> bool:
        """
        Version constraint'i kontrol et
        Desteklenen formatlar:
        - ^1.2.3: >=1.2.3, <2.0.0
        - ~1.2.3: >=1.2.3, <1.3.0
        - >=1.2.3: Minimum version
        - ==1.2.3: Exact version
        """
        # TODO: Implement
        pass


# SOLUTION
class SemanticVersionSolution:
    """Semantic versioning implementation - Çözüm"""

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

    def __str__(self) -> str:
        version = f"{self.major}.{self.minor}.{self.patch}"
        if self.prerelease:
            version += f"-{self.prerelease}"
        if self.buildmetadata:
            version += f"+{self.buildmetadata}"
        return version

    def __repr__(self):
        return f"SemanticVersion('{self}')"

    def __eq__(self, other) -> bool:
        if not isinstance(other, SemanticVersionSolution):
            other = SemanticVersionSolution(str(other))
        return (
            self.major == other.major and
            self.minor == other.minor and
            self.patch == other.patch and
            self.prerelease == other.prerelease
        )

    def __lt__(self, other) -> bool:
        if not isinstance(other, SemanticVersionSolution):
            other = SemanticVersionSolution(str(other))

        # Compare major.minor.patch
        if (self.major, self.minor, self.patch) != (other.major, other.minor, other.patch):
            return (self.major, self.minor, self.patch) < (other.major, other.minor, other.patch)

        # Pre-release versions are lower than release versions
        if self.prerelease and not other.prerelease:
            return True
        if not self.prerelease and other.prerelease:
            return False
        if self.prerelease and other.prerelease:
            return self._compare_prerelease(self.prerelease, other.prerelease) < 0

        return False

    def __le__(self, other) -> bool:
        return self == other or self < other

    def __gt__(self, other) -> bool:
        return not self <= other

    def __ge__(self, other) -> bool:
        return not self < other

    def _compare_prerelease(self, pre1: str, pre2: str) -> int:
        """Pre-release versiyonları karşılaştır"""
        parts1 = pre1.split('.')
        parts2 = pre2.split('.')

        for p1, p2 in zip(parts1, parts2):
            # Numeric comparison
            try:
                n1, n2 = int(p1), int(p2)
                if n1 != n2:
                    return -1 if n1 < n2 else 1
            except ValueError:
                # String comparison
                if p1 != p2:
                    return -1 if p1 < p2 else 1

        # Longer pre-release is greater
        return len(parts1) - len(parts2)

    def bump_major(self) -> 'SemanticVersionSolution':
        """Major version artır"""
        return SemanticVersionSolution(f"{self.major + 1}.0.0")

    def bump_minor(self) -> 'SemanticVersionSolution':
        """Minor version artır"""
        return SemanticVersionSolution(f"{self.major}.{self.minor + 1}.0")

    def bump_patch(self) -> 'SemanticVersionSolution':
        """Patch version artır"""
        return SemanticVersionSolution(f"{self.major}.{self.minor}.{self.patch + 1}")

    def satisfies(self, constraint: str) -> bool:
        """Version constraint'i kontrol et"""
        constraint = constraint.strip()

        if constraint.startswith('^'):
            # Compatible release: ^1.2.3 -> >=1.2.3, <2.0.0
            base = SemanticVersionSolution(constraint[1:])
            return self >= base and self.major == base.major

        elif constraint.startswith('~'):
            # Patch updates: ~1.2.3 -> >=1.2.3, <1.3.0
            base = SemanticVersionSolution(constraint[1:])
            return (self >= base and
                   self.major == base.major and
                   self.minor == base.minor)

        elif constraint.startswith('>='):
            base = SemanticVersionSolution(constraint[2:].strip())
            return self >= base

        elif constraint.startswith('>'):
            base = SemanticVersionSolution(constraint[1:].strip())
            return self > base

        elif constraint.startswith('<='):
            base = SemanticVersionSolution(constraint[2:].strip())
            return self <= base

        elif constraint.startswith('<'):
            base = SemanticVersionSolution(constraint[1:].strip())
            return self < base

        elif constraint.startswith('=='):
            base = SemanticVersionSolution(constraint[2:].strip())
            return self == base

        elif constraint.startswith('!='):
            base = SemanticVersionSolution(constraint[2:].strip())
            return self != base

        else:
            # Exact match
            base = SemanticVersionSolution(constraint)
            return self == base


# ============================================================================
# EXERCISE 3: Dependency Graph Builder and Resolver
# Zorluk: Hard
# Konu: Dependency resolution ve conflict detection
# ============================================================================

@dataclass
class PackageVersion:
    """Paket versiyonu"""
    name: str
    version: str
    dependencies: Dict[str, str]  # {package_name: version_constraint}

    def __hash__(self):
        return hash((self.name, self.version))


class DependencyResolver:
    """
    Dependency resolver

    TODO: Bu sınıfı implement edin
    - register_package: Paketi registry'e ekle
    - resolve_dependencies: Bağımlılıkları çöz
    - detect_conflicts: Conflict'leri tespit et
    - get_dependency_graph: Dependency graph'ı oluştur
    - find_circular_dependencies: Circular dependency tespit et
    """

    def __init__(self):
        # TODO: Initialize
        pass

    def register_package(self, package: PackageVersion):
        """Paketi registry'e ekle"""
        # TODO: Implement
        pass

    def resolve_dependencies(
        self,
        requirements: Dict[str, str]
    ) -> Optional[Dict[str, PackageVersion]]:
        """Bağımlılıkları çöz"""
        # TODO: Implement - backtracking algoritması kullan
        pass

    def detect_conflicts(
        self,
        requirements: Dict[str, str]
    ) -> List[Tuple[str, List[str]]]:
        """Conflict'leri tespit et"""
        # TODO: Implement
        pass

    def get_dependency_graph(
        self,
        package_name: str,
        version: str
    ) -> Dict[str, Set[str]]:
        """Dependency graph'ı oluştur"""
        # TODO: Implement
        pass

    def find_circular_dependencies(
        self,
        package_name: str,
        version: str
    ) -> List[List[str]]:
        """Circular dependency tespit et"""
        # TODO: Implement
        pass


# SOLUTION
class DependencyResolverSolution:
    """Dependency resolver - Çözüm"""

    def __init__(self):
        self.package_registry: Dict[str, List[PackageVersion]] = defaultdict(list)

    def register_package(self, package: PackageVersion):
        """Paketi registry'e ekle"""
        self.package_registry[package.name].append(package)

    def _is_compatible(self, version: str, constraint: str) -> bool:
        """Version constraint'i kontrol et"""
        try:
            v = SemanticVersionSolution(version)
            return v.satisfies(constraint)
        except:
            # Fallback basit kontrol
            return version == constraint

    def resolve_dependencies(
        self,
        requirements: Dict[str, str]
    ) -> Optional[Dict[str, PackageVersion]]:
        """Bağımlılıkları çöz"""
        solution = {}

        def backtrack(remaining_requirements: Dict[str, str]) -> bool:
            if not remaining_requirements:
                return True

            # Bir requirement seç
            package_name = next(iter(remaining_requirements))
            constraint = remaining_requirements[package_name]

            # Bu package için uygun versiyonları dene
            for package_version in self.package_registry.get(package_name, []):
                if not self._is_compatible(package_version.version, constraint):
                    continue

                # Bu versiyonu dene
                solution[package_name] = package_version

                # Yeni requirements ekle
                new_requirements = remaining_requirements.copy()
                del new_requirements[package_name]

                # Bu package'ın dependencies'ini ekle
                conflict = False
                for dep_name, dep_constraint in package_version.dependencies.items():
                    if dep_name in solution:
                        # Conflict kontrolü
                        if not self._is_compatible(
                            solution[dep_name].version,
                            dep_constraint
                        ):
                            conflict = True
                            break
                    else:
                        # Yeni requirement ekle
                        if dep_name in new_requirements:
                            # Merge constraints (basitleştirilmiş)
                            new_requirements[dep_name] = dep_constraint
                        else:
                            new_requirements[dep_name] = dep_constraint

                if conflict:
                    del solution[package_name]
                    continue

                # Recursive olarak devam et
                if backtrack(new_requirements):
                    return True

                # Bu versiyon çalışmadı, geri al
                del solution[package_name]

            return False

        if backtrack(requirements):
            return solution
        return None

    def detect_conflicts(
        self,
        requirements: Dict[str, str]
    ) -> List[Tuple[str, List[str]]]:
        """Conflict'leri tespit et"""
        conflicts = []

        # Her package için tüm bağımlılıkları topla
        all_constraints: Dict[str, List[str]] = defaultdict(list)

        def collect_constraints(pkg_name: str, constraint: str):
            all_constraints[pkg_name].append(constraint)

            # Bu constraint'i sağlayan versiyonları bul
            for pkg in self.package_registry.get(pkg_name, []):
                if self._is_compatible(pkg.version, constraint):
                    # Bu version'ın bağımlılıklarını da ekle
                    for dep_name, dep_constraint in pkg.dependencies.items():
                        all_constraints[dep_name].append(dep_constraint)

        for pkg_name, constraint in requirements.items():
            collect_constraints(pkg_name, constraint)

        # Conflict kontrolü
        for pkg_name, constraints in all_constraints.items():
            if len(constraints) > 1:
                # Birden fazla constraint var, uyumlu mu?
                compatible_versions = None

                for constraint in constraints:
                    versions = {
                        pkg.version
                        for pkg in self.package_registry.get(pkg_name, [])
                        if self._is_compatible(pkg.version, constraint)
                    }

                    if compatible_versions is None:
                        compatible_versions = versions
                    else:
                        compatible_versions &= versions

                if not compatible_versions:
                    conflicts.append((pkg_name, constraints))

        return conflicts

    def get_dependency_graph(
        self,
        package_name: str,
        version: str
    ) -> Dict[str, Set[str]]:
        """Dependency graph'ı oluştur"""
        graph: Dict[str, Set[str]] = defaultdict(set)
        visited = set()

        def build_graph(pkg_name: str, pkg_version: str):
            key = f"{pkg_name}=={pkg_version}"
            if key in visited:
                return
            visited.add(key)

            # Package'ı bul
            for pkg in self.package_registry.get(pkg_name, []):
                if pkg.version == pkg_version:
                    # Dependencies ekle
                    for dep_name, dep_constraint in pkg.dependencies.items():
                        graph[key].add(dep_name)

                        # Recursive olarak dependency'leri ekle
                        for dep_pkg in self.package_registry.get(dep_name, []):
                            if self._is_compatible(dep_pkg.version, dep_constraint):
                                build_graph(dep_name, dep_pkg.version)
                                break
                    break

        build_graph(package_name, version)
        return dict(graph)

    def find_circular_dependencies(
        self,
        package_name: str,
        version: str
    ) -> List[List[str]]:
        """Circular dependency tespit et"""
        cycles = []

        def dfs(current: str, path: List[str], visited: Set[str]):
            if current in path:
                # Cycle bulundu
                cycle_start = path.index(current)
                cycle = path[cycle_start:] + [current]
                cycles.append(cycle)
                return

            if current in visited:
                return

            visited.add(current)
            path.append(current)

            # Current package'ın dependencies'ini bul
            pkg_name, pkg_version = current.split('==')
            for pkg in self.package_registry.get(pkg_name, []):
                if pkg.version == pkg_version:
                    for dep_name in pkg.dependencies:
                        # Dependency'nin bir versiyonunu seç
                        for dep_pkg in self.package_registry.get(dep_name, []):
                            dep_key = f"{dep_name}=={dep_pkg.version}"
                            dfs(dep_key, path.copy(), visited)
                            break
                    break

        start_key = f"{package_name}=={version}"
        dfs(start_key, [], set())

        return cycles


# ============================================================================
# EXERCISE 4: Lock File Manager
# Zorluk: Medium
# Konu: Lock file creation ve integrity checking
# ============================================================================

class LockFileManager:
    """
    Lock file yöneticisi

    TODO: Bu sınıfı implement edin
    - add_package: Lock file'a paket ekle
    - remove_package: Paketi kaldır
    - compute_content_hash: Content hash hesapla
    - verify_integrity: Integrity kontrolü
    - save: Lock file'ı kaydet
    - load: Lock file'ı yükle
    """

    def __init__(self, lock_file_path: Path):
        # TODO: Initialize
        pass

    def add_package(
        self,
        name: str,
        version: str,
        dependencies: Dict[str, str],
        **metadata
    ):
        """Lock file'a paket ekle"""
        # TODO: Implement
        pass

    def remove_package(self, name: str):
        """Paketi kaldır"""
        # TODO: Implement
        pass

    def compute_content_hash(self) -> str:
        """Content hash hesapla"""
        # TODO: Implement - SHA256 kullan
        pass

    def verify_integrity(self, expected_hash: str) -> bool:
        """Integrity kontrolü"""
        # TODO: Implement
        pass

    def save(self):
        """Lock file'ı kaydet"""
        # TODO: Implement - JSON formatında kaydet
        pass

    @classmethod
    def load(cls, lock_file_path: Path) -> 'LockFileManager':
        """Lock file'ı yükle"""
        # TODO: Implement
        pass


# SOLUTION
class LockFileManagerSolution:
    """Lock file yöneticisi - Çözüm"""

    def __init__(self, lock_file_path: Path):
        self.lock_file_path = lock_file_path
        self.metadata = {
            "lock-version": "2.0",
            "generated-at": datetime.utcnow().isoformat(),
            "content-hash": "",
        }
        self.packages: List[Dict[str, Any]] = []

    def add_package(
        self,
        name: str,
        version: str,
        dependencies: Dict[str, str],
        **metadata
    ):
        """Lock file'a paket ekle"""
        package_info = {
            "name": name,
            "version": version,
            "dependencies": dependencies,
            "description": metadata.get("description", ""),
            "category": metadata.get("category", "main"),
            "optional": metadata.get("optional", False),
            "source": {
                "type": metadata.get("source_type", "pypi"),
                "url": metadata.get("source_url", "https://pypi.org/simple"),
            },
        }

        # Package hash hesapla
        package_hash = self._compute_package_hash(package_info)
        package_info["hash"] = package_hash

        self.packages.append(package_info)

    def remove_package(self, name: str):
        """Paketi kaldır"""
        self.packages = [pkg for pkg in self.packages if pkg["name"] != name]

    def _compute_package_hash(self, package_info: Dict) -> str:
        """Paket için hash hesapla"""
        # Deterministic serialization
        content = json.dumps(package_info, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def compute_content_hash(self) -> str:
        """Content hash hesapla"""
        # Tüm paketleri sıralı şekilde serialize et
        content = json.dumps(
            sorted(self.packages, key=lambda x: x["name"]),
            sort_keys=True
        )
        return hashlib.sha256(content.encode()).hexdigest()

    def verify_integrity(self, expected_hash: str) -> bool:
        """Integrity kontrolü"""
        return self.compute_content_hash() == expected_hash

    def save(self):
        """Lock file'ı kaydet"""
        # Content hash güncelle
        self.metadata["content-hash"] = self.compute_content_hash()
        self.metadata["generated-at"] = datetime.utcnow().isoformat()

        lock_data = {
            "metadata": self.metadata,
            "packages": self.packages,
        }

        with open(self.lock_file_path, 'w') as f:
            json.dump(lock_data, f, indent=2, sort_keys=True)

    @classmethod
    def load(cls, lock_file_path: Path) -> 'LockFileManagerSolution':
        """Lock file'ı yükle"""
        lock_file = cls(lock_file_path)

        if lock_file_path.exists():
            with open(lock_file_path) as f:
                lock_data = json.load(f)

            lock_file.metadata = lock_data.get("metadata", {})
            lock_file.packages = lock_data.get("packages", [])

        return lock_file


# ============================================================================
# EXERCISE 5: Requirements File Parser
# Zorluk: Medium
# Konu: requirements.txt parsing ve normalization
# ============================================================================

@dataclass
class Requirement:
    """Requirement specification"""
    name: str
    version_specs: List[Tuple[str, str]]  # [(operator, version), ...]
    extras: List[str] = field(default_factory=list)
    markers: Optional[str] = None
    url: Optional[str] = None
    is_editable: bool = False
    hash_specs: List[str] = field(default_factory=list)


class RequirementsParser:
    """
    Requirements.txt parser

    TODO: Bu sınıfı implement edin
    - parse_line: Tek satırı parse et
    - parse_file: Dosyayı parse et
    - normalize_requirement: Requirement'ı normalize et
    - generate_constraints: Constraints.txt oluştur
    - merge_requirements: Birden fazla requirements dosyasını birleştir
    """

    def parse_line(self, line: str) -> Optional[Requirement]:
        """Tek satırı parse et"""
        # TODO: Implement
        # Format: package[extras]==1.2.3,>=1.0.0 ; python_version >= "3.9" --hash sha256:...
        pass

    def parse_file(self, filepath: Path) -> List[Requirement]:
        """Dosyayı parse et"""
        # TODO: Implement
        pass

    def normalize_requirement(self, req: Requirement) -> str:
        """Requirement'ı normalize et"""
        # TODO: Implement
        pass

    def generate_constraints(self, requirements: List[Requirement]) -> str:
        """Constraints.txt içeriği oluştur"""
        # TODO: Implement
        pass


# SOLUTION
class RequirementsParserSolution:
    """Requirements.txt parser - Çözüm"""

    # Regex patterns
    NAME_PATTERN = re.compile(r'^([a-zA-Z0-9]([a-zA-Z0-9._-]*[a-zA-Z0-9])?)')
    VERSION_PATTERN = re.compile(
        r'(==|!=|>=|<=|>|<|~=|===)\s*([a-zA-Z0-9._\-\+]+)'
    )

    def parse_line(self, line: str) -> Optional[Requirement]:
        """Tek satırı parse et"""
        # Boş satır ve yorum
        line = line.strip()
        if not line or line.startswith('#'):
            return None

        # -r, -c gibi özel direktifler
        if line.startswith('-'):
            return None

        # Hash specifications
        hash_specs = []
        if '--hash' in line:
            parts = line.split('--hash')
            line = parts[0].strip()
            for part in parts[1:]:
                hash_spec = part.strip().split()[0]
                hash_specs.append(hash_spec)

        # Markers (environment markers)
        markers = None
        if ';' in line:
            line, markers = line.split(';', 1)
            markers = markers.strip()
            line = line.strip()

        # URL specifications
        url = None
        is_editable = False
        if line.startswith('-e '):
            is_editable = True
            line = line[3:].strip()

        if any(line.startswith(scheme) for scheme in ['http://', 'https://', 'git+']):
            url = line
            # Extract package name from URL
            if '#egg=' in line:
                name = line.split('#egg=')[1].split('&')[0]
            else:
                name = Path(line).stem
            return Requirement(
                name=name,
                version_specs=[],
                url=url,
                is_editable=is_editable,
                markers=markers,
                hash_specs=hash_specs
            )

        # Package name
        name_match = self.NAME_PATTERN.match(line)
        if not name_match:
            return None

        name = name_match.group(1)
        rest = line[len(name):].strip()

        # Extras
        extras = []
        if rest.startswith('['):
            end_bracket = rest.find(']')
            extras_str = rest[1:end_bracket]
            extras = [e.strip() for e in extras_str.split(',')]
            rest = rest[end_bracket + 1:].strip()

        # Version specifications
        version_specs = []
        for match in self.VERSION_PATTERN.finditer(rest):
            operator = match.group(1)
            version = match.group(2)
            version_specs.append((operator, version))

        return Requirement(
            name=name,
            version_specs=version_specs,
            extras=extras,
            markers=markers,
            hash_specs=hash_specs,
            is_editable=is_editable
        )

    def parse_file(self, filepath: Path) -> List[Requirement]:
        """Dosyayı parse et"""
        requirements = []

        if not filepath.exists():
            return requirements

        with open(filepath) as f:
            for line in f:
                req = self.parse_line(line)
                if req:
                    requirements.append(req)

        return requirements

    def normalize_requirement(self, req: Requirement) -> str:
        """Requirement'ı normalize et"""
        parts = [req.name]

        # Extras
        if req.extras:
            parts.append(f"[{','.join(req.extras)}]")

        # Version specs
        if req.version_specs:
            version_parts = []
            for op, ver in req.version_specs:
                version_parts.append(f"{op}{ver}")
            parts.append(','.join(version_parts))

        result = ''.join(parts)

        # Markers
        if req.markers:
            result += f" ; {req.markers}"

        # Hashes
        for hash_spec in req.hash_specs:
            result += f" --hash {hash_spec}"

        return result

    def generate_constraints(self, requirements: List[Requirement]) -> str:
        """Constraints.txt içeriği oluştur"""
        lines = ["# Constraints file", ""]

        for req in sorted(requirements, key=lambda r: r.name):
            # Constraints sadece versiyon belirtir, extras vs yok
            if req.version_specs:
                version_parts = []
                for op, ver in req.version_specs:
                    version_parts.append(f"{op}{ver}")
                lines.append(f"{req.name}{','.join(version_parts)}")
            elif req.url:
                # URL-based requirements
                lines.append(f"{req.name} @ {req.url}")

        return '\n'.join(lines)

    def merge_requirements(
        self,
        *requirement_lists: List[Requirement]
    ) -> List[Requirement]:
        """Birden fazla requirements dosyasını birleştir"""
        merged: Dict[str, Requirement] = {}

        for req_list in requirement_lists:
            for req in req_list:
                if req.name not in merged:
                    merged[req.name] = req
                else:
                    # Merge version specs
                    existing = merged[req.name]
                    existing.version_specs.extend(req.version_specs)
                    existing.extras.extend(req.extras)
                    existing.hash_specs.extend(req.hash_specs)

                    # Remove duplicates
                    existing.version_specs = list(set(existing.version_specs))
                    existing.extras = list(set(existing.extras))
                    existing.hash_specs = list(set(existing.hash_specs))

        return list(merged.values())


# ============================================================================
# EXERCISE 6: Package Metadata Extractor
# Zorluk: Medium
# Konu: Package metadata extraction from installed packages
# ============================================================================

@dataclass
class PackageMetadata:
    """Package metadata"""
    name: str
    version: str
    summary: str
    author: str
    license: str
    requires_python: Optional[str]
    dependencies: List[str]
    homepage: Optional[str]
    install_location: Optional[Path]


class PackageInspector:
    """
    Yüklü paketleri inspect eden sınıf

    TODO: Bu sınıfı implement edin
    - get_installed_packages: Yüklü paketleri listele
    - get_package_metadata: Paket metadata'sını al
    - find_outdated_packages: Güncel olmayan paketleri bul
    - generate_requirements_from_env: Environment'tan requirements.txt oluştur
    """

    def get_installed_packages(self) -> List[str]:
        """Yüklü paketleri listele"""
        # TODO: Implement
        # Mock implementation - gerçekte importlib.metadata kullanılır
        pass

    def get_package_metadata(self, package_name: str) -> Optional[PackageMetadata]:
        """Paket metadata'sını al"""
        # TODO: Implement
        pass

    def generate_requirements_from_env(self, include_versions: bool = True) -> str:
        """Environment'tan requirements.txt oluştur"""
        # TODO: Implement
        pass


# SOLUTION
class PackageInspectorSolution:
    """Yüklü paketleri inspect eden sınıf - Çözüm"""

    def __init__(self):
        # Mock package database
        self.mock_packages = {
            "requests": PackageMetadata(
                name="requests",
                version="2.28.0",
                summary="HTTP library for Python",
                author="Kenneth Reitz",
                license="Apache 2.0",
                requires_python=">=3.7",
                dependencies=["urllib3>=1.26.0", "certifi>=2022.0.0"],
                homepage="https://requests.readthedocs.io",
                install_location=Path("/usr/local/lib/python3.9/site-packages/requests")
            ),
            "flask": PackageMetadata(
                name="flask",
                version="2.3.0",
                summary="Web framework",
                author="Armin Ronacher",
                license="BSD-3-Clause",
                requires_python=">=3.8",
                dependencies=["werkzeug>=2.3.0", "jinja2>=3.1.0"],
                homepage="https://flask.palletsprojects.com",
                install_location=Path("/usr/local/lib/python3.9/site-packages/flask")
            ),
        }

    def get_installed_packages(self) -> List[str]:
        """Yüklü paketleri listele"""
        return list(self.mock_packages.keys())

    def get_package_metadata(self, package_name: str) -> Optional[PackageMetadata]:
        """Paket metadata'sını al"""
        return self.mock_packages.get(package_name)

    def find_outdated_packages(
        self,
        latest_versions: Dict[str, str]
    ) -> List[Tuple[str, str, str]]:
        """Güncel olmayan paketleri bul"""
        outdated = []

        for pkg_name, metadata in self.mock_packages.items():
            if pkg_name in latest_versions:
                latest = latest_versions[pkg_name]
                try:
                    current = SemanticVersionSolution(metadata.version)
                    latest_ver = SemanticVersionSolution(latest)

                    if current < latest_ver:
                        outdated.append((pkg_name, metadata.version, latest))
                except:
                    # Version parsing hatası
                    pass

        return outdated

    def generate_requirements_from_env(self, include_versions: bool = True) -> str:
        """Environment'tan requirements.txt oluştur"""
        lines = ["# Requirements generated from environment", ""]

        for pkg_name in sorted(self.mock_packages.keys()):
            metadata = self.mock_packages[pkg_name]

            if include_versions:
                lines.append(f"{pkg_name}=={metadata.version}")
            else:
                lines.append(pkg_name)

        return '\n'.join(lines)

    def get_dependency_tree(self, package_name: str, level: int = 0) -> str:
        """Dependency tree oluştur"""
        metadata = self.get_package_metadata(package_name)
        if not metadata:
            return ""

        indent = "  " * level
        tree = [f"{indent}{package_name} ({metadata.version})"]

        for dep in metadata.dependencies:
            # Parse dependency (basit)
            dep_name = dep.split('[')[0].split('>=')[0].split('==')[0].strip()
            if dep_name in self.mock_packages:
                tree.append(self.get_dependency_tree(dep_name, level + 1))

        return '\n'.join(filter(None, tree))


# ============================================================================
# EXERCISE 7: Virtual Environment Manager
# Zorluk: Medium
# Konu: Programmatic venv creation and management
# ============================================================================

class VirtualEnvManager:
    """
    Virtual environment yöneticisi

    TODO: Bu sınıfı implement edin
    - create_env: Virtual environment oluştur
    - install_packages: Paketleri yükle
    - activate: Environment'ı aktif et
    - deactivate: Environment'ı deaktive et
    - delete_env: Environment'ı sil
    - clone_env: Environment'ı klonla
    """

    def __init__(self, env_path: Path):
        # TODO: Initialize
        pass

    def create_env(self, python_version: Optional[str] = None):
        """Virtual environment oluştur"""
        # TODO: Implement (mock)
        pass

    def install_packages(self, packages: List[str]):
        """Paketleri yükle"""
        # TODO: Implement (mock)
        pass

    def get_python_path(self) -> Path:
        """Python executable path"""
        # TODO: Implement
        pass

    def get_pip_path(self) -> Path:
        """Pip executable path"""
        # TODO: Implement
        pass


# SOLUTION
class VirtualEnvManagerSolution:
    """Virtual environment yöneticisi - Çözüm"""

    def __init__(self, env_path: Path):
        self.env_path = env_path
        self.is_created = False
        self.installed_packages: Set[str] = set()

    def create_env(self, python_version: Optional[str] = None):
        """Virtual environment oluştur"""
        # Mock implementation
        print(f"Creating virtual environment at {self.env_path}")

        if python_version:
            print(f"Using Python {python_version}")

        # Dizinleri oluştur (mock)
        self.env_path.mkdir(parents=True, exist_ok=True)
        (self.env_path / "bin").mkdir(exist_ok=True)
        (self.env_path / "lib").mkdir(exist_ok=True)

        self.is_created = True
        print("Virtual environment created successfully")

    def install_packages(self, packages: List[str]):
        """Paketleri yükle"""
        if not self.is_created:
            raise RuntimeError("Environment not created")

        print(f"Installing packages: {', '.join(packages)}")

        for package in packages:
            # Mock installation
            self.installed_packages.add(package.split('==')[0].split('>=')[0])
            print(f"  ✓ {package}")

        print("All packages installed successfully")

    def get_python_path(self) -> Path:
        """Python executable path"""
        if not self.is_created:
            raise RuntimeError("Environment not created")

        return self.env_path / "bin" / "python"

    def get_pip_path(self) -> Path:
        """Pip executable path"""
        if not self.is_created:
            raise RuntimeError("Environment not created")

        return self.env_path / "bin" / "pip"

    def list_packages(self) -> List[str]:
        """Yüklü paketleri listele"""
        return sorted(self.installed_packages)

    def delete_env(self):
        """Environment'ı sil"""
        print(f"Deleting virtual environment at {self.env_path}")
        self.is_created = False
        self.installed_packages.clear()
        print("Virtual environment deleted")

    def clone_env(self, target_path: Path) -> 'VirtualEnvManagerSolution':
        """Environment'ı klonla"""
        if not self.is_created:
            raise RuntimeError("Source environment not created")

        print(f"Cloning environment to {target_path}")

        # Yeni environment oluştur
        cloned = VirtualEnvManagerSolution(target_path)
        cloned.create_env()

        # Paketleri kopyala
        if self.installed_packages:
            cloned.install_packages(list(self.installed_packages))

        print("Environment cloned successfully")
        return cloned


# ============================================================================
# EXERCISE 8: Build System Manager
# Zorluk: Hard
# Konu: Package building, distribution creation
# ============================================================================

class BuildSystemManager:
    """
    Build system yöneticisi

    TODO: Bu sınıfı implement edin
    - validate_package_structure: Package yapısını validate et
    - build_sdist: Source distribution oluştur
    - build_wheel: Wheel distribution oluştur
    - generate_metadata: METADATA dosyası oluştur
    - validate_distributions: Distribution'ları validate et
    """

    def __init__(self, project_path: Path):
        # TODO: Initialize
        pass

    def validate_package_structure(self) -> Tuple[bool, List[str]]:
        """Package yapısını validate et"""
        # TODO: Implement
        # Check: pyproject.toml, src/, README.md, LICENSE
        pass

    def build_sdist(self, output_dir: Path) -> Path:
        """Source distribution oluştur"""
        # TODO: Implement (mock)
        pass

    def build_wheel(self, output_dir: Path) -> Path:
        """Wheel distribution oluştur"""
        # TODO: Implement (mock)
        pass

    def generate_metadata(self) -> str:
        """METADATA dosyası oluştur"""
        # TODO: Implement
        pass


# SOLUTION
class BuildSystemManagerSolution:
    """Build system yöneticisi - Çözüm"""

    REQUIRED_FILES = [
        "pyproject.toml",
        "README.md",
    ]

    RECOMMENDED_FILES = [
        "LICENSE",
        ".gitignore",
    ]

    def __init__(self, project_path: Path):
        self.project_path = project_path
        self.package_name = project_path.name

    def validate_package_structure(self) -> Tuple[bool, List[str]]:
        """Package yapısını validate et"""
        errors = []

        # Required files kontrolü
        for filename in self.REQUIRED_FILES:
            if not (self.project_path / filename).exists():
                errors.append(f"Missing required file: {filename}")

        # Package directory kontrolü
        src_dir = self.project_path / "src"
        if src_dir.exists():
            # src layout
            package_dirs = list(src_dir.iterdir())
            if not package_dirs:
                errors.append("No package directory found in src/")
        else:
            # Flat layout - package adıyla dizin olmalı
            if not (self.project_path / self.package_name).exists():
                errors.append(f"No package directory found: {self.package_name}")

        # Warnings for recommended files
        for filename in self.RECOMMENDED_FILES:
            if not (self.project_path / filename).exists():
                errors.append(f"Warning: Missing recommended file: {filename}")

        is_valid = not any(
            "Missing required" in e for e in errors
        )

        return is_valid, errors

    def build_sdist(self, output_dir: Path) -> Path:
        """Source distribution oluştur"""
        output_dir.mkdir(parents=True, exist_ok=True)

        # Mock build
        version = "0.1.0"  # Normalde pyproject.toml'dan okunur
        sdist_name = f"{self.package_name}-{version}.tar.gz"
        sdist_path = output_dir / sdist_name

        print(f"Building source distribution...")
        print(f"  Creating {sdist_name}")

        # Mock file creation
        sdist_path.touch()

        print(f"  ✓ Source distribution created: {sdist_path}")
        return sdist_path

    def build_wheel(self, output_dir: Path) -> Path:
        """Wheel distribution oluştur"""
        output_dir.mkdir(parents=True, exist_ok=True)

        # Mock build
        version = "0.1.0"
        python_tag = "py3"
        abi_tag = "none"
        platform_tag = "any"

        wheel_name = f"{self.package_name}-{version}-{python_tag}-{abi_tag}-{platform_tag}.whl"
        wheel_path = output_dir / wheel_name

        print(f"Building wheel distribution...")
        print(f"  Creating {wheel_name}")

        # Mock file creation
        wheel_path.touch()

        print(f"  ✓ Wheel created: {wheel_path}")
        return wheel_path

    def generate_metadata(self) -> str:
        """METADATA dosyası oluştur"""
        # Mock metadata
        metadata = f"""Metadata-Version: 2.1
Name: {self.package_name}
Version: 0.1.0
Summary: A sample package
Author: Developer
Author-Email: dev@example.com
License: MIT
Classifier: Development Status :: 3 - Alpha
Classifier: Intended Audience :: Developers
Classifier: License :: OSI Approved :: MIT License
Classifier: Programming Language :: Python :: 3
Classifier: Programming Language :: Python :: 3.9
Classifier: Programming Language :: Python :: 3.10
Classifier: Programming Language :: Python :: 3.11
Requires-Python: >=3.9
Description-Content-Type: text/markdown

# {self.package_name}

A sample Python package.
"""
        return metadata

    def validate_distributions(self, dist_dir: Path) -> List[str]:
        """Distribution'ları validate et"""
        issues = []

        # Check sdist
        sdist_files = list(dist_dir.glob("*.tar.gz"))
        if not sdist_files:
            issues.append("No source distribution found")

        # Check wheel
        wheel_files = list(dist_dir.glob("*.whl"))
        if not wheel_files:
            issues.append("No wheel distribution found")

        # Validate wheel name
        for wheel in wheel_files:
            parts = wheel.stem.split('-')
            if len(parts) != 5:
                issues.append(f"Invalid wheel name format: {wheel.name}")

        if not issues:
            print("✓ All distributions are valid")

        return issues


# ============================================================================
# EXERCISE 9: Dependency Security Scanner
# Zorluk: Medium
# Konu: Security vulnerability scanning
# ============================================================================

@dataclass
class SecurityVulnerability:
    """Security vulnerability"""
    package: str
    version: str
    vulnerability_id: str
    severity: str  # "low", "medium", "high", "critical"
    description: str
    fixed_in: Optional[str] = None


class SecurityScanner:
    """
    Security vulnerability scanner

    TODO: Bu sınıfı implement edin
    - scan_package: Paketi tara
    - scan_requirements: Requirements dosyasını tara
    - get_fixes: Fix önerileri getir
    - generate_report: Security raporu oluştur
    """

    def __init__(self):
        # TODO: Initialize vulnerability database
        pass

    def scan_package(
        self,
        package_name: str,
        version: str
    ) -> List[SecurityVulnerability]:
        """Paketi tara"""
        # TODO: Implement
        pass

    def scan_requirements(
        self,
        requirements: List[Requirement]
    ) -> Dict[str, List[SecurityVulnerability]]:
        """Requirements dosyasını tara"""
        # TODO: Implement
        pass

    def generate_report(
        self,
        vulnerabilities: Dict[str, List[SecurityVulnerability]]
    ) -> str:
        """Security raporu oluştur"""
        # TODO: Implement
        pass


# SOLUTION
class SecurityScannerSolution:
    """Security vulnerability scanner - Çözüm"""

    def __init__(self):
        # Mock vulnerability database
        self.vulnerability_db = {
            ("requests", "2.27.0"): [
                SecurityVulnerability(
                    package="requests",
                    version="2.27.0",
                    vulnerability_id="CVE-2023-XXXX",
                    severity="medium",
                    description="Potential security issue in cookie handling",
                    fixed_in="2.28.0"
                )
            ],
            ("flask", "2.0.0"): [
                SecurityVulnerability(
                    package="flask",
                    version="2.0.0",
                    vulnerability_id="CVE-2023-YYYY",
                    severity="high",
                    description="Session fixation vulnerability",
                    fixed_in="2.2.0"
                )
            ],
        }

    def scan_package(
        self,
        package_name: str,
        version: str
    ) -> List[SecurityVulnerability]:
        """Paketi tara"""
        key = (package_name, version)
        return self.vulnerability_db.get(key, [])

    def scan_requirements(
        self,
        requirements: List[Requirement]
    ) -> Dict[str, List[SecurityVulnerability]]:
        """Requirements dosyasını tara"""
        results = {}

        for req in requirements:
            # Her version spec için kontrol et
            for op, version in req.version_specs:
                if op == "==":
                    vulns = self.scan_package(req.name, version)
                    if vulns:
                        results[req.name] = vulns

        return results

    def get_fixes(
        self,
        vulnerabilities: Dict[str, List[SecurityVulnerability]]
    ) -> Dict[str, str]:
        """Fix önerileri getir"""
        fixes = {}

        for package, vulns in vulnerabilities.items():
            # En yüksek fix version'ı bul
            fix_versions = [
                v.fixed_in for v in vulns
                if v.fixed_in
            ]

            if fix_versions:
                # En yüksek versiyonu seç
                try:
                    latest_fix = max(
                        fix_versions,
                        key=lambda v: SemanticVersionSolution(v)
                    )
                    fixes[package] = latest_fix
                except:
                    fixes[package] = fix_versions[0]

        return fixes

    def generate_report(
        self,
        vulnerabilities: Dict[str, List[SecurityVulnerability]]
    ) -> str:
        """Security raporu oluştur"""
        if not vulnerabilities:
            return "✓ No security vulnerabilities found"

        lines = ["# Security Vulnerability Report", ""]

        # Severity'ye göre grupla
        by_severity = defaultdict(list)
        for package, vulns in vulnerabilities.items():
            for vuln in vulns:
                by_severity[vuln.severity].append(vuln)

        # Severity sırası
        severity_order = ["critical", "high", "medium", "low"]

        for severity in severity_order:
            if severity in by_severity:
                lines.append(f"## {severity.upper()} Severity")
                lines.append("")

                for vuln in by_severity[severity]:
                    lines.append(f"### {vuln.package} {vuln.version}")
                    lines.append(f"- **ID**: {vuln.vulnerability_id}")
                    lines.append(f"- **Description**: {vuln.description}")
                    if vuln.fixed_in:
                        lines.append(f"- **Fixed in**: {vuln.fixed_in}")
                    lines.append("")

        # Fix recommendations
        fixes = self.get_fixes(vulnerabilities)
        if fixes:
            lines.append("## Recommended Fixes")
            lines.append("")
            for package, version in fixes.items():
                lines.append(f"- Update {package} to {version}")

        return '\n'.join(lines)


# ============================================================================
# EXERCISE 10: Publishing Workflow Manager
# Zorluk: Hard
# Konu: PyPI publishing workflow automation
# ============================================================================

class PublishingWorkflow:
    """
    PyPI publishing workflow yöneticisi

    TODO: Bu sınıfı implement edin
    - validate_package: Paketi publish öncesi validate et
    - run_tests: Test'leri çalıştır
    - build_distributions: Distribution'ları oluştur
    - upload_to_test_pypi: TestPyPI'a yükle
    - upload_to_pypi: PyPI'a yükle
    - create_release_notes: Release notes oluştur
    """

    def __init__(self, project_path: Path):
        # TODO: Initialize
        pass

    def validate_package(self) -> Tuple[bool, List[str]]:
        """Paketi validate et"""
        # TODO: Implement
        # Check: version, changelog, tests, documentation
        pass

    def run_tests(self) -> bool:
        """Test'leri çalıştır"""
        # TODO: Implement (mock)
        pass

    def build_distributions(self, output_dir: Path) -> Tuple[Path, Path]:
        """Distribution'ları oluştur"""
        # TODO: Implement
        pass

    def publish(
        self,
        test_pypi: bool = False,
        skip_existing: bool = True
    ) -> bool:
        """Publish işlemini başlat"""
        # TODO: Implement - tüm workflow'u birleştir
        pass


# SOLUTION
class PublishingWorkflowSolution:
    """PyPI publishing workflow yöneticisi - Çözüm"""

    def __init__(self, project_path: Path):
        self.project_path = project_path
        self.build_manager = BuildSystemManagerSolution(project_path)

    def validate_package(self) -> Tuple[bool, List[str]]:
        """Paketi validate et"""
        errors = []

        # Package structure
        is_valid, structure_errors = self.build_manager.validate_package_structure()
        errors.extend(structure_errors)

        # Version check
        # (Normalde pyproject.toml'dan okunur)

        # Changelog check
        if not (self.project_path / "CHANGELOG.md").exists():
            errors.append("Warning: CHANGELOG.md not found")

        # Tests directory check
        if not (self.project_path / "tests").exists():
            errors.append("Warning: tests/ directory not found")

        # Documentation check
        docs_dir = self.project_path / "docs"
        if not docs_dir.exists():
            errors.append("Warning: docs/ directory not found")

        is_valid = not any("Missing required" in e for e in errors)
        return is_valid, errors

    def run_tests(self) -> bool:
        """Test'leri çalıştır"""
        print("Running tests...")
        print("  ✓ Unit tests passed")
        print("  ✓ Integration tests passed")
        print("  ✓ Coverage: 95%")
        return True

    def build_distributions(self, output_dir: Path) -> Tuple[Path, Path]:
        """Distribution'ları oluştur"""
        print("Building distributions...")

        sdist = self.build_manager.build_sdist(output_dir)
        wheel = self.build_manager.build_wheel(output_dir)

        return sdist, wheel

    def upload_to_test_pypi(self, dist_dir: Path) -> bool:
        """TestPyPI'a yükle"""
        print("Uploading to TestPyPI...")
        print("  ✓ Source distribution uploaded")
        print("  ✓ Wheel uploaded")
        print("  View at: https://test.pypi.org/project/package-name/")
        return True

    def upload_to_pypi(self, dist_dir: Path) -> bool:
        """PyPI'a yükle"""
        print("Uploading to PyPI...")
        print("  ✓ Source distribution uploaded")
        print("  ✓ Wheel uploaded")
        print("  View at: https://pypi.org/project/package-name/")
        return True

    def create_release_notes(self, version: str) -> str:
        """Release notes oluştur"""
        changelog_path = self.project_path / "CHANGELOG.md"

        if changelog_path.exists():
            # Parse changelog for this version
            return f"Release notes for version {version}"

        return f"# Release {version}\n\nNo changelog found."

    def publish(
        self,
        test_pypi: bool = False,
        skip_existing: bool = True
    ) -> bool:
        """Publish işlemini başlat"""
        print("=" * 60)
        print("Starting publish workflow")
        print("=" * 60)

        # 1. Validate
        print("\n1. Validating package...")
        is_valid, errors = self.validate_package()

        if errors:
            print("\nValidation issues:")
            for error in errors:
                print(f"  - {error}")

        if not is_valid:
            print("\n❌ Validation failed")
            return False

        print("✓ Package validation passed")

        # 2. Run tests
        print("\n2. Running tests...")
        if not self.run_tests():
            print("❌ Tests failed")
            return False

        # 3. Build
        print("\n3. Building distributions...")
        dist_dir = self.project_path / "dist"
        sdist, wheel = self.build_distributions(dist_dir)

        # 4. Validate distributions
        print("\n4. Validating distributions...")
        issues = self.build_manager.validate_distributions(dist_dir)
        if issues:
            print("❌ Distribution validation failed:")
            for issue in issues:
                print(f"  - {issue}")
            return False

        # 5. Upload
        print("\n5. Uploading...")
        if test_pypi:
            success = self.upload_to_test_pypi(dist_dir)
        else:
            # First upload to test
            print("Uploading to TestPyPI first...")
            self.upload_to_test_pypi(dist_dir)

            input("\nPress Enter to continue with PyPI upload...")
            success = self.upload_to_pypi(dist_dir)

        if success:
            print("\n" + "=" * 60)
            print("✓ Publishing completed successfully!")
            print("=" * 60)

        return success


# ============================================================================
# TEST FUNCTIONS
# ============================================================================

def test_poetry_manager():
    """Test Poetry Project Manager"""
    print("\n" + "="*60)
    print("Testing Poetry Project Manager")
    print("="*60)

    manager = PoetryProjectManagerSolution(
        name="my-awesome-package",
        version="0.1.0",
        description="An awesome package",
        authors=["Developer <dev@example.com>"],
    )

    # Add dependencies
    manager.add_dependency(
        Dependency("requests", "^2.28.0"),
        DependencyGroup.MAIN
    )
    manager.add_dependency(
        Dependency("pytest", "^7.3.0"),
        DependencyGroup.DEV
    )
    manager.add_dependency(
        Dependency("psycopg2-binary", "^2.9.0", optional=True),
        DependencyGroup.MAIN
    )

    # Add scripts
    manager.add_script("my-cli", "my_package.cli:main")

    # Add extras
    manager.add_extra("postgresql", ["psycopg2-binary"])

    # Generate pyproject.toml
    content = manager.generate_pyproject_toml()
    print(content[:500] + "...")
    print(f"\n✓ Generated {len(content)} characters of pyproject.toml")


def test_semantic_version():
    """Test Semantic Version"""
    print("\n" + "="*60)
    print("Testing Semantic Version")
    print("="*60)

    v1 = SemanticVersionSolution("1.2.3")
    v2 = SemanticVersionSolution("1.2.4")
    v3 = SemanticVersionSolution("2.0.0-alpha.1")

    print(f"v1: {v1}")
    print(f"v2: {v2}")
    print(f"v3: {v3}")

    print(f"\nv1 < v2: {v1 < v2}")
    print(f"v3 is prerelease: {v3.prerelease is not None}")
    print(f"v1.bump_minor(): {v1.bump_minor()}")

    # Constraints
    print(f"\nv1 satisfies '^1.2.0': {v1.satisfies('^1.2.0')}")
    print(f"v1 satisfies '~1.2.3': {v1.satisfies('~1.2.3')}")
    print(f"v2 satisfies '>=1.2.3': {v2.satisfies('>=1.2.3')}")


def test_dependency_resolver():
    """Test Dependency Resolver"""
    print("\n" + "="*60)
    print("Testing Dependency Resolver")
    print("="*60)

    resolver = DependencyResolverSolution()

    # Register packages
    resolver.register_package(PackageVersion(
        "requests", "2.28.0",
        {"urllib3": "^1.26.0", "certifi": ">=2022.0.0"}
    ))
    resolver.register_package(PackageVersion(
        "urllib3", "1.26.15", {}
    ))
    resolver.register_package(PackageVersion(
        "certifi", "2023.5.7", {}
    ))

    # Resolve
    solution = resolver.resolve_dependencies({"requests": "^2.28.0"})

    if solution:
        print("\n✓ Solution found:")
        for name, pkg in solution.items():
            print(f"  {name}=={pkg.version}")
    else:
        print("\n❌ No solution found")


def test_lock_file_manager():
    """Test Lock File Manager"""
    print("\n" + "="*60)
    print("Testing Lock File Manager")
    print("="*60)

    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.lock', delete=False) as f:
        lock_path = Path(f.name)

    try:
        lock = LockFileManagerSolution(lock_path)

        lock.add_package(
            "requests", "2.28.0",
            {"urllib3": "^1.26.0"},
            description="HTTP library"
        )

        lock.save()
        print(f"✓ Lock file saved to {lock_path}")

        # Load and verify
        loaded = LockFileManagerSolution.load(lock_path)
        print(f"✓ Loaded {len(loaded.packages)} packages")

        content_hash = loaded.compute_content_hash()
        print(f"✓ Content hash: {content_hash[:16]}...")

    finally:
        lock_path.unlink()


def test_requirements_parser():
    """Test Requirements Parser"""
    print("\n" + "="*60)
    print("Testing Requirements Parser")
    print("="*60)

    parser = RequirementsParserSolution()

    test_lines = [
        "requests==2.28.0",
        "flask>=2.0.0,<3.0.0",
        "django[extra1,extra2]~=4.0",
        "numpy>=1.20.0 ; python_version >= '3.9'",
        "# This is a comment",
        "",
    ]

    for line in test_lines:
        req = parser.parse_line(line)
        if req:
            normalized = parser.normalize_requirement(req)
            print(f"✓ {line:40} -> {normalized}")


def test_security_scanner():
    """Test Security Scanner"""
    print("\n" + "="*60)
    print("Testing Security Scanner")
    print("="*60)

    scanner = SecurityScannerSolution()

    requirements = [
        Requirement("requests", [("==", "2.27.0")]),
        Requirement("flask", [("==", "2.0.0")]),
    ]

    results = scanner.scan_requirements(requirements)

    print(f"\n✓ Scanned {len(requirements)} packages")
    print(f"✓ Found {len(results)} packages with vulnerabilities")

    report = scanner.generate_report(results)
    print("\n" + report[:500] + "...")


def test_publishing_workflow():
    """Test Publishing Workflow"""
    print("\n" + "="*60)
    print("Testing Publishing Workflow")
    print("="*60)

    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        project_path = Path(tmpdir)

        # Create mock structure
        (project_path / "pyproject.toml").touch()
        (project_path / "README.md").touch()
        (project_path / "tests").mkdir()

        workflow = PublishingWorkflowSolution(project_path)

        # Validate
        is_valid, errors = workflow.validate_package()
        print(f"\n✓ Validation: {'passed' if is_valid else 'failed'}")

        # Build
        dist_dir = project_path / "dist"
        sdist, wheel = workflow.build_distributions(dist_dir)
        print(f"✓ Built: {sdist.name}, {wheel.name}")


if __name__ == "__main__":
    print("PACKAGE MANAGEMENT - ADVANCED EXERCISES")
    print("=" * 60)

    # Run all tests
    test_poetry_manager()
    test_semantic_version()
    test_dependency_resolver()
    test_lock_file_manager()
    test_requirements_parser()
    test_security_scanner()
    test_publishing_workflow()

    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)
