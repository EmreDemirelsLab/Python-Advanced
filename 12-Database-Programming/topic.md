# Database Programming - İleri Seviye

## İçindekiler
1. [SQLite Advanced](#sqlite-advanced)
2. [SQLAlchemy ORM](#sqlalchemy-orm)
3. [Models ve Relationships](#models-ve-relationships)
4. [Advanced Queries](#advanced-queries)
5. [Connection Pooling](#connection-pooling)
6. [Transactions & ACID](#transactions--acid)
7. [Query Optimization](#query-optimization)
8. [Database Migrations (Alembic)](#database-migrations-alembic)
9. [Raw SQL vs ORM](#raw-sql-vs-orm)
10. [NoSQL Basics (MongoDB)](#nosql-basics-mongodb)

---

## SQLite Advanced

### 1. Context Manager ile SQLite Bağlantısı
**Açıklama**: Production ortamında her zaman context manager kullanarak veritabanı bağlantılarını yönetin.

```python
import sqlite3
from contextlib import contextmanager
from typing import Generator

@contextmanager
def get_db_connection(db_path: str) -> Generator[sqlite3.Connection, None, None]:
    """
    Thread-safe SQLite bağlantısı yönetimi.
    Otomatik commit ve rollback desteği.
    """
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row  # Dict-like access
    try:
        yield conn
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()

# Kullanım
with get_db_connection('app.db') as conn:
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE id = ?", (1,))
    user = cursor.fetchone()
    print(dict(user))  # Row factory sayesinde dict'e çevrilebilir
```

### 2. SQLite ile Full-Text Search
**Açıklama**: FTS5 modülü ile gelişmiş metin arama özellikleri.

```python
import sqlite3

def setup_fts_search():
    """Full-Text Search için tablo ve indeks oluşturma"""
    conn = sqlite3.connect('articles.db')
    cursor = conn.cursor()

    # FTS5 sanal tablosu oluşturma
    cursor.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS articles_fts
        USING fts5(title, content, author, tokenize='porter')
    """)

    # Normal tabloya eklenen veriler FTS tablosuna da eklenir
    cursor.execute("""
        CREATE TRIGGER IF NOT EXISTS articles_ai AFTER INSERT ON articles
        BEGIN
            INSERT INTO articles_fts(rowid, title, content, author)
            VALUES (new.id, new.title, new.content, new.author);
        END
    """)

    conn.commit()
    conn.close()

def search_articles(query: str) -> list:
    """FTS kullanarak makale arama"""
    conn = sqlite3.connect('articles.db')
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # FTS5 MATCH operatörü ile arama
    cursor.execute("""
        SELECT articles.*, rank
        FROM articles_fts
        JOIN articles ON articles_fts.rowid = articles.id
        WHERE articles_fts MATCH ?
        ORDER BY rank
        LIMIT 10
    """, (query,))

    results = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return results

# Örnek arama
results = search_articles('python OR database')
```

---

## SQLAlchemy ORM

### 3. SQLAlchemy Engine ve Session Yapılandırması
**Açıklama**: Production-ready engine ve session yönetimi.

```python
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy.pool import QueuePool
import logging

# Logging yapılandırması
logging.basicConfig()
logging.getLogger('sqlalchemy.engine').setLevel(logging.INFO)

class DatabaseManager:
    """Singleton pattern ile database manager"""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        # Engine oluşturma - production settings
        self.engine = create_engine(
            'postgresql://user:pass@localhost/dbname',
            poolclass=QueuePool,
            pool_size=10,  # Varsayılan connection pool boyutu
            max_overflow=20,  # Pool doluysa ek connection sayısı
            pool_timeout=30,  # Connection bekle süresi (saniye)
            pool_recycle=3600,  # Connection yenileme süresi
            echo=False,  # Production'da False olmalı
            echo_pool=False,
        )

        # Event listener - connection'a özel ayarlar
        @event.listens_for(self.engine, "connect")
        def set_sqlite_pragma(dbapi_conn, connection_record):
            """SQLite için pragma ayarları"""
            cursor = dbapi_conn.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.execute("PRAGMA journal_mode=WAL")  # Write-Ahead Logging
            cursor.close()

        # Session factory
        session_factory = sessionmaker(
            bind=self.engine,
            autocommit=False,
            autoflush=False,
            expire_on_commit=False
        )

        # Thread-safe session
        self.Session = scoped_session(session_factory)
        self._initialized = True

    def get_session(self):
        """Yeni session döndür"""
        return self.Session()

    def close_session(self):
        """Mevcut thread'in session'ını kapat"""
        self.Session.remove()

# Kullanım
db_manager = DatabaseManager()
session = db_manager.get_session()
```

### 4. Declarative Base ve Model Tanımlama
**Açıklama**: SQLAlchemy ile model oluşturmanın modern yolu.

```python
from sqlalchemy import Column, Integer, String, DateTime, Boolean, Text, Enum
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func
from datetime import datetime
import enum

Base = declarative_base()

class UserRole(enum.Enum):
    """Enum kullanarak sabit değerler"""
    ADMIN = "admin"
    USER = "user"
    MODERATOR = "moderator"

class TimestampMixin:
    """Tüm modellerde kullanılacak timestamp alanları"""
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class SoftDeleteMixin:
    """Soft delete pattern için mixin"""
    deleted_at = Column(DateTime, nullable=True)
    is_deleted = Column(Boolean, default=False, nullable=False)

    def soft_delete(self):
        """Kaydı soft delete yap"""
        self.deleted_at = datetime.utcnow()
        self.is_deleted = True

class User(Base, TimestampMixin, SoftDeleteMixin):
    """User modeli - mixin'ler ile genişletilmiş"""
    __tablename__ = 'users'

    id = Column(Integer, primary_key=True)
    username = Column(String(50), unique=True, nullable=False, index=True)
    email = Column(String(120), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    role = Column(Enum(UserRole), default=UserRole.USER, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    last_login = Column(DateTime, nullable=True)

    def __repr__(self):
        return f"<User(id={self.id}, username='{self.username}', role={self.role.value})>"

    def to_dict(self):
        """Model'i dictionary'e çevir"""
        return {
            'id': self.id,
            'username': self.username,
            'email': self.email,
            'role': self.role.value,
            'is_active': self.is_active,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'last_login': self.last_login.isoformat() if self.last_login else None
        }

# Tablo oluşturma
Base.metadata.create_all(engine)
```

---

## Models ve Relationships

### 5. One-to-Many İlişkisi
**Açıklama**: Bir kullanıcının birden fazla gönderisi olabilir.

```python
from sqlalchemy import ForeignKey
from sqlalchemy.orm import relationship, backref

class Post(Base, TimestampMixin):
    __tablename__ = 'posts'

    id = Column(Integer, primary_key=True)
    title = Column(String(200), nullable=False)
    content = Column(Text, nullable=False)
    user_id = Column(Integer, ForeignKey('users.id', ondelete='CASCADE'), nullable=False)

    # Relationship tanımlama
    user = relationship('User', backref=backref('posts', lazy='dynamic', cascade='all, delete-orphan'))

    def __repr__(self):
        return f"<Post(id={self.id}, title='{self.title}', user_id={self.user_id})>"

# Kullanım
user = session.query(User).first()
print(user.posts.count())  # Lazy loading
for post in user.posts:
    print(post.title)
```

### 6. Many-to-Many İlişkisi
**Açıklama**: Association table ile çok-a-çok ilişki.

```python
from sqlalchemy import Table

# Association table
post_tags = Table('post_tags', Base.metadata,
    Column('post_id', Integer, ForeignKey('posts.id', ondelete='CASCADE'), primary_key=True),
    Column('tag_id', Integer, ForeignKey('tags.id', ondelete='CASCADE'), primary_key=True),
    Column('created_at', DateTime, default=datetime.utcnow)
)

class Tag(Base):
    __tablename__ = 'tags'

    id = Column(Integer, primary_key=True)
    name = Column(String(50), unique=True, nullable=False)

    # Many-to-many relationship
    posts = relationship('Post', secondary=post_tags, backref='tags', lazy='dynamic')

    def __repr__(self):
        return f"<Tag(name='{self.name}')>"

# Relationship'i Post modeline ekle
Post.tags = relationship('Tag', secondary=post_tags, backref='tagged_posts', lazy='dynamic')

# Kullanım
post = session.query(Post).first()
tag = session.query(Tag).filter_by(name='Python').first()
post.tags.append(tag)
session.commit()
```

### 7. Self-Referential İlişki (Ağaç Yapısı)
**Açıklama**: Kategori ağacı veya yorumlarda kullanılan parent-child ilişkisi.

```python
class Category(Base):
    __tablename__ = 'categories'

    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    parent_id = Column(Integer, ForeignKey('categories.id'), nullable=True)

    # Self-referential relationship
    children = relationship(
        'Category',
        backref=backref('parent', remote_side=[id]),
        lazy='dynamic'
    )

    def get_path(self) -> list:
        """Kategorinin root'a kadar olan path'ini döndür"""
        path = [self.name]
        parent = self.parent
        while parent:
            path.insert(0, parent.name)
            parent = parent.parent
        return path

    def get_all_children(self) -> list:
        """Tüm alt kategorileri recursive olarak getir"""
        result = [self]
        for child in self.children:
            result.extend(child.get_all_children())
        return result

# Kullanım
electronics = Category(name='Electronics')
computers = Category(name='Computers', parent=electronics)
laptops = Category(name='Laptops', parent=computers)

session.add_all([electronics, computers, laptops])
session.commit()

print(laptops.get_path())  # ['Electronics', 'Computers', 'Laptops']
```

### 8. Association Object Pattern
**Açıklama**: Many-to-many ilişkisine ekstra veri eklemek için.

```python
class OrderItem(Base, TimestampMixin):
    """Order ve Product arasındaki association object"""
    __tablename__ = 'order_items'

    order_id = Column(Integer, ForeignKey('orders.id'), primary_key=True)
    product_id = Column(Integer, ForeignKey('products.id'), primary_key=True)
    quantity = Column(Integer, nullable=False, default=1)
    unit_price = Column(Integer, nullable=False)  # Fiyat o andaki fiyat

    # Relationships
    order = relationship('Order', backref='order_items')
    product = relationship('Product', backref='order_items')

    @property
    def total_price(self):
        return self.quantity * self.unit_price

class Order(Base, TimestampMixin):
    __tablename__ = 'orders'

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    status = Column(String(20), default='pending')

    user = relationship('User', backref='orders')

    @property
    def total_amount(self):
        """Toplam sipariş tutarı"""
        return sum(item.total_price for item in self.order_items)

class Product(Base):
    __tablename__ = 'products'

    id = Column(Integer, primary_key=True)
    name = Column(String(200), nullable=False)
    price = Column(Integer, nullable=False)
    stock = Column(Integer, default=0)

# Kullanım
order = Order(user_id=1)
product = session.query(Product).first()

order_item = OrderItem(
    order=order,
    product=product,
    quantity=2,
    unit_price=product.price
)

session.add(order)
session.commit()
```

---

## Advanced Queries

### 9. Complex Query Builder
**Açıklama**: Dinamik ve karmaşık sorgular oluşturma.

```python
from sqlalchemy import and_, or_, not_, case, cast, func
from sqlalchemy.sql import label
from typing import Optional, List, Dict, Any

class UserQueryBuilder:
    """Advanced query builder pattern"""

    def __init__(self, session):
        self.session = session
        self.query = session.query(User)
        self.filters = []

    def with_role(self, role: UserRole):
        """Role'e göre filtrele"""
        self.filters.append(User.role == role)
        return self

    def active_only(self):
        """Sadece aktif kullanıcılar"""
        self.filters.append(User.is_active == True)
        self.filters.append(User.is_deleted == False)
        return self

    def search(self, term: str):
        """Username veya email'de arama"""
        search_filter = or_(
            User.username.ilike(f'%{term}%'),
            User.email.ilike(f'%{term}%')
        )
        self.filters.append(search_filter)
        return self

    def registered_between(self, start_date, end_date):
        """Belirli tarih aralığında kayıt olanlar"""
        self.filters.append(and_(
            User.created_at >= start_date,
            User.created_at <= end_date
        ))
        return self

    def with_posts_count(self):
        """Post sayısını da getir"""
        from sqlalchemy import func
        self.query = self.query.outerjoin(Post).group_by(User.id)
        self.query = self.query.add_columns(
            func.count(Post.id).label('posts_count')
        )
        return self

    def paginate(self, page: int = 1, per_page: int = 20):
        """Sayfalama"""
        offset = (page - 1) * per_page
        self.query = self.query.limit(per_page).offset(offset)
        return self

    def build(self):
        """Query'yi oluştur ve çalıştır"""
        if self.filters:
            self.query = self.query.filter(and_(*self.filters))
        return self.query

# Kullanım
builder = UserQueryBuilder(session)
results = (builder
    .active_only()
    .with_role(UserRole.USER)
    .search('john')
    .with_posts_count()
    .paginate(page=1, per_page=10)
    .build()
    .all())

for user, posts_count in results:
    print(f"{user.username}: {posts_count} posts")
```

### 10. Subqueries ve CTEs (Common Table Expressions)
**Açıklama**: Karmaşık veri analizi için alt sorgular.

```python
from sqlalchemy import select, exists

def get_users_with_recent_activity(session, days: int = 30):
    """Son N gün içinde aktif olan kullanıcılar"""

    # Subquery: Son N günde post atmış kullanıcılar
    recent_posts = (
        session.query(Post.user_id)
        .filter(Post.created_at >= datetime.utcnow() - timedelta(days=days))
        .subquery()
    )

    # Ana sorgu
    active_users = (
        session.query(User)
        .filter(User.id.in_(select([recent_posts.c.user_id])))
        .all()
    )

    return active_users

def get_users_statistics(session):
    """CTE kullanarak kullanıcı istatistikleri"""
    from sqlalchemy import literal_column

    # CTE: Her kullanıcının post sayısı
    posts_cte = (
        session.query(
            Post.user_id,
            func.count(Post.id).label('post_count'),
            func.max(Post.created_at).label('last_post_date')
        )
        .group_by(Post.user_id)
        .cte('posts_stats')
    )

    # Ana sorgu: User bilgisi + istatistikler
    query = (
        session.query(
            User.username,
            User.email,
            func.coalesce(posts_cte.c.post_count, 0).label('total_posts'),
            posts_cte.c.last_post_date,
            case(
                (posts_cte.c.post_count > 100, 'power_user'),
                (posts_cte.c.post_count > 10, 'active_user'),
                else_='casual_user'
            ).label('user_type')
        )
        .outerjoin(posts_cte, User.id == posts_cte.c.user_id)
        .order_by(posts_cte.c.post_count.desc().nullslast())
    )

    return query.all()

# Kullanım
stats = get_users_statistics(session)
for username, email, total_posts, last_post, user_type in stats:
    print(f"{username}: {total_posts} posts ({user_type})")
```

### 11. Window Functions
**Açıklama**: Gelişmiş analitik sorgular için window functions.

```python
from sqlalchemy import func, over

def get_ranked_posts(session):
    """Her kullanıcının postlarını tarihine göre rankle"""

    # Window function: Her kullanıcı için post sıralaması
    ranked_posts = (
        session.query(
            Post.id,
            Post.title,
            Post.user_id,
            Post.created_at,
            func.row_number().over(
                partition_by=Post.user_id,
                order_by=Post.created_at.desc()
            ).label('post_rank'),
            func.count(Post.id).over(
                partition_by=Post.user_id
            ).label('user_total_posts')
        )
        .all()
    )

    return ranked_posts

def get_running_totals(session):
    """Her gün toplam kullanıcı sayısı (running total)"""

    daily_users = (
        session.query(
            func.date(User.created_at).label('date'),
            func.count(User.id).label('new_users'),
            func.sum(func.count(User.id)).over(
                order_by=func.date(User.created_at)
            ).label('total_users')
        )
        .group_by(func.date(User.created_at))
        .all()
    )

    return daily_users

# Kullanım
ranked = get_ranked_posts(session)
for post_id, title, user_id, created, rank, total in ranked:
    print(f"User {user_id}: Post #{rank}/{total} - {title}")
```

---

## Connection Pooling

### 12. Custom Connection Pool
**Açıklama**: Production için optimize edilmiş connection pool.

```python
from sqlalchemy.pool import QueuePool, NullPool, StaticPool
from sqlalchemy import event, pool
import time

class MonitoredConnectionPool:
    """Connection pool monitoring wrapper"""

    def __init__(self, engine):
        self.engine = engine
        self.connection_stats = {
            'total_connections': 0,
            'active_connections': 0,
            'checkouts': 0,
            'checkins': 0,
            'errors': 0
        }
        self._setup_listeners()

    def _setup_listeners(self):
        """Pool event listener'ları kur"""

        @event.listens_for(self.engine, "connect")
        def receive_connect(dbapi_conn, connection_record):
            self.connection_stats['total_connections'] += 1
            connection_record.info['connect_time'] = time.time()

        @event.listens_for(self.engine, "checkout")
        def receive_checkout(dbapi_conn, connection_record, connection_proxy):
            self.connection_stats['checkouts'] += 1
            self.connection_stats['active_connections'] += 1
            connection_record.info['checkout_time'] = time.time()

        @event.listens_for(self.engine, "checkin")
        def receive_checkin(dbapi_conn, connection_record):
            self.connection_stats['checkins'] += 1
            self.connection_stats['active_connections'] -= 1

            # Connection kullanım süresini logla
            checkout_time = connection_record.info.get('checkout_time')
            if checkout_time:
                duration = time.time() - checkout_time
                if duration > 5:  # 5 saniyeden uzun sürerse warn
                    print(f"Warning: Long-running connection ({duration:.2f}s)")

    def get_stats(self):
        """Pool istatistiklerini döndür"""
        pool = self.engine.pool
        return {
            **self.connection_stats,
            'pool_size': pool.size(),
            'checked_out': pool.checkedout(),
            'overflow': pool.overflow(),
            'available': pool.size() - pool.checkedout()
        }

# Production engine with optimal pooling
def create_production_engine(db_url: str):
    """Production için optimize edilmiş engine"""

    engine = create_engine(
        db_url,
        poolclass=QueuePool,
        pool_size=20,  # Minimum connection sayısı
        max_overflow=10,  # Pool doluysa ek connection
        pool_timeout=30,  # Connection bekleme süresi
        pool_recycle=3600,  # 1 saatte bir connection yenile
        pool_pre_ping=True,  # Connection'ı kullanmadan önce test et
        echo=False
    )

    return MonitoredConnectionPool(engine)

# Kullanım
pool_manager = create_production_engine('postgresql://localhost/mydb')
print(pool_manager.get_stats())
```

---

## Transactions & ACID

### 13. Transaction Yönetimi
**Açıklama**: ACID özelliklerini garanti altına alan transaction yönetimi.

```python
from sqlalchemy.orm import Session
from contextlib import contextmanager
from typing import Callable, Any

@contextmanager
def transaction_scope(session: Session):
    """
    Transaction context manager.
    Otomatik commit/rollback yönetimi.
    """
    try:
        yield session
        session.commit()
    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()

class TransactionalService:
    """Service layer with transaction management"""

    def __init__(self, session_factory):
        self.session_factory = session_factory

    def execute_in_transaction(self, func: Callable, *args, **kwargs) -> Any:
        """Fonksiyonu transaction içinde çalıştır"""
        session = self.session_factory()
        try:
            result = func(session, *args, **kwargs)
            session.commit()
            return result
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()

    def transfer_money(self, from_user_id: int, to_user_id: int, amount: int):
        """Para transferi - atomic operation örneği"""

        def _transfer(session):
            # Pessimistic locking ile kullanıcıları getir
            from_user = (
                session.query(User)
                .filter(User.id == from_user_id)
                .with_for_update()  # SELECT ... FOR UPDATE
                .first()
            )

            to_user = (
                session.query(User)
                .filter(User.id == to_user_id)
                .with_for_update()
                .first()
            )

            if not from_user or not to_user:
                raise ValueError("User not found")

            if from_user.balance < amount:
                raise ValueError("Insufficient balance")

            # Atomic update
            from_user.balance -= amount
            to_user.balance += amount

            # Transaction log
            log = TransactionLog(
                from_user_id=from_user_id,
                to_user_id=to_user_id,
                amount=amount,
                status='completed'
            )
            session.add(log)

            return log

        return self.execute_in_transaction(_transfer)

# Kullanım
service = TransactionalService(db_manager.Session)
try:
    log = service.transfer_money(from_user_id=1, to_user_id=2, amount=100)
    print(f"Transfer successful: {log.id}")
except Exception as e:
    print(f"Transfer failed: {e}")
```

### 14. Isolation Levels
**Açıklama**: Transaction isolation level'larını anlama ve kullanma.

```python
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

def demonstrate_isolation_levels():
    """Farklı isolation level'ların etkilerini gösterir"""

    # READ UNCOMMITTED - En düşük isolation
    engine_read_uncommitted = create_engine(
        'postgresql://localhost/mydb',
        isolation_level="READ UNCOMMITTED"
    )

    # READ COMMITTED - PostgreSQL default
    engine_read_committed = create_engine(
        'postgresql://localhost/mydb',
        isolation_level="READ COMMITTED"
    )

    # REPEATABLE READ - Phantom read'leri engeller
    engine_repeatable_read = create_engine(
        'postgresql://localhost/mydb',
        isolation_level="REPEATABLE READ"
    )

    # SERIALIZABLE - En yüksek isolation
    engine_serializable = create_engine(
        'postgresql://localhost/mydb',
        isolation_level="SERIALIZABLE"
    )

    # Session bazında isolation level ayarlama
    Session = sessionmaker(bind=engine_read_committed)
    session = Session()

    # Transaction içinde isolation level değiştirme
    session.connection(execution_options={'isolation_level': 'SERIALIZABLE'})

    return session

def handle_serialization_failure():
    """SERIALIZABLE isolation'da retry logic"""
    from sqlalchemy.exc import OperationalError
    import time

    max_retries = 3
    retry_count = 0

    while retry_count < max_retries:
        session = demonstrate_isolation_levels()
        try:
            # Critical transaction
            user = session.query(User).filter_by(id=1).with_for_update().first()
            user.balance += 100
            session.commit()
            break
        except OperationalError as e:
            if 'could not serialize' in str(e):
                retry_count += 1
                time.sleep(0.1 * retry_count)  # Exponential backoff
                session.rollback()
            else:
                raise
        finally:
            session.close()
```

---

## Query Optimization

### 15. N+1 Query Problem Çözümü
**Açıklama**: Lazy loading'in neden olduğu performans sorunlarını çözme.

```python
from sqlalchemy.orm import joinedload, subqueryload, selectinload

# BAD: N+1 query problemi
def get_users_with_posts_bad():
    """Her user için ayrı query - KÖTÜ!"""
    users = session.query(User).all()
    for user in users:
        print(f"{user.username}: {user.posts.count()} posts")  # Her user için +1 query!

# GOOD: Eager loading ile tek query
def get_users_with_posts_good():
    """Tüm data tek query'de - İYİ!"""
    users = (
        session.query(User)
        .options(selectinload(User.posts))  # Posts'ları da getir
        .all()
    )
    for user in users:
        print(f"{user.username}: {len(user.posts)} posts")  # Ek query yok

# joinedload: LEFT OUTER JOIN ile
def get_posts_with_user_joinedload():
    """User bilgisini de JOIN ile getir"""
    posts = (
        session.query(Post)
        .options(joinedload(Post.user))  # JOIN ile user'ı getir
        .all()
    )
    for post in posts:
        print(f"{post.title} by {post.user.username}")  # Ek query yok

# subqueryload: İkinci query ile
def get_posts_with_tags_subqueryload():
    """Tag'leri subquery ile getir"""
    posts = (
        session.query(Post)
        .options(subqueryload(Post.tags))  # Ayrı query ama optimize
        .all()
    )
    return posts

# selectinload: Modern ve verimli
def get_users_with_posts_and_tags():
    """Nested relationship'leri efficiently load et"""
    users = (
        session.query(User)
        .options(
            selectinload(User.posts).selectinload(Post.tags)
        )
        .all()
    )
    return users
```

### 16. Index Kullanımı ve Query Profiling
**Açıklama**: Database index'leri ve query performans analizi.

```python
from sqlalchemy import Index, text
from sqlalchemy.dialects import postgresql
import time

class QueryProfiler:
    """Query performans analizi"""

    def __init__(self, session):
        self.session = session

    def explain_query(self, query):
        """Query execution plan'ı göster"""
        # PostgreSQL için EXPLAIN ANALYZE
        explained = self.session.execute(
            text(f"EXPLAIN ANALYZE {str(query.statement)}")
        )
        return [row for row in explained]

    def profile_query(self, query):
        """Query çalışma süresini ölç"""
        start = time.time()
        result = query.all()
        duration = time.time() - start

        return {
            'duration': duration,
            'count': len(result),
            'query': str(query.statement)
        }

# Index tanımlama
class OptimizedUser(Base):
    __tablename__ = 'optimized_users'

    id = Column(Integer, primary_key=True)
    username = Column(String(50), nullable=False)
    email = Column(String(120), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Composite index
    __table_args__ = (
        Index('idx_username_email', 'username', 'email'),
        Index('idx_created_at_desc', created_at.desc()),  # Descending index
        Index('idx_email_partial', 'email',
              postgresql_where=text("email IS NOT NULL")),  # Partial index
    )

# Kullanım
profiler = QueryProfiler(session)

# Kötü query
bad_query = session.query(User).filter(User.username.like('%john%'))
print(profiler.profile_query(bad_query))

# İyi query - index kullanıyor
good_query = session.query(User).filter(User.username == 'john_doe')
print(profiler.profile_query(good_query))
```

---

## Database Migrations (Alembic)

### 17. Alembic Setup ve Migration
**Açıklama**: Database schema değişikliklerini version control altında tutma.

```python
# alembic.ini ve env.py yapılandırması sonrası

# Migration script örneği
"""Add user profile table

Revision ID: 001_add_user_profile
Revises:
Create Date: 2024-01-01 12:00:00
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers
revision = '001_add_user_profile'
down_revision = None
branch_labels = None
depends_on = None

def upgrade():
    """Upgrade migration"""
    # Tablo oluştur
    op.create_table(
        'user_profiles',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('bio', sa.Text(), nullable=True),
        sa.Column('avatar_url', sa.String(255), nullable=True),
        sa.Column('location', sa.String(100), nullable=True),
        sa.Column('website', sa.String(200), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
    )

    # Index ekle
    op.create_index('idx_user_profiles_user_id', 'user_profiles', ['user_id'], unique=True)

    # Mevcut tabloya kolon ekle
    op.add_column('users', sa.Column('is_verified', sa.Boolean(),
                                      nullable=False, server_default='false'))

    # Data migration
    op.execute("""
        INSERT INTO user_profiles (user_id, bio, created_at)
        SELECT id, '', NOW() FROM users
    """)

def downgrade():
    """Downgrade migration"""
    # Reverse operations
    op.drop_column('users', 'is_verified')
    op.drop_index('idx_user_profiles_user_id', table_name='user_profiles')
    op.drop_table('user_profiles')
```

### 18. Alembic Helper Functions
**Açıklama**: Migration'larda kullanılacak helper fonksiyonlar.

```python
# migrations/utils.py
from alembic import op
import sqlalchemy as sa

def create_updated_at_trigger(table_name: str):
    """PostgreSQL'de otomatik updated_at trigger'ı oluştur"""
    op.execute(f"""
        CREATE OR REPLACE FUNCTION update_{table_name}_updated_at()
        RETURNS TRIGGER AS $$
        BEGIN
            NEW.updated_at = NOW();
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;

        CREATE TRIGGER trigger_{table_name}_updated_at
        BEFORE UPDATE ON {table_name}
        FOR EACH ROW
        EXECUTE FUNCTION update_{table_name}_updated_at();
    """)

def drop_updated_at_trigger(table_name: str):
    """Trigger'ı kaldır"""
    op.execute(f"""
        DROP TRIGGER IF EXISTS trigger_{table_name}_updated_at ON {table_name};
        DROP FUNCTION IF EXISTS update_{table_name}_updated_at();
    """)

def add_audit_columns(table_name: str):
    """Tüm tablolara audit kolonları ekle"""
    op.add_column(table_name, sa.Column('created_at', sa.DateTime(),
                                        nullable=False, server_default=sa.func.now()))
    op.add_column(table_name, sa.Column('updated_at', sa.DateTime(),
                                        onupdate=sa.func.now()))
    op.add_column(table_name, sa.Column('created_by', sa.Integer(), nullable=True))
    op.add_column(table_name, sa.Column('updated_by', sa.Integer(), nullable=True))

# Migration'da kullanım
def upgrade():
    op.create_table('articles', ...)
    add_audit_columns('articles')
    create_updated_at_trigger('articles')

def downgrade():
    drop_updated_at_trigger('articles')
    op.drop_table('articles')
```

---

## Raw SQL vs ORM

### 19. Hybrid Approach - ORM + Raw SQL
**Açıklama**: ORM ve Raw SQL'i birlikte kullanma.

```python
from sqlalchemy import text

class HybridRepository:
    """ORM ve Raw SQL'i birlikte kullanan repository"""

    def __init__(self, session):
        self.session = session

    # ORM kullanımı - basit CRUD
    def create_user(self, username: str, email: str) -> User:
        """ORM ile create - type-safe ve kolay"""
        user = User(username=username, email=email)
        self.session.add(user)
        self.session.commit()
        return user

    # Raw SQL - complex query
    def get_user_statistics(self, user_id: int) -> dict:
        """Karmaşık analitik query için raw SQL"""
        query = text("""
            WITH user_stats AS (
                SELECT
                    u.id,
                    u.username,
                    COUNT(DISTINCT p.id) as total_posts,
                    COUNT(DISTINCT c.id) as total_comments,
                    AVG(p.view_count) as avg_views,
                    MAX(p.created_at) as last_post_date
                FROM users u
                LEFT JOIN posts p ON u.id = p.user_id
                LEFT JOIN comments c ON u.id = c.user_id
                WHERE u.id = :user_id
                GROUP BY u.id, u.username
            )
            SELECT * FROM user_stats
        """)

        result = self.session.execute(query, {'user_id': user_id}).fetchone()

        if result:
            return {
                'id': result[0],
                'username': result[1],
                'total_posts': result[2],
                'total_comments': result[3],
                'avg_views': float(result[4]) if result[4] else 0,
                'last_post_date': result[5]
            }
        return None

    # ORM + Raw SQL combo
    def bulk_update_with_subquery(self, tag_name: str, new_status: str):
        """ORM model + Raw SQL subquery"""
        subquery = text("""
            SELECT p.id
            FROM posts p
            JOIN post_tags pt ON p.id = pt.post_id
            JOIN tags t ON pt.tag_id = t.id
            WHERE t.name = :tag_name
        """)

        # ORM update with raw SQL subquery
        self.session.query(Post).filter(
            Post.id.in_(subquery)
        ).update(
            {'status': new_status},
            synchronize_session=False
        )
        self.session.commit()

    # Raw SQL ile batch insert
    def bulk_insert_users(self, users_data: list):
        """Toplu insert için raw SQL - daha hızlı"""
        query = text("""
            INSERT INTO users (username, email, password_hash, created_at)
            VALUES (:username, :email, :password_hash, NOW())
        """)

        self.session.execute(query, users_data)
        self.session.commit()

# Kullanım
repo = HybridRepository(session)

# ORM - basit işlem
user = repo.create_user('john', 'john@example.com')

# Raw SQL - complex analytic
stats = repo.get_user_statistics(user.id)
print(stats)

# Batch insert
repo.bulk_insert_users([
    {'username': 'user1', 'email': 'user1@example.com', 'password_hash': 'hash1'},
    {'username': 'user2', 'email': 'user2@example.com', 'password_hash': 'hash2'},
])
```

---

## NoSQL Basics (MongoDB)

### 20. MongoDB ile PyMongo
**Açıklama**: NoSQL database kullanımı ve document-based storage.

```python
from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.errors import DuplicateKeyError, ConnectionFailure
from datetime import datetime
from typing import Optional, List, Dict
import logging

class MongoDBManager:
    """MongoDB connection ve operation manager"""

    def __init__(self, connection_string: str, database_name: str):
        try:
            self.client = MongoClient(
                connection_string,
                serverSelectionTimeoutMS=5000,
                maxPoolSize=50,
                minPoolSize=10,
            )
            # Connection test
            self.client.admin.command('ping')
            self.db = self.client[database_name]
            logging.info(f"Connected to MongoDB: {database_name}")
        except ConnectionFailure as e:
            logging.error(f"MongoDB connection failed: {e}")
            raise

    def close(self):
        """Connection'ı kapat"""
        self.client.close()

class UserRepository:
    """MongoDB user repository"""

    def __init__(self, db):
        self.collection = db.users
        self._create_indexes()

    def _create_indexes(self):
        """Index'leri oluştur"""
        # Unique index
        self.collection.create_index('email', unique=True)
        self.collection.create_index('username', unique=True)

        # Compound index
        self.collection.create_index([
            ('created_at', DESCENDING),
            ('is_active', ASCENDING)
        ])

        # Text index for full-text search
        self.collection.create_index([
            ('username', 'text'),
            ('bio', 'text')
        ])

    def create_user(self, user_data: dict) -> str:
        """Yeni user oluştur"""
        user_data['created_at'] = datetime.utcnow()
        user_data['updated_at'] = datetime.utcnow()

        try:
            result = self.collection.insert_one(user_data)
            return str(result.inserted_id)
        except DuplicateKeyError:
            raise ValueError("User already exists")

    def find_user(self, query: dict) -> Optional[dict]:
        """User bul"""
        return self.collection.find_one(query)

    def update_user(self, user_id: str, update_data: dict) -> bool:
        """User güncelle"""
        from bson.objectid import ObjectId

        update_data['updated_at'] = datetime.utcnow()
        result = self.collection.update_one(
            {'_id': ObjectId(user_id)},
            {'$set': update_data}
        )
        return result.modified_count > 0

    def aggregate_user_stats(self) -> List[dict]:
        """Aggregation pipeline ile istatistik"""
        pipeline = [
            # Group by role
            {
                '$group': {
                    '_id': '$role',
                    'count': {'$sum': 1},
                    'avg_age': {'$avg': '$age'},
                    'active_users': {
                        '$sum': {'$cond': ['$is_active', 1, 0]}
                    }
                }
            },
            # Sort by count
            {'$sort': {'count': -1}},
            # Project final shape
            {
                '$project': {
                    'role': '$_id',
                    'total_users': '$count',
                    'average_age': {'$round': ['$avg_age', 1]},
                    'active_users': 1,
                    '_id': 0
                }
            }
        ]

        return list(self.collection.aggregate(pipeline))

    def text_search(self, search_term: str) -> List[dict]:
        """Full-text search"""
        return list(self.collection.find(
            {'$text': {'$search': search_term}},
            {'score': {'$meta': 'textScore'}}
        ).sort([('score', {'$meta': 'textScore'})]))

# Kullanım
mongo_manager = MongoDBManager('mongodb://localhost:27017/', 'myapp')
user_repo = UserRepository(mongo_manager.db)

# Create
user_id = user_repo.create_user({
    'username': 'john_doe',
    'email': 'john@example.com',
    'role': 'user',
    'age': 25,
    'is_active': True,
    'bio': 'Python developer'
})

# Find
user = user_repo.find_user({'username': 'john_doe'})

# Update
user_repo.update_user(user_id, {'age': 26})

# Aggregate
stats = user_repo.aggregate_user_stats()
print(stats)

# Search
results = user_repo.text_search('python developer')
```

---

## Production Best Practices

### Örnek: Complete Production Setup

```python
"""
Production-ready database setup
- Connection pooling
- Session management
- Error handling
- Logging
- Metrics
"""

from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy.pool import QueuePool
from contextlib import contextmanager
import logging
from typing import Generator
import time

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProductionDatabase:
    """Production veritabanı yönetimi"""

    def __init__(self, database_url: str):
        # Engine oluşturma
        self.engine = create_engine(
            database_url,
            poolclass=QueuePool,
            pool_size=20,
            max_overflow=10,
            pool_timeout=30,
            pool_recycle=3600,
            pool_pre_ping=True,
            echo=False,
            connect_args={
                'connect_timeout': 10,
                'options': '-c statement_timeout=30000'  # 30 saniye
            }
        )

        # Session factory
        session_factory = sessionmaker(
            bind=self.engine,
            autocommit=False,
            autoflush=False,
            expire_on_commit=False
        )

        self.Session = scoped_session(session_factory)

        # Event listeners
        self._setup_event_listeners()

        # Metrics
        self.metrics = {
            'queries': 0,
            'slow_queries': 0,
            'errors': 0,
        }

    def _setup_event_listeners(self):
        """Event listener'ları kur"""

        @event.listens_for(self.engine, "before_cursor_execute")
        def before_cursor_execute(conn, cursor, statement, parameters, context, executemany):
            context._query_start_time = time.time()

        @event.listens_for(self.engine, "after_cursor_execute")
        def after_cursor_execute(conn, cursor, statement, parameters, context, executemany):
            total_time = time.time() - context._query_start_time
            self.metrics['queries'] += 1

            if total_time > 1.0:  # 1 saniyeden uzun
                self.metrics['slow_queries'] += 1
                logger.warning(f"Slow query ({total_time:.2f}s): {statement[:100]}")

    @contextmanager
    def session_scope(self) -> Generator:
        """Session context manager"""
        session = self.Session()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            self.metrics['errors'] += 1
            logger.error(f"Database error: {e}")
            raise
        finally:
            session.close()

    def health_check(self) -> bool:
        """Veritabanı sağlık kontrolü"""
        try:
            with self.session_scope() as session:
                session.execute(text("SELECT 1"))
            return True
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False

    def get_metrics(self) -> dict:
        """Metrikleri döndür"""
        pool = self.engine.pool
        return {
            **self.metrics,
            'pool_size': pool.size(),
            'checked_out': pool.checkedout(),
            'overflow': pool.overflow(),
        }

# Global instance
db = ProductionDatabase('postgresql://localhost/myapp')

# Kullanım
with db.session_scope() as session:
    users = session.query(User).filter_by(is_active=True).all()
    for user in users:
        print(user.username)

# Metrics
print(db.get_metrics())
```

---

## Özet

Bu dokümanda ele alınan konular:

1. **SQLite Advanced**: Context managers, FTS, WAL mode
2. **SQLAlchemy ORM**: Engine, session, model tanımlama
3. **Relationships**: One-to-many, many-to-many, self-referential, association objects
4. **Advanced Queries**: Query builders, subqueries, CTEs, window functions
5. **Connection Pooling**: QueuePool, monitoring, optimization
6. **Transactions**: ACID, isolation levels, pessimistic locking
7. **Query Optimization**: N+1 problem, eager loading, indexing
8. **Alembic**: Database migrations, version control
9. **Raw SQL vs ORM**: Hybrid approach, best practices
10. **NoSQL (MongoDB)**: PyMongo, aggregations, text search

### Production Checklist
- ✅ Connection pooling kullan
- ✅ Session'ları doğru yönet (context managers)
- ✅ Transaction'ları isolation level ile koru
- ✅ N+1 query probleminden kaçın
- ✅ Index'leri doğru kullan
- ✅ Slow query monitoring yap
- ✅ Migration'ları version control'de tut
- ✅ Health check endpoint'i ekle
- ✅ Metrics ve logging ekle
- ✅ Connection timeout'ları ayarla
