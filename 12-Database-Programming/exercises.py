"""
Database Programming - Advanced Exercises
Her exercise TODO ve solution içerir.
Gerçek dünya senaryoları: User management, E-commerce, Blog system
"""

from sqlalchemy import (
    create_engine, Column, Integer, String, DateTime, Boolean,
    Text, ForeignKey, Table, Float, Enum, Index, func, and_, or_
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, backref, scoped_session
from sqlalchemy.pool import QueuePool
from contextlib import contextmanager
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
import enum
import logging

Base = declarative_base()

# ============================================================================
# EXERCISE 1: Advanced Model Design - E-Commerce System
# ============================================================================
"""
TODO: Kompleks bir e-commerce sistemi için model tasarımı yapın:
- Product (id, name, description, price, stock, category_id)
- Category (id, name, parent_id) - Self-referential
- Order (id, user_id, status, total_amount, created_at)
- OrderItem (order_id, product_id, quantity, unit_price)
- Review (id, product_id, user_id, rating, comment)

Gereksinimler:
1. Soft delete mixin kullanın
2. Timestamp mixin kullanın
3. Tüm ilişkileri doğru kurun
4. Uygun index'leri ekleyin
5. Enum kullanarak status ve rating tanımlayın
"""

# SOLUTION:

class TimestampMixin:
    """Her model için otomatik timestamp alanları"""
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class SoftDeleteMixin:
    """Soft delete pattern için mixin"""
    deleted_at = Column(DateTime, nullable=True)
    is_deleted = Column(Boolean, default=False, nullable=False)

    def soft_delete(self):
        self.deleted_at = datetime.utcnow()
        self.is_deleted = True

class OrderStatus(enum.Enum):
    """Sipariş durumları"""
    PENDING = "pending"
    PROCESSING = "processing"
    SHIPPED = "shipped"
    DELIVERED = "delivered"
    CANCELLED = "cancelled"

class Rating(enum.Enum):
    """Ürün değerlendirme puanları"""
    ONE = 1
    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5

class Category(Base, TimestampMixin, SoftDeleteMixin):
    """Kategori modeli - self-referential relationship"""
    __tablename__ = 'categories'

    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False, index=True)
    parent_id = Column(Integer, ForeignKey('categories.id'), nullable=True)

    # Self-referential relationship
    children = relationship('Category', backref=backref('parent', remote_side=[id]))
    products = relationship('Product', back_populates='category')

    def get_path(self) -> List[str]:
        """Kategorinin root'a kadar path'ini döndür"""
        path = [self.name]
        parent = self.parent
        while parent:
            path.insert(0, parent.name)
            parent = parent.parent
        return path

class Product(Base, TimestampMixin, SoftDeleteMixin):
    """Ürün modeli"""
    __tablename__ = 'products'

    id = Column(Integer, primary_key=True)
    name = Column(String(200), nullable=False, index=True)
    description = Column(Text, nullable=True)
    price = Column(Float, nullable=False)
    stock = Column(Integer, default=0, nullable=False)
    category_id = Column(Integer, ForeignKey('categories.id'), nullable=False)

    # Relationships
    category = relationship('Category', back_populates='products')
    reviews = relationship('Review', back_populates='product', cascade='all, delete-orphan')
    order_items = relationship('OrderItem', back_populates='product')

    # Indexes
    __table_args__ = (
        Index('idx_product_category_price', 'category_id', 'price'),
        Index('idx_product_stock', 'stock'),
    )

    @property
    def average_rating(self) -> float:
        """Ortalama rating hesapla"""
        if not self.reviews:
            return 0.0
        return sum(r.rating.value for r in self.reviews) / len(self.reviews)

    @property
    def is_in_stock(self) -> bool:
        """Stokta var mı?"""
        return self.stock > 0

class Order(Base, TimestampMixin):
    """Sipariş modeli"""
    __tablename__ = 'orders'

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, nullable=False, index=True)
    status = Column(Enum(OrderStatus), default=OrderStatus.PENDING, nullable=False)
    total_amount = Column(Float, default=0.0, nullable=False)

    # Relationships
    order_items = relationship('OrderItem', back_populates='order', cascade='all, delete-orphan')

    # Indexes
    __table_args__ = (
        Index('idx_order_user_status', 'user_id', 'status'),
        Index('idx_order_created_at', 'created_at'),
    )

    def calculate_total(self):
        """Toplam tutarı hesapla"""
        self.total_amount = sum(item.subtotal for item in self.order_items)

class OrderItem(Base):
    """Sipariş kalemi - Association Object Pattern"""
    __tablename__ = 'order_items'

    order_id = Column(Integer, ForeignKey('orders.id', ondelete='CASCADE'), primary_key=True)
    product_id = Column(Integer, ForeignKey('products.id'), primary_key=True)
    quantity = Column(Integer, nullable=False, default=1)
    unit_price = Column(Float, nullable=False)

    # Relationships
    order = relationship('Order', back_populates='order_items')
    product = relationship('Product', back_populates='order_items')

    @property
    def subtotal(self) -> float:
        """Satır toplamı"""
        return self.quantity * self.unit_price

class Review(Base, TimestampMixin):
    """Ürün değerlendirme modeli"""
    __tablename__ = 'reviews'

    id = Column(Integer, primary_key=True)
    product_id = Column(Integer, ForeignKey('products.id', ondelete='CASCADE'), nullable=False)
    user_id = Column(Integer, nullable=False)
    rating = Column(Enum(Rating), nullable=False)
    comment = Column(Text, nullable=True)

    # Relationships
    product = relationship('Product', back_populates='reviews')

    # Indexes
    __table_args__ = (
        Index('idx_review_product_user', 'product_id', 'user_id', unique=True),
    )


# ============================================================================
# EXERCISE 2: Advanced Query Builder Pattern
# ============================================================================
"""
TODO: ProductQueryBuilder sınıfı oluşturun:
- Filtreleme: category, price range, in stock, rating
- Sıralama: price, rating, created_at
- Sayfalama
- Eager loading ile reviews ve category

Method chaining kullanarak fluent API tasarlayın.
"""

# SOLUTION:

class ProductQueryBuilder:
    """Advanced product query builder with method chaining"""

    def __init__(self, session):
        self.session = session
        self.query = session.query(Product)
        self.filters = []

    def in_category(self, category_id: int):
        """Kategoriye göre filtrele"""
        self.filters.append(Product.category_id == category_id)
        return self

    def price_range(self, min_price: float, max_price: float):
        """Fiyat aralığında filtrele"""
        self.filters.append(and_(
            Product.price >= min_price,
            Product.price <= max_price
        ))
        return self

    def in_stock_only(self):
        """Sadece stokta olanlar"""
        self.filters.append(Product.stock > 0)
        self.filters.append(Product.is_deleted == False)
        return self

    def min_rating(self, min_rating: float):
        """Minimum rating'e göre filtrele"""
        # Subquery ile ortalama rating hesapla
        avg_rating_subquery = (
            self.session.query(
                Review.product_id,
                func.avg(Review.rating).label('avg_rating')
            )
            .group_by(Review.product_id)
            .subquery()
        )

        self.query = self.query.outerjoin(
            avg_rating_subquery,
            Product.id == avg_rating_subquery.c.product_id
        )
        self.filters.append(avg_rating_subquery.c.avg_rating >= min_rating)
        return self

    def search(self, term: str):
        """İsim veya açıklamada arama"""
        search_filter = or_(
            Product.name.ilike(f'%{term}%'),
            Product.description.ilike(f'%{term}%')
        )
        self.filters.append(search_filter)
        return self

    def order_by_price(self, ascending: bool = True):
        """Fiyata göre sırala"""
        if ascending:
            self.query = self.query.order_by(Product.price.asc())
        else:
            self.query = self.query.order_by(Product.price.desc())
        return self

    def order_by_rating(self):
        """Rating'e göre sırala (en yüksek önce)"""
        # Subquery ile ortalama rating
        avg_rating = (
            self.session.query(
                Review.product_id,
                func.avg(Review.rating).label('avg_rating')
            )
            .group_by(Review.product_id)
            .subquery()
        )

        self.query = (
            self.query
            .outerjoin(avg_rating, Product.id == avg_rating.c.product_id)
            .order_by(avg_rating.c.avg_rating.desc().nullslast())
        )
        return self

    def with_details(self):
        """Eager loading - reviews ve category"""
        from sqlalchemy.orm import selectinload
        self.query = self.query.options(
            selectinload(Product.reviews),
            selectinload(Product.category)
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

    def all(self):
        """Tüm sonuçları getir"""
        return self.build().all()

    def count(self):
        """Toplam sonuç sayısı"""
        return self.build().count()

    def first(self):
        """İlk sonucu getir"""
        return self.build().first()


# Test ProductQueryBuilder
def test_product_query_builder(session):
    """ProductQueryBuilder test fonksiyonu"""
    builder = ProductQueryBuilder(session)

    # 100-500 TL arası, stokta olan, Electronics kategorisindeki ürünler
    products = (
        builder
        .in_category(1)
        .price_range(100, 500)
        .in_stock_only()
        .with_details()
        .order_by_price(ascending=True)
        .paginate(page=1, per_page=10)
        .all()
    )

    for product in products:
        print(f"{product.name} - {product.price} TL - Stock: {product.stock}")
        print(f"Category: {' > '.join(product.category.get_path())}")
        print(f"Average Rating: {product.average_rating:.1f}")


# ============================================================================
# EXERCISE 3: Transaction Management - Order Processing
# ============================================================================
"""
TODO: Sipariş oluşturma sistemi geliştirin:
1. Stok kontrolü yapın
2. Transaction içinde sipariş ve sipariş kalemlerini oluşturun
3. Stoktan düşün
4. Hata durumunda rollback yapın
5. Pessimistic locking kullanın

create_order(user_id, items: List[Dict]) fonksiyonu yazın.
items = [{'product_id': 1, 'quantity': 2}, ...]
"""

# SOLUTION:

class OrderService:
    """Sipariş işlemleri için servis katmanı"""

    def __init__(self, session):
        self.session = session

    @contextmanager
    def transaction_scope(self):
        """Transaction context manager"""
        try:
            yield self.session
            self.session.commit()
        except Exception as e:
            self.session.rollback()
            logging.error(f"Transaction failed: {e}")
            raise
        finally:
            self.session.close()

    def create_order(self, user_id: int, items: List[Dict[str, int]]) -> Order:
        """
        Sipariş oluşturma - atomic operation

        Args:
            user_id: Kullanıcı ID
            items: [{'product_id': 1, 'quantity': 2}, ...]

        Returns:
            Order objesi

        Raises:
            ValueError: Stok yetersizse veya ürün bulunamazsa
        """
        with self.transaction_scope():
            # Yeni sipariş oluştur
            order = Order(user_id=user_id, status=OrderStatus.PENDING)
            self.session.add(order)
            self.session.flush()  # order.id'yi al

            total_amount = 0.0

            for item_data in items:
                product_id = item_data['product_id']
                quantity = item_data['quantity']

                # Pessimistic locking ile ürünü getir
                product = (
                    self.session.query(Product)
                    .filter(Product.id == product_id)
                    .filter(Product.is_deleted == False)
                    .with_for_update()  # SELECT ... FOR UPDATE
                    .first()
                )

                if not product:
                    raise ValueError(f"Product {product_id} not found")

                # Stok kontrolü
                if product.stock < quantity:
                    raise ValueError(
                        f"Insufficient stock for {product.name}. "
                        f"Available: {product.stock}, Requested: {quantity}"
                    )

                # Sipariş kalemi oluştur
                order_item = OrderItem(
                    order_id=order.id,
                    product_id=product_id,
                    quantity=quantity,
                    unit_price=product.price
                )
                self.session.add(order_item)

                # Stoktan düş
                product.stock -= quantity
                total_amount += order_item.subtotal

            # Toplam tutarı güncelle
            order.total_amount = total_amount

            return order

    def cancel_order(self, order_id: int) -> bool:
        """
        Siparişi iptal et ve stokları geri yükle

        Args:
            order_id: Sipariş ID

        Returns:
            Başarılı ise True
        """
        with self.transaction_scope():
            # Siparişi getir
            order = (
                self.session.query(Order)
                .filter(Order.id == order_id)
                .with_for_update()
                .first()
            )

            if not order:
                raise ValueError("Order not found")

            if order.status == OrderStatus.DELIVERED:
                raise ValueError("Cannot cancel delivered order")

            # Stokları geri yükle
            for item in order.order_items:
                product = (
                    self.session.query(Product)
                    .filter(Product.id == item.product_id)
                    .with_for_update()
                    .first()
                )

                if product:
                    product.stock += item.quantity

            # Sipariş durumunu güncelle
            order.status = OrderStatus.CANCELLED

            return True

    def get_order_summary(self, order_id: int) -> Dict[str, Any]:
        """Sipariş özeti döndür"""
        from sqlalchemy.orm import joinedload

        order = (
            self.session.query(Order)
            .options(
                joinedload(Order.order_items).joinedload(OrderItem.product)
            )
            .filter(Order.id == order_id)
            .first()
        )

        if not order:
            return None

        return {
            'order_id': order.id,
            'user_id': order.user_id,
            'status': order.status.value,
            'total_amount': order.total_amount,
            'created_at': order.created_at.isoformat(),
            'items': [
                {
                    'product_name': item.product.name,
                    'quantity': item.quantity,
                    'unit_price': item.unit_price,
                    'subtotal': item.subtotal
                }
                for item in order.order_items
            ]
        }


# Test OrderService
def test_order_service(session):
    """OrderService test fonksiyonu"""
    service = OrderService(session)

    # Sipariş oluştur
    try:
        order = service.create_order(
            user_id=1,
            items=[
                {'product_id': 1, 'quantity': 2},
                {'product_id': 2, 'quantity': 1},
            ]
        )
        print(f"Order created: {order.id}")
        print(f"Total: {order.total_amount} TL")

        # Sipariş özeti
        summary = service.get_order_summary(order.id)
        print(f"Order summary: {summary}")

        # Siparişi iptal et
        service.cancel_order(order.id)
        print("Order cancelled successfully")

    except ValueError as e:
        print(f"Error: {e}")


# ============================================================================
# EXERCISE 4: Complex Aggregation Queries
# ============================================================================
"""
TODO: Gelişmiş raporlama sorguları yazın:
1. En çok satan ürünler (son 30 gün)
2. Kategoriye göre toplam satış
3. Kullanıcıların ortalama sipariş tutarı
4. Günlük satış trendi
5. En çok yorum yapan kullanıcılar

Window functions ve subquery kullanın.
"""

# SOLUTION:

class ReportingService:
    """Raporlama ve analitik sorgular"""

    def __init__(self, session):
        self.session = session

    def get_top_selling_products(self, days: int = 30, limit: int = 10) -> List[Dict]:
        """
        En çok satan ürünler

        Returns:
            [{'product_id', 'product_name', 'total_quantity', 'total_revenue'}, ...]
        """
        since_date = datetime.utcnow() - timedelta(days=days)

        query = (
            self.session.query(
                Product.id.label('product_id'),
                Product.name.label('product_name'),
                func.sum(OrderItem.quantity).label('total_quantity'),
                func.sum(OrderItem.quantity * OrderItem.unit_price).label('total_revenue'),
                func.count(func.distinct(Order.id)).label('order_count')
            )
            .join(OrderItem, Product.id == OrderItem.product_id)
            .join(Order, OrderItem.order_id == Order.id)
            .filter(Order.created_at >= since_date)
            .filter(Order.status.in_([OrderStatus.DELIVERED, OrderStatus.SHIPPED]))
            .group_by(Product.id, Product.name)
            .order_by(func.sum(OrderItem.quantity).desc())
            .limit(limit)
        )

        results = []
        for row in query.all():
            results.append({
                'product_id': row.product_id,
                'product_name': row.product_name,
                'total_quantity': row.total_quantity,
                'total_revenue': float(row.total_revenue),
                'order_count': row.order_count
            })

        return results

    def get_sales_by_category(self) -> List[Dict]:
        """
        Kategoriye göre satış istatistikleri

        Returns:
            [{'category_name', 'total_products', 'total_sales', 'avg_product_price'}, ...]
        """
        query = (
            self.session.query(
                Category.name.label('category_name'),
                func.count(func.distinct(Product.id)).label('total_products'),
                func.sum(OrderItem.quantity * OrderItem.unit_price).label('total_sales'),
                func.avg(Product.price).label('avg_product_price')
            )
            .join(Product, Category.id == Product.category_id)
            .outerjoin(OrderItem, Product.id == OrderItem.product_id)
            .group_by(Category.id, Category.name)
            .order_by(func.sum(OrderItem.quantity * OrderItem.unit_price).desc().nullslast())
        )

        results = []
        for row in query.all():
            results.append({
                'category_name': row.category_name,
                'total_products': row.total_products,
                'total_sales': float(row.total_sales) if row.total_sales else 0.0,
                'avg_product_price': float(row.avg_product_price)
            })

        return results

    def get_user_order_statistics(self, user_id: int) -> Dict:
        """
        Kullanıcının sipariş istatistikleri

        Returns:
            {'total_orders', 'total_spent', 'avg_order_amount', 'first_order', 'last_order'}
        """
        stats = (
            self.session.query(
                func.count(Order.id).label('total_orders'),
                func.sum(Order.total_amount).label('total_spent'),
                func.avg(Order.total_amount).label('avg_order_amount'),
                func.min(Order.created_at).label('first_order'),
                func.max(Order.created_at).label('last_order')
            )
            .filter(Order.user_id == user_id)
            .filter(Order.status != OrderStatus.CANCELLED)
            .first()
        )

        return {
            'total_orders': stats.total_orders or 0,
            'total_spent': float(stats.total_spent) if stats.total_spent else 0.0,
            'avg_order_amount': float(stats.avg_order_amount) if stats.avg_order_amount else 0.0,
            'first_order': stats.first_order.isoformat() if stats.first_order else None,
            'last_order': stats.last_order.isoformat() if stats.last_order else None
        }

    def get_daily_sales_trend(self, days: int = 30) -> List[Dict]:
        """
        Günlük satış trendi

        Returns:
            [{'date', 'total_orders', 'total_revenue', 'avg_order_amount'}, ...]
        """
        since_date = datetime.utcnow() - timedelta(days=days)

        query = (
            self.session.query(
                func.date(Order.created_at).label('date'),
                func.count(Order.id).label('total_orders'),
                func.sum(Order.total_amount).label('total_revenue'),
                func.avg(Order.total_amount).label('avg_order_amount')
            )
            .filter(Order.created_at >= since_date)
            .filter(Order.status != OrderStatus.CANCELLED)
            .group_by(func.date(Order.created_at))
            .order_by(func.date(Order.created_at))
        )

        results = []
        for row in query.all():
            results.append({
                'date': str(row.date),
                'total_orders': row.total_orders,
                'total_revenue': float(row.total_revenue),
                'avg_order_amount': float(row.avg_order_amount)
            })

        return results

    def get_top_reviewers(self, limit: int = 10) -> List[Dict]:
        """
        En çok yorum yapan kullanıcılar

        Returns:
            [{'user_id', 'total_reviews', 'avg_rating'}, ...]
        """
        query = (
            self.session.query(
                Review.user_id,
                func.count(Review.id).label('total_reviews'),
                func.avg(Review.rating).label('avg_rating')
            )
            .group_by(Review.user_id)
            .order_by(func.count(Review.id).desc())
            .limit(limit)
        )

        results = []
        for row in query.all():
            results.append({
                'user_id': row.user_id,
                'total_reviews': row.total_reviews,
                'avg_rating': float(row.avg_rating)
            })

        return results


# Test ReportingService
def test_reporting_service(session):
    """ReportingService test fonksiyonu"""
    reporting = ReportingService(session)

    # En çok satan ürünler
    print("\n=== Top Selling Products ===")
    top_products = reporting.get_top_selling_products(days=30, limit=5)
    for product in top_products:
        print(f"{product['product_name']}: {product['total_quantity']} sold, "
              f"{product['total_revenue']} TL revenue")

    # Kategoriye göre satışlar
    print("\n=== Sales by Category ===")
    category_sales = reporting.get_sales_by_category()
    for cat in category_sales:
        print(f"{cat['category_name']}: {cat['total_sales']} TL")

    # Günlük trend
    print("\n=== Daily Sales Trend ===")
    trend = reporting.get_daily_sales_trend(days=7)
    for day in trend:
        print(f"{day['date']}: {day['total_orders']} orders, {day['total_revenue']} TL")


# ============================================================================
# EXERCISE 5: Query Optimization - N+1 Problem
# ============================================================================
"""
TODO: N+1 query problemini gösterin ve çözün:
1. Kötü örnek: Lazy loading ile her ürün için ayrı query
2. İyi örnek: Eager loading ile tek query
3. Her iki yöntemi benchmark edin
4. joinedload, selectinload, subqueryload farklarını gösterin
"""

# SOLUTION:

import time

class PerformanceDemo:
    """Query performance demonstration"""

    def __init__(self, session):
        self.session = session

    def bad_approach_lazy_loading(self) -> float:
        """
        KÖTÜ: Lazy loading - N+1 query problem
        Her ürün için ayrı review query'si
        """
        start = time.time()

        products = self.session.query(Product).limit(20).all()

        for product in products:
            # Her iterasyonda yeni query! (N+1 problem)
            review_count = len(product.reviews)
            category_name = product.category.name
            print(f"{product.name}: {review_count} reviews, Category: {category_name}")

        duration = time.time() - start
        print(f"\nLazy loading duration: {duration:.3f}s")
        return duration

    def good_approach_selectinload(self) -> float:
        """
        İYİ: selectinload kullanımı
        2 query ile tüm data (products + reviews)
        """
        from sqlalchemy.orm import selectinload

        start = time.time()

        products = (
            self.session.query(Product)
            .options(
                selectinload(Product.reviews),
                selectinload(Product.category)
            )
            .limit(20)
            .all()
        )

        for product in products:
            # Ek query yok - data zaten yüklü
            review_count = len(product.reviews)
            category_name = product.category.name
            print(f"{product.name}: {review_count} reviews, Category: {category_name}")

        duration = time.time() - start
        print(f"\nselectinload duration: {duration:.3f}s")
        return duration

    def good_approach_joinedload(self) -> float:
        """
        İYİ: joinedload kullanımı
        JOIN ile tek query
        """
        from sqlalchemy.orm import joinedload

        start = time.time()

        products = (
            self.session.query(Product)
            .options(
                joinedload(Product.category)
            )
            .limit(20)
            .all()
        )

        for product in products:
            category_name = product.category.name
            print(f"{product.name}: Category: {category_name}")

        duration = time.time() - start
        print(f"\njoinedload duration: {duration:.3f}s")
        return duration

    def benchmark_comparison(self):
        """Performance karşılaştırması"""
        print("\n" + "="*60)
        print("PERFORMANCE BENCHMARK")
        print("="*60)

        # Lazy loading
        print("\n1. Lazy Loading (BAD)")
        lazy_time = self.bad_approach_lazy_loading()

        # selectinload
        print("\n2. selectinload (GOOD)")
        select_time = self.good_approach_selectinload()

        # joinedload
        print("\n3. joinedload (GOOD)")
        joined_time = self.good_approach_joinedload()

        # Karşılaştırma
        print("\n" + "="*60)
        print(f"Lazy loading: {lazy_time:.3f}s (baseline)")
        print(f"selectinload: {select_time:.3f}s ({lazy_time/select_time:.1f}x faster)")
        print(f"joinedload: {joined_time:.3f}s ({lazy_time/joined_time:.1f}x faster)")
        print("="*60)


# ============================================================================
# EXERCISE 6: Database Connection Pool Management
# ============================================================================
"""
TODO: Production-ready database manager oluşturun:
1. QueuePool ile connection pooling
2. Connection monitoring ve metrics
3. Health check endpoint
4. Slow query logging
5. Event listeners ile connection lifecycle yönetimi
"""

# SOLUTION:

class ProductionDatabaseManager:
    """Production için database manager"""

    def __init__(self, database_url: str):
        # Logging setup
        self.logger = logging.getLogger(__name__)

        # Engine oluşturma
        self.engine = create_engine(
            database_url,
            poolclass=QueuePool,
            pool_size=20,  # Minimum connection
            max_overflow=10,  # Extra connection limit
            pool_timeout=30,  # Connection wait timeout
            pool_recycle=3600,  # Recycle after 1 hour
            pool_pre_ping=True,  # Test connection before use
            echo=False,  # SQL logging (production'da False)
        )

        # Session factory
        session_factory = sessionmaker(
            bind=self.engine,
            autocommit=False,
            autoflush=False,
            expire_on_commit=False
        )

        self.Session = scoped_session(session_factory)

        # Metrics
        self.metrics = {
            'total_connections': 0,
            'total_queries': 0,
            'slow_queries': 0,
            'errors': 0,
        }

        # Event listeners
        self._setup_event_listeners()

        self.logger.info("Database manager initialized")

    def _setup_event_listeners(self):
        """Event listener'ları kur"""
        from sqlalchemy import event

        @event.listens_for(self.engine, "connect")
        def receive_connect(dbapi_conn, connection_record):
            """Yeni connection oluşturulduğunda"""
            self.metrics['total_connections'] += 1
            connection_record.info['connect_time'] = time.time()
            self.logger.debug("New database connection established")

        @event.listens_for(self.engine, "checkout")
        def receive_checkout(dbapi_conn, connection_record, connection_proxy):
            """Connection pool'dan alındığında"""
            connection_record.info['checkout_time'] = time.time()

        @event.listens_for(self.engine, "checkin")
        def receive_checkin(dbapi_conn, connection_record):
            """Connection pool'a geri verildiğinde"""
            checkout_time = connection_record.info.get('checkout_time')
            if checkout_time:
                duration = time.time() - checkout_time
                if duration > 5.0:  # 5 saniyeden uzun
                    self.logger.warning(f"Long connection checkout: {duration:.2f}s")

        @event.listens_for(self.engine, "before_cursor_execute")
        def before_cursor_execute(conn, cursor, statement, parameters, context, executemany):
            """Query çalıştırılmadan önce"""
            context._query_start_time = time.time()
            self.metrics['total_queries'] += 1

        @event.listens_for(self.engine, "after_cursor_execute")
        def after_cursor_execute(conn, cursor, statement, parameters, context, executemany):
            """Query çalıştırıldıktan sonra"""
            total_time = time.time() - context._query_start_time

            if total_time > 1.0:  # 1 saniyeden uzun
                self.metrics['slow_queries'] += 1
                self.logger.warning(
                    f"Slow query ({total_time:.2f}s): {statement[:200]}"
                )

    @contextmanager
    def session_scope(self):
        """Session context manager - auto commit/rollback"""
        session = self.Session()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            self.metrics['errors'] += 1
            self.logger.error(f"Database error: {e}")
            raise
        finally:
            session.close()

    def get_session(self):
        """Yeni session döndür"""
        return self.Session()

    def close_session(self):
        """Mevcut thread'in session'ını kapat"""
        self.Session.remove()

    def health_check(self) -> Dict[str, Any]:
        """Database health check"""
        try:
            with self.session_scope() as session:
                session.execute("SELECT 1")

            pool = self.engine.pool
            pool_status = {
                'pool_size': pool.size(),
                'checked_out': pool.checkedout(),
                'overflow': pool.overflow(),
                'available': pool.size() - pool.checkedout()
            }

            return {
                'status': 'healthy',
                'pool': pool_status,
                'metrics': self.metrics
            }
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return {
                'status': 'unhealthy',
                'error': str(e)
            }

    def get_pool_stats(self) -> Dict[str, int]:
        """Connection pool istatistikleri"""
        pool = self.engine.pool
        return {
            'pool_size': pool.size(),
            'checked_out_connections': pool.checkedout(),
            'overflow_connections': pool.overflow(),
            'available_connections': pool.size() - pool.checkedout(),
            'total_connections': self.metrics['total_connections'],
            'total_queries': self.metrics['total_queries'],
            'slow_queries': self.metrics['slow_queries'],
            'errors': self.metrics['errors']
        }


# Test ProductionDatabaseManager
def test_database_manager():
    """Database manager test"""
    db = ProductionDatabaseManager('sqlite:///ecommerce.db')

    # Tabloları oluştur
    Base.metadata.create_all(db.engine)

    # Health check
    print("\n=== Health Check ===")
    health = db.health_check()
    print(f"Status: {health['status']}")
    print(f"Pool stats: {health['pool']}")

    # Session kullanımı
    with db.session_scope() as session:
        category = Category(name='Electronics')
        session.add(category)

    # Pool istatistikleri
    print("\n=== Pool Statistics ===")
    stats = db.get_pool_stats()
    for key, value in stats.items():
        print(f"{key}: {value}")


# ============================================================================
# EXERCISE 7: Raw SQL vs ORM Hybrid Approach
# ============================================================================
"""
TODO: Hybrid repository pattern oluşturun:
1. Basit CRUD için ORM kullanın
2. Complex analytic query'ler için raw SQL
3. Bulk operations için raw SQL
4. Type-safe sonuçlar döndürün
"""

# SOLUTION:

from sqlalchemy import text
from typing import List, Dict, Any

class HybridProductRepository:
    """ORM ve Raw SQL'i birlikte kullanan repository"""

    def __init__(self, session):
        self.session = session

    # ===== ORM Methods =====

    def create(self, product_data: Dict[str, Any]) -> Product:
        """ORM ile create - type-safe"""
        product = Product(**product_data)
        self.session.add(product)
        self.session.commit()
        return product

    def get_by_id(self, product_id: int) -> Optional[Product]:
        """ORM ile get by id"""
        return self.session.query(Product).filter(Product.id == product_id).first()

    def update(self, product_id: int, update_data: Dict[str, Any]) -> bool:
        """ORM ile update"""
        result = (
            self.session.query(Product)
            .filter(Product.id == product_id)
            .update(update_data)
        )
        self.session.commit()
        return result > 0

    def soft_delete(self, product_id: int) -> bool:
        """ORM ile soft delete"""
        product = self.get_by_id(product_id)
        if product:
            product.soft_delete()
            self.session.commit()
            return True
        return False

    # ===== Raw SQL Methods - Complex Queries =====

    def get_product_analytics(self, product_id: int) -> Optional[Dict]:
        """
        Complex analytic query - Raw SQL kullanarak
        Ürün için detaylı istatistikler
        """
        query = text("""
            WITH product_stats AS (
                SELECT
                    p.id,
                    p.name,
                    p.price,
                    p.stock,
                    COUNT(DISTINCT r.id) as review_count,
                    AVG(CAST(r.rating AS FLOAT)) as avg_rating,
                    COUNT(DISTINCT oi.order_id) as total_orders,
                    SUM(oi.quantity) as total_sold,
                    SUM(oi.quantity * oi.unit_price) as total_revenue
                FROM products p
                LEFT JOIN reviews r ON p.id = r.product_id
                LEFT JOIN order_items oi ON p.id = oi.product_id
                WHERE p.id = :product_id
                GROUP BY p.id, p.name, p.price, p.stock
            )
            SELECT * FROM product_stats
        """)

        result = self.session.execute(query, {'product_id': product_id}).fetchone()

        if result:
            return {
                'id': result.id,
                'name': result.name,
                'price': result.price,
                'stock': result.stock,
                'review_count': result.review_count,
                'avg_rating': float(result.avg_rating) if result.avg_rating else 0.0,
                'total_orders': result.total_orders,
                'total_sold': result.total_sold or 0,
                'total_revenue': float(result.total_revenue) if result.total_revenue else 0.0
            }
        return None

    def bulk_update_prices(self, category_id: int, discount_percent: float):
        """
        Bulk update - Raw SQL ile daha hızlı
        Bir kategorideki tüm ürünlere indirim uygula
        """
        query = text("""
            UPDATE products
            SET price = price * (1 - :discount / 100.0),
                updated_at = CURRENT_TIMESTAMP
            WHERE category_id = :category_id
              AND is_deleted = 0
        """)

        result = self.session.execute(
            query,
            {'category_id': category_id, 'discount': discount_percent}
        )
        self.session.commit()
        return result.rowcount

    def get_low_stock_products(self, threshold: int = 10) -> List[Dict]:
        """
        Raw SQL ile low stock uyarısı
        """
        query = text("""
            SELECT
                p.id,
                p.name,
                p.stock,
                p.price,
                c.name as category_name,
                COALESCE(SUM(oi.quantity), 0) as total_sold_30d
            FROM products p
            LEFT JOIN categories c ON p.category_id = c.id
            LEFT JOIN order_items oi ON p.id = oi.product_id
            LEFT JOIN orders o ON oi.order_id = o.id
                AND o.created_at >= datetime('now', '-30 days')
            WHERE p.stock <= :threshold
              AND p.is_deleted = 0
            GROUP BY p.id, p.name, p.stock, p.price, c.name
            ORDER BY p.stock ASC
        """)

        results = []
        for row in self.session.execute(query, {'threshold': threshold}):
            results.append({
                'id': row.id,
                'name': row.name,
                'stock': row.stock,
                'price': row.price,
                'category_name': row.category_name,
                'sold_last_30_days': row.total_sold_30d
            })

        return results


# Test HybridProductRepository
def test_hybrid_repository(session):
    """Hybrid repository test"""
    repo = HybridProductRepository(session)

    # ORM - Create
    product = repo.create({
        'name': 'Laptop',
        'price': 1500.0,
        'stock': 10,
        'category_id': 1
    })
    print(f"Created product: {product.id}")

    # Raw SQL - Analytics
    analytics = repo.get_product_analytics(product.id)
    print(f"Analytics: {analytics}")

    # Raw SQL - Bulk update
    affected = repo.bulk_update_prices(category_id=1, discount_percent=10)
    print(f"Updated {affected} products")

    # Raw SQL - Low stock
    low_stock = repo.get_low_stock_products(threshold=20)
    print(f"Low stock products: {len(low_stock)}")


# ============================================================================
# EXERCISE 8: Database Migration with Alembic Simulation
# ============================================================================
"""
TODO: Migration pattern simülasyonu:
1. Schema version tracking
2. Up/Down migration simulation
3. Data migration örneği
4. Rollback scenario
"""

# SOLUTION:

class MigrationSimulator:
    """Alembic-style migration simulator"""

    def __init__(self, session):
        self.session = session
        self.migrations = []
        self._create_migration_table()

    def _create_migration_table(self):
        """Migration tracking tablosu oluştur"""
        self.session.execute(text("""
            CREATE TABLE IF NOT EXISTS schema_migrations (
                version VARCHAR(50) PRIMARY KEY,
                applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """))
        self.session.commit()

    def register_migration(self, version: str, upgrade_func, downgrade_func):
        """Migration kaydı"""
        self.migrations.append({
            'version': version,
            'upgrade': upgrade_func,
            'downgrade': downgrade_func
        })

    def get_current_version(self) -> Optional[str]:
        """Mevcut schema version"""
        result = self.session.execute(text("""
            SELECT version FROM schema_migrations
            ORDER BY applied_at DESC LIMIT 1
        """)).fetchone()

        return result[0] if result else None

    def migrate_up(self, target_version: Optional[str] = None):
        """Upgrade migration"""
        current = self.get_current_version()
        print(f"Current version: {current}")

        for migration in self.migrations:
            version = migration['version']

            # Skip if already applied
            if current and version <= current:
                continue

            # Apply migration
            print(f"Applying migration: {version}")
            try:
                migration['upgrade'](self.session)

                # Mark as applied
                self.session.execute(
                    text("INSERT INTO schema_migrations (version) VALUES (:version)"),
                    {'version': version}
                )
                self.session.commit()
                print(f"✓ Migration {version} applied successfully")

                # Stop if target reached
                if target_version and version == target_version:
                    break

            except Exception as e:
                self.session.rollback()
                print(f"✗ Migration {version} failed: {e}")
                raise

    def migrate_down(self, target_version: str):
        """Downgrade migration"""
        current = self.get_current_version()
        print(f"Current version: {current}")

        # Reverse order
        for migration in reversed(self.migrations):
            version = migration['version']

            # Skip if not applied
            if not current or version > current:
                continue

            # Skip if target reached
            if version <= target_version:
                break

            # Rollback migration
            print(f"Rolling back migration: {version}")
            try:
                migration['downgrade'](self.session)

                # Remove from tracking
                self.session.execute(
                    text("DELETE FROM schema_migrations WHERE version = :version"),
                    {'version': version}
                )
                self.session.commit()
                print(f"✓ Migration {version} rolled back successfully")

            except Exception as e:
                self.session.rollback()
                print(f"✗ Rollback {version} failed: {e}")
                raise


# Migration fonksiyonları
def migration_001_add_product_sku_upgrade(session):
    """Add SKU column to products"""
    session.execute(text("""
        ALTER TABLE products ADD COLUMN sku VARCHAR(50)
    """))

def migration_001_add_product_sku_downgrade(session):
    """Remove SKU column from products"""
    session.execute(text("""
        ALTER TABLE products DROP COLUMN sku
    """))

def migration_002_add_product_tags_upgrade(session):
    """Add tags column to products"""
    session.execute(text("""
        ALTER TABLE products ADD COLUMN tags TEXT
    """))

def migration_002_add_product_tags_downgrade(session):
    """Remove tags column"""
    session.execute(text("""
        ALTER TABLE products DROP COLUMN tags
    """))


# Test MigrationSimulator
def test_migrations(session):
    """Migration test"""
    migrator = MigrationSimulator(session)

    # Register migrations
    migrator.register_migration(
        '001_add_product_sku',
        migration_001_add_product_sku_upgrade,
        migration_001_add_product_sku_downgrade
    )

    migrator.register_migration(
        '002_add_product_tags',
        migration_002_add_product_tags_upgrade,
        migration_002_add_product_tags_downgrade
    )

    # Apply all migrations
    print("\n=== Migrating Up ===")
    migrator.migrate_up()

    # Rollback one migration
    print("\n=== Migrating Down ===")
    migrator.migrate_down('001_add_product_sku')


# ============================================================================
# MAIN - Test All Exercises
# ============================================================================

def main():
    """Tüm exercise'ları test et"""

    # Database setup
    engine = create_engine('sqlite:///ecommerce_advanced.db', echo=False)
    Base.metadata.create_all(engine)

    Session = sessionmaker(bind=engine)
    session = Session()

    print("="*80)
    print("DATABASE PROGRAMMING - ADVANCED EXERCISES")
    print("="*80)

    # Sample data oluştur
    print("\n[1] Creating sample data...")
    create_sample_data(session)

    # Test Query Builder
    print("\n[2] Testing ProductQueryBuilder...")
    test_product_query_builder(session)

    # Test Order Service
    print("\n[3] Testing OrderService...")
    test_order_service(session)

    # Test Reporting
    print("\n[4] Testing ReportingService...")
    test_reporting_service(session)

    # Test Performance
    print("\n[5] Testing Performance (N+1 Problem)...")
    perf = PerformanceDemo(session)
    perf.benchmark_comparison()

    # Test Database Manager
    print("\n[6] Testing ProductionDatabaseManager...")
    test_database_manager()

    # Test Hybrid Repository
    print("\n[7] Testing HybridProductRepository...")
    test_hybrid_repository(session)

    # Test Migrations
    print("\n[8] Testing MigrationSimulator...")
    test_migrations(session)

    session.close()
    print("\n" + "="*80)
    print("ALL EXERCISES COMPLETED!")
    print("="*80)


def create_sample_data(session):
    """Test için örnek data oluştur"""
    try:
        # Categories
        electronics = Category(name='Electronics')
        computers = Category(name='Computers', parent=electronics)
        phones = Category(name='Phones', parent=electronics)

        session.add_all([electronics, computers, phones])
        session.flush()

        # Products
        laptop = Product(
            name='Gaming Laptop',
            description='High-end gaming laptop',
            price=1500.0,
            stock=10,
            category_id=computers.id
        )

        phone = Product(
            name='Smartphone',
            description='Latest model smartphone',
            price=800.0,
            stock=20,
            category_id=phones.id
        )

        session.add_all([laptop, phone])
        session.flush()

        # Reviews
        review1 = Review(
            product_id=laptop.id,
            user_id=1,
            rating=Rating.FIVE,
            comment='Excellent product!'
        )

        review2 = Review(
            product_id=laptop.id,
            user_id=2,
            rating=Rating.FOUR,
            comment='Good but expensive'
        )

        session.add_all([review1, review2])

        session.commit()
        print("Sample data created successfully!")

    except Exception as e:
        session.rollback()
        print(f"Error creating sample data: {e}")


if __name__ == '__main__':
    main()
