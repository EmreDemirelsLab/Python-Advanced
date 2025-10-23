"""
Web Scraping - İleri Düzey Egzersizler

Bu dosya, web scraping konusunda ileri düzey egzersizler içerir.
Her egzersiz gerçek dünya senaryolarını simüle eder.

Konular:
- E-commerce scraping
- News aggregation
- API data collection
- Dynamic content handling
- Rate limiting
- Error handling
- Session management
- Distributed scraping
"""

import requests
from bs4 import BeautifulSoup
from typing import List, Dict, Optional
import time
import json
from datetime import datetime
import logging

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# EGZERSIZ 1: E-Commerce Product Scraper (Medium)
# ============================================================================
"""
Senaryo: Bir e-commerce sitesinden ürün bilgilerini scrape eden bir sınıf yazın.

Gereksinimler:
- Ürün adı, fiyat, rating, stok durumu extract edin
- Birden fazla sayfadan veri toplayın (pagination)
- Rate limiting uygulayın
- Verileri JSON olarak kaydedin

TODO: ProductScraper sınıfını implement edin
"""

class ProductScraper:
    """E-commerce ürün scraper'ı"""

    def __init__(self, base_url: str, rate_limit: float = 1.0):
        # TODO: Session oluştur
        # TODO: Rate limiter ekle
        # TODO: Headers yapılandır
        pass

    def scrape_product(self, product_url: str) -> Dict:
        """
        Tek bir ürün sayfasından bilgi extract et

        Args:
            product_url: Ürün sayfası URL'i

        Returns:
            Dict: Ürün bilgileri (name, price, rating, in_stock)
        """
        # TODO: Sayfayı fetch et
        # TODO: BeautifulSoup ile parse et
        # TODO: Ürün bilgilerini extract et
        # TODO: Structured data döndür
        pass

    def scrape_category(self, category_url: str, max_pages: int = 3) -> List[Dict]:
        """
        Kategori sayfasından tüm ürünleri scrape et

        Args:
            category_url: Kategori URL'i
            max_pages: Maksimum sayfa sayısı

        Returns:
            List[Dict]: Tüm ürünlerin bilgileri
        """
        # TODO: Pagination handle et
        # TODO: Her sayfadaki ürünleri topla
        # TODO: Rate limiting uygula
        # TODO: Tüm ürünleri döndür
        pass

    def save_to_json(self, products: List[Dict], filename: str):
        """Ürünleri JSON dosyasına kaydet"""
        # TODO: JSON dosyasına yaz
        pass


# ÇÖZÜM:
class ProductScraperSolution:
    """E-commerce ürün scraper'ı - Çözüm"""

    def __init__(self, base_url: str, rate_limit: float = 1.0):
        self.base_url = base_url
        self.rate_limit = rate_limit
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        self.last_request_time = 0

    def _rate_limit_wait(self):
        """Rate limiting için bekle"""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.rate_limit:
            time.sleep(self.rate_limit - elapsed)
        self.last_request_time = time.time()

    def scrape_product(self, product_url: str) -> Dict:
        """Tek bir ürün sayfasından bilgi extract et"""
        self._rate_limit_wait()

        try:
            response = self.session.get(product_url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'lxml')

            # Extract product data
            product = {
                'url': product_url,
                'name': soup.find('h1', class_='product-title').get_text(strip=True),
                'price': self._extract_price(soup),
                'rating': self._extract_rating(soup),
                'in_stock': self._check_stock(soup),
                'scraped_at': datetime.now().isoformat()
            }

            return product

        except Exception as e:
            logger.error(f"Error scraping {product_url}: {e}")
            return None

    def _extract_price(self, soup):
        """Fiyat extract et"""
        price_elem = soup.find('span', class_='price')
        if price_elem:
            # "$19.99" -> 19.99
            price_text = price_elem.get_text(strip=True)
            return float(price_text.replace('$', '').replace(',', ''))
        return None

    def _extract_rating(self, soup):
        """Rating extract et"""
        rating_elem = soup.find('div', class_='rating')
        if rating_elem:
            # "4.5 stars" -> 4.5
            rating_text = rating_elem.get('data-rating')
            return float(rating_text) if rating_text else None
        return None

    def _check_stock(self, soup):
        """Stok durumu kontrol et"""
        stock_elem = soup.find('span', class_='stock-status')
        if stock_elem:
            return 'in stock' in stock_elem.get_text().lower()
        return False

    def scrape_category(self, category_url: str, max_pages: int = 3) -> List[Dict]:
        """Kategori sayfasından tüm ürünleri scrape et"""
        all_products = []

        for page in range(1, max_pages + 1):
            page_url = f"{category_url}?page={page}"
            logger.info(f"Scraping page {page}: {page_url}")

            self._rate_limit_wait()

            try:
                response = self.session.get(page_url, timeout=10)
                response.raise_for_status()
                soup = BeautifulSoup(response.content, 'lxml')

                # Sayfadaki tüm ürün linklerini bul
                product_links = soup.find_all('a', class_='product-link')

                if not product_links:
                    logger.info(f"No products found on page {page}")
                    break

                # Her ürünü scrape et
                for link in product_links:
                    product_url = link.get('href')
                    if not product_url.startswith('http'):
                        product_url = self.base_url + product_url

                    product = self.scrape_product(product_url)
                    if product:
                        all_products.append(product)

            except Exception as e:
                logger.error(f"Error scraping page {page}: {e}")
                break

        return all_products

    def save_to_json(self, products: List[Dict], filename: str):
        """Ürünleri JSON dosyasına kaydet"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(products, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved {len(products)} products to {filename}")


# ============================================================================
# EGZERSIZ 2: News Aggregator (Medium)
# ============================================================================
"""
Senaryo: Farklı haber sitelerinden başlıkları ve özet bilgileri toplayan bir aggregator yazın.

Gereksinimler:
- Birden fazla haber kaynağından veri toplama
- Makale başlığı, özet, yazar, tarih extract etme
- Duplicate detection
- Verileri kategorize etme

TODO: NewsAggregator sınıfını implement edin
"""

class NewsAggregator:
    """Haber toplama ve aggregation"""

    def __init__(self, sources: List[str]):
        """
        Args:
            sources: Haber kaynağı URL'leri
        """
        # TODO: Session oluştur
        # TODO: Seen articles için set oluştur
        pass

    def scrape_source(self, source_url: str) -> List[Dict]:
        """
        Bir haber kaynağından makaleleri scrape et

        Returns:
            List[Dict]: Makale bilgileri (title, summary, author, date, url)
        """
        # TODO: Sayfayı fetch et
        # TODO: Article elementlerini bul
        # TODO: Her article'dan bilgi extract et
        # TODO: Duplicate check yap
        pass

    def scrape_all_sources(self) -> List[Dict]:
        """Tüm kaynaklardan haber topla"""
        # TODO: Her kaynağı scrape et
        # TODO: Sonuçları birleştir
        # TODO: Tarihe göre sırala
        pass

    def categorize_articles(self, articles: List[Dict]) -> Dict[str, List[Dict]]:
        """Makaleleri kategorilere ayır"""
        # TODO: Keyword'lere göre kategorize et
        # TODO: Categories: politics, technology, sports, business, etc.
        pass


# ÇÖZÜM:
class NewsAggregatorSolution:
    """Haber toplama ve aggregation - Çözüm"""

    def __init__(self, sources: List[str]):
        self.sources = sources
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'NewsAggregator/1.0'
        })
        self.seen_articles = set()  # Duplicate detection için

        # Kategori keywords
        self.category_keywords = {
            'technology': ['tech', 'ai', 'software', 'computer', 'digital'],
            'politics': ['election', 'government', 'president', 'policy'],
            'sports': ['game', 'match', 'player', 'team', 'score'],
            'business': ['market', 'stock', 'economy', 'company', 'trade'],
            'science': ['research', 'study', 'scientist', 'discovery']
        }

    def scrape_source(self, source_url: str) -> List[Dict]:
        """Bir haber kaynağından makaleleri scrape et"""
        articles = []

        try:
            response = self.session.get(source_url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'lxml')

            # Article elementlerini bul (yaygın class isimleri)
            article_elements = soup.find_all(['article', 'div'],
                                            class_=['article', 'post', 'news-item'])

            for element in article_elements:
                article = self._extract_article_data(element, source_url)

                # Duplicate check
                if article and article['title'] not in self.seen_articles:
                    self.seen_articles.add(article['title'])
                    articles.append(article)

            logger.info(f"Scraped {len(articles)} articles from {source_url}")

        except Exception as e:
            logger.error(f"Error scraping {source_url}: {e}")

        return articles

    def _extract_article_data(self, element, source_url: str) -> Optional[Dict]:
        """Article element'inden bilgi extract et"""
        try:
            # Title
            title_elem = element.find(['h1', 'h2', 'h3'], class_=['title', 'headline'])
            title = title_elem.get_text(strip=True) if title_elem else None

            if not title:
                return None

            # Summary
            summary_elem = element.find(['p', 'div'], class_=['summary', 'excerpt', 'description'])
            summary = summary_elem.get_text(strip=True) if summary_elem else ''

            # Author
            author_elem = element.find(['span', 'a'], class_=['author', 'byline'])
            author = author_elem.get_text(strip=True) if author_elem else 'Unknown'

            # Date
            date_elem = element.find('time')
            date = date_elem.get('datetime') if date_elem else None

            # URL
            link_elem = element.find('a')
            url = link_elem.get('href') if link_elem else None
            if url and not url.startswith('http'):
                from urllib.parse import urljoin
                url = urljoin(source_url, url)

            return {
                'title': title,
                'summary': summary[:200],  # İlk 200 karakter
                'author': author,
                'date': date,
                'url': url,
                'source': source_url,
                'scraped_at': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Error extracting article data: {e}")
            return None

    def scrape_all_sources(self) -> List[Dict]:
        """Tüm kaynaklardan haber topla"""
        all_articles = []

        for source in self.sources:
            articles = self.scrape_source(source)
            all_articles.extend(articles)
            time.sleep(1)  # Rate limiting

        # Tarihe göre sırala (en yeni önce)
        all_articles.sort(key=lambda x: x.get('date', ''), reverse=True)

        return all_articles

    def categorize_articles(self, articles: List[Dict]) -> Dict[str, List[Dict]]:
        """Makaleleri kategorilere ayır"""
        categorized = {category: [] for category in self.category_keywords.keys()}
        categorized['other'] = []

        for article in articles:
            # Title ve summary'yi birleştir ve lowercase yap
            text = (article['title'] + ' ' + article['summary']).lower()

            # Kategori belirle
            assigned_category = 'other'
            max_matches = 0

            for category, keywords in self.category_keywords.items():
                matches = sum(1 for keyword in keywords if keyword in text)
                if matches > max_matches:
                    max_matches = matches
                    assigned_category = category

            categorized[assigned_category].append(article)

        return categorized


# ============================================================================
# EGZERSIZ 3: Dynamic Content Scraper with Selenium (Hard)
# ============================================================================
"""
Senaryo: JavaScript ile render edilen bir SPA'dan veri scrape edin.

Gereksinimler:
- Selenium ile dinamik içerik scraping
- Infinite scroll handling
- AJAX request'lerin tamamlanmasını bekleme
- Screenshot alma
- Data extraction

TODO: DynamicScraper sınıfını implement edin
"""

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

class DynamicScraper:
    """JavaScript ile render edilen sayfalar için scraper"""

    def __init__(self, headless: bool = True):
        # TODO: Chrome options yapılandır
        # TODO: WebDriver oluştur
        # TODO: Wait oluştur
        pass

    def scrape_infinite_scroll(self, url: str, scroll_count: int = 5) -> List[Dict]:
        """
        Infinite scroll ile yüklenen içeriği scrape et

        Args:
            url: Scrape edilecek URL
            scroll_count: Kaç kez scroll yapılacağı

        Returns:
            List[Dict]: Scrape edilen veriler
        """
        # TODO: Sayfayı aç
        # TODO: Scroll işlemini gerçekleştir
        # TODO: Her scroll'da yeni içeriği bekle
        # TODO: Verileri extract et
        pass

    def wait_for_ajax(self, timeout: int = 10):
        """AJAX request'lerin tamamlanmasını bekle"""
        # TODO: jQuery.active == 0 kontrolü
        # TODO: Document ready state kontrolü
        pass

    def extract_items(self) -> List[Dict]:
        """Sayfadan item'ları extract et"""
        # TODO: Item elementlerini bul
        # TODO: Her item'dan veri extract et
        pass

    def take_screenshot(self, filename: str):
        """Sayfa screenshot'ı al"""
        # TODO: Screenshot al ve kaydet
        pass

    def close(self):
        """Driver'ı kapat"""
        # TODO: WebDriver'ı kapat
        pass


# ÇÖZÜM:
class DynamicScraperSolution:
    """JavaScript ile render edilen sayfalar için scraper - Çözüm"""

    def __init__(self, headless: bool = True):
        options = webdriver.ChromeOptions()

        if headless:
            options.add_argument('--headless')

        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--disable-blink-features=AutomationControlled')

        self.driver = webdriver.Chrome(options=options)
        self.wait = WebDriverWait(self.driver, 10)

    def scrape_infinite_scroll(self, url: str, scroll_count: int = 5) -> List[Dict]:
        """Infinite scroll ile yüklenen içeriği scrape et"""
        self.driver.get(url)

        # Initial load'u bekle
        self.wait.until(
            EC.presence_of_element_located((By.CLASS_NAME, "item"))
        )

        all_items = []
        previous_count = 0

        for i in range(scroll_count):
            # Scroll to bottom
            self.driver.execute_script(
                "window.scrollTo(0, document.body.scrollHeight);"
            )

            # Yeni içeriğin yüklenmesini bekle
            time.sleep(2)

            # AJAX tamamlanmasını bekle
            self.wait_for_ajax()

            # Şu anki item sayısı
            items = self.driver.find_elements(By.CLASS_NAME, "item")
            current_count = len(items)

            if current_count == previous_count:
                logger.info(f"No new items loaded after scroll {i + 1}")
                break

            logger.info(f"Scroll {i + 1}: Found {current_count} items")
            previous_count = current_count

        # Tüm item'ları extract et
        all_items = self.extract_items()

        return all_items

    def wait_for_ajax(self, timeout: int = 10):
        """AJAX request'lerin tamamlanmasını bekle"""
        try:
            # jQuery varsa
            self.driver.execute_script("return jQuery.active") == 0
        except:
            pass

        # Document ready
        WebDriverWait(self.driver, timeout).until(
            lambda driver: driver.execute_script("return document.readyState") == "complete"
        )

    def extract_items(self) -> List[Dict]:
        """Sayfadan item'ları extract et"""
        items = []
        elements = self.driver.find_elements(By.CLASS_NAME, "item")

        for element in elements:
            try:
                item = {
                    'title': element.find_element(By.CLASS_NAME, "title").text,
                    'description': element.find_element(By.CLASS_NAME, "description").text,
                    'link': element.find_element(By.TAG_NAME, "a").get_attribute('href'),
                }
                items.append(item)
            except Exception as e:
                logger.error(f"Error extracting item: {e}")
                continue

        return items

    def take_screenshot(self, filename: str):
        """Sayfa screenshot'ı al"""
        self.driver.save_screenshot(filename)
        logger.info(f"Screenshot saved to {filename}")

    def close(self):
        """Driver'ı kapat"""
        self.driver.quit()


# ============================================================================
# EGZERSIZ 4: API Data Collector (Medium)
# ============================================================================
"""
Senaryo: REST API'den veri toplayan ve pagination handle eden bir collector yazın.

Gereksinimler:
- Pagination (offset-based ve cursor-based)
- Rate limiting
- Authentication (API key)
- Error handling ve retry
- Data caching

TODO: APICollector sınıfını implement edin
"""

class APICollector:
    """API data collection with pagination"""

    def __init__(self, base_url: str, api_key: str):
        # TODO: Session oluştur
        # TODO: Authentication header ekle
        # TODO: Cache için dict oluştur
        pass

    def fetch_with_offset_pagination(self, endpoint: str, limit: int = 100) -> List[Dict]:
        """
        Offset-based pagination ile veri topla

        Args:
            endpoint: API endpoint
            limit: Her sayfada kaç item

        Returns:
            List[Dict]: Tüm veriler
        """
        # TODO: Offset-based pagination implement et
        # TODO: Her sayfayı fetch et
        # TODO: Tüm veriyi birleştir
        pass

    def fetch_with_cursor_pagination(self, endpoint: str) -> List[Dict]:
        """Cursor-based pagination ile veri topla"""
        # TODO: Cursor-based pagination implement et
        # TODO: next_cursor varken devam et
        pass

    def fetch_with_retry(self, url: str, max_retries: int = 3) -> Dict:
        """Retry logic ile API call"""
        # TODO: Retry logic implement et
        # TODO: Exponential backoff kullan
        pass


# ÇÖZÜM:
class APICollectorSolution:
    """API data collection with pagination - Çözüm"""

    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'Bearer {api_key}',
            'Accept': 'application/json'
        })
        self.cache = {}

    def fetch_with_offset_pagination(self, endpoint: str, limit: int = 100) -> List[Dict]:
        """Offset-based pagination ile veri topla"""
        all_data = []
        offset = 0

        while True:
            url = f"{self.base_url}/{endpoint}?limit={limit}&offset={offset}"

            logger.info(f"Fetching: offset={offset}, limit={limit}")

            try:
                response = self.fetch_with_retry(url)

                if not response or 'data' not in response:
                    break

                data = response['data']

                if not data:
                    break

                all_data.extend(data)
                offset += limit

                # Total count varsa optimize et
                if 'total' in response:
                    if offset >= response['total']:
                        break

                time.sleep(0.5)  # Rate limiting

            except Exception as e:
                logger.error(f"Error fetching data: {e}")
                break

        logger.info(f"Collected {len(all_data)} items")
        return all_data

    def fetch_with_cursor_pagination(self, endpoint: str) -> List[Dict]:
        """Cursor-based pagination ile veri topla"""
        all_data = []
        cursor = None

        while True:
            url = f"{self.base_url}/{endpoint}"
            if cursor:
                url += f"?cursor={cursor}"

            logger.info(f"Fetching with cursor: {cursor}")

            try:
                response = self.fetch_with_retry(url)

                if not response or 'data' not in response:
                    break

                data = response['data']
                all_data.extend(data)

                # Next cursor
                cursor = response.get('next_cursor')

                if not cursor:
                    break

                time.sleep(0.5)

            except Exception as e:
                logger.error(f"Error fetching data: {e}")
                break

        logger.info(f"Collected {len(all_data)} items")
        return all_data

    def fetch_with_retry(self, url: str, max_retries: int = 3) -> Dict:
        """Retry logic ile API call"""
        for attempt in range(max_retries):
            try:
                response = self.session.get(url, timeout=10)

                # Rate limit handling
                if response.status_code == 429:
                    retry_after = int(response.headers.get('Retry-After', 60))
                    logger.warning(f"Rate limited. Waiting {retry_after}s")
                    time.sleep(retry_after)
                    continue

                response.raise_for_status()
                return response.json()

            except requests.RequestException as e:
                if attempt == max_retries - 1:
                    raise

                wait_time = 2 ** attempt
                logger.warning(f"Retry {attempt + 1}/{max_retries} after {wait_time}s")
                time.sleep(wait_time)

        return None


# ============================================================================
# EGZERSIZ 5: Robust Error Handler (Medium)
# ============================================================================
"""
Senaryo: Comprehensive error handling ile scraper yazın.

Gereksinimler:
- Farklı HTTP error'ları handle etme
- Network error'ları handle etme
- Retry logic with exponential backoff
- Logging
- Error reporting

TODO: RobustScraper sınıfını implement edin
"""

from requests.exceptions import RequestException, HTTPError, ConnectionError, Timeout

class RobustScraper:
    """Comprehensive error handling ile scraper"""

    def __init__(self):
        # TODO: Session oluştur
        # TODO: Error log dosyası oluştur
        pass

    def fetch_with_error_handling(self, url: str) -> Optional[str]:
        """Comprehensive error handling ile fetch"""
        # TODO: Try-except blokları ekle
        # TODO: Farklı error türlerini handle et
        # TODO: Error'ları logla
        pass

    def retry_with_backoff(self, url: str, max_retries: int = 3) -> Optional[str]:
        """Exponential backoff ile retry"""
        # TODO: Retry loop oluştur
        # TODO: Exponential backoff uygula
        # TODO: Error sayısını logla
        pass

    def handle_http_errors(self, response):
        """HTTP error'ları handle et"""
        # TODO: Status code kontrolü
        # TODO: 404, 403, 429, 500 etc. handle et
        pass


# ÇÖZÜM:
class RobustScraperSolution:
    """Comprehensive error handling ile scraper - Çözüm"""

    def __init__(self):
        self.session = requests.Session()
        self.error_log = []

        # Logging setup
        self.logger = logging.getLogger(self.__class__.__name__)
        handler = logging.FileHandler('scraping_errors.log')
        handler.setFormatter(
            logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        )
        self.logger.addHandler(handler)

    def fetch_with_error_handling(self, url: str) -> Optional[str]:
        """Comprehensive error handling ile fetch"""
        try:
            response = self.session.get(url, timeout=10)
            self.handle_http_errors(response)
            return response.text

        except HTTPError as e:
            self.logger.error(f"HTTP Error for {url}: {e}")
            self.error_log.append({
                'url': url,
                'error': 'HTTPError',
                'status': e.response.status_code,
                'timestamp': datetime.now().isoformat()
            })
            return None

        except ConnectionError as e:
            self.logger.error(f"Connection Error for {url}: {e}")
            self.error_log.append({
                'url': url,
                'error': 'ConnectionError',
                'timestamp': datetime.now().isoformat()
            })
            return None

        except Timeout as e:
            self.logger.error(f"Timeout for {url}: {e}")
            self.error_log.append({
                'url': url,
                'error': 'Timeout',
                'timestamp': datetime.now().isoformat()
            })
            return None

        except RequestException as e:
            self.logger.error(f"Request Exception for {url}: {e}")
            self.error_log.append({
                'url': url,
                'error': 'RequestException',
                'timestamp': datetime.now().isoformat()
            })
            return None

        except Exception as e:
            self.logger.error(f"Unexpected error for {url}: {e}")
            self.error_log.append({
                'url': url,
                'error': 'UnexpectedException',
                'message': str(e),
                'timestamp': datetime.now().isoformat()
            })
            return None

    def retry_with_backoff(self, url: str, max_retries: int = 3) -> Optional[str]:
        """Exponential backoff ile retry"""
        for attempt in range(max_retries):
            try:
                response = self.session.get(url, timeout=10)
                self.handle_http_errors(response)
                self.logger.info(f"Successfully fetched {url} on attempt {attempt + 1}")
                return response.text

            except (ConnectionError, Timeout) as e:
                if attempt == max_retries - 1:
                    self.logger.error(f"Max retries reached for {url}")
                    return None

                wait_time = 2 ** attempt  # 1, 2, 4, 8...
                self.logger.warning(
                    f"Attempt {attempt + 1} failed for {url}. Retrying in {wait_time}s"
                )
                time.sleep(wait_time)

            except HTTPError as e:
                # HTTP errors genelde retry edilmez
                self.logger.error(f"HTTP error {e.response.status_code} for {url}")
                return None

        return None

    def handle_http_errors(self, response):
        """HTTP error'ları handle et"""
        if response.status_code == 404:
            raise HTTPError(f"Page not found", response=response)

        elif response.status_code == 403:
            raise HTTPError(f"Access forbidden", response=response)

        elif response.status_code == 429:
            # Rate limit - special handling
            retry_after = response.headers.get('Retry-After', 60)
            self.logger.warning(f"Rate limited. Should wait {retry_after}s")
            raise HTTPError(f"Rate limited", response=response)

        elif response.status_code >= 500:
            raise HTTPError(f"Server error: {response.status_code}", response=response)

        response.raise_for_status()

    def get_error_report(self) -> Dict:
        """Error raporu oluştur"""
        total_errors = len(self.error_log)

        if total_errors == 0:
            return {'total_errors': 0}

        # Error type'lara göre grupla
        error_types = {}
        for error in self.error_log:
            error_type = error['error']
            error_types[error_type] = error_types.get(error_type, 0) + 1

        return {
            'total_errors': total_errors,
            'error_types': error_types,
            'recent_errors': self.error_log[-5:]  # Son 5 error
        }


# ============================================================================
# EGZERSIZ 6: Session Manager (Medium)
# ============================================================================
"""
Senaryo: Login state'i koruyan ve persistent olan bir session manager yazın.

Gereksinimler:
- Login fonksiyonalitesi
- Cookie persistence (dosyaya kaydetme)
- Session expiration handling
- Auto-refresh

TODO: SessionManager sınıfını implement edin
"""

import pickle
import os
from datetime import datetime, timedelta

class SessionManager:
    """Persistent session management"""

    def __init__(self, session_file: str = 'session.pkl'):
        # TODO: Session file path
        # TODO: Session oluştur veya yükle
        pass

    def login(self, login_url: str, credentials: Dict) -> bool:
        """Login ve session kaydet"""
        # TODO: Login POST request
        # TODO: Success kontrolü
        # TODO: Session'ı kaydet
        pass

    def save_session(self, expires_in_hours: int = 24):
        """Session'ı dosyaya kaydet"""
        # TODO: Session data'yı serialize et
        # TODO: Expiry time ekle
        # TODO: Dosyaya yaz
        pass

    def load_session(self) -> bool:
        """Session'ı dosyadan yükle"""
        # TODO: Dosyayı oku
        # TODO: Expiry kontrolü
        # TODO: Session'ı restore et
        pass

    def is_logged_in(self) -> bool:
        """Login durumunu kontrol et"""
        # TODO: Session geçerliliğini kontrol et
        pass


# ÇÖZÜM:
class SessionManagerSolution:
    """Persistent session management - Çözüm"""

    def __init__(self, session_file: str = 'session.pkl'):
        self.session_file = session_file
        self.session = requests.Session()
        self.expires_at = None

        # Mevcut session'ı yükle
        if not self.load_session():
            self.logger = logging.getLogger(self.__class__.__name__)
            self.logger.info("New session created")

    def login(self, login_url: str, credentials: Dict) -> bool:
        """Login ve session kaydet"""
        try:
            response = self.session.post(login_url, data=credentials, timeout=10)

            # Login başarılı mı kontrol et
            if response.ok and self._verify_login(response):
                self.save_session()
                logging.info("Login successful")
                return True
            else:
                logging.error(f"Login failed: {response.status_code}")
                return False

        except Exception as e:
            logging.error(f"Login error: {e}")
            return False

    def _verify_login(self, response) -> bool:
        """Login başarısını verify et"""
        # Response'da success indicator ara
        # Örnek: JSON response'da 'success' field'ı
        try:
            data = response.json()
            return data.get('success', False)
        except:
            # HTML response için
            return 'dashboard' in response.url or 'logout' in response.text

    def save_session(self, expires_in_hours: int = 24):
        """Session'ı dosyaya kaydet"""
        session_data = {
            'cookies': dict(self.session.cookies),
            'headers': dict(self.session.headers),
            'expires_at': datetime.now() + timedelta(hours=expires_in_hours)
        }

        with open(self.session_file, 'wb') as f:
            pickle.dump(session_data, f)

        self.expires_at = session_data['expires_at']
        logging.info(f"Session saved to {self.session_file}")

    def load_session(self) -> bool:
        """Session'ı dosyadan yükle"""
        if not os.path.exists(self.session_file):
            return False

        try:
            with open(self.session_file, 'rb') as f:
                session_data = pickle.load(f)

            # Expiry kontrolü
            expires_at = session_data.get('expires_at')
            if expires_at and datetime.now() >= expires_at:
                logging.info("Session expired")
                os.remove(self.session_file)
                return False

            # Session'ı restore et
            self.session.cookies.update(session_data['cookies'])
            self.session.headers.update(session_data['headers'])
            self.expires_at = expires_at

            logging.info("Session loaded from file")
            return True

        except Exception as e:
            logging.error(f"Error loading session: {e}")
            return False

    def is_logged_in(self) -> bool:
        """Login durumunu kontrol et"""
        if not self.expires_at:
            return False

        if datetime.now() >= self.expires_at:
            logging.info("Session expired")
            return False

        return True

    def fetch_protected(self, url: str) -> Optional[str]:
        """Protected content fetch"""
        if not self.is_logged_in():
            logging.error("Not logged in")
            return None

        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            return response.text
        except Exception as e:
            logging.error(f"Error fetching protected content: {e}")
            return None

    def logout(self):
        """Logout ve session temizle"""
        self.session.close()
        if os.path.exists(self.session_file):
            os.remove(self.session_file)
        self.expires_at = None
        logging.info("Logged out and session cleared")


# ============================================================================
# EGZERSIZ 7: Rate Limiter (Medium)
# ============================================================================
"""
Senaryo: Adaptive rate limiting implement edin.

Gereksinimler:
- Token bucket veya sliding window algorithm
- Adaptive rate adjustment
- Multiple domains için farklı limits
- Request queuing

TODO: AdaptiveRateLimiter sınıfını implement edin
"""

from collections import deque
from threading import Lock

class AdaptiveRateLimiter:
    """Adaptive rate limiting"""

    def __init__(self, initial_rate: float = 1.0):
        # TODO: Rate parameters
        # TODO: Request history
        # TODO: Lock for thread safety
        pass

    def wait(self):
        """Rate limit için bekle"""
        # TODO: Son request'ten bu yana geçen süreyi hesapla
        # TODO: Gerekirse bekle
        pass

    def adjust_rate(self, success: bool):
        """Rate'i başarı/başarısızlığa göre ayarla"""
        # TODO: Başarılıysa rate'i artır
        # TODO: Başarısızsa rate'i azalt
        pass


# ÇÖZÜM:
class AdaptiveRateLimiterSolution:
    """Adaptive rate limiting - Çözüm"""

    def __init__(self, initial_rate: float = 1.0, min_rate: float = 0.5, max_rate: float = 5.0):
        self.rate = initial_rate  # Requests per second
        self.min_rate = min_rate
        self.max_rate = max_rate

        self.last_request_time = 0
        self.success_count = 0
        self.error_count = 0

        self.lock = Lock()

    def wait(self):
        """Rate limit için bekle"""
        with self.lock:
            current_time = time.time()
            time_since_last = current_time - self.last_request_time

            # Beklenmesi gereken süre
            required_delay = 1.0 / self.rate

            if time_since_last < required_delay:
                sleep_time = required_delay - time_since_last
                time.sleep(sleep_time)

            self.last_request_time = time.time()

    def on_success(self):
        """Başarılı request sonrası"""
        self.success_count += 1
        self.error_count = 0

        # 5 başarılı request'ten sonra rate'i artır
        if self.success_count >= 5:
            old_rate = self.rate
            self.rate = min(self.max_rate, self.rate * 1.2)
            self.success_count = 0

            if old_rate != self.rate:
                logging.info(f"Rate increased: {old_rate:.2f} -> {self.rate:.2f} req/s")

    def on_error(self, status_code: Optional[int] = None):
        """Hata sonrası"""
        self.error_count += 1
        self.success_count = 0

        # 429 (Rate Limit) özel durumu
        if status_code == 429:
            old_rate = self.rate
            self.rate = max(self.min_rate, self.rate * 0.5)
            logging.warning(f"Rate decreased (429): {old_rate:.2f} -> {self.rate:.2f} req/s")

        # 3 hata sonrası rate'i azalt
        elif self.error_count >= 3:
            old_rate = self.rate
            self.rate = max(self.min_rate, self.rate * 0.8)
            self.error_count = 0
            logging.warning(f"Rate decreased (errors): {old_rate:.2f} -> {self.rate:.2f} req/s")

    def get_current_delay(self) -> float:
        """Şu anki delay'i döndür"""
        return 1.0 / self.rate


class MultiDomainRateLimiter:
    """Domain başına farklı rate limiting"""

    def __init__(self):
        self.limiters = {}  # domain -> AdaptiveRateLimiter
        self.lock = Lock()

    def get_limiter(self, domain: str) -> AdaptiveRateLimiterSolution:
        """Domain için limiter al veya oluştur"""
        with self.lock:
            if domain not in self.limiters:
                self.limiters[domain] = AdaptiveRateLimiterSolution()
            return self.limiters[domain]

    def wait_for_domain(self, url: str):
        """URL'in domain'i için bekle"""
        from urllib.parse import urlparse
        domain = urlparse(url).netloc

        limiter = self.get_limiter(domain)
        limiter.wait()

    def report_success(self, url: str):
        """Başarılı request rapor et"""
        from urllib.parse import urlparse
        domain = urlparse(url).netloc

        limiter = self.get_limiter(domain)
        limiter.on_success()

    def report_error(self, url: str, status_code: Optional[int] = None):
        """Hata rapor et"""
        from urllib.parse import urlparse
        domain = urlparse(url).netloc

        limiter = self.get_limiter(domain)
        limiter.on_error(status_code)


# ============================================================================
# EGZERSIZ 8: Data Validator (Medium)
# ============================================================================
"""
Senaryo: Scrape edilen verileri validate eden bir sistem yazın.

Gereksinimler:
- Schema validation
- Data type checking
- Required fields kontrolü
- Custom validation rules
- Validation reporting

TODO: DataValidator sınıfını implement edin
"""

from typing import Any, Callable

class DataValidator:
    """Scrape edilen veri validasyonu"""

    def __init__(self, schema: Dict):
        """
        Args:
            schema: Validation schema
                {
                    'field_name': {
                        'type': str/int/float/bool,
                        'required': bool,
                        'min_length': int,
                        'max_length': int,
                        'pattern': str (regex),
                        'custom': Callable
                    }
                }
        """
        # TODO: Schema'yı kaydet
        pass

    def validate(self, data: Dict) -> tuple[bool, List[str]]:
        """
        Veriyi validate et

        Returns:
            (is_valid, errors): Validation sonucu ve error listesi
        """
        # TODO: Her field için validation yap
        # TODO: Type checking
        # TODO: Required fields
        # TODO: Custom validations
        # TODO: Error mesajları topla
        pass

    def validate_field(self, field_name: str, value: Any, rules: Dict) -> List[str]:
        """Tek bir field'ı validate et"""
        # TODO: Type check
        # TODO: Length check
        # TODO: Pattern match
        # TODO: Custom validation
        pass


# ÇÖZÜM:
import re

class DataValidatorSolution:
    """Scrape edilen veri validasyonu - Çözüm"""

    def __init__(self, schema: Dict):
        self.schema = schema

    def validate(self, data: Dict) -> tuple[bool, List[str]]:
        """Veriyi validate et"""
        errors = []

        # Required fields kontrolü
        for field_name, rules in self.schema.items():
            if rules.get('required', False) and field_name not in data:
                errors.append(f"Required field missing: {field_name}")
                continue

            # Field varsa validate et
            if field_name in data:
                field_errors = self.validate_field(
                    field_name,
                    data[field_name],
                    rules
                )
                errors.extend(field_errors)

        return (len(errors) == 0, errors)

    def validate_field(self, field_name: str, value: Any, rules: Dict) -> List[str]:
        """Tek bir field'ı validate et"""
        errors = []

        # Type check
        expected_type = rules.get('type')
        if expected_type and not isinstance(value, expected_type):
            errors.append(
                f"{field_name}: Expected {expected_type.__name__}, got {type(value).__name__}"
            )
            return errors  # Type yanlışsa diğer kontrolleri yapma

        # String validations
        if isinstance(value, str):
            # Min length
            min_length = rules.get('min_length')
            if min_length and len(value) < min_length:
                errors.append(
                    f"{field_name}: Length {len(value)} < minimum {min_length}"
                )

            # Max length
            max_length = rules.get('max_length')
            if max_length and len(value) > max_length:
                errors.append(
                    f"{field_name}: Length {len(value)} > maximum {max_length}"
                )

            # Pattern match
            pattern = rules.get('pattern')
            if pattern and not re.match(pattern, value):
                errors.append(
                    f"{field_name}: Does not match pattern {pattern}"
                )

        # Numeric validations
        if isinstance(value, (int, float)):
            # Min value
            min_value = rules.get('min_value')
            if min_value is not None and value < min_value:
                errors.append(
                    f"{field_name}: Value {value} < minimum {min_value}"
                )

            # Max value
            max_value = rules.get('max_value')
            if max_value is not None and value > max_value:
                errors.append(
                    f"{field_name}: Value {value} > maximum {max_value}"
                )

        # Custom validation
        custom_validator = rules.get('custom')
        if custom_validator and callable(custom_validator):
            try:
                if not custom_validator(value):
                    errors.append(
                        f"{field_name}: Failed custom validation"
                    )
            except Exception as e:
                errors.append(
                    f"{field_name}: Custom validation error: {e}"
                )

        return errors

    def validate_batch(self, data_list: List[Dict]) -> Dict:
        """Birden fazla veriyi validate et"""
        results = {
            'total': len(data_list),
            'valid': 0,
            'invalid': 0,
            'errors': []
        }

        for i, data in enumerate(data_list):
            is_valid, errors = self.validate(data)

            if is_valid:
                results['valid'] += 1
            else:
                results['invalid'] += 1
                results['errors'].append({
                    'index': i,
                    'data': data,
                    'errors': errors
                })

        return results


# Test DataValidator
if __name__ == "__main__":
    # Schema tanımla
    product_schema = {
        'name': {
            'type': str,
            'required': True,
            'min_length': 3,
            'max_length': 100
        },
        'price': {
            'type': float,
            'required': True,
            'min_value': 0.01,
            'max_value': 999999.99
        },
        'url': {
            'type': str,
            'required': True,
            'pattern': r'^https?://.+'
        },
        'rating': {
            'type': float,
            'required': False,
            'min_value': 0.0,
            'max_value': 5.0
        }
    }

    validator = DataValidatorSolution(product_schema)

    # Test data
    test_products = [
        {
            'name': 'Product 1',
            'price': 19.99,
            'url': 'https://example.com/product1',
            'rating': 4.5
        },
        {
            'name': 'AB',  # Too short
            'price': -5.0,  # Invalid price
            'url': 'invalid-url',  # Invalid URL
            'rating': 6.0  # Out of range
        },
        {
            # Missing required fields
            'name': 'Product 3'
        }
    ]

    # Batch validation
    results = validator.validate_batch(test_products)
    print(f"Valid: {results['valid']}/{results['total']}")
    print(f"Invalid: {results['invalid']}/{results['total']}")

    for error_item in results['errors']:
        print(f"\nItem {error_item['index']}:")
        for error in error_item['errors']:
            print(f"  - {error}")


# ============================================================================
# EGZERSIZ 9: Robots.txt Compliance Checker (Easy-Medium)
# ============================================================================
"""
Senaryo: Robots.txt'i parse eden ve compliance kontrol yapan bir sınıf yazın.

Gereksinimler:
- Robots.txt parsing
- URL permission checking
- Crawl delay extraction
- Sitemap URL extraction

TODO: RobotsChecker sınıfını implement edin
"""

from urllib.robotparser import RobotFileParser
from urllib.parse import urlparse, urljoin

class RobotsChecker:
    """Robots.txt compliance checker"""

    def __init__(self, user_agent: str = '*'):
        # TODO: User agent kaydet
        # TODO: Parser dictionary oluştur (domain -> parser)
        pass

    def can_fetch(self, url: str) -> bool:
        """URL'in fetch edilip edilemeyeceğini kontrol et"""
        # TODO: Domain için robots.txt parse et
        # TODO: Permission kontrolü yap
        pass

    def get_crawl_delay(self, url: str) -> float:
        """Crawl delay'i al"""
        # TODO: Parser'dan crawl delay al
        # TODO: Default değer döndür
        pass

    def get_sitemaps(self, domain: str) -> List[str]:
        """Sitemap URL'lerini al"""
        # TODO: Robots.txt'den sitemap'leri extract et
        pass


# ÇÖZÜM:
class RobotsCheckerSolution:
    """Robots.txt compliance checker - Çözüm"""

    def __init__(self, user_agent: str = '*'):
        self.user_agent = user_agent
        self.parsers = {}  # domain -> RobotFileParser

    def _get_parser(self, domain: str) -> RobotFileParser:
        """Domain için parser al veya oluştur"""
        if domain not in self.parsers:
            robots_url = urljoin(domain, '/robots.txt')
            parser = RobotFileParser()
            parser.set_url(robots_url)

            try:
                parser.read()
                logging.info(f"Loaded robots.txt from {robots_url}")
            except Exception as e:
                logging.warning(f"Could not load robots.txt from {robots_url}: {e}")

            self.parsers[domain] = parser

        return self.parsers[domain]

    def can_fetch(self, url: str) -> bool:
        """URL'in fetch edilip edilemeyeceğini kontrol et"""
        parsed = urlparse(url)
        domain = f"{parsed.scheme}://{parsed.netloc}"

        parser = self._get_parser(domain)
        can_fetch = parser.can_fetch(self.user_agent, url)

        if not can_fetch:
            logging.warning(f"Robots.txt forbids: {url}")

        return can_fetch

    def get_crawl_delay(self, url: str) -> float:
        """Crawl delay'i al"""
        parsed = urlparse(url)
        domain = f"{parsed.scheme}://{parsed.netloc}"

        parser = self._get_parser(domain)
        delay = parser.crawl_delay(self.user_agent)

        # Default 1 saniye
        return delay if delay else 1.0

    def get_sitemaps(self, domain: str) -> List[str]:
        """Sitemap URL'lerini al"""
        parser = self._get_parser(domain)

        try:
            # RobotFileParser'da sitemap method yok,
            # manuel parse gerekiyor
            sitemaps = []
            robots_url = urljoin(domain, '/robots.txt')

            response = requests.get(robots_url, timeout=5)
            if response.ok:
                for line in response.text.split('\n'):
                    if line.lower().startswith('sitemap:'):
                        sitemap_url = line.split(':', 1)[1].strip()
                        sitemaps.append(sitemap_url)

            return sitemaps

        except Exception as e:
            logging.error(f"Error getting sitemaps: {e}")
            return []

    def check_compliance_report(self, urls: List[str]) -> Dict:
        """URL listesi için compliance raporu"""
        report = {
            'total': len(urls),
            'allowed': 0,
            'forbidden': 0,
            'forbidden_urls': []
        }

        for url in urls:
            if self.can_fetch(url):
                report['allowed'] += 1
            else:
                report['forbidden'] += 1
                report['forbidden_urls'].append(url)

        return report


# ============================================================================
# EGZERSIZ 10: Distributed Scraper with Queue (Hard)
# ============================================================================
"""
Senaryo: Redis queue kullanarak distributed scraping yapın.

Gereksinimler:
- Redis-based task queue
- Worker pattern
- Task deduplication
- Progress tracking
- Result storage

TODO: DistributedScraper ve Worker sınıflarını implement edin
"""

# Not: Redis kurulumu gerektirir
# pip install redis

class TaskQueue:
    """Redis-based task queue"""

    def __init__(self, redis_host: str = 'localhost'):
        # TODO: Redis connection
        # TODO: Queue keys
        pass

    def add_task(self, url: str):
        """Queue'ya task ekle"""
        # TODO: Duplicate check
        # TODO: Queue'ya push
        pass

    def get_task(self) -> Optional[str]:
        """Queue'dan task al"""
        # TODO: Queue'dan pop
        # TODO: Processing set'ine ekle
        pass

    def mark_completed(self, url: str):
        """Task'ı tamamlandı olarak işaretle"""
        # TODO: Processing'den kaldır
        # TODO: Completed'a ekle
        pass

    def get_stats(self) -> Dict:
        """Queue istatistikleri"""
        # TODO: Queue size, processing, completed
        pass


class ScraperWorker:
    """Distributed scraping worker"""

    def __init__(self, worker_id: str, queue: TaskQueue):
        # TODO: Worker ID
        # TODO: Queue reference
        # TODO: Scraper oluştur
        pass

    def run(self):
        """Worker loop"""
        # TODO: Queue'dan task al
        # TODO: Scrape et
        # TODO: Result kaydet
        # TODO: Task'ı complete işaretle
        pass


# ÇÖZÜM:
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logging.warning("Redis not available. Install with: pip install redis")

if REDIS_AVAILABLE:
    import hashlib

    class TaskQueueSolution:
        """Redis-based task queue - Çözüm"""

        def __init__(self, redis_host: str = 'localhost', redis_port: int = 6379):
            self.redis = redis.Redis(
                host=redis_host,
                port=redis_port,
                decode_responses=True
            )

            # Queue keys
            self.queue_key = 'scraping:queue'
            self.processing_key = 'scraping:processing'
            self.completed_key = 'scraping:completed'

        def add_task(self, url: str):
            """Queue'ya task ekle"""
            url_hash = hashlib.md5(url.encode()).hexdigest()

            # Duplicate check
            if self.redis.sismember(self.completed_key, url_hash):
                logging.info(f"URL already completed: {url}")
                return False

            # Queue'ya ekle
            task_data = json.dumps({
                'url': url,
                'hash': url_hash,
                'added_at': datetime.now().isoformat()
            })

            self.redis.lpush(self.queue_key, task_data)
            logging.info(f"Added to queue: {url}")
            return True

        def add_tasks_batch(self, urls: List[str]):
            """Toplu task ekleme"""
            added = 0
            for url in urls:
                if self.add_task(url):
                    added += 1
            logging.info(f"Added {added}/{len(urls)} tasks to queue")

        def get_task(self) -> Optional[Dict]:
            """Queue'dan task al"""
            # Queue'dan al ve processing'e taşı (atomic)
            task_data = self.redis.rpoplpush(self.queue_key, self.processing_key)

            if task_data:
                return json.loads(task_data)
            return None

        def mark_completed(self, task: Dict):
            """Task'ı tamamlandı olarak işaretle"""
            url_hash = task['hash']

            # Processing'den kaldır
            task_data = json.dumps(task)
            self.redis.lrem(self.processing_key, 0, task_data)

            # Completed'a ekle
            self.redis.sadd(self.completed_key, url_hash)

            logging.info(f"Task completed: {task['url']}")

        def mark_failed(self, task: Dict):
            """Failed task'ı processing'den kaldır"""
            task_data = json.dumps(task)
            self.redis.lrem(self.processing_key, 0, task_data)

            # Failed set'ine ekle
            self.redis.sadd('scraping:failed', task['hash'])

        def get_stats(self) -> Dict:
            """Queue istatistikleri"""
            return {
                'queue_size': self.redis.llen(self.queue_key),
                'processing': self.redis.llen(self.processing_key),
                'completed': self.redis.scard(self.completed_key),
                'failed': self.redis.scard('scraping:failed')
            }

        def clear_all(self):
            """Tüm queue'ları temizle"""
            self.redis.delete(
                self.queue_key,
                self.processing_key,
                self.completed_key,
                'scraping:failed'
            )


    class ScraperWorkerSolution:
        """Distributed scraping worker - Çözüm"""

        def __init__(self, worker_id: str, queue: TaskQueueSolution):
            self.worker_id = worker_id
            self.queue = queue
            self.scraper = RobustScraperSolution()
            self.logger = logging.getLogger(f'Worker-{worker_id}')

            self.stats = {
                'processed': 0,
                'successful': 0,
                'failed': 0
            }

        def run(self, max_tasks: Optional[int] = None):
            """Worker loop"""
            self.logger.info(f"Worker {self.worker_id} started")

            tasks_processed = 0

            while True:
                # Max tasks limiti
                if max_tasks and tasks_processed >= max_tasks:
                    self.logger.info(f"Reached max tasks: {max_tasks}")
                    break

                # Queue'dan task al
                task = self.queue.get_task()

                if not task:
                    self.logger.info("No tasks in queue")
                    time.sleep(5)
                    continue

                url = task['url']
                self.logger.info(f"Processing: {url}")

                try:
                    # Scrape
                    content = self.scraper.fetch_with_error_handling(url)

                    if content:
                        # Success
                        self._save_result(url, content)
                        self.queue.mark_completed(task)

                        self.stats['successful'] += 1
                        self.logger.info(f"Success: {url}")
                    else:
                        # Failed
                        self.queue.mark_failed(task)
                        self.stats['failed'] += 1
                        self.logger.error(f"Failed: {url}")

                except Exception as e:
                    self.logger.error(f"Error processing {url}: {e}")
                    self.queue.mark_failed(task)
                    self.stats['failed'] += 1

                self.stats['processed'] += 1
                tasks_processed += 1

                # Rate limiting
                time.sleep(1)

            self.logger.info(f"Worker {self.worker_id} stopped")
            self.logger.info(f"Stats: {self.stats}")

        def _save_result(self, url: str, content: str):
            """Sonucu kaydet"""
            # Filename oluştur
            url_hash = hashlib.md5(url.encode()).hexdigest()
            filename = f"results/{url_hash}.html"

            # Directory oluştur
            os.makedirs('results', exist_ok=True)

            # Kaydet
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(content)

        def get_stats(self) -> Dict:
            """Worker istatistikleri"""
            return {
                'worker_id': self.worker_id,
                **self.stats
            }


# ============================================================================
# Test ve Kullanım Örnekleri
# ============================================================================

def test_exercises():
    """Egzersizleri test et"""

    print("=" * 80)
    print("Web Scraping Egzersizleri - Test")
    print("=" * 80)

    # Test 1: Product Scraper
    print("\n1. Product Scraper Test")
    print("-" * 40)
    # Not: Gerçek bir e-commerce sitesi kullanmak yerine httpbin kullanabiliriz
    # scraper = ProductScraperSolution('https://httpbin.org')
    print("Skipped: Requires real e-commerce site")

    # Test 2: News Aggregator
    print("\n2. News Aggregator Test")
    print("-" * 40)
    # aggregator = NewsAggregatorSolution(['https://example.com'])
    print("Skipped: Requires real news sites")

    # Test 3: Dynamic Scraper
    print("\n3. Dynamic Scraper Test")
    print("-" * 40)
    print("Skipped: Requires Selenium setup")

    # Test 4: API Collector
    print("\n4. API Collector Test")
    print("-" * 40)
    # collector = APICollectorSolution('https://api.example.com', 'fake-key')
    print("Skipped: Requires real API")

    # Test 5: Robust Scraper
    print("\n5. Robust Scraper Test")
    print("-" * 40)
    scraper = RobustScraperSolution()

    test_urls = [
        'https://httpbin.org/status/200',
        'https://httpbin.org/status/404',
        'https://httpbin.org/status/500',
    ]

    for url in test_urls:
        result = scraper.fetch_with_error_handling(url)
        status = "Success" if result else "Failed"
        print(f"{url}: {status}")

    print(f"\nError report: {scraper.get_error_report()}")

    # Test 7: Rate Limiter
    print("\n7. Rate Limiter Test")
    print("-" * 40)
    limiter = AdaptiveRateLimiterSolution(initial_rate=2.0)

    for i in range(5):
        limiter.wait()
        print(f"Request {i + 1} - Current delay: {limiter.get_current_delay():.2f}s")
        limiter.on_success()

    # Test 8: Data Validator
    print("\n8. Data Validator Test")
    print("-" * 40)
    # Already tested in the DataValidator section
    print("See DataValidator section for detailed test")

    # Test 9: Robots Checker
    print("\n9. Robots Checker Test")
    print("-" * 40)
    robots = RobotsCheckerSolution('TestBot/1.0')

    test_url = 'https://www.python.org/about/'
    can_fetch = robots.can_fetch(test_url)
    crawl_delay = robots.get_crawl_delay(test_url)

    print(f"URL: {test_url}")
    print(f"Can fetch: {can_fetch}")
    print(f"Crawl delay: {crawl_delay}s")

    print("\n" + "=" * 80)
    print("Test tamamlandı!")
    print("=" * 80)


if __name__ == "__main__":
    test_exercises()


"""
BONUS EGZERSIZLER (Kendiniz deneyin!)

1. Proxy Pool Manager
   - Proxy listesi yönetimi
   - Proxy health checking
   - Automatic rotation
   - Failure handling

2. Screenshot Comparison
   - Selenium ile screenshot alma
   - İki screenshot'ı karşılaştırma
   - Visual regression testing

3. CAPTCHA Handler
   - CAPTCHA detection
   - 2captcha/anticaptcha entegrasyonu
   - Retry logic

4. Data Pipeline
   - Scraper -> Validator -> Storage pipeline
   - ETL operations
   - Data transformation

5. Monitoring Dashboard
   - Scraping metrics
   - Real-time monitoring
   - Alerting system

6. Incremental Scraping
   - Change detection
   - Delta updates
   - Timestamp tracking

7. Multi-format Exporter
   - JSON, CSV, XML export
   - Database storage
   - Cloud storage (S3, etc.)

8. Authentication Handler
   - OAuth implementation
   - JWT handling
   - Session refresh

9. Sitemap Crawler
   - Sitemap.xml parsing
   - Automatic URL discovery
   - Priority-based crawling

10. Advanced Anti-Detection
    - Browser fingerprinting
    - Canvas fingerprinting avoidance
    - WebRTC leak prevention
"""
