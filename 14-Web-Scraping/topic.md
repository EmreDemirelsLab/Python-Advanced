# Web Scraping - İleri Düzey

## İçindekiler
1. [Giriş](#giriş)
2. [Requests Modülü - İleri Düzey](#requests-modülü---ileri-düzey)
3. [BeautifulSoup4](#beautifulsoup4)
4. [CSS Selectors ve XPath](#css-selectors-ve-xpath)
5. [Selenium Automation](#selenium-automation)
6. [Headless Browsers](#headless-browsers)
7. [Rate Limiting ve Throttling](#rate-limiting-ve-throttling)
8. [User Agents ve Headers](#user-agents-ve-headers)
9. [Session Management](#session-management)
10. [Error Handling ve Retry Logic](#error-handling-ve-retry-logic)
11. [Robots.txt ve Etik](#robotstxt-ve-etik)
12. [Production Patterns](#production-patterns)

---

## Giriş

Web scraping, web sitelerinden veri çıkarma işlemidir. Modern web scraping, karmaşık JavaScript uygulamalarını, anti-scraping mekanizmalarını ve dinamik içeriği yönetmeyi gerektirir.

**Önemli Kavramlar:**
- **Etik Scraping**: Robots.txt'e uyma, rate limiting, telif hakları
- **Legal Uyumluluk**: GDPR, ToS (Terms of Service)
- **Performans**: Asenkron requests, connection pooling
- **Dayanıklılık**: Retry logic, error handling, logging

---

## Requests Modülü - İleri Düzey

### Örnek 1: Advanced Session Management

```python
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import logging

# Logging yapılandırması
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvancedScraper:
    """
    İleri düzey web scraper sınıfı.
    - Connection pooling
    - Automatic retries
    - Custom timeout
    - Session persistence
    """

    def __init__(self, max_retries=3, pool_connections=10, pool_maxsize=10):
        self.session = requests.Session()

        # Retry stratejisi
        retry_strategy = Retry(
            total=max_retries,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS", "POST"],
            backoff_factor=1  # 1, 2, 4, 8 saniye bekler
        )

        # HTTP Adapter ile connection pooling
        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=pool_connections,
            pool_maxsize=pool_maxsize
        )

        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

        # Default headers
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        })

    def get(self, url, timeout=10, **kwargs):
        """GET request with error handling"""
        try:
            response = self.session.get(url, timeout=timeout, **kwargs)
            response.raise_for_status()
            logger.info(f"Successfully fetched: {url}")
            return response
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching {url}: {str(e)}")
            raise

    def close(self):
        """Session'ı kapat"""
        self.session.close()

# Kullanım
scraper = AdvancedScraper(max_retries=5)
try:
    response = scraper.get('https://httpbin.org/delay/2')
    print(f"Status: {response.status_code}")
    print(f"Response time: {response.elapsed.total_seconds()}s")
finally:
    scraper.close()
```

### Örnek 2: Proxy Rotation

```python
import requests
from itertools import cycle
import random

class ProxyRotator:
    """
    Proxy rotation ile scraping.
    IP bloklama riskini azaltır.
    """

    def __init__(self, proxy_list):
        """
        proxy_list: ['http://proxy1:port', 'http://proxy2:port', ...]
        """
        self.proxy_pool = cycle(proxy_list)
        self.current_proxy = None

    def get_proxy(self):
        """Sıradaki proxy'yi al"""
        self.current_proxy = next(self.proxy_pool)
        return {'http': self.current_proxy, 'https': self.current_proxy}

    def fetch(self, url, max_retries=3):
        """Proxy rotation ile URL fetch"""
        for attempt in range(max_retries):
            proxy = self.get_proxy()
            try:
                response = requests.get(
                    url,
                    proxies=proxy,
                    timeout=10,
                    headers={'User-Agent': 'Mozilla/5.0'}
                )
                response.raise_for_status()
                print(f"Success with proxy: {self.current_proxy}")
                return response
            except requests.exceptions.RequestException as e:
                print(f"Attempt {attempt + 1} failed with {self.current_proxy}: {e}")
                if attempt == max_retries - 1:
                    raise
        return None

# Kullanım (örnek proxy'ler)
proxies = [
    'http://proxy1.example.com:8080',
    'http://proxy2.example.com:8080',
    'http://proxy3.example.com:8080'
]
# rotator = ProxyRotator(proxies)
# response = rotator.fetch('https://httpbin.org/ip')
```

### Örnek 3: Cookie Management

```python
import requests
from http.cookiejar import LWPCookieJar
import os

class CookieManager:
    """
    Cookie persistence ve yönetimi.
    Login state'i korumak için kullanılır.
    """

    def __init__(self, cookie_file='cookies.txt'):
        self.cookie_file = cookie_file
        self.session = requests.Session()

        # Cookie jar oluştur
        self.session.cookies = LWPCookieJar(cookie_file)

        # Mevcut cookie'leri yükle
        if os.path.exists(cookie_file):
            self.session.cookies.load(ignore_discard=True, ignore_expires=True)

    def login(self, login_url, credentials):
        """Login ve cookie kaydet"""
        response = self.session.post(login_url, data=credentials)

        if response.ok:
            # Cookie'leri kaydet
            self.session.cookies.save(ignore_discard=True, ignore_expires=True)
            print("Login successful, cookies saved")
            return True
        return False

    def fetch_protected(self, url):
        """Cookie ile protected content fetch"""
        response = self.session.get(url)
        return response

    def clear_cookies(self):
        """Tüm cookie'leri temizle"""
        self.session.cookies.clear()
        if os.path.exists(self.cookie_file):
            os.remove(self.cookie_file)

# Kullanım
# manager = CookieManager()
# manager.login('https://example.com/login', {'user': 'test', 'pass': 'test'})
# content = manager.fetch_protected('https://example.com/dashboard')
```

---

## BeautifulSoup4

### Örnek 4: Advanced Parsing

```python
from bs4 import BeautifulSoup
import requests
from typing import List, Dict

class HTMLParser:
    """
    BeautifulSoup ile gelişmiş HTML parsing.
    """

    def __init__(self, html_content, parser='lxml'):
        """
        parser: 'lxml' (hızlı), 'html.parser' (built-in), 'html5lib' (tolerant)
        """
        self.soup = BeautifulSoup(html_content, parser)

    def extract_articles(self) -> List[Dict]:
        """Article'ları extract et"""
        articles = []

        for article in self.soup.find_all('article', class_='post'):
            data = {
                'title': self._extract_title(article),
                'author': self._extract_author(article),
                'date': self._extract_date(article),
                'content': self._extract_content(article),
                'tags': self._extract_tags(article),
                'image': self._extract_image(article)
            }
            articles.append(data)

        return articles

    def _extract_title(self, element):
        """Title'ı güvenli şekilde extract et"""
        title_elem = element.find('h2', class_='title')
        return title_elem.get_text(strip=True) if title_elem else None

    def _extract_author(self, element):
        """Author bilgisini extract et"""
        author_elem = element.find('span', class_='author')
        return author_elem.get_text(strip=True) if author_elem else 'Anonymous'

    def _extract_date(self, element):
        """Tarih bilgisini extract et"""
        date_elem = element.find('time')
        if date_elem:
            return date_elem.get('datetime') or date_elem.get_text(strip=True)
        return None

    def _extract_content(self, element):
        """İçeriği extract et"""
        content_elem = element.find('div', class_='content')
        if content_elem:
            # Script ve style tag'lerini kaldır
            for script in content_elem(['script', 'style']):
                script.decompose()
            return content_elem.get_text(strip=True, separator=' ')
        return None

    def _extract_tags(self, element):
        """Tag'leri extract et"""
        tag_container = element.find('div', class_='tags')
        if tag_container:
            return [tag.get_text(strip=True) for tag in tag_container.find_all('a')]
        return []

    def _extract_image(self, element):
        """Image URL'ini extract et"""
        img = element.find('img')
        if img:
            return img.get('src') or img.get('data-src')  # Lazy loading için
        return None

# Kullanım örneği
html_example = """
<article class="post">
    <h2 class="title">Sample Article</h2>
    <span class="author">John Doe</span>
    <time datetime="2024-01-15">January 15, 2024</time>
    <div class="content">
        <p>This is the article content.</p>
    </div>
    <div class="tags">
        <a>Python</a>
        <a>Web Scraping</a>
    </div>
    <img src="https://example.com/image.jpg" alt="Article image">
</article>
"""

parser = HTMLParser(html_example)
articles = parser.extract_articles()
print(articles)
```

### Örnek 5: Navigating Complex HTML

```python
from bs4 import BeautifulSoup, NavigableString

def advanced_navigation(html):
    """
    BeautifulSoup ile complex navigation.
    """
    soup = BeautifulSoup(html, 'lxml')

    # 1. Parent navigation
    link = soup.find('a', class_='read-more')
    if link:
        article = link.find_parent('article')
        print(f"Parent article: {article.get('id')}")

    # 2. Sibling navigation
    heading = soup.find('h2')
    if heading:
        next_paragraph = heading.find_next_sibling('p')
        print(f"Next paragraph: {next_paragraph.text[:50]}")

    # 3. Recursive search
    all_text_nodes = []
    for element in soup.descendants:
        if isinstance(element, NavigableString) and element.strip():
            all_text_nodes.append(element.strip())

    # 4. Find by attribute
    elements_with_data_id = soup.find_all(attrs={'data-id': True})

    # 5. Complex selector combinations
    # Birden fazla class'a sahip elementler
    multi_class = soup.find_all(class_=['featured', 'sticky'])

    # Belirli attribute değerine sahip elementler
    specific_attr = soup.find_all('div', attrs={'data-type': 'product'})

    return {
        'text_nodes': all_text_nodes[:5],
        'data_elements': len(elements_with_data_id),
        'multi_class': len(multi_class)
    }

# Test
html = """
<article id="post-1">
    <h2>Title</h2>
    <p>First paragraph</p>
    <a class="read-more">Read more</a>
</article>
<div class="featured sticky" data-id="123">Featured content</div>
"""
result = advanced_navigation(html)
print(result)
```

---

## CSS Selectors ve XPath

### Örnek 6: CSS Selectors

```python
from bs4 import BeautifulSoup

class CSSSelector:
    """
    CSS selector'ları kullanarak element seçimi.
    """

    def __init__(self, html):
        self.soup = BeautifulSoup(html, 'lxml')

    def demonstrate_selectors(self):
        """Farklı CSS selector türlerini göster"""

        # 1. Basic selectors
        all_divs = self.soup.select('div')
        print(f"All divs: {len(all_divs)}")

        # 2. Class selector
        featured = self.soup.select('.featured')
        print(f"Featured items: {len(featured)}")

        # 3. ID selector
        header = self.soup.select('#header')

        # 4. Attribute selector
        links_with_target = self.soup.select('a[target="_blank"]')
        external_links = self.soup.select('a[href^="http"]')
        pdf_links = self.soup.select('a[href$=".pdf"]')

        # 5. Descendant selector
        article_paragraphs = self.soup.select('article p')

        # 6. Direct child selector
        direct_children = self.soup.select('div > p')

        # 7. Adjacent sibling selector
        adjacent = self.soup.select('h2 + p')

        # 8. General sibling selector
        siblings = self.soup.select('h2 ~ p')

        # 9. Pseudo-class selectors
        first_child = self.soup.select('p:first-child')
        last_child = self.soup.select('p:last-child')
        nth_child = self.soup.select('li:nth-child(2)')

        # 10. Multiple selectors
        headers = self.soup.select('h1, h2, h3')

        # 11. Complex combinations
        complex = self.soup.select('article.featured div.content p:not(.meta)')

        return {
            'external_links': [link.get('href') for link in external_links],
            'article_paragraphs': len(article_paragraphs),
            'headers': len(headers)
        }

# Test HTML
html = """
<div id="header">
    <h1>Main Title</h1>
</div>
<article class="featured">
    <h2>Article Title</h2>
    <p>First paragraph</p>
    <div class="content">
        <p>Content paragraph</p>
        <p class="meta">Metadata</p>
    </div>
</article>
<a href="https://example.com" target="_blank">External</a>
<a href="/internal">Internal</a>
<a href="document.pdf">PDF</a>
"""

selector = CSSSelector(html)
results = selector.demonstrate_selectors()
print(results)
```

### Örnek 7: XPath

```python
from lxml import html
import requests

class XPathParser:
    """
    XPath kullanarak element seçimi.
    XPath, CSS selectors'tan daha güçlüdür.
    """

    def __init__(self, html_content):
        self.tree = html.fromstring(html_content)

    def demonstrate_xpath(self):
        """XPath örnekleri"""

        # 1. Basic path
        all_divs = self.tree.xpath('//div')

        # 2. Attribute filtering
        featured = self.tree.xpath('//article[@class="featured"]')

        # 3. Text content filtering
        specific_text = self.tree.xpath('//p[contains(text(), "Python")]')

        # 4. Multiple conditions
        complex = self.tree.xpath('//div[@class="content" and @data-type="article"]')

        # 5. Parent selection
        parent = self.tree.xpath('//a[@class="read-more"]/parent::article')

        # 6. Ancestor selection
        ancestors = self.tree.xpath('//a[@class="read-more"]/ancestor::div')

        # 7. Following sibling
        next_sibling = self.tree.xpath('//h2/following-sibling::p[1]')

        # 8. Attribute extraction
        hrefs = self.tree.xpath('//a/@href')

        # 9. Text extraction
        texts = self.tree.xpath('//p/text()')

        # 10. Position-based selection
        first_paragraph = self.tree.xpath('(//p)[1]')
        last_paragraph = self.tree.xpath('(//p)[last()]')

        # 11. Starts-with ve ends-with
        external_links = self.tree.xpath('//a[starts-with(@href, "http")]')

        # 12. Not operator
        non_meta_paragraphs = self.tree.xpath('//p[not(@class="meta")]')

        # 13. Or operator
        headers = self.tree.xpath('//h1 | //h2 | //h3')

        return {
            'all_divs': len(all_divs),
            'hrefs': hrefs,
            'texts': texts[:3],
            'headers': len(headers)
        }

    def extract_table_data(self):
        """Table'dan veri extract etme"""
        rows = []
        for tr in self.tree.xpath('//table//tr'):
            row_data = []
            # Header cells (th) ve data cells (td)
            cells = tr.xpath('.//th/text() | .//td/text()')
            if cells:
                rows.append(cells)
        return rows

# Test
html_content = """
<div>
    <article class="featured">
        <h2>Python Tutorial</h2>
        <p>Learn Python programming</p>
        <a class="read-more" href="/article/1">Read more</a>
    </article>
    <p class="meta">Published: 2024</p>
    <p>Regular paragraph</p>
</div>
<table>
    <tr><th>Name</th><th>Age</th></tr>
    <tr><td>John</td><td>30</td></tr>
    <tr><td>Jane</td><td>25</td></tr>
</table>
"""

parser = XPathParser(html_content)
results = parser.demonstrate_xpath()
print("XPath results:", results)

table_data = parser.extract_table_data()
print("Table data:", table_data)
```

---

## Selenium Automation

### Örnek 8: Selenium Basics

```python
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import time

class SeleniumScraper:
    """
    Selenium ile dinamik içerik scraping.
    JavaScript ile render edilen sayfalar için gerekli.
    """

    def __init__(self, headless=True):
        options = webdriver.ChromeOptions()

        if headless:
            options.add_argument('--headless')

        # Performance optimizations
        options.add_argument('--disable-gpu')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--disable-blink-features=AutomationControlled')

        # Anti-detection
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_experimental_option('useAutomationExtension', False)

        self.driver = webdriver.Chrome(options=options)
        self.wait = WebDriverWait(self.driver, 10)

    def scrape_dynamic_content(self, url):
        """JavaScript ile yüklenen içeriği scrape et"""
        try:
            self.driver.get(url)

            # Sayfa yüklenene kadar bekle
            self.wait.until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )

            # Scroll to bottom (lazy loading için)
            self.scroll_to_bottom()

            # Elementi bekle ve bul
            element = self.wait.until(
                EC.presence_of_element_located((By.CLASS_NAME, "content"))
            )

            return element.text

        except TimeoutException:
            print("Timeout: Element bulunamadı")
            return None

    def scroll_to_bottom(self, pause_time=1):
        """Sayfayı en alta scroll et (infinite scroll için)"""
        last_height = self.driver.execute_script("return document.body.scrollHeight")

        while True:
            # Scroll down
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(pause_time)

            # Yeni height hesapla
            new_height = self.driver.execute_script("return document.body.scrollHeight")

            if new_height == last_height:
                break

            last_height = new_height

    def handle_pagination(self, url, max_pages=5):
        """Pagination ile tüm sayfaları scrape et"""
        all_data = []

        self.driver.get(url)

        for page in range(max_pages):
            try:
                # Sayfadaki verileri topla
                items = self.driver.find_elements(By.CLASS_NAME, "item")
                page_data = [item.text for item in items]
                all_data.extend(page_data)

                # Next button'a tıkla
                next_button = self.wait.until(
                    EC.element_to_be_clickable((By.CLASS_NAME, "next"))
                )
                next_button.click()

                # Yeni sayfa yüklenene kadar bekle
                time.sleep(2)

            except (TimeoutException, NoSuchElementException):
                print(f"Pagination ended at page {page + 1}")
                break

        return all_data

    def fill_form(self, url, form_data):
        """Form doldurma ve submit etme"""
        self.driver.get(url)

        try:
            # Input field'ları doldur
            for field_name, value in form_data.items():
                input_field = self.wait.until(
                    EC.presence_of_element_located((By.NAME, field_name))
                )
                input_field.clear()
                input_field.send_keys(value)

            # Submit button
            submit_button = self.driver.find_element(By.CSS_SELECTOR, 'button[type="submit"]')
            submit_button.click()

            # Sonuç sayfasını bekle
            self.wait.until(
                EC.url_changes(url)
            )

            return self.driver.current_url

        except Exception as e:
            print(f"Form submission error: {e}")
            return None

    def handle_javascript_click(self, url, selector):
        """JavaScript click event'i handle et"""
        self.driver.get(url)

        # Element'i bul ve görünür olmasını bekle
        element = self.wait.until(
            EC.visibility_of_element_located((By.CSS_SELECTOR, selector))
        )

        # JavaScript ile click (bazı durumlarda normal click çalışmaz)
        self.driver.execute_script("arguments[0].click();", element)

        time.sleep(2)  # İçerik yüklensin

        return self.driver.page_source

    def close(self):
        """Browser'ı kapat"""
        self.driver.quit()

# Kullanım
# scraper = SeleniumScraper(headless=True)
# try:
#     content = scraper.scrape_dynamic_content('https://example.com')
#     print(content)
# finally:
#     scraper.close()
```

### Örnek 9: Advanced Selenium Patterns

```python
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait, Select
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
import time

class AdvancedSeleniumPatterns:
    """
    Selenium ile advanced pattern'ler.
    """

    def __init__(self):
        self.driver = webdriver.Chrome()
        self.wait = WebDriverWait(self.driver, 10)
        self.actions = ActionChains(self.driver)

    def handle_dropdown(self, url, dropdown_selector):
        """Dropdown menu'leri handle et"""
        self.driver.get(url)

        # Select element'i bul
        dropdown = Select(self.driver.find_element(By.ID, dropdown_selector))

        # Farklı seçim methodları
        dropdown.select_by_index(1)  # Index ile
        dropdown.select_by_value('value')  # Value attribute ile
        dropdown.select_by_visible_text('Visible Text')  # Görünen text ile

        # Tüm option'ları al
        all_options = dropdown.options
        for option in all_options:
            print(option.text)

    def handle_alerts(self):
        """JavaScript alert'leri handle et"""
        try:
            # Alert'i bekle
            alert = self.wait.until(EC.alert_is_present())

            # Alert text'ini al
            alert_text = alert.text
            print(f"Alert message: {alert_text}")

            # Alert'i accept et
            alert.accept()

            # Veya cancel et
            # alert.dismiss()

        except TimeoutException:
            print("No alert found")

    def handle_iframes(self, iframe_selector):
        """iframe içindeki elementleri handle et"""
        # iframe'e geç
        iframe = self.driver.find_element(By.CSS_SELECTOR, iframe_selector)
        self.driver.switch_to.frame(iframe)

        # iframe içindeki elementlerle çalış
        content = self.driver.find_element(By.ID, "content")
        text = content.text

        # Ana sayfaya geri dön
        self.driver.switch_to.default_content()

        return text

    def handle_multiple_windows(self):
        """Multiple window/tab'ları handle et"""
        # Orijinal window handle
        original_window = self.driver.current_window_handle

        # Yeni window açan link'e tıkla
        link = self.driver.find_element(By.LINK_TEXT, "Open New Window")
        link.click()

        # Yeni window'un açılmasını bekle
        self.wait.until(EC.number_of_windows_to_be(2))

        # Tüm window handle'ları al
        all_windows = self.driver.window_handles

        # Yeni window'a geç
        for window in all_windows:
            if window != original_window:
                self.driver.switch_to.window(window)
                break

        # Yeni window'da işlem yap
        content = self.driver.find_element(By.TAG_NAME, "body").text

        # Yeni window'u kapat ve orijinale dön
        self.driver.close()
        self.driver.switch_to.window(original_window)

        return content

    def hover_and_click(self, hover_selector, click_selector):
        """Hover ve click işlemleri"""
        # Hover yapılacak element
        hover_element = self.driver.find_element(By.CSS_SELECTOR, hover_selector)

        # Hover action
        self.actions.move_to_element(hover_element).perform()

        # Hover sonrası görünen element'i bekle
        click_element = self.wait.until(
            EC.visibility_of_element_located((By.CSS_SELECTOR, click_selector))
        )

        # Click
        click_element.click()

    def drag_and_drop(self, source_selector, target_selector):
        """Drag and drop işlemi"""
        source = self.driver.find_element(By.CSS_SELECTOR, source_selector)
        target = self.driver.find_element(By.CSS_SELECTOR, target_selector)

        # Drag and drop action
        self.actions.drag_and_drop(source, target).perform()

    def wait_for_ajax(self, timeout=10):
        """AJAX request'lerin tamamlanmasını bekle"""
        wait = WebDriverWait(self.driver, timeout)

        # jQuery kullanıyorsa
        wait.until(lambda driver: driver.execute_script(
            "return jQuery.active == 0"
        ))

        # Document ready state
        wait.until(lambda driver: driver.execute_script(
            "return document.readyState"
        ) == "complete")

    def close(self):
        self.driver.quit()
```

---

## Headless Browsers

### Örnek 10: Playwright

```python
"""
Playwright: Modern, headless browser automation.
Selenium'a alternatif, daha hızlı ve güvenilir.

Kurulum: pip install playwright
         playwright install
"""

from playwright.sync_api import sync_playwright
import time

class PlaywrightScraper:
    """
    Playwright ile modern web scraping.
    """

    def __init__(self, headless=True):
        self.headless = headless
        self.playwright = None
        self.browser = None
        self.context = None

    def __enter__(self):
        self.playwright = sync_playwright().start()
        self.browser = self.playwright.chromium.launch(headless=self.headless)

        # Browser context oluştur (cookie'ler, cache vs için)
        self.context = self.browser.new_context(
            viewport={'width': 1920, 'height': 1080},
            user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        )

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.context:
            self.context.close()
        if self.browser:
            self.browser.close()
        if self.playwright:
            self.playwright.stop()

    def scrape_page(self, url):
        """Basit sayfa scraping"""
        page = self.context.new_page()

        try:
            # Sayfayı yükle
            page.goto(url, wait_until='networkidle')

            # Element'i bekle
            page.wait_for_selector('.content')

            # İçeriği al
            content = page.inner_text('.content')

            # Screenshot al (debugging için)
            page.screenshot(path='screenshot.png')

            return content

        finally:
            page.close()

    def handle_spa(self, url):
        """Single Page Application scraping"""
        page = self.context.new_page()

        try:
            page.goto(url)

            # JavaScript'in render etmesini bekle
            page.wait_for_load_state('networkidle')

            # Belirli bir element'in render edilmesini bekle
            page.wait_for_selector('[data-testid="content"]')

            # Lazy loading için scroll
            for _ in range(5):
                page.evaluate('window.scrollTo(0, document.body.scrollHeight)')
                time.sleep(1)

            # Tüm içeriği al
            html = page.content()

            return html

        finally:
            page.close()

    def intercept_requests(self, url):
        """Network request'leri intercept et"""
        page = self.context.new_page()

        api_responses = []

        # Request handler
        def handle_response(response):
            if '/api/' in response.url:
                api_responses.append({
                    'url': response.url,
                    'status': response.status,
                    'data': response.json() if 'json' in response.headers.get('content-type', '') else None
                })

        page.on('response', handle_response)

        try:
            page.goto(url)
            page.wait_for_load_state('networkidle')

            return api_responses

        finally:
            page.close()

    def execute_javascript(self, url):
        """JavaScript execution"""
        page = self.context.new_page()

        try:
            page.goto(url)

            # JavaScript kodu çalıştır
            result = page.evaluate('''() => {
                const title = document.querySelector('h1').textContent;
                const links = Array.from(document.querySelectorAll('a'))
                    .map(a => ({
                        text: a.textContent,
                        href: a.href
                    }));
                return { title, links };
            }''')

            return result

        finally:
            page.close()

# Kullanım
# with PlaywrightScraper(headless=True) as scraper:
#     content = scraper.scrape_page('https://example.com')
#     print(content)
```

---

## Rate Limiting ve Throttling

### Örnek 11: Rate Limiter

```python
import time
from functools import wraps
from collections import deque
import threading

class RateLimiter:
    """
    Rate limiting implementation.
    Belirli bir sürede maksimum request sayısını sınırlar.
    """

    def __init__(self, max_calls, time_window):
        """
        max_calls: Maksimum çağrı sayısı
        time_window: Zaman penceresi (saniye)
        """
        self.max_calls = max_calls
        self.time_window = time_window
        self.calls = deque()
        self.lock = threading.Lock()

    def __call__(self, func):
        """Decorator olarak kullanım"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            with self.lock:
                now = time.time()

                # Eski çağrıları temizle
                while self.calls and self.calls[0] < now - self.time_window:
                    self.calls.popleft()

                # Rate limit kontrolü
                if len(self.calls) >= self.max_calls:
                    sleep_time = self.time_window - (now - self.calls[0])
                    if sleep_time > 0:
                        print(f"Rate limit reached. Sleeping for {sleep_time:.2f}s")
                        time.sleep(sleep_time)
                        # Yeniden temizle
                        self.calls.clear()

                # Çağrıyı kaydet
                self.calls.append(time.time())

            return func(*args, **kwargs)

        return wrapper

# Kullanım
@RateLimiter(max_calls=10, time_window=60)  # 60 saniyede max 10 request
def fetch_url(url):
    """URL'den veri çek"""
    print(f"Fetching: {url}")
    # Simulated request
    time.sleep(0.5)
    return f"Data from {url}"

# Test
# for i in range(15):
#     result = fetch_url(f'https://example.com/page/{i}')
#     print(result)
```

### Örnek 12: Adaptive Rate Limiting

```python
import time
import random

class AdaptiveRateLimiter:
    """
    Adaptive rate limiting.
    Server response'una göre rate'i otomatik ayarlar.
    """

    def __init__(self, initial_delay=1.0, min_delay=0.5, max_delay=10.0):
        self.delay = initial_delay
        self.min_delay = min_delay
        self.max_delay = max_delay
        self.success_count = 0
        self.error_count = 0

    def wait(self):
        """Bekle ve jitter ekle"""
        # Jitter: randomize delay to avoid thundering herd
        jitter = random.uniform(0, 0.1 * self.delay)
        time.sleep(self.delay + jitter)

    def on_success(self):
        """Başarılı request sonrası"""
        self.success_count += 1
        self.error_count = 0

        # 5 başarılı request'ten sonra delay'i azalt
        if self.success_count >= 5:
            self.delay = max(self.min_delay, self.delay * 0.9)
            self.success_count = 0
            print(f"Decreased delay to {self.delay:.2f}s")

    def on_error(self, status_code=None):
        """Hata sonrası"""
        self.error_count += 1
        self.success_count = 0

        # 429 (Too Many Requests) özel durumu
        if status_code == 429:
            self.delay = min(self.max_delay, self.delay * 2)
            print(f"429 error: Increased delay to {self.delay:.2f}s")
        # Diğer hatalar
        elif self.error_count >= 3:
            self.delay = min(self.max_delay, self.delay * 1.5)
            print(f"Multiple errors: Increased delay to {self.delay:.2f}s")

    def reset(self):
        """Rate limiter'ı sıfırla"""
        self.delay = self.min_delay
        self.success_count = 0
        self.error_count = 0

# Kullanım
import requests

def scrape_with_adaptive_rate(urls):
    """Adaptive rate limiting ile scraping"""
    limiter = AdaptiveRateLimiter(initial_delay=1.0)
    results = []

    for url in urls:
        limiter.wait()

        try:
            response = requests.get(url, timeout=10)

            if response.status_code == 200:
                limiter.on_success()
                results.append(response.text)
            elif response.status_code == 429:
                limiter.on_error(429)
                # Retry aynı URL
                time.sleep(limiter.delay)
                continue
            else:
                limiter.on_error(response.status_code)

        except requests.RequestException as e:
            limiter.on_error()
            print(f"Error: {e}")

    return results

# Test URLs
# test_urls = [f'https://httpbin.org/delay/{i%3}' for i in range(10)]
# results = scrape_with_adaptive_rate(test_urls)
```

---

## User Agents ve Headers

### Örnek 13: User Agent Rotation

```python
import random
import requests

class UserAgentRotator:
    """
    User agent rotation.
    Bot detection'ı engellemek için farklı user agent'lar kullan.
    """

    def __init__(self):
        self.user_agents = [
            # Chrome on Windows
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            # Chrome on Mac
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            # Firefox on Windows
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0',
            # Firefox on Mac
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:121.0) Gecko/20100101 Firefox/121.0',
            # Safari on Mac
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15',
            # Edge on Windows
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0',
        ]

    def get_random_user_agent(self):
        """Rastgele user agent döndür"""
        return random.choice(self.user_agents)

    def get_headers(self, referer=None):
        """Realistic browser headers"""
        headers = {
            'User-Agent': self.get_random_user_agent(),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Cache-Control': 'max-age=0',
        }

        if referer:
            headers['Referer'] = referer

        return headers

class SmartScraper:
    """
    User agent rotation ile smart scraping.
    """

    def __init__(self):
        self.ua_rotator = UserAgentRotator()
        self.session = requests.Session()

    def fetch(self, url, referer=None):
        """Rotating user agent ile fetch"""
        headers = self.ua_rotator.get_headers(referer)

        try:
            response = self.session.get(url, headers=headers, timeout=10)
            response.raise_for_status()

            print(f"Success with UA: {headers['User-Agent'][:50]}...")
            return response

        except requests.RequestException as e:
            print(f"Error: {e}")
            raise

    def scrape_with_referer_chain(self, urls):
        """Referer chain ile scraping (daha realistic)"""
        results = []
        previous_url = None

        for url in urls:
            response = self.fetch(url, referer=previous_url)
            results.append(response.text)
            previous_url = url

        return results

# Kullanım
scraper = SmartScraper()
# response = scraper.fetch('https://httpbin.org/headers')
# print(response.json())
```

---

## Session Management

### Örnek 14: Advanced Session

```python
import requests
import pickle
import os
from datetime import datetime, timedelta

class PersistentSession:
    """
    Persistent session management.
    Cookie'leri ve session state'i dosyaya kaydeder.
    """

    def __init__(self, session_file='session.pkl'):
        self.session_file = session_file
        self.session = self._load_session()

    def _load_session(self):
        """Session'ı dosyadan yükle"""
        if os.path.exists(self.session_file):
            try:
                with open(self.session_file, 'rb') as f:
                    session_data = pickle.load(f)

                # Expiry kontrolü
                if session_data.get('expires_at') and \
                   datetime.now() < session_data['expires_at']:
                    session = requests.Session()
                    session.cookies.update(session_data['cookies'])
                    session.headers.update(session_data['headers'])
                    print("Session loaded from file")
                    return session
            except Exception as e:
                print(f"Error loading session: {e}")

        # Yeni session oluştur
        return requests.Session()

    def save_session(self, expires_in_hours=24):
        """Session'ı dosyaya kaydet"""
        session_data = {
            'cookies': dict(self.session.cookies),
            'headers': dict(self.session.headers),
            'expires_at': datetime.now() + timedelta(hours=expires_in_hours)
        }

        with open(self.session_file, 'wb') as f:
            pickle.dump(session_data, f)

        print("Session saved to file")

    def login(self, login_url, credentials):
        """Login ve session kaydet"""
        response = self.session.post(login_url, data=credentials)

        if response.ok:
            self.save_session()
            return True
        return False

    def get(self, url, **kwargs):
        """GET request"""
        return self.session.get(url, **kwargs)

    def post(self, url, **kwargs):
        """POST request"""
        return self.session.post(url, **kwargs)

    def clear_session(self):
        """Session'ı temizle"""
        if os.path.exists(self.session_file):
            os.remove(self.session_file)
        self.session = requests.Session()

# Kullanım
# session = PersistentSession()
# session.login('https://example.com/login', {'user': 'test', 'pass': 'test'})
# response = session.get('https://example.com/protected')
```

---

## Error Handling ve Retry Logic

### Örnek 15: Robust Error Handling

```python
import requests
from requests.exceptions import (
    RequestException, HTTPError, ConnectionError,
    Timeout, TooManyRedirects
)
import time
import logging
from functools import wraps

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ScrapingError(Exception):
    """Custom scraping error"""
    pass

def retry_with_backoff(max_retries=3, backoff_factor=2):
    """
    Retry decorator with exponential backoff.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            retries = 0

            while retries < max_retries:
                try:
                    return func(*args, **kwargs)

                except (ConnectionError, Timeout) as e:
                    retries += 1
                    if retries >= max_retries:
                        logger.error(f"Max retries reached: {e}")
                        raise ScrapingError(f"Failed after {max_retries} retries") from e

                    wait_time = backoff_factor ** retries
                    logger.warning(f"Retry {retries}/{max_retries} after {wait_time}s")
                    time.sleep(wait_time)

                except HTTPError as e:
                    # HTTP error'lar için retry yapma
                    logger.error(f"HTTP error: {e}")
                    raise ScrapingError(f"HTTP error: {e.response.status_code}") from e

                except Exception as e:
                    logger.error(f"Unexpected error: {e}")
                    raise ScrapingError(f"Unexpected error: {str(e)}") from e

        return wrapper
    return decorator

class RobustScraper:
    """
    Comprehensive error handling ile scraper.
    """

    def __init__(self):
        self.session = requests.Session()

    @retry_with_backoff(max_retries=3, backoff_factor=2)
    def fetch_url(self, url, timeout=10):
        """Robust URL fetching"""
        try:
            response = self.session.get(url, timeout=timeout)

            # Status code kontrolü
            if response.status_code == 404:
                raise ScrapingError(f"Page not found: {url}")
            elif response.status_code == 403:
                raise ScrapingError(f"Access forbidden: {url}")
            elif response.status_code == 429:
                # Rate limit - retry edilebilir
                retry_after = response.headers.get('Retry-After', 60)
                logger.warning(f"Rate limited. Retry after {retry_after}s")
                time.sleep(int(retry_after))
                raise ConnectionError("Rate limited")

            response.raise_for_status()
            return response

        except Timeout:
            logger.error(f"Timeout: {url}")
            raise

        except ConnectionError as e:
            logger.error(f"Connection error: {url}")
            raise

        except TooManyRedirects:
            raise ScrapingError(f"Too many redirects: {url}")

    def safe_scrape(self, urls):
        """Tüm URL'leri güvenli şekilde scrape et"""
        results = {'success': [], 'failed': []}

        for url in urls:
            try:
                response = self.fetch_url(url)
                results['success'].append({
                    'url': url,
                    'content': response.text,
                    'status': response.status_code
                })
                logger.info(f"Successfully scraped: {url}")

            except ScrapingError as e:
                results['failed'].append({
                    'url': url,
                    'error': str(e)
                })
                logger.error(f"Failed to scrape {url}: {e}")

            # Rate limiting
            time.sleep(1)

        return results

# Kullanım
scraper = RobustScraper()
test_urls = [
    'https://httpbin.org/status/200',
    'https://httpbin.org/status/404',
    'https://httpbin.org/delay/2',
]
# results = scraper.safe_scrape(test_urls)
# print(f"Success: {len(results['success'])}, Failed: {len(results['failed'])}")
```

---

## Robots.txt ve Etik

### Örnek 16: Robots.txt Parser

```python
from urllib.robotparser import RobotFileParser
from urllib.parse import urljoin, urlparse
import time

class RobotsChecker:
    """
    Robots.txt kontrolü yapan sınıf.
    Etik scraping için gerekli.
    """

    def __init__(self, user_agent='*'):
        self.user_agent = user_agent
        self.parsers = {}  # Domain -> RobotFileParser mapping

    def can_fetch(self, url):
        """URL'in fetch edilip edilemeyeceğini kontrol et"""
        parsed = urlparse(url)
        domain = f"{parsed.scheme}://{parsed.netloc}"

        # Domain için parser yoksa oluştur
        if domain not in self.parsers:
            robots_url = urljoin(domain, '/robots.txt')
            parser = RobotFileParser()
            parser.set_url(robots_url)

            try:
                parser.read()
                self.parsers[domain] = parser
            except Exception as e:
                print(f"Error reading robots.txt from {domain}: {e}")
                # robots.txt okunamazsa, her şeye izin ver (varsayılan)
                return True

        parser = self.parsers[domain]
        return parser.can_fetch(self.user_agent, url)

    def get_crawl_delay(self, url):
        """Domain için crawl delay'i al"""
        parsed = urlparse(url)
        domain = f"{parsed.scheme}://{parsed.netloc}"

        if domain in self.parsers:
            parser = self.parsers[domain]
            delay = parser.crawl_delay(self.user_agent)
            return delay if delay else 1  # Default 1 saniye

        return 1

class EthicalScraper:
    """
    Etik kurallara uyan scraper.
    """

    def __init__(self, user_agent='MyBot/1.0'):
        self.user_agent = user_agent
        self.robots_checker = RobotsChecker(user_agent)
        self.last_fetch_time = {}  # Domain -> timestamp

    def fetch(self, url):
        """Etik kurallara uygun fetch"""
        # 1. Robots.txt kontrolü
        if not self.robots_checker.can_fetch(url):
            raise PermissionError(f"Robots.txt forbids fetching: {url}")

        # 2. Crawl delay kontrolü
        parsed = urlparse(url)
        domain = f"{parsed.scheme}://{parsed.netloc}"
        crawl_delay = self.robots_checker.get_crawl_delay(url)

        # Son fetch'ten bu yana geçen süre
        if domain in self.last_fetch_time:
            elapsed = time.time() - self.last_fetch_time[domain]
            if elapsed < crawl_delay:
                wait_time = crawl_delay - elapsed
                print(f"Waiting {wait_time:.2f}s for crawl delay...")
                time.sleep(wait_time)

        # 3. Fetch
        import requests
        headers = {'User-Agent': self.user_agent}
        response = requests.get(url, headers=headers, timeout=10)

        # Son fetch zamanını kaydet
        self.last_fetch_time[domain] = time.time()

        return response

# Kullanım
scraper = EthicalScraper(user_agent='MyResearchBot/1.0')

urls_to_scrape = [
    'https://www.python.org/',
    'https://www.python.org/about/',
]

for url in urls_to_scrape:
    try:
        response = scraper.fetch(url)
        print(f"Fetched {url}: {response.status_code}")
    except PermissionError as e:
        print(f"Permission denied: {e}")
```

### Örnek 17: Etik Scraping Best Practices

```python
"""
Etik Web Scraping Kuralları:

1. Robots.txt'e Uyun
   - Her zaman robots.txt'i kontrol edin
   - Yasak URL'leri scrape etmeyin

2. Rate Limiting
   - Crawl-delay direktifine uyun
   - Server'ı overwhelming etmeyin
   - Makul bekleme süreleri kullanın (1-2 saniye)

3. User Agent
   - Descriptive user agent kullanın
   - Bot olduğunuzu belirtin
   - İletişim bilgisi ekleyin

4. Terms of Service (ToS)
   - Web sitesinin ToS'unu okuyun
   - Yasal sınırlamalara uyun

5. Kişisel Veriler
   - GDPR, CCPA gibi düzenlemelere uyun
   - Kişisel verileri toplarken dikkatli olun

6. Attribution
   - Scrape edilen veriyi kullanırken kaynak belirtin
   - Telif haklarına saygı gösterin

7. Server Load
   - Peak hours'ta scraping yapmaktan kaçının
   - Parallel requests'i sınırlayın

8. Error Handling
   - Server error'larında scraping'i durdurun
   - 503, 500 gibi error'lara saygı gösterin
"""

class BestPracticesScraper:
    """
    En iyi pratikleri uygulayan scraper.
    """

    def __init__(self, bot_name, contact_email):
        self.user_agent = f"{bot_name} (+{contact_email})"
        self.robots_checker = RobotsChecker(bot_name)
        self.request_count = 0
        self.max_requests_per_session = 1000

    def check_compliance(self, url):
        """Compliance kontrolü"""
        checks = {
            'robots_allowed': self.robots_checker.can_fetch(url),
            'under_request_limit': self.request_count < self.max_requests_per_session,
        }

        if not all(checks.values()):
            failed = [k for k, v in checks.items() if not v]
            raise PermissionError(f"Compliance checks failed: {failed}")

        return True

    def log_request(self, url, status_code):
        """Request'i log'la"""
        self.request_count += 1
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        log_entry = f"[{timestamp}] {self.user_agent} -> {url} (Status: {status_code})"

        # Log dosyasına yaz
        with open('scraping.log', 'a') as f:
            f.write(log_entry + '\n')

# Good User Agent Examples:
GOOD_USER_AGENTS = [
    "ResearchBot/1.0 (+http://example.com/bot.html)",
    "MyCompanyBot/2.1 (contact@example.com)",
    "DataCollector/1.0 (+https://example.com; support@example.com)",
]

# Bad User Agent Examples:
BAD_USER_AGENTS = [
    "Mozilla/5.0",  # Pretending to be a browser
    "Python-urllib/3.9",  # Too generic
    "",  # Empty user agent
]
```

---

## Production Patterns

### Örnek 18: Production-Ready Scraper

```python
import requests
from bs4 import BeautifulSoup
import logging
from typing import List, Dict, Optional
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('scraper.log'),
        logging.StreamHandler()
    ]
)

@dataclass
class ScrapedData:
    """Scrape edilen veri modeli"""
    url: str
    title: str
    content: str
    metadata: Dict
    scraped_at: datetime

    def to_dict(self):
        return {
            'url': self.url,
            'title': self.title,
            'content': self.content,
            'metadata': self.metadata,
            'scraped_at': self.scraped_at.isoformat()
        }

class ProductionScraper:
    """
    Production-ready web scraper.

    Features:
    - Structured logging
    - Data validation
    - Error recovery
    - Data persistence
    - Monitoring
    """

    def __init__(self, config_file='config.json'):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config = self._load_config(config_file)
        self.session = self._setup_session()
        self.stats = {
            'total_requests': 0,
            'successful': 0,
            'failed': 0,
            'start_time': datetime.now()
        }

    def _load_config(self, config_file):
        """Configuration yükle"""
        default_config = {
            'user_agent': 'ProductionScraper/1.0',
            'timeout': 10,
            'max_retries': 3,
            'rate_limit': 1.0,
            'output_dir': 'scraped_data'
        }

        if Path(config_file).exists():
            with open(config_file) as f:
                config = json.load(f)
                default_config.update(config)

        return default_config

    def _setup_session(self):
        """Session setup"""
        session = requests.Session()
        session.headers.update({
            'User-Agent': self.config['user_agent']
        })
        return session

    def scrape(self, url: str) -> Optional[ScrapedData]:
        """Ana scraping metodu"""
        self.stats['total_requests'] += 1

        try:
            # Fetch
            response = self._fetch_with_retry(url)

            # Parse
            data = self._parse_response(url, response)

            # Validate
            if self._validate_data(data):
                self.stats['successful'] += 1

                # Save
                self._save_data(data)

                return data
            else:
                self.logger.warning(f"Data validation failed for {url}")
                self.stats['failed'] += 1

        except Exception as e:
            self.logger.error(f"Scraping failed for {url}: {e}", exc_info=True)
            self.stats['failed'] += 1

        return None

    def _fetch_with_retry(self, url: str) -> requests.Response:
        """Retry logic ile fetch"""
        max_retries = self.config['max_retries']

        for attempt in range(max_retries):
            try:
                response = self.session.get(
                    url,
                    timeout=self.config['timeout']
                )
                response.raise_for_status()

                self.logger.info(f"Successfully fetched: {url}")
                return response

            except requests.RequestException as e:
                if attempt == max_retries - 1:
                    raise

                self.logger.warning(f"Retry {attempt + 1}/{max_retries} for {url}")
                time.sleep(2 ** attempt)

    def _parse_response(self, url: str, response: requests.Response) -> ScrapedData:
        """Response'u parse et"""
        soup = BeautifulSoup(response.content, 'lxml')

        # Extract data
        title = soup.find('h1')
        title_text = title.get_text(strip=True) if title else 'No title'

        content = soup.find('div', class_='content')
        content_text = content.get_text(strip=True) if content else ''

        metadata = {
            'status_code': response.status_code,
            'content_type': response.headers.get('content-type'),
            'content_length': len(response.content)
        }

        return ScrapedData(
            url=url,
            title=title_text,
            content=content_text,
            metadata=metadata,
            scraped_at=datetime.now()
        )

    def _validate_data(self, data: ScrapedData) -> bool:
        """Veri validasyonu"""
        if not data.title or data.title == 'No title':
            return False

        if not data.content or len(data.content) < 100:
            return False

        return True

    def _save_data(self, data: ScrapedData):
        """Veriyi kaydet"""
        output_dir = Path(self.config['output_dir'])
        output_dir.mkdir(exist_ok=True)

        # Filename oluştur
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = output_dir / f"data_{timestamp}.json"

        # JSON olarak kaydet
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data.to_dict(), f, ensure_ascii=False, indent=2)

        self.logger.info(f"Data saved to {filename}")

    def get_stats(self) -> Dict:
        """Scraping istatistiklerini al"""
        duration = (datetime.now() - self.stats['start_time']).total_seconds()

        return {
            **self.stats,
            'duration_seconds': duration,
            'success_rate': self.stats['successful'] / max(self.stats['total_requests'], 1),
            'requests_per_second': self.stats['total_requests'] / max(duration, 1)
        }

# Kullanım
# scraper = ProductionScraper()
# data = scraper.scrape('https://example.com')
# print(scraper.get_stats())
```

### Örnek 19: Distributed Scraping

```python
"""
Distributed scraping için pattern'ler.
"""

import redis
import json
from typing import List
import hashlib

class DistributedQueue:
    """
    Redis-based distributed queue.
    Birden fazla scraper instance'ı için.
    """

    def __init__(self, redis_host='localhost', redis_port=6379):
        self.redis = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
        self.queue_key = 'scraping:queue'
        self.processing_key = 'scraping:processing'
        self.completed_key = 'scraping:completed'

    def add_urls(self, urls: List[str]):
        """URL'leri queue'ya ekle"""
        for url in urls:
            url_hash = hashlib.md5(url.encode()).hexdigest()

            # Duplicate check
            if not self.redis.sismember(self.completed_key, url_hash):
                self.redis.lpush(self.queue_key, json.dumps({
                    'url': url,
                    'hash': url_hash
                }))

    def get_next_url(self) -> Optional[Dict]:
        """Sıradaki URL'i al"""
        # Queue'dan al ve processing'e taşı
        data = self.redis.rpoplpush(self.queue_key, self.processing_key)

        if data:
            return json.loads(data)
        return None

    def mark_completed(self, url_hash: str):
        """URL'i tamamlandı olarak işaretle"""
        # Processing'den kaldır
        self.redis.lrem(self.processing_key, 0, url_hash)

        # Completed'a ekle
        self.redis.sadd(self.completed_key, url_hash)

    def get_stats(self):
        """Queue istatistikleri"""
        return {
            'queue_size': self.redis.llen(self.queue_key),
            'processing': self.redis.llen(self.processing_key),
            'completed': self.redis.scard(self.completed_key)
        }

class DistributedWorker:
    """
    Distributed scraping worker.
    """

    def __init__(self, worker_id: str):
        self.worker_id = worker_id
        self.queue = DistributedQueue()
        self.scraper = ProductionScraper()
        self.logger = logging.getLogger(f'Worker-{worker_id}')

    def run(self):
        """Worker'ı çalıştır"""
        self.logger.info(f"Worker {self.worker_id} started")

        while True:
            # Sıradaki URL'i al
            item = self.queue.get_next_url()

            if not item:
                self.logger.info("No URLs in queue, waiting...")
                time.sleep(5)
                continue

            url = item['url']
            url_hash = item['hash']

            try:
                # Scrape
                self.logger.info(f"Processing: {url}")
                data = self.scraper.scrape(url)

                if data:
                    # Tamamlandı olarak işaretle
                    self.queue.mark_completed(url_hash)
                    self.logger.info(f"Completed: {url}")

            except Exception as e:
                self.logger.error(f"Error processing {url}: {e}")

            # Rate limiting
            time.sleep(1)

# Kullanım (birden fazla worker çalıştırabilirsiniz)
# worker = DistributedWorker(worker_id='worker-1')
# worker.run()
```

### Örnek 20: Monitoring ve Alerting

```python
import smtplib
from email.mime.text import MIMEText
from datetime import datetime, timedelta

class ScraperMonitor:
    """
    Scraper monitoring ve alerting.
    """

    def __init__(self, alert_email=None):
        self.alert_email = alert_email
        self.metrics = {
            'requests': [],
            'errors': [],
            'response_times': []
        }

    def record_request(self, url, status_code, response_time):
        """Request'i kaydet"""
        self.metrics['requests'].append({
            'url': url,
            'status': status_code,
            'time': response_time,
            'timestamp': datetime.now()
        })

    def record_error(self, url, error):
        """Error'u kaydet"""
        self.metrics['errors'].append({
            'url': url,
            'error': str(error),
            'timestamp': datetime.now()
        })

    def get_health_status(self):
        """Health status kontrolü"""
        now = datetime.now()
        last_hour = now - timedelta(hours=1)

        # Son 1 saatteki metrikler
        recent_requests = [r for r in self.metrics['requests']
                          if r['timestamp'] > last_hour]
        recent_errors = [e for e in self.metrics['errors']
                        if e['timestamp'] > last_hour]

        total_requests = len(recent_requests)
        error_count = len(recent_errors)

        # Error rate
        error_rate = error_count / max(total_requests, 1)

        # Average response time
        if recent_requests:
            avg_response_time = sum(r['time'] for r in recent_requests) / len(recent_requests)
        else:
            avg_response_time = 0

        # Health status
        if error_rate > 0.5:
            status = 'CRITICAL'
        elif error_rate > 0.2:
            status = 'WARNING'
        elif avg_response_time > 5.0:
            status = 'WARNING'
        else:
            status = 'HEALTHY'

        return {
            'status': status,
            'total_requests': total_requests,
            'error_count': error_count,
            'error_rate': error_rate,
            'avg_response_time': avg_response_time
        }

    def send_alert(self, subject, message):
        """Email alert gönder"""
        if not self.alert_email:
            print(f"ALERT: {subject}\n{message}")
            return

        # Email gönderme (örnek)
        # Gerçek kullanımda SMTP ayarlarını yapılandırın
        print(f"Alert email would be sent to {self.alert_email}")
        print(f"Subject: {subject}")
        print(f"Message: {message}")

    def check_and_alert(self):
        """Health check ve gerekirse alert"""
        health = self.get_health_status()

        if health['status'] in ['CRITICAL', 'WARNING']:
            subject = f"Scraper Health: {health['status']}"
            message = f"""
            Scraper health check failed:

            Status: {health['status']}
            Total Requests: {health['total_requests']}
            Error Count: {health['error_count']}
            Error Rate: {health['error_rate']:.2%}
            Avg Response Time: {health['avg_response_time']:.2f}s
            """

            self.send_alert(subject, message)

# Kullanım
# monitor = ScraperMonitor(alert_email='admin@example.com')
# monitor.record_request('https://example.com', 200, 1.5)
# monitor.check_and_alert()
```

---

## Özet

Bu dokümanda web scraping'in ileri düzey konularını ele aldık:

### Temel Prensipler
1. **Etik Scraping**: Robots.txt, rate limiting, ToS
2. **Error Handling**: Retry logic, backoff strategies
3. **Performance**: Connection pooling, session management
4. **Anti-Detection**: User agent rotation, proxy rotation

### Teknolojiler
- **requests**: Session management, retry strategies
- **BeautifulSoup**: HTML parsing, CSS selectors
- **Selenium**: Dynamic content, JavaScript rendering
- **Playwright**: Modern headless browser automation
- **lxml**: XPath, fast parsing

### Production Patterns
- Structured logging
- Data validation
- Distributed scraping
- Monitoring ve alerting
- Configuration management

### Best Practices
1. Her zaman robots.txt'i kontrol et
2. Rate limiting kullan
3. Descriptive user agent kullan
4. Error handling yap
5. Verileri validate et
6. İstatistikleri logla
7. Yasal sınırlamalara uy

Web scraping güçlü bir araçtır, ancak sorumlu ve etik bir şekilde kullanılmalıdır!
