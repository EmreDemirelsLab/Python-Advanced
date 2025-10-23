# Regular Expressions Advanced (İleri Düzey Düzenli İfadeler)

## İçindekiler
1. [Kompleks Pattern'ler](#kompleks-patternler)
2. [Lookahead ve Lookbehind](#lookahead-ve-lookbehind)
3. [Named Groups (İsimlendirilmiş Gruplar)](#named-groups)
4. [Non-Capturing Groups](#non-capturing-groups)
5. [Backreferences (Geri Referanslar)](#backreferences)
6. [Greedy vs Lazy Matching](#greedy-vs-lazy-matching)
7. [Performance Optimization](#performance-optimization)
8. [re.compile Kullanımı](#recompile-kullanımı)
9. [Real-World Parsing](#real-world-parsing)

## Kompleks Pattern'ler

### Teori
Kompleks regex pattern'leri, birden fazla koşulu aynı anda kontrol eden ve sofistike text işleme gerektiren durumlar için kullanılır. Character classes, quantifiers ve alternation'ı birleştirerek güçlü pattern'ler oluşturabiliriz.

### Örnekler

```python
import re

# Örnek 1: Gelişmiş Email Validation
# En az bir nokta içeren ve geçerli karakterler kullanan email pattern'i
email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'

emails = [
    'user@example.com',
    'john.doe+filter@company.co.uk',
    'invalid@',
    '@invalid.com',
    'valid_email@sub.domain.com'
]

for email in emails:
    if re.match(email_pattern, email):
        print(f"✓ Geçerli: {email}")
    else:
        print(f"✗ Geçersiz: {email}")

# Örnek 2: Telefon Numarası Pattern'leri (Çoklu Format)
# Farklı formatlardaki telefon numaralarını yakalar
phone_patterns = {
    'TR_Mobile': r'^(?:0|\+90)?\s?5\d{2}\s?\d{3}\s?\d{2}\s?\d{2}$',
    'US_Format': r'^(?:\+1\s?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}$',
    'International': r'^\+\d{1,3}\s?\d{1,14}$'
}

test_phones = [
    '0555 123 45 67',
    '+90 532 111 22 33',
    '(555) 123-4567',
    '+1 415-555-0132',
    '+44 20 7123 4567'
]

for phone in test_phones:
    for format_name, pattern in phone_patterns.items():
        if re.match(pattern, phone):
            print(f"{phone} -> {format_name}")
            break

# Örnek 3: URL Parsing (Protokol, Domain, Path, Query)
url_pattern = r'^(https?|ftp)://([^/\s]+)(/[^\s?]*)?(\?[^\s]*)?$'

urls = [
    'https://www.example.com/path/to/page',
    'http://domain.com',
    'ftp://files.server.org/download?file=test.zip'
]

for url in urls:
    match = re.match(url_pattern, url)
    if match:
        protocol, domain, path, query = match.groups()
        print(f"\nURL: {url}")
        print(f"  Protokol: {protocol}")
        print(f"  Domain: {domain}")
        print(f"  Path: {path or 'yok'}")
        print(f"  Query: {query or 'yok'}")

# Örnek 4: IP Adresi Validation (IPv4)
# 0-255 arası değerleri doğru kontrol eder
ipv4_pattern = r'^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$'

ip_addresses = [
    '192.168.1.1',
    '255.255.255.255',
    '256.1.1.1',  # Geçersiz
    '192.168.1',  # Geçersiz
    '10.0.0.255'
]

for ip in ip_addresses:
    if re.match(ipv4_pattern, ip):
        print(f"✓ Geçerli IP: {ip}")
    else:
        print(f"✗ Geçersiz IP: {ip}")

# Örnek 5: Kredi Kartı Validation (Luhn Algorithm öncesi format check)
# Visa, MasterCard, Amex formatları
cc_patterns = {
    'Visa': r'^4\d{3}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}$',
    'MasterCard': r'^5[1-5]\d{2}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}$',
    'Amex': r'^3[47]\d{2}[\s-]?\d{6}[\s-]?\d{5}$'
}

test_cards = [
    '4532-1234-5678-9010',
    '5425 2334 3010 9903',
    '3782 822463 10005',
    '6011-1234-5678-9010'
]

for card in test_cards:
    for card_type, pattern in cc_patterns.items():
        if re.match(pattern, card):
            print(f"{card} -> {card_type}")
            break
    else:
        print(f"{card} -> Bilinmeyen format")
```

## Lookahead ve Lookbehind

### Teori
Lookahead ve lookbehind assertions, karakterleri "tüketmeden" pattern kontrolü yapar:
- **Positive Lookahead** `(?=...)`: Sonrasında belirtilen pattern varsa eşleşir
- **Negative Lookahead** `(?!...)`: Sonrasında belirtilen pattern yoksa eşleşir
- **Positive Lookbehind** `(?<=...)`: Öncesinde belirtilen pattern varsa eşleşir
- **Negative Lookbehind** `(?<!...)`: Öncesinde belirtilen pattern yoksa eşleşir

### Örnekler

```python
# Örnek 6: Positive Lookahead - Güçlü Şifre Kontrolü
# En az 8 karakter, büyük harf, küçük harf, rakam ve özel karakter içermeli
password_pattern = r'^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]{8,}$'

passwords = [
    'Weak123',           # Özel karakter yok
    'Strong@123',        # Geçerli
    'NoNumber@',         # Rakam yok
    'NOLOWER@123',       # Küçük harf yok
    'nouppercase@123',   # Büyük harf yok
    'Secure#Pass2024'    # Geçerli
]

for pwd in passwords:
    if re.match(password_pattern, pwd):
        print(f"✓ Güçlü şifre: {pwd}")
    else:
        print(f"✗ Zayıf şifre: {pwd}")

# Örnek 7: Negative Lookahead - JavaScript yorumları hariç kod
# // ile başlayan satırları hariç tut
text = """
function test() {
    console.log("Hello");
    // This is a comment
    return true;
    // Another comment
}
"""

# // ile başlamayan satırları bul
code_lines = re.findall(r'^(?!.*\/\/).*\S.*$', text, re.MULTILINE)
print("Yorum olmayan satırlar:")
for line in code_lines:
    print(f"  {line}")

# Örnek 8: Positive Lookbehind - Para birimi sonrası değerler
# $ işaretinden sonraki sayıları bul
price_text = "Product costs $49.99 and shipping is $5.99. Total: $55.98"
prices = re.findall(r'(?<=\$)\d+\.?\d*', price_text)
print(f"\nBulunan fiyatlar: {prices}")
print(f"Toplam: ${sum(float(p) for p in prices):.2f}")

# Örnek 9: Negative Lookbehind - @ işareti olmayan kelimeleri bul
# Mention (@username) olmayan kelimeleri seç
social_text = "Hello @john, please contact @support or check the documentation"
non_mentions = re.findall(r'(?<!@)\b[a-z]+\b(?!@)', social_text, re.IGNORECASE)
print(f"\nMention olmayan kelimeler: {non_mentions}")

# Örnek 10: Kombinasyon - HTML tag'leri arasındaki içerik
# <p> tag'i içindeki metni al, ama nested tag'leri dahil etme
html = """
<p>This is valid</p>
<p>Another paragraph</p>
<div>Not a paragraph</div>
<p>Last one</p>
"""

# <p> ile </p> arasındaki içeriği bul
paragraphs = re.findall(r'(?<=<p>).*?(?=</p>)', html)
print("\nParagraflar:")
for i, p in enumerate(paragraphs, 1):
    print(f"  {i}. {p}")

# Örnek 11: Kompleks Lookahead - Dosya uzantısı kontrolü
# Belirli uzantılara sahip olmayan dosya isimlerini bul
filenames = [
    'document.pdf',
    'image.jpg',
    'script.js',
    'config.json',
    'readme.txt',
    'data.xml'
]

# .pdf, .jpg, .png olmayan dosyalar
non_media_files = [f for f in filenames if re.match(r'^.*(?<!\.pdf)(?<!\.jpg)(?<!\.png)$', f)]
print(f"\nMedya olmayan dosyalar: {non_media_files}")
```

## Named Groups

### Teori
Named groups, pattern'deki grupları isimlendirerek daha okunabilir ve maintainable kod yazmanızı sağlar. Syntax: `(?P<name>...)`

### Örnekler

```python
# Örnek 12: Log Parsing ile Named Groups
log_pattern = r'(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) \[(?P<level>\w+)\] (?P<message>.*)'

log_entries = [
    '2024-01-15 10:30:45 [INFO] Application started',
    '2024-01-15 10:30:47 [ERROR] Database connection failed',
    '2024-01-15 10:31:00 [WARNING] High memory usage detected'
]

for log in log_entries:
    match = re.match(log_pattern, log)
    if match:
        log_data = match.groupdict()
        print(f"Zaman: {log_data['timestamp']}")
        print(f"Seviye: {log_data['level']}")
        print(f"Mesaj: {log_data['message']}\n")

# Örnek 13: URL Parsing Named Groups
url_pattern = r'(?P<protocol>https?|ftp)://(?P<domain>[^/:]+)(?::(?P<port>\d+))?(?P<path>/[^?]*)?(?:\?(?P<query>.*))?'

url = 'https://api.example.com:8080/v1/users?active=true&limit=10'
match = re.match(url_pattern, url)

if match:
    url_parts = match.groupdict()
    print("URL Bileşenleri:")
    for key, value in url_parts.items():
        print(f"  {key}: {value or 'N/A'}")

# Örnek 14: Tarih Formatları ile Named Groups
date_pattern = r'(?P<day>\d{2})[./](?P<month>\d{2})[./](?P<year>\d{4})'

dates = ['15.01.2024', '25/12/2023', '01.06.2024']

for date in dates:
    match = re.match(date_pattern, date)
    if match:
        d = match.groupdict()
        print(f"{date} -> Gün: {d['day']}, Ay: {d['month']}, Yıl: {d['year']}")

# Örnek 15: SQL Query Parsing
sql_pattern = r'(?P<action>SELECT|INSERT|UPDATE|DELETE)\s+(?P<columns>.*?)\s+FROM\s+(?P<table>\w+)(?:\s+WHERE\s+(?P<condition>.*))?'

queries = [
    'SELECT id, name FROM users WHERE active = 1',
    'INSERT name, email FROM customers',
    'DELETE * FROM logs WHERE date < 2024-01-01'
]

for query in queries:
    match = re.match(sql_pattern, query, re.IGNORECASE)
    if match:
        q = match.groupdict()
        print(f"\nQuery: {query}")
        print(f"  Action: {q['action']}")
        print(f"  Columns: {q['columns']}")
        print(f"  Table: {q['table']}")
        print(f"  Condition: {q['condition'] or 'None'}")
```

## Non-Capturing Groups

### Teori
Non-capturing groups `(?:...)`, gruplamaya izin verir ama yakalama yapmazlar. Bu, performans açısından faydalıdır ve group indexing'i basitleştirir.

### Örnekler

```python
# Örnek 16: Capturing vs Non-Capturing Karşılaştırma
text = "Contact: john@example.com or jane@company.org"

# Capturing groups kullanan pattern
capturing_pattern = r'(\w+)@([\w.]+)'
matches = re.findall(capturing_pattern, text)
print("Capturing groups:")
print(matches)  # [('john', 'example.com'), ('jane', 'company.org')]

# Non-capturing groups kullanan pattern (sadece email'i al)
non_capturing_pattern = r'\w+@(?:[\w.]+)'
matches = re.findall(non_capturing_pattern, text)
print("\nNon-capturing groups:")
print(matches)  # ['john@example.com', 'jane@company.org']

# Örnek 17: Telefon Numarası (Opsiyonel Alan Kodu)
# Alan kodu opsiyonel ama yakalanmamalı
phone_pattern = r'(?:\+90\s?)?(?:0)?(5\d{2})\s?(\d{3})\s?(\d{2})\s?(\d{2})'

phones = [
    '+90 532 123 45 67',
    '0555 987 65 43',
    '542 111 22 33'
]

for phone in phones:
    match = re.search(phone_pattern, phone)
    if match:
        # Sadece asıl numara grupları yakalanır
        print(f"{phone} -> Gruplar: {match.groups()}")

# Örnek 18: URL Protocol Non-Capturing
# Protocol'u kontrol et ama yakalama
url_pattern = r'(?:https?|ftp)://([a-zA-Z0-9.-]+)(/.*)?'

urls = [
    'https://example.com/path',
    'http://test.org',
    'ftp://files.server.net/download'
]

for url in urls:
    match = re.match(url_pattern, url)
    if match:
        domain, path = match.groups()
        print(f"Domain: {domain}, Path: {path or 'yok'}")
```

## Backreferences

### Teori
Backreferences, daha önce yakalanan grupları tekrar kullanmanıza izin verir. `\1`, `\2` gibi numaralı veya `(?P=name)` şeklinde isimli backreference'lar kullanabilirsiniz.

### Örnekler

```python
# Örnek 19: Tekrarlanan Kelime Bulma
text = "This is is a test test of the the code"

# Tekrarlanan kelimeleri bul
duplicate_pattern = r'\b(\w+)\s+\1\b'
duplicates = re.findall(duplicate_pattern, text)
print(f"Tekrarlanan kelimeler: {duplicates}")

# Tekrarları temizle
cleaned = re.sub(duplicate_pattern, r'\1', text)
print(f"Temizlenmiş: {cleaned}")

# Örnek 20: HTML/XML Tag Eşleştirme
html = """
<div>Content</div>
<span>Text</span>
<p>Paragraph</div>
<h1>Title</h1>
"""

# Eşleşen tag çiftlerini bul
tag_pattern = r'<(\w+)>.*?</\1>'
valid_tags = re.findall(tag_pattern, html, re.DOTALL)
print(f"\nGeçerli eşleşen tag'ler: {valid_tags}")

# Eşleşmeyen tag'leri bul
unmatched_pattern = r'<(\w+)>.*?</(?!\1)(\w+)>'
unmatched = re.findall(unmatched_pattern, html, re.DOTALL)
print(f"Eşleşmeyen tag çiftleri: {unmatched}")

# Örnek 21: Quote Matching (Aynı tırnak işareti)
code = """
message = "Hello World"
char = 'A'
invalid = "Mixed'
"""

# Aynı quote karakteri ile başlayıp biten string'leri bul
quote_pattern = r'(["\']).*?\1'
strings = re.findall(quote_pattern, code)
print(f"\nGeçerli string'ler: {re.findall(r'(["\']).*?\1', code)}")

# Tam string'i almak için
full_strings = re.findall(r'(["\'])(.*?)\1', code)
for quote, content in full_strings:
    print(f"Quote: {quote}, İçerik: {content}")

# Örnek 22: Named Backreference
# Aynı kelime ile başlayıp biten cümleler
text = "Start this is a test Start"
pattern = r'(?P<word>\w+).*(?P=word)'
match = re.search(pattern, text)
if match:
    print(f"\nEşleşen kelime: {match.group('word')}")
    print(f"Tam eşleşme: {match.group(0)}")
```

## Greedy vs Lazy Matching

### Teori
- **Greedy (Açgözlü)**: Varsayılan davranış, mümkün olan en uzun eşleşmeyi bulur (`*`, `+`, `?`, `{n,m}`)
- **Lazy (Tembel)**: Mümkün olan en kısa eşleşmeyi bulur (`*?`, `+?`, `??`, `{n,m}?`)

### Örnekler

```python
# Örnek 23: HTML Tag İçeriği - Greedy vs Lazy
html = "<p>First paragraph</p><p>Second paragraph</p>"

# Greedy - tüm içeriği yakalar
greedy_pattern = r'<p>.*</p>'
greedy_match = re.search(greedy_pattern, html)
print("Greedy match:")
print(f"  {greedy_match.group()}")

# Lazy - sadece ilk tag çiftini yakalar
lazy_pattern = r'<p>.*?</p>'
lazy_matches = re.findall(lazy_pattern, html)
print("\nLazy matches:")
for match in lazy_matches:
    print(f"  {match}")

# Örnek 24: JSON String Extraction
json_text = '{"name": "John", "age": 30, "city": "New York"}'

# Greedy - tüm JSON'u yakalar
greedy = re.search(r'".*"', json_text)
print(f"\nGreedy JSON: {greedy.group()}")

# Lazy - ilk key-value çiftini yakalar
lazy = re.findall(r'".*?"', json_text)
print(f"Lazy JSON: {lazy}")

# Örnek 25: Kod Yorumları (Çoklu yorum blokları)
code = """
/* Comment 1 */
int x = 5;
/* Comment 2 */
int y = 10;
/* Comment 3 */
"""

# Greedy - tüm yorumları tek bir match olarak yakalar
greedy_comments = re.findall(r'/\*.*\*/', code, re.DOTALL)
print(f"\nGreedy comments (count: {len(greedy_comments)}):")
for c in greedy_comments:
    print(f"  {c[:30]}...")

# Lazy - her yorumu ayrı yakalar
lazy_comments = re.findall(r'/\*.*?\*/', code, re.DOTALL)
print(f"\nLazy comments (count: {len(lazy_comments)}):")
for c in lazy_comments:
    print(f"  {c.strip()}")

# Örnek 26: Path Extraction
paths = "C:\\Users\\John\\Documents\\file.txt and D:\\Projects\\code.py"

# Greedy
greedy_path = re.findall(r'[A-Z]:\\.*\\', paths)
print(f"\nGreedy paths: {greedy_path}")

# Lazy
lazy_path = re.findall(r'[A-Z]:\\.*?\\', paths)
print(f"Lazy paths: {lazy_path}")
```

## Performance Optimization

### Teori
Regex performansını optimize etmek için:
1. Compiled pattern'leri kullan (`re.compile`)
2. Gereksiz capturing groups'tan kaçın
3. Specific pattern'leri genel olanların önüne koy
4. Catastrophic backtracking'den kaçın
5. Anchor'ları (`^`, `$`) kullan

### Örnekler

```python
import time

# Örnek 27: Performans Karşılaştırması
test_text = "email1@test.com, email2@example.org, " * 1000

# Yöntem 1: Her seferinde compile
start = time.time()
for _ in range(100):
    re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', test_text)
uncompiled_time = time.time() - start

# Yöntem 2: Pre-compiled
email_regex = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
start = time.time()
for _ in range(100):
    email_regex.findall(test_text)
compiled_time = time.time() - start

print(f"Uncompiled: {uncompiled_time:.4f}s")
print(f"Compiled: {compiled_time:.4f}s")
print(f"Speedup: {uncompiled_time/compiled_time:.2f}x")

# Örnek 28: Anchor Kullanımı
# Tüm string'i taramak yerine baştan kontrol
text_lines = ["Valid line", "Invalid", "Another valid line"] * 100

# Anchor'sız (tüm satırı tarar)
start = time.time()
for line in text_lines:
    re.search(r'Valid', line)
no_anchor = time.time() - start

# Anchor'lu (sadece baştan bakar)
start = time.time()
for line in text_lines:
    re.search(r'^Valid', line)
with_anchor = time.time() - start

print(f"\nAnchor'sız: {no_anchor:.4f}s")
print(f"Anchor'lu: {with_anchor:.4f}s")

# Örnek 29: Non-Capturing Groups Performansı
text = "test@example.com, user@domain.org" * 500

# Capturing groups
pattern1 = r'(\w+)@([\w.]+)'
start = time.time()
re.findall(pattern1, text)
capturing_time = time.time() - start

# Non-capturing groups
pattern2 = r'(?:\w+)@(?:[\w.]+)'
start = time.time()
re.findall(pattern2, text)
non_capturing_time = time.time() - start

print(f"\nCapturing groups: {capturing_time:.4f}s")
print(f"Non-capturing groups: {non_capturing_time:.4f}s")
```

## re.compile Kullanımı

### Teori
`re.compile()` ile regex pattern'lerini önceden derleyerek tekrarlı kullanımlarda performans kazancı sağlayabilirsiniz. Ayrıca kod okunabilirliğini artırır.

### Örnekler

```python
# Örnek 30: Compile Edilmiş Pattern Sınıfı
class TextValidator:
    def __init__(self):
        # Tüm pattern'leri önceden compile et
        self.email_regex = re.compile(
            r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        )
        self.phone_regex = re.compile(
            r'^(?:0|\+90)?\s?5\d{2}\s?\d{3}\s?\d{2}\s?\d{2}$'
        )
        self.url_regex = re.compile(
            r'^(https?|ftp)://[^\s/$.?#].[^\s]*$',
            re.IGNORECASE
        )
        self.ipv4_regex = re.compile(
            r'^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}'
            r'(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$'
        )

    def validate_email(self, email):
        return bool(self.email_regex.match(email))

    def validate_phone(self, phone):
        return bool(self.phone_regex.match(phone))

    def validate_url(self, url):
        return bool(self.url_regex.match(url))

    def validate_ip(self, ip):
        return bool(self.ipv4_regex.match(ip))

# Kullanım
validator = TextValidator()
print(validator.validate_email("test@example.com"))  # True
print(validator.validate_phone("0532 123 45 67"))    # True
print(validator.validate_url("https://example.com")) # True
print(validator.validate_ip("192.168.1.1"))          # True

# Örnek 31: Flags ile Compile
# Case-insensitive ve multiline search
log_pattern = re.compile(
    r'^(?P<level>ERROR|WARNING|INFO):(?P<message>.*)$',
    re.IGNORECASE | re.MULTILINE
)

log_text = """
INFO: Application started
error: Database connection failed
WARNING: Low disk space
info: User logged in
"""

matches = log_pattern.finditer(log_text)
for match in matches:
    print(f"{match.group('level').upper()}: {match.group('message').strip()}")

# Örnek 32: Verbose Mode (Okunabilir Regex)
# VERBOSE flag ile regex'i açıklamalarla yazabilirsiniz
complex_email = re.compile(r'''
    ^                           # String başlangıcı
    [a-zA-Z0-9._%+-]+           # Kullanıcı adı kısmı
    @                           # @ işareti
    [a-zA-Z0-9.-]+              # Domain adı
    \.                          # Nokta
    [a-zA-Z]{2,}                # TLD (en az 2 karakter)
    $                           # String sonu
''', re.VERBOSE)

print(f"\nVerbose email validation: {complex_email.match('user@example.com')}")
```

## Real-World Parsing

### Teori
Gerçek dünya senaryolarında regex kullanımı: log parsing, data extraction, text cleaning ve format conversion.

### Örnekler

```python
# Örnek 33: Apache Log Parsing
apache_log_pattern = re.compile(
    r'(?P<ip>[\d.]+)\s+\S+\s+\S+\s+'
    r'\[(?P<datetime>[^\]]+)\]\s+'
    r'"(?P<method>\w+)\s+(?P<path>[^\s]+)\s+HTTP/[\d.]+"\s+'
    r'(?P<status>\d+)\s+'
    r'(?P<size>\d+|-)'
)

log_line = '192.168.1.1 - - [15/Jan/2024:10:30:45 +0000] "GET /index.html HTTP/1.1" 200 1234'
match = apache_log_pattern.match(log_line)

if match:
    log_data = match.groupdict()
    print("Apache Log Analizi:")
    for key, value in log_data.items():
        print(f"  {key}: {value}")

# Örnek 34: CSV Parsing (Quoted Fields)
# Virgüller quote içinde olabilir
csv_pattern = re.compile(r'''
    (?:^|,)                     # Satır başı veya virgül
    (?:                         # Group başlangıcı
        "([^"]*)"               # Quoted field
        |                       # VEYA
        ([^,]*)                 # Unquoted field
    )
''', re.VERBOSE)

csv_line = 'John,Doe,"New York, NY",30,"Software Engineer, Senior"'
fields = []

for match in csv_pattern.finditer(csv_line):
    # Quoted veya unquoted değeri al
    field = match.group(1) if match.group(1) is not None else match.group(2)
    if field:  # Boş değerleri atla
        fields.append(field)

print(f"\nCSV Fields: {fields}")

# Örnek 35: JSON-like Data Extraction
data = '''
{
    "users": [
        {"id": 1, "name": "John", "email": "john@example.com"},
        {"id": 2, "name": "Jane", "email": "jane@example.com"}
    ]
}
'''

# Email'leri çıkar
email_pattern = re.compile(r'"email":\s*"([^"]+)"')
emails = email_pattern.findall(data)
print(f"\nExtracted emails: {emails}")

# ID ve name'leri birlikte çıkar
user_pattern = re.compile(r'"id":\s*(\d+),\s*"name":\s*"([^"]+)"')
users = user_pattern.findall(data)
print(f"User ID-Name pairs: {users}")

# Örnek 36: Markdown Link Extraction
markdown = """
Check out [Python](https://python.org) and [GitHub](https://github.com).
Also see [Documentation](https://docs.python.org/3/).
"""

# [text](url) formatındaki linkleri bul
link_pattern = re.compile(r'\[([^\]]+)\]\(([^)]+)\)')
links = link_pattern.findall(markdown)

print("\nMarkdown Links:")
for text, url in links:
    print(f"  Text: {text}, URL: {url}")

# Örnek 37: SQL Injection Detection
dangerous_patterns = re.compile(
    r"('|(--)|;|\b(union|select|insert|update|delete|drop|create|alter)\b)",
    re.IGNORECASE
)

inputs = [
    "John Doe",
    "admin' OR '1'='1",
    "test; DROP TABLE users;",
    "normal_user_123",
    "1' UNION SELECT * FROM passwords--"
]

print("\nSQL Injection Taraması:")
for inp in inputs:
    if dangerous_patterns.search(inp):
        print(f"  ⚠️  Tehlikeli: {inp}")
    else:
        print(f"  ✓ Güvenli: {inp}")

# Örnek 38: HTML Cleaning (Script ve Style Temizleme)
html_content = """
<html>
<head>
    <style>body { color: red; }</style>
    <script>alert('XSS');</script>
</head>
<body>
    <p>Valid content</p>
    <script>malicious();</script>
    <div>More content</div>
</body>
</html>
"""

# Script ve style tag'lerini temizle
clean_pattern = re.compile(r'<(script|style)[^>]*>.*?</\1>', re.DOTALL | re.IGNORECASE)
cleaned_html = clean_pattern.sub('', html_content)

print("\nTemizlenmiş HTML:")
print(cleaned_html)

# Örnek 39: Environment Variable Substitution
config_text = """
database_host=${DB_HOST}
database_port=${DB_PORT}
api_key=${API_KEY}
debug_mode=${DEBUG}
"""

env_vars = {
    'DB_HOST': 'localhost',
    'DB_PORT': '5432',
    'API_KEY': 'secret_key_123',
    'DEBUG': 'true'
}

# ${VAR} formatındaki değişkenleri değiştir
var_pattern = re.compile(r'\$\{([^}]+)\}')

def replace_env_var(match):
    var_name = match.group(1)
    return env_vars.get(var_name, match.group(0))

resolved_config = var_pattern.sub(replace_env_var, config_text)
print("\nResolved Configuration:")
print(resolved_config)

# Örnek 40: Time Duration Parsing (Human Readable)
duration_pattern = re.compile(
    r'(?:(\d+)\s*h(?:our)?s?)?\s*'
    r'(?:(\d+)\s*m(?:in(?:ute)?s?)?)?\s*'
    r'(?:(\d+)\s*s(?:ec(?:ond)?s?)?)?',
    re.IGNORECASE
)

durations = [
    "2 hours 30 minutes",
    "1h 15m 30s",
    "45 minutes",
    "3h",
    "90 seconds"
]

print("\nZaman Süresi Parsing:")
for duration in durations:
    match = duration_pattern.search(duration)
    if match:
        hours, minutes, seconds = match.groups()
        total_seconds = (
            int(hours or 0) * 3600 +
            int(minutes or 0) * 60 +
            int(seconds or 0)
        )
        print(f"  {duration} -> {total_seconds} saniye")
```

## Best Practices

### Performans İpuçları
1. Sık kullanılan pattern'leri `re.compile()` ile derleyin
2. Mümkün olduğunca specific pattern'ler kullanın
3. Non-capturing groups `(?:...)` kullanarak gereksiz yakalamalardan kaçının
4. Lookahead/lookbehind yerine basit pattern'ler tercih edin (performans için)
5. `re.search()` yerine `re.match()` kullanın (eğer baştan eşleşme yeterli ise)

### Güvenlik
1. User input'ları regex'e vermeden önce sanitize edin
2. ReDoS (Regular Expression Denial of Service) saldırılarına karşı dikkatli olun
3. Catastrophic backtracking yapabilecek pattern'lerden kaçının
4. Input uzunluğunu sınırlayın

### Okunabilirlik
1. Karmaşık pattern'ler için `re.VERBOSE` flag kullanın
2. Named groups ile anlamlı isimler verin
3. Pattern'leri değişkenlerde saklayın ve açıklayıcı isimler kullanın
4. Kompleks pattern'leri daha küçük parçalara bölün

## Özet
İleri düzey regex kullanımı, kompleks text processing görevlerinde büyük güç sağlar. Lookahead/lookbehind assertions, named groups ve backreferences ile daha okunabilir ve maintainable kod yazabilirsiniz. Performance optimization ve güvenlik best practice'lerini göz önünde bulundurarak production-ready regex pattern'leri oluşturabilirsiniz.
