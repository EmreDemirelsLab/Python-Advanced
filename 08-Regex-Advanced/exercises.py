"""
REGULAR EXPRESSIONS ADVANCED - EXERCISES
İleri Düzey Düzenli İfadeler Alıştırmaları

Bu alıştırmalar gerçek dünya senaryolarını içerir:
- Log parsing ve analiz
- Email/URL extraction ve validation
- HTML/XML cleaning
- Data validation ve transformation
- Complex pattern matching
"""

import re
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import json

# ============================================================================
# EXERCISE 1: Advanced Log Parser (EXPERT)
# ============================================================================
# TODO: Kompleks log formatını parse edin ve structured data oluşturun
#
# Log formatı:
# [TIMESTAMP] [LEVEL] [MODULE:FUNCTION:LINE] MESSAGE {JSON_CONTEXT}
#
# Gereksinimler:
# 1. Timestamp'i datetime objesine çevirin
# 2. Level'ı normalize edin (uppercase)
# 3. Module, function ve line numarasını ayrıştırın
# 4. JSON context varsa parse edin
# 5. Tüm bilgileri dict olarak döndürün

def parse_advanced_log(log_line: str) -> Optional[Dict]:
    """
    Advanced log line'ı parse eder.

    Örnek:
    >>> log = '[2024-01-15 10:30:45.123] [ERROR] [auth:login:42] Login failed {"user_id": 123, "ip": "192.168.1.1"}'
    >>> result = parse_advanced_log(log)
    >>> result['level']
    'ERROR'
    >>> result['module']
    'auth'
    """
    # TODO: Implementation
    pass

# SOLUTION:
def parse_advanced_log_solution(log_line: str) -> Optional[Dict]:
    """Advanced log parser with context extraction."""
    pattern = re.compile(
        r'\[(?P<timestamp>[^\]]+)\]\s+'
        r'\[(?P<level>\w+)\]\s+'
        r'\[(?P<module>\w+):(?P<function>\w+):(?P<line>\d+)\]\s+'
        r'(?P<message>[^{]+?)\s*'
        r'(?:\{(?P<context>.*)\})?$'
    )

    match = pattern.match(log_line)
    if not match:
        return None

    data = match.groupdict()

    # Timestamp'i parse et
    try:
        data['timestamp'] = datetime.strptime(
            data['timestamp'],
            '%Y-%m-%d %H:%M:%S.%f'
        )
    except ValueError:
        data['timestamp'] = data['timestamp']

    # Level'ı normalize et
    data['level'] = data['level'].upper()

    # Line'ı integer'a çevir
    data['line'] = int(data['line'])

    # Message'ı trim et
    data['message'] = data['message'].strip()

    # JSON context'i parse et
    if data['context']:
        try:
            data['context'] = json.loads('{' + data['context'] + '}')
        except json.JSONDecodeError:
            pass

    return data


# ============================================================================
# EXERCISE 2: Email Extractor with Validation (HARD)
# ============================================================================
# TODO: Metinden email adreslerini çıkarın ve kategorize edin
#
# Gereksinimler:
# 1. Tüm email adreslerini bulun
# 2. Domain extension'a göre kategorize edin (.com, .org, .edu, etc.)
# 3. Geçersiz email formatlarını atlayın
# 4. Duplicate'leri kaldırın
# 5. Dict olarak kategorize edilmiş email listesi döndürün

def extract_and_categorize_emails(text: str) -> Dict[str, List[str]]:
    """
    Metinden email'leri çıkarır ve kategorize eder.

    Örnek:
    >>> text = "Contact: admin@company.com, info@university.edu, invalid@, test@org.org"
    >>> result = extract_and_categorize_emails(text)
    >>> 'com' in result
    True
    >>> len(result['edu'])
    1
    """
    # TODO: Implementation
    pass

# SOLUTION:
def extract_and_categorize_emails_solution(text: str) -> Dict[str, List[str]]:
    """Extract and categorize emails by TLD."""
    # Gelişmiş email pattern
    email_pattern = re.compile(
        r'\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.([a-zA-Z]{2,})\b'
    )

    categorized = {}
    seen = set()

    for match in email_pattern.finditer(text):
        email = match.group(0)
        tld = match.group(1)

        # Duplicate kontrolü
        if email.lower() not in seen:
            seen.add(email.lower())

            if tld not in categorized:
                categorized[tld] = []

            categorized[tld].append(email)

    return categorized


# ============================================================================
# EXERCISE 3: URL Parser and Validator (HARD)
# ============================================================================
# TODO: URL'leri parse edin ve bileşenlerine ayırın
#
# Gereksinimler:
# 1. Protocol, domain, port, path, query parameters, fragment'i ayırın
# 2. Query parameters'ı dict olarak parse edin
# 3. Geçersiz URL'leri None döndürün
# 4. IPv4 ve domain name destekleyin
# 5. Default port'ları ekleyin (http:80, https:443)

def parse_url(url: str) -> Optional[Dict]:
    """
    URL'i parse eder ve bileşenlerine ayırır.

    Örnek:
    >>> url = "https://api.example.com:8080/v1/users?active=true&limit=10#section"
    >>> result = parse_url(url)
    >>> result['domain']
    'api.example.com'
    >>> result['query']
    {'active': 'true', 'limit': '10'}
    """
    # TODO: Implementation
    pass

# SOLUTION:
def parse_url_solution(url: str) -> Optional[Dict]:
    """Parse URL into components."""
    pattern = re.compile(
        r'^(?P<protocol>https?|ftp)://'
        r'(?P<domain>[\w.-]+|\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'
        r'(?::(?P<port>\d+))?'
        r'(?P<path>/[^?#]*)?'
        r'(?:\?(?P<query>[^#]*))?'
        r'(?:#(?P<fragment>.*))?$'
    )

    match = pattern.match(url)
    if not match:
        return None

    result = match.groupdict()

    # Default port ekle
    if not result['port']:
        default_ports = {'http': '80', 'https': '443', 'ftp': '21'}
        result['port'] = default_ports.get(result['protocol'])

    # Query parameters'ı parse et
    if result['query']:
        query_dict = {}
        for param in result['query'].split('&'):
            if '=' in param:
                key, value = param.split('=', 1)
                query_dict[key] = value
        result['query'] = query_dict
    else:
        result['query'] = {}

    return result


# ============================================================================
# EXERCISE 4: HTML Tag Cleaner with Whitelist (HARD)
# ============================================================================
# TODO: HTML içeriğinden tehlikeli tag'leri temizleyin
#
# Gereksinimler:
# 1. Script, style, iframe tag'lerini tamamen kaldırın
# 2. Whitelist'te olmayan tag'leri strip edin
# 3. Onclick, onerror gibi event attribute'larını kaldırın
# 4. Nested tag'leri doğru şekilde handle edin
# 5. Temizlenmiş HTML döndürün

def sanitize_html(html: str, allowed_tags: List[str] = None) -> str:
    """
    HTML'i sanitize eder ve güvenli hale getirir.

    Örnek:
    >>> html = '<p>Safe</p><script>alert("XSS")</script><div onclick="bad()">Text</div>'
    >>> result = sanitize_html(html, allowed_tags=['p', 'div'])
    >>> '<script>' not in result
    True
    >>> 'onclick' not in result
    True
    """
    # TODO: Implementation
    pass

# SOLUTION:
def sanitize_html_solution(html: str, allowed_tags: List[str] = None) -> str:
    """Sanitize HTML by removing dangerous tags and attributes."""
    if allowed_tags is None:
        allowed_tags = ['p', 'div', 'span', 'a', 'b', 'i', 'u', 'strong', 'em']

    # 1. Tehlikeli tag'leri tamamen kaldır (içeriğiyle birlikte)
    dangerous_tags = re.compile(
        r'<(script|style|iframe|object|embed)[^>]*>.*?</\1>',
        re.DOTALL | re.IGNORECASE
    )
    html = dangerous_tags.sub('', html)

    # 2. Event handler'ları kaldır
    event_attrs = re.compile(
        r'\s+on\w+\s*=\s*["\'][^"\']*["\']',
        re.IGNORECASE
    )
    html = event_attrs.sub('', html)

    # 3. İzin verilmeyen tag'leri strip et (içeriği koru)
    def replace_tag(match):
        tag_name = match.group(1).lower()
        if tag_name in allowed_tags:
            return match.group(0)
        return ''

    # Opening ve closing tag'leri kontrol et
    tag_pattern = re.compile(r'</?(\w+)[^>]*>', re.IGNORECASE)
    html = tag_pattern.sub(replace_tag, html)

    return html


# ============================================================================
# EXERCISE 5: Credit Card Validator with Luhn Algorithm (EXPERT)
# ============================================================================
# TODO: Kredi kartı numaralarını validate edin
#
# Gereksinimler:
# 1. Farklı formatları destekleyin (spaces, dashes)
# 2. Kart tipini belirleyin (Visa, MasterCard, Amex, etc.)
# 3. Luhn algoritmasını uygulayın
# 4. Geçerli ise kart bilgilerini döndürün
# 5. Maskelenmiş versiyonunu oluşturun (****-****-****-1234)

def validate_credit_card(card_number: str) -> Optional[Dict]:
    """
    Kredi kartı numarasını validate eder.

    Örnek:
    >>> card = "4532-1234-5678-9010"
    >>> result = validate_credit_card(card)
    >>> result['type']
    'Visa'
    >>> result['valid']
    True
    """
    # TODO: Implementation
    pass

# SOLUTION:
def validate_credit_card_solution(card_number: str) -> Optional[Dict]:
    """Validate credit card with Luhn algorithm."""
    # Sadece rakamları al
    digits = re.sub(r'[\s-]', '', card_number)

    if not re.match(r'^\d+$', digits):
        return None

    # Kart tipini belirle
    card_patterns = {
        'Visa': r'^4\d{15}$',
        'MasterCard': r'^5[1-5]\d{14}$',
        'Amex': r'^3[47]\d{13}$',
        'Discover': r'^6(?:011|5\d{2})\d{12}$',
        'Diners': r'^3(?:0[0-5]|[68]\d)\d{11}$'
    }

    card_type = None
    for name, pattern in card_patterns.items():
        if re.match(pattern, digits):
            card_type = name
            break

    if not card_type:
        return None

    # Luhn Algorithm
    def luhn_check(card_digits):
        total = 0
        reverse_digits = card_digits[::-1]

        for i, digit in enumerate(reverse_digits):
            n = int(digit)
            if i % 2 == 1:  # Her ikinci rakamı iki katına çıkar
                n *= 2
                if n > 9:
                    n -= 9
            total += n

        return total % 10 == 0

    is_valid = luhn_check(digits)

    # Maskelenmiş versiyon
    masked = '*' * (len(digits) - 4) + digits[-4:]
    if len(digits) == 16:
        masked = '-'.join([masked[i:i+4] for i in range(0, 16, 4)])
    elif len(digits) == 15:  # Amex
        masked = masked[:4] + '-' + masked[4:10] + '-' + masked[10:]

    return {
        'type': card_type,
        'valid': is_valid,
        'masked': masked,
        'length': len(digits)
    }


# ============================================================================
# EXERCISE 6: Password Strength Analyzer (MEDIUM)
# ============================================================================
# TODO: Şifre gücünü analiz edin ve score hesaplayın
#
# Gereksinimler:
# 1. Minimum 8 karakter kontrolü
# 2. Büyük harf, küçük harf, rakam, özel karakter kontrolü
# 3. Yaygın şifreleri reddet (password123, qwerty, etc.)
# 4. Tekrarlayan karakter pattern'lerini penalize et
# 5. 0-100 arası güç skoru döndürün

def analyze_password_strength(password: str) -> Dict:
    """
    Şifre gücünü analiz eder.

    Örnek:
    >>> result = analyze_password_strength("Weak123")
    >>> result['score'] < 50
    True
    >>> result = analyze_password_strength("Str0ng!P@ssw0rd#2024")
    >>> result['score'] > 80
    True
    """
    # TODO: Implementation
    pass

# SOLUTION:
def analyze_password_strength_solution(password: str) -> Dict:
    """Analyze password strength and return score."""
    score = 0
    feedback = []

    # Length check
    length = len(password)
    if length < 8:
        feedback.append("En az 8 karakter olmalı")
    elif length < 12:
        score += 20
    elif length < 16:
        score += 30
    else:
        score += 40

    # Character type checks
    if re.search(r'[a-z]', password):
        score += 10
    else:
        feedback.append("Küçük harf ekleyin")

    if re.search(r'[A-Z]', password):
        score += 10
    else:
        feedback.append("Büyük harf ekleyin")

    if re.search(r'\d', password):
        score += 10
    else:
        feedback.append("Rakam ekleyin")

    if re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
        score += 15
    else:
        feedback.append("Özel karakter ekleyin")

    # Complexity bonus
    if re.search(r'[a-z].*[A-Z]|[A-Z].*[a-z]', password):
        score += 5

    if re.search(r'\d.*[!@#$%^&*]|[!@#$%^&*].*\d', password):
        score += 10

    # Common passwords penalty
    common_passwords = [
        'password', '123456', 'qwerty', 'abc123',
        'password123', 'admin', 'letmein', 'welcome'
    ]
    if password.lower() in common_passwords:
        score = max(0, score - 50)
        feedback.append("Yaygın şifre kullanmayın")

    # Repeating characters penalty
    if re.search(r'(.)\1{2,}', password):
        score = max(0, score - 10)
        feedback.append("Tekrarlayan karakterlerden kaçının")

    # Sequential characters penalty
    if re.search(r'(012|123|234|345|456|567|678|789|abc|bcd|cde)', password.lower()):
        score = max(0, score - 10)
        feedback.append("Ardışık karakterlerden kaçının")

    # Strength level
    if score < 30:
        strength = "Çok Zayıf"
    elif score < 50:
        strength = "Zayıf"
    elif score < 70:
        strength = "Orta"
    elif score < 90:
        strength = "Güçlü"
    else:
        strength = "Çok Güçlü"

    return {
        'score': min(100, score),
        'strength': strength,
        'feedback': feedback
    }


# ============================================================================
# EXERCISE 7: IPv4/IPv6 Address Extractor (HARD)
# ============================================================================
# TODO: Metinden IPv4 ve IPv6 adreslerini çıkarın
#
# Gereksinimler:
# 1. IPv4 formatını validate edin (0-255 range)
# 2. IPv6 formatını destekleyin (compressed ve full format)
# 3. Private vs Public IP'leri ayırın
# 4. CIDR notation'ı parse edin (/24, /64 etc.)
# 5. Her tip için ayrı listeler döndürün

def extract_ip_addresses(text: str) -> Dict[str, List[str]]:
    """
    Metinden IP adreslerini çıkarır ve kategorize eder.

    Örnek:
    >>> text = "Server: 192.168.1.1/24, Public: 8.8.8.8, IPv6: 2001:0db8::1"
    >>> result = extract_ip_addresses(text)
    >>> len(result['ipv4_private']) > 0
    True
    >>> len(result['ipv4_public']) > 0
    True
    """
    # TODO: Implementation
    pass

# SOLUTION:
def extract_ip_addresses_solution(text: str) -> Dict[str, List[str]]:
    """Extract and categorize IPv4 and IPv6 addresses."""
    result = {
        'ipv4_private': [],
        'ipv4_public': [],
        'ipv6': []
    }

    # IPv4 pattern with CIDR
    ipv4_pattern = re.compile(
        r'\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}'
        r'(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)'
        r'(?:/\d{1,2})?\b'
    )

    # IPv6 pattern (simplified)
    ipv6_pattern = re.compile(
        r'\b(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b|'
        r'\b(?:[0-9a-fA-F]{1,4}:){1,7}:\b|'
        r'\b::(?:[0-9a-fA-F]{1,4}:){0,6}[0-9a-fA-F]{1,4}\b'
    )

    # IPv4 extraction
    for match in ipv4_pattern.finditer(text):
        ip = match.group(0)
        base_ip = ip.split('/')[0]

        # Private IP ranges
        if (base_ip.startswith('10.') or
            base_ip.startswith('192.168.') or
            re.match(r'^172\.(1[6-9]|2[0-9]|3[0-1])\.', base_ip)):
            result['ipv4_private'].append(ip)
        else:
            result['ipv4_public'].append(ip)

    # IPv6 extraction
    for match in ipv6_pattern.finditer(text):
        result['ipv6'].append(match.group(0))

    return result


# ============================================================================
# EXERCISE 8: SQL Query Parser (EXPERT)
# ============================================================================
# TODO: SQL sorgularını parse edin ve analiz edin
#
# Gereksinimler:
# 1. SELECT, INSERT, UPDATE, DELETE sorgu tiplerini destekleyin
# 2. Table isimlerini çıkarın
# 3. Column isimlerini parse edin
# 4. WHERE, JOIN, ORDER BY clause'larını ayırın
# 5. Potential SQL injection pattern'lerini tespit edin

def parse_sql_query(query: str) -> Optional[Dict]:
    """
    SQL sorgusunu parse eder.

    Örnek:
    >>> query = "SELECT id, name FROM users WHERE active = 1 ORDER BY created_at"
    >>> result = parse_sql_query(query)
    >>> result['type']
    'SELECT'
    >>> 'users' in result['tables']
    True
    """
    # TODO: Implementation
    pass

# SOLUTION:
def parse_sql_query_solution(query: str) -> Optional[Dict]:
    """Parse SQL query into components."""
    # SQL injection detection
    dangerous_patterns = [
        r"('|(--)|;|\bunion\b|\bdrop\b)",
        r"(\bor\b\s+['\"]?1['\"]?\s*=\s*['\"]?1)",
    ]

    has_injection = any(re.search(p, query, re.IGNORECASE) for p in dangerous_patterns)

    # SELECT query pattern
    select_pattern = re.compile(
        r'\b(?P<type>SELECT)\s+(?P<columns>.+?)\s+'
        r'FROM\s+(?P<tables>[\w,\s]+)'
        r'(?:\s+(?P<joins>(?:INNER|LEFT|RIGHT|FULL)?\s*JOIN\s+.+?))?'
        r'(?:\s+WHERE\s+(?P<where>.+?))?'
        r'(?:\s+ORDER\s+BY\s+(?P<order>.+?))?'
        r'(?:\s+LIMIT\s+(?P<limit>\d+))?',
        re.IGNORECASE | re.DOTALL
    )

    # INSERT query pattern
    insert_pattern = re.compile(
        r'\b(?P<type>INSERT)\s+INTO\s+(?P<table>\w+)\s*'
        r'\((?P<columns>[^)]+)\)\s*'
        r'VALUES\s*\((?P<values>.+?)\)',
        re.IGNORECASE
    )

    # UPDATE query pattern
    update_pattern = re.compile(
        r'\b(?P<type>UPDATE)\s+(?P<table>\w+)\s+'
        r'SET\s+(?P<sets>.+?)'
        r'(?:\s+WHERE\s+(?P<where>.+?))?',
        re.IGNORECASE | re.DOTALL
    )

    # DELETE query pattern
    delete_pattern = re.compile(
        r'\b(?P<type>DELETE)\s+FROM\s+(?P<table>\w+)'
        r'(?:\s+WHERE\s+(?P<where>.+?))?',
        re.IGNORECASE
    )

    # Try each pattern
    for pattern in [select_pattern, insert_pattern, update_pattern, delete_pattern]:
        match = pattern.search(query)
        if match:
            result = match.groupdict()
            result['has_injection_risk'] = has_injection
            result['original_query'] = query.strip()

            # Parse tables
            if 'tables' in result and result['tables']:
                result['tables'] = [t.strip() for t in result['tables'].split(',')]
            elif 'table' in result and result['table']:
                result['tables'] = [result['table']]

            # Parse columns
            if 'columns' in result and result['columns']:
                result['columns'] = [c.strip() for c in result['columns'].split(',')]

            return result

    return None


# ============================================================================
# EXERCISE 9: Phone Number Formatter (MEDIUM)
# ============================================================================
# TODO: Farklı formatlardaki telefon numaralarını standardize edin
#
# Gereksinimler:
# 1. TR, US, UK, international formatları destekleyin
# 2. Ülke kodunu tespit edin
# 3. Standart formata çevirin
# 4. Geçersiz numaraları None döndürün
# 5. Extension numaralarını handle edin

def format_phone_number(phone: str, country: str = 'TR') -> Optional[str]:
    """
    Telefon numarasını standart formata çevirir.

    Örnek:
    >>> format_phone_number("05321234567", "TR")
    '+90 532 123 45 67'
    >>> format_phone_number("(555) 123-4567", "US")
    '+1 555 123 4567'
    """
    # TODO: Implementation
    pass

# SOLUTION:
def format_phone_number_solution(phone: str, country: str = 'TR') -> Optional[str]:
    """Format phone number to standard international format."""
    # Sadece rakamları ve + işaretini al
    cleaned = re.sub(r'[^\d+]', '', phone)

    patterns = {
        'TR': {
            'pattern': r'^(?:\+90|0)?(\d{3})(\d{3})(\d{2})(\d{2})$',
            'format': '+90 {} {} {} {}',
            'prefix': '+90'
        },
        'US': {
            'pattern': r'^(?:\+?1)?(\d{3})(\d{3})(\d{4})$',
            'format': '+1 {} {} {}',
            'prefix': '+1'
        },
        'UK': {
            'pattern': r'^(?:\+?44|0)?(\d{2,5})(\d{6,8})$',
            'format': '+44 {} {}',
            'prefix': '+44'
        }
    }

    if country not in patterns:
        return None

    config = patterns[country]
    match = re.match(config['pattern'], cleaned)

    if not match:
        return None

    return config['format'].format(*match.groups())


# ============================================================================
# EXERCISE 10: Markdown to HTML Converter (HARD)
# ============================================================================
# TODO: Basit markdown'ı HTML'e çevirin
#
# Gereksinimler:
# 1. Headers (# ## ###)
# 2. Bold (**text**) ve italic (*text*)
# 3. Links [text](url)
# 4. Code blocks (```code```)
# 5. Lists (- item veya 1. item)

def markdown_to_html(markdown: str) -> str:
    """
    Markdown'ı HTML'e çevirir.

    Örnek:
    >>> md = "# Title\\n**bold** and *italic*"
    >>> html = markdown_to_html(md)
    >>> '<h1>Title</h1>' in html
    True
    >>> '<strong>bold</strong>' in html
    True
    """
    # TODO: Implementation
    pass

# SOLUTION:
def markdown_to_html_solution(markdown: str) -> str:
    """Convert markdown to HTML."""
    html = markdown

    # Headers (# -> h1, ## -> h2, etc.)
    def replace_header(match):
        level = len(match.group(1))
        text = match.group(2)
        return f'<h{level}>{text}</h{level}>'

    html = re.sub(r'^(#{1,6})\s+(.+)$', replace_header, html, flags=re.MULTILINE)

    # Bold (**text**)
    html = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', html)

    # Italic (*text*)
    html = re.sub(r'\*(.+?)\*', r'<em>\1</em>', html)

    # Code blocks (```code```)
    html = re.sub(r'```(.+?)```', r'<pre><code>\1</code></pre>', html, flags=re.DOTALL)

    # Inline code (`code`)
    html = re.sub(r'`(.+?)`', r'<code>\1</code>', html)

    # Links [text](url)
    html = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'<a href="\2">\1</a>', html)

    # Unordered lists (- item)
    def replace_unordered_list(text):
        lines = text.split('\n')
        in_list = False
        result = []

        for line in lines:
            if re.match(r'^-\s+(.+)$', line):
                if not in_list:
                    result.append('<ul>')
                    in_list = True
                match = re.match(r'^-\s+(.+)$', line)
                result.append(f'<li>{match.group(1)}</li>')
            else:
                if in_list:
                    result.append('</ul>')
                    in_list = False
                result.append(line)

        if in_list:
            result.append('</ul>')

        return '\n'.join(result)

    html = replace_unordered_list(html)

    # Ordered lists (1. item)
    def replace_ordered_list(text):
        lines = text.split('\n')
        in_list = False
        result = []

        for line in lines:
            if re.match(r'^\d+\.\s+(.+)$', line):
                if not in_list:
                    result.append('<ol>')
                    in_list = True
                match = re.match(r'^\d+\.\s+(.+)$', line)
                result.append(f'<li>{match.group(1)}</li>')
            else:
                if in_list:
                    result.append('</ol>')
                    in_list = False
                result.append(line)

        if in_list:
            result.append('</ol>')

        return '\n'.join(result)

    html = replace_ordered_list(html)

    # Paragraphs (empty line separated)
    paragraphs = re.split(r'\n\s*\n', html)
    formatted_paragraphs = []

    for para in paragraphs:
        para = para.strip()
        if para and not para.startswith('<'):
            formatted_paragraphs.append(f'<p>{para}</p>')
        else:
            formatted_paragraphs.append(para)

    return '\n'.join(formatted_paragraphs)


# ============================================================================
# EXERCISE 11: Date/Time Parser (MEDIUM)
# ============================================================================
# TODO: Farklı formatlardaki tarihleri parse edin
#
# Gereksinimler:
# 1. Multiple format desteği (DD/MM/YYYY, MM-DD-YYYY, YYYY.MM.DD, etc.)
# 2. Ay isimlerini destekleyin (January, Jan, Ocak, etc.)
# 3. Relative dates (yesterday, last week, 3 days ago)
# 4. Time parsing (HH:MM:SS, HH:MM AM/PM)
# 5. datetime objesi döndürün

def parse_flexible_date(date_string: str) -> Optional[datetime]:
    """
    Farklı formatlardaki tarihleri parse eder.

    Örnek:
    >>> parse_flexible_date("15/01/2024")
    datetime.datetime(2024, 1, 15, 0, 0)
    >>> parse_flexible_date("January 15, 2024")
    datetime.datetime(2024, 1, 15, 0, 0)
    """
    # TODO: Implementation
    pass

# SOLUTION:
def parse_flexible_date_solution(date_string: str) -> Optional[datetime]:
    """Parse various date formats."""
    from datetime import timedelta

    # Date format patterns
    formats = [
        # DD/MM/YYYY variants
        (r'(\d{1,2})[/.-](\d{1,2})[/.-](\d{4})', '%d/%m/%Y'),
        # YYYY-MM-DD
        (r'(\d{4})[/.-](\d{1,2})[/.-](\d{1,2})', '%Y-%m-%d'),
        # Month DD, YYYY
        (r'(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2}),?\s+(\d{4})', '%B %d %Y'),
        # DD Month YYYY
        (r'(\d{1,2})\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{4})', '%d %B %Y'),
    ]

    # Try each format
    for pattern, date_format in formats:
        match = re.search(pattern, date_string, re.IGNORECASE)
        if match:
            try:
                matched_str = match.group(0)
                return datetime.strptime(matched_str, date_format)
            except ValueError:
                continue

    # Relative dates
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

    relative_patterns = {
        r'\btoday\b': timedelta(days=0),
        r'\byesterday\b': timedelta(days=-1),
        r'\btomorrow\b': timedelta(days=1),
        r'(\d+)\s+days?\s+ago': lambda m: timedelta(days=-int(m.group(1))),
        r'(\d+)\s+weeks?\s+ago': lambda m: timedelta(weeks=-int(m.group(1))),
    }

    for pattern, delta in relative_patterns.items():
        match = re.search(pattern, date_string, re.IGNORECASE)
        if match:
            if callable(delta):
                delta = delta(match)
            return today + delta

    return None


# ============================================================================
# EXERCISE 12: JSON-like Data Extractor (EXPERT)
# ============================================================================
# TODO: Text içindeki JSON-benzeri yapıları çıkarın
#
# Gereksinimler:
# 1. Nested object'leri handle edin
# 2. Array'leri parse edin
# 3. String, number, boolean, null değerleri destekleyin
# 4. Quoted key'leri ve unquoted key'leri destekleyin
# 5. Python dict'e çevirin

def extract_json_like_data(text: str) -> List[Dict]:
    """
    Text'ten JSON-benzeri yapıları çıkarır.

    Örnek:
    >>> text = 'User: {name: "John", age: 30, active: true}'
    >>> result = extract_json_like_data(text)
    >>> result[0]['name']
    'John'
    """
    # TODO: Implementation
    pass

# SOLUTION:
def extract_json_like_data_solution(text: str) -> List[Dict]:
    """Extract JSON-like structures from text."""
    # Basit JSON pattern (nested olmayan)
    json_pattern = re.compile(
        r'\{([^{}]*)\}',
        re.DOTALL
    )

    results = []

    for match in json_pattern.finditer(text):
        content = match.group(1)

        # Key-value çiftlerini parse et
        # Hem "key": value hem de key: value formatını destekle
        kv_pattern = re.compile(
            r'(?:"([^"]+)"|(\w+))\s*:\s*'
            r'(?:'
            r'"([^"]*)"'  # String value
            r'|(\d+(?:\.\d+)?)'  # Number value
            r'|(true|false)'  # Boolean value
            r'|(null)'  # Null value
            r')'
        )

        data = {}
        for kv_match in kv_pattern.finditer(content):
            # Key (quoted veya unquoted)
            key = kv_match.group(1) or kv_match.group(2)

            # Value (string, number, boolean veya null)
            if kv_match.group(3) is not None:  # String
                value = kv_match.group(3)
            elif kv_match.group(4) is not None:  # Number
                num_str = kv_match.group(4)
                value = float(num_str) if '.' in num_str else int(num_str)
            elif kv_match.group(5) is not None:  # Boolean
                value = kv_match.group(5).lower() == 'true'
            elif kv_match.group(6) is not None:  # Null
                value = None
            else:
                continue

            data[key] = value

        if data:
            results.append(data)

    return results


# ============================================================================
# EXERCISE 13: Code Comment Extractor (HARD)
# ============================================================================
# TODO: Kaynak koddan yorumları çıkarın (multiple languages)
#
# Gereksinimler:
# 1. Single-line comments (//, #)
# 2. Multi-line comments (/* */, ''' ''', """ """)
# 3. Docstrings'i ayırın
# 4. Nested comments'leri handle edin
# 5. String içindeki comment-like yapıları ignore edin

def extract_code_comments(code: str, language: str = 'python') -> Dict[str, List[str]]:
    """
    Kaynak koddan yorumları çıkarır.

    Örnek:
    >>> code = '# Comment\\nprint("test")  # Inline comment'
    >>> result = extract_code_comments(code, 'python')
    >>> len(result['single_line']) >= 2
    True
    """
    # TODO: Implementation
    pass

# SOLUTION:
def extract_code_comments_solution(code: str, language: str = 'python') -> Dict[str, List[str]]:
    """Extract comments from source code."""
    comments = {
        'single_line': [],
        'multi_line': [],
        'docstrings': []
    }

    if language == 'python':
        # Single-line comments (#)
        single_pattern = re.compile(r'#(.+)$', re.MULTILINE)
        for match in single_pattern.finditer(code):
            comments['single_line'].append(match.group(1).strip())

        # Multi-line strings/docstrings (''' or """)
        multi_pattern = re.compile(r'(["\'])\1\1(.+?)\1\1\1', re.DOTALL)
        for match in multi_pattern.finditer(code):
            content = match.group(2).strip()
            # Docstring ise (fonksiyon/class tanımından sonra geliyorsa)
            before = code[:match.start()]
            if re.search(r'(def|class)\s+\w+.*:\s*$', before, re.MULTILINE):
                comments['docstrings'].append(content)
            else:
                comments['multi_line'].append(content)

    elif language in ['javascript', 'java', 'c', 'cpp']:
        # Single-line comments (//)
        single_pattern = re.compile(r'//(.+)$', re.MULTILINE)
        for match in single_pattern.finditer(code):
            comments['single_line'].append(match.group(1).strip())

        # Multi-line comments (/* */)
        multi_pattern = re.compile(r'/\*(.+?)\*/', re.DOTALL)
        for match in multi_pattern.finditer(code):
            comments['multi_line'].append(match.group(1).strip())

    return comments


# ============================================================================
# EXERCISE 14: File Path Validator and Parser (MEDIUM)
# ============================================================================
# TODO: Dosya yollarını validate edin ve parse edin
#
# Gereksinimler:
# 1. Windows ve Unix path'lerini destekleyin
# 2. Relative ve absolute path'leri ayırt edin
# 3. Extension, filename, directory'yi ayırın
# 4. Invalid karakterleri tespit edin
# 5. Path normalization yapın

def parse_file_path(path: str) -> Optional[Dict]:
    """
    Dosya yolunu parse eder.

    Örnek:
    >>> result = parse_file_path("/home/user/documents/file.txt")
    >>> result['filename']
    'file.txt'
    >>> result['extension']
    'txt'
    """
    # TODO: Implementation
    pass

# SOLUTION:
def parse_file_path_solution(path: str) -> Optional[Dict]:
    """Parse and validate file paths."""
    # Windows vs Unix detection
    is_windows = '\\' in path or re.match(r'^[A-Za-z]:', path)

    if is_windows:
        # Windows path pattern
        pattern = re.compile(
            r'^(?P<drive>[A-Za-z]:)?'
            r'(?P<path>(?:[\\\/]?[\w\s.-]+)*)'
            r'[\\\/]?(?P<filename>[\w\s.-]+?(?:\.(?P<ext>\w+))?)$'
        )
        separator = '\\'
    else:
        # Unix path pattern
        pattern = re.compile(
            r'^(?P<path>(?:\/[\w\s.-]+)*)'
            r'\/?(?P<filename>[\w\s.-]+?(?:\.(?P<ext>\w+))?)$'
        )
        separator = '/'

    match = pattern.match(path)
    if not match:
        return None

    result = match.groupdict()

    # Absolute vs relative
    if is_windows:
        result['is_absolute'] = bool(result.get('drive')) or path.startswith('\\\\')
    else:
        result['is_absolute'] = path.startswith('/')

    result['is_relative'] = not result['is_absolute']
    result['separator'] = separator
    result['is_windows'] = is_windows

    # Directory path
    if result['path']:
        result['directory'] = result['path'].strip(separator)
    else:
        result['directory'] = None

    # Invalid character check
    invalid_chars = r'[<>:"|?*]' if is_windows else r'[\0]'
    result['has_invalid_chars'] = bool(re.search(invalid_chars, path))

    return result


# ============================================================================
# EXERCISE 15: Performance: Compiled Regex Benchmark (ADVANCED)
# ============================================================================
# TODO: Compiled vs uncompiled regex performans karşılaştırması
#
# Gereksinimler:
# 1. Email extraction için her iki yöntemi test edin
# 2. 1000+ email içeren text'te test yapın
# 3. Execution time'ı ölçün
# 4. Memory usage farkını hesaplayın
# 5. Speedup faktörünü raporlayın

def benchmark_regex_compilation() -> Dict:
    """
    Compiled vs uncompiled regex performansını karşılaştırır.

    Returns:
        Dict with timing and performance metrics
    """
    # TODO: Implementation
    pass

# SOLUTION:
def benchmark_regex_compilation_solution() -> Dict:
    """Benchmark compiled vs uncompiled regex."""
    import time

    # Test data oluştur
    test_emails = [f"user{i}@example{i%100}.com" for i in range(1000)]
    test_text = ", ".join(test_emails) * 10  # 10,000 emails

    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'

    # Uncompiled version
    start = time.perf_counter()
    for _ in range(100):
        re.findall(email_pattern, test_text)
    uncompiled_time = time.perf_counter() - start

    # Compiled version
    compiled_pattern = re.compile(email_pattern)
    start = time.perf_counter()
    for _ in range(100):
        compiled_pattern.findall(test_text)
    compiled_time = time.perf_counter() - start

    return {
        'uncompiled_time': round(uncompiled_time, 4),
        'compiled_time': round(compiled_time, 4),
        'speedup': round(uncompiled_time / compiled_time, 2),
        'time_saved': round(uncompiled_time - compiled_time, 4),
        'improvement_percent': round(
            ((uncompiled_time - compiled_time) / uncompiled_time) * 100, 2
        )
    }


# ============================================================================
# TEST SUITE
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("REGEX ADVANCED - EXERCISE TESTS")
    print("=" * 80)

    # Test 1: Log Parser
    print("\n1. Advanced Log Parser:")
    log = '[2024-01-15 10:30:45.123] [ERROR] [auth:login:42] Login failed {"user_id": 123, "ip": "192.168.1.1"}'
    result = parse_advanced_log_solution(log)
    print(f"   Level: {result['level']}")
    print(f"   Module: {result['module']}")
    print(f"   Context: {result['context']}")

    # Test 2: Email Categorization
    print("\n2. Email Categorization:")
    text = "Contact: admin@company.com, info@university.edu, test@org.org"
    emails = extract_and_categorize_emails_solution(text)
    print(f"   Categories: {list(emails.keys())}")
    print(f"   .com emails: {emails.get('com', [])}")

    # Test 3: URL Parser
    print("\n3. URL Parser:")
    url = "https://api.example.com:8080/v1/users?active=true&limit=10#section"
    url_data = parse_url_solution(url)
    print(f"   Domain: {url_data['domain']}")
    print(f"   Port: {url_data['port']}")
    print(f"   Query: {url_data['query']}")

    # Test 4: HTML Sanitizer
    print("\n4. HTML Sanitizer:")
    html = '<p>Safe</p><script>alert("XSS")</script><div onclick="bad()">Text</div>'
    clean = sanitize_html_solution(html, allowed_tags=['p', 'div'])
    print(f"   Has script: {'<script>' in clean}")
    print(f"   Has onclick: {'onclick' in clean}")
    print(f"   Cleaned: {clean[:50]}...")

    # Test 5: Credit Card Validator
    print("\n5. Credit Card Validator:")
    card = "4532-1234-5678-9010"
    card_info = validate_credit_card_solution(card)
    if card_info:
        print(f"   Type: {card_info['type']}")
        print(f"   Valid: {card_info['valid']}")
        print(f"   Masked: {card_info['masked']}")

    # Test 6: Password Strength
    print("\n6. Password Strength:")
    pwd = "Str0ng!P@ssw0rd#2024"
    strength = analyze_password_strength_solution(pwd)
    print(f"   Score: {strength['score']}/100")
    print(f"   Strength: {strength['strength']}")

    # Test 7: IP Extractor
    print("\n7. IP Address Extractor:")
    ip_text = "Server: 192.168.1.1/24, Public: 8.8.8.8"
    ips = extract_ip_addresses_solution(ip_text)
    print(f"   Private: {ips['ipv4_private']}")
    print(f"   Public: {ips['ipv4_public']}")

    # Test 8: SQL Parser
    print("\n8. SQL Query Parser:")
    sql = "SELECT id, name FROM users WHERE active = 1 ORDER BY created_at"
    sql_data = parse_sql_query_solution(sql)
    if sql_data:
        print(f"   Type: {sql_data['type']}")
        print(f"   Tables: {sql_data['tables']}")
        print(f"   Columns: {sql_data['columns']}")

    # Test 9: Phone Formatter
    print("\n9. Phone Number Formatter:")
    formatted = format_phone_number_solution("05321234567", "TR")
    print(f"   Formatted: {formatted}")

    # Test 10: Markdown Converter
    print("\n10. Markdown to HTML:")
    md = "# Title\n**bold** and *italic*"
    html = markdown_to_html_solution(md)
    print(f"    Has h1: {'<h1>' in html}")
    print(f"    Has strong: {'<strong>' in html}")

    # Test 11: Date Parser
    print("\n11. Flexible Date Parser:")
    date = parse_flexible_date_solution("15/01/2024")
    if date:
        print(f"    Parsed: {date.strftime('%Y-%m-%d')}")

    # Test 12: JSON Extractor
    print("\n12. JSON-like Data Extractor:")
    json_text = 'User: {name: "John", age: 30, active: true}'
    json_data = extract_json_like_data_solution(json_text)
    if json_data:
        print(f"    Data: {json_data[0]}")

    # Test 13: Comment Extractor
    print("\n13. Code Comment Extractor:")
    code = '# Comment\nprint("test")  # Inline'
    comments = extract_code_comments_solution(code, 'python')
    print(f"    Single-line: {len(comments['single_line'])} comments")

    # Test 14: Path Parser
    print("\n14. File Path Parser:")
    path_info = parse_file_path_solution("/home/user/file.txt")
    if path_info:
        print(f"    Filename: {path_info['filename']}")
        print(f"    Extension: {path_info['ext']}")

    # Test 15: Performance Benchmark
    print("\n15. Regex Compilation Benchmark:")
    perf = benchmark_regex_compilation_solution()
    print(f"    Speedup: {perf['speedup']}x")
    print(f"    Improvement: {perf['improvement_percent']}%")

    print("\n" + "=" * 80)
    print("Tüm testler tamamlandı!")
    print("=" * 80)
