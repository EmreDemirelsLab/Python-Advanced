# Advanced Python Networking & Socket Programming

## İçindekiler
1. [Socket Programming Temelleri](#socket-programming-temelleri)
2. [TCP Socket Programming](#tcp-socket-programming)
3. [UDP Socket Programming](#udp-socket-programming)
4. [Client-Server Architecture](#client-server-architecture)
5. [Non-blocking Sockets](#non-blocking-sockets)
6. [Multiplexing: select, poll, epoll](#multiplexing-select-poll-epoll)
7. [Asyncio ile Network Programming](#asyncio-ile-network-programming)
8. [HTTP Requests (Advanced)](#http-requests-advanced)
9. [WebSocket Programming](#websocket-programming)
10. [SSL/TLS Güvenliği](#ssltls-güvenliği)
11. [Network Protocols](#network-protocols)
12. [Error Handling & Best Practices](#error-handling-best-practices)

---

## Socket Programming Temelleri

Socket'ler, ağ üzerinden iletişim kurmak için kullanılan düşük seviyeli API'lerdir. Python'da `socket` modülü ile TCP/UDP protokolleri üzerinden veri alışverişi yapabiliriz.

### Örnek 1: Basit TCP Socket Oluşturma

```python
import socket
import sys

def create_tcp_socket():
    """
    TCP socket oluşturma ve temel ayarlar
    Production'da socket options dikkatle ayarlanmalı
    """
    try:
        # AF_INET: IPv4, SOCK_STREAM: TCP
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        # Socket seçenekleri
        # SO_REUSEADDR: Aynı portu tekrar kullanmaya izin ver
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        # TCP_NODELAY: Nagle algoritmasını devre dışı bırak (düşük latency için)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

        # Timeout ayarı (production'da önemli)
        sock.settimeout(30.0)  # 30 saniye

        # Buffer boyutları
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 65536)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 65536)

        print(f"Socket oluşturuldu: {sock}")
        print(f"Socket ailesi: {sock.family}")
        print(f"Socket tipi: {sock.type}")
        print(f"Socket protokolü: {sock.proto}")

        return sock

    except socket.error as e:
        print(f"Socket oluşturma hatası: {e}", file=sys.stderr)
        return None
    finally:
        if 'sock' in locals():
            sock.close()

# Kullanım
sock = create_tcp_socket()
```

### Örnek 2: Socket Address Çözümleme

```python
import socket
from typing import List, Tuple

def resolve_hostname(hostname: str, port: int = 80) -> List[Tuple]:
    """
    Hostname'i IP adreslerine çözümle (DNS lookup)
    IPv4 ve IPv6 adreslerini destekler
    """
    try:
        # getaddrinfo: Hostname'i çözümle, tüm olası adresleri getir
        results = socket.getaddrinfo(
            hostname,
            port,
            socket.AF_UNSPEC,  # IPv4 veya IPv6
            socket.SOCK_STREAM  # TCP
        )

        print(f"DNS Çözümleme: {hostname}")
        print("-" * 50)

        addresses = []
        for family, socktype, proto, canonname, sockaddr in results:
            family_name = "IPv4" if family == socket.AF_INET else "IPv6"
            addresses.append((family_name, sockaddr[0], sockaddr[1]))
            print(f"Aile: {family_name}, Adres: {sockaddr[0]}, Port: {sockaddr[1]}")

        return addresses

    except socket.gaierror as e:
        print(f"DNS çözümleme hatası: {e}")
        return []

def get_local_info():
    """Local makine bilgilerini al"""
    hostname = socket.gethostname()
    fqdn = socket.getfqdn()
    local_ip = socket.gethostbyname(hostname)

    print(f"Hostname: {hostname}")
    print(f"FQDN: {fqdn}")
    print(f"Local IP: {local_ip}")

# Kullanım
resolve_hostname("www.google.com", 443)
resolve_hostname("www.python.org", 80)
get_local_info()
```

---

## TCP Socket Programming

TCP (Transmission Control Protocol), bağlantı odaklı, güvenilir bir protokoldür. Veri sırası korunur ve kayıp yaşanmaz.

### Örnek 3: Production-Ready TCP Server

```python
import socket
import threading
import logging
from typing import Optional
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class TCPServer:
    """
    Production-ready TCP server
    - Multi-threaded client handling
    - Graceful shutdown
    - Error handling
    - Connection pooling
    """

    def __init__(self, host: str = '0.0.0.0', port: int = 9000,
                 max_connections: int = 5):
        self.host = host
        self.port = port
        self.max_connections = max_connections
        self.server_socket: Optional[socket.socket] = None
        self.running = False
        self.clients = []

    def start(self):
        """Server'ı başlat"""
        try:
            # Socket oluştur
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

            # Bind ve listen
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(self.max_connections)

            self.running = True
            logging.info(f"Server başlatıldı: {self.host}:{self.port}")
            logging.info(f"Maksimum bağlantı: {self.max_connections}")

            # Client bekleme döngüsü
            while self.running:
                try:
                    # Accept client connection
                    self.server_socket.settimeout(1.0)  # Shutdown kontrolü için
                    client_socket, address = self.server_socket.accept()

                    logging.info(f"Yeni bağlantı: {address}")

                    # Her client için thread oluştur
                    client_thread = threading.Thread(
                        target=self.handle_client,
                        args=(client_socket, address),
                        daemon=True
                    )
                    client_thread.start()
                    self.clients.append(client_thread)

                except socket.timeout:
                    continue
                except Exception as e:
                    if self.running:
                        logging.error(f"Accept hatası: {e}")

        except Exception as e:
            logging.error(f"Server başlatma hatası: {e}")
        finally:
            self.shutdown()

    def handle_client(self, client_socket: socket.socket, address: tuple):
        """Client ile iletişimi yönet"""
        try:
            client_socket.settimeout(60.0)  # 60 saniye timeout

            # Welcome mesajı gönder
            welcome = f"Hoşgeldiniz! Bağlantı zamanı: {datetime.now()}\n"
            client_socket.sendall(welcome.encode('utf-8'))

            # Echo server mantığı
            while self.running:
                try:
                    # Veri al (max 4096 byte)
                    data = client_socket.recv(4096)

                    if not data:
                        # Client bağlantıyı kapattı
                        logging.info(f"Client ayrıldı: {address}")
                        break

                    # Veriyi işle
                    message = data.decode('utf-8').strip()
                    logging.info(f"[{address}] Mesaj: {message}")

                    # Komutları işle
                    if message.lower() == 'quit':
                        response = "Güle güle!\n"
                        client_socket.sendall(response.encode('utf-8'))
                        break
                    elif message.lower() == 'time':
                        response = f"Sunucu zamanı: {datetime.now()}\n"
                        client_socket.sendall(response.encode('utf-8'))
                    else:
                        # Echo geri gönder
                        response = f"Echo: {message}\n"
                        client_socket.sendall(response.encode('utf-8'))

                except socket.timeout:
                    logging.warning(f"Client timeout: {address}")
                    break
                except Exception as e:
                    logging.error(f"Client iletişim hatası: {e}")
                    break

        finally:
            client_socket.close()
            logging.info(f"Bağlantı kapatıldı: {address}")

    def shutdown(self):
        """Server'ı güvenli şekilde kapat"""
        logging.info("Server kapatılıyor...")
        self.running = False

        if self.server_socket:
            self.server_socket.close()

        # Tüm client thread'lerini bekle
        for client in self.clients:
            if client.is_alive():
                client.join(timeout=2.0)

        logging.info("Server kapatıldı")

# Kullanım
if __name__ == '__main__':
    server = TCPServer(host='127.0.0.1', port=9000)
    try:
        server.start()
    except KeyboardInterrupt:
        logging.info("Keyboard interrupt alındı")
        server.shutdown()
```

### Örnek 4: Production-Ready TCP Client

```python
import socket
import logging
from typing import Optional

class TCPClient:
    """
    Production-ready TCP client
    - Automatic reconnection
    - Timeout handling
    - Buffered I/O
    """

    def __init__(self, host: str, port: int, timeout: float = 10.0):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.socket: Optional[socket.socket] = None
        self.connected = False

    def connect(self) -> bool:
        """Server'a bağlan"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(self.timeout)

            logging.info(f"Bağlanılıyor: {self.host}:{self.port}")
            self.socket.connect((self.host, self.port))

            self.connected = True
            logging.info("Bağlantı başarılı")
            return True

        except socket.timeout:
            logging.error("Bağlantı timeout!")
            return False
        except ConnectionRefusedError:
            logging.error("Bağlantı reddedildi!")
            return False
        except Exception as e:
            logging.error(f"Bağlantı hatası: {e}")
            return False

    def send(self, message: str) -> bool:
        """Mesaj gönder"""
        if not self.connected or not self.socket:
            logging.error("Bağlantı yok!")
            return False

        try:
            data = message.encode('utf-8')
            self.socket.sendall(data)
            return True
        except Exception as e:
            logging.error(f"Gönderme hatası: {e}")
            self.connected = False
            return False

    def receive(self, buffer_size: int = 4096) -> Optional[str]:
        """Veri al"""
        if not self.connected or not self.socket:
            logging.error("Bağlantı yok!")
            return None

        try:
            data = self.socket.recv(buffer_size)
            if not data:
                logging.warning("Bağlantı karşı tarafça kapatıldı")
                self.connected = False
                return None

            return data.decode('utf-8')

        except socket.timeout:
            logging.warning("Alma timeout!")
            return None
        except Exception as e:
            logging.error(f"Alma hatası: {e}")
            self.connected = False
            return None

    def close(self):
        """Bağlantıyı kapat"""
        if self.socket:
            self.socket.close()
            self.connected = False
            logging.info("Bağlantı kapatıldı")

# Kullanım
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    client = TCPClient('127.0.0.1', 9000)

    if client.connect():
        # Welcome mesajını al
        response = client.receive()
        print(f"Server: {response}")

        # Mesajlar gönder
        messages = ['Hello Server', 'time', 'Test message', 'quit']

        for msg in messages:
            print(f"\nGönderiliyor: {msg}")
            client.send(msg + '\n')

            response = client.receive()
            print(f"Cevap: {response}")

        client.close()
```

---

## UDP Socket Programming

UDP (User Datagram Protocol), bağlantısız, hızlı ama güvenilir olmayan bir protokoldür. Streaming, gaming, DNS gibi uygulamalarda kullanılır.

### Örnek 5: UDP Server ve Client

```python
import socket
import threading
import logging
from datetime import datetime

class UDPServer:
    """
    UDP server - connectionless protocol
    Her datagram bağımsızdır
    """

    def __init__(self, host: str = '0.0.0.0', port: int = 9001):
        self.host = host
        self.port = port
        self.socket = None
        self.running = False

    def start(self):
        """Server'ı başlat"""
        try:
            # UDP socket oluştur (SOCK_DGRAM)
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

            # Bind
            self.socket.bind((self.host, self.port))
            self.running = True

            logging.info(f"UDP Server başlatıldı: {self.host}:{self.port}")

            # Datagram alma döngüsü
            while self.running:
                try:
                    # recvfrom: veri + sender adresi
                    data, address = self.socket.recvfrom(4096)

                    message = data.decode('utf-8').strip()
                    logging.info(f"[{address}] Mesaj: {message}")

                    # Response hazırla
                    response = f"Echo: {message} (Zaman: {datetime.now()})"

                    # Geri gönder (sendto: address gerekir)
                    self.socket.sendto(response.encode('utf-8'), address)

                except Exception as e:
                    if self.running:
                        logging.error(f"Hata: {e}")

        except Exception as e:
            logging.error(f"Server başlatma hatası: {e}")
        finally:
            if self.socket:
                self.socket.close()
            logging.info("UDP Server kapatıldı")

class UDPClient:
    """UDP client"""

    def __init__(self, host: str, port: int, timeout: float = 5.0):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.socket = None

    def send(self, message: str) -> Optional[str]:
        """Mesaj gönder ve cevap al"""
        try:
            # UDP socket oluştur
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.socket.settimeout(self.timeout)

            # Gönder (connect gerekmez)
            self.socket.sendto(message.encode('utf-8'), (self.host, self.port))

            # Cevap bekle
            data, server = self.socket.recvfrom(4096)
            return data.decode('utf-8')

        except socket.timeout:
            logging.error("Timeout!")
            return None
        except Exception as e:
            logging.error(f"Hata: {e}")
            return None
        finally:
            if self.socket:
                self.socket.close()

# Kullanım
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    # Server'ı thread'de başlat
    server = UDPServer('127.0.0.1', 9001)
    server_thread = threading.Thread(target=server.start, daemon=True)
    server_thread.start()

    import time
    time.sleep(1)  # Server'ın başlamasını bekle

    # Client ile mesaj gönder
    client = UDPClient('127.0.0.1', 9001)

    messages = ['Hello UDP', 'Test message', 'Goodbye']
    for msg in messages:
        print(f"\nGönderiliyor: {msg}")
        response = client.send(msg)
        print(f"Cevap: {response}")
```

---

## Non-blocking Sockets

Non-blocking socket'ler, I/O işlemlerinin thread'i bloke etmemesini sağlar. Bu sayede tek thread ile birçok connection yönetilebilir.

### Örnek 6: Non-blocking TCP Server

```python
import socket
import errno
import logging
from typing import Dict, Set

class NonBlockingServer:
    """
    Non-blocking socket server
    Tek thread ile birçok client'ı yönetir
    """

    def __init__(self, host: str = '127.0.0.1', port: int = 9002):
        self.host = host
        self.port = port
        self.server_socket = None
        self.clients: Dict[socket.socket, str] = {}
        self.running = False

    def start(self):
        """Server'ı başlat"""
        try:
            # Socket oluştur
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

            # Non-blocking yap
            self.server_socket.setblocking(False)

            # Bind ve listen
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(5)

            self.running = True
            logging.info(f"Non-blocking server başlatıldı: {self.host}:{self.port}")

            # Ana döngü
            while self.running:
                # Yeni bağlantıları kabul et
                self._accept_new_clients()

                # Mevcut client'lardan veri al
                self._handle_clients()

        except Exception as e:
            logging.error(f"Server hatası: {e}")
        finally:
            self.shutdown()

    def _accept_new_clients(self):
        """Yeni client'ları kabul et"""
        try:
            while True:
                try:
                    client_socket, address = self.server_socket.accept()
                    client_socket.setblocking(False)

                    self.clients[client_socket] = f"{address[0]}:{address[1]}"
                    logging.info(f"Yeni client: {self.clients[client_socket]}")

                    # Welcome mesajı
                    welcome = b"Welcome to non-blocking server!\n"
                    client_socket.send(welcome)

                except BlockingIOError:
                    # Yeni bağlantı yok, devam et
                    break

        except Exception as e:
            logging.error(f"Accept hatası: {e}")

    def _handle_clients(self):
        """Mevcut client'ları işle"""
        clients_to_remove = []

        for client_socket, client_info in list(self.clients.items()):
            try:
                # Veri almayı dene
                data = client_socket.recv(4096)

                if not data:
                    # Client bağlantıyı kapattı
                    logging.info(f"Client ayrıldı: {client_info}")
                    clients_to_remove.append(client_socket)
                    continue

                message = data.decode('utf-8').strip()
                logging.info(f"[{client_info}] Mesaj: {message}")

                # Echo gönder
                response = f"Echo: {message}\n".encode('utf-8')
                client_socket.send(response)

            except BlockingIOError:
                # Veri yok, devam et
                continue
            except Exception as e:
                logging.error(f"Client hatası [{client_info}]: {e}")
                clients_to_remove.append(client_socket)

        # Kopan client'ları temizle
        for client_socket in clients_to_remove:
            self._remove_client(client_socket)

    def _remove_client(self, client_socket: socket.socket):
        """Client'ı kaldır"""
        if client_socket in self.clients:
            del self.clients[client_socket]
        client_socket.close()

    def shutdown(self):
        """Server'ı kapat"""
        logging.info("Server kapatılıyor...")
        self.running = False

        # Tüm client'ları kapat
        for client_socket in list(self.clients.keys()):
            self._remove_client(client_socket)

        # Server socket'i kapat
        if self.server_socket:
            self.server_socket.close()

        logging.info("Server kapatıldı")

# Kullanım
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    server = NonBlockingServer()
    try:
        server.start()
    except KeyboardInterrupt:
        server.shutdown()
```

---

## Multiplexing: select, poll, epoll

`select`, `poll` ve `epoll` ile birden fazla socket'i eşzamanlı olarak izleyebiliriz. Bu, performanslı network uygulamaları için kritiktir.

### Örnek 7: Select ile Multiplexing

```python
import socket
import select
import logging
from typing import List

class SelectServer:
    """
    select() kullanarak multiplexing
    Birden fazla socket'i aynı anda izler
    """

    def __init__(self, host: str = '127.0.0.1', port: int = 9003):
        self.host = host
        self.port = port
        self.server_socket = None
        self.sockets_list: List[socket.socket] = []
        self.clients = {}

    def start(self):
        """Server'ı başlat"""
        try:
            # Server socket oluştur
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(5)

            # Socket listesine ekle
            self.sockets_list.append(self.server_socket)

            logging.info(f"Select server başlatıldı: {self.host}:{self.port}")

            while True:
                # select: hazır olan socket'leri bekle
                # timeout=None: süresiz bekle
                read_sockets, _, exception_sockets = select.select(
                    self.sockets_list,  # Okumak için izlenenler
                    [],                 # Yazmak için izlenenler
                    self.sockets_list,  # Exception için izlenenler
                    1.0                 # Timeout (saniye)
                )

                # Okunabilir socket'leri işle
                for notified_socket in read_sockets:
                    # Yeni bağlantı
                    if notified_socket == self.server_socket:
                        self._accept_client()

                    # Mevcut client'tan veri
                    else:
                        self._handle_client_data(notified_socket)

                # Exception olan socket'leri işle
                for notified_socket in exception_sockets:
                    self._remove_client(notified_socket)

        except KeyboardInterrupt:
            logging.info("Keyboard interrupt")
        except Exception as e:
            logging.error(f"Server hatası: {e}")
        finally:
            self.shutdown()

    def _accept_client(self):
        """Yeni client kabul et"""
        try:
            client_socket, address = self.server_socket.accept()

            self.sockets_list.append(client_socket)
            self.clients[client_socket] = f"{address[0]}:{address[1]}"

            logging.info(f"Yeni bağlantı: {self.clients[client_socket]}")

            welcome = b"Welcome to select server!\n"
            client_socket.send(welcome)

        except Exception as e:
            logging.error(f"Accept hatası: {e}")

    def _handle_client_data(self, client_socket: socket.socket):
        """Client'tan gelen veriyi işle"""
        try:
            data = client_socket.recv(4096)

            if not data:
                # Client bağlantıyı kapattı
                logging.info(f"Client ayrıldı: {self.clients[client_socket]}")
                self._remove_client(client_socket)
                return

            message = data.decode('utf-8').strip()
            client_info = self.clients[client_socket]

            logging.info(f"[{client_info}] Mesaj: {message}")

            # Tüm client'lara broadcast (chat server gibi)
            broadcast_msg = f"[{client_info}]: {message}\n".encode('utf-8')

            for sock in self.sockets_list:
                if sock != self.server_socket and sock != client_socket:
                    try:
                        sock.send(broadcast_msg)
                    except:
                        self._remove_client(sock)

            # Echo gönderen client'a
            echo = f"Echo: {message}\n".encode('utf-8')
            client_socket.send(echo)

        except Exception as e:
            logging.error(f"Veri işleme hatası: {e}")
            self._remove_client(client_socket)

    def _remove_client(self, client_socket: socket.socket):
        """Client'ı kaldır"""
        if client_socket in self.sockets_list:
            self.sockets_list.remove(client_socket)

        if client_socket in self.clients:
            del self.clients[client_socket]

        try:
            client_socket.close()
        except:
            pass

    def shutdown(self):
        """Server'ı kapat"""
        logging.info("Server kapatılıyor...")

        for sock in self.sockets_list:
            try:
                sock.close()
            except:
                pass

        logging.info("Server kapatıldı")

# Kullanım
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    server = SelectServer()
    server.start()
```

### Örnek 8: Selectors Modülü (High-level API)

```python
import socket
import selectors
import logging
from typing import Optional

class SelectorsServer:
    """
    selectors modülü ile modern multiplexing
    Platform'a göre en iyi selector'ı otomatik seçer
    (Linux: epoll, Windows: select, BSD: kqueue)
    """

    def __init__(self, host: str = '127.0.0.1', port: int = 9004):
        self.host = host
        self.port = port
        self.selector = selectors.DefaultSelector()
        self.server_socket = None

    def start(self):
        """Server'ı başlat"""
        try:
            # Server socket
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(5)

            # Non-blocking yap
            self.server_socket.setblocking(False)

            # Selector'a kaydet (READ eventi için)
            self.selector.register(
                self.server_socket,
                selectors.EVENT_READ,
                data=None  # Server socket için data yok
            )

            logging.info(f"Selectors server başlatıldı: {self.host}:{self.port}")
            logging.info(f"Selector tipi: {type(self.selector).__name__}")

            # Event loop
            while True:
                # Hazır olan event'leri al (timeout: 1 saniye)
                events = self.selector.select(timeout=1.0)

                for key, mask in events:
                    if key.data is None:
                        # Server socket: yeni bağlantı
                        self._accept_client()
                    else:
                        # Client socket: veri geldi
                        self._handle_client(key, mask)

        except KeyboardInterrupt:
            logging.info("Keyboard interrupt")
        except Exception as e:
            logging.error(f"Server hatası: {e}")
        finally:
            self.shutdown()

    def _accept_client(self):
        """Yeni client kabul et"""
        try:
            client_socket, address = self.server_socket.accept()
            client_socket.setblocking(False)

            logging.info(f"Yeni bağlantı: {address}")

            # Client bilgilerini data olarak sakla
            client_data = {
                'address': address,
                'inb': b'',  # Incoming buffer
                'outb': b''  # Outgoing buffer
            }

            # Selector'a kaydet (READ eventi)
            self.selector.register(
                client_socket,
                selectors.EVENT_READ | selectors.EVENT_WRITE,
                data=client_data
            )

            # Welcome mesajı
            welcome = b"Welcome to selectors server!\n"
            client_data['outb'] += welcome

        except Exception as e:
            logging.error(f"Accept hatası: {e}")

    def _handle_client(self, key, mask):
        """Client event'ini işle"""
        client_socket = key.fileobj
        client_data = key.data

        try:
            # READ event: veri geldi
            if mask & selectors.EVENT_READ:
                recv_data = client_socket.recv(4096)

                if recv_data:
                    # Veriyi buffer'a ekle
                    client_data['inb'] += recv_data

                    # Satır satır işle
                    while b'\n' in client_data['inb']:
                        line, client_data['inb'] = client_data['inb'].split(b'\n', 1)
                        message = line.decode('utf-8').strip()

                        logging.info(f"[{client_data['address']}] Mesaj: {message}")

                        # Response hazırla
                        response = f"Echo: {message}\n".encode('utf-8')
                        client_data['outb'] += response
                else:
                    # Client bağlantıyı kapattı
                    logging.info(f"Client ayrıldı: {client_data['address']}")
                    self._remove_client(client_socket)
                    return

            # WRITE event: yazmaya hazır
            if mask & selectors.EVENT_WRITE:
                if client_data['outb']:
                    # Gönder
                    sent = client_socket.send(client_data['outb'])
                    client_data['outb'] = client_data['outb'][sent:]

        except Exception as e:
            logging.error(f"Client işleme hatası: {e}")
            self._remove_client(client_socket)

    def _remove_client(self, client_socket: socket.socket):
        """Client'ı kaldır"""
        try:
            self.selector.unregister(client_socket)
            client_socket.close()
        except Exception as e:
            logging.error(f"Client kaldırma hatası: {e}")

    def shutdown(self):
        """Server'ı kapat"""
        logging.info("Server kapatılıyor...")

        try:
            self.selector.close()
        except:
            pass

        if self.server_socket:
            self.server_socket.close()

        logging.info("Server kapatıldı")

# Kullanım
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    server = SelectorsServer()
    server.start()
```

---

## Asyncio ile Network Programming

Asyncio, modern Python'da asenkron network programlama için en iyi yöntemdir. Coroutine'ler ile yüksek performanslı server'lar yazabiliriz.

### Örnek 9: Asyncio TCP Server

```python
import asyncio
import logging
from datetime import datetime

class AsyncioTCPServer:
    """
    Asyncio ile modern TCP server
    Coroutine'ler ile binlerce bağlantı yönetilebilir
    """

    def __init__(self, host: str = '127.0.0.1', port: int = 9005):
        self.host = host
        self.port = port
        self.clients = {}

    async def handle_client(self, reader: asyncio.StreamReader,
                           writer: asyncio.StreamWriter):
        """Client ile iletişimi yönet (coroutine)"""
        address = writer.get_extra_info('peername')
        client_id = f"{address[0]}:{address[1]}"

        self.clients[client_id] = writer
        logging.info(f"Yeni bağlantı: {client_id}")

        try:
            # Welcome mesajı
            welcome = f"Hoşgeldiniz! Zaman: {datetime.now()}\n"
            writer.write(welcome.encode('utf-8'))
            await writer.drain()  # Göndermeyi bekle

            # Client loop
            while True:
                # Veri al (await ile asenkron)
                data = await reader.read(4096)

                if not data:
                    # Bağlantı kapandı
                    logging.info(f"Client ayrıldı: {client_id}")
                    break

                message = data.decode('utf-8').strip()
                logging.info(f"[{client_id}] Mesaj: {message}")

                # Komutları işle
                if message.lower() == 'quit':
                    response = "Güle güle!\n"
                    writer.write(response.encode('utf-8'))
                    await writer.drain()
                    break

                elif message.lower() == 'time':
                    response = f"Sunucu zamanı: {datetime.now()}\n"
                    writer.write(response.encode('utf-8'))
                    await writer.drain()

                elif message.lower() == 'clients':
                    response = f"Bağlı client sayısı: {len(self.clients)}\n"
                    writer.write(response.encode('utf-8'))
                    await writer.drain()

                elif message.lower().startswith('broadcast:'):
                    # Tüm client'lara mesaj gönder
                    broadcast_msg = message[10:].strip()
                    await self.broadcast(f"[{client_id}]: {broadcast_msg}\n",
                                       exclude=client_id)

                    response = "Broadcast gönderildi\n"
                    writer.write(response.encode('utf-8'))
                    await writer.drain()

                else:
                    # Echo
                    response = f"Echo: {message}\n"
                    writer.write(response.encode('utf-8'))
                    await writer.drain()

        except asyncio.CancelledError:
            logging.info(f"Client görevi iptal edildi: {client_id}")
        except Exception as e:
            logging.error(f"Client hatası [{client_id}]: {e}")
        finally:
            # Cleanup
            if client_id in self.clients:
                del self.clients[client_id]

            writer.close()
            await writer.wait_closed()
            logging.info(f"Bağlantı kapatıldı: {client_id}")

    async def broadcast(self, message: str, exclude: str = None):
        """Tüm client'lara mesaj gönder"""
        data = message.encode('utf-8')

        for client_id, writer in list(self.clients.items()):
            if client_id != exclude:
                try:
                    writer.write(data)
                    await writer.drain()
                except Exception as e:
                    logging.error(f"Broadcast hatası [{client_id}]: {e}")

    async def start(self):
        """Server'ı başlat"""
        # start_server: her client için callback çağırır
        server = await asyncio.start_server(
            self.handle_client,
            self.host,
            self.port
        )

        address = server.sockets[0].getsockname()
        logging.info(f"Asyncio server başlatıldı: {address}")

        # Server'ı süresiz çalıştır
        async with server:
            await server.serve_forever()

# Kullanım
async def main():
    logging.basicConfig(level=logging.INFO)
    server = AsyncioTCPServer()
    await server.start()

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("Keyboard interrupt")
```

### Örnek 10: Asyncio Client

```python
import asyncio
import logging

class AsyncioTCPClient:
    """Asyncio ile asenkron TCP client"""

    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self.reader = None
        self.writer = None

    async def connect(self) -> bool:
        """Server'a bağlan"""
        try:
            logging.info(f"Bağlanılıyor: {self.host}:{self.port}")

            # Asenkron bağlan
            self.reader, self.writer = await asyncio.open_connection(
                self.host,
                self.port
            )

            logging.info("Bağlantı başarılı")
            return True

        except Exception as e:
            logging.error(f"Bağlantı hatası: {e}")
            return False

    async def send(self, message: str):
        """Mesaj gönder"""
        if not self.writer:
            logging.error("Bağlantı yok!")
            return

        self.writer.write(message.encode('utf-8'))
        await self.writer.drain()

    async def receive(self) -> str:
        """Veri al"""
        if not self.reader:
            logging.error("Bağlantı yok!")
            return ""

        data = await self.reader.read(4096)
        return data.decode('utf-8')

    async def close(self):
        """Bağlantıyı kapat"""
        if self.writer:
            self.writer.close()
            await self.writer.wait_closed()
            logging.info("Bağlantı kapatıldı")

async def client_example():
    """Client kullanım örneği"""
    logging.basicConfig(level=logging.INFO)

    client = AsyncioTCPClient('127.0.0.1', 9005)

    if await client.connect():
        # Welcome mesajını al
        response = await client.receive()
        print(f"Server: {response}")

        # Mesajlar gönder
        messages = [
            'Hello Asyncio Server',
            'time',
            'clients',
            'broadcast: Hello everyone!',
            'quit'
        ]

        for msg in messages:
            print(f"\nGönderiliyor: {msg}")
            await client.send(msg + '\n')

            response = await client.receive()
            print(f"Cevap: {response}")

            await asyncio.sleep(0.5)  # Kısa bekle

        await client.close()

if __name__ == '__main__':
    asyncio.run(client_example())
```

---

## HTTP Requests (Advanced)

HTTP iletişimi için `requests` ve `httpx` (async) kütüphanelerini kullanabiliriz. Production'da session pooling, retry, timeout gibi özellikler kritiktir.

### Örnek 11: Advanced HTTP Client

```python
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import logging
from typing import Optional, Dict
import time

class AdvancedHTTPClient:
    """
    Production-ready HTTP client
    - Connection pooling
    - Automatic retries
    - Timeout handling
    - Session management
    """

    def __init__(self, base_url: str = "", timeout: float = 10.0):
        self.base_url = base_url
        self.timeout = timeout
        self.session = self._create_session()

    def _create_session(self) -> requests.Session:
        """Retry ve adapter'lı session oluştur"""
        session = requests.Session()

        # Retry stratejisi
        retry_strategy = Retry(
            total=3,  # 3 deneme
            backoff_factor=1,  # 1, 2, 4 saniye bekle
            status_forcelist=[429, 500, 502, 503, 504],  # Bu kodlarda tekrar dene
            method_whitelist=["HEAD", "GET", "OPTIONS", "POST"]
        )

        # HTTP adapter
        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=10,  # Connection pool
            pool_maxsize=20
        )

        # Hem HTTP hem HTTPS için
        session.mount("http://", adapter)
        session.mount("https://", adapter)

        # Default headers
        session.headers.update({
            'User-Agent': 'AdvancedHTTPClient/1.0',
            'Accept': 'application/json'
        })

        return session

    def get(self, endpoint: str, params: Optional[Dict] = None) -> Optional[Dict]:
        """GET request"""
        url = self.base_url + endpoint

        try:
            logging.info(f"GET: {url}")
            start_time = time.time()

            response = self.session.get(
                url,
                params=params,
                timeout=self.timeout
            )

            elapsed = time.time() - start_time
            logging.info(f"Yanıt alındı: {response.status_code} ({elapsed:.2f}s)")

            response.raise_for_status()  # 4xx, 5xx için exception

            return response.json()

        except requests.Timeout:
            logging.error(f"Timeout: {url}")
            return None
        except requests.ConnectionError:
            logging.error(f"Bağlantı hatası: {url}")
            return None
        except requests.HTTPError as e:
            logging.error(f"HTTP hatası: {e}")
            return None
        except Exception as e:
            logging.error(f"İstek hatası: {e}")
            return None

    def post(self, endpoint: str, data: Optional[Dict] = None,
             json: Optional[Dict] = None) -> Optional[Dict]:
        """POST request"""
        url = self.base_url + endpoint

        try:
            logging.info(f"POST: {url}")
            start_time = time.time()

            response = self.session.post(
                url,
                data=data,
                json=json,
                timeout=self.timeout
            )

            elapsed = time.time() - start_time
            logging.info(f"Yanıt alındı: {response.status_code} ({elapsed:.2f}s)")

            response.raise_for_status()

            return response.json() if response.content else {}

        except Exception as e:
            logging.error(f"POST hatası: {e}")
            return None

    def close(self):
        """Session'ı kapat"""
        self.session.close()

# Kullanım
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    # JSONPlaceholder API ile test
    client = AdvancedHTTPClient(base_url='https://jsonplaceholder.typicode.com')

    # GET örneği
    posts = client.get('/posts', params={'_limit': 5})
    if posts:
        print(f"\nİlk 5 post:")
        for post in posts:
            print(f"  {post['id']}: {post['title']}")

    # POST örneği
    new_post = {
        'title': 'Test Post',
        'body': 'Test content',
        'userId': 1
    }
    result = client.post('/posts', json=new_post)
    if result:
        print(f"\nYeni post oluşturuldu: ID={result['id']}")

    client.close()
```

### Örnek 12: Async HTTP Client (httpx)

```python
import asyncio
import httpx
import logging
from typing import Optional, Dict, List

class AsyncHTTPClient:
    """
    Asyncio ile HTTP client (httpx kullanarak)
    Birden fazla request'i paralel yapabilir
    """

    def __init__(self, base_url: str = "", timeout: float = 10.0):
        self.base_url = base_url
        self.timeout = timeout

    async def get(self, endpoint: str, params: Optional[Dict] = None) -> Optional[Dict]:
        """Async GET request"""
        url = self.base_url + endpoint

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                logging.info(f"GET: {url}")
                response = await client.get(url, params=params)
                response.raise_for_status()
                return response.json()

        except httpx.TimeoutException:
            logging.error(f"Timeout: {url}")
            return None
        except Exception as e:
            logging.error(f"GET hatası: {e}")
            return None

    async def post(self, endpoint: str, json: Optional[Dict] = None) -> Optional[Dict]:
        """Async POST request"""
        url = self.base_url + endpoint

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                logging.info(f"POST: {url}")
                response = await client.post(url, json=json)
                response.raise_for_status()
                return response.json() if response.content else {}

        except Exception as e:
            logging.error(f"POST hatası: {e}")
            return None

    async def fetch_multiple(self, endpoints: List[str]) -> List[Optional[Dict]]:
        """Birden fazla endpoint'i paralel fetch et"""
        tasks = [self.get(endpoint) for endpoint in endpoints]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Exception'ları None'a çevir
        return [r if not isinstance(r, Exception) else None for r in results]

# Kullanım
async def async_http_example():
    logging.basicConfig(level=logging.INFO)

    client = AsyncHTTPClient(base_url='https://jsonplaceholder.typicode.com')

    # Tek request
    post = await client.get('/posts/1')
    if post:
        print(f"\nPost 1: {post['title']}")

    # Paralel multiple requests
    endpoints = [f'/posts/{i}' for i in range(1, 6)]
    posts = await client.fetch_multiple(endpoints)

    print(f"\nParalel fetch edilen {len(posts)} post:")
    for post in posts:
        if post:
            print(f"  {post['id']}: {post['title']}")

if __name__ == '__main__':
    asyncio.run(async_http_example())
```

---

## WebSocket Programming

WebSocket, full-duplex iletişim sağlayan bir protokoldür. Real-time uygulamalar (chat, notifications) için idealdir.

### Örnek 13: WebSocket Server (websockets kütüphanesi)

```python
import asyncio
import websockets
import json
import logging
from datetime import datetime
from typing import Set

class WebSocketChatServer:
    """
    WebSocket chat server
    Real-time bidirectional communication
    """

    def __init__(self, host: str = 'localhost', port: int = 8765):
        self.host = host
        self.port = port
        self.clients: Set[websockets.WebSocketServerProtocol] = set()

    async def register(self, websocket):
        """Client'ı kaydet"""
        self.clients.add(websocket)
        logging.info(f"Yeni client: {websocket.remote_address}")
        logging.info(f"Toplam client: {len(self.clients)}")

        # Welcome mesajı
        welcome = {
            'type': 'system',
            'message': 'Hoşgeldiniz!',
            'timestamp': str(datetime.now()),
            'clients_count': len(self.clients)
        }
        await websocket.send(json.dumps(welcome))

    async def unregister(self, websocket):
        """Client'ı kaldır"""
        self.clients.discard(websocket)
        logging.info(f"Client ayrıldı: {websocket.remote_address}")
        logging.info(f"Kalan client: {len(self.clients)}")

    async def broadcast(self, message: dict, exclude=None):
        """Tüm client'lara mesaj gönder"""
        if self.clients:
            message_str = json.dumps(message)

            # Asenkron olarak tüm client'lara gönder
            tasks = [
                client.send(message_str)
                for client in self.clients
                if client != exclude
            ]

            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

    async def handle_client(self, websocket, path):
        """Her client için handler"""
        await self.register(websocket)

        try:
            async for message in websocket:
                try:
                    # JSON parse
                    data = json.loads(message)

                    logging.info(f"[{websocket.remote_address}] Mesaj: {data}")

                    # Mesaj tipine göre işle
                    if data.get('type') == 'chat':
                        # Chat mesajı - broadcast
                        broadcast_msg = {
                            'type': 'chat',
                            'sender': str(websocket.remote_address),
                            'message': data.get('message', ''),
                            'timestamp': str(datetime.now())
                        }
                        await self.broadcast(broadcast_msg, exclude=websocket)

                        # Gönderene confirmation
                        confirm = {
                            'type': 'confirm',
                            'message': 'Mesaj gönderildi'
                        }
                        await websocket.send(json.dumps(confirm))

                    elif data.get('type') == 'ping':
                        # Ping - pong gönder
                        pong = {
                            'type': 'pong',
                            'timestamp': str(datetime.now())
                        }
                        await websocket.send(json.dumps(pong))

                except json.JSONDecodeError:
                    error = {
                        'type': 'error',
                        'message': 'Geçersiz JSON'
                    }
                    await websocket.send(json.dumps(error))

        except websockets.exceptions.ConnectionClosed:
            logging.info(f"Bağlantı kapandı: {websocket.remote_address}")
        finally:
            await self.unregister(websocket)

    async def start(self):
        """Server'ı başlat"""
        logging.info(f"WebSocket server başlatılıyor: ws://{self.host}:{self.port}")

        async with websockets.serve(self.handle_client, self.host, self.port):
            logging.info("WebSocket server hazır")
            await asyncio.Future()  # Süresiz çalıştır

# Kullanım
async def main():
    logging.basicConfig(level=logging.INFO)
    server = WebSocketChatServer()
    await server.start()

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("Server kapatıldı")
```

### Örnek 14: WebSocket Client

```python
import asyncio
import websockets
import json
import logging

class WebSocketClient:
    """WebSocket client"""

    def __init__(self, uri: str):
        self.uri = uri
        self.websocket = None

    async def connect(self):
        """Server'a bağlan"""
        try:
            logging.info(f"Bağlanılıyor: {self.uri}")
            self.websocket = await websockets.connect(self.uri)
            logging.info("Bağlantı başarılı")
            return True
        except Exception as e:
            logging.error(f"Bağlantı hatası: {e}")
            return False

    async def send_message(self, message: str):
        """Chat mesajı gönder"""
        if not self.websocket:
            return

        data = {
            'type': 'chat',
            'message': message
        }
        await self.websocket.send(json.dumps(data))

    async def send_ping(self):
        """Ping gönder"""
        if not self.websocket:
            return

        data = {'type': 'ping'}
        await self.websocket.send(json.dumps(data))

    async def receive_loop(self):
        """Mesaj alma döngüsü"""
        try:
            async for message in self.websocket:
                data = json.loads(message)

                msg_type = data.get('type')

                if msg_type == 'system':
                    print(f"\n[SISTEM] {data.get('message')}")
                    print(f"Bağlı kullanıcı: {data.get('clients_count')}")

                elif msg_type == 'chat':
                    print(f"\n[{data.get('sender')}] {data.get('message')}")

                elif msg_type == 'pong':
                    print(f"\n[PONG] {data.get('timestamp')}")

                elif msg_type == 'confirm':
                    print(f"\n[OK] {data.get('message')}")

        except websockets.exceptions.ConnectionClosed:
            logging.info("Bağlantı kapandı")

    async def close(self):
        """Bağlantıyı kapat"""
        if self.websocket:
            await self.websocket.close()
            logging.info("Bağlantı kapatıldı")

# Kullanım
async def client_example():
    logging.basicConfig(level=logging.INFO)

    client = WebSocketClient('ws://localhost:8765')

    if await client.connect():
        # Receive loop'u task olarak başlat
        receive_task = asyncio.create_task(client.receive_loop())

        # Mesajlar gönder
        await asyncio.sleep(1)

        await client.send_message("Merhaba!")
        await asyncio.sleep(1)

        await client.send_ping()
        await asyncio.sleep(1)

        await client.send_message("Test mesajı")
        await asyncio.sleep(2)

        # Kapat
        await client.close()
        receive_task.cancel()

if __name__ == '__main__':
    asyncio.run(client_example())
```

---

## SSL/TLS Güvenliği

SSL/TLS, network iletişimini şifreler. Production'da mutlaka kullanılmalıdır.

### Örnek 15: SSL ile TCP Server

```python
import socket
import ssl
import threading
import logging

class SSLTCPServer:
    """
    SSL/TLS ile güvenli TCP server
    Sertifika ve private key gerektirir
    """

    def __init__(self, host: str = '127.0.0.1', port: int = 9006,
                 certfile: str = 'server.crt', keyfile: str = 'server.key'):
        self.host = host
        self.port = port
        self.certfile = certfile
        self.keyfile = keyfile
        self.server_socket = None
        self.running = False

    def create_ssl_context(self) -> ssl.SSLContext:
        """SSL context oluştur"""
        # TLS server context
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)

        # Sertifika yükle
        context.load_cert_chain(self.certfile, self.keyfile)

        # Güvenlik ayarları
        context.minimum_version = ssl.TLSVersion.TLSv1_2
        context.maximum_version = ssl.TLSVersion.TLSv1_3

        # Cipher suite'ler
        context.set_ciphers('ECDHE+AESGCM:ECDHE+CHACHA20:DHE+AESGCM:DHE+CHACHA20:!aNULL:!MD5:!DSS')

        return context

    def start(self):
        """Server'ı başlat"""
        try:
            # SSL context
            ssl_context = self.create_ssl_context()

            # Socket oluştur
            raw_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            raw_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            raw_socket.bind((self.host, self.port))
            raw_socket.listen(5)

            # SSL wrap
            self.server_socket = ssl_context.wrap_socket(raw_socket, server_side=True)

            self.running = True
            logging.info(f"SSL Server başlatıldı: {self.host}:{self.port}")

            while self.running:
                try:
                    client_socket, address = self.server_socket.accept()

                    logging.info(f"SSL bağlantı: {address}")
                    logging.info(f"SSL version: {client_socket.version()}")
                    logging.info(f"Cipher: {client_socket.cipher()}")

                    # Thread ile handle et
                    client_thread = threading.Thread(
                        target=self.handle_client,
                        args=(client_socket, address),
                        daemon=True
                    )
                    client_thread.start()

                except Exception as e:
                    if self.running:
                        logging.error(f"Accept hatası: {e}")

        except Exception as e:
            logging.error(f"Server başlatma hatası: {e}")
        finally:
            self.shutdown()

    def handle_client(self, client_socket: ssl.SSLSocket, address: tuple):
        """Client ile iletişim"""
        try:
            client_socket.settimeout(30.0)

            # Welcome
            welcome = b"Secure SSL Server\n"
            client_socket.sendall(welcome)

            while self.running:
                data = client_socket.recv(4096)

                if not data:
                    break

                message = data.decode('utf-8').strip()
                logging.info(f"[{address}] Mesaj: {message}")

                if message.lower() == 'quit':
                    break

                # Echo
                response = f"SSL Echo: {message}\n".encode('utf-8')
                client_socket.sendall(response)

        except Exception as e:
            logging.error(f"Client hatası: {e}")
        finally:
            client_socket.close()
            logging.info(f"SSL bağlantı kapatıldı: {address}")

    def shutdown(self):
        """Server'ı kapat"""
        logging.info("SSL Server kapatılıyor...")
        self.running = False

        if self.server_socket:
            self.server_socket.close()

class SSLTCPClient:
    """SSL/TLS client"""

    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self.socket = None

    def connect(self) -> bool:
        """SSL ile bağlan"""
        try:
            # SSL context
            context = ssl.create_default_context()

            # Self-signed sertifika için (production'da kullanma!)
            context.check_hostname = False
            context.verify_mode = ssl.CERT_NONE

            # Raw socket
            raw_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

            # SSL wrap
            self.socket = context.wrap_socket(raw_socket, server_hostname=self.host)

            # Bağlan
            self.socket.connect((self.host, self.port))

            logging.info("SSL bağlantı başarılı")
            logging.info(f"SSL version: {self.socket.version()}")
            logging.info(f"Cipher: {self.socket.cipher()}")

            return True

        except Exception as e:
            logging.error(f"SSL bağlantı hatası: {e}")
            return False

    def send(self, message: str):
        """Mesaj gönder"""
        if self.socket:
            self.socket.sendall(message.encode('utf-8'))

    def receive(self) -> str:
        """Veri al"""
        if self.socket:
            data = self.socket.recv(4096)
            return data.decode('utf-8')
        return ""

    def close(self):
        """Bağlantıyı kapat"""
        if self.socket:
            self.socket.close()

# NOT: Bu örneği çalıştırmak için sertifika dosyaları gerekir:
# openssl req -x509 -newkey rsa:4096 -keyout server.key -out server.crt -days 365 -nodes
```

---

## Error Handling & Best Practices

Network programlamada robust error handling ve best practice'ler kritiktir.

### Örnek 16: Comprehensive Error Handling

```python
import socket
import errno
import logging
from typing import Optional, Tuple
from contextlib import contextmanager

class NetworkError(Exception):
    """Base network exception"""
    pass

class ConnectionTimeout(NetworkError):
    """Connection timeout exception"""
    pass

class ConnectionRefused(NetworkError):
    """Connection refused exception"""
    pass

class DataTransferError(NetworkError):
    """Data transfer error"""
    pass

class RobustTCPClient:
    """
    Comprehensive error handling ile TCP client
    Her türlü network hatasını yakalayıp handle eder
    """

    def __init__(self, host: str, port: int, timeout: float = 10.0):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.socket: Optional[socket.socket] = None

    @contextmanager
    def _error_handler(self, operation: str):
        """Context manager for error handling"""
        try:
            yield
        except socket.timeout:
            raise ConnectionTimeout(f"{operation}: Timeout ({self.timeout}s)")
        except ConnectionRefusedError:
            raise ConnectionRefused(f"{operation}: Bağlantı reddedildi")
        except socket.gaierror as e:
            raise NetworkError(f"{operation}: DNS çözümleme hatası - {e}")
        except OSError as e:
            if e.errno == errno.ECONNRESET:
                raise DataTransferError(f"{operation}: Bağlantı resetlendi")
            elif e.errno == errno.EPIPE:
                raise DataTransferError(f"{operation}: Broken pipe")
            elif e.errno == errno.ENETUNREACH:
                raise NetworkError(f"{operation}: Network unreachable")
            else:
                raise NetworkError(f"{operation}: OS hatası - {e}")
        except Exception as e:
            raise NetworkError(f"{operation}: Beklenmeyen hata - {e}")

    def connect(self, retries: int = 3) -> bool:
        """
        Bağlan (retry logic ile)
        """
        for attempt in range(1, retries + 1):
            try:
                with self._error_handler("Connect"):
                    logging.info(f"Bağlanma denemesi {attempt}/{retries}")

                    self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    self.socket.settimeout(self.timeout)
                    self.socket.connect((self.host, self.port))

                    logging.info("Bağlantı başarılı")
                    return True

            except NetworkError as e:
                logging.error(f"Deneme {attempt} başarısız: {e}")

                if self.socket:
                    self.socket.close()
                    self.socket = None

                if attempt < retries:
                    import time
                    wait_time = 2 ** attempt  # Exponential backoff
                    logging.info(f"{wait_time} saniye bekleniyor...")
                    time.sleep(wait_time)
                else:
                    logging.error("Tüm denemeler başarısız")
                    return False

        return False

    def send_with_retry(self, data: bytes, retries: int = 3) -> bool:
        """
        Veri gönder (retry ile)
        """
        if not self.socket:
            raise NetworkError("Socket bağlı değil")

        total_sent = 0
        data_length = len(data)

        for attempt in range(1, retries + 1):
            try:
                with self._error_handler("Send"):
                    while total_sent < data_length:
                        sent = self.socket.send(data[total_sent:])

                        if sent == 0:
                            raise DataTransferError("Socket bağlantısı koptu")

                        total_sent += sent

                    logging.info(f"{total_sent} byte gönderildi")
                    return True

            except DataTransferError as e:
                logging.error(f"Gönderme denemesi {attempt} başarısız: {e}")

                if attempt < retries:
                    # Yeniden bağlan
                    self.reconnect()
                else:
                    return False

        return False

    def receive_exact(self, num_bytes: int, timeout: Optional[float] = None) -> Optional[bytes]:
        """
        Tam olarak belirtilen sayıda byte al
        """
        if not self.socket:
            raise NetworkError("Socket bağlı değil")

        if timeout:
            old_timeout = self.socket.gettimeout()
            self.socket.settimeout(timeout)

        try:
            with self._error_handler("Receive"):
                data = b''

                while len(data) < num_bytes:
                    chunk = self.socket.recv(num_bytes - len(data))

                    if not chunk:
                        raise DataTransferError("Bağlantı kapandı")

                    data += chunk

                logging.info(f"{len(data)} byte alındı")
                return data

        finally:
            if timeout:
                self.socket.settimeout(old_timeout)

    def reconnect(self) -> bool:
        """Bağlantıyı yeniden kur"""
        logging.info("Yeniden bağlanılıyor...")
        self.close()
        return self.connect()

    def close(self):
        """Bağlantıyı kapat"""
        if self.socket:
            try:
                # Graceful shutdown
                self.socket.shutdown(socket.SHUT_RDWR)
            except:
                pass

            self.socket.close()
            self.socket = None
            logging.info("Bağlantı kapatıldı")

# Kullanım örneği
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    client = RobustTCPClient('127.0.0.1', 9000)

    try:
        if client.connect(retries=3):
            # Veri gönder
            message = b"Hello Server\n"
            client.send_with_retry(message)

            # Cevap al
            response = client.receive_exact(1024, timeout=5.0)
            if response:
                print(f"Cevap: {response.decode('utf-8')}")

    except NetworkError as e:
        logging.error(f"Network hatası: {e}")

    finally:
        client.close()
```

---

## Özet: Production Best Practices

### 1. Socket Configuration
```python
# Her zaman SO_REUSEADDR kullan
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

# TCP için Nagle algoritmasını devre dışı bırak (low latency)
sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

# Timeout ayarla
sock.settimeout(30.0)

# Buffer boyutlarını optimize et
sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 65536)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 65536)
```

### 2. Error Handling
```python
# Her zaman spesifik exception'ları yakala
try:
    sock.connect((host, port))
except socket.timeout:
    # Timeout handling
    pass
except ConnectionRefusedError:
    # Connection refused handling
    pass
except socket.gaierror:
    # DNS error handling
    pass
except OSError as e:
    # OS-level error handling
    pass
```

### 3. Resource Management
```python
# Context manager kullan
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.connect((host, port))
    # ...

# Veya try-finally
sock = None
try:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # ...
finally:
    if sock:
        sock.close()
```

### 4. Scalability
```python
# Asyncio kullan (modern Python)
async def handle_client(reader, writer):
    data = await reader.read(4096)
    # Process data
    writer.write(response)
    await writer.drain()

# Veya selectors (multiplexing)
import selectors
selector = selectors.DefaultSelector()
selector.register(sock, selectors.EVENT_READ, data=callback)
```

### 5. Security
```python
# SSL/TLS kullan (production)
import ssl
context = ssl.create_default_context()
secure_sock = context.wrap_socket(sock, server_hostname=hostname)

# Input validation
if len(data) > MAX_SIZE:
    raise ValueError("Data too large")

# Rate limiting implement et
```

### 6. Monitoring & Logging
```python
import logging

logging.info(f"Connection from {address}")
logging.error(f"Error: {error}")
logging.warning(f"Slow response: {elapsed}s")

# Metrics
total_connections += 1
avg_response_time = calculate_average()
```

Bu advanced networking guide, production-ready socket programming için gerekli tüm konuları kapsar!
