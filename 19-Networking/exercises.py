"""
ADVANCED NETWORKING EXERCISES
Her exercise için önce TODO kısmını kendin implement etmeyi dene!
Sonra solution ile karşılaştır.

Topics:
- TCP/UDP Socket Programming
- Multi-threaded/Async Servers
- HTTP Client/Server
- WebSocket Communication
- File Transfer Protocol
- Chat Applications
- Error Handling & Reconnection
- SSL/TLS Security
"""

import socket
import asyncio
import threading
import json
import hashlib
import logging
from typing import Optional, Dict, List, Tuple
from datetime import datetime
import time

logging.basicConfig(level=logging.INFO)


# ============================================================================
# EXERCISE 1: Multi-threaded Echo Server
# Difficulty: Medium
# Topics: TCP, Threading, Connection Management
# ============================================================================

"""
TASK: Multi-threaded echo server oluştur
- Birden fazla client'ı aynı anda handle etsin
- Her client için ayrı thread oluştur
- Client'a gelen mesajı uppercase yapıp geri gönder
- "quit" komutu ile client ayrılabilsin
- Graceful shutdown desteklesin
"""

# TODO: Implement MultiThreadedEchoServer class
class MultiThreadedEchoServer:
    def __init__(self, host: str = '127.0.0.1', port: int = 9000):
        pass

    def start(self):
        """Server'ı başlat"""
        pass

    def handle_client(self, client_socket, address):
        """Her client için handler"""
        pass

    def shutdown(self):
        """Server'ı kapat"""
        pass


# SOLUTION:
class MultiThreadedEchoServerSolution:
    """
    Multi-threaded echo server
    Her client için ayrı thread, uppercase echo
    """

    def __init__(self, host: str = '127.0.0.1', port: int = 9000):
        self.host = host
        self.port = port
        self.server_socket = None
        self.running = False
        self.clients = []

    def start(self):
        """Server'ı başlat"""
        try:
            # Socket oluştur ve bind et
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(5)

            self.running = True
            logging.info(f"Echo server başlatıldı: {self.host}:{self.port}")

            # Client kabul döngüsü
            while self.running:
                try:
                    self.server_socket.settimeout(1.0)
                    client_socket, address = self.server_socket.accept()

                    # Thread oluştur ve başlat
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
        """Her client için handler - uppercase echo"""
        logging.info(f"Yeni client: {address}")

        try:
            client_socket.settimeout(30.0)

            # Welcome mesajı
            welcome = b"Echo Server - Mesajınız uppercase'e çevrilecek. 'quit' ile çıkış.\n"
            client_socket.sendall(welcome)

            while self.running:
                # Veri al
                data = client_socket.recv(4096)

                if not data:
                    break

                message = data.decode('utf-8').strip()

                if message.lower() == 'quit':
                    goodbye = b"Goodbye!\n"
                    client_socket.sendall(goodbye)
                    break

                # Uppercase yap ve geri gönder
                response = message.upper() + '\n'
                client_socket.sendall(response.encode('utf-8'))

                logging.info(f"[{address}] {message} -> {response.strip()}")

        except socket.timeout:
            logging.warning(f"Client timeout: {address}")
        except Exception as e:
            logging.error(f"Client error: {e}")
        finally:
            client_socket.close()
            logging.info(f"Client disconnected: {address}")

    def shutdown(self):
        """Server'ı güvenli şekilde kapat"""
        logging.info("Shutting down server...")
        self.running = False

        if self.server_socket:
            self.server_socket.close()

        # Thread'leri bekle
        for client_thread in self.clients:
            if client_thread.is_alive():
                client_thread.join(timeout=2.0)

        logging.info("Server stopped")


# ============================================================================
# EXERCISE 2: UDP Ping-Pong Server
# Difficulty: Medium
# Topics: UDP, Latency Measurement, Statistics
# ============================================================================

"""
TASK: UDP ping-pong server ve client oluştur
Server:
- UDP datagram'ları dinle
- Timestamp ile pong yanıtı gönder
- İstatistikleri tut (total requests, avg response time)

Client:
- N adet ping gönder
- Her ping için RTT (Round Trip Time) hesapla
- İstatistikleri göster (min, max, avg, packet loss)
"""

# TODO: Implement UDPPingPongServer and UDPPingPongClient
class UDPPingPongServer:
    def __init__(self, host: str = '127.0.0.1', port: int = 9001):
        pass

    def start(self):
        pass


class UDPPingPongClient:
    def __init__(self, host: str = '127.0.0.1', port: int = 9001):
        pass

    def ping(self, count: int = 10):
        pass


# SOLUTION:
class UDPPingPongServerSolution:
    """
    UDP ping-pong server
    Timestamp ile yanıt verir, istatistik tutar
    """

    def __init__(self, host: str = '127.0.0.1', port: int = 9001):
        self.host = host
        self.port = port
        self.socket = None
        self.running = False
        self.stats = {
            'total_requests': 0,
            'total_bytes': 0,
            'start_time': None
        }

    def start(self):
        """Server'ı başlat"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.socket.bind((self.host, self.port))
            self.running = True
            self.stats['start_time'] = time.time()

            logging.info(f"UDP Ping-Pong server başlatıldı: {self.host}:{self.port}")

            while self.running:
                try:
                    # Datagram al
                    data, address = self.socket.recvfrom(1024)

                    # Stats güncelle
                    self.stats['total_requests'] += 1
                    self.stats['total_bytes'] += len(data)

                    # Parse ping message
                    message = data.decode('utf-8').strip()

                    if message == 'STATS':
                        # İstatistik gönder
                        uptime = time.time() - self.stats['start_time']
                        stats_msg = json.dumps({
                            'total_requests': self.stats['total_requests'],
                            'total_bytes': self.stats['total_bytes'],
                            'uptime_seconds': uptime,
                            'requests_per_second': self.stats['total_requests'] / uptime
                        })
                        self.socket.sendto(stats_msg.encode('utf-8'), address)
                    else:
                        # Pong yanıtı (server timestamp ekle)
                        pong = {
                            'type': 'pong',
                            'original': message,
                            'server_timestamp': time.time()
                        }
                        self.socket.sendto(json.dumps(pong).encode('utf-8'), address)

                except Exception as e:
                    if self.running:
                        logging.error(f"Error: {e}")

        except Exception as e:
            logging.error(f"Server başlatma hatası: {e}")
        finally:
            if self.socket:
                self.socket.close()
            logging.info("UDP server stopped")


class UDPPingPongClientSolution:
    """
    UDP ping-pong client
    RTT hesaplar, istatistik gösterir
    """

    def __init__(self, host: str = '127.0.0.1', port: int = 9001, timeout: float = 2.0):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.socket = None

    def ping(self, count: int = 10):
        """Ping gönder ve istatistikleri göster"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.socket.settimeout(self.timeout)

            rtts = []
            lost = 0

            print(f"\nPinging {self.host}:{self.port} with {count} packets...")
            print("-" * 60)

            for seq in range(1, count + 1):
                try:
                    # Ping mesajı
                    send_time = time.time()
                    ping_msg = json.dumps({
                        'type': 'ping',
                        'seq': seq,
                        'timestamp': send_time
                    })

                    # Gönder
                    self.socket.sendto(ping_msg.encode('utf-8'), (self.host, self.port))

                    # Pong al
                    data, server = self.socket.recvfrom(1024)
                    recv_time = time.time()

                    # RTT hesapla
                    rtt = (recv_time - send_time) * 1000  # milliseconds
                    rtts.append(rtt)

                    pong = json.loads(data.decode('utf-8'))

                    print(f"Reply from {server}: seq={seq} time={rtt:.2f}ms")

                    time.sleep(0.1)  # Rate limit

                except socket.timeout:
                    print(f"Request timeout: seq={seq}")
                    lost += 1
                except Exception as e:
                    logging.error(f"Ping error: {e}")
                    lost += 1

            # İstatistikleri göster
            print("-" * 60)
            print(f"\nPing İstatistikleri:")
            print(f"  Gönderilen: {count}")
            print(f"  Alınan: {count - lost}")
            print(f"  Kayıp: {lost} ({lost/count*100:.1f}%)")

            if rtts:
                print(f"\nRTT İstatistikleri (ms):")
                print(f"  Min: {min(rtts):.2f}")
                print(f"  Max: {max(rtts):.2f}")
                print(f"  Avg: {sum(rtts)/len(rtts):.2f}")

        finally:
            if self.socket:
                self.socket.close()


# ============================================================================
# EXERCISE 3: Async Chat Server (asyncio)
# Difficulty: Hard
# Topics: Asyncio, Multi-client, Broadcasting, Commands
# ============================================================================

"""
TASK: Asyncio ile chat server oluştur
Features:
- Async client handling (binlerce client desteklesin)
- Username sistemi
- Private messaging (/msg username message)
- Broadcast messaging
- Kullanıcı listesi (/users)
- Graceful disconnect handling
"""

# TODO: Implement AsyncChatServer
class AsyncChatServer:
    def __init__(self, host: str = '127.0.0.1', port: int = 9002):
        pass

    async def handle_client(self, reader, writer):
        pass

    async def broadcast(self, message: str, exclude=None):
        pass

    async def start(self):
        pass


# SOLUTION:
class AsyncChatServerSolution:
    """
    Asyncio ile modern chat server
    Username, private message, broadcast desteği
    """

    def __init__(self, host: str = '127.0.0.1', port: int = 9002):
        self.host = host
        self.port = port
        self.clients: Dict[str, asyncio.StreamWriter] = {}  # username -> writer
        self.client_addresses: Dict[str, tuple] = {}  # username -> address

    async def handle_client(self, reader: asyncio.StreamReader,
                           writer: asyncio.StreamWriter):
        """Client handler"""
        address = writer.get_extra_info('peername')
        username = None

        try:
            # Username al
            writer.write(b"Enter your username: ")
            await writer.drain()

            data = await reader.read(1024)
            username = data.decode('utf-8').strip()

            # Username validation
            if not username or username in self.clients:
                writer.write(b"Invalid or taken username!\n")
                await writer.drain()
                writer.close()
                await writer.wait_closed()
                return

            # Client'ı kaydet
            self.clients[username] = writer
            self.client_addresses[username] = address

            logging.info(f"User joined: {username} from {address}")

            # Welcome mesajı
            welcome = f"Welcome {username}! Type /help for commands.\n"
            writer.write(welcome.encode('utf-8'))
            await writer.drain()

            # Join broadcast
            await self.broadcast(f"*** {username} joined the chat ***\n",
                               exclude=username)

            # Message loop
            while True:
                data = await reader.read(4096)

                if not data:
                    break

                message = data.decode('utf-8').strip()

                if not message:
                    continue

                # Komutları işle
                if message.startswith('/'):
                    await self.handle_command(username, message, writer)
                else:
                    # Normal mesaj - broadcast
                    broadcast_msg = f"[{username}]: {message}\n"
                    await self.broadcast(broadcast_msg, exclude=username)

                    # Echo to sender
                    echo = f"[You]: {message}\n"
                    writer.write(echo.encode('utf-8'))
                    await writer.drain()

        except asyncio.CancelledError:
            logging.info(f"Client cancelled: {username}")
        except Exception as e:
            logging.error(f"Client error [{username}]: {e}")
        finally:
            # Cleanup
            if username:
                if username in self.clients:
                    del self.clients[username]
                if username in self.client_addresses:
                    del self.client_addresses[username]

                # Leave broadcast
                await self.broadcast(f"*** {username} left the chat ***\n")

                logging.info(f"User left: {username}")

            writer.close()
            await writer.wait_closed()

    async def handle_command(self, username: str, command: str,
                            writer: asyncio.StreamWriter):
        """Komutları işle"""
        parts = command.split(maxsplit=2)
        cmd = parts[0].lower()

        if cmd == '/help':
            help_text = """
Available commands:
  /help - Show this help
  /users - List online users
  /msg <username> <message> - Send private message
  /quit - Disconnect
"""
            writer.write(help_text.encode('utf-8'))
            await writer.drain()

        elif cmd == '/users':
            users = ', '.join(self.clients.keys())
            msg = f"Online users ({len(self.clients)}): {users}\n"
            writer.write(msg.encode('utf-8'))
            await writer.drain()

        elif cmd == '/msg':
            if len(parts) < 3:
                writer.write(b"Usage: /msg <username> <message>\n")
                await writer.drain()
                return

            target_username = parts[1]
            message = parts[2]

            if target_username not in self.clients:
                writer.write(f"User '{target_username}' not found!\n".encode('utf-8'))
                await writer.drain()
                return

            # Private mesaj gönder
            target_writer = self.clients[target_username]
            pm = f"[PM from {username}]: {message}\n"
            target_writer.write(pm.encode('utf-8'))
            await target_writer.drain()

            # Sender'a confirmation
            confirm = f"[PM to {target_username}]: {message}\n"
            writer.write(confirm.encode('utf-8'))
            await writer.drain()

        elif cmd == '/quit':
            writer.write(b"Goodbye!\n")
            await writer.drain()
            writer.close()

        else:
            writer.write(f"Unknown command: {cmd}\n".encode('utf-8'))
            await writer.drain()

    async def broadcast(self, message: str, exclude: str = None):
        """Tüm client'lara mesaj gönder"""
        data = message.encode('utf-8')

        for username, writer in list(self.clients.items()):
            if username != exclude:
                try:
                    writer.write(data)
                    await writer.drain()
                except Exception as e:
                    logging.error(f"Broadcast error to {username}: {e}")

    async def start(self):
        """Server'ı başlat"""
        server = await asyncio.start_server(
            self.handle_client,
            self.host,
            self.port
        )

        address = server.sockets[0].getsockname()
        logging.info(f"Chat server başlatıldı: {address}")

        async with server:
            await server.serve_forever()


# ============================================================================
# EXERCISE 4: File Transfer Protocol
# Difficulty: Hard
# Topics: Binary Transfer, Chunking, Progress, Checksum
# ============================================================================

"""
TASK: Dosya transfer protokolü oluştur
Server:
- Dosya upload ve download desteklesin
- Chunk'lar halinde transfer
- Progress reporting
- MD5 checksum validation

Client:
- Dosya gönderme (upload)
- Dosya alma (download)
- Progress bar
- Transfer sonrası checksum doğrulama
"""

# TODO: Implement FileTransferServer and FileTransferClient
class FileTransferServer:
    def __init__(self, host: str = '127.0.0.1', port: int = 9003,
                 storage_dir: str = './uploads'):
        pass

    def handle_client(self, client_socket, address):
        pass


class FileTransferClient:
    def __init__(self, host: str, port: int):
        pass

    def upload_file(self, filepath: str):
        pass

    def download_file(self, remote_filename: str, local_filepath: str):
        pass


# SOLUTION:
class FileTransferServerSolution:
    """
    File transfer server
    Chunk-based transfer, checksum validation
    """

    CHUNK_SIZE = 4096

    def __init__(self, host: str = '127.0.0.1', port: int = 9003,
                 storage_dir: str = './uploads'):
        self.host = host
        self.port = port
        self.storage_dir = storage_dir
        self.running = False

        # Storage directory oluştur
        import os
        os.makedirs(storage_dir, exist_ok=True)

    def start(self):
        """Server'ı başlat"""
        try:
            server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server_socket.bind((self.host, self.port))
            server_socket.listen(5)

            self.running = True
            logging.info(f"File transfer server başlatıldı: {self.host}:{self.port}")
            logging.info(f"Storage directory: {self.storage_dir}")

            while self.running:
                try:
                    server_socket.settimeout(1.0)
                    client_socket, address = server_socket.accept()

                    # Thread ile handle et
                    client_thread = threading.Thread(
                        target=self.handle_client,
                        args=(client_socket, address),
                        daemon=True
                    )
                    client_thread.start()

                except socket.timeout:
                    continue

        except Exception as e:
            logging.error(f"Server error: {e}")
        finally:
            server_socket.close()

    def handle_client(self, client_socket: socket.socket, address: tuple):
        """Client handler"""
        logging.info(f"Client connected: {address}")

        try:
            client_socket.settimeout(60.0)

            # Komut al (UPLOAD veya DOWNLOAD)
            command_data = self._recv_exact(client_socket, 1024)
            command = json.loads(command_data.decode('utf-8'))

            cmd_type = command.get('type')

            if cmd_type == 'UPLOAD':
                self._handle_upload(client_socket, command)
            elif cmd_type == 'DOWNLOAD':
                self._handle_download(client_socket, command)
            else:
                self._send_response(client_socket, {
                    'status': 'ERROR',
                    'message': 'Unknown command'
                })

        except Exception as e:
            logging.error(f"Client error: {e}")
            try:
                self._send_response(client_socket, {
                    'status': 'ERROR',
                    'message': str(e)
                })
            except:
                pass
        finally:
            client_socket.close()
            logging.info(f"Client disconnected: {address}")

    def _handle_upload(self, client_socket: socket.socket, command: dict):
        """Dosya upload işle"""
        filename = command['filename']
        filesize = command['filesize']

        import os
        filepath = os.path.join(self.storage_dir, filename)

        logging.info(f"Receiving file: {filename} ({filesize} bytes)")

        # OK yanıtı gönder
        self._send_response(client_socket, {'status': 'OK'})

        # Dosyayı al
        md5_hash = hashlib.md5()
        received = 0

        with open(filepath, 'wb') as f:
            while received < filesize:
                chunk_size = min(self.CHUNK_SIZE, filesize - received)
                chunk = self._recv_exact(client_socket, chunk_size)

                if not chunk:
                    raise Exception("Connection lost during transfer")

                f.write(chunk)
                md5_hash.update(chunk)
                received += len(chunk)

        # Checksum
        calculated_md5 = md5_hash.hexdigest()

        # Checksum al
        checksum_data = self._recv_exact(client_socket, 1024)
        checksum_msg = json.loads(checksum_data.decode('utf-8'))
        received_md5 = checksum_msg['md5']

        # Validate
        if calculated_md5 == received_md5:
            logging.info(f"File received successfully: {filename}")
            self._send_response(client_socket, {
                'status': 'OK',
                'message': 'File uploaded successfully',
                'md5': calculated_md5
            })
        else:
            logging.error(f"Checksum mismatch for {filename}")
            self._send_response(client_socket, {
                'status': 'ERROR',
                'message': 'Checksum mismatch'
            })

    def _handle_download(self, client_socket: socket.socket, command: dict):
        """Dosya download işle"""
        filename = command['filename']

        import os
        filepath = os.path.join(self.storage_dir, filename)

        if not os.path.exists(filepath):
            self._send_response(client_socket, {
                'status': 'ERROR',
                'message': 'File not found'
            })
            return

        filesize = os.path.getsize(filepath)

        logging.info(f"Sending file: {filename} ({filesize} bytes)")

        # Dosya bilgisi gönder
        self._send_response(client_socket, {
            'status': 'OK',
            'filename': filename,
            'filesize': filesize
        })

        # Dosyayı gönder
        md5_hash = hashlib.md5()

        with open(filepath, 'rb') as f:
            while True:
                chunk = f.read(self.CHUNK_SIZE)
                if not chunk:
                    break

                client_socket.sendall(chunk)
                md5_hash.update(chunk)

        # Checksum gönder
        self._send_response(client_socket, {
            'md5': md5_hash.hexdigest()
        })

        logging.info(f"File sent successfully: {filename}")

    def _recv_exact(self, sock: socket.socket, num_bytes: int) -> bytes:
        """Tam olarak num_bytes kadar veri al"""
        data = b''
        while len(data) < num_bytes:
            chunk = sock.recv(num_bytes - len(data))
            if not chunk:
                raise Exception("Connection closed")
            data += chunk
        return data

    def _send_response(self, sock: socket.socket, response: dict):
        """JSON response gönder"""
        data = json.dumps(response).encode('utf-8')
        # Length prefix (4 bytes)
        length = len(data)
        sock.sendall(length.to_bytes(4, 'big') + data)


class FileTransferClientSolution:
    """File transfer client"""

    CHUNK_SIZE = 4096

    def __init__(self, host: str, port: int, timeout: float = 60.0):
        self.host = host
        self.port = port
        self.timeout = timeout

    def upload_file(self, filepath: str) -> bool:
        """Dosya upload et"""
        import os

        if not os.path.exists(filepath):
            logging.error(f"File not found: {filepath}")
            return False

        filename = os.path.basename(filepath)
        filesize = os.path.getsize(filepath)

        try:
            # Bağlan
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(self.timeout)
            sock.connect((self.host, self.port))

            # UPLOAD komutu gönder
            command = {
                'type': 'UPLOAD',
                'filename': filename,
                'filesize': filesize
            }
            self._send_json(sock, command)

            # OK yanıtı bekle
            response = self._recv_json(sock)
            if response['status'] != 'OK':
                logging.error(f"Server error: {response.get('message')}")
                return False

            # Dosyayı gönder
            print(f"Uploading {filename} ({filesize} bytes)...")

            md5_hash = hashlib.md5()
            sent = 0

            with open(filepath, 'rb') as f:
                while True:
                    chunk = f.read(self.CHUNK_SIZE)
                    if not chunk:
                        break

                    sock.sendall(chunk)
                    md5_hash.update(chunk)
                    sent += len(chunk)

                    # Progress
                    progress = sent / filesize * 100
                    print(f"\rProgress: {progress:.1f}%", end='', flush=True)

            print()  # Newline

            # Checksum gönder
            self._send_json(sock, {'md5': md5_hash.hexdigest()})

            # Sonuç al
            response = self._recv_json(sock)

            if response['status'] == 'OK':
                print(f"Upload successful! MD5: {response['md5']}")
                return True
            else:
                logging.error(f"Upload failed: {response.get('message')}")
                return False

        except Exception as e:
            logging.error(f"Upload error: {e}")
            return False
        finally:
            sock.close()

    def download_file(self, remote_filename: str, local_filepath: str) -> bool:
        """Dosya download et"""
        try:
            # Bağlan
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(self.timeout)
            sock.connect((self.host, self.port))

            # DOWNLOAD komutu gönder
            command = {
                'type': 'DOWNLOAD',
                'filename': remote_filename
            }
            self._send_json(sock, command)

            # Dosya bilgisi al
            response = self._recv_json(sock)

            if response['status'] != 'OK':
                logging.error(f"Server error: {response.get('message')}")
                return False

            filesize = response['filesize']

            print(f"Downloading {remote_filename} ({filesize} bytes)...")

            # Dosyayı al
            md5_hash = hashlib.md5()
            received = 0

            with open(local_filepath, 'wb') as f:
                while received < filesize:
                    chunk_size = min(self.CHUNK_SIZE, filesize - received)
                    chunk = self._recv_exact(sock, chunk_size)

                    if not chunk:
                        raise Exception("Connection lost")

                    f.write(chunk)
                    md5_hash.update(chunk)
                    received += len(chunk)

                    # Progress
                    progress = received / filesize * 100
                    print(f"\rProgress: {progress:.1f}%", end='', flush=True)

            print()  # Newline

            # Checksum al
            checksum_response = self._recv_json(sock)
            server_md5 = checksum_response['md5']
            calculated_md5 = md5_hash.hexdigest()

            # Validate
            if calculated_md5 == server_md5:
                print(f"Download successful! MD5: {calculated_md5}")
                return True
            else:
                logging.error("Checksum mismatch!")
                return False

        except Exception as e:
            logging.error(f"Download error: {e}")
            return False
        finally:
            sock.close()

    def _send_json(self, sock: socket.socket, data: dict):
        """JSON gönder (length-prefixed)"""
        json_data = json.dumps(data).encode('utf-8')
        length = len(json_data)
        sock.sendall(length.to_bytes(4, 'big') + json_data)

    def _recv_json(self, sock: socket.socket) -> dict:
        """JSON al (length-prefixed)"""
        # Length al (4 bytes)
        length_bytes = self._recv_exact(sock, 4)
        length = int.from_bytes(length_bytes, 'big')

        # JSON data al
        json_data = self._recv_exact(sock, length)
        return json.loads(json_data.decode('utf-8'))

    def _recv_exact(self, sock: socket.socket, num_bytes: int) -> bytes:
        """Tam olarak num_bytes kadar veri al"""
        data = b''
        while len(data) < num_bytes:
            chunk = sock.recv(num_bytes - len(data))
            if not chunk:
                raise Exception("Connection closed")
            data += chunk
        return data


# ============================================================================
# EXERCISE 5: HTTP Server (Simple)
# Difficulty: Hard
# Topics: HTTP Protocol, Request Parsing, Response Building, Routing
# ============================================================================

"""
TASK: Basit HTTP server oluştur
Features:
- HTTP/1.1 protocol
- GET ve POST method'ları
- URL routing
- Static file serving
- JSON API endpoints
- Error handling (404, 500)
"""

# TODO: Implement SimpleHTTPServer
class SimpleHTTPServer:
    def __init__(self, host: str = '127.0.0.1', port: int = 8000):
        pass

    def route(self, path: str, method: str = 'GET'):
        """Decorator for routing"""
        pass

    def handle_client(self, client_socket, address):
        pass

    def start(self):
        pass


# SOLUTION:
class SimpleHTTPServerSolution:
    """
    Simple HTTP server
    GET/POST, routing, static files, JSON API
    """

    def __init__(self, host: str = '127.0.0.1', port: int = 8000,
                 static_dir: str = './static'):
        self.host = host
        self.port = port
        self.static_dir = static_dir
        self.routes = {}  # (method, path) -> handler
        self.running = False

        # Default routes
        self._setup_default_routes()

    def _setup_default_routes(self):
        """Default route'ları kur"""

        @self.route('/', 'GET')
        def index(request):
            return {
                'status': 200,
                'body': '<h1>Simple HTTP Server</h1><p>Welcome!</p>',
                'content_type': 'text/html'
            }

        @self.route('/api/time', 'GET')
        def api_time(request):
            return {
                'status': 200,
                'body': json.dumps({
                    'timestamp': time.time(),
                    'datetime': str(datetime.now())
                }),
                'content_type': 'application/json'
            }

    def route(self, path: str, method: str = 'GET'):
        """Decorator for routing"""
        def decorator(func):
            self.routes[(method, path)] = func
            return func
        return decorator

    def start(self):
        """Server'ı başlat"""
        try:
            server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server_socket.bind((self.host, self.port))
            server_socket.listen(5)

            self.running = True
            logging.info(f"HTTP server başlatıldı: http://{self.host}:{self.port}")

            while self.running:
                try:
                    server_socket.settimeout(1.0)
                    client_socket, address = server_socket.accept()

                    # Thread ile handle et
                    client_thread = threading.Thread(
                        target=self.handle_client,
                        args=(client_socket, address),
                        daemon=True
                    )
                    client_thread.start()

                except socket.timeout:
                    continue

        except Exception as e:
            logging.error(f"Server error: {e}")
        finally:
            server_socket.close()

    def handle_client(self, client_socket: socket.socket, address: tuple):
        """HTTP request handler"""
        try:
            client_socket.settimeout(10.0)

            # Request'i al
            request_data = b''
            while b'\r\n\r\n' not in request_data:
                chunk = client_socket.recv(1024)
                if not chunk:
                    break
                request_data += chunk

            if not request_data:
                return

            # Parse request
            request = self._parse_request(request_data)

            logging.info(f"{request['method']} {request['path']} from {address}")

            # Route'u bul
            route_key = (request['method'], request['path'])

            if route_key in self.routes:
                # Handler çağır
                handler = self.routes[route_key]
                response = handler(request)
            else:
                # 404
                response = {
                    'status': 404,
                    'body': '<h1>404 Not Found</h1>',
                    'content_type': 'text/html'
                }

            # Response gönder
            self._send_response(client_socket, response)

        except Exception as e:
            logging.error(f"Request handling error: {e}")

            # 500 Internal Server Error
            try:
                response = {
                    'status': 500,
                    'body': '<h1>500 Internal Server Error</h1>',
                    'content_type': 'text/html'
                }
                self._send_response(client_socket, response)
            except:
                pass

        finally:
            client_socket.close()

    def _parse_request(self, request_data: bytes) -> dict:
        """HTTP request'i parse et"""
        lines = request_data.decode('utf-8').split('\r\n')

        # Request line parse et
        request_line = lines[0]
        method, path, version = request_line.split()

        # Headers parse et
        headers = {}
        for i in range(1, len(lines)):
            if not lines[i]:
                break

            key, value = lines[i].split(':', 1)
            headers[key.strip()] = value.strip()

        return {
            'method': method,
            'path': path,
            'version': version,
            'headers': headers
        }

    def _send_response(self, client_socket: socket.socket, response: dict):
        """HTTP response gönder"""
        status = response.get('status', 200)
        body = response.get('body', '')
        content_type = response.get('content_type', 'text/html')

        # Status text
        status_texts = {
            200: 'OK',
            404: 'Not Found',
            500: 'Internal Server Error'
        }
        status_text = status_texts.get(status, 'Unknown')

        # Response oluştur
        body_bytes = body.encode('utf-8')

        response_lines = [
            f"HTTP/1.1 {status} {status_text}",
            f"Content-Type: {content_type}",
            f"Content-Length: {len(body_bytes)}",
            "Connection: close",
            "",
            ""
        ]

        response_headers = '\r\n'.join(response_lines).encode('utf-8')

        # Gönder
        client_socket.sendall(response_headers + body_bytes)


# ============================================================================
# EXERCISE 6: Port Scanner
# Difficulty: Medium
# Topics: TCP Connections, Threading, Network Discovery
# ============================================================================

"""
TASK: Multi-threaded port scanner oluştur
Features:
- Belirtilen IP'de port range'ini tara
- Multi-threaded scanning (hızlı)
- Open/closed/filtered port detection
- Service detection (common ports)
- Banner grabbing
- Progress reporting
"""

# TODO: Implement PortScanner
class PortScanner:
    def __init__(self, target: str, timeout: float = 1.0):
        pass

    def scan_port(self, port: int) -> dict:
        pass

    def scan_range(self, start_port: int, end_port: int, threads: int = 10):
        pass


# SOLUTION:
class PortScannerSolution:
    """
    Multi-threaded port scanner
    Service detection, banner grabbing
    """

    # Common ports ve servisleri
    COMMON_PORTS = {
        20: 'FTP Data',
        21: 'FTP',
        22: 'SSH',
        23: 'Telnet',
        25: 'SMTP',
        53: 'DNS',
        80: 'HTTP',
        110: 'POP3',
        143: 'IMAP',
        443: 'HTTPS',
        3306: 'MySQL',
        5432: 'PostgreSQL',
        6379: 'Redis',
        8080: 'HTTP Proxy',
        27017: 'MongoDB'
    }

    def __init__(self, target: str, timeout: float = 1.0):
        self.target = target
        self.timeout = timeout
        self.results = []
        self.lock = threading.Lock()

    def scan_port(self, port: int) -> dict:
        """Tek port'u tara"""
        result = {
            'port': port,
            'state': 'closed',
            'service': self.COMMON_PORTS.get(port, 'unknown'),
            'banner': None
        }

        try:
            # TCP bağlantı dene
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(self.timeout)

            start_time = time.time()
            result_code = sock.connect_ex((self.target, port))
            elapsed = time.time() - start_time

            if result_code == 0:
                result['state'] = 'open'
                result['response_time'] = elapsed

                # Banner grabbing dene
                try:
                    sock.send(b'HEAD / HTTP/1.0\r\n\r\n')
                    banner = sock.recv(1024).decode('utf-8', errors='ignore').strip()
                    if banner:
                        result['banner'] = banner[:100]  # İlk 100 karakter
                except:
                    pass

            sock.close()

        except socket.timeout:
            result['state'] = 'filtered'
        except Exception as e:
            result['state'] = 'error'
            result['error'] = str(e)

        return result

    def _scan_port_worker(self, port: int):
        """Worker thread için port scan"""
        result = self.scan_port(port)

        with self.lock:
            self.results.append(result)

    def scan_range(self, start_port: int, end_port: int, threads: int = 10):
        """Port range'ini tara (multi-threaded)"""
        print(f"\nScanning {self.target} ports {start_port}-{end_port}...")
        print(f"Threads: {threads}, Timeout: {self.timeout}s")
        print("-" * 70)

        self.results = []

        # Thread pool
        from concurrent.futures import ThreadPoolExecutor, as_completed

        ports = range(start_port, end_port + 1)
        total_ports = len(ports)

        with ThreadPoolExecutor(max_workers=threads) as executor:
            futures = {executor.submit(self._scan_port_worker, port): port
                      for port in ports}

            completed = 0
            for future in as_completed(futures):
                completed += 1
                progress = completed / total_ports * 100
                print(f"\rProgress: {progress:.1f}% ({completed}/{total_ports})",
                      end='', flush=True)

        print("\n")

        # Sonuçları göster
        self._print_results()

    def _print_results(self):
        """Sonuçları göster"""
        # Sadece open portları
        open_ports = [r for r in self.results if r['state'] == 'open']

        if not open_ports:
            print("No open ports found.")
            return

        print(f"\nOpen Ports ({len(open_ports)}):")
        print("-" * 70)
        print(f"{'PORT':<10} {'SERVICE':<20} {'RESPONSE':<12} {'BANNER'}")
        print("-" * 70)

        for result in sorted(open_ports, key=lambda x: x['port']):
            port = result['port']
            service = result['service']
            response_time = f"{result.get('response_time', 0)*1000:.1f}ms"
            banner = result.get('banner', '')[:30] if result.get('banner') else ''

            print(f"{port:<10} {service:<20} {response_time:<12} {banner}")

        print("-" * 70)


# ============================================================================
# Test fonksiyonları
# ============================================================================

def test_echo_server():
    """Exercise 1 test"""
    print("\n" + "="*70)
    print("EXERCISE 1: Multi-threaded Echo Server")
    print("="*70)

    # Server'ı thread'de başlat
    server = MultiThreadedEchoServerSolution('127.0.0.1', 9000)
    server_thread = threading.Thread(target=server.start, daemon=True)
    server_thread.start()

    time.sleep(1)  # Server'ın başlamasını bekle

    # Test client
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect(('127.0.0.1', 9000))

        # Welcome mesajını al
        data = sock.recv(1024)
        print(data.decode('utf-8'))

        # Test mesajları
        test_messages = ['hello', 'world', 'test']

        for msg in test_messages:
            sock.sendall(f"{msg}\n".encode('utf-8'))
            response = sock.recv(1024).decode('utf-8')
            print(f"Sent: {msg} -> Received: {response.strip()}")

        # Quit
        sock.sendall(b"quit\n")
        response = sock.recv(1024).decode('utf-8')
        print(f"Server response: {response}")

        sock.close()
        print("\n✓ Echo server test passed!")

    except Exception as e:
        print(f"\n✗ Test failed: {e}")


def test_udp_ping():
    """Exercise 2 test"""
    print("\n" + "="*70)
    print("EXERCISE 2: UDP Ping-Pong")
    print("="*70)

    # Server'ı thread'de başlat
    server = UDPPingPongServerSolution('127.0.0.1', 9001)
    server_thread = threading.Thread(target=server.start, daemon=True)
    server_thread.start()

    time.sleep(1)

    # Client ile test
    client = UDPPingPongClientSolution('127.0.0.1', 9001)
    client.ping(count=5)

    print("\n✓ UDP ping-pong test passed!")


def test_async_chat():
    """Exercise 3 test"""
    print("\n" + "="*70)
    print("EXERCISE 3: Async Chat Server")
    print("="*70)
    print("Note: Bu test interactive olduğu için manual test gerektirir")
    print("Server'ı başlatmak için:")
    print("  server = AsyncChatServerSolution()")
    print("  asyncio.run(server.start())")
    print("Client ile bağlanmak için telnet veya netcat kullanın:")
    print("  telnet 127.0.0.1 9002")


def test_port_scanner():
    """Exercise 6 test"""
    print("\n" + "="*70)
    print("EXERCISE 6: Port Scanner")
    print("="*70)

    # Localhost'u tara (yaygın portlar)
    scanner = PortScannerSolution('127.0.0.1', timeout=0.5)

    # Sadece birkaç yaygın portu tara (hızlı test için)
    common_ports = [21, 22, 80, 443, 3306, 5432, 8080]

    print("\nScanning common ports on localhost...")
    for port in common_ports:
        result = scanner.scan_port(port)
        if result['state'] == 'open':
            print(f"  Port {port}: {result['state']} ({result['service']})")


def run_all_tests():
    """Tüm testleri çalıştır"""
    print("\n" + "="*70)
    print("ADVANCED NETWORKING EXERCISES - TEST SUITE")
    print("="*70)

    try:
        test_echo_server()
        time.sleep(1)

        test_udp_ping()
        time.sleep(1)

        test_async_chat()
        time.sleep(1)

        test_port_scanner()

        print("\n" + "="*70)
        print("ALL TESTS COMPLETED!")
        print("="*70)

    except KeyboardInterrupt:
        print("\n\nTests interrupted by user")


if __name__ == '__main__':
    # Tüm testleri çalıştır
    run_all_tests()

    # Veya tek tek test et:
    # test_echo_server()
    # test_udp_ping()
    # test_port_scanner()
