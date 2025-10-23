"""
Advanced Data Structures - İleri Seviye Egzersizler
Bu dosya, Python'un ileri seviye veri yapılarını kullanan zorlu egzersizler içerir.
Her egzersiz TODO kısmı ve çözümü ile birlikte verilmiştir.
"""

from collections import deque, Counter, defaultdict, OrderedDict, ChainMap
import heapq
import bisect
from typing import List, Dict, Tuple, Optional, Set
import time


# ============================================================================
# EXERCISE 1: Time-Based Key-Value Store (Medium-Hard)
# ============================================================================
"""
Timestamp'li key-value store implemente edin. Her set işlemi timestamp ile kaydedilir.
get(key, timestamp) çağrısı, verilen timestamp'ten önce veya o anda set edilen
en son değeri döndürmelidir.

Gereksinimler:
- set(key, value, timestamp): O(1) amortized
- get(key, timestamp): O(log n) where n is number of timestamps for that key
- bisect kullanarak timestamp araması yapın
"""

class TimeMap:
    def __init__(self):
        # TODO: Data structure tasarlayın
        # İpucu: Her key için (timestamp, value) pair'lerini tutun
        pass

    def set(self, key: str, value: str, timestamp: int) -> None:
        # TODO: Key-value pair'i timestamp ile kaydedin
        pass

    def get(self, key: str, timestamp: int) -> str:
        # TODO: Verilen timestamp'ten önce/o anda set edilen son değeri bulun
        # İpucu: bisect_right kullanın
        pass


# ============== SOLUTION ==============
class TimeMapSolution:
    """
    Time Complexity:
    - set: O(1) amortized
    - get: O(log n) binary search on timestamps

    Space Complexity: O(N) where N is total number of set operations
    """

    def __init__(self):
        # Her key için (timestamp, value) pair listesi
        self.store: Dict[str, List[Tuple[int, str]]] = defaultdict(list)

    def set(self, key: str, value: str, timestamp: int) -> None:
        """Timestamp'li value ekle"""
        self.store[key].append((timestamp, value))

    def get(self, key: str, timestamp: int) -> str:
        """Timestamp'ten önce/o anda set edilen son değeri bul"""
        if key not in self.store:
            return ""

        values = self.store[key]

        # Binary search ile timestamp'i bul
        # bisect_right: timestamp'ten büyük olan ilk index
        idx = bisect.bisect_right(values, (timestamp, chr(127)))

        # idx 0 ise, timestamp'ten önce hiç değer yok
        if idx == 0:
            return ""

        return values[idx - 1][1]


# Test
def test_timemap():
    tm = TimeMapSolution()
    tm.set("key1", "value1", 1)
    tm.set("key1", "value2", 3)
    tm.set("key1", "value3", 5)

    assert tm.get("key1", 2) == "value1"  # timestamp 1'deki değer
    assert tm.get("key1", 3) == "value2"  # timestamp 3'teki değer
    assert tm.get("key1", 4) == "value2"  # timestamp 3'teki değer
    assert tm.get("key1", 6) == "value3"  # timestamp 5'teki değer
    assert tm.get("key1", 0) == ""        # Hiç değer yok
    print("✓ TimeMap tests passed")


# ============================================================================
# EXERCISE 2: LFU Cache (Hard)
# ============================================================================
"""
LFU (Least Frequently Used) Cache implemente edin.
LRU'dan farkı: En az kullanılan elemanı çıkarır (frequency'ye göre).
Aynı frequency'de birden fazla eleman varsa, en az yakın zamanda kullanılanı çıkarır.

Gereksinimler:
- get(key): O(1)
- put(key, value): O(1)
- Counter ve OrderedDict kombinasyonu kullanabilirsiniz
"""

class LFUCache:
    def __init__(self, capacity: int):
        # TODO: Data structures tasarlayın
        # İpucu: frequency -> OrderedDict(key -> value) mapping
        # İpucu: key -> frequency mapping
        # İpucu: min_freq tutun
        pass

    def get(self, key: int) -> int:
        # TODO: Değeri getir ve frequency'yi artır
        # O(1) olmalı
        pass

    def put(self, key: int, value: int) -> None:
        # TODO: Değer ekle/güncelle, kapasite doluysa LFU elemanı çıkar
        # O(1) olmalı
        pass


# ============== SOLUTION ==============
class LFUCacheSolution:
    """
    LFU Cache with O(1) operations

    Data Structures:
    1. freq_map: frequency -> OrderedDict(key -> value)
    2. key_freq: key -> frequency
    3. min_freq: minimum frequency tracker
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.min_freq = 0
        self.key_freq: Dict[int, int] = {}  # key -> frequency
        self.freq_map: Dict[int, OrderedDict] = defaultdict(OrderedDict)  # freq -> {key: value}

    def _update_freq(self, key: int) -> None:
        """Key'in frequency'sini artır ve yeniden konumlandır"""
        freq = self.key_freq[key]
        value = self.freq_map[freq][key]

        # Eski frequency'den çıkar
        del self.freq_map[freq][key]

        # Eski frequency boş kaldıysa ve min_freq ise, min_freq'i artır
        if not self.freq_map[freq] and freq == self.min_freq:
            self.min_freq += 1

        # Yeni frequency'e ekle
        new_freq = freq + 1
        self.key_freq[key] = new_freq
        self.freq_map[new_freq][key] = value

    def get(self, key: int) -> int:
        """Değeri getir ve frequency artır - O(1)"""
        if key not in self.key_freq:
            return -1

        self._update_freq(key)
        return self.freq_map[self.key_freq[key]][key]

    def put(self, key: int, value: int) -> None:
        """Değer ekle/güncelle - O(1)"""
        if self.capacity <= 0:
            return

        # Key zaten varsa güncelle
        if key in self.key_freq:
            self.freq_map[self.key_freq[key]][key] = value
            self._update_freq(key)
            return

        # Kapasite doluysa LFU elemanı çıkar
        if len(self.key_freq) >= self.capacity:
            # min_freq'teki ilk elemanı (en eski) çıkar
            evict_key, _ = self.freq_map[self.min_freq].popitem(last=False)
            del self.key_freq[evict_key]

        # Yeni key ekle
        self.key_freq[key] = 1
        self.freq_map[1][key] = value
        self.min_freq = 1


# Test
def test_lfu_cache():
    cache = LFUCacheSolution(2)
    cache.put(1, 1)
    cache.put(2, 2)
    assert cache.get(1) == 1  # freq: {1: 2, 2: 1}
    cache.put(3, 3)            # evicts key 2 (LFU with freq=1)
    assert cache.get(2) == -1
    assert cache.get(3) == 3   # freq: {1: 2, 3: 2}, both accessed
    cache.put(4, 4)            # evicts key 1 (LRU among freq=2, since 1 was accessed before 3)
    assert cache.get(1) == -1  # key 1 was evicted
    assert cache.get(3) == 3   # key 3 still there
    assert cache.get(4) == 4
    print("✓ LFU Cache tests passed")


# ============================================================================
# EXERCISE 3: Design Twitter Feed (Medium-Hard)
# ============================================================================
"""
Twitter benzeri bir feed sistemi tasarlayın:
- postTweet(userId, tweetId): User bir tweet atar
- getNewsFeed(userId): User ve takip ettiklerinin son 10 tweet'ini döndürür
- follow(followerId, followeeId): Takip et
- unfollow(followerId, followeeId): Takibi bırak

Gereksinimler:
- getNewsFeed: O(N log K) where N is number of followed users
- heapq kullanarak K-way merge yapın
"""

class Twitter:
    def __init__(self):
        # TODO: Data structures tasarlayın
        # İpucu: user -> tweets mapping
        # İpucu: user -> following set mapping
        # İpucu: global timestamp counter
        pass

    def postTweet(self, userId: int, tweetId: int) -> None:
        # TODO: Tweet ekle
        pass

    def getNewsFeed(self, userId: int) -> List[int]:
        # TODO: Son 10 tweet'i getir (heapq ile merge)
        pass

    def follow(self, followerId: int, followeeId: int) -> None:
        # TODO: Takip et
        pass

    def unfollow(self, followerId: int, followeeId: int) -> None:
        # TODO: Takibi bırak
        pass


# ============== SOLUTION ==============
class TwitterSolution:
    """
    Twitter feed with efficient timeline merge using heapq

    Time Complexity:
    - postTweet: O(1)
    - follow/unfollow: O(1)
    - getNewsFeed: O(N log K) where N is total tweets, K is feed size
    """

    def __init__(self):
        self.timestamp = 0
        self.tweets: Dict[int, List[Tuple[int, int]]] = defaultdict(list)  # userId -> [(time, tweetId)]
        self.following: Dict[int, Set[int]] = defaultdict(set)  # userId -> {followeeIds}

    def postTweet(self, userId: int, tweetId: int) -> None:
        """Tweet at - O(1)"""
        self.tweets[userId].append((self.timestamp, tweetId))
        self.timestamp += 1

    def getNewsFeed(self, userId: int) -> List[int]:
        """
        Son 10 tweet'i getir - O(N log K)
        K-way merge with max heap
        """
        # Max heap (timestamp'e göre büyükten küçüğe)
        max_heap = []

        # User'ın kendi tweet'leri
        if userId in self.tweets:
            for time, tweet_id in self.tweets[userId]:
                heapq.heappush(max_heap, (-time, tweet_id))

        # Takip edilenlerin tweet'leri
        for followee in self.following[userId]:
            if followee in self.tweets:
                for time, tweet_id in self.tweets[followee]:
                    heapq.heappush(max_heap, (-time, tweet_id))

        # En yeni 10 tweet
        feed = []
        for _ in range(min(10, len(max_heap))):
            _, tweet_id = heapq.heappop(max_heap)
            feed.append(tweet_id)

        return feed

    def follow(self, followerId: int, followeeId: int) -> None:
        """Takip et - O(1)"""
        if followerId != followeeId:  # Kendini takip edemez
            self.following[followerId].add(followeeId)

    def unfollow(self, followerId: int, followeeId: int) -> None:
        """Takibi bırak - O(1)"""
        self.following[followerId].discard(followeeId)


# Test
def test_twitter():
    twitter = TwitterSolution()
    twitter.postTweet(1, 5)
    assert twitter.getNewsFeed(1) == [5]
    twitter.follow(1, 2)
    twitter.postTweet(2, 6)
    assert twitter.getNewsFeed(1) == [6, 5]
    twitter.unfollow(1, 2)
    assert twitter.getNewsFeed(1) == [5]
    print("✓ Twitter tests passed")


# ============================================================================
# EXERCISE 4: Design Search Autocomplete System (Hard)
# ============================================================================
"""
Arama autocomplete sistemi tasarlayın:
- Trie kullanarak prefix matching
- Popularity'ye göre sıralama
- Real-time search suggestions

Gereksinimimler:
- insert(sentence, times): O(M) where M is sentence length
- search(prefix): O(P + N log K) where P is prefix length, N is matches, K is top results
"""

class AutocompleteSystem:
    def __init__(self):
        # TODO: Trie ve popularity counter tasarlayın
        pass

    def insert(self, sentence: str, times: int = 1) -> None:
        # TODO: Sentence'ı trie'a ekle ve popularity güncelle
        pass

    def search(self, prefix: str, top_k: int = 3) -> List[str]:
        # TODO: Prefix'e uyan top K suggestion döndür
        pass


# ============== SOLUTION ==============
class TrieNode:
    def __init__(self):
        self.children: Dict[str, 'TrieNode'] = {}
        self.sentences: Set[str] = set()  # Bu node'dan geçen sentence'lar

class AutocompleteSystemSolution:
    """
    Autocomplete with Trie + popularity ranking

    Time Complexity:
    - insert: O(M) where M is sentence length
    - search: O(P + N log K) where P is prefix, N is matches, K is top results
    """

    def __init__(self):
        self.root = TrieNode()
        self.popularity: Counter = Counter()

    def insert(self, sentence: str, times: int = 1) -> None:
        """Sentence ekle ve popularity artır - O(M)"""
        # Trie'a ekle
        node = self.root
        for char in sentence:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
            node.sentences.add(sentence)

        # Popularity güncelle
        self.popularity[sentence] += times

    def search(self, prefix: str, top_k: int = 3) -> List[str]:
        """
        Prefix'e uyan top K suggestion döndür - O(P + N log K)

        Priority: higher count > lexicographically smaller
        """
        # Prefix node'unu bul
        node = self.root
        for char in prefix:
            if char not in node.children:
                return []
            node = node.children[char]

        # Bu node'dan geçen tüm sentence'ları al
        candidates = list(node.sentences)

        # Popularity ve alfabetik sıraya göre sırala
        # (-count, sentence) tuple: count'a göre descending, sentence'a göre ascending
        candidates.sort(key=lambda s: (-self.popularity[s], s))

        return candidates[:top_k]


# Test
def test_autocomplete():
    system = AutocompleteSystemSolution()
    system.insert("python tutorial", 5)
    system.insert("python advanced", 3)
    system.insert("python basics", 2)
    system.insert("java tutorial", 1)

    results = system.search("python", 3)
    assert results == ["python tutorial", "python advanced", "python basics"]

    results = system.search("pyth", 2)
    assert results == ["python tutorial", "python advanced"]

    results = system.search("java", 1)
    assert results == ["java tutorial"]

    print("✓ Autocomplete tests passed")


# ============================================================================
# EXERCISE 5: Meeting Rooms II (Medium)
# ============================================================================
"""
Verilen meeting time interval'larına göre, gerekli minimum toplantı odası sayısını bulun.
heapq kullanarak overlap'leri tespit edin.

Örnek:
Input: [[0, 30], [5, 10], [15, 20]]
Output: 2 (0-30 ve 5-10 overlap olur)

Gereksinimler:
- O(N log N) time complexity
- O(N) space complexity
"""

def min_meeting_rooms(intervals: List[List[int]]) -> int:
    # TODO: heapq ile çözün
    # İpucu: Start time'a göre sırala, end time'ları heap'te tut
    pass


# ============== SOLUTION ==============
def min_meeting_rooms_solution(intervals: List[List[int]]) -> int:
    """
    Minimum meeting room sayısını hesapla - O(N log N)

    Algorithm:
    1. Interval'ları start time'a göre sırala
    2. Min heap ile end time'ları takip et
    3. Yeni meeting başlarken, bitmiş olanları heap'ten çıkar
    4. Heap size = gerekli oda sayısı
    """
    if not intervals:
        return 0

    # Start time'a göre sırala - O(N log N)
    intervals.sort(key=lambda x: x[0])

    # Min heap: end time'ları tutar
    heap = []
    heapq.heappush(heap, intervals[0][1])  # İlk meeting'in end time'ı

    for start, end in intervals[1:]:
        # En erken biten meeting, yeni meeting başlamadan bittiyse çıkar
        if heap[0] <= start:
            heapq.heappop(heap)

        # Yeni meeting'in end time'ını ekle
        heapq.heappush(heap, end)

    # Heap size = aynı anda devam eden max meeting sayısı
    return len(heap)


# Test
def test_meeting_rooms():
    assert min_meeting_rooms_solution([[0, 30], [5, 10], [15, 20]]) == 2
    assert min_meeting_rooms_solution([[7, 10], [2, 4]]) == 1
    assert min_meeting_rooms_solution([[1, 5], [8, 9], [8, 9]]) == 2
    assert min_meeting_rooms_solution([[1, 10], [2, 7], [3, 19], [8, 12], [10, 20]]) == 3
    print("✓ Meeting Rooms tests passed")


# ============================================================================
# EXERCISE 6: Sliding Window Maximum (Hard)
# ============================================================================
"""
Sliding window içindeki maximum değeri bul.
deque kullanarak O(N) çözüm implemente edin.

Örnek:
Input: nums = [1,3,-1,-3,5,3,6,7], k = 3
Output: [3,3,5,5,6,7]

Gereksinimler:
- O(N) time complexity
- O(K) space complexity
- deque kullanın
"""

def max_sliding_window(nums: List[int], k: int) -> List[int]:
    # TODO: deque ile monotonic decreasing queue oluşturun
    # İpucu: Queue'da her zaman büyükten küçüğe sıralı index'ler olsun
    pass


# ============== SOLUTION ==============
def max_sliding_window_solution(nums: List[int], k: int) -> List[int]:
    """
    Sliding window maximum - O(N) with monotonic deque

    Algorithm:
    - Deque'da decreasing order'da index'ler tut
    - Her eleman için:
      1. Window dışındaki index'leri çıkar
      2. Mevcut elemandan küçük olanları çıkar
      3. Mevcut index'i ekle
      4. Window tam boyuttaysa, deque[0] = max
    """
    if not nums or k == 0:
        return []

    dq = deque()  # Index'leri tutar
    result = []

    for i, num in enumerate(nums):
        # Window dışındaki index'leri çıkar
        while dq and dq[0] <= i - k:
            dq.popleft()

        # Mevcut sayıdan küçük olanları çıkar (monotonic decreasing)
        while dq and nums[dq[-1]] < num:
            dq.pop()

        dq.append(i)

        # Window tam boyuttaysa result'a ekle
        if i >= k - 1:
            result.append(nums[dq[0]])

    return result


# Test
def test_sliding_window():
    assert max_sliding_window_solution([1,3,-1,-3,5,3,6,7], 3) == [3,3,5,5,6,7]
    assert max_sliding_window_solution([1], 1) == [1]
    assert max_sliding_window_solution([1,-1], 1) == [1,-1]
    assert max_sliding_window_solution([9,11], 2) == [11]
    print("✓ Sliding Window Maximum tests passed")


# ============================================================================
# EXERCISE 7: Task Scheduler (Medium-Hard)
# ============================================================================
"""
CPU task scheduler: Aynı task'lar arasında n cooldown süresi olmalı.
Counter ve heapq kullanarak optimal sıralamayı bulun.

Örnek:
Input: tasks = ["A","A","A","B","B","B"], n = 2
Output: 8
Explanation: A -> B -> idle -> A -> B -> idle -> A -> B

Gereksinimler:
- O(N log K) where K is unique tasks
- Counter ve heapq kullanın
"""

def least_interval(tasks: List[str], n: int) -> int:
    # TODO: Task frequency'leri hesapla ve greedy schedule yap
    # İpucu: En çok olan task'tan başla, cooldown'u manage et
    pass


# ============== SOLUTION ==============
def least_interval_solution(tasks: List[str], n: int) -> int:
    """
    Minimum time to complete tasks with cooldown - O(N log K)

    Algorithm:
    1. Count task frequencies
    2. Use max heap to always schedule most frequent task
    3. Track cooldown with queue
    """
    # Task frequency'lerini say
    task_counts = Counter(tasks)

    # Max heap (frequency'ye göre)
    max_heap = [-count for count in task_counts.values()]
    heapq.heapify(max_heap)

    time = 0
    # Cooldown'dan çıkacak task'ları tut: (time_available, frequency)
    cooldown_queue = deque()

    while max_heap or cooldown_queue:
        time += 1

        if max_heap:
            # En sık task'ı al
            count = heapq.heappop(max_heap)
            count += 1  # Bir tane execute ettik (negative olduğu için +1)

            # Hala kalan varsa cooldown'a koy
            if count < 0:
                cooldown_queue.append((time + n, count))

        # Cooldown'dan çıkanları heap'e geri koy
        if cooldown_queue and cooldown_queue[0][0] == time:
            _, count = cooldown_queue.popleft()
            heapq.heappush(max_heap, count)

    return time


# Test
def test_task_scheduler():
    assert least_interval_solution(["A","A","A","B","B","B"], 2) == 8
    assert least_interval_solution(["A","A","A","B","B","B"], 0) == 6
    assert least_interval_solution(["A","A","A","A","A","A","B","C","D","E","F","G"], 2) == 16
    print("✓ Task Scheduler tests passed")


# ============================================================================
# EXERCISE 8: Implement Trie with Word Search (Medium)
# ============================================================================
"""
Trie implemente edin ve wildcard (.) destekleyen search fonksiyonu ekleyin.

Örnek:
addWord("bad")
addWord("dad")
addWord("mad")
search("pad") -> False
search("bad") -> True
search(".ad") -> True
search("b..") -> True
"""

class WordDictionary:
    def __init__(self):
        # TODO: Trie node yapısı oluşturun
        pass

    def addWord(self, word: str) -> None:
        # TODO: Kelime ekle - O(M)
        pass

    def search(self, word: str) -> bool:
        # TODO: Wildcard destekli arama (DFS) - O(M * 26^K) K=wildcards
        pass


# ============== SOLUTION ==============
class TrieNodeDict:
    def __init__(self):
        self.children: Dict[str, 'TrieNodeDict'] = {}
        self.is_word = False

class WordDictionarySolution:
    """Trie with wildcard search using DFS"""

    def __init__(self):
        self.root = TrieNodeDict()

    def addWord(self, word: str) -> None:
        """Kelime ekle - O(M)"""
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNodeDict()
            node = node.children[char]
        node.is_word = True

    def search(self, word: str) -> bool:
        """Wildcard destekli arama - O(M * 26^K)"""
        def dfs(node: TrieNodeDict, index: int) -> bool:
            if index == len(word):
                return node.is_word

            char = word[index]

            if char == '.':
                # Wildcard: tüm child'ları dene
                for child in node.children.values():
                    if dfs(child, index + 1):
                        return True
                return False
            else:
                # Normal karakter
                if char not in node.children:
                    return False
                return dfs(node.children[char], index + 1)

        return dfs(self.root, 0)


# Test
def test_word_dictionary():
    wd = WordDictionarySolution()
    wd.addWord("bad")
    wd.addWord("dad")
    wd.addWord("mad")
    assert wd.search("pad") == False
    assert wd.search("bad") == True
    assert wd.search(".ad") == True
    assert wd.search("b..") == True
    assert wd.search("...") == True
    assert wd.search("....") == False
    print("✓ Word Dictionary tests passed")


# ============================================================================
# EXERCISE 9: Graph - Course Schedule II (Medium-Hard)
# ============================================================================
"""
Course schedule II: Topological sort ile ders sırasını bulun.

Örnek:
Input: numCourses = 4, prerequisites = [[1,0],[2,0],[3,1],[3,2]]
Output: [0,2,1,3] veya [0,1,2,3]

Gereksinimler:
- DFS veya BFS ile topological sort - O(V + E)
- Cycle detection
"""

def find_order(numCourses: int, prerequisites: List[List[int]]) -> List[int]:
    # TODO: Topological sort ile ders sırasını bulun
    # İpucu: BFS (Kahn's algorithm) veya DFS kullanın
    pass


# ============== SOLUTION ==============
def find_order_solution(numCourses: int, prerequisites: List[List[int]]) -> List[int]:
    """
    Topological sort with BFS (Kahn's algorithm) - O(V + E)

    Algorithm:
    1. Build adjacency list and in-degree count
    2. Start with courses having no prerequisites (in-degree = 0)
    3. Process courses and reduce in-degree of dependents
    4. If all courses processed, return order; else return [] (cycle exists)
    """
    # Graph oluştur
    graph = defaultdict(list)
    in_degree = [0] * numCourses

    for course, prereq in prerequisites:
        graph[prereq].append(course)
        in_degree[course] += 1

    # In-degree 0 olan course'ları queue'ya ekle
    queue = deque([i for i in range(numCourses) if in_degree[i] == 0])
    order = []

    while queue:
        course = queue.popleft()
        order.append(course)

        # Bu course'u alan diğer course'ların in-degree'sini azalt
        for next_course in graph[course]:
            in_degree[next_course] -= 1
            if in_degree[next_course] == 0:
                queue.append(next_course)

    # Tüm course'lar işlendiyse geçerli sıralama
    return order if len(order) == numCourses else []


# Test
def test_course_schedule():
    result = find_order_solution(4, [[1,0],[2,0],[3,1],[3,2]])
    assert len(result) == 4
    assert result.index(0) < result.index(1)
    assert result.index(0) < result.index(2)
    assert result.index(1) < result.index(3)
    assert result.index(2) < result.index(3)

    result = find_order_solution(2, [[1,0]])
    assert result == [0, 1]

    result = find_order_solution(2, [[1,0],[0,1]])
    assert result == []  # Cycle

    print("✓ Course Schedule tests passed")


# ============================================================================
# EXERCISE 10: Design HashMap (Medium)
# ============================================================================
"""
Custom HashMap implemente edin (Python dict kullanmadan).
Separate chaining ile collision handling.

Gereksinimler:
- put(key, value): O(1) average
- get(key): O(1) average
- remove(key): O(1) average
- Load factor > 0.75 olunca resize
"""

class MyHashMap:
    def __init__(self):
        # TODO: Bucket array ve hash function tasarlayın
        # İpucu: Linked list veya list of lists
        pass

    def put(self, key: int, value: int) -> None:
        # TODO: Key-value pair ekle/güncelle
        pass

    def get(self, key: int) -> int:
        # TODO: Value getir veya -1
        pass

    def remove(self, key: int) -> None:
        # TODO: Key-value pair sil
        pass


# ============== SOLUTION ==============
class MyHashMapSolution:
    """
    Custom HashMap with separate chaining

    Time Complexity: O(1) average for all operations
    Space Complexity: O(N + M) where N is entries, M is buckets
    """

    def __init__(self, capacity: int = 16):
        self.capacity = capacity
        self.size = 0
        self.buckets: List[List[Tuple[int, int]]] = [[] for _ in range(capacity)]
        self.load_factor = 0.75

    def _hash(self, key: int) -> int:
        """Hash function"""
        return key % self.capacity

    def _resize(self) -> None:
        """Capacity'yi iki katına çıkar ve rehash yap"""
        old_buckets = self.buckets
        self.capacity *= 2
        self.buckets = [[] for _ in range(self.capacity)]
        self.size = 0

        # Tüm elemanları yeniden ekle
        for bucket in old_buckets:
            for key, value in bucket:
                self.put(key, value)

    def put(self, key: int, value: int) -> None:
        """Key-value pair ekle/güncelle - O(1) average"""
        index = self._hash(key)
        bucket = self.buckets[index]

        # Key zaten varsa güncelle
        for i, (k, v) in enumerate(bucket):
            if k == key:
                bucket[i] = (key, value)
                return

        # Yeni key ekle
        bucket.append((key, value))
        self.size += 1

        # Load factor kontrolü
        if self.size / self.capacity > self.load_factor:
            self._resize()

    def get(self, key: int) -> int:
        """Value getir - O(1) average"""
        index = self._hash(key)
        bucket = self.buckets[index]

        for k, v in bucket:
            if k == key:
                return v

        return -1

    def remove(self, key: int) -> None:
        """Key-value pair sil - O(1) average"""
        index = self._hash(key)
        bucket = self.buckets[index]

        for i, (k, v) in enumerate(bucket):
            if k == key:
                bucket.pop(i)
                self.size -= 1
                return


# Test
def test_hashmap():
    hashmap = MyHashMapSolution()
    hashmap.put(1, 1)
    hashmap.put(2, 2)
    assert hashmap.get(1) == 1
    assert hashmap.get(3) == -1
    hashmap.put(2, 1)
    assert hashmap.get(2) == 1
    hashmap.remove(2)
    assert hashmap.get(2) == -1
    print("✓ HashMap tests passed")


# ============================================================================
# EXERCISE 11: Design Circular Queue (Medium)
# ============================================================================
"""
Circular queue (ring buffer) implemente edin.

Gereksinimler:
- enQueue(value): O(1)
- deQueue(): O(1)
- Front(): O(1)
- Rear(): O(1)
- isFull(): O(1)
- isEmpty(): O(1)
"""

class MyCircularQueue:
    def __init__(self, k: int):
        # TODO: Fixed size array ile circular queue tasarlayın
        # İpucu: head, tail, size tracker'lar kullanın
        pass

    def enQueue(self, value: int) -> bool:
        # TODO: Queue'ya ekle
        pass

    def deQueue(self) -> bool:
        # TODO: Queue'dan çıkar
        pass

    def Front(self) -> int:
        # TODO: İlk elemanı döndür
        pass

    def Rear(self) -> int:
        # TODO: Son elemanı döndür
        pass

    def isEmpty(self) -> bool:
        # TODO: Boş mu?
        pass

    def isFull(self) -> bool:
        # TODO: Dolu mu?
        pass


# ============== SOLUTION ==============
class MyCircularQueueSolution:
    """
    Circular Queue with fixed-size array
    All operations: O(1) time, O(K) space
    """

    def __init__(self, k: int):
        self.queue = [0] * k
        self.capacity = k
        self.head = 0
        self.tail = 0
        self.size = 0

    def enQueue(self, value: int) -> bool:
        """Queue'ya ekle - O(1)"""
        if self.isFull():
            return False

        self.queue[self.tail] = value
        self.tail = (self.tail + 1) % self.capacity
        self.size += 1
        return True

    def deQueue(self) -> bool:
        """Queue'dan çıkar - O(1)"""
        if self.isEmpty():
            return False

        self.head = (self.head + 1) % self.capacity
        self.size -= 1
        return True

    def Front(self) -> int:
        """İlk eleman - O(1)"""
        return -1 if self.isEmpty() else self.queue[self.head]

    def Rear(self) -> int:
        """Son eleman - O(1)"""
        return -1 if self.isEmpty() else self.queue[(self.tail - 1) % self.capacity]

    def isEmpty(self) -> bool:
        """Boş mu - O(1)"""
        return self.size == 0

    def isFull(self) -> bool:
        """Dolu mu - O(1)"""
        return self.size == self.capacity


# Test
def test_circular_queue():
    queue = MyCircularQueueSolution(3)
    assert queue.enQueue(1) == True
    assert queue.enQueue(2) == True
    assert queue.enQueue(3) == True
    assert queue.enQueue(4) == False
    assert queue.Rear() == 3
    assert queue.isFull() == True
    assert queue.deQueue() == True
    assert queue.enQueue(4) == True
    assert queue.Rear() == 4
    print("✓ Circular Queue tests passed")


# ============================================================================
# EXERCISE 12: Median of Data Stream (Hard)
# ============================================================================
"""
Sürekli gelen sayıların median'ını O(log n) ile hesaplayan data structure.
İki heap (max heap + min heap) kullanın.

Gereksinimler:
- addNum(num): O(log n)
- findMedian(): O(1)
"""

class MedianFinder:
    def __init__(self):
        # TODO: İki heap tasarlayın
        # İpucu: small (max heap), large (min heap)
        pass

    def addNum(self, num: int) -> None:
        # TODO: Sayı ekle ve heap'leri balance et
        pass

    def findMedian(self) -> float:
        # TODO: Median hesapla
        pass


# ============== SOLUTION ==============
class MedianFinderSolution:
    """
    Median finder with two heaps

    Time Complexity:
    - addNum: O(log n)
    - findMedian: O(1)

    Space Complexity: O(n)
    """

    def __init__(self):
        # Max heap for lower half (negated values)
        self.small = []
        # Min heap for upper half
        self.large = []

    def addNum(self, num: int) -> None:
        """Sayı ekle ve heap'leri balance et - O(log n)"""
        # Her zaman önce small'a ekle
        heapq.heappush(self.small, -num)

        # Balance: small'ın max'ı large'ın min'inden büyükse taşı
        if self.small and self.large and (-self.small[0] > self.large[0]):
            val = -heapq.heappop(self.small)
            heapq.heappush(self.large, val)

        # Size balance: small en fazla large + 1 olabilir
        if len(self.small) > len(self.large) + 1:
            val = -heapq.heappop(self.small)
            heapq.heappush(self.large, val)

        if len(self.large) > len(self.small):
            val = heapq.heappop(self.large)
            heapq.heappush(self.small, -val)

    def findMedian(self) -> float:
        """Median hesapla - O(1)"""
        if len(self.small) > len(self.large):
            return -self.small[0]
        return (-self.small[0] + self.large[0]) / 2.0


# Test
def test_median_finder():
    mf = MedianFinderSolution()
    mf.addNum(1)
    mf.addNum(2)
    assert mf.findMedian() == 1.5
    mf.addNum(3)
    assert mf.findMedian() == 2.0
    mf.addNum(4)
    mf.addNum(5)
    assert mf.findMedian() == 3.0
    print("✓ Median Finder tests passed")


# ============================================================================
# EXERCISE 13: Design Skip List (Expert)
# ============================================================================
"""
Skip list implemente edin (probabilistic balanced tree alternative).

Gereksinimler:
- search(target): O(log n) average
- add(num): O(log n) average
- erase(num): O(log n) average
"""

class Skiplist:
    def __init__(self):
        # TODO: Multi-level linked list tasarlayın
        # İpucu: Random level assignment for balancing
        pass

    def search(self, target: int) -> bool:
        # TODO: Logarithmic search
        pass

    def add(self, num: int) -> None:
        # TODO: Add with random level
        pass

    def erase(self, num: int) -> bool:
        # TODO: Remove from all levels
        pass


# ============== SOLUTION ==============
import random

class SkipListNode:
    def __init__(self, val: int, level: int):
        self.val = val
        self.next = [None] * (level + 1)

class SkiplistSolution:
    """
    Skip List: Probabilistic balanced search structure

    Average Time Complexity: O(log n) for all operations
    Space Complexity: O(n log n) average
    """

    def __init__(self):
        self.max_level = 16
        self.p = 0.5  # Probability for level promotion
        self.head = SkipListNode(-1, self.max_level)
        self.level = 0

    def _random_level(self) -> int:
        """Random level for new node"""
        level = 0
        while random.random() < self.p and level < self.max_level:
            level += 1
        return level

    def search(self, target: int) -> bool:
        """Search for target - O(log n) average"""
        current = self.head

        # Top-down, left-right search
        for i in range(self.level, -1, -1):
            while current.next[i] and current.next[i].val < target:
                current = current.next[i]

        current = current.next[0]
        return current is not None and current.val == target

    def add(self, num: int) -> None:
        """Add number - O(log n) average"""
        update = [None] * (self.max_level + 1)
        current = self.head

        # Find insertion position at each level
        for i in range(self.level, -1, -1):
            while current.next[i] and current.next[i].val < num:
                current = current.next[i]
            update[i] = current

        # Create new node with random level
        new_level = self._random_level()
        if new_level > self.level:
            for i in range(self.level + 1, new_level + 1):
                update[i] = self.head
            self.level = new_level

        new_node = SkipListNode(num, new_level)

        # Insert at each level
        for i in range(new_level + 1):
            new_node.next[i] = update[i].next[i]
            update[i].next[i] = new_node

    def erase(self, num: int) -> bool:
        """Remove number - O(log n) average"""
        update = [None] * (self.max_level + 1)
        current = self.head

        # Find node to delete
        for i in range(self.level, -1, -1):
            while current.next[i] and current.next[i].val < num:
                current = current.next[i]
            update[i] = current

        current = current.next[0]

        if current is None or current.val != num:
            return False

        # Remove from each level
        for i in range(self.level + 1):
            if update[i].next[i] != current:
                break
            update[i].next[i] = current.next[i]

        # Update max level
        while self.level > 0 and self.head.next[self.level] is None:
            self.level -= 1

        return True


# Test
def test_skiplist():
    skiplist = SkiplistSolution()
    skiplist.add(1)
    skiplist.add(2)
    skiplist.add(3)
    assert skiplist.search(0) == False
    skiplist.add(4)
    assert skiplist.search(1) == True
    assert skiplist.erase(0) == False
    assert skiplist.erase(1) == True
    assert skiplist.search(1) == False
    print("✓ Skiplist tests passed")


# ============================================================================
# EXERCISE 14: Design Browser History (Medium)
# ============================================================================
"""
Browser history sistemi tasarlayın (back/forward navigation).

Gereksinimler:
- visit(url): O(1)
- back(steps): O(min(steps, history_size))
- forward(steps): O(min(steps, future_size))
"""

class BrowserHistory:
    def __init__(self, homepage: str):
        # TODO: History tracking tasarlayın
        # İpucu: List veya doubly linked list
        pass

    def visit(self, url: str) -> None:
        # TODO: Yeni URL'e git (forward history silinir)
        pass

    def back(self, steps: int) -> str:
        # TODO: steps kadar geri git
        pass

    def forward(self, steps: int) -> str:
        # TODO: steps kadar ileri git
        pass


# ============== SOLUTION ==============
class BrowserHistorySolution:
    """Browser history with list-based navigation"""

    def __init__(self, homepage: str):
        self.history = [homepage]
        self.current = 0

    def visit(self, url: str) -> None:
        """Yeni URL'e git - O(1)"""
        # Forward history'yi sil
        self.current += 1
        self.history = self.history[:self.current]
        self.history.append(url)

    def back(self, steps: int) -> str:
        """Geri git - O(1)"""
        self.current = max(0, self.current - steps)
        return self.history[self.current]

    def forward(self, steps: int) -> str:
        """İleri git - O(1)"""
        self.current = min(len(self.history) - 1, self.current + steps)
        return self.history[self.current]


# Test
def test_browser_history():
    browser = BrowserHistorySolution("leetcode.com")
    browser.visit("google.com")
    browser.visit("facebook.com")
    browser.visit("youtube.com")
    assert browser.back(1) == "facebook.com"
    assert browser.back(1) == "google.com"
    assert browser.forward(1) == "facebook.com"
    browser.visit("linkedin.com")
    assert browser.forward(2) == "linkedin.com"
    assert browser.back(2) == "google.com"
    assert browser.back(7) == "leetcode.com"
    print("✓ Browser History tests passed")


# ============================================================================
# EXERCISE 15: Design Underground System (Medium-Hard)
# ============================================================================
"""
Metro card sistemi tasarlayın:
- checkIn(id, stationName, t): Yolcu giriş yapar
- checkOut(id, stationName, t): Yolcu çıkış yapar
- getAverageTime(startStation, endStation): Ortalama süre

Gereksinimler:
- Tüm operasyonlar O(1)
- defaultdict kullanın
"""

class UndergroundSystem:
    def __init__(self):
        # TODO: Check-in/check-out tracking + route statistics
        pass

    def checkIn(self, id: int, stationName: str, t: int) -> None:
        # TODO: Yolcu giriş
        pass

    def checkOut(self, id: int, stationName: str, t: int) -> None:
        # TODO: Yolcu çıkış ve istatistik güncelle
        pass

    def getAverageTime(self, startStation: str, endStation: str) -> float:
        # TODO: Ortalama süre hesapla
        pass


# ============== SOLUTION ==============
class UndergroundSystemSolution:
    """
    Metro system with O(1) operations

    Data structures:
    - check_ins: id -> (station, time)
    - routes: (start, end) -> (total_time, count)
    """

    def __init__(self):
        self.check_ins: Dict[int, Tuple[str, int]] = {}
        # (start_station, end_station) -> (total_time, trip_count)
        self.routes: Dict[Tuple[str, str], List[int]] = defaultdict(lambda: [0, 0])

    def checkIn(self, id: int, stationName: str, t: int) -> None:
        """Check-in - O(1)"""
        self.check_ins[id] = (stationName, t)

    def checkOut(self, id: int, stationName: str, t: int) -> None:
        """Check-out ve route istatistiği güncelle - O(1)"""
        start_station, start_time = self.check_ins[id]
        route = (start_station, stationName)

        travel_time = t - start_time
        self.routes[route][0] += travel_time
        self.routes[route][1] += 1

        del self.check_ins[id]

    def getAverageTime(self, startStation: str, endStation: str) -> float:
        """Ortalama süre - O(1)"""
        total_time, count = self.routes[(startStation, endStation)]
        return total_time / count


# Test
def test_underground_system():
    system = UndergroundSystemSolution()
    system.checkIn(45, "Leyton", 3)
    system.checkIn(32, "Paradise", 8)
    system.checkIn(27, "Leyton", 10)
    system.checkOut(45, "Waterloo", 15)
    system.checkOut(27, "Waterloo", 20)
    system.checkOut(32, "Cambridge", 22)
    assert system.getAverageTime("Paradise", "Cambridge") == 14.0
    assert system.getAverageTime("Leyton", "Waterloo") == 11.0
    print("✓ Underground System tests passed")


# ============================================================================
# RUN ALL TESTS
# ============================================================================

def run_all_tests():
    """Tüm testleri çalıştır"""
    print("\n" + "="*80)
    print("ADVANCED DATA STRUCTURES - EXERCISE TESTS")
    print("="*80 + "\n")

    tests = [
        ("TimeMap", test_timemap),
        ("LFU Cache", test_lfu_cache),
        ("Twitter Feed", test_twitter),
        ("Autocomplete", test_autocomplete),
        ("Meeting Rooms", test_meeting_rooms),
        ("Sliding Window Maximum", test_sliding_window),
        ("Task Scheduler", test_task_scheduler),
        ("Word Dictionary", test_word_dictionary),
        ("Course Schedule", test_course_schedule),
        ("HashMap", test_hashmap),
        ("Circular Queue", test_circular_queue),
        ("Median Finder", test_median_finder),
        ("Skiplist", test_skiplist),
        ("Browser History", test_browser_history),
        ("Underground System", test_underground_system),
    ]

    passed = 0
    failed = 0

    for name, test_func in tests:
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            print(f"✗ {name} failed: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ {name} error: {e}")
            failed += 1

    print("\n" + "="*80)
    print(f"RESULTS: {passed} passed, {failed} failed out of {len(tests)} tests")
    print("="*80 + "\n")


if __name__ == "__main__":
    run_all_tests()
