import asyncio
import aiohttp
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import re
import pandas as pd
from pathlib import Path
from datetime import datetime
import logging
import hashlib
import sys
import random

if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

START_URLS = [
    'https://en.wikipedia.org/wiki/Python_(programming_language)',
    'https://en.wikipedia.org/wiki/Web_scraping',
    'https://en.wikipedia.org/wiki/Data_mining',
    'https://en.wikipedia.org/wiki/Artificial_intelligence',
    'https://en.wikipedia.org/wiki/Machine_learning',
    'https://en.wikipedia.org/wiki/Cybersecurity',
    'https://en.wikipedia.org/wiki/Blockchain',
    'https://en.wikipedia.org/wiki/Cloud_computing',
    'https://en.wikipedia.org/wiki/Internet_of_things',

    'https://en.wikipedia.org/wiki/Physics',
    'https://en.wikipedia.org/wiki/Chemistry',
    'https://en.wikipedia.org/wiki/Biology',
    'https://en.wikipedia.org/wiki/Astronomy',
    'https://en.wikipedia.org/wiki/Geology',
    'https://en.wikipedia.org/wiki/Quantum_mechanics',
    'https://en.wikipedia.org/wiki/Ecology',
    'https://en.wikipedia.org/wiki/Climate_change',

    'https://en.wikipedia.org/wiki/Medicine',
    'https://en.wikipedia.org/wiki/Neuroscience',
    'https://en.wikipedia.org/wiki/Genetics',
    'https://en.wikipedia.org/wiki/Public_health',

    'https://en.wikipedia.org/wiki/History',
    'https://en.wikipedia.org/wiki/Ancient_history',
    'https://en.wikipedia.org/wiki/Renaissance',
    'https://en.wikipedia.org/wiki/World_War_II',
    'https://en.wikipedia.org/wiki/Cold_War',
    'https://en.wikipedia.org/wiki/Philosophy',
    'https://en.wikipedia.org/wiki/Religion',
    'https://en.wikipedia.org/wiki/Mythology',

    'https://en.wikipedia.org/wiki/Art',
    'https://en.wikipedia.org/wiki/Music',
    'https://en.wikipedia.org/wiki/Literature',
    'https://en.wikipedia.org/wiki/Film',
    'https://en.wikipedia.org/wiki/Theatre',
    'https://en.wikipedia.org/wiki/Architecture',

    'https://en.wikipedia.org/wiki/Psychology',
    'https://en.wikipedia.org/wiki/Sociology',
    'https://en.wikipedia.org/wiki/Economics',
    'https://en.wikipedia.org/wiki/Political_science',
    'https://en.wikipedia.org/wiki/Linguistics',
    'https://en.wikipedia.org/wiki/Law',
    'https://en.wikipedia.org/wiki/Anthropology',

    'https://en.wikipedia.org/wiki/Geography',
    'https://en.wikipedia.org/wiki/Asia',
    'https://en.wikipedia.org/wiki/Africa',
    'https://en.wikipedia.org/wiki/Europe',
    'https://en.wikipedia.org/wiki/North_America',
    'https://en.wikipedia.org/wiki/South_America',
    'https://en.wikipedia.org/wiki/Oceania',
    'https://en.wikipedia.org/wiki/Environmentalism',

    'https://en.wikipedia.org/wiki/Engineering',
    'https://en.wikipedia.org/wiki/Agriculture',
    'https://en.wikipedia.org/wiki/Space_exploration',
    'https://en.wikipedia.org/wiki/Energy',
    'https://en.wikipedia.org/wiki/Transportation',

    'https://en.wikipedia.org/wiki/Olympic_Games',
    'https://en.wikipedia.org/wiki/National_Basketball_Association',
    'https://en.wikipedia.org/wiki/Skiing',
    'https://en.wikipedia.org/wiki/Football',
    'https://en.wikipedia.org/wiki/Business',
    'https://en.wikipedia.org/wiki/Finance',
    'https://en.wikipedia.org/wiki/Education',
    'https://en.wikipedia.org/wiki/Social_media',
    'https://en.wikipedia.org/wiki/Artificial_general_intelligence'
]





class AsyncWikiCrawler:
    """
    Asynchronous Wikipedia crawler with partitioned start URLs.
    """
    
    def __init__(self, start_urls, output_dir, target_count=50, max_depth=3, 
                 concurrency_limit=5, worker_target_count=20):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.target_count = target_count
        self.max_depth = max_depth
        self.concurrency_limit = concurrency_limit
        self.worker_target_count = worker_target_count

        self.link_allow_regex = re.compile(r'^/wiki/[^:]+$')
        self.link_deny_regex = re.compile(r'/wiki/Main_Page')
        self.allowed_domain = 'en.wikipedia.org'

        # Shuffle and partition start URLs among workers
        shuffled_urls = random.sample(start_urls, len(start_urls))
        self.worker_url_partitions = self._partition_urls(
            shuffled_urls, 
            concurrency_limit
        )
        
        self.session = None
        self.semaphore = asyncio.Semaphore(self.concurrency_limit)
    
        self.results = []
        self.seen_content_hashes = set()
        self.lock = asyncio.Lock()
        
        self.stop_crawling = asyncio.Event()
        self.stats = {
            'total_requests': 0,
            'failed_requests': 0,
            'items_collected': 0,
            'duplicate_content': 0,
            'parse_failures': 0,
            'urls_discovered': 0,
        }
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def _partition_urls(self, urls, num_partitions):
        """Divide URLs into N roughly equal partitions."""
        partitions = [[] for _ in range(num_partitions)]
        for i, url in enumerate(urls):
            partitions[i % num_partitions].append(url)
        return partitions

    async def run(self):
        """Main entry point to start the crawl."""
        self.logger.info(
            f"Starting crawl: Target={self.target_count}, "
            f"Depth={self.max_depth}, Workers={self.concurrency_limit}"
        )
        
        for i, partition in enumerate(self.worker_url_partitions):
            self.logger.info(f"Worker {i} assigned {len(partition)} start URLs")
        
        timeout = aiohttp.ClientTimeout(total=30, connect=10)
        connector = aiohttp.TCPConnector(limit_per_host=self.concurrency_limit)
        
        async with aiohttp.ClientSession(
            headers={'User-Agent': 'AsyncWikiCrawler/2.0 (Educational)'},
            timeout=timeout,
            connector=connector
        ) as session:
            self.session = session
            
            # Create worker tasks with their assigned URL partitions
            tasks = [
                asyncio.create_task(
                    self.worker(
                        worker_id=i, 
                        assigned_urls=self.worker_url_partitions[i]
                    )
                ) 
                for i in range(self.concurrency_limit)
            ]
            
            # Wait for all workers to finish
            await asyncio.gather(*tasks, return_exceptions=True)
            
        self.logger.info("Crawl finished. Saving data...")
        self._save_data()
        self.logger.info(f"Final Stats: {self.stats}")

    async def worker(self, worker_id, assigned_urls):
        """Worker task that processes its assigned start URLs."""
        self.logger.info(
            f"Worker {worker_id} starting with {len(assigned_urls)} URLs"
        )
        
        # Each worker has its own seen_urls set
        worker_seen_urls = set()
        
        for start_url in assigned_urls:
            # Check for stopping condition
            if self.stop_crawling.is_set():
                self.logger.info(
                    f"Worker {worker_id}: Stop signal received. Exiting."
                )
                break
            
            async with self.lock:
                if len(self.results) >= self.target_count:
                    self.logger.info(
                        f"Worker {worker_id}: Global target reached. Exiting."
                    )
                    self.stop_crawling.set()
                    break
            
            self.logger.info(
                f"Worker {worker_id} starting sub-crawl from: {start_url}"
            )
            
            await self._process_subcrawl(worker_id, start_url, worker_seen_urls)
        
        self.logger.info(
            f"Worker {worker_id} completed all assigned URLs "
            f"(visited {len(worker_seen_urls)} unique URLs)"
        )

    async def _process_subcrawl(self, worker_id, start_url, worker_seen_urls):
        """Process a single sub-crawl from a start URL."""
        local_queue = asyncio.Queue()
        worker_seen_urls.add(start_url)
        await local_queue.put((start_url, 0))

        worker_scraped_count = 0
        
        while (worker_scraped_count < self.worker_target_count and 
               not self.stop_crawling.is_set()):
            
            try:
                url, depth = await asyncio.wait_for(
                    local_queue.get(), 
                    timeout=0.5
                )
            except asyncio.TimeoutError:
                if local_queue.empty():
                    break
                continue

            async with self.lock:
                if len(self.results) >= self.target_count:
                    self.stop_crawling.set()
                    break

            if depth > self.max_depth:
                self.logger.debug(f"Skipping {url} - exceeded max depth")
                continue

            html_content = await self.fetch(url)
            
            if html_content:
                scraped = await self.parse(
                    html_content, url, depth, local_queue, worker_seen_urls
                )
                if scraped:
                    worker_scraped_count += 1

        self.logger.info(
            f"Worker {worker_id} finished sub-crawl from {start_url} "
            f"(scraped {worker_scraped_count} pages)"
        )

    async def fetch(self, url):
        """Fetches a single URL using aiohttp with rate limiting."""
        async with self.semaphore:
            async with self.lock:
                self.stats['total_requests'] += 1
            
            try:
                await asyncio.sleep(random.uniform(0.5, 1.0))
                
                async with self.session.get(url) as response:
                    if response.status == 200:
                        content_type = response.headers.get('Content-Type', '')
                        if 'text/html' in content_type:
                            return await response.text()
                    
                    self.logger.warning(
                        f"Failed fetch: {url} (Status: {response.status})"
                    )
                    async with self.lock:
                        self.stats['failed_requests'] += 1
                    return None
                    
            except asyncio.TimeoutError:
                self.logger.error(f"Timeout fetching: {url}")
                async with self.lock:
                    self.stats['failed_requests'] += 1
                return None
            except Exception as e:
                self.logger.error(f"Exception fetching {url}: {e}")
                async with self.lock:
                    self.stats['failed_requests'] += 1
                return None

    async def parse(self, html_content, url, depth, local_queue, worker_seen_urls):
        """
        Parses HTML content, extracts data, and discovers links.
        Returns True if an item was collected.
        """
        try:
            soup = BeautifulSoup(html_content, 'lxml')
            
            # Extract data
            item = self._extract_data(soup, url)
            item_collected = False
            
            if item:
                async with self.lock:
                    if (item['content_hash'] not in self.seen_content_hashes and 
                        len(self.results) < self.target_count):
                        
                        self.seen_content_hashes.add(item['content_hash'])
                        self.results.append(item)
                        self.stats['items_collected'] += 1
                        item_collected = True
                        
                        self.logger.info(
                            f"Scraped ({self.stats['items_collected']}/"
                            f"{self.target_count}): {item['title']} "
                            f"({item['word_count']} words)"
                        )
                        
                        if len(self.results) >= self.target_count:
                            self.stop_crawling.set()
                    elif item['content_hash'] in self.seen_content_hashes:
                        self.stats['duplicate_content'] += 1
            
            if (not self.stop_crawling.is_set() and 
                depth < self.max_depth):
                await self._find_links(soup, url, depth, local_queue, worker_seen_urls)
            
            return item_collected
            
        except Exception as e:
            self.logger.error(f"Failed to parse {url}: {e}")
            async with self.lock:
                self.stats['parse_failures'] += 1
            return False

    def _extract_data(self, soup, url):
        """Extract article data from parsed HTML."""
        try:
            # Extract title
            title_tag = soup.find('h1', id='firstHeading')
            if not title_tag:
                return None
            title = title_tag.get_text().strip()
            
            # Extract text from paragraphs
            text_parts = []
            content_div = soup.find('div', class_='mw-parser-output')
            if content_div:
                for p in content_div.find_all('p', recursive=False):
                    text = p.get_text().strip()
                    if text:
                        text_parts.append(text)
            
            full_text = ' '.join(text_parts)
            full_text = ' '.join(full_text.split())
            
            # Skip if too short
            if len(full_text) < 200:
                return None
            
            word_count = len(full_text.split())
            char_count = len(full_text)
            
            return {
                'url': url,
                'title': title,
                'text': full_text,
                'domain': 'wikipedia.org',
                'timestamp': datetime.now().isoformat(),
                'word_count': word_count,
                'char_count': char_count,
                'content_hash': hashlib.md5(full_text.encode()).hexdigest(),
            }
            
        except Exception as e:
            self.logger.error(f"Error extracting data from {url}: {e}")
            return None

    async def _find_links(self, soup, base_url, depth, local_queue, worker_seen_urls):
        """Discover and queue new links using worker's private seen set."""
        new_urls = []
        
        for a_tag in soup.find_all('a', href=True):
            href = a_tag['href']
            abs_url = urljoin(base_url, href)
            parsed_url = urlparse(abs_url)
            
            if parsed_url.netloc != self.allowed_domain:
                continue

            if not (self.link_allow_regex.match(parsed_url.path) and 
                    not self.link_deny_regex.match(parsed_url.path)):
                continue
            
            clean_url = abs_url.split('#')[0]
            
            if clean_url not in worker_seen_urls:
                worker_seen_urls.add(clean_url)
                new_urls.append(clean_url)
        
        for clean_url in new_urls:
            await local_queue.put((clean_url, depth + 1))
        
        if new_urls:
            async with self.lock:
                self.stats['urls_discovered'] += len(new_urls)

    def _save_data(self):
        """Save scraped data to disk."""
        if not self.results:
            self.logger.warning("No items to save!")
            return
        
        df = pd.DataFrame(self.results)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        csv_path = self.output_dir / f'wiki_articles_{timestamp}.csv'
        df.to_csv(csv_path, index=False, encoding='utf-8')
        self.logger.info(f"Saved {len(df)} articles to {csv_path}")
