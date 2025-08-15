#!/usr/bin/env python3.12
import requests
from bs4 import BeautifulSoup, Tag
import json
from urllib.parse import urljoin, urlparse
import pdfplumber
import signal
from functools import wraps
from hashlib import md5
import tempfile
import time
import random
from typing import Optional, Set, Dict, List, Callable, Union, TypedDict
from dataclasses import dataclass
from enum import Enum, auto
import logging
from pathlib import Path


class CrawlerTimeoutError(Exception):
    pass


class Record(TypedDict):
    objectID: str
    url: str
    title: str
    content: str
    type: str
    timestamp: int


class SiteType(Enum):
    SPHINX = auto()
    FLASK = auto()
    REPOSITORY = auto()


@dataclass
class CrawlerConfig:
    sites: Dict[str, List[str]]
    repositories: Dict[str, List[str]]
    settings: Dict[str, Union[int, List[str]]]


def timeout(seconds: int = 10, error_message: str = "Function timed out") -> Callable:
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            def handle_timeout(_signum, _frame):
                raise CrawlerTimeoutError(error_message)

            signal.signal(signal.SIGALRM, handle_timeout)
            signal.alarm(seconds)
            try:
                result = func(*args, **kwargs)
            finally:
                signal.alarm(0)
            return result

        return wrapper

    return decorator


def load_config() -> CrawlerConfig:
    """Load and validate configuration"""
    script_dir = Path(__file__).parent
    config_path = script_dir / 'crawler_config.json'

    with config_path.open('r') as f:
        config_data = json.load(f)

    # Validate config structure
    required_sections = {'sites', 'repositories', 'settings'}
    if not required_sections.issubset(config_data.keys()):
        raise ValueError("Invalid config: missing required sections")

    return CrawlerConfig(**config_data)


config = load_config()
logger = logging.getLogger(__name__)


def get_random_user_agent() -> str:
    """Return a random user agent from config"""
    agents: List[str] = config.settings.get('user_agents', [])
    return random.choice(agents) if agents else "Mozilla/5.0"


@timeout(15)
def fetch_url(url: str) -> Optional[requests.Response]:
    """Fetch URL with timeout and retry logic"""
    try:
        time.sleep(config.settings.get('request_delay', 1))
        response = requests.get(
            url,
            timeout=10,
            headers={'User-Agent': get_random_user_agent()},
            allow_redirects=True
        )
        response.raise_for_status()
        return response
    except requests.exceptions.RequestException as e:
        logger.warning(f"Error fetching {url}: {str(e)[:100]}")
        return None


def clean_text(element: Optional[Union[str, Tag, BeautifulSoup]]) -> str:
    """Extract and clean text from HTML element"""
    if element is None:
        return ""
    if isinstance(element, str):
        return element.strip()
    if isinstance(element, (Tag, BeautifulSoup)):
        return ' '.join(element.stripped_strings)
    return str(element).strip()


def clean_content(text: str, max_length: int = 8000) -> str:
    """Clean and truncate content text"""
    if not text:
        return ""

    text = ' '.join(text.split())
    boilerplate = [
        "Unseen University, 2024",
        "Ty Myrddin Home",
        "Improbability Blog"
    ]

    for phrase in boilerplate:
        text = text.replace(phrase, "")

    if len(text) > max_length:
        return text[:max_length].removesuffix(' ').rsplit(' ', 1)[0] + "..."
    return text


def create_optimized_record(url: str, title: Optional[str], content: str, source_type: str) -> Record:
    """Create a minimized record with essential data"""
    return {
        "objectID": md5(url.encode()).hexdigest(),
        "url": url,
        "title": title[:200] if title else "",
        "content": clean_content(content),
        "type": source_type,
        "timestamp": int(time.time())
    }


def should_crawl(url: str) -> bool:
    """Check if URL matches any of our target domains"""
    parsed = urlparse(url)
    domain = parsed.netloc.lower()

    # Check against all sites
    for category in ['sphinx', 'flask']:
        if domain in [d.lower() for d in config.sites.get(category, [])]:
            return True

    # Check against repositories
    for repo_url in config.repositories.get('users', []) + config.repositories.get('individual_repos', []):
        if urlparse(repo_url).netloc.lower() == domain:
            return True

    return False


@timeout(30)
def extract_pdf_text(pdf_url: str) -> str:
    """Extract text content from PDF files"""
    tmp_file_path: Optional[Path] = None
    text = ""
    response: Optional[requests.Response] = None

    try:
        # Download with retries
        for attempt in range(3):
            try:
                response = requests.get(
                    str(pdf_url),  # Explicitly convert to str
                    stream=True,
                    timeout=10,
                    headers={'User-Agent': get_random_user_agent()}
                )
                response.raise_for_status()
                break
            except requests.exceptions.RequestException:
                if attempt == 2:
                    logger.error(f"Failed PDF download after 3 attempts: {pdf_url}")
                    return ""
                time.sleep(1)
                continue

        if not response:
            return ""

        # Process PDF
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            for chunk in response.iter_content(chunk_size=8192):
                tmp_file.write(chunk)
            tmp_file_path = Path(tmp_file.name)

        with pdfplumber.open(tmp_file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text() or ""  # Ensure str type
                text += page_text + " "

    except requests.exceptions.RequestException:
        logger.error(f"PDF download error for {pdf_url}")
        return ""
    except Exception as e:  # Replace PDFSyntaxError with Exception
        logger.error(f"Invalid PDF syntax for {pdf_url}: {str(e)}")
        return ""
    finally:
        if tmp_file_path and tmp_file_path.exists():
            try:
                tmp_file_path.unlink()
            except OSError:
                pass
    pdf_url = str(pdf_url)  # Explicit conversion ensures str type
    return text.strip()


def get_all_links(soup: BeautifulSoup, current_url: str) -> Set[str]:
    """Extract all crawlable links from page"""
    links: Set[str] = set()
    parsed = urlparse(current_url)
    base_url = f"{parsed.scheme}://{parsed.netloc}"

    for link in soup.find_all('a', href=True):
        href = link['href']

        # Skip non-HTTP links
        if href.startswith(('mailto:', 'tel:', 'javascript:', '#')):
            continue

        # Normalize URL
        if href.startswith('/'):
            absolute_url = f"{base_url}{href}"
        elif href.startswith(('http://', 'https://')):
            absolute_url = href
        else:
            absolute_url = urljoin(current_url, href)

        # Remove fragments/query params
        parsed = urlparse(absolute_url)
        normalized_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"

        if should_crawl(normalized_url):
            links.add(normalized_url)

    return links


@timeout(30)
def scrape_repository(url: str) -> Optional[Record]:
    """Enhanced repository scraper with profile/repo distinction"""
    try:
        response = fetch_url(url)
        if not response:
            return None

        soup = BeautifulSoup(response.text, 'html.parser')
        is_profile = url.count('/') <= 4 and not url.endswith('.git')

        if is_profile:
            # Handle user/organization profiles
            title = clean_text(soup.find('meta', property='og:title') or url.split('/')[-1])
            platform = "GitHub" if "github" in url else "GitLab"

            # Get repositories
            repos = []
            for repo in soup.select('a[itemprop*="codeRepository"], a.project'):
                repo_text = clean_text(repo)
                if repo_text and len(repo_text) < 50:  # Filter out noise
                    repos.append(repo_text)

            return create_optimized_record(
                url=url,
                title=f"{title} ({platform} Profile)",
                content=f"Repositories: {', '.join(repos[:8])}{'...' if len(repos) > 8 else ''}",
                source_type="profile"
            )
        else:
            # Handle individual repositories
            title = clean_text(soup.find('strong', itemprop='name') or
                           soup.find('meta', property='og:title') or
                           url.split('/')[-1])

            description = clean_text(
                soup.find('meta', property='og:description') or
                soup.find('p', itemprop='description') or
                soup.select_one('.repository-meta-content, .project-description')
            )

            # Get additional metadata
            stars = clean_text(soup.select_one('[aria-label="Stars"]'))
            last_commit = clean_text(soup.find('relative-time'))

            return create_optimized_record(
                url=url,
                title=title,
                content=f"{description}\n★ {stars} | Last commit: {last_commit}",
                source_type="repository"
            )

    except (AttributeError, ValueError, requests.exceptions.RequestException) as e:
        logger.error(f"Repository scrape error: {e}")
        return None


def crawl_site(base_url: str, site_type: SiteType) -> List[Record]:
    """Main crawling function with site-type specific logic"""
    visited: Set[str] = set()
    to_visit = [base_url]
    records: List[Record] = []
    page_count = 0
    seen_hashes: Set[str] = set()

    logger.info(f"Crawling {base_url} as {site_type.name} site")

    while to_visit and page_count < config.settings.get('max_pages', 50):
        url = to_visit.pop()

        if url in visited:
            continue

        try:
            page_count += 1
            logger.debug(f"Processing [{page_count}]: {url}")

            # Handle special URL types first
            if any(d in url for d in ["github.com", "gitlab.com"]):
                if record := scrape_repository(url):
                    records.append(record)
                visited.add(url)
                continue

            if url.lower().endswith('.pdf'):
                if content := extract_pdf_text(url):
                    records.append(create_optimized_record(
                        url=url,
                        title=url.split('/')[-1],
                        content=content,
                        source_type="pdf"
                    ))
                visited.add(url)
                continue

            response = fetch_url(url)
            if not response:
                continue

            content_type = response.headers.get('Content-Type', '')
            if 'text/html' not in content_type.lower():
                continue

            soup = BeautifulSoup(response.text, 'html.parser')
            visited.add(url)
            current_url = response.url

            # Add new links to queue
            new_links = get_all_links(soup, current_url)
            to_visit.extend(link for link in new_links if link not in visited and link not in to_visit)

            # Skip content extraction for Sphinx sites (link-only)
            if site_type == SiteType.SPHINX:
                continue

            # Extract content for Flask sites
            title = clean_text(soup.find('h1') or soup.find('title'))
            main_content = (soup.find(class_="document") or
                          soup.find('article') or
                          soup.find('main') or
                          soup)

            content = clean_text(main_content)
            if len(content.split()) < 50:  # Skip low-content pages
                continue

            # Deduplicate content
            content_hash = md5(content.encode()).hexdigest()
            if content_hash in seen_hashes:
                continue
            seen_hashes.add(content_hash)

            records.append(create_optimized_record(
                url=current_url,
                title=title,
                content=content,
                source_type="flask" if site_type == SiteType.FLASK else "html"
            ))

        except CrawlerTimeoutError:
            logger.warning(f"Timeout processing {url}")
        except Exception as e:
            logger.error(f"Processing error: {e}")

    logger.info(f"Finished: {len(records)} records found")
    return records


def main() -> None:
    """Main execution function"""
    logging.basicConfig(level=logging.INFO)
    all_records: List[Record] = []

    # Crawl all configured sites
    for site_type, urls in config.sites.items():
        stype = SiteType.SPHINX if site_type == "sphinx" else SiteType.FLASK
        for url in urls:
            full_url = f"https://{url}/" if not url.startswith('http') else url
            all_records.extend(crawl_site(full_url, stype))

    # Crawl repositories
    for repo_type in ['users', 'individual_repos']:
        for repo_url in config.repositories.get(repo_type, []):
            all_records.extend(crawl_site(repo_url, SiteType.REPOSITORY))

    # Save results
    output_dir = Path(__file__).parent.parent / 'docs' / 'indices'
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "index.json"
    with output_path.open('w', encoding='utf-8') as f:
        json.dump(all_records, f, indent=2, ensure_ascii=False)

    logger.info(f"Success! Saved {len(all_records)} records to {output_path}")


if __name__ == "__main__":
    signal.signal(signal.SIGINT,
                  lambda _s, _f: (logger.info("\nCrawler stopped by user"), exit(1)))
    logger.info("Starting crawler with config:")
    logger.info(json.dumps(config.__dict__, indent=2))
    main()