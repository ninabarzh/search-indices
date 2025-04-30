import requests
from bs4 import BeautifulSoup
import json
from urllib.parse import urljoin, urlparse
import pdfplumber
import signal
from functools import wraps
from hashlib import md5
import tempfile
import os
import time
import random


class CrawlerTimeoutError(Exception):
    pass


def timeout(seconds=10, error_message="Function timed out"):
    def decorator(func):
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


def load_config():
    # Get the directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, 'crawler_config.json')

    with open(config_path, 'r') as f:
        return json.load(f)


config = load_config()


def get_random_user_agent():
    return random.choice(config['settings']['user_agents'])


@timeout(15)
def fetch_url(url):
    try:
        # Add delay between requests
        time.sleep(config['settings']['request_delay'])

        response = requests.get(
            url,
            timeout=10,
            headers={'User-Agent': get_random_user_agent()},
            allow_redirects=True
        )
        response.raise_for_status()
        return response
    except requests.exceptions.RequestException as e:
        print(f"Error fetching {url}: {str(e)[:100]}")
        return None


def clean_text(element):
    """Original function - keep this for HTML element cleaning"""
    if element is None:
        return ""
    return ' '.join(element.stripped_strings)


def clean_content(text, max_length=8000):
    """NEW FUNCTION - optimizes content after clean_text()"""
    if not text:
        return ""

    # First apply the same cleaning as clean_text()
    text = ' '.join(text.split())

    # Then add additional optimizations
    boilerplate = ["Unseen University, 2024, with a forest garden fostered by /ut7.", "Ty Myrddin Home",
                   "Unseen University", "Improbability Blog", "About", "Contact"]
    for phrase in boilerplate:
        text = text.replace(phrase, "")

    if len(text) > max_length:
        text = text[:max_length].rsplit(' ', 1)[0] + "..."

    return text


def create_optimized_record(url, title, content, category):
    """NEW FUNCTION - creates minimized records"""
    return {
        "objectID": md5(url.encode()).hexdigest(),
        "u": url,
        "t": title[:200] if title else "",
        "c": clean_content(content),  # Uses both cleaning functions
        "cat": category,
        "type": "pdf" if url.lower().endswith('.pdf') else "html"
    }


def categorize_url(url):
    parsed = urlparse(url)
    domain = parsed.netloc.lower()

    for category, domains in config['domains'].items():
        if domain in domains:
            return category
    return "uncategorized"


@timeout(30)
def extract_pdf_text(pdf_url):
    tmp_file_path = None
    response = None
    text = ""

    try:
        # Download the PDF with retry logic
        for attempt in range(3):
            try:
                response = requests.get(
                    pdf_url,
                    stream=True,
                    timeout=10,
                    headers={'User-Agent': get_random_user_agent()}
                )
                response.raise_for_status()
                break
            except requests.exceptions.RequestException as e:
                if attempt == 2:
                    print(f"Failed to download PDF after 3 attempts: {pdf_url} - Error: {str(e)}")
                    return ""
                print(f"Retrying PDF download ({attempt + 1}/3) for {pdf_url} - Error: {str(e)}")
                continue

        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            for chunk in response.iter_content(chunk_size=8192):
                tmp_file.write(chunk)
            tmp_file_path = tmp_file.name

        # Process PDF
        with pdfplumber.open(tmp_file_path) as pdf:
            for page in pdf.pages:
                try:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + " "
                except Exception as page_error:
                    print(f"Warning: Error processing page in {pdf_url}: {str(page_error)}")
                    continue

    except Exception as e:
        print(f"Error processing PDF {pdf_url}: {str(e)}")
    finally:
        # Clean up temporary file
        if tmp_file_path and os.path.exists(tmp_file_path):
            try:
                os.unlink(tmp_file_path)
            except Exception as cleanup_error:
                print(f"Warning: Could not delete temp file {tmp_file_path}: {str(cleanup_error)}")

    return text.strip()


def get_all_links(soup, current_url, url_filter_func):
    """Extract links respecting the domain filter"""
    links = set()
    current_parsed = urlparse(current_url)
    base_scheme = current_parsed.scheme
    base_netloc = current_parsed.netloc

    for link in soup.find_all('a', href=True):
        href = link['href']
        if href.startswith(('mailto:', 'tel:', 'javascript:', '#')):
            continue

        # Handle relative URLs
        if href.startswith('/'):
            absolute_url = f"{base_scheme}://{base_netloc}{href}"
        elif href.startswith(('http://', 'https://')):
            absolute_url = href
        else:
            absolute_url = urljoin(current_url, href)

        # Normalize URL by removing fragments and query parameters
        parsed = urlparse(absolute_url)
        normalized_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"

        if url_filter_func(normalized_url):
            links.add(normalized_url)

    # Special handling for Sphinx toctree links
    for div in soup.find_all('div', class_='toctree-wrapper'):
        for link in div.find_all('a', class_='reference external', href=True):
            href = link['href']
            if href.startswith(('http://', 'https://')):
                parsed = urlparse(href)
                normalized_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
                if url_filter_func(normalized_url):
                    links.add(normalized_url)

    return links


def crawl_site(base_url, site_type, max_pages=500, url_filter_func=None):
    visited = set()
    to_visit = [base_url]
    records = []
    page_count = 0
    seen_hashes = set()

    print(f"Starting crawl of {base_url} (max {max_pages} pages)...")

    while to_visit and page_count < max_pages:
        url = to_visit.pop()

        if url in visited:
            continue

        try:
            page_count += 1
            print(f"Processing ({page_count}): {url}")

            if url.lower().endswith('.pdf'):
                content_text = extract_pdf_text(url)
                if content_text:
                    record = create_optimized_record(
                        url=url,
                        title=url.split('/')[-1],
                        content=content_text,
                        category=categorize_url(url)
                    )
                    records.append(record)
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

            new_links = get_all_links(soup, current_url, url_filter_func)
            for link in new_links:
                if link not in visited and link not in to_visit and url_filter_func(link):
                    to_visit.append(link)

            if site_type == "sphinx-links":
                continue

            title_element = soup.find('h1') or soup.find('title')
            content = (soup.find(class_="document") or
                       soup.find('article') or
                       soup.find('main') or
                       soup)

            content_text = clean_text(content)
            if len(content_text.split()) < 50:
                continue

            content_hash = md5(content_text.encode()).hexdigest()
            if content_hash in seen_hashes:
                continue
            seen_hashes.add(content_hash)

            record = create_optimized_record(
                url=current_url,
                title=clean_text(title_element),
                content=content_text,
                category=categorize_url(current_url)
            )
            records.append(record)

        except CrawlerTimeoutError:
            print(f"Timeout processing {url}, skipping...")
        except Exception as e:
            print(f"Error processing {url}: {str(e)[:100]}")

    print(f"Finished crawling {base_url}. Processed {page_count} pages.")
    return records


def main():
    # Initialize indices with all categories
    indices = {category: [] for category in config['domains'].keys()}
    indices["uncategorized"] = []

    # Build all domains list with https:// prefix
    all_domains = []
    for category, domains in config['domains'].items():
        all_domains.extend([f"https://{domain}/" for domain in domains])

    for domain in all_domains:
        category = categorize_url(domain)
        print(f"\n=== Crawling {domain} for {category.upper()} index ===")

        # Determine site type - main domains are Flask, others are Sphinx
        site_type = "sphinx"
        if domain in [f"https://{d}/" for d in config['domains']['main']]:
            site_type = "flask"

        records = crawl_site(
            base_url=domain,
            site_type=site_type,
            max_pages=config['settings']['max_pages'],
            url_filter_func=lambda url: categorize_url(url) == category
        )
        indices[category].extend(records)

    # Save to separate files
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.abspath(os.path.join(script_dir, '../docs/indices'))
    os.makedirs(output_dir, exist_ok=True)

    for index_name, records in indices.items():
        if not records:
            continue

        filename = os.path.join(output_dir, f"{index_name}.json")
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(records, f, indent=2, ensure_ascii=False)
        print(f"Saved {len(records)} records to {filename}")


if __name__ == "__main__":
    signal.signal(signal.SIGINT, lambda _s, _f: (print("\nKeyboard interrupt received"), exit(1)))
    print("Starting crawler (Ctrl+C to interrupt)...")
    main()