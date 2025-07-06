import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import time

# Set your base website URL here
BASE_URL = "https://growvy-website.vercel.app/"

# To avoid infinite loops or external links
visited = set()

def is_valid_url(url):
    parsed = urlparse(url)
    return parsed.netloc == urlparse(BASE_URL).netloc

def get_text_from_page(url):
    try:
        res = requests.get(url)
        soup = BeautifulSoup(res.text, "html.parser")

        # Remove scripts, styles, etc.
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()

        # Extract visible text
        text = soup.get_text(separator="\n")
        text = "\n".join([line.strip() for line in text.splitlines() if line.strip()])
        return text
    except Exception as e:
        print(f"[ERROR] Failed to load {url}: {e}")
        return ""

def crawl(url):
    if url in visited or not is_valid_url(url):
        return ""

    print(f"[INFO] Crawling: {url}")
    visited.add(url)
    page_text = get_text_from_page(url)

    try:
        res = requests.get(url)
        soup = BeautifulSoup(res.text, "html.parser")
        links = [urljoin(url, link.get("href")) for link in soup.find_all("a", href=True)]
    except:
        links = []

    # Recursively crawl inner pages
    for link in links:
        time.sleep(1)  # be nice to the server
        page_text += "\n" + crawl(link)

    return page_text

if __name__ == "__main__":
    final_text = crawl(BASE_URL)

    with open("website_content.txt", "w", encoding="utf-8") as f:
        f.write(final_text)

    print("[✅] Website content saved to 'website_content.txt'")
