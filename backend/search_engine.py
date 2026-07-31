import time
import urllib.parse
from bs4 import BeautifulSoup
import cloudscraper

def search_and_scrape(topic: str, max_results: int = 15):
    """
    1. Uses Bing News HTML scraping with Pagination to guarantee we find enough articles.
    2. Uses cloudscraper + BeautifulSoup to scrape the full raw text, bypassing Cloudflare 403s.
    """
    print(f"\n[Search Engine] Searching Bing News for: '{topic}'")
    articles_data = []
    
    # cloudscraper perfectly mimics a real Chrome browser's TLS fingerprint to bypass 403 Forbidden errors
    scraper = cloudscraper.create_scraper(browser={'browser': 'chrome', 'platform': 'windows', 'desktop': True})
    
    try:
        encoded_topic = urllib.parse.quote(topic)
        unique_links = []
        seen_urls = set()
        
        # Paginate through Bing News (first=1, 11, 21...) to gather a large pool of links
        for page_offset in [1, 11, 21, 31]:
            if len(unique_links) >= max_results * 2: # Get twice as many links as we need to account for failed scrapes
                break
                
            bing_url = f"https://www.bing.com/news/search?q={encoded_topic}&first={page_offset}"
            bing_response = scraper.get(bing_url, timeout=10)
            bing_soup = BeautifulSoup(bing_response.text, "html.parser")
            
            # Extract links from the current page
            page_links = []
            for a_tag in bing_soup.find_all("a", class_="title"):
                href = a_tag.get("href")
                title = a_tag.get_text(strip=True)
                if href and title and href.startswith("http"):
                    page_links.append({"href": href, "title": title})
                    
            if not page_links:
                for a_tag in bing_soup.find_all("a"):
                    href = a_tag.get("href")
                    title = a_tag.get_text(strip=True)
                    if href and href.startswith("http") and len(title) > 30 and "bing.com" not in href:
                        page_links.append({"href": href, "title": title})
            
            for link in page_links:
                if link["href"] not in seen_urls:
                    unique_links.append(link)
                    seen_urls.add(link["href"])
                    
            time.sleep(1) # Be polite to Bing to avoid blocks
            
        print(f"[Search Engine] Found {len(unique_links)} search results across multiple Bing pages. Scraping full text...")
        
        for res in unique_links:
            if len(articles_data) >= max_results:
                break
                
            url = res['href']
            title = res['title']
            domain = urllib.parse.urlparse(url).netloc.replace("www.", "")
            
            print(f" -> Scraping: {domain} | {title[:50]}...")
            
            try:
                response = scraper.get(url, timeout=12)
                
                if response.status_code != 200:
                    print(f"    [!] Blocked by {domain} (Status: {response.status_code}). Skipping.")
                    continue
                
                soup = BeautifulSoup(response.content, 'html.parser')
                
                paragraphs = soup.find_all('p')
                full_text = " ".join([p.get_text(strip=True) for p in paragraphs])
                
                if not full_text or len(full_text) < 150:
                    print("    [!] Failed to extract sufficient text. Skipping.")
                    continue
                    
                articles_data.append({
                    "url": url,
                    "title": title,
                    "domain": domain,
                    "full_text": full_text
                })
                
            except Exception as e:
                print(f"    [!] Error scraping {url[:40]}...: {e}")
                
    except Exception as e:
        print(f"[Search Engine] Critical search error: {e}")
        
    print(f"[Search Engine] Successfully scraped {len(articles_data)} full articles.")
    return articles_data
