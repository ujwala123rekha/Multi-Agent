from typing import List, Dict
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.document_loaders import WebBaseLoader
import httpx
from bs4 import BeautifulSoup
import asyncio

class SearchTools:
    def __init__(self):
        self.search = DuckDuckGoSearchResults()

    async def search_and_fetch(self, query: str) -> List[Dict[str, str]]:
        """Perform deep search and fetch actual page contents."""
        try:
            # 1. Get search results (this returns a string of results with URLs)
            raw_results = self.search.run(query)
            
            # 2. Extract URLs from the raw string (simplified for this example)
            # In a production app, we'd use a regex or a more structured search tool
            import re
            urls = re.findall(r'https?://[^\s,\]]+', raw_results)[:3] # Top 3 links
            
            if not urls:
                return [{"title": "Search Snippet", "url": "N/A", "content": raw_results}]

            detailed_results = []
            
            # 3. Define a fetcher function
            async def fetch_url(client, url):
                try:
                    response = await client.get(url)
                    if response.status_code == 200:
                        soup = BeautifulSoup(response.text, 'lxml')
                        # Strip scripts and styles
                        for s in soup(['script', 'style']): s.decompose()
                        text = soup.get_text(separator=' ', strip=True)[:3000] # Cap per page
                        return {
                            "title": "Web Page",
                            "url": url,
                            "content": text
                        }
                except Exception as e:
                    print(f"Failed to fetch {url}: {e}")
                return None

            # 4. Fetch URLs in parallel
            async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
                tasks = [fetch_url(client, url) for url in urls]
                results = await asyncio.gather(*tasks)
                detailed_results = [r for r in results if r is not None]

            return detailed_results if detailed_results else [{"title": "Search Snippet", "url": "N/A", "content": raw_results}]

        except Exception as e:
            print(f"Deep search failed: {e}")
            return []
