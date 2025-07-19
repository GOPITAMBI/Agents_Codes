# src/custom_tool.py

import os
import requests

class FireCrawlWebSearchTool:
    def __init__(self):
        self.api_key = os.getenv("FIRECRAWL_API_KEY")
        self.api_url = "https://api.firecrawl.dev/v1/search"

    def run(self, query: str):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "query": query,
            "numResults": 3,
            "includeContent": True
        }

        try:
            response = requests.post(self.api_url, headers=headers, json=payload)
            response.raise_for_status()
            results = response.json().get("results", [])
            return "\n\n".join([r.get("content", "") for r in results]) or "No relevant content found."
        except Exception as e:
            return f"Error during web search: {str(e)}"
