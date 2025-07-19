# src/custom_tool.py

import os
import requests
import json

class FireCrawlWebSearchTool:
    def __init__(self):
        self.api_key = os.getenv("FIRECRAWL_API_KEY")
        self.api_url = "https://api.firecrawl.dev/v1/search"

    def run(self, query: str) -> str:
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
            # Send request as raw JSON string
            response = requests.post(self.api_url, headers=headers, data=json.dumps(payload))
            response.raise_for_status()

            results = response.json().get("results", [])

            # Extract and clean text for use by text/PDF/CSV agents
            clean_texts = []
            for i, r in enumerate(results, start=1):
                content = r.get("content", "").strip()
                url = r.get("url", "")
                if content:
                    clean_texts.append(f"Result {i}:\nURL: {url}\n\n{content}")

            return "\n\n---\n\n".join(clean_texts) or "No relevant content found."

        except Exception as e:
            return f"Error during web search: {str(e)}"
