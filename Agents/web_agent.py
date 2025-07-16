from langchain.utilities.tavily_search import TavilySearchAPIWrapper

def web_search(query):
    tavily = TavilySearchAPIWrapper()
    return tavily.run(query)
