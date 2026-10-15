from langchain_mcp_adapters import client
import os
import asyncio
import json
from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient


load_dotenv()

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
if not TAVILY_API_KEY:
    raise ValueError("TAVILY API, Invalid API key, give right one")

async def tavily_search(query :str):

    client =  MultiServerMCPClient(
        {
        "tavily-remote-mcp": {
        "transport": "streamable_http",
        "url": f"https://mcp.tavily.com/mcp/?tavilyApiKey={TAVILY_API_KEY}"
            }
        }
    )

    tools = await client.get_tools()
    tool = { tool.name : tool for tool in tools}

    tavily_search = tool['tavily_search'] 

    result = await tavily_search.ainvoke({
        "query" : query
    })

    data = json.loads(result[0]['text'])

    context = "\n\n".join(
        item["content"]
        for item in data["results"]
    )

    return context


if __name__ == "__main__":

    query = input("ask : ")
    asyncio.run(tavily_search(query))


    
    

