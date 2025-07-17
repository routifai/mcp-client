"""
Fixed Google Search Tool - Your original tool with bugs fixed
"""

from typing import List, Literal, Optional
import asyncio
import logging

from googleapiclient.discovery import build
from pydantic import Field

from atomic_agents.agents.base_agent import BaseIOSchema
from atomic_agents.lib.base.base_tool import BaseTool, BaseToolConfig

logger = logging.getLogger(__name__)

################
# SCHEMAS - FIXED VERSIONS #
################

class GoogleSearchApiToolInputSchema(BaseIOSchema):
    """
    Schema for input to a tool for searching for information, news, references, and other content using Google Custom Search Engine.
    Returns a list of search results with a short description or content snippet and URLs for further exploration.
    """
    
    queries: List[str] = Field(..., description="List of search queries.")
    category: Optional[Literal["general", "news", "social_media"]] = Field(
        "general", description="Category of the search queries."
    )

class GoogleSearchApiResultItemSchema(BaseIOSchema):
    """This schema represents a single search result item."""
    
    url: str = Field(..., description="The URL of the search result")
    title: str = Field(..., description="The title of the search result")
    content: Optional[str] = Field(
        None, description="The content snippet of the search result"
    )
    query: str = Field(..., description="The query used to obtain this search result")

class GoogleSearchApiToolOutputSchema(BaseIOSchema):
    """This schema represents the output of the Google Search API tool."""
    
    results: List[GoogleSearchApiResultItemSchema] = Field(
        ..., description="List of search result items"
    )
    category: Optional[str] = Field(
        None, description="The category of the search results"
    )

################
# CONFIGURATION #
################

class GoogleSearchApiToolConfig(BaseToolConfig):
    api_key: str = Field(..., description="Google API key for Custom Search")
    cx: str = Field(..., description="Google Custom Search Engine ID (CX)")
    max_results: int = 10

################
# FIXED GOOGLE SEARCH TOOL #
################

class FixedGoogleSearchApiTool(BaseTool):
    """
    Fixed version of your original Google Search tool - all bugs resolved
    """
    
    input_schema = GoogleSearchApiToolInputSchema
    output_schema = GoogleSearchApiToolOutputSchema
    
    def __init__(self, config: GoogleSearchApiToolConfig):
        super().__init__(config)
        self.api_key = config.api_key
        self.cx = config.cx
        self.max_results = config.max_results
    
    def _fetch_search_results(self, query: str, category: Optional[str]) -> List[dict]:
        """
        Fixed version of your original search method
        """
        try:
            service = build("customsearch", "v1", developerKey=self.api_key)
            query_params = {
                "q": query,
                "cx": self.cx,
                "num": min(self.max_results, 10),
                "lr": "lang_en",
            }
            
            if category == "news":
                query_params["sort"] = "date"
            elif category == "social_media":
                query_params["q"] = f"{query} site:twitter.com OR site:linkedin.com"
            
            logger.info(f"Fetching results for query: {query}")
            
            res = service.cse().list(**query_params).execute()
            results = res.get("items", [])
            
            # FIXED: Add query field to each result (this was missing in your original)
            for result in results:
                result["query"] = query
                
            logger.info(f"Received {len(results)} results for query '{query}'")
            return results
            
        except Exception as e:
            logger.error(f"Failed to fetch results for query '{query}': {str(e)}")
            raise
    
    async def run_async(self, params: GoogleSearchApiToolInputSchema, max_results: Optional[int] = None) -> GoogleSearchApiToolOutputSchema:
        """
        Fixed async implementation - resolves lambda and filtering bugs
        """
        if not self.api_key or not self.cx:
            raise ValueError("API key and CX must be provided in the configuration.")
        
        loop = asyncio.get_event_loop()
        tasks = []
        
        # FIXED: Proper lambda with default parameter capture (was causing syntax error)
        for query in params.queries:
            task = loop.run_in_executor(
                None,
                lambda q=query: self._fetch_search_results(q, params.category)  # Fixed: capture query in closure
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        all_results = []
        for query, result in zip(params.queries, results):
            if isinstance(result, Exception):
                logger.error(f"Error for query '{query}': {str(result)}")
                continue
            all_results.extend(result)
        
        logger.info(f"Total results before filtering: {len(all_results)}")
        
        # Remove duplicates and filter results
        seen_urls = set()
        unique_results = []
        for result in all_results:
            url = result.get("link")
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)
            unique_results.append({
                "url": url,
                "title": result.get("title", ""),
                "content": result.get("snippet", ""),
                "query": result.get("query", ""),  # FIXED: This field was missing
            })
        
        # FIXED: Proper filtering logic (was broken with incomplete if statement)
        if params.category and params.category != "general":
            filtered_results = []
            for result in unique_results:
                should_include = False
                
                if params.category == "general":
                    should_include = True
                elif params.category == "news":
                    # For news, we rely on the search parameters, so include all
                    should_include = True
                elif params.category == "social_media":
                    # Check if URL contains social media sites
                    should_include = any(
                        site in result["url"].lower()
                        for site in ["twitter.com", "linkedin.com", "reddit.com"]
                    )
                else:
                    # For other categories, include all (can be enhanced later)
                    should_include = True
                
                if should_include:
                    filtered_results.append(result)
        else:
            filtered_results = unique_results
        
        # Apply max_results limit
        max_limit = max_results or self.max_results
        filtered_results = filtered_results[:max_limit]
        logger.info(f"Final results after filtering: {len(filtered_results)}")
        
        return GoogleSearchApiToolOutputSchema(
            results=[
                GoogleSearchApiResultItemSchema(
                    url=result["url"],
                    title=result["title"],
                    content=result["content"],
                    query=result["query"],
                )
                for result in filtered_results
            ],
            category=params.category,
        )
    
    def run(self, params: GoogleSearchApiToolInputSchema, max_results: Optional[int] = None) -> GoogleSearchApiToolOutputSchema:
        """
        Synchronous wrapper for the async method
        """
        return asyncio.run(self.run_async(params, max_results))