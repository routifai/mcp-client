"""
Enhanced Google Search Tool - Intelligent Google Search with LLM-powered query analysis
"""

import asyncio
import time
import logging
from typing import List, Dict, Any, Optional

from pydantic import BaseModel, Field
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from atomic_agents.lib.base.base_tool import BaseTool, BaseToolConfig
from query_intelligence_agent import (
    QueryIntelligenceAgent, 
    QueryAnalysis, 
    SearchCategory, 
    ScrapingStrategy
)
from intelligent_cache import intelligent_cache

logger = logging.getLogger(__name__)

################
# CONFIGURATION #
################

class EnhancedGoogleSearchConfig(BaseToolConfig):
    """Enhanced configuration for Google Search"""
    api_key: str = Field(..., description="Google API key")
    cx: str = Field(..., description="Google Custom Search Engine ID")
    openai_api_key: str = Field(..., description="OpenAI API key for intelligence agent")
    max_results: int = Field(10, description="Default max results")
    enable_caching: bool = Field(True, description="Enable result caching")
    cache_ttl: int = Field(300, description="Cache TTL in seconds")
    intelligence_model: str = Field("gpt-4o-mini", description="Model for query analysis")

################
# RESPONSE MODELS #
################

class SearchResult(BaseModel):
    """Enhanced search result with intelligence metadata"""
    url: str
    title: str
    snippet: str
    display_url: Optional[str] = None
    formatted_url: Optional[str] = None
    page_map: Optional[Dict[str, Any]] = None
    mime_type: Optional[str] = None
    relevance_score: Optional[float] = None
    source_query: str = Field(description="Which query generated this result")
    scraped_content: Optional[str] = None
    scraping_error: Optional[str] = None
    is_scraped: bool = False

class SearchMetadata(BaseModel):
    """Metadata about the search execution"""
    total_execution_time: float
    analysis_time: float
    search_time: float
    scraping_time: float
    queries_executed: int
    pages_scraped: int
    cache_hits: int
    api_calls_made: int
    intelligence_used: bool
    query_analysis: QueryAnalysis

class EnhancedSearchResponse(BaseModel):
    """Complete search response with intelligence"""
    results: List[SearchResult]
    metadata: SearchMetadata
    original_query: str
    query_analysis: QueryAnalysis
    suggestions: Optional[List[str]] = None

################
# ENHANCED GOOGLE SEARCH TOOL #
################

class IntelligentGoogleSearchTool(BaseTool):
    """Intelligent Google Search with LLM-powered query analysis"""
    
    def __init__(self, config: EnhancedGoogleSearchConfig):
        super().__init__(config)
        self.config = config
        self.intelligence_agent = QueryIntelligenceAgent(
            api_key=config.openai_api_key,
            model=config.intelligence_model
        )
        
    async def execute_intelligent_search(self, query: str, user_context: Optional[Dict] = None) -> EnhancedSearchResponse:
        """Execute intelligent search with LLM analysis"""
        start_time = time.time()
        
        # Step 1: Analyze query with intelligence agent
        analysis_start = time.time()
        
        # Check cache for analysis first
        cached_analysis = intelligent_cache.get(query, content_type="analysis")
        if cached_analysis and self.config.enable_caching:
            query_analysis = QueryAnalysis(**cached_analysis)
            logger.info(f"Using cached analysis for: {query[:50]}...")
        else:
            query_analysis = await self.intelligence_agent.analyze_query(query, user_context)
            if self.config.enable_caching:
                intelligent_cache.set(query, query_analysis.model_dump(), content_type="analysis")
        
        analysis_time = time.time() - analysis_start
        
        # Step 2: Execute search strategy
        search_start = time.time()
        search_results = await self._execute_search_strategy(query, query_analysis)
        search_time = time.time() - search_start
        
        # Step 3: Execute scraping strategy
        scraping_start = time.time()
        scraped_results, pages_scraped = await self._execute_scraping_strategy(search_results, query_analysis)
        scraping_time = time.time() - scraping_start
        
        # Step 4: Compile response
        total_time = time.time() - start_time
        
        metadata = SearchMetadata(
            total_execution_time=total_time,
            analysis_time=analysis_time,
            search_time=search_time,
            scraping_time=scraping_time,
            queries_executed=len(query_analysis.decomposed_queries) if query_analysis.should_decompose else 1,
            pages_scraped=pages_scraped,
            cache_hits=0,  # Could track this more precisely
            api_calls_made=1,  # Could track this more precisely
            intelligence_used=True,
            query_analysis=query_analysis
        )
        
        return EnhancedSearchResponse(
            results=scraped_results,
            metadata=metadata,
            original_query=query,
            query_analysis=query_analysis
        )
    
    async def _execute_search_strategy(self, original_query: str, analysis: QueryAnalysis) -> List[SearchResult]:
        """Execute search based on analysis strategy"""
        results = []
        
        # Determine which queries to execute
        if analysis.should_decompose and analysis.decomposed_queries:
            queries_to_execute = analysis.decomposed_queries
            logger.info(f"Executing decomposed search with {len(queries_to_execute)} queries")
        else:
            queries_to_execute = [analysis.optimized_query]
            logger.info(f"Executing single optimized query")
        
        # Execute searches in parallel
        search_tasks = []
        for query in queries_to_execute[:5]:  # Limit to 5 queries max
            search_tasks.append(self._single_google_search(query, analysis))
        
        search_results = await asyncio.gather(*search_tasks, return_exceptions=True)
        
        # Process results
        seen_urls = set()
        for i, result in enumerate(search_results):
            if isinstance(result, Exception):
                logger.error(f"Search failed for query {i}: {result}")
                continue
            
            for item in result:
                if item.url not in seen_urls:
                    seen_urls.add(item.url)
                    results.append(item)
        
        # Sort by relevance and limit
        results.sort(key=lambda x: x.relevance_score or 0, reverse=True)
        return results[:analysis.max_results_needed]
    
    async def _single_google_search(self, query: str, analysis: QueryAnalysis) -> List[SearchResult]:
        """Execute single Google search with category awareness"""
        
        # Check cache first
        cache_key = f"{query}:{analysis.primary_category}"
        if self.config.enable_caching:
            cached_results = intelligent_cache.get(cache_key, str(analysis.primary_category))
            if cached_results:
                return [SearchResult(**result) for result in cached_results]
        
        # Build search parameters
        search_params = {
            "q": query,
            "cx": self.config.cx,
            "num": min(analysis.max_results_needed, 10),
            "safe": "moderate"
        }
        
        # Apply category-specific parameters
        if analysis.primary_category == SearchCategory.NEWS:
            search_params["tbm"] = "nws"
            search_params["sort"] = "date"
        elif analysis.primary_category == SearchCategory.IMAGES:
            search_params["tbm"] = "isch"
        elif analysis.primary_category == SearchCategory.SCIENCE:
            search_params["q"] = f"{query} site:scholar.google.com OR site:arxiv.org OR site:pubmed.ncbi.nlm.nih.gov"
        elif analysis.primary_category == SearchCategory.IT:
            search_params["q"] = f"{query} site:stackoverflow.com OR site:github.com OR documentation"
        elif analysis.primary_category == SearchCategory.ACADEMIC:
            search_params["q"] = f"{query} site:scholar.google.com OR site:edu OR academic"
        elif analysis.primary_category == SearchCategory.SOCIAL_MEDIA:
            search_params["q"] = f"{query} site:twitter.com OR site:linkedin.com OR site:reddit.com"
        
        # Apply filters based on analysis
        if analysis.date_filter:
            date_mapping = {"day": "d1", "week": "w1", "month": "m1", "year": "y1"}
            search_params["dateRestrict"] = date_mapping[analysis.date_filter]
        
        if analysis.suggested_region:
            search_params["gl"] = analysis.suggested_region
        
        if analysis.suggested_language != "en":
            search_params["lr"] = f"lang_{analysis.suggested_language}"
        
        try:
            # Execute Google search
            service = build("customsearch", "v1", developerKey=self.config.api_key)
            response = service.cse().list(**search_params).execute()
            
            # Process results
            results = []
            for item in response.get("items", []):
                result = SearchResult(
                    url=item.get("link", ""),
                    title=item.get("title", ""),
                    snippet=item.get("snippet", ""),
                    display_url=item.get("displayLink", ""),
                    formatted_url=item.get("formattedUrl", ""),
                    page_map=item.get("pagemap", {}),
                    mime_type=item.get("mime", ""),
                    relevance_score=self._calculate_relevance(item, query),
                    source_query=query
                )
                results.append(result)
            
            # Cache results
            if self.config.enable_caching:
                intelligent_cache.set(
                    cache_key, 
                    [result.model_dump() for result in results], 
                    str(analysis.primary_category)
                )
            
            return results
            
        except Exception as e:
            logger.error(f"Google search failed for query '{query}': {e}")
            return []
    
    def _calculate_relevance(self, search_item: Dict, query: str) -> float:
        """Calculate relevance score for search result"""
        title = search_item.get("title", "").lower()
        snippet = search_item.get("snippet", "").lower()
        query_lower = query.lower()
        query_words = query_lower.split()
        
        if not query_words:
            return 0.0
        
        # Score based on keyword matches
        title_matches = sum(1 for word in query_words if word in title)
        snippet_matches = sum(1 for word in query_words if word in snippet)
        
        # Weighted relevance (title matches worth more)
        relevance = (title_matches * 2 + snippet_matches) / (len(query_words) * 3)
        
        return min(relevance, 1.0)
    
    async def _execute_scraping_strategy(self, search_results: List[SearchResult], analysis: QueryAnalysis) -> tuple[List[SearchResult], int]:
        """Execute scraping based on analysis strategy using your existing scraper"""
        
        if analysis.scraping_strategy == ScrapingStrategy.SNIPPET_ONLY:
            logger.info("Using snippet-only strategy")
            return search_results, 0
        
        # Determine how many pages to scrape
        if analysis.scraping_strategy == ScrapingStrategy.SELECTIVE_SCRAPE:
            pages_to_scrape = min(analysis.max_pages_to_scrape, 5, len(search_results))
        else:  # FULL_RESEARCH
            pages_to_scrape = min(analysis.max_pages_to_scrape, len(search_results))
        
        if pages_to_scrape == 0:
            return search_results, 0
        
        logger.info(f"Scraping {pages_to_scrape} pages using {analysis.scraping_strategy} strategy")
        
        # Select best results for scraping (by relevance)
        results_to_scrape = sorted(search_results, key=lambda x: x.relevance_score or 0, reverse=True)[:pages_to_scrape]
        urls_to_scrape = [result.url for result in results_to_scrape]
        
        # Use your existing scraping approach - exactly like your original code
        try:
            # Import your existing scraper components - UPDATE THIS LINE
            try:
                from your_scraper_module import WebpageScraperTool, WebpageScraperToolInputSchema
            except ImportError:
                logger.warning("WebpageScraperTool not found - please update import in enhanced_google_search.py")
                return search_results, 0
            
            # Your exact scraping pattern from search_web function
            async def scrape_single_url(index: int, url: str) -> tuple[int, bool, str]:
                """Scrape a single URL and return index, success status, and content."""
                try:
                    logger.debug(f"Scraping URL {index+1}/{len(urls_to_scrape)}: {url}")
                    scraper_tool = WebpageScraperTool()
                    
                    scraped_result = await scraper_tool.run(
                        WebpageScraperToolInputSchema(url=url, include_links=True)
                    )
                    
                    if scraped_result.error:
                        logger.warning(f"Failed to scrape {url}: {scraped_result.error}")
                        return index, False, ""
                    else:
                        if scraped_result.content:
                            logger.debug(f"Successfully scraped {url}")
                            
                            # Apply your existing content limiting logic
                            content = self._limit_to_500_tokens(scraped_result.content)
                            return index, True, content
                        else:
                            return index, False, ""
                            
                except Exception as e:
                    logger.error(f"Error scraping {url}: {e}", exc_info=True)
                    return index, False, ""
            
            # Run all scraping tasks in parallel - exactly like your original
            scraping_tasks = [
                scrape_single_url(i, url)
                for i, url in enumerate(urls_to_scrape)
            ]
            
            # Wait for all scraping tasks to complete
            scraping_results = await asyncio.gather(*scraping_tasks, return_exceptions=True)
            
            # Process results and update search_results - exactly like your original
            successful_scrapes = 0
            for result in scraping_results:
                if isinstance(result, Exception):
                    logger.error(f"Scraping task failed with exception: {result}")
                    continue
                
                index, success, content = result
                if success and content:
                    # Update the corresponding result
                    if index < len(results_to_scrape):
                        # Find this result in the original search_results list
                        url = urls_to_scrape[index]
                        for search_result in search_results:
                            if search_result.url == url:
                                search_result.scraped_content = content
                                search_result.is_scraped = True
                                successful_scrapes += 1
                                break
            
            logger.debug(f"Successfully scraped {successful_scrapes} out of {pages_to_scrape} pages in parallel")
            return search_results, successful_scrapes
            
        except Exception as e:
            logger.error(f"Scraping strategy execution failed: {e}")
            return search_results, 0
    
    def _limit_to_500_tokens(self, text):
        """
        Your exact existing content limiting function
        Helper function to limit content to approximately 500 tokens
        """
        if not text:
            return ""
        # A very rough approximation: average English word is ~5 chars + space = 6 chars per token
        # 500 tokens = 3000 characters
        max_chars = 3000
        if len(text) <= max_chars:
            return text
        
        # Truncate and add indication that content was trimmed
        truncated = text[:max_chars].rstrip()
        return truncated + "...\n[Content truncated to approximately 500 tokens...]"
    
    async def close(self):
        """Cleanup resources"""
        # Any cleanup needed
        pass