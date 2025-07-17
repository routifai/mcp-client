"""
Clean search_web tool - just the enhanced tool without custom formatting
Drop this directly into your server.py
"""

import os
import asyncio
import logging
from typing import List, Dict, Any, Optional

# Import the intelligent search components
from query_intelligence_agent import QueryIntelligenceAgent, QueryAnalysis
from enhanced_google_search import IntelligentGoogleSearchTool, EnhancedGoogleSearchConfig
from fixed_google_search_tool import GoogleSearchApiTool, GoogleSearchApiToolConfig, GoogleSearchApiToolInputSchema

logger = logging.getLogger(__name__)

# Constants
DEFAULT_SCRAPE_RESULTS = True
DEFAULT_MAX_SCRAPED_PAGES = 3
DEFAULT_NUM_QUERIES = 2
DEFAULT_MAX_RESULTS = 10

@app.tool("search_web")
async def search_web(
    query: str,
    scrape_results: bool = DEFAULT_SCRAPE_RESULTS,
    max_scraped_pages: int = DEFAULT_MAX_SCRAPED_PAGES
) -> List[types.TextContent]:
    """
    Enhanced web search - same interface as your original but with intelligence and bug fixes
    """
    
    logger.debug(f"Executing search_web with query: {query}, scrape_results: {scrape_results}, max_scraped_pages: {max_scraped_pages}")
    
    try:
        # Check if intelligence is available
        openai_api_key = os.getenv("OPENAI_API_KEY")
        use_intelligence = openai_api_key is not None
        
        if use_intelligence:
            # Use intelligent search
            return await _intelligent_search_web(query, scrape_results, max_scraped_pages)
        else:
            # Use fixed original logic
            return await _original_search_web_fixed(query, scrape_results, max_scraped_pages)
    
    except Exception as e:
        error_msg = f"Error in search_web: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return format_mcp_response(error_msg)

async def _intelligent_search_web(query: str, scrape_results: bool, max_scraped_pages: int):
    """Intelligent search with LLM analysis"""
    
    # Validate API credentials
    google_api_key = os.getenv("GOOGLE_API_KEY")
    google_cx = os.getenv("GOOGLE_CX")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    
    if not google_api_key or not google_cx:
        return format_mcp_response("Google API credentials not configured")
    
    # Create intelligent search tool
    config = EnhancedGoogleSearchConfig(
        api_key=google_api_key,
        cx=google_cx,
        openai_api_key=openai_api_key,
        max_results=DEFAULT_MAX_RESULTS,
        enable_caching=True
    )
    
    search_tool = IntelligentGoogleSearchTool(config)
    
    # Execute intelligent search
    search_response = await search_tool.execute_intelligent_search(query=query)
    
    # Convert to your format
    formatted_results = []
    for result in search_response.results:
        formatted_results.append({
            "headline": result.title,
            "url": result.url,
            "content": result.scraped_content if result.is_scraped else result.snippet,
            "scraped": result.is_scraped,
            "query": result.source_query
        })
    
    await search_tool.close()
    return format_mcp_response(formatted_results)

async def _original_search_web_fixed(query: str, scrape_results: bool, max_scraped_pages: int):
    """Your original search logic but with bugs fixed"""
    
    try:
        # Step 1: Generate optimized search queries (using your existing query agent if available)
        try:
            from your_existing_query_agent import get_query_agent, QueryAgentInputSchema
            query_agent = get_query_agent()
            query_agent_output = query_agent.run(
                QueryAgentInputSchema(instruction=query, num_queries=DEFAULT_NUM_QUERIES)
            )
            logger.debug(f"Generated queries: {query_agent_output.queries}")
        except ImportError:
            # Fallback if query agent not available
            logger.warning("Query agent not available, using original query")
            query_agent_output = type('obj', (object,), {'queries': [query]})()
        
        # Step 2: Validate API credentials
        api_key = os.getenv("GOOGLE_API_KEY")
        cx = os.getenv("GOOGLE_CX")
        if not api_key or not cx:
            return format_mcp_response("Google API credentials not configured")
        
        # Step 3: Execute search using fixed Google Search tool
        search_tool = GoogleSearchApiTool(
            config=GoogleSearchApiToolConfig(
                api_key=api_key, cx=cx, max_results=DEFAULT_MAX_RESULTS
            )
        )
        
        search_results = await search_tool.run_async(
            GoogleSearchApiToolInputSchema(queries=query_agent_output.queries)
        )
        
        logger.debug(f"Retrieved {len(search_results.results)} search results")
        
        # Step 4: Convert search results to your format
        formatted_results = []
        for result in search_results.results:
            formatted_results.append({
                "headline": result.title or "No title",
                "url": result.url,
                "content": result.content or "No preview available",
                "scraped": False,
                "query": result.query
            })
        
        # Step 5: Scraping (your exact logic)
        if scrape_results and formatted_results:
            max_scraped_pages = max(1, min(max_scraped_pages, 10, len(formatted_results)))
            
            logger.debug(f"Starting parallel scraping of top {max_scraped_pages} results")
            
            urls_to_scrape = [result["url"] for result in formatted_results[:max_scraped_pages]]
            
            async def scrape_single_url(index: int, url: str) -> tuple[int, bool, str]:
                """Your exact scraping pattern"""
                try:
                    logger.debug(f"Scraping URL {index+1}/{len(urls_to_scrape)}: {url}")
                    
                    # UPDATE THIS IMPORT to match your scraper
                    from your_scraper_module import WebpageScraperTool, WebpageScraperToolInputSchema
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
                            return index, True, scraped_result.content
                        else:
                            return index, False, ""
                            
                except Exception as e:
                    logger.error(f"Error scraping {url}: {e}", exc_info=True)
                    return index, False, ""
            
            # Run scraping in parallel
            scraping_tasks = [
                scrape_single_url(i, url)
                for i, url in enumerate(urls_to_scrape)
            ]
            
            scraping_results = await asyncio.gather(*scraping_tasks, return_exceptions=True)
            
            # Your exact content limiting function
            def limit_to_500_tokens(text):
                if not text:
                    return ""
                max_chars = 3000
                if len(text) <= max_chars:
                    return text
                truncated = text[:max_chars].rstrip()
                return truncated + "...\n[Content truncated to approximately 500 tokens...]"
            
            # Process results
            successful_scrapes = 0
            for result in scraping_results:
                if isinstance(result, Exception):
                    logger.error(f"Scraping task failed with exception: {result}")
                    continue
                
                index, success, content = result
                if success and content:
                    formatted_results[index]["content"] = limit_to_500_tokens(content)
                    formatted_results[index]["scraped"] = True
                    successful_scrapes += 1
            
            logger.debug(f"Successfully scraped and truncated {successful_scrapes} out of {max_scraped_pages} pages in parallel")
        
        return format_mcp_response(formatted_results)
        
    except Exception as e:
        error_msg = f"Error in original search pattern: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return format_mcp_response(error_msg)