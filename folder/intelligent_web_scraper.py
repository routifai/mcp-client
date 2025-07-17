"""
Intelligent Web Scraper Wrapper - Uses your existing crawl4ai scraper as black box
"""

import logging
import re
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

################
# SCRAPER WRAPPER #
################

class IntelligentWebScraper:
    """
    Wrapper around your existing crawl4ai scraper
    Keeps your scraper as black box but adds intelligence for result processing
    """
    
    def __init__(self, timeout: int = 30):
        self.timeout = timeout
        # Your existing scraper will be imported and used
    
    async def scrape_url(self, url: str, max_length: int = 5000) -> Dict[str, Any]:
        """
        Wrapper around your existing single URL scraping
        This will call your existing WebpageScraperTool
        """
        try:
            # Import your existing scraper - UPDATE THIS LINE to match your actual imports
            # from your_existing_scraper import WebpageScraperTool, WebpageScraperToolInputSchema
            # For now, we'll create a placeholder that follows your pattern
            
            # Placeholder import - replace with your actual scraper
            try:
                # Try to import your actual scraper
                from your_scraper_module import WebpageScraperTool, WebpageScraperToolInputSchema
            except ImportError:
                # Fallback error if scraper not found
                return {
                    'success': False,
                    'error': 'WebpageScraperTool not found - please update import in intelligent_web_scraper.py',
                    'content': None,
                    'quality_score': 0.0
                }
            
            # Use your existing scraper as black box
            scraper_tool = WebpageScraperTool()
            scraped_result = await scraper_tool.run(
                WebpageScraperToolInputSchema(url=url, include_links=True)
            )
            
            if scraped_result.error:
                return {
                    'success': False,
                    'error': scraped_result.error,
                    'content': None,
                    'quality_score': 0.0
                }
            else:
                content = scraped_result.content or ""
                
                # Apply length limiting if needed
                if len(content) > max_length:
                    content = self._limit_content_smartly(content, max_length)
                
                # Calculate quality score after scraping
                quality_score = self._calculate_content_quality(content)
                
                return {
                    'success': True,
                    'error': None,
                    'content': content,
                    'quality_score': quality_score
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': f"Scraping failed: {str(e)}",
                'content': None,
                'quality_score': 0.0
            }
    
    async def scrape_multiple_urls(self, urls: List[str], max_length: int = 5000) -> List[Dict[str, Any]]:
        """
        Wrapper for scraping multiple URLs using your existing batch scraping logic
        This maintains compatibility with your existing parallel scraping approach
        """
        try:
            # Use your existing parallel scraping logic (the one from your search_web function)
            # This is the exact pattern you already have working
            
            async def scrape_single_url(index: int, url: str) -> tuple[int, bool, str, str]:
                """Your existing scrape_single_url function pattern"""
                try:
                    # Import your existing scraper - UPDATE THIS LINE
                    try:
                        from your_scraper_module import WebpageScraperTool, WebpageScraperToolInputSchema
                    except ImportError:
                        return index, False, "", "WebpageScraperTool not found - please update import"
                    
                    scraper_tool = WebpageScraperTool()
                    scraped_result = await scraper_tool.run(
                        WebpageScraperToolInputSchema(url=url, include_links=True)
                    )
                    
                    if scraped_result.error:
                        return index, False, "", scraped_result.error
                    else:
                        content = scraped_result.content or ""
                        if len(content) > max_length:
                            content = self._limit_content_smartly(content, max_length)
                        return index, True, content, ""
                        
                except Exception as e:
                    return index, False, "", str(e)
            
            # Create scraping tasks exactly like your existing code
            scraping_tasks = [
                scrape_single_url(i, url)
                for i, url in enumerate(urls)
            ]
            
            # Execute in parallel like your existing code
            import asyncio
            scraping_results = await asyncio.gather(*scraping_tasks, return_exceptions=True)
            
            # Process results
            results = []
            for i, result in enumerate(scraping_results):
                if isinstance(result, Exception):
                    results.append({
                        'success': False,
                        'error': str(result),
                        'content': None,
                        'quality_score': 0.0,
                        'url': urls[i] if i < len(urls) else "unknown"
                    })
                else:
                    index, success, content, error = result
                    quality_score = self._calculate_content_quality(content) if success else 0.0
                    results.append({
                        'success': success,
                        'error': error if not success else None,
                        'content': content if success else None,
                        'quality_score': quality_score,
                        'url': urls[index] if index < len(urls) else "unknown"
                    })
            
            return results
            
        except Exception as e:
            # Return error for all URLs if batch fails
            return [{
                'success': False,
                'error': f"Batch scraping failed: {str(e)}",
                'content': None,
                'quality_score': 0.0,
                'url': url
            } for url in urls]
    
    def _limit_content_smartly(self, content: str, max_length: int) -> str:
        """
        Smart content limiting that preserves structure
        This enhances your existing 500-token limiting logic
        """
        if len(content) <= max_length:
            return content
        
        # Try to cut at sentence boundaries
        truncated = content[:max_length]
        
        # Find the last complete sentence
        last_period = truncated.rfind('. ')
        last_newline = truncated.rfind('\n')
        
        # Cut at the latest sentence or paragraph boundary
        cut_point = max(last_period, last_newline)
        if cut_point > max_length * 0.8:  # Only if we don't lose too much content
            truncated = truncated[:cut_point + 1]
        
        return truncated + "...\n[Content truncated to preserve length limits]"
    
    def _calculate_content_quality(self, content: str) -> float:
        """
        Calculate content quality score (0-1)
        This adds intelligence to assess your scraped content quality
        """
        if not content:
            return 0.0
        
        score = 0.5  # Base score
        
        # Length indicators
        word_count = len(content.split())
        if word_count > 100: score += 0.2
        if word_count > 500: score += 0.1
        if word_count < 50: score -= 0.2  # Too short content
        
        # Structure indicators
        sentence_count = content.count('. ')
        if sentence_count > 5: score += 0.1
        
        paragraph_indicators = content.count('\n')
        if paragraph_indicators > 2: score += 0.1
        
        # Content quality indicators
        if not re.search(r'cookie|privacy policy|terms of service|subscribe|newsletter', content.lower()):
            score += 0.1  # Not just boilerplate
        
        return min(max(score, 0.0), 1.0)
    
    async def close(self):
        """Close any resources - matches your existing pattern"""
        # Your existing scraper cleanup if needed
        pass