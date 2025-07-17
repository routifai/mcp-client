"""
Query Intelligence Agent - LLM-powered query analysis using Instructor library
"""

from typing import List, Literal, Optional, Dict, Any
import json
import logging
from enum import Enum

from pydantic import BaseModel, Field
import instructor
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

################
# SCHEMAS #
################

class ComplexityLevel(str, Enum):
    SIMPLE = "simple"
    MODERATE = "moderate" 
    COMPLEX = "complex"

class SearchCategory(str, Enum):
    GENERAL = "general"
    NEWS = "news"
    SCIENCE = "science"
    IT = "it"
    ACADEMIC = "academic"
    SOCIAL_MEDIA = "social_media"
    SHOPPING = "shopping"
    IMAGES = "images"
    VIDEOS = "videos"
    MAPS = "maps"
    MUSIC = "music"

class ScrapingStrategy(str, Enum):
    SNIPPET_ONLY = "snippet_only"
    SELECTIVE_SCRAPE = "selective_scrape"
    FULL_RESEARCH = "full_research"

class QueryIntent(BaseModel):
    """Detected intent and characteristics of the user query"""
    primary_intent: Literal["factual_lookup", "comparison", "research", "current_events", "how_to", "opinion"] = Field(
        description="Primary intent behind the query"
    )
    requires_recent_data: bool = Field(description="Whether query needs current/recent information")
    geographic_context: Optional[str] = Field(description="Geographic context if location-specific")
    temporal_context: Optional[str] = Field(description="Time-related context (recent, historical, etc.)")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in intent detection")

class QueryAnalysis(BaseModel):
    """Complete analysis of user query by the intelligence agent"""
    
    # Core classification
    complexity_level: ComplexityLevel = Field(description="Assessed complexity of the query")
    primary_category: SearchCategory = Field(description="Primary search category")
    secondary_categories: List[SearchCategory] = Field(
        default=[], description="Additional relevant categories"
    )
    
    # Intent analysis
    intent: QueryIntent = Field(description="Detected user intent and context")
    
    # Query processing strategy
    should_decompose: bool = Field(description="Whether to break query into sub-queries")
    decomposed_queries: List[str] = Field(
        default=[], description="Sub-queries if decomposition is beneficial"
    )
    optimized_query: str = Field(description="Optimized version of original query")
    
    # Search strategy
    scraping_strategy: ScrapingStrategy = Field(description="Recommended scraping approach")
    max_results_needed: int = Field(ge=1, le=50, description="Optimal number of search results")
    max_pages_to_scrape: int = Field(ge=0, le=20, description="Number of pages to scrape")
    
    # Search parameters
    suggested_language: str = Field(default="en", description="Suggested language for search")
    suggested_region: Optional[str] = Field(description="Suggested region if location-relevant")
    date_filter: Optional[Literal["day", "week", "month", "year"]] = Field(
        description="Date filtering if recency matters"
    )
    
    # Quality metrics
    analysis_confidence: float = Field(ge=0.0, le=1.0, description="Overall confidence in analysis")
    reasoning: str = Field(description="Explanation of the analysis decisions")

################
# QUERY INTELLIGENCE AGENT #
################

class QueryIntelligenceAgent:
    """LLM-powered agent for intelligent query analysis using Instructor"""
    
    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        self.client = instructor.from_openai(AsyncOpenAI(api_key=api_key))
        self.model = model
        
    async def analyze_query(self, query: str, user_context: Optional[Dict] = None) -> QueryAnalysis:
        """Analyze query and return structured analysis"""
        
        system_prompt = """You are a search query intelligence agent. Your job is to analyze user queries and provide optimal search strategies.

        Analyze the query considering:
        1. COMPLEXITY: Simple (factual lookup), Moderate (comparison/multiple concepts), Complex (research/synthesis)
        2. CATEGORY: Choose the most appropriate search category
        3. INTENT: What is the user trying to accomplish?
        4. DECOMPOSITION: Break complex queries into focused sub-queries
        5. SCRAPING STRATEGY: When is full content needed vs snippets?

        Guidelines:
        - Simple queries: Single concept, factual lookup → snippet_only
        - Moderate queries: Comparisons, multiple concepts → selective_scrape (2-5 pages)
        - Complex queries: Research, analysis, synthesis → full_research (5-20 pages)
        
        Categories:
        - news: Current events, breaking news, recent developments
        - science: Research, studies, scientific information
        - it: Programming, software, technical documentation
        - academic: Scholarly research, educational content
        - social_media: Social platforms, public opinion
        - general: Everything else
        
        Be precise and practical in your analysis."""
        
        user_prompt = f"""Analyze this search query: "{query}"
        
        {"User context: " + json.dumps(user_context) if user_context else ""}
        
        Provide a complete analysis with reasoning for your decisions."""
        
        try:
            analysis = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_model=QueryAnalysis,
                temperature=0.1  # Low temperature for consistent analysis
            )
            
            logger.info(f"Query analysis completed for: {query[:50]}...")
            logger.debug(f"Analysis result: {analysis.model_dump()}")
            
            return analysis
            
        except Exception as e:
            logger.error(f"Query analysis failed: {e}")
            # Fallback to basic analysis
            return self._create_fallback_analysis(query)
    
    def _create_fallback_analysis(self, query: str) -> QueryAnalysis:
        """Create basic analysis when LLM fails"""
        word_count = len(query.split())
        
        # Simple heuristics for fallback
        if word_count <= 3:
            complexity = ComplexityLevel.SIMPLE
            scraping = ScrapingStrategy.SNIPPET_ONLY
        elif word_count <= 8:
            complexity = ComplexityLevel.MODERATE
            scraping = ScrapingStrategy.SELECTIVE_SCRAPE
        else:
            complexity = ComplexityLevel.COMPLEX
            scraping = ScrapingStrategy.FULL_RESEARCH
        
        return QueryAnalysis(
            complexity_level=complexity,
            primary_category=SearchCategory.GENERAL,
            intent=QueryIntent(
                primary_intent="factual_lookup",
                requires_recent_data=False,
                confidence=0.5
            ),
            should_decompose=False,
            optimized_query=query,
            scraping_strategy=scraping,
            max_results_needed=10,
            max_pages_to_scrape=3,
            analysis_confidence=0.3,
            reasoning="Fallback analysis due to LLM failure"
        )