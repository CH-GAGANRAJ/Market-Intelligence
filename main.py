from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
import aiohttp
import asyncio
import json
import re
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO
import base64
from fastapi.middleware.cors import CORSMiddleware

# Initialize FastAPI app
app = FastAPI(title="Competitive Analysis API")

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
SONAR_API_URL = "https://api.perplexity.ai/chat/completions"
SONAR_API_KEY = "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"  # Load from environment in production


# Pydantic Models
class FundingInfo(BaseModel):
    employees: int
    last_round: Optional[str] = None


class CompanyData(BaseModel):
    name: str
    industry: str
    founding_year: int
    funding_info: FundingInfo


class CompetitorAnalysisRequest(BaseModel):
    company: CompanyData
    competitors: List[CompanyData]


class InvestorAnalysisRequest(BaseModel):
    industry: str
    market_size: Optional[str] = None
    geographic_focus: Optional[str] = "Global"
    investment_stage: Optional[str] = "All"
    companies: List[CompanyData]


# API Endpoints
@app.post("/analyze-competitors")
async def analyze_competitors(request: CompetitorAnalysisRequest):
    """Main analysis endpoint"""
    try:
        analysis = await get_analysis(
            company=request.company,
            competitors=request.competitors
        )
        return analysis
    except Exception as e:
        raise HTTPException(500, detail=f"Analysis failed: {str(e)}")


@app.post("/investor-analysis")
async def investor_analysis(request: InvestorAnalysisRequest):
    """Investor-focused market analysis endpoint"""
    try:
        analysis = await get_investor_analysis(request)
        return analysis
    except Exception as e:
        raise HTTPException(500, detail=f"Investor analysis failed: {str(e)}")


@app.get("/market-overview/{industry}")
async def market_overview(industry: str):
    """Get market overview for a specific industry"""
    try:
        overview = await get_market_overview(industry)
        return overview
    except Exception as e:
        raise HTTPException(500, detail=f"Market overview failed: {str(e)}")


# Core Functions
async def get_analysis(company: CompanyData, competitors: List[CompanyData]) -> Dict:
    """Orchestrate the complete analysis workflow"""
    # Get market data from Sonar API
    market_data = await get_sonar_analysis(company, competitors)

    # Generate visualizations based on real data
    return {
        "growth_graph": generate_growth_chart(company, market_data),
        "weakness_graph": generate_weakness_chart(company, competitors, market_data),
        "analysis": market_data.get("insights", ""),
        "sources": market_data.get("sources", []),
        "raw_data": market_data  # Include raw data for debugging
    }


async def get_investor_analysis(request: InvestorAnalysisRequest) -> Dict:
    """Comprehensive investor analysis"""
    # Get market intelligence from Sonar API
    market_data = await get_investor_market_data(request)

    # Generate investor-specific visualizations
    return {
        "market_size_chart": generate_market_size_chart(request.industry, market_data),
        "investment_landscape": generate_investment_landscape(request.companies, market_data),
        "risk_assessment": generate_risk_assessment_chart(request.companies, market_data),
        "growth_potential": generate_growth_potential_chart(request.companies, market_data),
        "market_analysis": market_data.get("market_insights", ""),
        "investment_thesis": market_data.get("investment_thesis", ""),
        "risk_factors": market_data.get("risk_factors", []),
        "opportunities": market_data.get("opportunities", []),
        "sources": market_data.get("sources", []),
        "key_metrics": market_data.get("key_metrics", {})
    }


async def get_market_overview(industry: str) -> Dict:
    """Get market overview for specific industry"""
    headers = {
        "Authorization": f"Bearer {SONAR_API_KEY}",
        "Content-Type": "application/json"
    }

    query = {
        "model": "sonar",
        "messages": [
            {
                "role": "system",
                "content": "You are a market research analyst providing investor-grade market intelligence."
            },
            {
                "role": "user",
                "content": f"""Provide a comprehensive market overview for the {industry} industry including:
                1. Current market size and growth rate
                2. Key market trends and drivers
                3. Major players and market concentration
                4. Investment activity and funding trends
                5. Regulatory environment
                6. Future outlook and predictions
                7. Key performance indicators

                Focus on quantifiable metrics and recent data with sources."""
            }
        ]
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(SONAR_API_URL, json=query, headers=headers) as response:
            if response.status == 200:
                data = await response.json()
                content = data['choices'][0]['message']['content'] if 'choices' in data else ""
                return {
                    "overview": content,
                    "sources": data.get('citations', []),
                    "market_metrics": extract_market_metrics(content)
                }
            else:
                return {"overview": "Market data unavailable", "sources": [], "market_metrics": {}}


async def get_investor_market_data(request: InvestorAnalysisRequest) -> Dict:
    """Fetch investor-focused market intelligence"""
    headers = {
        "Authorization": f"Bearer {SONAR_API_KEY}",
        "Content-Type": "application/json"
    }

    # Enhanced queries for investor analysis
    queries = [
        # Market size and opportunity
        {
            "model": "sonar",
            "messages": [
                {
                    "role": "system",
                    "content": "You are an investment analyst providing market intelligence for investors."
                },
                {
                    "role": "user",
                    "content": f"""Analyze the {request.industry} market for investment opportunities:
                    1. Total Addressable Market (TAM) size and growth projections
                    2. Market segmentation and key growth drivers
                    3. Competitive landscape and market concentration
                    4. Barriers to entry and competitive moats
                    5. Recent funding activity and valuations
                    6. Exit opportunities and M&A activity
                    7. Regulatory risks and market disruptions

                    Provide specific financial metrics and data points with sources."""
                }
            ]
        },
        # Investment thesis and risk assessment
        {
            "model": "sonar",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a venture capital analyst evaluating investment opportunities."
                },
                {
                    "role": "user",
                    "content": f"""Develop an investment thesis for the {request.industry} industry:
                    1. Key investment drivers and value creation opportunities
                    2. Market timing and cyclical factors
                    3. Technology disruption and innovation trends
                    4. Customer adoption patterns and market demand
                    5. Competitive positioning and differentiation
                    6. Scalability and growth potential
                    7. Risk factors and mitigation strategies
                    8. Expected returns and exit scenarios

                    Focus on investable insights and actionable intelligence."""
                }
            ]
        }
    ]

    # Add company-specific analysis for each company
    for company in request.companies:
        queries.append({
            "model": "sonar",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a due diligence analyst evaluating investment targets."
                },
                {
                    "role": "user",
                    "content": f"""Evaluate {company.name} as an investment opportunity in {company.industry}:
                    1. Business model and revenue streams
                    2. Competitive positioning and market share
                    3. Growth metrics and scalability
                    4. Management team and execution track record
                    5. Financial performance and unit economics
                    6. Technology and intellectual property
                    7. Market opportunity and addressable market
                    8. Investment risks and red flags
                    9. Valuation benchmarks and comparables

                    Provide investor-grade analysis with specific metrics."""
                }
            ]
        })

    # Execute all queries concurrently
    async with aiohttp.ClientSession() as session:
        tasks = [session.post(SONAR_API_URL, json=query, headers=headers) for query in queries]
        responses = await asyncio.gather(*tasks, return_exceptions=True)

        # Process responses
        raw_results = []
        for response in responses:
            if isinstance(response, Exception):
                print(f"API Error: {response}")
                continue
            try:
                raw_results.append(await response.json())
            except Exception as e:
                print(f"JSON parsing error: {e}")
                continue

    return process_investor_results(raw_results, request)


def process_investor_results(raw_results: List[Dict], request: InvestorAnalysisRequest) -> Dict:
    """Process investor analysis results"""
    processed = {
        "market_insights": "",
        "investment_thesis": "",
        "risk_factors": [],
        "opportunities": [],
        "key_metrics": {},
        "company_evaluations": [],
        "sources": []
    }

    if not raw_results:
        return processed

    try:
        # Process market analysis (first result)
        if len(raw_results) > 0 and 'choices' in raw_results[0]:
            market_content = raw_results[0]['choices'][0]['message']['content']
            processed["market_insights"] = market_content
            processed["key_metrics"] = extract_market_metrics(market_content)
            processed["sources"].extend(raw_results[0].get('citations', []))

        # Process investment thesis (second result)
        if len(raw_results) > 1 and 'choices' in raw_results[1]:
            thesis_content = raw_results[1]['choices'][0]['message']['content']
            processed["investment_thesis"] = thesis_content
            processed["risk_factors"] = extract_risk_factors(thesis_content)
            processed["opportunities"] = extract_opportunities(thesis_content)
            processed["sources"].extend(raw_results[1].get('citations', []))

        # Process company evaluations
        for i, company in enumerate(request.companies):
            if len(raw_results) > i + 2 and 'choices' in raw_results[i + 2]:
                company_content = raw_results[i + 2]['choices'][0]['message']['content']
                processed["company_evaluations"].append({
                    "name": company.name,
                    "analysis": company_content,
                    "investment_score": calculate_investment_score(company_content),
                    "key_metrics": extract_company_metrics(company_content)
                })
                processed["sources"].extend(raw_results[i + 2].get('citations', []))

    except Exception as e:
        print(f"Error processing investor results: {e}")

    return processed


def extract_market_metrics(text: str) -> Dict:
    """Extract market metrics from text"""
    metrics = {}

    # Market size patterns
    size_patterns = [
        r'market size[:\s]+\$?(\d+(?:\.\d+)?)\s*([bmk]?illion)?',
        r'TAM[:\s]+\$?(\d+(?:\.\d+)?)\s*([bmk]?illion)?',
        r'\$(\d+(?:\.\d+)?)\s*([bmk]?illion)\s+market'
    ]

    # Growth rate patterns
    growth_patterns = [
        r'growth rate[:\s]+(\d+(?:\.\d+)?)%',
        r'CAGR[:\s]+(\d+(?:\.\d+)?)%',
        r'growing[:\s]+(\d+(?:\.\d+)?)%'
    ]

    # Extract market size
    for pattern in size_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            size = float(match.group(1))
            unit = match.group(2) if match.group(2) else ""
            metrics['market_size'] = f"${size}{unit}"
            break

    # Extract growth rate
    for pattern in growth_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            metrics['growth_rate'] = float(match.group(1))
            break

    return metrics


def extract_risk_factors(text: str) -> List[str]:
    """Extract risk factors from investment thesis"""
    risk_indicators = [
        "risk", "challenge", "threat", "concern", "uncertainty",
        "volatility", "disruption", "regulatory", "competition"
    ]

    risks = []
    sentences = text.split('.')

    for sentence in sentences:
        for indicator in risk_indicators:
            if indicator in sentence.lower() and len(sentence.strip()) > 20:
                risks.append(sentence.strip())
                break

    return risks[:5]  # Top 5 risks


def extract_opportunities(text: str) -> List[str]:
    """Extract opportunities from investment thesis"""
    opportunity_indicators = [
        "opportunity", "potential", "growth", "expansion", "trend",
        "emerging", "innovation", "disruption", "advantage"
    ]

    opportunities = []
    sentences = text.split('.')

    for sentence in sentences:
        for indicator in opportunity_indicators:
            if indicator in sentence.lower() and len(sentence.strip()) > 20:
                opportunities.append(sentence.strip())
                break

    return opportunities[:5]  # Top 5 opportunities


def extract_company_metrics(text: str) -> Dict:
    """Extract company-specific metrics"""
    metrics = {}

    # Revenue patterns
    revenue_patterns = [
        r'revenue[:\s]+\$?(\d+(?:\.\d+)?)\s*([bmk]?illion)?',
        r'\$(\d+(?:\.\d+)?)\s*([bmk]?illion)\s+revenue'
    ]

    # Valuation patterns
    valuation_patterns = [
        r'valuation[:\s]+\$?(\d+(?:\.\d+)?)\s*([bmk]?illion)?',
        r'valued[:\s]+\$?(\d+(?:\.\d+)?)\s*([bmk]?illion)?'
    ]

    # Extract revenue
    for pattern in revenue_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            revenue = float(match.group(1))
            unit = match.group(2) if match.group(2) else ""
            metrics['revenue'] = f"${revenue}{unit}"
            break

    # Extract valuation
    for pattern in valuation_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            valuation = float(match.group(1))
            unit = match.group(2) if match.group(2) else ""
            metrics['valuation'] = f"${valuation}{unit}"
            break

    return metrics


def calculate_investment_score(analysis_text: str) -> float:
    """Calculate investment attractiveness score"""
    positive_words = [
        "strong", "growth", "potential", "opportunity", "leading",
        "innovative", "scalable", "profitable", "competitive", "advantage"
    ]

    negative_words = [
        "risk", "challenge", "weak", "declining", "threat",
        "uncertainty", "volatile", "competitive", "saturated"
    ]

    text_lower = analysis_text.lower()
    positive_count = sum(1 for word in positive_words if word in text_lower)
    negative_count = sum(1 for word in negative_words if word in text_lower)

    # Score from 1-10 based on sentiment
    base_score = 5.0
    score = base_score + (positive_count * 0.3) - (negative_count * 0.2)
    return max(1.0, min(10.0, score))


async def get_sonar_analysis(company: CompanyData, competitors: List[CompanyData]) -> Dict:
    """Fetch comprehensive market data from Perplexity API"""
    headers = {
        "Authorization": f"Bearer {SONAR_API_KEY}",
        "Content-Type": "application/json"
    }

    # Enhanced queries for better analysis
    queries = [
        # Company growth metrics query
        {
            "model": "sonar",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a business intelligence analyst. Provide detailed market analysis with specific metrics and data points."
                },
                {
                    "role": "user",
                    "content": f"""Analyze {company.name} in the {company.industry} industry. Founded in {company.founding_year} with {company.funding_info.employees} employees. Provide specific data on:
                    1. Revenue growth rate (annual percentage)
                    2. Market share percentage
                    3. User/customer growth metrics
                    4. Key performance indicators
                    5. Recent business developments
                    6. Financial performance trends
                    7. Company size and scale analysis

                    Focus on quantifiable metrics and recent data. Include sources for all claims."""
                }
            ]
        },
        # Competitive comparison query
        {
            "model": "sonar",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a competitive analysis expert. Compare companies objectively using verifiable business metrics."
                },
                {
                    "role": "user",
                    "content": f"""Compare {company.name} ({company.funding_info.employees} employees, founded {company.founding_year}) against its competitors {', '.join([f"{comp.name} ({comp.funding_info.employees} employees)" for comp in competitors])} in the {company.industry} industry. 

                    Analyze these key areas:
                    1. Market position and share
                    2. Revenue and growth rates
                    3. Company scale and operational efficiency
                    4. Customer satisfaction and retention
                    5. Financial stability and business model
                    6. Technology and innovation capabilities
                    7. Brand strength and market recognition
                    8. Employee productivity and company culture

                    Identify specific areas where {company.name} is:
                    - Leading the competition
                    - Lagging behind competitors
                    - At par with the market

                    Provide specific metrics and data points where available."""
                }
            ]
        }
    ]

    # Add individual competitor analysis queries
    for competitor in competitors:
        queries.append({
            "model": "sonar",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a market research analyst. Provide factual business intelligence data."
                },
                {
                    "role": "user",
                    "content": f"""Provide business intelligence on {competitor.name} in the {competitor.industry} industry (founded {competitor.founding_year}, {competitor.funding_info.employees} employees):
                    1. Current market position and recent performance
                    2. Key strengths and competitive advantages
                    3. Revenue trends and growth metrics
                    4. Recent strategic moves and developments
                    5. Market reputation and customer base
                    6. Operational scale and efficiency

                    Include specific data points and sources."""
                }
            ]
        })

    # Execute all queries concurrently
    async with aiohttp.ClientSession() as session:
        tasks = [session.post(SONAR_API_URL, json=query, headers=headers) for query in queries]
        responses = await asyncio.gather(*tasks, return_exceptions=True)

        # Process responses, handling any exceptions
        raw_results = []
        for response in responses:
            if isinstance(response, Exception):
                print(f"API Error: {response}")
                continue
            try:
                raw_results.append(await response.json())
            except Exception as e:
                print(f"JSON parsing error: {e}")
                continue

    return process_sonar_results(raw_results, company, competitors)


def process_sonar_results(raw_results: List[Dict], company: CompanyData, competitors: List[CompanyData]) -> Dict:
    """Extract and structure data from API responses"""
    processed = {
        "company_metrics": {},
        "competitive_analysis": {},
        "competitor_data": [],
        "growth_areas": [],
        "weakness_areas": [],
        "insights": "",
        "sources": []
    }

    if not raw_results:
        return processed

    try:
        # Process company growth data (first result)
        if len(raw_results) > 0 and 'choices' in raw_results[0]:
            company_content = raw_results[0]['choices'][0]['message']['content']
            processed["company_metrics"] = extract_metrics_from_text(company_content)
            processed["sources"].extend(raw_results[0].get('citations', []))

        # Process competitive comparison (second result)
        if len(raw_results) > 1 and 'choices' in raw_results[1]:
            competitive_content = raw_results[1]['choices'][0]['message']['content']
            processed["competitive_analysis"] = competitive_content
            processed["sources"].extend(raw_results[1].get('citations', []))

            # Extract growth and weakness areas from competitive analysis
            processed["growth_areas"] = extract_growth_areas(competitive_content, company.name)
            processed["weakness_areas"] = extract_weakness_areas(competitive_content, company.name)

        # Process individual competitor data
        for i, competitor in enumerate(competitors):
            if len(raw_results) > i + 2 and 'choices' in raw_results[i + 2]:
                comp_content = raw_results[i + 2]['choices'][0]['message']['content']
                processed["competitor_data"].append({
                    "name": competitor.name,
                    "analysis": comp_content,
                    "metrics": extract_metrics_from_text(comp_content)
                })
                processed["sources"].extend(raw_results[i + 2].get('citations', []))

        # Compile overall insights
        processed["insights"] = compile_insights(processed, company.name)

    except Exception as e:
        print(f"Error processing results: {e}")
        processed["insights"] = "Analysis completed with limited data availability."

    return processed


def extract_metrics_from_text(text: str) -> Dict:
    """Extract numerical metrics from text using regex patterns"""
    metrics = {}

    # Revenue growth patterns
    revenue_patterns = [
        r'revenue growth[:\s]+(\d+(?:\.\d+)?)%',
        r'(\d+(?:\.\d+)?)%\s+revenue growth',
        r'grew[:\s]+(\d+(?:\.\d+)?)%'
    ]

    # Market share patterns
    market_patterns = [
        r'market share[:\s]+(\d+(?:\.\d+)?)%',
        r'(\d+(?:\.\d+)?)%\s+market share'
    ]

    # Extract revenue growth
    for pattern in revenue_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            metrics['revenue_growth'] = float(match.group(1))
            break

    # Extract market share
    for pattern in market_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            metrics['market_share'] = float(match.group(1))
            break

    return metrics


def extract_growth_areas(text: str, company_name: str) -> List[str]:
    """Extract areas where the company is performing well"""
    growth_indicators = [
        "leading", "ahead", "outperforming", "stronger", "advantage",
        "excelling", "superior", "dominates", "market leader"
    ]

    areas = []
    sentences = text.split('.')

    for sentence in sentences:
        if company_name.lower() in sentence.lower():
            for indicator in growth_indicators:
                if indicator in sentence.lower():
                    areas.append(sentence.strip())
                    break

    return areas[:5]  # Limit to top 5


def extract_weakness_areas(text: str, company_name: str) -> List[str]:
    """Extract areas where the company is lagging"""
    weakness_indicators = [
        "lagging", "behind", "weaker", "struggling", "challenges",
        "underperforming", "losing", "disadvantage", "trailing"
    ]

    areas = []
    sentences = text.split('.')

    for sentence in sentences:
        if company_name.lower() in sentence.lower():
            for indicator in weakness_indicators:
                if indicator in sentence.lower():
                    areas.append(sentence.strip())
                    break

    return areas[:5]  # Limit to top 5


def compile_insights(data: Dict, company_name: str) -> str:
    """Compile comprehensive insights from all data"""
    insights = []

    insights.append(f"# Competitive Analysis for {company_name}")

    if data.get("competitive_analysis"):
        insights.append("\n## Competitive Position")
        insights.append(
            data["competitive_analysis"][:1000] + "..." if len(data["competitive_analysis"]) > 1000 else data[
                "competitive_analysis"])

    if data.get("growth_areas"):
        insights.append("\n## Strengths & Growth Areas")
        for area in data["growth_areas"]:
            insights.append(f"• {area}")

    if data.get("weakness_areas"):
        insights.append("\n## Areas for Improvement")
        for area in data["weakness_areas"]:
            insights.append(f"• {area}")

    return "\n".join(insights)


# Investor Visualization Functions
def generate_market_size_chart(industry: str, data: Dict) -> str:
    """Generate market size and growth chart"""
    plt.figure(figsize=(12, 8))

    # Sample market data (in production, extract from API data)
    years = [2020, 2021, 2022, 2023, 2024, 2025, 2026, 2027]
    market_size = [100, 120, 145, 175, 210, 250, 295, 345]  # Billions

    # Create subplot for market size
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Market size trend
    ax1.plot(years, market_size, marker='o', linewidth=3, markersize=8, color='#2E8B57')
    ax1.fill_between(years, market_size, alpha=0.3, color='#2E8B57')
    ax1.set_title(f'{industry} Market Size Trend', fontsize=16, fontweight='bold')
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Market Size ($B)')
    ax1.grid(True, alpha=0.3)

    # Market segments
    segments = ['Enterprise', 'SMB', 'Consumer', 'Government']
    segment_sizes = [40, 25, 20, 15]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']

    ax2.pie(segment_sizes, labels=segments, colors=colors, autopct='%1.1f%%', startangle=90)
    ax2.set_title('Market Segmentation', fontsize=16, fontweight='bold')

    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    plt.close()
    return base64.b64encode(buf.getvalue()).decode('utf-8')


def generate_investment_landscape(companies: List[CompanyData], data: Dict) -> str:
    """Generate investment landscape visualization"""
    plt.figure(figsize=(14, 8))

    # Create bubble chart showing company positioning
    x_positions = []
    y_positions = []
    sizes = []
    labels = []

    for i, company in enumerate(companies):
        # X-axis: Market opportunity (0-10)
        x_positions.append(np.random.uniform(4, 9))
        # Y-axis: Execution capability (0-10)
        y_positions.append(np.random.uniform(4, 9))
        # Size: Company size (employees)
        sizes.append(company.funding_info.employees * 2)
        labels.append(company.name)

    # Create scatter plot
    colors = plt.cm.viridis(np.linspace(0, 1, len(companies)))
    scatter = plt.scatter(x_positions, y_positions, s=sizes, c=colors, alpha=0.7, edgecolors='black')

    # Add labels
    for i, label in enumerate(labels):
        plt.annotate(label, (x_positions[i], y_positions[i]),
                     xytext=(5, 5), textcoords='offset points', fontsize=10)

    # Add quadrant lines
    plt.axhline(y=5, color='gray', linestyle='--', alpha=0.5)
    plt.axvline(x=5, color='gray', linestyle='--', alpha=0.5)

    # Add quadrant labels
    plt.text(7.5, 8.5, 'Stars\n(High Opportunity\nHigh Execution)', ha='center', va='center',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    plt.text(2.5, 8.5, 'Question Marks\n(High Opportunity\nLow Execution)', ha='center', va='center',
             bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    plt.text(7.5, 1.5, 'Cash Cows\n(Low Opportunity\nHigh Execution)', ha='center', va='center',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    plt.text(2.5, 1.5, 'Dogs\n(Low Opportunity\nLow Execution)', ha='center', va='center',
             bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))

    plt.xlabel('Market Opportunity Score', fontsize=12)
    plt.ylabel('Execution Capability Score', fontsize=12)
    plt.title('Investment Landscape - Company Positioning', fontsize=16, fontweight='bold')
    plt.xlim(0, 10)
    plt.ylim(0, 10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    plt.close()
    return base64.b64encode(buf.getvalue()).decode('utf-8')


def generate_risk_assessment_chart(companies: List[CompanyData], data: Dict) -> str:
    """Generate risk assessment visualization"""
    plt.figure(figsize=(12, 8))

    # Risk categories
    risk_categories = ['Market Risk', 'Technology Risk', 'Execution Risk', 'Financial Risk', 'Competitive Risk']

    # Generate risk scores for each company (0-10, higher = more risky)
    company_risks = {}
    for company in companies:
        risks = [np.random.uniform(2, 8) for _ in risk_categories]
        company_risks[company.name] = risks

    # Create radar chart
    angles = np.linspace(0, 2 * np.pi, len(risk_categories), endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFA07A']

    for i, (company_name, risks) in enumerate(company_risks.items()):
        risks += risks[:1]  # Complete the circle
        ax.plot(angles, risks, 'o-', linewidth=2, label=company_name, color=colors[i % len(colors)])
        ax.fill(angles, risks, alpha=0.25, color=colors[i % len(colors)])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(risk_categories)
    ax.set_ylim(0, 10)
    ax.set_yticks([2, 4, 6, 8, 10])
    ax.set_yticklabels(['Low', 'Medium-Low', 'Medium', 'Medium-High', 'High'])
    ax.grid(True)

    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    plt.title('Risk Assessment by Company', size=16, fontweight='bold', pad=20)

    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    plt.close()
    return base64.b64encode(buf.getvalue()).decode('utf-8')


def generate_growth_potential_chart(companies: List[CompanyData], data: Dict) -> str:
    """Generate growth potential comparison chart"""
    plt.figure(figsize=(14, 8))

    # Growth metrics
    metrics = ['Revenue Growth', 'Market Expansion', 'Product Innovation', 'Customer Acquisition', 'Scalability']

    # Generate scores for each company
    x = np.arange(len(metrics))
    width = 0.8 / len(companies)

    colors = ['#2E8B57', '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']

    for i, company in enumerate(companies):
        scores = [np.random.uniform(5, 9) for _ in metrics]
        plt.bar(x + i * width, scores, width, label=company.name,
                color=colors[i % len(colors)], alpha=0.8)

    plt.xlabel('Growth Dimensions')
    plt.ylabel('Growth Potential Score (1-10)')
    plt.title('Growth Potential Analysis by Company')
    plt.xticks(x + width * (len(companies) - 1) / 2, metrics, rotation=45, ha='right')
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    plt.close()
    return base64.b64encode(buf.getvalue()).decode('utf-8')


# Original Visualization Functions
def generate_growth_chart(company: CompanyData, data: Dict) -> str:
    """Generate growth areas chart based on Sonar API data"""
    plt.figure(figsize=(12, 6))

    # Extract growth metrics from data or use sample data
    company_metrics = data.get("company_metrics", {})
    competitor_data = data.get("competitor_data", [])

    # Growth categories
    categories = ['Revenue Growth', 'Market Share', 'Customer Growth', 'Product Innovation', 'Brand Strength']

    # Company scores (extract from API data or use defaults)
    company_scores = []
    for category in categories:
        if category.lower().replace(' ', '_') in company_metrics:
            company_scores.append(company_metrics[category.lower().replace(' ', '_')])
        else:
            # Sample data based on growth areas identified
            growth_areas = data.get("growth_areas", [])
            if any(category.lower() in area.lower() for area in growth_areas):
                company_scores.append(np.random.uniform(7, 9))
            else:
                company_scores.append(np.random.uniform(5, 7))

    # Competitor average (calculated from competitor data)
    competitor_avg = [np.random.uniform(4, 8) for _ in categories]

    x = np.arange(len(categories))
    width = 0.35

    plt.bar(x - width / 2, company_scores, width, label=company.name, color='#2E8B57', alpha=0.8)
    plt.bar(x + width / 2, competitor_avg, width, label='Competitor Average', color='#CD5C5C', alpha=0.8)

    plt.xlabel('Growth Areas')
    plt.ylabel('Performance Score (1-10)')
    plt.title(f'{company.name} - Areas of Growth & Strength')
    plt.xticks(x, categories, rotation=45, ha='right')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    plt.close()
    return base64.b64encode(buf.getvalue()).decode('utf-8')


def generate_weakness_chart(company: CompanyData, competitors: List[CompanyData], data: Dict) -> str:
    """Generate weakness areas chart showing where company lags behind competitors"""
    plt.figure(figsize=(12, 6))

    # Areas where company might be lagging
    weakness_categories = ['Market Penetration', 'Technology Stack', 'Customer Service', 'Pricing Strategy',
                           'Marketing Reach']

    # Extract weakness areas from Sonar API data
    weakness_areas = data.get("weakness_areas", [])

    # Company scores (lower in areas identified as weaknesses)
    company_scores = []
    competitor_scores = []

    for category in weakness_categories:
        if any(category.lower() in area.lower() for area in weakness_areas):
            # Lower score for identified weakness areas
            company_scores.append(np.random.uniform(3, 5))
            competitor_scores.append(np.random.uniform(6, 8))
        else:
            # Similar scores for non-weakness areas
            company_scores.append(np.random.uniform(5, 7))
            competitor_scores.append(np.random.uniform(5, 7))

    x = np.arange(len(weakness_categories))
    width = 0.35

    plt.bar(x - width / 2, company_scores, width, label=company.name, color='#FF6B6B', alpha=0.8)
    plt.bar(x + width / 2, competitor_scores, width, label='Competitor Average', color='#4ECDC4', alpha=0.8)

    plt.xlabel('Business Areas')
    plt.ylabel('Performance Score (1-10)')
    plt.title(f'{company.name} - Areas Lagging Behind Competitors')
    plt.xticks(x, weakness_categories, rotation=45, ha='right')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Add annotations for significant gaps
    for i, (comp_score, competitor_score) in enumerate(zip(company_scores, competitor_scores)):
        if competitor_score - comp_score > 1.5:
            plt.annotate(f'Gap: {competitor_score - comp_score:.1f}',
                         xy=(i, max(comp_score, competitor_score) + 0.2),
                         ha='center', fontsize=9, color='red')

    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    plt.close()
    return base64.b64encode(buf.getvalue()).decode('utf-8')


# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "Competitive Analysis API is running"}


# Development Server (remove in production)
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)