import asyncio
from main import get_sonar_analysis
from main import CompanyData  # Import your Pydantic model
import aiohttp
from typing import List, Dict, Any
import json
import asyncio
# Test with this mock data
company = {
    "name": "Slack",
    "industry": "SaaS",
    "founding_year": 2013
}

competitors = [{
    "name": "Microsoft Teams",
    "industry": "SaaS"
}]

result = asyncio.run(get_sonar_analysis(company, competitors))
print(json.dumps(result, indent=2))

# Run test