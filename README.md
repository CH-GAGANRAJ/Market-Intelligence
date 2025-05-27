# Market Intelligence

**AI-powered competitive analysis for startups and investors**  
*Perplexity Hackathon Submission*

## 🌟 Overview

This full-stack application provides:
- **Real-time competitive analysis** using Perplexity's Sonar API
- **Investor-grade market intelligence** with visual analytics
- **Interactive dashboards** for both companies and investors

## ✨ Key Features

### 🔍 For Companies
- Competitor benchmarking (revenue, market share, growth)
- Strength/weakness identification
- Industry trend analysis
- Automated report generation

### 💼 For Investors
- Market opportunity scoring
- Investment landscape visualization
- Risk assessment matrices
- Portfolio performance tracking

### 📊 Data & Visualization
- AI-extracted metrics from unstructured data
- Dynamic chart generation
- Exportable insights
- Interactive dashboards

## 🛠️ Tech Stack

### Backend (API)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Perplexity](https://img.shields.io/badge/Perplexity_Sonar-AA336A?style=for-the-badge)

### Frontend (Dashboard)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly)
![Pandas](https://img.shields.io/badge/Pandas-2C2D72?style=for-the-badge&logo=pandas&logoColor=white)

## 🚀 Quick Start

### Backend Services
```mermaid
graph LR
    A[FastAPI] --> B[Perplexity Sonar]
    A --> C[Matplotlib]
    A --> D[Pydantic]
    A --> E[AsyncIO]

### 1. Backend Setup
```bash
# Clone repository
git clone https://github.com/your-username/market-intelligence-dashboard.git
cd market-intelligence-dashboard

# Install dependencies
pip install -r requirements.txt

# Set API key (get from Perplexity)
echo "PERPLEXITY_API_KEY=your_key_here" > .env

# Start FastAPI server
uvicorn main:app --reload
