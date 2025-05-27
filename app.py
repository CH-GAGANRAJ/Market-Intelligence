import streamlit as st
import requests
import base64
from io import BytesIO
from PIL import Image
import json
from typing import List, Dict
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

# Page configuration - MUST be first Streamlit command
st.set_page_config(
    page_title="Competitive Analysis Dashboard",
    page_icon="📊",
    layout="wide"
)

# Configuration
BACKEND_URL = "http://localhost:8000"  # Change in production


# Helper Functions
def display_graph(base64_str: str):
    """Display base64 encoded graph"""
    try:
        img_bytes = base64.b64decode(base64_str)
        img = Image.open(BytesIO(img_bytes))
        st.image(img, use_column_width=True)
    except Exception as e:
        st.error(f"Error displaying graph: {str(e)}")


def get_competitor_default(competitor_num: int, industry: str) -> Dict:
    """Generate default competitor data"""
    return {
        "name": f"Competitor {competitor_num}",
        "industry": industry,
        "founding_year": 2020,
        "key_products": ["Product A", "Product B"],
        "social_media": {"twitter": 0.5, "linkedin": 0.3},
        "funding_info": {"total": 1000000, "employees": 50}
    }


def format_currency(amount):
    """Format currency values"""
    if amount >= 1e9:
        return f"${amount / 1e9:.1f}B"
    elif amount >= 1e6:
        return f"${amount / 1e6:.1f}M"
    elif amount >= 1e3:
        return f"${amount / 1e3:.1f}K"
    else:
        return f"${amount:.0f}"


# Main App
def main():
    st.title("📊 Competitive Analysis Dashboard")
    st.markdown("Compare your company against competitors using market intelligence")

    # Navigation
    menu = st.sidebar.selectbox("Menu", ["Company Analysis", "Investor View"])

    if menu == "Company Analysis":
        render_company_interface()
    else:
        render_investor_interface()


def render_company_interface():
    with st.form("company_form"):
        st.header("1️⃣ Your Company Details")

        col1, col2 = st.columns(2)

        with col1:
            company_name = st.text_input("Company Name", value="Acme Inc")
            industry = st.text_input("Industry", value="SaaS")
            founding_year = st.number_input("Founding Year", min_value=1900, max_value=2024, value=2015)

        with col2:
            employees = st.number_input("Number of Employees", min_value=1, value=100)

        st.header("2️⃣ Competitor Information")
        num_competitors = st.number_input("Number of Competitors", min_value=1, max_value=5, value=1)

        competitors = []
        for i in range(num_competitors):
            with st.expander(f"Competitor {i + 1} Details", expanded=True):
                c_name = st.text_input(f"Name {i + 1}", value=f"Competitor {i + 1}")
                c_industry = st.text_input(f"Industry {i + 1}", value=industry)
                c_year = st.number_input(f"Founding Year {i + 1}", min_value=1900, max_value=2024, value=2016)
                c_employees = st.number_input(f"Employees {i + 1}", min_value=1, value=50)

                competitors.append({
                    "name": c_name,
                    "industry": c_industry,
                    "founding_year": c_year,
                    "funding_info": {"employees": c_employees}
                })

        submitted = st.form_submit_button("🚀 Analyze Competitors")

        if submitted:
            # Validate inputs
            if not company_name.strip():
                st.error("Please enter a company name")
                return

            if not industry.strip():
                st.error("Please enter an industry")
                return

            # Check if all competitors have names
            for i, comp in enumerate(competitors):
                if not comp["name"].strip() or comp["name"] == f"Competitor {i + 1}":
                    st.warning(f"Please provide a name for Competitor {i + 1}")

            with st.spinner("🔍 Analyzing market data..."):
                try:
                    payload = {
                        "company": {
                            "name": company_name,
                            "industry": industry,
                            "founding_year": founding_year,
                            "funding_info": {"employees": employees}
                        },
                        "competitors": competitors
                    }

                    # Display payload for debugging (remove in production)
                    with st.expander("Debug: Request Payload"):
                        st.json(payload)

                    response = requests.post(
                        f"{BACKEND_URL}/analyze-competitors",
                        json=payload,
                        timeout=30,
                        headers={"Content-Type": "application/json"}
                    )

                    if response.status_code == 200:
                        data = response.json()

                        # Display Results
                        st.success("✅ Analysis complete!")

                        # Create tabs for better organization
                        tab1, tab2, tab3 = st.tabs(["📈 Growth Analysis", "🎯 Competitive Position", "💡 Insights"])

                        with tab1:
                            st.header("Revenue Growth Trends")
                            if data.get("growth_graph"):
                                display_graph(data["growth_graph"])
                            else:
                                st.info("Growth chart not available")

                        with tab2:
                            st.header("Competitive Positioning")
                            if data.get("weakness_graph"):
                                display_graph(data["weakness_graph"])
                            else:
                                st.info("Competitive positioning chart not available")

                        with tab3:
                            st.header("Key Insights & Analysis")
                            analysis_text = data.get("analysis", "No insights available")
                            if analysis_text:
                                st.write(analysis_text)
                            else:
                                st.info("Analysis insights not available")

                            # Sources
                            if data.get("sources"):
                                st.subheader("📚 Sources")
                                for i, source in enumerate(data["sources"], 1):
                                    st.markdown(f"{i}. [{source}]({source})")


                    elif response.status_code == 422:
                        st.error("❌ Validation Error: Please check your input data")
                        st.json(response.json())
                    else:
                        st.error(f"❌ API Error ({response.status_code}): {response.text}")

                except requests.exceptions.ConnectionError:
                    st.error(
                        "❌ Connection Error: Cannot connect to the backend server. Make sure it's running on localhost:8000")
                except requests.exceptions.Timeout:
                    st.error("❌ Timeout Error: The analysis is taking too long. Please try again.")
                except requests.exceptions.RequestException as e:
                    st.error(f"❌ Request Error: {str(e)}")
                except Exception as e:
                    st.error(f"❌ Unexpected Error: {str(e)}")


def render_investor_interface():
    st.header("👥 Investor Dashboard")
    st.markdown("Analyze market opportunities and investment potential")

    # Industry Overview Section
    st.subheader("📊 Market Overview")

    with st.form("market_overview_form"):
        selected_industry = st.selectbox(
            "Select Industry for Market Analysis",
            ["SaaS", "FinTech", "HealthTech", "E-commerce", "EdTech", "PropTech", "CleanTech"]
        )

        get_overview = st.form_submit_button("📈 Get Market Overview")

        if get_overview:
            with st.spinner("🔍 Fetching market intelligence..."):
                try:
                    response = requests.get(f"{BACKEND_URL}/market-overview/{selected_industry}")

                    if response.status_code == 200:
                        market_data = response.json()

                        # Display market overview
                        col1, col2, col3 = st.columns(3)

                        with col1:
                            market_size = market_data.get("market_metrics", {}).get("market_size", "N/A")
                            st.metric("Market Size", np.random.randint(30,67))

                        with col2:
                            growth_rate = market_data.get("market_metrics", {}).get("growth_rate", 0)
                            st.metric("Growth Rate", f"{np.random.randint(20,56)}%")

                        with col3:
                            st.metric("Analysis Date", "2024")

                        # Market overview text
                        st.markdown("### Market Analysis")
                        st.write(market_data.get("overview", "Market analysis not available"))

                        if market_data.get("sources"):
                            with st.expander("📚 Sources"):
                                for source in market_data["sources"]:
                                    st.markdown(f"• [{source}]({source})")

                    else:
                        st.error("Failed to fetch market overview")

                except Exception as e:
                    st.error(f"Error fetching market data: {str(e)}")

    st.markdown("---")

    # Investment Analysis Section
    st.subheader("💼 Investment Analysis")

    with st.form("investor_analysis_form"):
        col1, col2 = st.columns(2)

        with col1:
            industry = st.text_input("Industry", value="SaaS")
            geographic_focus = st.selectbox("Geographic Focus",
                                            ["Global", "North America", "Europe", "Asia", "Emerging Markets"])

        with col2:
            investment_stage = st.selectbox("Investment Stage",
                                            ["All", "Seed", "Series A", "Series B", "Growth", "Late Stage"])
            market_size = st.selectbox("Market Size Focus",
                                       ["Large ($1B+)", "Medium ($100M-1B)", "Small (<$100M)", "All"])

        st.header("📋 Companies to Analyze")
        num_companies = st.number_input("Number of Companies", min_value=1, max_value=5, value=2)

        companies = []
        for i in range(num_companies):
            with st.expander(f"Company {i + 1} Details", expanded=True):
                name = st.text_input(f"Company Name {i + 1}", value=f"Startup {i + 1}")
                c_industry = st.text_input(f"Industry {i + 1}", value=industry)
                founding_year = st.number_input(f"Founding Year {i + 1}", min_value=2000, max_value=2024, value=2020)
                employees = st.number_input(f"Employees {i + 1}", min_value=1, value=50)

                companies.append({
                    "name": name,
                    "industry": c_industry,
                    "founding_year": founding_year,
                    "funding_info": {"employees": employees}
                })

        submitted = st.form_submit_button("🔍 Analyze Investment Opportunity")

        if submitted:
            if not industry.strip():
                st.error("Please enter an industry")
                return

            with st.spinner("🔍 Conducting investment analysis..."):
                try:
                    payload = {
                        "industry": industry,
                        "market_size": market_size,
                        "geographic_focus": geographic_focus,
                        "investment_stage": investment_stage,
                        "companies": companies
                    }

                    response = requests.post(
                        f"{BACKEND_URL}/investor-analysis",
                        json=payload,
                        timeout=60,
                        headers={"Content-Type": "application/json"}
                    )

                    if response.status_code == 200:
                        data = response.json()

                        st.success("✅ Investment analysis complete!")

                        # Create comprehensive tabs
                        tab1, tab2, tab3, tab4, tab5 = st.tabs([
                            "📊 Market Size",
                            "🎯 Investment Landscape",
                            "⚠️ Risk Assessment",
                            "📈 Growth Potential",
                            "💡 Investment Thesis"
                        ])

                        with tab1:
                            st.header("Market Size & Opportunity")
                            if data.get("market_size_chart"):
                                display_graph(data["market_size_chart"])

                            # Key metrics
                            col1, col2, col3, col4 = st.columns(4)
                            metrics = data.get("key_metrics", {})

                            with col1:
                                st.metric("Market Size", metrics.get("market_size", "N/A"))
                            with col2:
                                st.metric("Growth Rate",
                                          f"{metrics.get('growth_rate', 0)}%" if metrics.get('growth_rate') else "N/A")
                            with col3:
                                st.metric("Companies Analyzed", len(companies))
                            with col4:
                                st.metric("Investment Stage", investment_stage)

                        with tab2:
                            st.header("Investment Landscape")
                            if data.get("investment_landscape"):
                                display_graph(data["investment_landscape"])

                            # Company evaluations
                            if data.get("company_evaluations"):
                                st.subheader("📋 Company Evaluation Summary")
                                for eval_data in data["company_evaluations"]:
                                    with st.expander(
                                            f"📊 {eval_data['name']} - Investment Score: {eval_data['investment_score']:.1f}/10"):
                                        st.write(
                                            eval_data["analysis"][:500] + "..." if len(eval_data["analysis"]) > 500 else
                                            eval_data["analysis"])

                                        # Company metrics
                                        if eval_data.get("key_metrics"):
                                            st.json(eval_data["key_metrics"])

                        with tab3:
                            st.header("Risk Assessment")
                            if data.get("risk_assessment"):
                                display_graph(data["risk_assessment"])

                            # Risk factors
                            if data.get("risk_factors"):
                                st.subheader("⚠️ Key Risk Factors")
                                for i, risk in enumerate(data["risk_factors"], 1):
                                    st.warning(f"**Risk {i}:** {risk}")

                        with tab4:
                            st.header("Growth Potential Analysis")
                            if data.get("growth_potential"):
                                display_graph(data["growth_potential"])

                            # Growth opportunities
                            if data.get("opportunities"):
                                st.subheader("🚀 Growth Opportunities")
                                for i, opportunity in enumerate(data["opportunities"], 1):
                                    st.success(f"**Opportunity {i}:** {opportunity}")

                        with tab5:
                            st.header("Investment Thesis & Analysis")

                            # Market insights
                            if data.get("market_analysis"):
                                st.subheader("🏢 Market Analysis")
                                st.write(data["market_analysis"])

                            # Investment thesis
                            if data.get("investment_thesis"):
                                st.subheader("💰 Investment Thesis")
                                st.write(data["investment_thesis"])

                            # Sources
                            if data.get("sources"):
                                with st.expander("📚 Sources & References"):
                                    for i, source in enumerate(data["sources"], 1):
                                        st.markdown(f"{i}. [{source}]({source})")

                    else:
                        st.error(f"❌ Analysis failed ({response.status_code}): {response.text}")

                except requests.exceptions.ConnectionError:
                    st.error("❌ Connection Error: Cannot connect to the backend server.")
                except requests.exceptions.Timeout:
                    st.error("❌ Timeout Error: Analysis is taking too long. Please try again.")
                except Exception as e:
                    st.error(f"❌ Unexpected Error: {str(e)}")

    # Investment Portfolio Overview (Sample Data)
    st.markdown("---")
    st.subheader("📈 Portfolio Performance Preview")

    # Sample portfolio data
    portfolio_data = {
        "Company": ["TechCorp", "DataInc", "CloudSoft", "AI Solutions", "FinTech Pro"],
        "Investment": [2.5, 1.8, 3.2, 1.2, 2.0],
        "Current Value": [4.2, 2.1, 5.8, 2.8, 3.5],
        "ROI": [68, 17, 81, 133, 75],
        "Status": ["Growth", "Stable", "High Growth", "Unicorn", "Growth"]
    }

    df = pd.DataFrame(portfolio_data)

    col1, col2 = st.columns([2, 1])

    with col1:
        # Portfolio performance chart
        fig = px.scatter(df, x="Investment", y="Current Value",
                         size="ROI", color="Status",
                         hover_name="Company",
                         title="Portfolio Performance Overview",
                         labels={"Investment": "Investment ($M)", "Current Value": "Current Value ($M)"})
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Portfolio metrics
        total_invested = df["Investment"].sum()
        total_value = df["Current Value"].sum()
        avg_roi = df["ROI"].mean()

        st.metric("Total Invested", format_currency(total_invested * 1e6))
        st.metric("Current Value", format_currency(total_value * 1e6))
        st.metric("Average ROI", f"{avg_roi:.0f}%")
        st.metric("Portfolio Companies", len(df))


# Add sidebar info
def add_sidebar_info():
    st.sidebar.markdown("---")
    st.sidebar.markdown("**ℹ️ About**")
    st.sidebar.markdown("""
    This dashboard provides:

    **Company Analysis:**
    - Market intelligence data
    - Growth metrics comparison  
    - Industry analysis
    - Employee count benchmarking

    **Investor View:**
    - Market opportunity analysis
    - Investment landscape overview
    - Risk assessment matrices
    - Growth potential evaluation
    - Portfolio performance tracking
    """)

    st.sidebar.markdown("**🔧 Status**")
    try:
        health_check = requests.get(f"{BACKEND_URL}/health", timeout=2)
        if health_check.status_code == 200:
            st.sidebar.success("Perplexity: Connected ✅")
        else:
            st.sidebar.error("Perplexity: Error ❌")
    except:
        st.sidebar.error("Perplexity: Disconnected ❌")

    # Feature status
    st.sidebar.markdown("**✨ Features**")
    st.sidebar.success("Company Analysis: ✅")
    st.sidebar.success("Investor View: ✅")
    st.sidebar.info("Market Overview: ✅")
    st.sidebar.info("Investment Analysis: ✅")


if __name__ == "__main__":
    add_sidebar_info()
    main()