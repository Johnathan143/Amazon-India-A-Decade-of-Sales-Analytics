import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from page_1_to_5 import revenue_growth_analysis, seasonality_trends, category_performance, payment_method_evolution, customer_segmentation_rfm
from page_6_to_09 import prime_membership_impact, geographic_analysis, festival_sales_impact, age_group_behavior
from page_10_to_14 import price_demand_analysis, delivery_performance_analysis,return_analysis, brand_performance_analysis, customer_lifetime_value_analysis
from page_15_to_20 import discount_promotion_analysis, product_rating_analysis,customer_journey_analysis, product_lifecycle_analysis, competitive_pricing_analysis, business_health_dashboard


st.set_page_config(
    page_title="Amazon India Sales Dashboard (2015–2025)",
    layout="wide",
    page_icon="🛒"
)

# Sidebar Navigation
st.sidebar.title("📚 Navigation")
page = st.sidebar.radio(
    "Go to:",
    [
        "🏠 Overview Dashboard",
        "📈 Revenue & Growth",
        "🗓️ Seasonality & Trends",
        "👥 Customer Segmentation (RFM)",
        "💵 Payment Method Evolution",
        "📦 Category Performance",
        "💎 Prime Membership Impact",
        "🧭 Geographic Analysis",
        "🎉 Festival Sales Impact",
        "🧠 Age Group Behavior",
        "💰 Price vs Demand Analysis",
        "🚚 Delivery Performance Analysis",
        "↩️ Product Return & Customer Satisfaction Analysis",
        "🏷️ Brand Performance & Market Share Evolution",
        "💎 Customer Lifetime Value (CLV) & Cohort Analysis",
        "🎯 Discount & Promotion Effectiveness",
        "⭐ Product Rating Impact on Sales Performance",
        "🛤️ Customer Journey & Purchase Evolution Analysis",
        "📦 Inventory & Product Lifecycle Analysis",
        "📊 Competitive Pricing Analysis",
        "📊 Business Health Dashboard (2015–2025)",
    ]
)

@st.cache_data
def load_data():
    df = pd.read_csv("Amazon_india_decade_sales_data(cleaned).csv")
    df['order_date'] = pd.to_datetime(df['order_date'], errors='coerce')
    df['order_year'] = df['order_date'].dt.year
    df['order_month'] = df['order_date'].dt.month_name()
    return df

df = load_data()

# Sidebar filters
st.sidebar.markdown("---")
st.sidebar.header("🔍 Filters")
selected_year = st.sidebar.multiselect("Select Year(s):", sorted(df['order_year'].unique()), default=df['order_year'].max())
selected_category = st.sidebar.multiselect("Select Category:", df['category'].dropna().unique())
selected_brand = st.sidebar.multiselect("Select Brand:", df['brand'].dropna().unique())

filtered_df = df[
    (df['order_year'].isin(selected_year)) &
    (df['category'].isin(selected_category) if selected_category else True) &
    (df['brand'].isin(selected_brand) if selected_brand else True)
]

# 🏠 Overview Dashboard
if page == "🏠 Overview Dashboard":
    st.title("🛒 Amazon India – A Decade of E-commerce (2015–2025)")
    st.caption("Built by Mizaru | Advanced Analytics Dashboard")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("💰 Total Revenue", f"₹{filtered_df['final_amount_inr'].sum():,.0f}")
    col2.metric("📦 Orders", f"{filtered_df['transaction_id'].nunique():,}")
    col3.metric("👥 Unique Customers", f"{filtered_df['customer_id'].nunique():,}")
    col4.metric("⭐ Avg Rating", f"{filtered_df['customer_rating'].astype(float).mean():.2f}")

    st.markdown("### 📈 Revenue Growth Over Time")
    yearly_rev = filtered_df.groupby('order_year')['final_amount_inr'].sum().reset_index()
    fig = px.line(yearly_rev, x='order_year', y='final_amount_inr', markers=True,
                  title="Total Revenue Trend (2015–2025)", color_discrete_sequence=['#0078D4'])
    st.plotly_chart(fig, use_container_width=True)

elif page == "📈 Revenue & Growth":
    revenue_growth_analysis(filtered_df)

elif page == "🗓️ Seasonality & Trends":
    seasonality_trends(filtered_df)

elif page == "📦 Category Performance":
    category_performance(filtered_df)

elif page == "💵 Payment Method Evolution":
    payment_method_evolution(filtered_df)

elif page == "👥 Customer Segmentation (RFM)":
    customer_segmentation_rfm(filtered_df)

elif page == "💎 Prime Membership Impact":
    prime_membership_impact(filtered_df)

elif page == "🧭 Geographic Analysis":
    geographic_analysis(filtered_df)

elif page == "🎉 Festival Sales Impact":
    festival_sales_impact(filtered_df)

elif page == "🧠 Age Group Behavior":
    age_group_behavior(filtered_df)

elif page == "💰 Price vs Demand Analysis":
    price_demand_analysis(filtered_df)

elif page == "🚚 Delivery Performance Analysis":
    delivery_performance_analysis(filtered_df)

elif page == "↩️ Product Return & Customer Satisfaction Analysis":
    return_analysis(filtered_df)

elif page == "🏷️ Brand Performance & Market Share Evolution":
    brand_performance_analysis(filtered_df)

elif page == "💎 Customer Lifetime Value (CLV) & Cohort Analysis":
    customer_lifetime_value_analysis(filtered_df)

elif page == "🎯 Discount & Promotion Effectiveness":
    discount_promotion_analysis(filtered_df)

elif page == "⭐ Product Rating Impact on Sales Performance":
    product_rating_analysis(filtered_df)

elif page == "🛤️ Customer Journey & Purchase Evolution Analysis":
    customer_journey_analysis(filtered_df)

elif page == "📦 Inventory & Product Lifecycle Analysis":
    product_lifecycle_analysis(filtered_df)

elif page == "📊 Competitive Pricing Analysis":
    competitive_pricing_analysis(filtered_df)

elif page == "📊 Business Health Dashboard (2015–2025)":
    business_health_dashboard(filtered_df)