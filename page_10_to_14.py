import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json, os, requests, urllib.request

#### PAGE 10 ####
# --------------------------------------------------------
# "💰 Price vs Demand Analysis"
# --------------------------------------------------------
def price_demand_analysis(filtered_df):
    st.header("💰 Price vs Demand Relationship Analysis")

    # --- Filter relevant columns ---
    price_df = filtered_df[['category', 'brand', 'discounted_price_inr', 'quantity',
                            'original_price_inr', 'discount_percent', 'final_amount_inr']].copy()

    price_df = price_df[price_df['quantity'] > 0]
    price_df['Revenue'] = price_df['final_amount_inr']

    # --- Category Selector ---
    selected_category = st.selectbox("Select Category", sorted(price_df['category'].dropna().unique()))
    cat_df = price_df[price_df['category'] == selected_category]

    # --- Scatter Plot: Price vs Quantity (Demand Curve) ---
    st.subheader(f"📉 Price vs Quantity Sold — {selected_category}")
    fig_scatter = px.scatter(
        cat_df,
        x='discounted_price_inr',
        y='quantity',
        color='brand',
        trendline='ols',
        opacity=0.7,
        size='Revenue',
        hover_data=['brand', 'discount_percent'],
        title=f"Demand Curve for {selected_category}",
        color_discrete_sequence=px.colors.qualitative.Vivid
    )
    fig_scatter.update_layout(
        xaxis_title="Discounted Price (INR)",
        yaxis_title="Quantity Sold",
        template="plotly_white"
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    # --- Correlation Matrix ---
    st.subheader("🔗 Correlation Between Pricing, Discounts & Demand")

    corr_df = cat_df[['original_price_inr', 'discount_percent', 'discounted_price_inr', 'quantity', 'Revenue']]
    corr_matrix = corr_df.corr()

    fig_corr = px.imshow(
        corr_matrix,
        text_auto=True,
        color_continuous_scale="RdBu_r",
        title=f"Correlation Matrix — {selected_category}"
    )
    fig_corr.update_layout(
        xaxis_title="Metrics",
        yaxis_title="Metrics",
        coloraxis_colorbar=dict(title="Correlation"),
        font=dict(size=13),
        title_font=dict(size=20, color="#0077b6"),
    )
    st.plotly_chart(fig_corr, use_container_width=True)

    # --- Category-Wide Price Elasticity ---
    st.subheader("📊 Category-Wise Price Elasticity Overview")

    elasticity_data = (
        price_df.groupby('category')
        .apply(lambda g: g['discounted_price_inr'].corr(g['quantity']))
        .reset_index(name='Price_Demand_Correlation')
        .sort_values('Price_Demand_Correlation')
    )

    fig_elasticity = px.bar(
        elasticity_data,
        x='category',
        y='Price_Demand_Correlation',
        color='Price_Demand_Correlation',
        color_continuous_scale='Tealgrn',
        title="Category-Wise Price vs Demand Correlation"
    )
    fig_elasticity.update_layout(
        xaxis_title="Category",
        yaxis_title="Correlation (Price vs Demand)",
        template="plotly_white"
    )
    st.plotly_chart(fig_elasticity, use_container_width=True)

    # --- Insights ---
    st.markdown("""
    ### 💡 Key Analytical Insights:
    - **Higher prices** generally correlate with **lower demand**, except for premium or niche categories.
    - **Discount Percent** often shows a **negative correlation** with price and **positive correlation** with quantity sold.
    - **Revenue** correlation helps pinpoint optimal price ranges balancing **volume and profit**.
    - This data helps identify:
        - **Elastic vs Inelastic** product categories.
        - **Optimal discount thresholds** for maximum profitability.
        - **Luxury products** that defy standard demand curves.
    """)

#### PAGE 11 ####

# --------------------------------------------------------
#  "🚚 Delivery Performance Analysis"
# --------------------------------------------------------
def delivery_performance_analysis(filtered_df):
    st.header("🚚 Delivery Performance Analysis")

    # --- Data Preparation ---
    if 'delivery_days' not in filtered_df.columns:
        st.error("❌ 'delivery_days' column missing. Please ensure dataset includes delivery duration per order.")
        return

    delivery_df = filtered_df.copy()

    # Clean invalid or missing delivery values
    delivery_df = delivery_df[delivery_df['delivery_days'].notna()]
    delivery_df = delivery_df[delivery_df['delivery_days'] >= 0]

    # Define On-Time Delivery
    delivery_df['on_time'] = delivery_df['delivery_days'] <= 3

    # --- Distribution of Delivery Days ---
    st.subheader("📦 Distribution of Delivery Days")

    fig_hist = px.histogram(
        delivery_df,
        x='delivery_days',
        nbins=20,
        color='customer_tier',
        marginal='box',
        title="Distribution of Delivery Speed Across Tiers",
        color_discrete_sequence=px.colors.qualitative.Bold
    )
    fig_hist.update_layout(
        xaxis_title="Delivery Days",
        yaxis_title="Number of Orders",
        template="plotly_white"
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    # --- On-Time Delivery Performance ---
    st.subheader("⏱️ On-Time Delivery Rate by City and Tier")

    on_time_df = (
        delivery_df.groupby(['customer_city', 'customer_tier'], as_index=False)
        .agg(on_time_rate=('on_time', 'mean'))
    )
    on_time_df['on_time_rate'] *= 100

    fig_heatmap = px.density_heatmap(
        on_time_df,
        x='customer_tier',
        y='customer_city',
        z='on_time_rate',
        color_continuous_scale='Greens',
        title="On-Time Delivery Rate by City and Tier (%)"
    )
    fig_heatmap.update_layout(
        xaxis_title="Customer Tier",
        yaxis_title="City",
        template="plotly_white"
    )
    st.plotly_chart(fig_heatmap, use_container_width=True)

    # --- Correlation Between Delivery Speed & Rating ---
    if 'customer_rating' in delivery_df.columns:
        st.subheader("⭐ Customer Satisfaction vs Delivery Speed")

        corr_value = delivery_df['delivery_days'].corr(delivery_df['customer_rating'])
        st.caption(f"📊 Correlation between Delivery Speed and Rating: **{corr_value:.2f}**")

        fig_scatter = px.scatter(
            delivery_df,
            x='delivery_days',
            y='customer_rating',
            color='customer_tier',
            trendline='ols',
            opacity=0.7,
            title="Customer Rating vs Delivery Speed",
            color_discrete_sequence=px.colors.qualitative.Vivid
        )
        fig_scatter.update_layout(
            xaxis_title="Delivery Days",
            yaxis_title="Customer Rating",
            template="plotly_white"
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
    else:
        st.warning("⚠️ 'customer_rating' column not found — skipping satisfaction analysis.")

    # --- Tier-Wise Summary ---
    st.subheader("🏙️ Average Delivery Performance by Tier")

    tier_summary = (
        delivery_df.groupby('customer_tier', as_index=False)
        .agg(
            avg_delivery_days=('delivery_days', 'mean'),
            on_time_rate=('on_time', 'mean'),
            avg_rating=('customer_rating', 'mean')
        )
    )
    tier_summary['on_time_rate'] *= 100
    st.dataframe(tier_summary.style.format({
        'avg_delivery_days': '{:.2f}',
        'on_time_rate': '{:.1f}%',
        'avg_rating': '{:.2f}'
    }))

    # --- Insights ---
    st.markdown("""
    ### 💡 Key Insights:
    - **Delivery Speed Distribution** reveals variation across **customer tiers and cities**.
    - **On-time rate** tends to be higher in **Metro and Tier 1** regions due to better logistics infrastructure.
    - Negative correlation between **delivery days** and **customer ratings** suggests faster delivery improves satisfaction.
    - Consider:
        - Strengthening **last-mile delivery** in Tier 2 and rural regions.
        - Offering **Prime or Express Delivery** in high-delay zones.
        - Monitoring **partner performance** in low on-time cities.
    """)

#### PAGE 12 #### 
# --------------------------------------------------------
# "↩️ Product Return & Customer Satisfaction Analysis"
# --------------------------------------------------------

def return_analysis(filtered_df):
    st.header("↩️ Product Return & Customer Satisfaction Analysis")

    df = filtered_df.copy()

    # 🔍 --- STEP 1: Auto-detect Return Column ---
    possible_cols = [col for col in df.columns if 'return' in col.lower()]
    if not possible_cols:
        st.error("❌ No return-related column found (e.g. 'return_status', 'is_returned'). Please verify dataset.")
        st.write("Available columns:", list(df.columns))
        return

    return_col = possible_cols[0]  # Use first detected column
    st.caption(f"✅ Using column **'{return_col}'** as return indicator.")

    # Normalize return column to boolean
    df['is_returned'] = df[return_col].astype(str).str.lower().isin(['yes', 'true', 'returned', '1'])
    df['return_flag'] = df['is_returned'].apply(lambda x: 'Returned' if x else 'Kept')

    # --- Return Rate by Category ---
    st.subheader("📦 Return Rate by Product Category")

    cat_return = (
        df.groupby('category', as_index=False)['is_returned']
        .mean()
        .rename(columns={'is_returned': 'return_rate'})
        .sort_values('return_rate', ascending=False)
    )
    cat_return['return_rate'] *= 100

    fig_cat = px.bar(
        cat_return,
        x='category',
        y='return_rate',
        color='return_rate',
        color_continuous_scale='Reds',
        title="Return Rate by Category (%)",
        text_auto='.2f'
    )
    fig_cat.update_layout(xaxis_title="Category", yaxis_title="Return Rate (%)")
    st.plotly_chart(fig_cat, use_container_width=True)

    # --- Return Rate by Price Range ---
    st.subheader("💰 Return Rate by Price Range")
    df['price_range'] = pd.cut(
        df['discounted_price_inr'],
        bins=[0, 500, 1000, 5000, 10000, 50000, df['discounted_price_inr'].max()],
        labels=['<₹500', '₹500–₹1k', '₹1k–₹5k', '₹5k–₹10k', '₹10k–₹50k', '>₹50k']
    )

    price_return = (
        df.groupby('price_range', as_index=False)['is_returned']
        .mean()
        .rename(columns={'is_returned': 'return_rate'})
    )
    price_return['return_rate'] *= 100

    fig_price = px.line(
        price_return,
        x='price_range',
        y='return_rate',
        markers=True,
        title="Return Rate vs Price Range (%)",
        color_discrete_sequence=['#e63946']
    )
    st.plotly_chart(fig_price, use_container_width=True)

    # --- Correlation with Ratings, Price & Discount ---
    st.subheader("🔗 Correlation Between Returns and Product Attributes")

    corr_df = df[['is_returned', 'customer_rating', 'discount_percent', 'discounted_price_inr', 'final_amount_inr']].copy()
    corr_df['is_returned'] = corr_df['is_returned'].astype(int)

    corr_matrix = corr_df.corr()

    fig_corr = px.imshow(
        corr_matrix,
        text_auto=True,
        color_continuous_scale='RdBu_r',
        title="Return Correlation Matrix"
    )
    st.plotly_chart(fig_corr, use_container_width=True)

    # --- Return Rate by Rating ---
    if 'customer_rating' in df.columns:
        st.subheader("⭐ Return Rate by Customer Rating")

        rating_return = (
            df.groupby('customer_rating', as_index=False)['is_returned']
            .mean()
            .rename(columns={'is_returned': 'return_rate'})
        )
        rating_return['return_rate'] *= 100

        fig_rating = px.bar(
            rating_return,
            x='customer_rating',
            y='return_rate',
            text_auto='.1f',
            color='return_rate',
            color_continuous_scale='YlOrRd',
            title="Return Rate by Product Rating (%)"
        )
        st.plotly_chart(fig_rating, use_container_width=True)

    # --- Treemap: Category → Brand → Return Flag ---
    st.subheader("🌳 Category–Brand Return Treemap")

    fig_tree = px.treemap(
        df,
        path=['category', 'brand', 'return_flag'],
        values='final_amount_inr',
        color='return_flag',
        color_discrete_map={'Returned': '#e76f51', 'Kept': '#2a9d8f'},
        title="Category and Brand Return Composition"
    )
    st.plotly_chart(fig_tree, use_container_width=True)

    # --- Insights ---
    st.markdown("""
    ### 💡 Key Insights:
    - **High return rates** may indicate quality or expectation mismatches.
    - **Mid-range prices** often face the most returns — balancing affordability & expectation.
    - **Low-rated products** have a strong correlation with **higher return probability**.
    - Returns often spike post-festivals or high-discount periods.
    """)

#### PAGE 13 ####

# --------------------------------------------------------
# 🏷️ Brand Performance & Market Share Evolution (2015–2025)
# --------------------------------------------------------
def brand_performance_analysis(filtered_df):
    st.header("🏷️ Brand Performance & Market Share Evolution (2015–2025)")

    df = filtered_df.copy()

    # --- 🧹 Clean data ---
    df = df[df['brand'].notna()]
    df['order_year'] = df['order_year'].astype(int)
    df['final_amount_inr'] = df['final_amount_inr'].astype(float)

    # --- Brand Revenue by Year ---
    st.subheader("💰 Yearly Brand Revenue Trend")
    brand_year = (
        df.groupby(['order_year', 'brand'], as_index=False)['final_amount_inr']
        .sum()
        .sort_values(['brand', 'order_year'])
    )

    top_brands = (
        brand_year.groupby('brand')['final_amount_inr']
        .sum()
        .nlargest(10)
        .index
    )

    top_brand_year = brand_year[brand_year['brand'].isin(top_brands)]

    fig1 = px.line(
        top_brand_year,
        x='order_year',
        y='final_amount_inr',
        color='brand',
        markers=True,
        title="Top 10 Brands by Revenue Over Time"
    )
    fig1.update_layout(
        xaxis_title="Year",
        yaxis_title="Revenue (INR)",
        legend_title="Brand"
    )
    st.plotly_chart(fig1, use_container_width=True)

    # ---  Market Share Evolution (Stacked Area) ---
    st.subheader("📈 Market Share Evolution")
    total_per_year = df.groupby('order_year')['final_amount_inr'].transform('sum')
    df['market_share'] = df['final_amount_inr'] / total_per_year * 100

    market_share = (
        df.groupby(['order_year', 'brand'], as_index=False)['market_share']
        .mean()
        .sort_values(['order_year', 'market_share'], ascending=[True, False])
    )

    fig2 = px.area(
        market_share[market_share['brand'].isin(top_brands)],
        x='order_year',
        y='market_share',
        color='brand',
        title="Brand Market Share Evolution (Top 10 Brands)",
        line_group='brand'
    )
    fig2.update_layout(
        xaxis_title="Year",
        yaxis_title="Market Share (%)",
        legend_title="Brand"
    )
    st.plotly_chart(fig2, use_container_width=True)

    # ---  Brand Performance by Category ---
    st.subheader("🏆 Top Brands by Category")

    cat_brand = (
        df.groupby(['category', 'brand'], as_index=False)['final_amount_inr']
        .sum()
        .sort_values('final_amount_inr', ascending=False)
    )

    selected_category = st.selectbox(
        "Select a category to compare brand performance:",
        sorted(df['category'].dropna().unique())
    )

    cat_data = cat_brand[cat_brand['category'] == selected_category].head(10)

    fig3 = px.bar(
        cat_data,
        x='brand',
        y='final_amount_inr',
        color='brand',
        text_auto='.2s',
        title=f"Top 10 Brands in {selected_category} Category"
    )
    fig3.update_layout(xaxis_title="Brand", yaxis_title="Revenue (INR)", showlegend=False)
    st.plotly_chart(fig3, use_container_width=True)

    # ---  Treemap: Category → Brand ---
    st.subheader("🌳 Brand Market Composition")
    fig4 = px.treemap(
        df,
        path=['category', 'brand','subcategory'],
        values='final_amount_inr',
        color='final_amount_inr',
        color_continuous_scale='Tealgrn',
        title="Category & Brand Revenue Composition"
    )
    st.plotly_chart(fig4, use_container_width=True)

    # ---  Insights ---
    st.markdown("""
    ### 💡 Key Insights:
    - Track how **brand dominance shifts** over time — e.g., emerging vs legacy brands.
    - **Market share trends** reveal competitive pressure and new entrant success.
    - Identify **category leaders** and **brands losing momentum**.
    - Treemaps show **revenue concentration**, useful for strategic partnerships or diversification.
    - Combine with **customer reviews** and **pricing data** for full competitive benchmarking.
    """)

#### PAGE 14 ####

# --------------------------------------------------------
#  "💎 Customer Lifetime Value (CLV) & Cohort Analysis"
# --------------------------------------------------------
def customer_lifetime_value_analysis(filtered_df):
    st.header("💎 Customer Lifetime Value (CLV) & Cohort Analysis")

    df = filtered_df.copy()
    df['order_date'] = pd.to_datetime(df['order_date'], errors='coerce')

    # ---  Define Customer Cohorts ---
    df['order_year'] = df['order_date'].dt.year
    df['order_month'] = df['order_date'].dt.to_period('M')

    first_purchase = df.groupby('customer_id')['order_month'].min().reset_index()
    first_purchase.columns = ['customer_id', 'acquisition_month']

    df = df.merge(first_purchase, on='customer_id')
    df['cohort_month'] = df['acquisition_month'].dt.to_timestamp()
    df['months_since_acquisition'] = (
        (df['order_date'].dt.year - df['cohort_month'].dt.year) * 12 +
        (df['order_date'].dt.month - df['cohort_month'].dt.month)
    )

    # ---  Cohort Retention Analysis ---
    cohort_data = (
        df.groupby(['cohort_month', 'months_since_acquisition'])['customer_id']
        .nunique()
        .reset_index()
    )

    cohort_size = (
        cohort_data[cohort_data['months_since_acquisition'] == 0]
        .set_index('cohort_month')['customer_id']
    )

    cohort_data['cohort_size'] = cohort_data['cohort_month'].map(cohort_size)
    cohort_data['retention'] = cohort_data['customer_id'] / cohort_data['cohort_size']

    # --- Heatmap: Retention Matrix ---
    retention_matrix = cohort_data.pivot(
        index='cohort_month',
        columns='months_since_acquisition',
        values='retention'
    )

    st.subheader("📈 Cohort Retention Heatmap")
    fig_retention = px.imshow(
        retention_matrix,
        aspect='auto',
        color_continuous_scale='Blues',
        title="Customer Retention by Cohort (Months Since Acquisition)",
        labels=dict(x="Months Since Acquisition", y="Cohort (Acquisition Month)", color="Retention Rate")
    )
    st.plotly_chart(fig_retention, use_container_width=True)

    # ---  Compute Customer Lifetime Value (CLV) ---
    clv_df = (
        df.groupby('customer_id', as_index=False)
        .agg({
            'final_amount_inr': 'sum',
            'order_year': 'min',
            'customer_tier': 'first',
            'customer_age_group': 'first'
        })
        .rename(columns={'final_amount_inr': 'CLV', 'order_year': 'Acquisition_Year'})
    )

    # ---  CLV Distribution ---
    st.subheader("💰 CLV Distribution Across Customers")
    fig_clv = px.histogram(
        clv_df,
        x='CLV',
        nbins=50,
        color='customer_tier',
        marginal='box',
        title="Customer Lifetime Value Distribution by Tier",
        color_discrete_sequence=px.colors.qualitative.Vivid
    )
    fig_clv.update_layout(xaxis_title="Customer Lifetime Value (₹)", yaxis_title="Customer Count")
    st.plotly_chart(fig_clv, use_container_width=True)

    # --- Average CLV by Segment ---
    clv_segment = (
        clv_df.groupby(['customer_tier', 'customer_age_group'], as_index=False)['CLV']
        .mean()
        .sort_values('CLV', ascending=False)
    )

    st.subheader("📊 Average CLV by Tier & Age Group")
    fig_segment = px.bar(
        clv_segment,
        x='customer_tier',
        y='CLV',
        color='customer_age_group',
        barmode='group',
        title="Average Customer Lifetime Value by Segment"
    )
    fig_segment.update_layout(xaxis_title="Customer Tier", yaxis_title="Avg CLV (₹)")
    st.plotly_chart(fig_segment, use_container_width=True)

    # --- CLV Over Acquisition Years ---
    st.subheader("📆 CLV by Acquisition Year")
    clv_year = (
        clv_df.groupby('Acquisition_Year', as_index=False)['CLV']
        .mean()
        .sort_values('Acquisition_Year')
    )

    fig_year = px.line(
        clv_year,
        x='Acquisition_Year',
        y='CLV',
        markers=True,
        title="Average CLV Over Acquisition Years"
    )
    fig_year.update_layout(xaxis_title="Acquisition Year", yaxis_title="Average CLV (₹)")
    st.plotly_chart(fig_year, use_container_width=True)

    # --- 7️⃣ Insights ---
    st.markdown("""
    ### 💡 Key Insights:
    - **Retention Heatmap** helps identify strong and weak cohorts — a drop after month 3 often signals engagement issues.
    - **CLV Distribution** shows a skew — a small % of customers generate most of the revenue.
    - **Higher-tier and older age groups** tend to have higher lifetime value.
    - **Cohort-based CLV growth** highlights successful customer acquisition campaigns.
    - Use this data to:
        - Design **loyalty programs** for high CLV segments.
        - Improve **retention rates** within first few months.
        - Identify **profitable acquisition periods**.
    """)
