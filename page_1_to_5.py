import streamlit as st
import plotly.express as px
import pandas as pd
import plotly.graph_objects as go

#### PAGE 1 ####

#--------------------------------------------------------
# 💰 Revenue & Growth Analysis
#--------------------------------------------------------

def revenue_growth_analysis(filtered_df):
    st.header("💰 Revenue & Growth Analysis")

    yearly_rev = (
        filtered_df.groupby('order_year')['final_amount_inr']
        .sum()
        .reset_index()
        .sort_values('order_year')
    )
    yearly_rev['YoY_Growth'] = yearly_rev['final_amount_inr'].pct_change() * 100

    fig1 = go.Figure()

    fig1.add_trace(
        go.Bar(
            x=yearly_rev['order_year'],
            y=yearly_rev['final_amount_inr'],
            name="Total Revenue",
            marker_color='royalblue'
        )
    )

    fig1.add_trace(
        go.Scatter(
            x=yearly_rev['order_year'],
            y=yearly_rev['YoY_Growth'],
            name="YoY Growth (%)",
            mode='lines+markers',
            line=dict(color='orange', width=3),
            yaxis='y2'
        )
    )

    fig1.update_layout(
        title="📈 Revenue & YoY Growth (2015–2025)",
        xaxis=dict(title="Year"),
        yaxis=dict(title="Total Revenue (₹)", showgrid=True),
        yaxis2=dict(title="YoY Growth (%)", overlaying='y', side='right'),
        template="plotly_white",
        legend=dict(orientation="h", y=1.15),
        height=550
    )

    st.plotly_chart(fig1, use_container_width=True)

    st.markdown("""
    ### 💡 Key Insights:
    - **YoY Growth** spikes highlight strong sales years, possibly linked to major events or festivals.
    - Identify **consistent growth years** versus **slowdown periods**.
    - Helps guide **strategic forecasting** and **seasonal budget planning**.
    """)

#### PAGE 2 ####

#--------------------------------------------------------
# 🗓️ Seasonality & Trends Analysis
#--------------------------------------------------------

def seasonality_trends(df):
    st.header("🗓️ Monthly & Seasonal Sales Trends (2015–2025)")

    monthly = (
        df.groupby(['order_year', 'order_month'])['final_amount_inr']
        .sum()
        .reset_index()
    )

    month_order = [
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December"
    ]
    monthly['order_month'] = pd.Categorical(monthly['order_month'], categories=month_order, ordered=True)
    monthly = monthly.sort_values(['order_year', 'order_month'])

    fig2 = px.density_heatmap(
        monthly,
        x='order_month',
        y='order_year',
        z='final_amount_inr',
        color_continuous_scale="YlGnBu",
        title="📊 Monthly Sales Heatmap (2015–2025)",
        nbinsx=12
    )
    fig2.update_layout(
        xaxis_title="Month",
        yaxis_title="Year",
        height=550,
        template="plotly_white"
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("📈 Average Monthly Revenue Trend")

    month_avg = (
        df.groupby('order_month')['final_amount_inr']
        .mean()
        .reindex(month_order)
        .reset_index()
    )

    fig3 = px.line(
        month_avg,
        x='order_month',
        y='final_amount_inr',
        markers=True,
        title="Average Monthly Revenue (Seasonal Behavior)",
        line_shape='spline',
        color_discrete_sequence=['#008080']
    )
    fig3.update_layout(xaxis_title="Month", yaxis_title="Avg Revenue (₹)")
    st.plotly_chart(fig3, use_container_width=True)


    st.markdown("""
    ### 💡 Key Insights:
    - Certain months (like **October–December**) typically show **spikes** due to **festive sales (Diwali, Christmas, Year-End Deals)**.
    - **May–July** may show slower performance — good for off-season campaigns.
    - Compare year-on-year heatmap patterns to detect **emerging seasonal consistency**.
    """)

#### PAGE 3 ####

#--------------------------------------------------------
# 📦 Category Performance Analysis
#--------------------------------------------------------

def payment_method_evolution(filtered_df):
    st.header("💵 Evolution of Payment Methods (2015–2025)")

    payment_trends = (
        filtered_df.groupby(['order_year', 'payment_method'])['final_amount_inr']
        .sum()
        .reset_index()
    )

    total_per_year = payment_trends.groupby('order_year')['final_amount_inr'].transform('sum')
    payment_trends['market_share'] = (payment_trends['final_amount_inr'] / total_per_year) * 100


    payment_trends = payment_trends.sort_values(['order_year', 'payment_method'])

    fig6 = px.area(
        payment_trends,
        x='order_year',
        y='market_share',
        color='payment_method',
        line_group='payment_method',
        title='Payment Method Market Share Evolution (2015–2025)',
        color_discrete_sequence=px.colors.qualitative.Bold
    )

    fig6.update_layout(
        xaxis_title="Year",
        yaxis_title="Market Share (%)",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#333", size=13),
        legend_title_text="Payment Method",
        hovermode="x unified",
        title_font=dict(size=20, color="#0077b6"),
    )

    fig6.update_traces(mode="none", hovertemplate="<b>%{x}</b><br>%{legendgroup}: %{y:.2f}%")

    st.plotly_chart(fig6, use_container_width=True)

    latest_year = payment_trends['order_year'].max()
    latest_data = payment_trends[payment_trends['order_year'] == latest_year]
    top_payment = latest_data.loc[latest_data['market_share'].idxmax(), 'payment_method']

    st.info(
        f"📈 **Insight:** In {latest_year}, **{top_payment}** dominated digital payments — "
        f"marking a major shift from traditional methods like COD."
    )

    st.caption("Data: Based on transaction revenue share by payment method per year (2015–2025).")


#### PAGE 4 ####

#--------------------------------------------------------
# "👥 Customer Segmentation (RFM)"
#--------------------------------------------------------

def customer_segmentation_rfm(filtered_df):
    st.header("👥 Customer Segmentation using RFM Analysis")
    st.subheader("📊 RFM Model Overview")
    filtered_df['order_date'] = pd.to_datetime(filtered_df['order_date'], errors='coerce')

    ref_date = filtered_df['order_date'].max()

    rfm = (
        filtered_df.groupby('customer_id', as_index=False)
        .agg({
            'order_date': lambda x: (ref_date - x.max()).days, 
            'transaction_id': 'nunique',                      
            'final_amount_inr': 'sum'                          
        })
        .rename(columns={
            'order_date': 'Recency',
            'transaction_id': 'Frequency',
            'final_amount_inr': 'Monetary'
        })
    )

    rfm['R_Score'] = pd.qcut(rfm['Recency'], 5, labels=[5,4,3,2,1]).astype(int)
    rfm['F_Score'] = pd.qcut(rfm['Frequency'].rank(method='first'), 5, labels=[1,2,3,4,5]).astype(int)
    rfm['M_Score'] = pd.qcut(rfm['Monetary'], 5, labels=[1,2,3,4,5]).astype(int)
    rfm['RFM_Score'] = rfm['R_Score'] + rfm['F_Score'] + rfm['M_Score']

    def segment_customer(row):
        if row['RFM_Score'] >= 13:
            return '💎 VIP / Champions'
        elif row['RFM_Score'] >= 10:
            return '🌟 Loyal Customers'
        elif row['RFM_Score'] >= 7:
            return '🛍️ Regular Buyers'
        elif row['RFM_Score'] >= 5:
            return '🌱 New Customers'
        else:
            return '💤 At Risk / Lost'

    rfm['Segment'] = rfm.apply(segment_customer, axis=1)

    st.subheader("📈 RFM Segmentation Summary")
    seg_counts = rfm['Segment'].value_counts().reset_index()
    seg_counts.columns = ['Segment', 'Customer Count']

    fig_seg = px.pie(
        seg_counts,
        names='Segment',
        values='Customer Count',
        color_discrete_sequence=px.colors.qualitative.Bold,
        title="Customer Segmentation Distribution"
    )
    st.plotly_chart(fig_seg, use_container_width=True)

    st.subheader("🎯 Customer Segmentation Scatter Plots")

    col1, col2 = st.columns(2)

    with col1:
        fig1 = px.scatter(
            rfm, x='Recency', y='Monetary',
            color='Segment',
            size='Frequency',
            hover_data=['customer_id', 'RFM_Score'],
            title="Recency vs Monetary (Bubble = Frequency)"
        )
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        fig2 = px.scatter_3d(
            rfm,
            x='Recency',
            y='Frequency',
            z='Monetary',
            color='Segment',
            title="3D View: RFM Segmentation",
            color_discrete_sequence=px.colors.sequential.Viridis
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("📋 Segment-Wise Summary (Actionable Insights)")
    seg_summary = (
        rfm.groupby('Segment')
        .agg({
            'Recency': 'mean',
            'Frequency': 'mean',
            'Monetary': 'mean',
            'customer_id': 'count'
        })
        .rename(columns={'customer_id': 'Customer Count'})
        .sort_values('Monetary', ascending=False)
        .round(2)
    )

    st.dataframe(seg_summary)

    st.markdown("""
    ### 💡 Key Insights:
    - **💎 VIP / Champions** → Recent, frequent, and high-value spenders — nurture with loyalty programs.
    - **🌟 Loyal Customers** → Buy often, slightly lower spend — target with personalized discounts.
    - **🛍️ Regular Buyers** → Stable but medium activity — engage with cross-sell recommendations.
    - **🌱 New Customers** → Recently joined — encourage repeat purchases with onboarding offers.
    - **💤 At Risk / Lost** → Haven’t purchased in a while — re-engage via retention campaigns.
    """)

#### PAGE 5 ####

#--------------------------------------------------------
# 📦 Category Performance Analysis
#--------------------------------------------------------

def category_performance(filtered_df):
    st.header("📦 Category-Wise Performance Analysis (2015–2025)")

    cat_rev = (
        filtered_df.groupby('category', as_index=False)
        .agg({'final_amount_inr': 'sum'})
        .sort_values('final_amount_inr', ascending=False)
    )

    cat_year = (
        filtered_df.groupby(['order_year', 'category'], as_index=False)['final_amount_inr']
        .sum()
        .sort_values(['category', 'order_year'])
    )

    cat_year['growth_rate'] = cat_year.groupby('category')['final_amount_inr'].pct_change() * 100
    cat_growth = (
        cat_year.groupby('category')['growth_rate']
        .mean()
        .reset_index()
        .fillna(0)
        .sort_values('growth_rate', ascending=False)
    )

    # -------------------------------
    #  Visualization 1: Revenue Bar Chart
    # -------------------------------
    st.subheader("💰 Category Revenue Contribution")
    fig_bar = px.bar(
        cat_rev,
        x='category',
        y='final_amount_inr',
        text_auto='.2s',
        color='final_amount_inr',
        color_continuous_scale='Tealgrn',
        title="Total Revenue by Category (2015–2025)"
    )
    fig_bar.update_layout(
        xaxis_title="Category",
        yaxis_title="Revenue (₹)",
        template="plotly_white",
        showlegend=False
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # -------------------------------
    # 🧩 Treemap Visualization
    # -------------------------------
    st.markdown("## 🪴 Category Revenue Breakdown")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🔹 Category → Brand → Product")
        fig_combined1 = px.treemap(
            filtered_df,
            path=['category', 'brand', 'product_name'],
            values='final_amount_inr',
            color='final_amount_inr',
            color_continuous_scale='RdYlGn',
            title="Revenue Breakdown by Category, Brand & Product"
        )
        fig_combined1.update_traces(textinfo="label+percent parent")
        st.plotly_chart(fig_combined1, use_container_width=True)

    with col2:
        st.markdown("### 🔸 Category → Subcategory")
        fig_combined2 = px.treemap(
            filtered_df,
            path=['category', 'subcategory'],
            values='final_amount_inr',
            color='final_amount_inr',
            color_continuous_scale='Viridis',
            title="Revenue Distribution by Category and Subcategory"
        )
        fig_combined2.update_traces(textinfo="label+percent parent")
        st.plotly_chart(fig_combined2, use_container_width=True)

    # -------------------------------
    # 🔹 Visualization 3: Pie Chart – Market Share
    # -------------------------------
    st.subheader("🥧 Market Share by Category")
    fig_pie = px.pie(
        cat_rev,
        names='category',
        values='final_amount_inr',
        hole=0.35,
        color_discrete_sequence=px.colors.qualitative.Set3,
        title="Category Revenue Share (%)"
    )
    fig_pie.update_traces(textinfo='percent+label', pull=[0.05]*len(cat_rev))
    st.plotly_chart(fig_pie, use_container_width=True)

    # -------------------------------
    # 🔹 Visualization 4: Category Growth Trend
    # -------------------------------
    st.subheader("📈 Category Revenue Growth Over Time")
    fig_line = px.line(
        cat_year,
        x='order_year',
        y='final_amount_inr',
        color='category',
        markers=True,
        line_shape='spline',
        title="Category Revenue Trends (2015–2025)"
    )
    fig_line.update_layout(
        xaxis_title="Year",
        yaxis_title="Revenue (₹)",
        template="plotly_white"
    )
    st.plotly_chart(fig_line, use_container_width=True)

    # -------------------------------
    # 📊 Growth Summary Table
    # -------------------------------
    st.subheader("📋 Average Growth Rate by Category")
    st.dataframe(cat_growth.style.format({'growth_rate': '{:.2f}%'}))

    # -------------------------------
    # 💡 Insights Section
    # -------------------------------
    st.markdown("""
    ### 💡 Key Insights:
    - **Top-performing categories** contribute the majority of total revenue — prioritize them for premium marketing.
    - Categories like **Electronics & Accessories** and **Home Appliances** often show **consistent growth**.
    - **Fashion and Lifestyle** categories exhibit **seasonal spikes**, especially during festive quarters.
    - Subcategories within fast-growing categories can reveal **untapped niches** worth targeting.
    """)

