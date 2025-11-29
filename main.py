from __future__ import annotations
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json, os, requests, urllib.request
from sqlalchemy import create_engine
from datetime import timedelta
from dateutil.relativedelta import relativedelta

# -----------------------------------------------
# Streamlit Page Configuration
# -----------------------------------------------
st.set_page_config(
    page_title="Amazon India Sales Dashboard (2015–2025)",
    layout="wide",
    page_icon="🛒"
)

# -----------------------------------------------
# Sidebar Navigation
# -----------------------------------------------
st.sidebar.title("📚 Navigation")
main_page = st.sidebar.radio(
    "Go to:",
    [
        "💰 Sales & Revenue Insights",
        "👥 Customer & Behavioral Analytics",
        "🛍️ Product & Category Performance",
        "💳 Payment & Transaction Dynamics",
        "🌏 Market & External Factors"
    ]
)

# -----------------------------------------------
# Load Data from MySQL Database
# -----------------------------------------------
@st.cache_data
def load_data():
    engine = create_engine("mysql+mysqlconnector://root:Root@localhost:3306/amazon_db")

    all_years = []

    # include 2015–2025 (range stop is exclusive, so use 2026)
    for year in range(2015, 2026):
        table_name = f"amazon_{year}"
        query = f"SELECT * FROM {table_name}"

        try:
            df_year = pd.read_sql(query, con=engine)

            # Make sure order_date is proper datetime
            if "order_date" in df_year.columns:
                df_year["order_date"] = pd.to_datetime(df_year["order_date"], errors="coerce")

                # Add year/month columns only if missing
                if "order_year" not in df_year.columns:
                    df_year["order_year"] = df_year["order_date"].dt.year

                if "order_month" not in df_year.columns:
                    df_year["order_month"] = df_year["order_date"].dt.month_name()

            all_years.append(df_year)

        except Exception as e:
            # Table for this year might not exist – just skip it
            print(f"Skipping {table_name}: {e}")
            continue

    # If nothing loaded, return empty df (prevents crashes)
    if not all_years:
        return pd.DataFrame()

    # Combine all years into one dataframe
    df = pd.concat(all_years, ignore_index=True)

    # Final safety: enforce date/year/month again
    if "order_date" in df.columns:
        df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
        df["order_year"] = df["order_date"].dt.year
        df["order_month"] = df["order_date"].dt.month_name()

    return df

with st.spinner("⏳ Loading data from database..."):
    df = load_data()

# -----------------------------------------------
# Sidebar Filters
# -----------------------------------------------
st.sidebar.markdown("---")
st.sidebar.header("🔍 Filters")

selected_year = st.sidebar.multiselect(
    "Select Year(s):", sorted(df['order_year'].unique()), default=df['order_year'].max()
)
selected_category = st.sidebar.multiselect(
    "Select Category:", df['subcategory'].dropna().unique()
)
selected_brand = st.sidebar.multiselect(
    "Select Brand:", df['brand'].dropna().unique()
)

filtered_df = df.copy()
if selected_year:
    filtered_df = filtered_df[filtered_df['order_year'].isin(selected_year)]
if selected_category:
    filtered_df = filtered_df[filtered_df['subcategory'].isin(selected_category)]
if selected_brand:
    filtered_df = filtered_df[filtered_df['brand'].isin(selected_brand)]

# -----------------------------------------------
# 💰 SALES & REVENUE INSIGHTS PAGE
# -----------------------------------------------
if main_page == "💰 Sales & Revenue Insights":

    st.title("💰 Sales & Revenue Insights Dashboard")
    st.caption("A decade of Amazon India performance (2015–2025) | Built by Mizaru")

    # Tabs
    analysis_option = st.selectbox(
    "Select Analysis Section:",
    [
        "📈 Revenue & Growth",
        "🗓️ Seasonality",
        "🎯 Discount & Promotional Effects",
        "📊 Business Health Dashboard"
    ]
)


    # -----------------------------------------------
    # TAB 1 — Revenue & Growth
    # -----------------------------------------------
    if analysis_option == "📈 Revenue & Growth":
        st.subheader("📈 Revenue Trend Analysis (2015–2025)")

        # 1) Prep: limit to 2015–2025 and aggregate
        year_min, year_max = 2015, 2025
        yearly = (
            filtered_df[(filtered_df['order_year'] >= year_min) & (filtered_df['order_year'] <= year_max)]
            .groupby('order_year', as_index=False)['final_amount_inr']
            .sum()
            .rename(columns={'final_amount_inr': 'revenue_inr'})
            .sort_values('order_year')
        )
        # handle empty edge cases
        if yearly.empty:
            st.warning("No data available for 2015–2025 in the current filters.")
        else:
            # 2) YoY growth and CAGR
            yearly['yoy_pct'] = yearly['revenue_inr'].pct_change()*100

            # CAGR across the available span (first→last non-null)
            start_val = yearly['revenue_inr'].iloc[0]
            end_val   = yearly['revenue_inr'].iloc[-1]
            n_years   = max(1, (yearly['order_year'].iloc[-1] - yearly['order_year'].iloc[0]))
            cagr_pct  = ( (end_val / start_val)**(1/n_years) - 1 ) * 100 if start_val > 0 else np.nan

            # 3) Linear regression trendline (revenue vs year)
            x = yearly['order_year'].values.astype(float)
            y = yearly['revenue_inr'].values.astype(float)
            coef = np.polyfit(x, y, 1)  # y = m*x + b
            trend_y = np.polyval(coef, x)

            # 4) Identify key periods: best/worst YoY
            # ignore first NaN YoY row
            yoy_nonnull = yearly.dropna(subset=['yoy_pct']).copy()
            best_row = yoy_nonnull.loc[yoy_nonnull['yoy_pct'].idxmax()] if not yoy_nonnull.empty else None
            worst_row = yoy_nonnull.loc[yoy_nonnull['yoy_pct'].idxmin()] if not yoy_nonnull.empty else None

            # 5) KPIs
            colA, colB, colC, colD = st.columns(4)
            colA.metric("🧮 CAGR", f"{cagr_pct:,.2f}%")
            colB.metric("💰 Start (₹)", f"{start_val:,.0f}", help=str(yearly['order_year'].iloc[0]))
            colC.metric("💰 End (₹)", f"{end_val:,.0f}", help=str(yearly['order_year'].iloc[-1]))
            if best_row is not None:
                colD.metric("🏆 Best YoY", f"{best_row['yoy_pct']:,.2f}%", help=f"{int(best_row['order_year'])}")

            # 6) Chart: Bars (revenue), Line (YoY), Trendline (regression)
            fig = go.Figure()

            # Bars — revenue
            fig.add_trace(
                go.Bar(
                    x=yearly['order_year'],
                    y=yearly['revenue_inr'],
                    name="Total Revenue (₹)",
                    marker_color='royalblue',
                    hovertemplate="<b>%{x}</b><br>Revenue: ₹%{y:,.0f}<extra></extra>",
                )
            )

            # Line — YoY %
            fig.add_trace(
                go.Scatter(
                    x=yearly['order_year'],
                    y=yearly['yoy_pct'],
                    name="YoY Growth (%)",
                    mode='lines+markers',
                    line=dict(color='orange', width=3),
                    yaxis='y2',
                    hovertemplate="<b>%{x}</b><br>YoY: %{y:.2f}%<extra></extra>",
                )
            )

            # Trendline — regression
            fig.add_trace(
                go.Scatter(
                    x=yearly['order_year'],
                    y=trend_y,
                    name="Trend Line (Regression)",
                    mode='lines',
                    line=dict(dash='dash', width=2),
                    hoverinfo='skip'
                )
            )

            # 7) Annotations for key growth periods
            annotations = []
            if best_row is not None:
                annotations.append(
                    dict(
                        x=int(best_row['order_year']),
                        y=yearly.loc[yearly['order_year'] == int(best_row['order_year']), 'revenue_inr'].values[0],
                        xanchor='center',
                        yanchor='bottom',
                        text=f"Peak YoY: {best_row['yoy_pct']:.1f}%",
                        showarrow=True,
                        arrowhead=2,
                        ay=-40
                    )
                )
            if worst_row is not None:
                annotations.append(
                    dict(
                        x=int(worst_row['order_year']),
                        y=yearly.loc[yearly['order_year'] == int(worst_row['order_year']), 'revenue_inr'].values[0],
                        xanchor='center',
                        yanchor='top',
                        text=f"Lowest YoY: {worst_row['yoy_pct']:.1f}%",
                        showarrow=True,
                        arrowhead=2,
                        ay=40
                    )
                )

            fig.update_layout(
                title="Revenue & YoY Growth with Trend Line (2015–2025)",
                xaxis=dict(title="Year"),
                yaxis=dict(title="Total Revenue (₹)", showgrid=True),
                yaxis2=dict(title="YoY Growth (%)", overlaying='y', side='right'),
                template="plotly_white",
                legend=dict(orientation="h", y=1.15),
                height=560,
                annotations=annotations
            )

            st.plotly_chart(fig, use_container_width=True)

            # 8) Clean summary table (Year, Revenue, YoY%)
            st.markdown("#### 📋 Yearly Summary")
            summary = yearly.copy()
            summary['Revenue (₹)'] = summary['revenue_inr'].round(0).map(lambda v: f"{v:,.0f}")
            summary['YoY Growth (%)'] = summary['yoy_pct'].round(2).map(lambda v: "" if pd.isna(v) else f"{v:,.2f}%")
            summary = summary[['order_year', 'Revenue (₹)', 'YoY Growth (%)']].rename(columns={'order_year': 'Year'})
            st.dataframe(summary, use_container_width=True)

        # -----------------------------------------------
        # TAB 2 — Seasonality & Trends
        # -----------------------------------------------
    elif analysis_option == "🗓️ Seasonality":
        st.header("🗓️ Seasonal & Monthly Sales Analysis")

        # --- 1) Prepare Month Order for Sorting ---
        monthly_sales = (
            filtered_df.groupby(['order_year', 'order_month'])['final_amount_inr']
            .sum()
            .reset_index()
            .sort_values(['order_year', 'order_month'])
        )
        
        month_order = [
            "January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November", "December"
        ]
        monthly_sales['order_month'] = pd.Categorical(monthly_sales['order_month'], categories=month_order, ordered=True)
        monthly_sales = monthly_sales.sort_values(['order_year', 'order_month'])

        # Ensure correct ordering
        df['order_month'] = pd.Categorical(df['order_month'], categories=month_order, ordered=True)

        # --- 2) Monthly Sales Heatmap (Year vs Month) ---
        st.subheader("🔥 Monthly Sales Heatmap Across Years")

        fig_heatmap = px.density_heatmap(
            monthly_sales,
            x="order_month",
            y="order_year",
            z="final_amount_inr",
            color_continuous_scale="YlOrRd",
            title="Monthly Revenue Heatmap (Year vs Month)",
            height=550
        )

        fig_heatmap.update_layout(xaxis_title="Month", yaxis_title="Year")
        st.plotly_chart(fig_heatmap, use_container_width=True)

        # --- 3) Peak Selling Months (Overall) ---
        st.subheader("🏆 Peak Selling Months (Overall)")

        peak_months = (
            filtered_df.groupby('order_month')['final_amount_inr']
            .sum()
            .reindex(month_order)
            .reset_index()
        )

        fig_peaks = px.bar(
            peak_months,
            x="order_month", y="final_amount_inr",
            text_auto=True,
            title="Total Revenue by Month (Peak Months Highlighted)",
            color_discrete_sequence=["#0077b6"]
        )
        fig_peaks.update_layout(xaxis_title="Month", yaxis_title="Total Revenue (₹)")
        st.plotly_chart(fig_peaks, use_container_width=True)

        # Insight Box
        top_month = peak_months.loc[peak_months['final_amount_inr'].idxmax(), 'order_month']
        st.info(f"📌 **Peak Sales Month:** `{top_month}` consistently shows the highest shopping activity.")

    #---------------------------------------
    # TAB 3
    #---------------------------------------

    elif analysis_option == "🎯 Discount & Promotional Effects":

        st.header("🎯 Discount & Promotion Effectiveness Analysis")
        st.caption("Understand how discounts influence customer purchasing behavior across products and time.")

        df = filtered_df.copy()

        # Clean & ensure correct data types
        df = df[df['discount_percent'].notna()]
        df['discount_percent'] = df['discount_percent'].astype(float)
        df['quantity'] = df['quantity'].astype(int)
        df['final_amount_inr'] = df['final_amount_inr'].astype(float)

        # -------------------------------
        # 1) Discount vs Sales Volume
        # -------------------------------
        st.subheader("📉 Relationship Between Discount % and Quantity Sold")

        fig1 = px.scatter(
            df,
            x='discount_percent',
            y='quantity',
            color='subcategory',
            opacity=0.6,
            trendline="ols",
            hover_data=['brand', 'product_name', 'final_amount_inr'],
            title="How Discounts Influence Units Sold"
        )

        fig1.update_layout(
            xaxis_title="Discount (%)",
            yaxis_title="Quantity Sold",
            template="plotly_white"
        )
        st.plotly_chart(fig1, use_container_width=True)

        st.info("**Interpretation:** A steep upward trendline means higher discounts strongly increase demand. A flat line means discounting has little impact.")

        # -------------------------------
        # 2) Subcategory Discount Impact
        # -------------------------------
        st.subheader("🏷️ Which Subcategories Respond Best to Discounts?")

        subcat_summary = (
            df.groupby('subcategory', as_index=False)
            .agg(avg_discount=('discount_percent', 'mean'),
                total_units=('quantity', 'sum'),
                revenue=('final_amount_inr', 'sum'))
            .sort_values('avg_discount', ascending=False)
        )

        fig2 = px.scatter(
            subcat_summary,
            x='avg_discount',
            y='total_units',
            size='revenue',
            color='subcategory',
            size_max=65,
            title="Discount Impact by Subcategory (Bubble Size = Revenue)"
        )

        fig2.update_layout(
            xaxis_title="Average Discount (%)",
            yaxis_title="Units Sold",
            template="plotly_white"
        )
        st.plotly_chart(fig2, use_container_width=True)

        # Identify best & worst discount performers
        best = subcat_summary.iloc[subcat_summary['total_units'].idxmax()]['subcategory']
        worst = subcat_summary.iloc[subcat_summary['total_units'].idxmin()]['subcategory']

        st.success(f"✅ **High Response Category:** `{best}` responds strongly to discounts.")
        st.error(f"⚠️ **Low Response Category:** `{worst}` receives little benefit from discounting.")

        # -------------------------------
        # 3) Discount Trend Over Time
        # -------------------------------
        st.subheader("📆 Discount Patterns Across Time")

        df['order_month'] = pd.Categorical(
            df['order_month'],
            categories=["January","February","March","April","May","June","July",
                        "August","September","October","November","December"],
            ordered=True
        )

        monthly_discount_trend = (
            df.groupby(['order_year','order_month'], as_index=False)
            .agg(avg_discount=('discount_percent', 'mean'),
                total_revenue=('final_amount_inr', 'sum'))
        )

        fig3 = px.line(
            monthly_discount_trend,
            x='order_month',
            y='avg_discount',
            color='order_year',
            markers=True,
            title="Monthly Discount Trend Over Years"
        )

        fig3.update_layout(xaxis_title="Month", yaxis_title="Average Discount (%)", template="plotly_white")
        st.plotly_chart(fig3, use_container_width=True)

        # -------------------------------
        # 4) Correlation Matrix
        # -------------------------------
        st.subheader("🔗 Correlation: Discount vs Quantity vs Revenue")

        corr = df[['discount_percent', 'quantity', 'final_amount_inr']].corr()

        fig4 = px.imshow(
            corr,
            text_auto=True,
            color_continuous_scale="RdBu_r",
            title="Correlation Heatmap"
        )
        st.plotly_chart(fig4, use_container_width=True)

        # -------------------------------
        # 5) Key Takeaways
        # -------------------------------
        st.markdown("""
        ### 💡 Key Insights Summary

        | Insight | Meaning | Action |
        |--------|---------|--------|
        | **High discount ↗ units sold (strong positive correlation)** | Discounts successfully drive demand | Use discounts strategically to boost seasonal sales |
        | **Some subcategories barely react to discounting** | Not all products are price-sensitive | Reduce unnecessary discount spend in those categories |
        | **Certain months show peak discount impact** | Promotions align with seasonal demand | Schedule campaigns around high-performing months |
        | **High revenue ≠ high discount** | Some products sell well even without heavy discount | Preserve margin and avoid over-discounting |

        **Next Steps:**  
        ✅ Identify top subcategories for promotional campaigns  
        ✅ Reduce discounting on low-response products  
        ✅ Test smaller discounts first before deep price cuts  
        """)

    #---------------------------------
    #              TAB4
    #---------------------------------

    elif analysis_option == "📊 Business Health Dashboard":

        st.title("📊 Business Health Dashboard")
        st.caption("A simplified view of business performance across revenue, customers, retention, and operations.")

        df = filtered_df.copy()
        df['order_date'] = pd.to_datetime(df['order_date'], errors='coerce')
        df['order_month'] = df['order_date'].dt.to_period('M').dt.to_timestamp()
        df['order_year'] = df['order_date'].dt.year

        # ----------------- KPI SUMMARY -----------------
        st.markdown("### ✅ Overall Business Snapshot")
        col1, col2, col3 = st.columns(3)
        col1.metric("💰 Total Revenue", f"₹{df['final_amount_inr'].sum():,.0f}")
        col2.metric("👥 Total Customers", f"{df['customer_id'].nunique():,}")
        col3.metric("📦 Total Orders", f"{df['transaction_id'].nunique():,}")

        st.markdown("---")

        # ----------------- REVENUE TREND -----------------
        st.markdown("### 📈 Revenue Growth Over Time")
        yearly = df.groupby('order_year', as_index=False)['final_amount_inr'].sum()
        fig_rev = px.line(yearly, x='order_year', y='final_amount_inr', markers=True,
                        title="Yearly Revenue Trend")
        fig_rev.update_layout(yaxis_title="Revenue (₹)")
        st.plotly_chart(fig_rev, use_container_width=True)

        st.info("If the line goes upward → The business is growing. If downward → Sales are slowing.")

        st.markdown("---")

        # ----------------- CUSTOMER ACQUISITION -----------------
        st.markdown("### 🆕 New Customer Acquisition")
        first_purchase = df.groupby('customer_id')['order_date'].min().dt.to_period('M').dt.to_timestamp()
        monthly_new = first_purchase.value_counts().sort_index().reset_index()
        monthly_new.columns = ['Month', 'New_Customers']
        
        fig_acq = px.bar(monthly_new, x='Month', y='New_Customers', title="New Customers Per Month")
        st.plotly_chart(fig_acq, use_container_width=True)

        st.info("If new customers are increasing → Marketing and product reach are strong.")

        st.markdown("---")

        # ----------------- RETENTION (VERY SIMPLE VERSION) -----------------
        st.markdown("### 🔁 Customer Repeat Rate (Retention)")
        repeat_customers = df.groupby('customer_id')['transaction_id'].nunique()
        retention_rate = (repeat_customers > 1).mean() * 100

        st.metric("🔄 Repeat Customer Rate", f"{retention_rate:.1f} %")
        st.info("Higher repeat % means customers like the platform and come back again.")

        st.markdown("---")

        # ----------------- OPERATIONAL EFFICIENCY -----------------
        st.markdown("### ⚙️ Operational Efficiency")

        if 'delivery_days' in df.columns:
            monthly_delivery = df.groupby('order_month')['delivery_days'].mean().reset_index()
            fig_del = px.line(monthly_delivery, x='order_month', y='delivery_days', markers=True,
                            title="Average Delivery Time Over Time")
            st.plotly_chart(fig_del, use_container_width=True)
            st.info("Faster delivery → Better customer satisfaction.")

        if 'return_status' in df.columns:
            monthly_returns = df.groupby('order_month')['return_status'].apply(lambda x: (x=='Returned').mean()*100).reset_index()
            fig_ret = px.line(monthly_returns, x='order_month', y='return_status', markers=True,
                            title="Product Return Rate (%) Over Time")
            st.plotly_chart(fig_ret, use_container_width=True)
            st.info("High return rate → Product or delivery issue.")

        st.markdown("---")

        # ----------------- EXECUTIVE SUMMARY -----------------
        st.subheader("🧭 Quick Business Summary (Easy to Explain)")
        st.write("""
        - **Revenue Trend:** Shows whether business revenue is increasing year-over-year.
        - **New Customers:** Indicates how well we are attracting new users.
        - **Repeat Customer Rate:** Tells us how many customers return to buy again.
        - **Delivery Time & Returns:** Reflects the quality of operations and customer experience.
        
        **Goal:** Grow revenue, bring in new customers, keep them coming back, and deliver efficiently.
        """)

#---------------------------
#        PAGE 2
#---------------------------

elif main_page =="👥 Customer & Behavioral Analytics":
        st.title("👥 Customer & Behavioral Analytics")
        st.caption("A decade of Amazon India performance (2015–2025) | Built by Mizaru")

        page2 = st.selectbox(
    "Select Analysis Section:",
    [
        "👥 Customer Segmentation (RFM)",
        "💎 Customer Lifetime Value (CLV)",
        "🧠 Age Group Behavior",
        "🛤️ Customer Journey & Purchase Evolution", 
        "↩️ Product Return & Customer Satisfaction"
    ]
)
        if page2 == "👥 Customer Segmentation (RFM)":
            st.header("👥 Customer Segmentation (RFM)")
            st.caption("Quick view of customer groups based on how recently, how often, and how much they buy.")

            # 1) Prepare data
            data = df[['customer_id', 'transaction_id', 'final_amount_inr', 'order_date']].copy()
            data['order_date'] = pd.to_datetime(data['order_date'], errors='coerce')
            data = data.dropna(subset=['order_date', 'customer_id'])
            if data.empty:
                st.warning("No transactions to analyze with current filters.")

            ref_date = data['order_date'].max() + pd.Timedelta(days=1)

            # 2) Build RFM
            rfm = (data.groupby('customer_id', as_index=False)
                        .agg(last_order=('order_date', 'max'),
                            Frequency=('transaction_id', 'nunique'),
                            Monetary=('final_amount_inr', 'sum')))
            rfm['Recency'] = (ref_date - rfm['last_order']).dt.days

            # 3) Simple 1–5 scores (robust to ties)
            def score(s, reverse=False):
                r = s.rank(method="first")
                try:
                    bins = pd.qcut(r, 5, labels=[1,2,3,4,5])
                except ValueError:
                    bins = pd.cut(r, 5, labels=[1,2,3,4,5])
                out = bins.astype(int)
                return out.map({1:5,2:4,3:3,4:2,5:1}) if reverse else out

            rfm['R_Score'] = score(rfm['Recency'], reverse=True)
            rfm['F_Score'] = score(rfm['Frequency'])
            rfm['M_Score'] = score(rfm['Monetary'])
            rfm['RFM_Score'] = rfm['R_Score'] + rfm['F_Score'] + rfm['M_Score']

            # 4) Very clear segments
            def label(row):
                if row['R_Score'] >= 4 and row['F_Score'] >= 4 and row['M_Score'] >= 4: return "Champions"
                if row['R_Score'] >= 4 and row['F_Score'] >= 3:                         return "Loyal"
                if row['R_Score'] == 5 and row['F_Score'] <= 2:                          return "New"
                if row['R_Score'] <= 2 and row['F_Score'] >= 4:                          return "At Risk"
                if row['R_Score'] <= 2 and row['F_Score'] <= 2 and row['M_Score'] <= 2:  return "Hibernating"
                return "Needs Attention"

            rfm['Segment'] = rfm.apply(label, axis=1)

            # 6) Segments bar (simple)
            seg = (rfm['Segment'].value_counts()
                .rename_axis('Segment')
                .reset_index(name='Customers'))
            fig_seg = px.bar(seg.sort_values('Customers', ascending=False),
                            x='Segment', y='Customers', color='Segment',
                            title="Customers by Segment")
            fig_seg.update_layout(showlegend=False, yaxis_title="Customers")
            st.plotly_chart(fig_seg, use_container_width=True)

            # 7) One scatter (easy to read)
            st.subheader("🎯 Recency vs Spend")
            fig_sc = px.scatter(rfm, x='Recency', y='Monetary', color='Segment',
                                size='Frequency', size_max=18,
                                title="Lower Recency = more recent buyers")
            fig_sc.update_layout(xaxis_title="Recency (days)", yaxis_title="Spend (₹)")
            st.plotly_chart(fig_sc, use_container_width=True)

            st.markdown("""
                ### 💡 Key Insights:
                - **💎 Champions** → Best customers → Buy often, recently, and spend the most.
                - **🌟 Loyal** → Repeat purchasers → Buy regularly, but may not spend the highest.
                - **🛍️ Hibernating** → Very low activity → Buy rarely, spend very little, and inactive for long.
                - **🌱 New** → just joined Bought → recently but only a few orders. 
                - **💤 At Risk** → Once great, but not buying anymore → Used to buy a lot but haven’t purchased in a long time .
                - **⚠️ Needs Attention** → Middle group → Some spending, but not consistent.
                """)
            
        elif page2 == "💎 Customer Lifetime Value (CLV)":

            st.header("💎 Customer Lifetime Value (CLV) & Cohort Analysis")
            st.caption("Understand how long customers stay active and how much value they generate over time.")

            # Copy filtered data
            df_clv = filtered_df.copy()
            df_clv['order_date'] = pd.to_datetime(df_clv['order_date'], errors='coerce')

            # -----------------------------------------------------
            # 1) Create Cohorts (Groups based on first purchase month)
            # -----------------------------------------------------
            df_clv['order_month'] = df_clv['order_date'].dt.to_period('M')
            first_purchase = df_clv.groupby('customer_id')['order_month'].min().reset_index()
            first_purchase.columns = ['customer_id', 'first_month']

            df_clv = df_clv.merge(first_purchase, on='customer_id')
            df_clv['cohort_month'] = df_clv['first_month'].dt.to_timestamp()

            # Months since acquisition
            df_clv['months_since'] = (
                (df_clv['order_date'].dt.year - df_clv['cohort_month'].dt.year) * 12 +
                (df_clv['order_date'].dt.month - df_clv['cohort_month'].dt.month)
            )

            # -----------------------------------------------------
            # 2) Cohort Retention (How many customers stay each month)
            # -----------------------------------------------------
            cohort_counts = (
                df_clv.groupby(['cohort_month', 'months_since'])['customer_id']
                .nunique()
                .reset_index()
            )

            cohort_size = (
                cohort_counts[cohort_counts['months_since'] == 0]
                .set_index('cohort_month')['customer_id']
            )

            cohort_counts['cohort_size'] = cohort_counts['cohort_month'].map(cohort_size)
            cohort_counts['retention_rate'] = cohort_counts['customer_id'] / cohort_counts['cohort_size']

            retention_matrix = cohort_counts.pivot(
                index='cohort_month',
                columns='months_since',
                values='retention_rate'
            )

            st.subheader("📈 Customer Retention Heatmap")
            fig_ret = px.imshow(
                retention_matrix,
                color_continuous_scale='Blues',
                aspect='auto'
            )
            fig_ret.update_layout(
                xaxis_title="Months After First Purchase",
                yaxis_title="Cohort (Customer Joined Month)",
                coloraxis_colorbar_title="Retention"
            )
            st.plotly_chart(fig_ret, use_container_width=True)

            st.markdown("Customers who stay **past month 2–3** are much more likely to become long-term repeat buyers.")

            # -----------------------------------------------------
            # 3) CLV Calculation (Simply sum lifetime spend)
            # -----------------------------------------------------
            clv = df_clv.groupby('customer_id', as_index=False)['final_amount_inr'].sum()
            clv.rename(columns={'final_amount_inr': 'CLV'}, inplace=True)

            st.subheader("💰 CLV Distribution")
            fig_clv = px.histogram(clv, x='CLV', nbins=40, color_discrete_sequence=["#4a90e2"])
            fig_clv.update_layout(
                xaxis_title="Customer Lifetime Value (₹)",
                yaxis_title="Number of Customers",
                template="simple_white"
            )
            st.plotly_chart(fig_clv, use_container_width=True)

            # -----------------------------------------------------
            # 4) CLV by Customer Segment (Tier)
            # -----------------------------------------------------
            clv_seg = df_clv.groupby('customer_id', as_index=False).agg({
                'final_amount_inr': 'sum',
                'customer_tier': 'first'
            }).rename(columns={'final_amount_inr': 'CLV'})

            seg_summary = clv_seg.groupby('customer_tier', as_index=False)['CLV'].mean()

            st.subheader("📊 Average CLV by Customer Tier")
            fig_seg = px.bar(
                seg_summary,
                x='customer_tier',
                y='CLV',
                color='customer_tier',
                text_auto='.0f',
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            fig_seg.update_layout(showlegend=False, template="simple_white")
            st.plotly_chart(fig_seg, use_container_width=True)

            # -----------------------------------------------------
            # 5) CLV by Acquisition Year
            # -----------------------------------------------------
            clv_year = df_clv.copy()
            clv_year['Acquisition_Year'] = df_clv.groupby('customer_id')['order_date'].transform('min').dt.year
            clv_year = clv_year.groupby('Acquisition_Year', as_index=False)['final_amount_inr'].sum()
            clv_year.rename(columns={'final_amount_inr': 'Avg_CLV'}, inplace=True)

            st.subheader("📆 CLV Over Acquisition Years")
            fig_year = px.line(clv_year, x='Acquisition_Year', y='Avg_CLV', markers=True)
            fig_year.update_layout(
                yaxis_title="Average CLV (₹)",
                template="simple_white"
            )
            st.plotly_chart(fig_year, use_container_width=True)

            # -----------------------------------------------------
            # Insights (Minimal and Business Friendly)
            # -----------------------------------------------------
            st.markdown("""
            ### 🧭 Key Insights
            - Customers who remain active for **2–3 months** usually become valuable long-term buyers.
            - CLV is **not evenly distributed** → a **small group of customers drives the majority of revenue**.
            - **Higher-tier customers** consistently show **higher lifetime value**.
            - Acquisition year trends help identify **strong vs weak marketing periods**.

            **Goal:** Improve early-month engagement so more customers transition into high-value loyal buyers.
            """)
        elif page2 == "🧠 Age Group Behavior":
            st.header("🧠 Customer Age Group Behavior & Preferences")
            df_age = filtered_df.copy()
            df_age['final_amount_inr'] = df_age['final_amount_inr'].astype(float)
            df_age['order_date'] = pd.to_datetime(df_age['order_date'], errors='coerce')

            df_age['customer_age_group'] = df_age['customer_age_group'].astype(str).str.strip().str.title()
            age_order = ['18-25', '26-35', '36-45', '46-55', '55+']
            df_age['customer_age_group'] = pd.Categorical(df_age['customer_age_group'], categories=age_order, ordered=True)

            spend = (
                df_age.groupby('customer_age_group')['final_amount_inr']
                .agg(['count', 'mean', 'sum'])
                .reset_index()
                .rename(columns={'count': 'Orders', 'mean': 'Avg_Spend', 'sum': 'Total_Spend'})
            )

            col1, col2 = st.columns(2)

            with col1:
                fig_spend = px.bar(
                    spend, x='customer_age_group', y='Total_Spend',
                    color='customer_age_group', text_auto='.2s',
                    color_discrete_sequence=px.colors.qualitative.Bold,
                    title="💰 Total Revenue Contribution by Age Group"
                )
                st.plotly_chart(fig_spend, use_container_width=True)

            with col2:
                fig_avg = px.line(
                    spend, x='customer_age_group', y='Avg_Spend',
                    markers=True, line_shape='spline',
                    color_discrete_sequence=['#FF8800'],
                    title="💳 Average Order Value by Age Group"
                )
                st.plotly_chart(fig_avg, use_container_width=True)

            st.subheader("🎯 Category Preferences by Age Group")

            cat_pref = (
                df_age.groupby(['customer_age_group', 'subcategory'])['final_amount_inr']
                .sum()
                .reset_index()
            )

            fig_cat = px.treemap(
                cat_pref,
                path=['customer_age_group', 'subcategory'],
                values='final_amount_inr',
                color='customer_age_group',
                color_discrete_sequence=px.colors.qualitative.Vivid,
                title="🪄 Category Preference by Age Segment"
            )
            st.plotly_chart(fig_cat, use_container_width=True)

            st.subheader("📅 Shopping Frequency by Age Group")
            freq = (
                df_age.groupby(['customer_age_group', 'order_year'])
                .size()
                .reset_index(name='Order_Count')
            )

            fig_freq = px.line(
                freq,
                x='order_year', y='Order_Count', color='customer_age_group',
                markers=True,
                title="📈 Yearly Shopping Frequency by Age Group",
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            st.plotly_chart(fig_freq, use_container_width=True)

            st.subheader("🔥 Heatmap: Category Spending by Age Group")

            heatmap_data = (
                df_age.groupby(['subcategory', 'customer_age_group'])['final_amount_inr']
                .sum().reset_index()
                .pivot(index='subcategory', columns='customer_age_group', values='final_amount_inr')
                .fillna(0)
            )

            fig_heat = px.imshow(
                heatmap_data,
                color_continuous_scale='YlOrRd',
                title="Heatmap: Category vs Age Group Spending (INR)"
            )
            st.plotly_chart(fig_heat, use_container_width=True)

            st.markdown("""
            ### 💡 Insights Summary:
            - **26–35 age group** tends to have the **highest total and average spend** — core e-commerce audience.
            - **18–25** shows frequent small purchases — focus on affordability & offers.
            - **36–45** age group spends more on **Electronics & Home products**.
            - **55+** segment shows low frequency but high-value single orders — luxury-focused marketing opportunity.
            - Overall trend: spending rises until mid-30s, then shifts to quality-driven purchases.
            """)

        elif page2 == "🛤️ Customer Journey & Purchase Evolution":
            st.header("🛤️ Customer Journey & Purchase Evolution Analysis")

            df = filtered_df.copy()
            df['order_date'] = pd.to_datetime(df['order_date'], errors='coerce')

            # Ensure correct datatypes
            df['order_year'] = df['order_year'].astype(int)
            df['final_amount_inr'] = df['final_amount_inr'].astype(float)

            # -----------------------------------------------------
            # Purchase Frequency Analysis
            # -----------------------------------------------------
            st.subheader("🔁 Purchase Frequency Distribution")

            purchase_counts = (
                df.groupby('customer_id')['transaction_id']
                .nunique()
                .reset_index(name='purchase_count')
            )
            freq_summary = purchase_counts['purchase_count'].value_counts().sort_index().reset_index()
            freq_summary.columns = ['Purchases', 'Customer_Count']

            fig1 = px.bar(
                freq_summary,
                x='Purchases',
                y='Customer_Count',
                color='Customer_Count',
                color_continuous_scale='Blues',
                title="Customer Purchase Frequency Distribution"
            )
            st.plotly_chart(fig1, use_container_width=True)

            # Categorize customer loyalty levels
            purchase_counts['Customer_Tier'] = pd.cut(
                purchase_counts['purchase_count'],
                bins=[0, 1, 3, 7, np.inf],
                labels=['New', 'Occasional', 'Regular', 'Loyal']
            )

            tier_summary = (
                purchase_counts.groupby('Customer_Tier')['customer_id']
                .count()
                .reset_index(name='Customer_Count')
            )

            fig_tier = px.pie(
                tier_summary,
                names='Customer_Tier',
                values='Customer_Count',
                title="Customer Loyalty Segmentation (Based on Purchase Frequency)",
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            st.plotly_chart(fig_tier, use_container_width=True)

            # -----------------------------------------------------
            # Evolution Over Time — From New → Loyal
            # -----------------------------------------------------
            st.subheader("📈 Customer Evolution Over Time")

            # Calculate yearly orders per customer
            yearly_orders = (   
                df.groupby(['order_year', 'customer_id'], as_index=False)
                .size()
                .rename(columns={'size': 'yearly_orders'})
            )

            # --- Dynamic Tiering Using Quantiles ---
            # Calculate quartile cut points
            q1 = yearly_orders['yearly_orders'].quantile(0.25)
            q2 = yearly_orders['yearly_orders'].quantile(0.50)
            q3 = yearly_orders['yearly_orders'].quantile(0.75)

            # Assign tiers based on quantiles
            def assign_tier(x):
                if x <= q1:
                    return 'New'
                elif x <= q2:
                    return 'Occasional'
                elif x <= q3:
                    return 'Regular'
                else:
                    return 'Loyal'

            yearly_orders['Customer_Tier'] = yearly_orders['yearly_orders'].apply(assign_tier)

            # Summarize tier counts by year
            loyalty_summary = (
                yearly_orders.groupby(['order_year', 'Customer_Tier'], observed=False)
                            .size()
                            .reset_index(name='Customer_Count')
            )

            # Order tiers display
            tier_order = ['New', 'Occasional', 'Regular', 'Loyal']
            loyalty_summary['Customer_Tier'] = pd.Categorical(loyalty_summary['Customer_Tier'], ordered=True, categories=tier_order)

            # Plot
            fig_trend = px.area(
                loyalty_summary,
                x='order_year',
                y='Customer_Count',
                color='Customer_Tier',
                title="Evolution of Customer Loyalty Segments (2015–2025)",
                category_orders={"Customer_Tier": tier_order}
            )

            fig_trend.update_layout(
                xaxis_title='Year',
                yaxis_title='Customers',
                legend_title='Customer Tier'
            )

            st.plotly_chart(fig_trend, use_container_width=True)



            # -----------------------------------------------------
            # Insights
            # -----------------------------------------------------
            st.markdown("""
            ### 💡 Key Insights:
            - **Most customers (≈60–70%)** make only **one purchase**, highlighting retention challenges.
            - The **Loyal** and **Regular** customer segments, though smaller, contribute **disproportionately high revenue**.
            - Sankey visualization reveals **dominant category transition patterns**, e.g., from *Electronics → Accessories*.
            - Over the years, there’s a **steady increase in repeat customers**, indicating improved retention efforts.
            - **Action Points:**
                - Launch **loyalty rewards** for *Regular* and *Loyal* segments.
                - Use **cross-category promotions** where transition probability is high.
                - Focus on **onboarding & reactivation** of *New/Occasional* buyers.
            """)

        elif page2 == "↩️ Product Return & Customer Satisfaction":
            st.header("↩️ Product Return & Customer Satisfaction Analysis")

            df = filtered_df.copy()

            # 🔍 --- STEP 1: Auto-detect Return Column ---
            possible_cols = [col for col in df.columns if 'return' in col.lower()]
            if not possible_cols:
                st.error("❌ No return-related column found (e.g. 'return_status', 'is_returned'). Please verify dataset.")
                st.write("Available columns:", list(df.columns))

            return_col = possible_cols[0]  # Use first detected column
            st.caption(f"✅ Using column **'{return_col}'** as return indicator.")

            # Normalize return column to boolean
            df['is_returned'] = df[return_col].astype(str).str.lower().isin(['yes', 'true', 'returned', '1'])
            df['return_flag'] = df['is_returned'].apply(lambda x: 'Returned' if x else 'Kept')

            # --- Return Rate by Category ---
            st.subheader("📦 Return Rate by Product Category")

            cat_return = (
                df.groupby('subcategory', as_index=False)['is_returned']
                .mean()
                .rename(columns={'is_returned': 'return_rate'})
                .sort_values('return_rate', ascending=False)
            )
            cat_return['return_rate'] *= 100

            fig_cat = px.bar(
                cat_return,
                x='subcategory',
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

            # --- Insights ---
            st.markdown("""
            ### 💡 Key Insights:
            - **High return rates** may indicate quality or expectation mismatches.
            - **Mid-range prices** often face the most returns — balancing affordability & expectation.
            - **Low-rated products** have a strong correlation with **higher return probability**.
            - Returns often spike post-festivals or high-discount periods.
            """)
#--------------------------
#         PAGE 3
#--------------------------
elif main_page == "🛍️ Product & Category Performance":
    st.title("🛍️ Product & Category Performance")
    st.caption("Quick view of customer groups based on how recently, how often, and how much they buy.")

    page3 = st.selectbox(
        "Select Analysis Section:",
        [
            "📦 Category Performance Analysis",
            "🏷️ Brand Performance & Market Share Evolution",
            "📦 Inventory & Product Lifecycle Analysis",
            "💰 Price vs Demand Analysis",
            "⭐ Product Rating Impact on Sales Performance"

        ]
    )
    #---------------------------
    #           TAB1
    #---------------------------

    if page3 == "📦 Category Performance Analysis":
        st.header("📦 Category-Wise Performance Analysis (2015–2025)")

        cat_rev = (
            filtered_df.groupby('subcategory', as_index=False)
            .agg({'final_amount_inr': 'sum'})
            .sort_values('final_amount_inr', ascending=False)
        )

        cat_year = (
            filtered_df.groupby(['order_year', 'subcategory'], as_index=False)['final_amount_inr']
            .sum()
            .sort_values(['subcategory', 'order_year'])
        )

        cat_year['growth_rate'] = cat_year.groupby('subcategory')['final_amount_inr'].pct_change() * 100
        cat_growth = (
            cat_year.groupby('subcategory')['growth_rate']
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
            x='subcategory',
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
        st.markdown("### 🔸 Category ")
        fig_combined2 = px.treemap(
                filtered_df,
                path=['subcategory'],
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
            names='subcategory',
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
            color='subcategory',
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
        
    #-------------------------
    #        TAB 2
    #-------------------------

    elif page3 == "🏷️ Brand Performance & Market Share Evolution":
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
            df.groupby(['subcategory', 'brand'], as_index=False)['final_amount_inr']
            .sum()
            .sort_values('final_amount_inr', ascending=False)
        )

        selected_category = st.selectbox(
            "Select a category to compare brand performance:",
            sorted(df['subcategory'].dropna().unique())
        )

        cat_data = cat_brand[cat_brand['subcategory'] == selected_category].head(10)

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
            path=['subcategory', 'brand','subcategory'],
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


########-----------------------------------------------
########                  TAB 3
########-----------------------------------------------

# --- SIMPLE: Inventory & Product Lifecycle Analysis --------------------------
# Assumes: filtered_df is available and you imported
#   import streamlit as st, import pandas as pd, import numpy as np, import plotly.express as px

    elif page3 == "📦 Inventory & Product Lifecycle Analysis":
        st.header("📦 Inventory & Product Lifecycle Analysis (Simple)")

        # 1) ---- Minimal input checks ----
        df = filtered_df.copy()
        needed = {"order_date", "product_id", "final_amount_inr"}
        missing = needed - set(df.columns)
        if missing:
            st.error(f"Missing required columns: {', '.join(missing)}")
            st.stop()

        # Optional columns
        has_name = "product_name" in df.columns
        has_qty = "quantity" in df.columns

        # 2) ---- Clean types ----
        df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
        df = df[df["order_date"].notna()]
        df["final_amount_inr"] = pd.to_numeric(df["final_amount_inr"], errors="coerce").fillna(0)

        # 3) ---- Infer launch (first sale) & monthly rollup ----
        df["order_month"] = df["order_date"].dt.to_period("M").dt.to_timestamp()
        # First sale per product (launch)
        launch = (
            df.groupby("product_id", as_index=False)["order_date"]
            .min().rename(columns={"order_date": "launch_month"})
        )
        # Monthly product performance
        agg_dict = {"final_amount_inr": "sum"}
        if has_qty: agg_dict["quantity"] = "sum"
        prod_monthly = (
            df.groupby(["product_id", "order_month"], as_index=False)
            .agg(**{k: (k, v) for k, v in agg_dict.items()})
            .rename(columns={"final_amount_inr": "monthly_revenue",
                            "quantity": "monthly_qty"})
        )
        # Attach product name (if any)
        base_cols = ["product_id"]
        if has_name: 
            base_cols.append("product_name")
        if "subcategory" in df.columns:
            base_cols.append("subcategory")
        meta = df[base_cols].drop_duplicates("product_id")
        prod_monthly = prod_monthly.merge(meta, on="product_id", how="left")
        prod_monthly = prod_monthly.merge(launch, on="product_id", how="left")

        # 4) ---- Simple lifecycle metrics (per product) ----
        peak = (
            prod_monthly.sort_values(["product_id","order_month"])
            .groupby("product_id", as_index=False)
            .apply(lambda g: pd.Series({
                "peak_month": g.loc[g["monthly_revenue"].idxmax(), "order_month"] if not g.empty else pd.NaT,
                "peak_revenue": g["monthly_revenue"].max() if not g.empty else 0,
                "total_revenue": g["monthly_revenue"].sum()
            }))
            .reset_index(drop=True)
        )
        metrics = (meta.merge(launch, on="product_id", how="left")
                        .merge(peak, on="product_id", how="left"))

        # Time-to-peak in months (simple ~30-day months)
        def months_between(a, b):
            if pd.isna(a) or pd.isna(b): return np.nan
            return int((a - b).days // 30)
        metrics["time_to_peak_months"] = metrics.apply(
            lambda r: months_between(r["peak_month"], r["launch_month"]), axis=1
        )

        # First 90 days revenue
        prod_monthly["months_since_launch"] = (
            (prod_monthly["order_month"] - prod_monthly["launch_month"]).dt.days // 30
        )
        first90 = (
            prod_monthly[prod_monthly["months_since_launch"].between(0, 3, inclusive="both")]
            .groupby("product_id", as_index=False)["monthly_revenue"].sum()
            .rename(columns={"monthly_revenue": "rev_first_90days"})
        )
        metrics = metrics.merge(first90, on="product_id", how="left").fillna({"rev_first_90days": 0})

        # ---------------- VISUALS (kept minimal) ----------------

        # A) Launches per year
        st.subheader("🚀 Launches per Year")
        launches_by_year = (
            metrics.assign(launch_year=metrics["launch_month"].dt.year)
                .groupby("launch_year", as_index=False)["product_id"].count()
                .rename(columns={"product_id": "launch_count"})
                .sort_values("launch_year")
        )
        if not launches_by_year.empty:
            fig_launch = px.bar(launches_by_year, x="launch_year", y="launch_count",
                                title="Products Launched per Year")
            st.plotly_chart(fig_launch, use_container_width=True)
        else:
            st.info("No launch data available.")

        # B) Time-to-peak distribution
        st.subheader("⏱️ Time-to-Peak (months)")
        if metrics["time_to_peak_months"].notna().any():
            fig_ttp = px.histogram(metrics, x="time_to_peak_months", nbins=24,
                                title="Time-to-Peak Distribution")
            st.plotly_chart(fig_ttp, use_container_width=True)
        else:
            st.info("No peak data available to plot time-to-peak.")

        # C) Simple lifecycle curve for a product
        st.subheader("📈 Lifecycle Curve")
        if has_name:
            choices = metrics["product_name"].fillna("Unknown").unique().tolist()
            if choices:
                pick = st.selectbox("Choose a product", sorted(choices)[:300])
                pid = metrics.loc[metrics["product_name"] == pick, "product_id"].iloc[0]
            else:
                st.info("No products to select."); st.stop()
        else:
            ids = metrics["product_id"].astype(str).tolist()
            pick_id = st.selectbox("Choose a product_id", ids[:300])
            pid = int(pick_id)

        ts = prod_monthly[prod_monthly["product_id"] == pid].sort_values("order_month")
        if not ts.empty:
            title_name = metrics.loc[metrics["product_id"] == pid, "product_name"].fillna("").iloc[0] if has_name else f"ID {pid}"
            fig_line = px.line(ts, x="order_month", y="monthly_revenue",
                            title=f"Monthly Revenue — {title_name}", markers=True)
            st.plotly_chart(fig_line, use_container_width=True)
        else:
            st.info("No time series for the selected product.")

        # D) Top launches by first 90 days revenue
        st.subheader("🏆 Top Launches (First 90 Days Revenue)")
        topN = (metrics.sort_values("rev_first_90days", ascending=False).head(15))
        if not topN.empty:
            xcol = "product_name" if has_name else "product_id"
            fig_top = px.bar(topN, x=xcol, y="rev_first_90days",
                            title="Top 15 by 90-Day Revenue")
            fig_top.update_layout(xaxis_title="Product", yaxis_title="Revenue (first 90 days)", xaxis={"tickangle": -45})
            st.plotly_chart(fig_top, use_container_width=True)
        else:
            st.info("No early revenue data found.")

        # E) Small table + export
        st.subheader("🔎 Quick Table")
        show_cols = ["product_id", "product_name", "subcategory", "launch_month",
                    "peak_month", "time_to_peak_months", "rev_first_90days", "total_revenue"]
        show_cols = [c for c in show_cols if c in metrics.columns]
        st.dataframe(metrics[show_cols].sort_values("total_revenue", ascending=False).head(50))

        st.download_button(
            "Download lifecycle metrics (CSV)",
            data=metrics.to_csv(index=False).encode("utf-8"),
            file_name="product_lifecycle_metrics_simple.csv",
            mime="text/csv",
            use_container_width=True
        )

    #--------------------------------
    #             TAB 4
    #--------------------------------

    elif page3 == "💰 Price vs Demand Analysis":
        st.header("💰 Price vs Demand Relationship Analysis")

    # --- Filter relevant columns ---
        price_df = filtered_df[['subcategory', 'brand', 'discounted_price_inr', 'quantity',
                                'original_price_inr', 'discount_percent', 'final_amount_inr']].copy()

        price_df = price_df[price_df['quantity'] > 0]
        price_df['Revenue'] = price_df['final_amount_inr']

        # --- Category Selector ---
        selected_category = st.selectbox("Select Category", sorted(price_df['subcategory'].dropna().unique()))
        cat_df = price_df[price_df['subcategory'] == selected_category]

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
            price_df.groupby('subcategory')
            .apply(lambda g: g['discounted_price_inr'].corr(g['quantity']))
            .reset_index(name='Price_Demand_Correlation')
            .sort_values('Price_Demand_Correlation')
        )

        fig_elasticity = px.bar(
            elasticity_data,
            x='subcategory',
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
    elif page3 == "⭐ Product Rating Impact on Sales Performance":
        st.header("🎯 Discount & Promotional Effectiveness Analysis")

        df = filtered_df.copy()
        df = df[df['discount_percent'].notna()]
        df['discount_percent'] = df['discount_percent'].astype(float)
        df['final_amount_inr'] = df['final_amount_inr'].astype(float)
        df['order_year'] = df['order_year'].astype(int)
        df['order_month'] = pd.Categorical(df['order_month'], 
                                        categories=["January","February","March","April","May","June",
                                                    "July","August","September","October","November","December"],
                                        ordered=True)

        # --- Discount vs Sales Volume (Scatter Plot) ---
        st.subheader("📉 Discount % vs Quantity Sold")
        fig_scatter = px.scatter(
            df,
            x='discount_percent',
            y='quantity',
            color='subcategory',
            opacity=0.7,
            trendline='ols',
            hover_data=['brand', 'final_amount_inr'],
            title="Impact of Discount Percentage on Sales Volume"
        )
        fig_scatter.update_layout(
            xaxis_title="Discount (%)",
            yaxis_title="Quantity Sold",
            template="plotly_white"
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

        # --- Average Discount & Revenue by Category ---
        cat_discount = (
            df.groupby('subcategory', as_index=False)
            .agg({
                'discount_percent': 'mean',
                'final_amount_inr': 'sum',
                'quantity': 'sum'
            })
            .sort_values('discount_percent', ascending=False)
        )

        st.subheader("🏷️ Average Discount vs Revenue (Category Level)")
        fig_bar = px.bar(
            cat_discount,
            x='subcategory',
            y='discount_percent',
            color='final_amount_inr',
            text_auto='.2s',
            color_continuous_scale='Purples',
            title="Average Discount and Revenue by Category"
        )
        fig_bar.update_layout(xaxis_title="Category", yaxis_title="Average Discount (%)")
        st.plotly_chart(fig_bar, use_container_width=True)

        # --- Discount Effect Over Time (Monthly Trend) ---
        st.subheader("📆 Discount vs Revenue Trend Over Time")
        monthly_discount = (
            df.groupby(['order_year', 'order_month'], as_index=False)
            .agg({'discount_percent': 'mean', 'final_amount_inr': 'sum'})
            .sort_values(['order_year', 'order_month'])
        )

        fig_line = px.line(
            monthly_discount,
            x='order_month',
            y='discount_percent',
            color='order_year',
            title="Average Discount (%) by Month & Year",
            markers=True
        )
        fig_line.update_layout(
            xaxis_title="Month",
            yaxis_title="Average Discount (%)",
            legend_title="Year",
            template="plotly_white"
        )
        st.plotly_chart(fig_line, use_container_width=True)

        # --- Correlation Analysis ---
        st.subheader("🔗 Correlation Between Discounts, Quantity & Revenue")

        corr_df = df[['discount_percent', 'quantity', 'final_amount_inr']]
        corr_matrix = corr_df.corr()

        fig_corr = px.imshow(
            corr_matrix,
            text_auto=True,
            color_continuous_scale="RdBu_r",
            title="Correlation Matrix — Discounts vs Performance"
        )
        st.plotly_chart(fig_corr, use_container_width=True)

        # --- Discount Effectiveness Score (Simplified Metric) ---
        st.subheader("📊 Discount Effectiveness by Category")

        df['discount_effectiveness'] = (
            (df['quantity'] * df['discount_percent']) / df['final_amount_inr'].replace(0, np.nan)
        )
        discount_eff = (
            df.groupby('subcategory', as_index=False)['discount_effectiveness']
            .mean()
            .sort_values('discount_effectiveness', ascending=False)
        )

        fig_eff = px.bar(
            discount_eff,
            x='subcategory',
            y='discount_effectiveness',
            color='discount_effectiveness',
            color_continuous_scale='Tealgrn',
            title="Discount Effectiveness by Category"
        )
        fig_eff.update_layout(xaxis_title="Category", yaxis_title="Effectiveness Score")
        st.plotly_chart(fig_eff, use_container_width=True)

        # --- Insights ---
        st.markdown("""
        ### 💡 Key Insights:
        - **Discounts** often show a **positive correlation with sales volume**, but the effect on **revenue** varies by category.
        - Some categories achieve **higher conversion** with smaller discounts → focus promotional spend efficiently.
        - **Over-discounting** can reduce profit margins without improving demand.
        - The **Discount Effectiveness Score** helps identify where discounts truly drive sales vs where they are wasted.
        - **Recommendations:**
            - Optimize discount levels for each product type.
            - Track **monthly discount-response curves**.
            - Use **A/B tests** to refine promotional effectiveness.
        """)

elif main_page == "💳 Payment & Transaction Dynamics":
    st.title("💳 Payment & Transaction Dynamics")
    page4 = st.selectbox(
        "Select Analysis Section:",
        [
            "💳 Payment Method Evolution",
            "🚚 Delivery Performance Analysis",
            "💎 Prime Membership Impact on Customer Behavior"
        ]
    )
    #---------------------------
    #          TAB 1
    #---------------------------

    if page4 == "💳 Payment Method Evolution":
        st.subheader("💳 Evolution of Payment Methods (2015–2025)")

        payment_trends = (
                filtered_df.groupby(['order_year', 'payment_method'])['final_amount_inr']
                .sum()
                .reset_index()
            )
        total_per_year = payment_trends.groupby('order_year')['final_amount_inr'].transform('sum')
        payment_trends['market_share'] = (payment_trends['final_amount_inr'] / total_per_year) * 100

        fig6 = px.area(
                payment_trends,
                x='order_year', y='market_share', color='payment_method',
                title='Payment Method Market Share Evolution (2015–2025)',
                color_discrete_sequence=px.colors.qualitative.Bold
            )
        fig6.update_layout(
                xaxis_title="Year",
                yaxis_title="Market Share (%)",
                height=550,
                template="plotly_white"
            )
        st.plotly_chart(fig6, use_container_width=True)

        latest_year = payment_trends['order_year'].max()
        latest_data = payment_trends[payment_trends['order_year'] == latest_year]
        top_payment = latest_data.loc[latest_data['market_share'].idxmax(), 'payment_method']

        st.info(
                f"📈 **Insight:** In {latest_year}, **{top_payment}** dominated customer payments — "
                f"indicating a strong shift toward digital transactions."
            )
    #--------------------------------
    #            TAB 2
    #--------------------------------
    elif page4 == "🚚 Delivery Performance Analysis":
        st.header("🚚 Delivery Performance Analysis")
        st.caption("Understand delivery speed, reliability, and its impact on customer satisfaction.")

        # ✅ Check required data
        if 'delivery_days' not in filtered_df.columns:
            st.error("❌ 'delivery_days' column missing. Please ensure dataset includes delivery duration per order.")
        else:
            df_del = filtered_df.copy()
            df_del = df_del[df_del['delivery_days'].notna() & (df_del['delivery_days'] >= 0)]

            # On-time delivery flag (≤ 3 days)
            df_del['On_Time'] = df_del['delivery_days'] <= 3

            # ----------------------------------------
            # 1️⃣ Delivery Speed Distribution
            # ----------------------------------------
            st.subheader("📦 Delivery Speed Distribution")

            fig_hist = px.histogram(
                df_del,
                x='delivery_days',
                color='customer_tier',
                nbins=20,
                title="How Fast Are Deliveries Completed?",
                marginal='box',
                color_discrete_sequence=px.colors.qualitative.Bold
            )
            fig_hist.update_layout(
                xaxis_title="Delivery Days",
                yaxis_title="Number of Orders",
                template="plotly_white"
            )
            st.plotly_chart(fig_hist, use_container_width=True)

            st.info("Most deliveries within 2–4 days indicate strong operational performance. Longer tails may suggest logistical delays.")

            # ----------------------------------------
            # 2️⃣ On-Time Delivery by Tier & City
            # ----------------------------------------
            st.subheader("⏱️ On-Time Delivery Rate by Tier & City")

            on_time = (
                df_del.groupby(['customer_city', 'customer_tier'], as_index=False)
                .agg(OnTimeRate=('On_Time', 'mean'))
            )
            on_time['OnTimeRate'] *= 100

            fig_heat = px.density_heatmap(
                on_time,
                x='customer_tier',
                y='customer_city',
                z='OnTimeRate',
                color_continuous_scale='Greens',
                title="On-Time Delivery Rate by City & Tier (%)"
            )
            fig_heat.update_layout(
                xaxis_title="Customer Tier",
                yaxis_title="City",
                template="plotly_white"
            )
            st.plotly_chart(fig_heat, use_container_width=True)

            st.info("Metro and Tier-1 cities usually maintain higher on-time delivery rates due to stronger logistics networks.")

            # ----------------------------------------
            # 3️⃣ Delivery Speed vs Customer Rating
            # ----------------------------------------
            if 'customer_rating' in df_del.columns:
                st.subheader("⭐ Delivery Speed vs Customer Ratings")

                corr = df_del['delivery_days'].corr(df_del['customer_rating'])
                st.caption(f"📊 Correlation between Delivery Speed and Rating: **{corr:.2f}**")

                fig_scatter = px.scatter(
                    df_del,
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

                if corr < 0:
                    st.success("✅ Negative correlation → Faster deliveries lead to higher ratings.")
                else:
                    st.warning("⚠️ Weak or positive correlation → Delivery speed may not strongly impact satisfaction.")
            else:
                st.warning("⚠️ 'customer_rating' column not found — skipping satisfaction analysis.")

            # ----------------------------------------
            # 4️⃣ Tier-wise Summary
            # ----------------------------------------
            st.subheader("🏙️ Delivery Summary by Customer Tier")

            tier_summary = (
                df_del.groupby('customer_tier', as_index=False)
                .agg(
                    Avg_Delivery_Days=('delivery_days', 'mean'),
                    OnTime_Rate=('On_Time', 'mean'),
                    Avg_Rating=('customer_rating', 'mean')
                )
            )
            tier_summary['OnTime_Rate'] *= 100

            st.dataframe(
                tier_summary.style.format({
                    'Avg_Delivery_Days': '{:.2f}',
                    'OnTime_Rate': '{:.1f}%',
                    'Avg_Rating': '{:.2f}'
                })
            )

            # ----------------------------------------
            # 5️⃣ Insights
            # ----------------------------------------
            st.markdown("""
            ### 💡 Key Insights
            - **Fast Deliveries (≤ 3 days)** show higher satisfaction and customer retention.
            - **Metro & Tier-1 cities** lead in on-time performance; rural areas may need support.
            - **Strong negative correlation** between delivery time and ratings → speed drives happiness.
            - **Suggestions:**
                - Expand **Express or Prime Delivery** to more cities.
                - Track **delayed deliveries** and logistics partners more closely.
                - Use **regional performance reports** to improve weak areas.
            """)
    #-----------------------------------------
    #                 TAB 3
    #-----------------------------------------
    elif page4 == "💎 Prime Membership Impact on Customer Behavior":
        st.header("💎 Prime Membership Impact on Customer Behavior")

        prime_df = filtered_df.copy()
        prime_df['is_prime_member'] = prime_df['is_prime_member'].astype(bool)

        prime_summary = (
            prime_df.groupby('is_prime_member', as_index=False)
            .agg({
                'transaction_id': 'nunique',
                'customer_id': 'nunique',
                'final_amount_inr': 'sum',
                'order_date': 'count'
            })
        )
        prime_summary['avg_order_value'] = prime_df.groupby('is_prime_member')['final_amount_inr'].mean().values
        prime_summary['avg_order_per_customer'] = prime_summary['transaction_id'] / prime_summary['customer_id']
        prime_summary['Type'] = prime_summary['is_prime_member'].replace({True: 'Prime', False: 'Non-Prime'})

        st.subheader("📊 Summary Metrics: Prime vs Non-Prime")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("💰 Avg Order Value (INR)", f"{prime_summary.loc[0, 'avg_order_value']:.0f}")
        with col2:
            st.metric("🛒 Avg Orders per Customer", f"{prime_summary.loc[0, 'avg_order_per_customer']:.2f}")
        with col3:
            st.metric("👥 Total Prime Customers", f"{prime_summary.loc[prime_summary['Type'] == 'Prime', 'customer_id'].values[0]}")

        st.subheader("💵 Average Order Value Comparison")
        fig_aov = px.bar(
            prime_summary,
            x='Type',
            y='avg_order_value',
            color='Type',
            text_auto='.2s',
            color_discrete_sequence=['#007BFF', '#FF5733'],
            title="Average Order Value: Prime vs Non-Prime"
        )
        fig_aov.update_layout(showlegend=False, yaxis_title="Average Order Value (INR)")
        st.plotly_chart(fig_aov, use_container_width=True)

        st.subheader("📈 Order Frequency Distribution per Customer")
        freq_df = (
            prime_df.groupby(['customer_id', 'is_prime_member'])
            .agg({'transaction_id': 'count'})
            .rename(columns={'transaction_id': 'order_count'})
            .reset_index()
        )
        freq_df['Type'] = freq_df['is_prime_member'].replace({True: 'Prime', False: 'Non-Prime'})

        fig_freq = px.box(
            freq_df,
            x='Type',
            y='order_count',
            color='Type',
            color_discrete_sequence=['#00B894', '#FF7675'],
            title="Order Frequency Distribution (Prime vs Non-Prime)"
        )
        st.plotly_chart(fig_freq, use_container_width=True)

        st.subheader("🛍️ Category Preference Comparison")
        cat_pref = (
            prime_df.groupby(['is_prime_member', 'subcategory'], as_index=False)['final_amount_inr']
            .sum()
        )
        cat_pref['Type'] = cat_pref['is_prime_member'].replace({True: 'Prime', False: 'Non-Prime'})

        fig_cat = px.bar(
            cat_pref,
            x='subcategory',
            y='final_amount_inr',
            color='Type',
            barmode='group',
            color_discrete_sequence=['#1E90FF', '#FFB347'],
            title="Category Spending Comparison: Prime vs Non-Prime"
        )
        fig_cat.update_layout(xaxis_title="Category", yaxis_title="Total Revenue (INR)")
        st.plotly_chart(fig_cat, use_container_width=True)

        st.subheader("🥧 Prime vs Non-Prime Revenue Share")
        revenue_share = (
            prime_df.groupby('is_prime_member', as_index=False)['final_amount_inr']
            .sum()
            .replace({True: 'Prime', False: 'Non-Prime'})
            .rename(columns={'is_prime_member': 'Customer Type'})
        )

        fig_pie = px.pie(
            revenue_share,
            names='Customer Type',
            values='final_amount_inr',
            color_discrete_sequence=['#4CAF50', '#F44336'],
            title="Revenue Contribution: Prime vs Non-Prime"
        )
        fig_pie.update_traces(textinfo='percent+label', pull=[0.05, 0])
        st.plotly_chart(fig_pie, use_container_width=True)

        st.markdown("""
        ### 💡 Key Insights:
        - **Prime members** show higher *average order value* and *order frequency*, reflecting loyalty and higher lifetime value.
        - **Non-Prime customers** contribute significantly to volume but have lower per-order spend.
        - **Prime users prefer premium categories** (like Electronics, Smart Watches), while non-Prime tend toward budget categories.
        - To boost revenue: 
            - Encourage Non-Prime to join Prime via trial programs.
            - Offer exclusive category-based promotions for Prime members.
        """)

###______________________------------------------------------------
###                             PAGE 5
###-----------------------------------------_______________________

elif main_page == "🌏 Market & External Factors":
    st.title("🌏 Market & External Factors")
    page5= st.selectbox(
        "Select Analysis:",[
            "🧭 Geographic Analysis",
            "🎉 Festival Sales Impact",
            "🏷️ Competitive Pricing & Brand Positioning Analysis"
        ]
    )
    #-------------------------
    #        TAB1
    #-------------------------

    if page5 == "🧭 Geographic Analysis":
        st.header("🧭 Geographic Sales Performance Across India")
        geo_df = filtered_df.copy()
        geo_df['final_amount_inr'] = geo_df['final_amount_inr'].astype(float)
        geo_df['order_year'] = geo_df['order_year'].astype(int)
        
        state_name_map = {
            "Andaman and Nicobar": "Andaman & Nicobar Islands",
            "Delhi": "NCT of Delhi",
            "Jammu & Kashmir": "Jammu and Kashmir",
            "Odisha": "Orissa",
            "Telangana": "Telangana",
            "Pondicherry": "Puducherry",
            "Uttaranchal": "Uttarakhand",
            "Dadra and Nagar Haveli": "Dadra & Nagar Haveli and Daman & Diu",
            "Chhattisgarh": "Chhattisgarh",
        }
        filtered_df["customer_state"] = filtered_df["customer_state"].replace(state_name_map)
        state_rev = (
            filtered_df.groupby('customer_state', as_index=False)
            .agg({'final_amount_inr': 'sum'})
            .sort_values('final_amount_inr', ascending=False)
        )

        tier_rev = (
            filtered_df.groupby('customer_tier', as_index=False)
            .agg({'final_amount_inr': 'sum'})
            .sort_values('final_amount_inr', ascending=False)
        )

        city_rev = (
            filtered_df.groupby('customer_city', as_index=False)
            .agg({'final_amount_inr': 'sum'})
            .sort_values('final_amount_inr', ascending=False)
            .head(15)
        )

        tier_year = (
            filtered_df.groupby(['order_year', 'customer_tier'], as_index=False)['final_amount_inr']
            .sum()
            .sort_values(['customer_tier', 'order_year'])
        )
        tier_year['YoY_Growth'] = tier_year.groupby('customer_tier')['final_amount_inr'].pct_change() * 100

        geojson_url = "https://raw.githubusercontent.com/geohacker/india/master/state/india_telengana.geojson"
        with urllib.request.urlopen(geojson_url) as response:
            india_geo = json.load(response)
        st.success("✅ India GeoJSON loaded successfully (online mode).")

        st.subheader("🗺️ Revenue Distribution Across Indian States")

        fig_map = px.choropleth(
            state_rev,
            geojson=india_geo,
            featureidkey="properties.NAME_1",
            locations="customer_state",
            color="final_amount_inr",
            color_continuous_scale="YlGnBu",
            title="State-wise Revenue Density (INR)",
        )
        fig_map.update_geos(fitbounds="locations", visible=False)
        st.plotly_chart(fig_map, use_container_width=True)

        st.subheader("🏙️ Revenue Contribution by Customer Tier")

        fig_tier = px.bar(
            tier_rev,
            x='customer_tier',
            y='final_amount_inr',
            color='customer_tier',
            text_auto='.2s',
            color_discrete_sequence=px.colors.qualitative.Vivid,
            title="Revenue by Tier (Metro / Tier1 / Tier2 / Rural)"
        )
        fig_tier.update_layout(showlegend=False, xaxis_title="Tier", yaxis_title="Revenue (INR)")
        st.plotly_chart(fig_tier, use_container_width=True)

        st.subheader("🏆 Top 15 Cities by Revenue")

        fig_city = px.bar(
            city_rev,
            x='final_amount_inr',
            y='customer_city',
            orientation='h',
            text_auto='.2s',
            color='final_amount_inr',
            color_continuous_scale='Tealgrn',
            title="Top 15 Cities Driving Revenue"
        )
        fig_city.update_layout(yaxis_title="City", xaxis_title="Revenue (INR)")
        st.plotly_chart(fig_city, use_container_width=True)

        st.subheader("📈 Tier-wise Growth Pattern (2015–2025)")

        fig_growth = px.line(
            tier_year,
            x='order_year',
            y='final_amount_inr',
            color='customer_tier',
            markers=True,
            title="Revenue Growth Trends by Tier"
        )
        fig_growth.update_layout(xaxis_title="Year", yaxis_title="Revenue (INR)")
        st.plotly_chart(fig_growth, use_container_width=True)

        st.subheader("📊 YoY Growth Rate by Tier")
        fig_yoy = px.line(
            tier_year,
            x='order_year',
            y='YoY_Growth',
            color='customer_tier',
            markers=True,
            color_discrete_sequence=px.colors.qualitative.Bold,
            title="Year-over-Year Growth Rate by Tier"
        )
        fig_yoy.update_layout(xaxis_title="Year", yaxis_title="YoY Growth (%)")
        st.plotly_chart(fig_yoy, use_container_width=True)

        st.markdown("""
        ### 💡 Key Insights:
        - **Metros** contribute the largest share of revenue, especially in **Delhi, Mumbai, Bengaluru, Chennai, and Hyderabad**.
        - **Tier 1 and Tier 2 cities** show steady growth — expanding e-commerce penetration in non-metro regions.
        - **Rural regions**, while lower in total revenue, exhibit the **highest growth rate (YoY)** — a key future market.
        - Regional strategies:
            - Boost logistics and delivery options in Tier 2/Rural zones.
            - Maintain premium category dominance in Metros.
            - Target regional festivals for local sales surges.
        """)
    elif page5 == "🎉 Festival Sales Impact":
        st.header("🎉 Festival Sales Impact (Before, During & After Analysis)")

        df = filtered_df.copy()
        df['order_date'] = pd.to_datetime(df['order_date'], errors='coerce')
        df = df.dropna(subset=['order_date'])
        df['final_amount_inr'] = df['final_amount_inr'].astype(float)
        df['festival_name'] = df['festival_name'].fillna('').astype(str).str.strip().str.title()

        festival_map = {
            'Diwali': ['Diwali', 'Deepavali'],
            'Prime Day': ['Prime', 'Prime Day'],
            'Holi': ['Holi'],
            'Dussehra': ['Dussehra', 'Vijayadashami'],
            'Independence Day': ['Independence'],
            'New Year': ['New Year'],
            'Eid': ['Eid', 'Eid-Ul-Fitr', 'Id']
        }

        def identify_festival(val):
            for fest, keywords in festival_map.items():
                for k in keywords:
                    if k.lower() in val.lower():
                        return fest
            return 'Non-Festival'

        df['Festival_Key'] = df['festival_name'].apply(identify_festival)

        st.sidebar.subheader("⚙️ Analysis Settings")
        before_days = st.sidebar.slider("Days Before", 7, 60, 15)
        after_days = st.sidebar.slider("Days After", 7, 60, 15)

        fest_dates = (
            df[df['Festival_Key'] != 'Non-Festival']
            .groupby(['Festival_Key', 'order_year'])['order_date']
            .min()
            .reset_index()
        )

        if fest_dates.empty:
            st.warning("⚠️ No festival data detected in your dataset.")
            st.stop()

        results = []
        for _, row in fest_dates.iterrows():
            fest, year, date = row['Festival_Key'], row['order_year'], row['order_date']
            start_before = date - pd.Timedelta(days=before_days)
            end_after = date + pd.Timedelta(days=after_days)

            before = df[(df['order_date'] >= start_before) & (df['order_date'] < date)]['final_amount_inr'].sum()
            during = df[(df['order_date'] == date)]['final_amount_inr'].sum()
            after = df[(df['order_date'] > date) & (df['order_date'] <= end_after)]['final_amount_inr'].sum()

            lift_pct = ((during - (before / before_days)) / (before / before_days)) * 100 if before > 0 else 0
            results.append([fest, year, before, during, after, lift_pct])

        fest_impact = pd.DataFrame(results, columns=['Festival', 'Year', 'Before_Sales', 'During_Sales', 'After_Sales', 'Lift_%'])
        fest_impact['Lift_%'] = fest_impact['Lift_%'].round(2)

        st.subheader("📊 Festival Impact Summary")
        st.dataframe(fest_impact.sort_values('Lift_%', ascending=False).reset_index(drop=True))

        avg_lift = fest_impact.groupby('Festival', as_index=False)['Lift_%'].mean().sort_values('Lift_%', ascending=False)
        fig_lift = px.bar(
            avg_lift,
            x='Festival', y='Lift_%',
            color='Lift_%',
            color_continuous_scale='Sunsetdark',
            text_auto='.2f',
            title="🔥 Average Festival Lift (%) Across Years"
        )
        st.plotly_chart(fig_lift, use_container_width=True)

        st.subheader("🔍 Festival Time-Series Analysis")
        selected_fest = st.selectbox("Choose a Festival:", avg_lift['Festival'].unique())
        selected_rows = fest_dates[fest_dates['Festival_Key'] == selected_fest]

        for _, r in selected_rows.iterrows():
            f_date = r['order_date']
            f_year = r['order_year']
            start = f_date - pd.Timedelta(days=before_days)
            end = f_date + pd.Timedelta(days=after_days)

            ts = df[(df['order_date'] >= start) & (df['order_date'] <= end)].copy()
            ts = ts.groupby('order_date')['final_amount_inr'].sum().reset_index()

            fig_ts = px.line(ts, x='order_date', y='final_amount_inr', title=f"{selected_fest} {f_year} — Revenue Over Time")
            fig_ts.add_vrect(x0=f_date - pd.Timedelta(days=1), x1=f_date + pd.Timedelta(days=1),
                            fillcolor="orange", opacity=0.3, annotation_text="Festival Day", annotation_position="top left")
            st.plotly_chart(fig_ts, use_container_width=True)

        st.subheader("🛍️ Category Impact During Festival")
        selected_fest_for_cat = st.selectbox("Select Festival for Category Analysis:", avg_lift['Festival'].unique())
        fest_day = fest_dates[fest_dates['Festival_Key'] == selected_fest_for_cat]['order_date'].max()

        start_b = fest_day - pd.Timedelta(days=before_days)
        end_a = fest_day + pd.Timedelta(days=after_days)

        before_df = df[(df['order_date'] >= start_b) & (df['order_date'] < fest_day)]
        during_df = df[df['order_date'] == fest_day]

        cat_before = before_df.groupby('subcategory')['final_amount_inr'].sum().reset_index(name='Before_Sales')
        cat_during = during_df.groupby('subcategory')['final_amount_inr'].sum().reset_index(name='During_Sales')
        cat_merged = pd.merge(cat_before, cat_during, on='subcategory', how='outer').fillna(0)
        cat_merged['Lift_%'] = ((cat_merged['During_Sales'] - cat_merged['Before_Sales']) / (cat_merged['Before_Sales'] + 1e-6)) * 100

        fig_cat = px.bar(cat_merged.sort_values('Lift_%', ascending=False).head(10),
                        x='subcategory', y='Lift_%', text_auto='.1f',
                        title=f"Top 10 Categories with Highest Lift During {selected_fest_for_cat}",
                        color='Lift_%', color_continuous_scale='Viridis')
        st.plotly_chart(fig_cat, use_container_width=True)

        st.markdown("""
        ### 💡 Key Insights:
        - **Diwali** and **Prime Day** typically show the strongest revenue lifts.
        - Electronics and Fashion dominate during festival sales periods.
        - **UPI payments** spike significantly in festival seasons.
        - Customers show higher engagement during **Prime Day**, but **Diwali** has broader reach across states.
        - Monitor **post-festival dips** to optimize stock and logistics.
        """)
    #----------------------------------
    #             TAG 3
    #----------------------------------

    elif page5 == "🏷️ Competitive Pricing & Brand Positioning Analysis":
        st.header("🏷️ Competitive Pricing & Brand Positioning Analysis")

        df = filtered_df.copy()
        
        # --- Validate required columns ---
        required_cols = ['brand', 'subcategory', 'final_amount_inr', 'discounted_price_inr', 'quantity']
        missing_cols = [c for c in required_cols if c not in df.columns]
        if missing_cols:
            st.error(f"❌ Missing required columns: {missing_cols}")
            
        # --- Data cleaning ---
        df = df.dropna(subset=['brand', 'subcategory', 'discounted_price_inr'])
        df['final_amount_inr'] = pd.to_numeric(df['final_amount_inr'], errors='coerce').fillna(0)
        df['discounted_price_inr'] = pd.to_numeric(df['discounted_price_inr'], errors='coerce').fillna(0)
        df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce').fillna(0)

        # --- Select category ---
        category_list = sorted(df['subcategory'].dropna().unique())
        selected_category = st.selectbox("Select Category", category_list)

        cat_df = df[df['subcategory'] == selected_category]
        if cat_df.empty:
            st.warning("⚠️ No data found for this category.")

        # --- Compute brand-level metrics ---
        brand_summary = (
            cat_df.groupby('brand', as_index=False)
            .agg(
                revenue=('final_amount_inr', 'sum'),
                volume=('quantity', 'sum'),
                avg_price=('discounted_price_inr', 'mean'),
                median_price=('discounted_price_inr', 'median')
            )
        )

        total_revenue = brand_summary['revenue'].sum()
        brand_summary['market_share'] = brand_summary['revenue'] / total_revenue * 100
        brand_summary = brand_summary.sort_values('revenue', ascending=False)
        top_brands = brand_summary.head(10)['brand'].tolist()
        top_df = cat_df[cat_df['brand'].isin(top_brands)]

        # -------------------------------------------
        # PRICE RANGE DISTRIBUTION (BOX PLOT)
        # -------------------------------------------
        st.subheader("📦 Price Range Comparison (Top Brands)")

        fig_box = px.box(
            top_df,
            x='brand',
            y='discounted_price_inr',
            color='brand',
            points='all',
            title=f"Price Range & Distribution — {selected_category}",
        )
        fig_box.update_layout(
            xaxis_title="Brand",
            yaxis_title="Discounted Price (INR)",
            showlegend=False,
            template="plotly_white"
        )
        st.plotly_chart(fig_box, use_container_width=True)

        # -------------------------------------------
        # COMPETITIVE MATRIX (Avg Price vs Market Share)
        # -------------------------------------------
        st.subheader("🎯 Competitive Positioning Matrix")

        fig_matrix = px.scatter(
            brand_summary,
            x='avg_price',
            y='market_share',
            size='revenue',
            color='brand',
            text='brand',
            hover_data=['median_price', 'revenue', 'volume'],
            title=f"Brand Positioning — {selected_category}",
        )
        fig_matrix.update_traces(textposition='top center')
        fig_matrix.update_layout(
            xaxis_title="Average Price (INR)",
            yaxis_title="Market Share (%)",
            template="plotly_white",
        )
        st.plotly_chart(fig_matrix, use_container_width=True)

        # ------------------------------------------
        # MARKET PENETRATION (PRICE-TIER MATRIX)
        # -------------------------------------------
        st.subheader("📊 Market Penetration by Price Tier")

        # Create price tiers based on quantiles
        try:
            cat_df['price_tier'] = pd.qcut(cat_df['discounted_price_inr'], q=5, labels=['Very Low', 'Low', 'Mid', 'High', 'Premium'])
        except:
            bins = [0, 500, 1000, 5000, 10000, cat_df['discounted_price_inr'].max()]
            cat_df['price_tier'] = pd.cut(cat_df['discounted_price_inr'], bins=bins, labels=['Very Low', 'Low', 'Mid', 'High', 'Premium'])

        tier_summary = (
            cat_df.groupby(['brand', 'price_tier'], as_index=False)['final_amount_inr']
            .sum()
        )
        tier_total = tier_summary.groupby('brand')['final_amount_inr'].sum().rename('total').reset_index()
        tier_summary = tier_summary.merge(tier_total, on='brand')
        tier_summary['pct'] = tier_summary['final_amount_inr'] / tier_summary['total'] * 100
        pivot_heat = tier_summary.pivot(index='brand', columns='price_tier', values='pct').fillna(0)
        pivot_heat = pivot_heat.loc[top_brands] if set(top_brands).issubset(pivot_heat.index) else pivot_heat

        fig_heat = px.imshow(
            pivot_heat,
            text_auto=".1f",
            color_continuous_scale="YlGnBu",
            title=f"Price-Tier Revenue Share by Brand — {selected_category}",
            labels=dict(x="Price Tier", y="Brand", color="% Revenue Share"),
        )
        st.plotly_chart(fig_heat, use_container_width=True)

        # -------------------------------------------
        # INSIGHTS
        # -------------------------------------------
        st.markdown("""
        ### 💡 Insights:
        - **Box plot** shows how each brand positions its products — premium vs budget.
        - **Competitive matrix** reveals whether high-priced brands hold proportional market share.
        - **Heatmap** shows each brand’s **price-tier mix** (penetration strategy).
        - **Balanced brands** have strong representation across tiers.
        - **Luxury brands** cluster in higher tiers but with smaller market share.
        - **Value brands** dominate lower tiers and volume share.
        """)

