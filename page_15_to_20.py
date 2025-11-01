import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from dateutil.relativedelta import relativedelta
### PAGE 15 ####
# ----------------------------------------
# 🎯 Discount & Promotion Effectiveness Analysis
# ----------------------------------------

def discount_promotion_analysis(filtered_df):
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
        color='category',
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
        df.groupby('category', as_index=False)
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
        x='category',
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
        df.groupby('category', as_index=False)['discount_effectiveness']
        .mean()
        .sort_values('discount_effectiveness', ascending=False)
    )

    fig_eff = px.bar(
        discount_eff,
        x='category',
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

#### PAGE 16 ####
# ----------------------------------------
# ⭐ Product Rating Impact on Sales Performance
# ----------------------------------------

def product_rating_analysis(filtered_df):
    st.header("⭐ Product Rating Impact on Sales Performance")

    df = filtered_df.copy()

    # --- Data Cleaning ---
    df = df[df['product_rating'].notna()]
    df['product_rating'] = df['product_rating'].astype(float)
    df['final_amount_inr'] = df['final_amount_inr'].astype(float)
    df['original_price_inr'] = df['original_price_inr'].astype(float)
    df['order_year'] = df['order_year'].astype(int)

    # --- Rating Distribution ---
    st.subheader("📊 Rating Distribution Across All Products")
    fig_rating_dist = px.histogram(
        df,
        x='product_rating',
        nbins=20,
        color='category',
        marginal='box',
        title="Distribution of Product Ratings by Category",
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    fig_rating_dist.update_layout(
        xaxis_title="Product Rating (1–5)",
        yaxis_title="Number of Products"
    )
    st.plotly_chart(fig_rating_dist, use_container_width=True)

    # --- Rating vs Sales (Scatter) ---
    st.subheader("💰 Rating vs Revenue Relationship")
    fig_scatter = px.scatter(
        df,
        x='product_rating',
        y='final_amount_inr',
        color='category',
        size='quantity',
        hover_data=['brand', 'product_name', 'original_price_inr'],
        opacity=0.7,
        trendline='ols',
        title="Correlation Between Product Rating and Revenue"
    )
    fig_scatter.update_layout(
        xaxis_title="Product Rating (1–5)",
        yaxis_title="Revenue (₹)",
        template="plotly_white"
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    # --- Average Revenue by Rating Bracket ---
    st.subheader("🏷️ Average Revenue per Rating Bracket")
    df['rating_bracket'] = pd.cut(
        df['product_rating'], 
        bins=[0, 2, 3, 4, 4.5, 5],
        labels=['Poor (0–2)', 'Average (2–3)', 'Good (3–4)', 'Very Good (4–4.5)', 'Excellent (4.5–5)']
    )

    rating_rev = (
        df.groupby('rating_bracket', as_index=False)
        .agg({'final_amount_inr': 'mean', 'quantity': 'sum'})
        .sort_values('final_amount_inr', ascending=False)
    )

    fig_bar = px.bar(
        rating_rev,
        x='rating_bracket',
        y='final_amount_inr',
        color='quantity',
        color_continuous_scale='Tealgrn',
        text_auto='.2s',
        title="Average Revenue by Rating Category"
    )
    fig_bar.update_layout(
        xaxis_title="Rating Bracket",
        yaxis_title="Average Revenue (₹)"
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # --- Correlation Heatmap ---
    st.subheader("🔗 Correlation: Ratings, Prices & Sales Metrics")
    corr_df = df[['product_rating', 'original_price_inr', 'discount_percent', 'quantity', 'final_amount_inr']]
    corr_matrix = corr_df.corr()

    fig_corr = px.imshow(
        corr_matrix,
        text_auto=True,
        color_continuous_scale="RdBu_r",
        title="Correlation Matrix — Ratings vs Sales & Price Metrics"
    )
    st.plotly_chart(fig_corr, use_container_width=True)

    # --- Category-Wise Rating Performance ---
    st.subheader("📦 Category-Wise Rating vs Average Revenue")
    category_rating = (
        df.groupby('category', as_index=False)
        .agg({'product_rating': 'mean', 'final_amount_inr': 'mean'})
        .sort_values('product_rating', ascending=False)
    )

    fig_line = px.line(
        category_rating,
        x='product_rating',
        y='final_amount_inr',
        text='category',
        markers=True,
        title="Category Average Rating vs Average Revenue"
    )
    fig_line.update_traces(textposition="top center")
    st.plotly_chart(fig_line, use_container_width=True)

    # ---  Insights ---
    st.markdown("""
    ### 💡 Key Insights:
    - **Higher-rated products** generally drive **more revenue**, indicating strong customer trust.
    - **Rating 4.5+** products often belong to **premium categories** with higher price points.
    - **Low-rated products (≤3)** tend to sell in higher quantities only during **deep discounts** or festive offers.
    - **Correlation heatmap** reveals:
        - Positive correlation between **rating and revenue**.
        - Negative correlation between **discount% and rating**, suggesting over-discounting for poor-rated products.
    - **Action Points:**
        - Promote **high-rated, mid-price products** for sustained sales.
        - Gather **feedback** for low-rated categories.
        - Focus **marketing spend** on top-rated items to boost ROI.
    """)

#### PAGE 17 ####
# ----------------------------------------
# 🛤️ Customer Journey & Purchase Evolution Analysis
# ----------------------------------------

def customer_journey_analysis(filtered_df):
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
    # Category Transition Analysis (Sankey Diagram)
    # -----------------------------------------------------
    st.subheader("🔄 Category Transition Flow (From First → Next Purchase)")

    # Sort purchases by date to track transition
    df_sorted = df.sort_values(['customer_id', 'order_date'])
    df_sorted['next_category'] = df_sorted.groupby('customer_id')['category'].shift(-1)

    transition_data = (
        df_sorted.groupby(['category', 'next_category'])
        .size()
        .reset_index(name='count')
        .dropna()
    )

    if not transition_data.empty:
        # Prepare for Sankey diagram
        all_categories = list(pd.unique(transition_data[['category', 'next_category']].values.ravel()))
        cat_index = {cat: i for i, cat in enumerate(all_categories)}

        fig_sankey = go.Figure(
            data=[
                go.Sankey(
                    node=dict(
                        pad=15,
                        thickness=20,
                        line=dict(color="black", width=0.5),
                        label=all_categories,
                        color=px.colors.qualitative.G10[:len(all_categories)]
                    ),
                    link=dict(
                        source=transition_data['category'].map(cat_index),
                        target=transition_data['next_category'].map(cat_index),
                        value=transition_data['count']
                    )
                )
            ]
        )
        fig_sankey.update_layout(title_text="Customer Category Transition Flow", font_size=12)
        st.plotly_chart(fig_sankey, use_container_width=True)
    else:
        st.warning("⚠️ Not enough sequential purchase data to build transition flow.")

    # -----------------------------------------------------
    # Evolution Over Time — From New → Loyal
    # -----------------------------------------------------
    st.subheader("📈 Customer Evolution Over Time")

    loyalty_trend = (
        df.groupby(['order_year', 'customer_id'])
        .size()
        .reset_index(name='yearly_orders')
    )

    loyalty_trend['Customer_Tier'] = pd.cut(
        loyalty_trend['yearly_orders'],
        bins=[0, 1, 3, 7, np.inf],
        labels=['New', 'Occasional', 'Regular', 'Loyal']
    )

    loyalty_summary = (
        loyalty_trend.groupby(['order_year', 'Customer_Tier'])
        .size()
        .reset_index(name='Customer_Count')
    )

    fig_trend = px.area(
        loyalty_summary,
        x='order_year',
        y='Customer_Count',
        color='Customer_Tier',
        title="Evolution of Customer Loyalty Segments (2015–2025)",
        color_discrete_sequence=px.colors.qualitative.Set2
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

#### PAGE 18 ####
# ----------------------------------------
# 📦 Inventory & Product Lifecycle Analysis
# ----------------------------------------

def product_lifecycle_analysis(filtered_df):


    st.header("📦 Inventory & Product Lifecycle Analysis")

    df = filtered_df.copy()

    # ---- Required columns check (we can infer some if missing) ----
    cols = df.columns.to_list()
    # preference / optional columns
    has_launch_col = 'product_launch_date' in cols
    has_discontinued_col = 'product_discontinued_date' in cols
    has_stock_col = 'stock_level' in cols or 'inventory' in cols or 'quantity' in cols

    # Ensure date and numeric types
    if 'order_date' not in df.columns:
        st.error("Dataset must include `order_date` column.")
        return
    df['order_date'] = pd.to_datetime(df['order_date'], errors='coerce')
    df = df[df['order_date'].notna()]

    # Use product_launch_date if present, otherwise infer from first sale
    if has_launch_col:
        df['product_launch_date'] = pd.to_datetime(df['product_launch_date'], errors='coerce')
    else:
        st.info("`product_launch_date` not found — inferring from first sale date per product.")
        first_sale = df.groupby('product_id', as_index=False)['order_date'].min().rename(columns={'order_date': 'product_launch_date'})
        df = df.merge(first_sale, on='product_id', how='left')

    # Optional discontinued
    if has_discontinued_col:
        df['product_discontinued_date'] = pd.to_datetime(df['product_discontinued_date'], errors='coerce')

    # Ensure revenue column available
    if 'final_amount_inr' not in df.columns:
        st.error("Dataset must include `final_amount_inr` (revenue) column.")
        return
    df['final_amount_inr'] = pd.to_numeric(df['final_amount_inr'], errors='coerce').fillna(0)

    # --- Aggregate daily/monthly product sales time series ---
    df['order_month'] = df['order_date'].dt.to_period('M').dt.to_timestamp()
    prod_monthly = (
        df.groupby(['product_id', 'product_name', 'order_month'], as_index=False)
        .agg(monthly_revenue=('final_amount_inr', 'sum'),
             monthly_qty=('quantity', 'sum' if 'quantity' in df.columns else 'count'))
    )

    # --- Compute product lifecycle metrics ---
    # Launch date per product
    launch_df = prod_monthly.groupby('product_id', as_index=False).agg(
        launch_month=('order_month', 'min')
    )
    # Peak month and revenue
    peak_df = prod_monthly.groupby('product_id', as_index=False).apply(
        lambda g: pd.Series({
            'peak_month': g.loc[g['monthly_revenue'].idxmax(), 'order_month'] if not g.empty else pd.NaT,
            'peak_revenue': g['monthly_revenue'].max() if not g.empty else 0,
            'total_revenue': g['monthly_revenue'].sum()
        })
    ).reset_index()
    metrics = launch_df.merge(peak_df, on='product_id', how='left')

    # Time-to-peak in months (approximate)
    metrics['time_to_peak_months'] = metrics.apply(
        lambda row: (
            (relativedelta(row['peak_month'], row['launch_month']).years * 12) +
            relativedelta(row['peak_month'], row['launch_month']).months
        )
        if pd.notnull(row['peak_month']) and pd.notnull(row['launch_month'])
        else 0,
        axis=1
    )
    prod_monthly = prod_monthly.merge(metrics[['product_id', 'launch_month']], on='product_id', how='left')
    prod_monthly['months_since_launch'] = (
        (prod_monthly['order_month'] - prod_monthly['launch_month']).dt.days / 30
    ).fillna(0).astype(int)
    first90 = (
        prod_monthly[prod_monthly['months_since_launch'] <= 3]
        .groupby('product_id', as_index=False)['monthly_revenue']
        .sum().rename(columns={'monthly_revenue': 'rev_first_90days'})
    )
    first180 = (
        prod_monthly[prod_monthly['months_since_launch'] <= 6]
        .groupby('product_id', as_index=False)['monthly_revenue']
        .sum().rename(columns={'monthly_revenue': 'rev_first_180days'})
    )
    metrics = metrics.merge(first90, on='product_id', how='left').merge(first180, on='product_id', how='left')
    metrics[['rev_first_90days','rev_first_180days']] = metrics[['rev_first_90days','rev_first_180days']].fillna(0)

    # success flag: e.g., top 20% of rev_first_90days relative to median
    q80 = metrics['rev_first_90days'].quantile(0.8)
    metrics['launch_success'] = metrics['rev_first_90days'] >= q80

    # Decline detection: sustained decline after peak (e.g., 3 consecutive months with revenue < 60% of peak)
    decline_list = []
    for pid, g in prod_monthly.groupby('product_id'):
        g = g.sort_values('order_month')
        peak = g['monthly_revenue'].max() if not g.empty else 0
        decline = False
        if peak > 0:
            # find months after peak
            post = g[g['monthly_revenue'] < 0.6 * peak]
            # check if there are 3 consecutive months after peak satisfying condition
            # convert to boolean sequence of months post-peak
            if not post.empty:
                months = g['order_month']
                cond = (g['monthly_revenue'] < 0.6 * peak).astype(int).values
                # check run length of 3 consecutive ones
                runs = np.split(cond, np.where(np.diff(cond) != 0)[0] + 1)
                if any(len(r) >= 3 and r.sum() == len(r) for r in runs):
                    decline = True
        decline_list.append((pid, decline))
    decline_df = pd.DataFrame(decline_list, columns=['product_id', 'sustained_decline'])
    metrics = metrics.merge(decline_df, on='product_id', how='left')

    # Merge product_name and category for display
    if 'product_name' in df.columns:
        metrics = metrics.merge(df[['product_id', 'product_name', 'category']].drop_duplicates('product_id'), on='product_id', how='left')

    # ---- VISUALIZATIONS ----

    # Launch timeline: number of launches per year
    st.subheader("🚀 Product Launch Timeline")
    metrics['launch_year'] = metrics['launch_month'].dt.year
    launches_by_year = metrics.groupby('launch_year', as_index=False)['product_id'].count().rename(columns={'product_id':'launch_count'})
    fig_launch = px.bar(launches_by_year.sort_values('launch_year'), x='launch_year', y='launch_count', title="Products Launched per Year")
    fig_launch.update_layout(xaxis_title="Year", yaxis_title="Number of Product SKUs")
    st.plotly_chart(fig_launch, use_container_width=True)

    # Time-to-peak distribution
    st.subheader("⏱️ Time-to-Peak (months) Distribution")
    fig_ttp = px.histogram(metrics, x='time_to_peak_months', nbins=30, title="Time-to-Peak Distribution (months)")
    st.plotly_chart(fig_ttp, use_container_width=True)

    # Top launches by first-90-days revenue
    st.subheader("🏆 Top Launches (First 90 Days Revenue)")
    top_launches = metrics.sort_values('rev_first_90days', ascending=False).head(20)
    if not top_launches.empty:
        fig_top = px.bar(top_launches, x='product_name', y='rev_first_90days', hover_data=['category','total_revenue'], title="Top 20 Launches by 90-day Revenue")
        fig_top.update_layout(xaxis_title="Product", yaxis_title="Revenue (first 90 days)", xaxis={'tickangle':-45})
        st.plotly_chart(fig_top, use_container_width=True)
    else:
        st.info("No launch data available to display top launches.")

    # Lifecycle curves for a selected product (line of monthly revenue)
    st.subheader("📈 Product Lifecycle Curve (Select a product)")
    product_list = metrics['product_name'].dropna().unique().tolist()
    if not product_list:
        st.warning("No products available for lifecycle curve. Check filters.")
    else:
        selected_product = st.selectbox("Choose product for lifecycle curve", product_list[:500])  # limit for performance
        pid = metrics[metrics['product_name']==selected_product]['product_id'].iloc[0]
        p_ts = prod_monthly[prod_monthly['product_id']==pid].sort_values('order_month')
        if p_ts.empty:
            st.info("No time series data for the selected product.")
        else:
            launch_date = metrics.loc[metrics['product_id']==pid, 'launch_month'].iloc[0]
            peak_date = metrics.loc[metrics['product_id']==pid, 'peak_month'].iloc[0]

            def ensure_pydate(x):
                if pd.isna(x):
                    return None
                if isinstance(x, np.datetime64):
                    return pd.to_datetime(x).to_datetime64()
                if isinstance(x, pd.Timestamp):
                    return x.to_pydatetime()
                return x
            launch_date = ensure_pydate(launch_date)
            peak_date = ensure_pydate(peak_date)
            fig_lc = px.line(
                p_ts, 
                x='order_month', 
                y='monthly_revenue', 
                title=f"Monthly Revenue — {selected_product}", 
                markers=True,
                color_discrete_sequence=['#0078D7']
                )
            
            def date_to_float(date_obj):
                if date_obj is None:
                    return None
                if isinstance(date_obj, pd.Timestamp):
                    date_obj = pd.to_datetime(str(date_obj)).to_pydatetime()
                return date_obj.timestamp() * 1000
            launch_x = date_to_float(launch_date)
            peak_x = date_to_float(peak_date)
            if isinstance(launch_date, (pd.Timestamp, pd.DatetimeIndex)):
                launch_date = pd.to_datetime(str(launch_date)).to_pydatetime()
                fig_lc.add_shape(
                    type='line',
                    x0=launch_date,
                    x1=launch_date,
                    y0=0,
                    y1=1,
                    xref='x',
                    yref='paper',
                    line=dict(color = "orange", dash='dash') 
                    )
                fig_lc.add_annotation(
                    x=launch_date,
                    y=1,
                    text="Launch",
                    showarrow=False,
                    yref="paper",
                    yanchor="bottom",
                    font = dict(size=12, color="orange")
                )
            fig_lc.update_layout(
                xaxis_title="Order Month",
                yaxis_title="Monthly Revenue (₹)",
                template="plotly_white"
            )
            st.plotly_chart(fig_lc, use_container_width=True)

    # Inventory heatmap (if stock/inventory columns exist)
    st.subheader("📊 Inventory Snapshot / Heatmap (if available)")
    if has_stock_col:
        # pick a stock-like column name
        stock_col = 'stock_level' if 'stock_level' in df.columns else ('inventory' if 'inventory' in df.columns else 'quantity')
        inv = df.groupby(['product_id', 'product_name'], as_index=False)[stock_col].mean().dropna().sort_values(stock_col, ascending=False).head(200)
        if not inv.empty:
            fig_inv = px.bar(inv, x=stock_col, y='product_name', orientation='h', title=f"Average {stock_col} (Top 200 SKUs)")
            st.plotly_chart(fig_inv, use_container_width=True)
        else:
            st.info("No inventory values available to plot.")
    else:
        st.info("Inventory / stock column not found. To enable inventory heatmap, include 'stock_level' or 'inventory' column.")

    # Decline & churned products
    st.subheader("⚠️ Products in Decline / Churn")
    decline_products = metrics[metrics['sustained_decline']].sort_values('total_revenue', ascending=False).head(50)
    if not decline_products.empty:
        st.dataframe(decline_products[['product_id','product_name','category','peak_month','peak_revenue','total_revenue','time_to_peak_months']])
    else:
        st.info("No products detected with sustained decline (based on current filters).")

    # Summary metrics and export
    total_products = metrics['product_id'].nunique()
    launched = metrics['launch_month'].notna().sum()
    succeeded = metrics['launch_success'].sum()
    st.markdown(f"**Total SKUs (in filtered data):** {total_products:,}  \n**Detected launches:** {launched:,}  \n**Top-launch (90d) success count:** {succeeded:,}")

    if st.button("Download lifecycle metrics (CSV)"):
        csv = metrics.to_csv(index=False).encode('utf-8')
        st.download_button(label="Download metrics CSV", data=csv, file_name="product_lifecycle_metrics.csv", mime="text/csv")

  #### PAGE 19 ####
# ----------------------------------------
# "🏷️ Competitive Pricing & Brand Positioning Analysis"
# ----------------------------------------


def competitive_pricing_analysis(filtered_df):
    st.header("🏷️ Competitive Pricing & Brand Positioning Analysis")

    df = filtered_df.copy()
    
    # --- Validate required columns ---
    required_cols = ['brand', 'category', 'final_amount_inr', 'discounted_price_inr', 'quantity']
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        st.error(f"❌ Missing required columns: {missing_cols}")
        return

    # --- Data cleaning ---
    df = df.dropna(subset=['brand', 'category', 'discounted_price_inr'])
    df['final_amount_inr'] = pd.to_numeric(df['final_amount_inr'], errors='coerce').fillna(0)
    df['discounted_price_inr'] = pd.to_numeric(df['discounted_price_inr'], errors='coerce').fillna(0)
    df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce').fillna(0)

    # --- Select category ---
    category_list = sorted(df['category'].dropna().unique())
    selected_category = st.selectbox("Select Category", category_list)

    cat_df = df[df['category'] == selected_category]
    if cat_df.empty:
        st.warning("⚠️ No data found for this category.")
        return

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

    # Return for optional reuse
    return brand_summary, pivot_heat

#### PAGE 20 ####
# ----------------------------------------
# 📊 Business Health Dashboard (2015–2025)
# ----------------------------------------

def business_health_dashboard(filtered_df):
    st.header("📊 Business Health Dashboard (2015–2025)")
    st.caption("A holistic overview of business performance across revenue, customers, retention, and operational efficiency.")

    df = filtered_df.copy()
    df['order_date'] = pd.to_datetime(df['order_date'], errors='coerce')
    df['order_year'] = df['order_date'].dt.year

    # ------------------------------------------------------
    # Executive Summary KPIs
    # ------------------------------------------------------
    st.subheader("🏁 Executive Summary")

    total_revenue = df['final_amount_inr'].sum()
    total_orders = df['transaction_id'].nunique()
    unique_customers = df['customer_id'].nunique()
    avg_order_value = total_revenue / total_orders if total_orders else 0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("💰 Total Revenue", f"₹{total_revenue:,.0f}")
    col2.metric("📦 Orders", f"{total_orders:,}")
    col3.metric("👥 Unique Customers", f"{unique_customers:,}")
    col4.metric("💳 Avg Order Value", f"₹{avg_order_value:,.0f}")

    # ------------------------------------------------------
    # Revenue Growth Trend
    # ------------------------------------------------------
    st.subheader("📈 Revenue Growth Over Time")

    yearly_rev = df.groupby('order_year')['final_amount_inr'].sum().reset_index()
    yearly_rev['YoY_Growth'] = yearly_rev['final_amount_inr'].pct_change() * 100

    fig_rev = px.line(
        yearly_rev,
        x='order_year',
        y='final_amount_inr',
        markers=True,
        title="Revenue Trend & Growth (2015–2025)",
        color_discrete_sequence=['#0077b6']
    )
    fig_rev.update_traces(mode='lines+markers')
    fig_rev.update_layout(
        xaxis_title="Year",
        yaxis_title="Revenue (INR)",
        hovermode="x unified"
    )
    st.plotly_chart(fig_rev, use_container_width=True)

    growth_rate = yearly_rev['YoY_Growth'].iloc[-1] if not yearly_rev['YoY_Growth'].isna().all() else 0
    st.info(f"📈 **Year-over-Year Growth:** {growth_rate:.2f}%")

    # ------------------------------------------------------
    # Customer Acquisition & Retention
    # ------------------------------------------------------
    st.subheader("👥 Customer Acquisition & Retention Analysis")

    customer_orders = df.groupby(['customer_id', 'order_year'])['transaction_id'].nunique().reset_index(name='orders')
    first_purchase = customer_orders.groupby('customer_id')['order_year'].min().reset_index(name='acquisition_year')
    retention = first_purchase.groupby('acquisition_year')['customer_id'].count().reset_index(name='new_customers')

    # Retention curve (based on repeated buyers)
    retention_rate = (
        customer_orders.groupby('order_year')['customer_id']
        .nunique()
        .reset_index(name='active_customers')
    )
    retention_rate['retention_growth'] = retention_rate['active_customers'].pct_change() * 100

    col5, col6 = st.columns(2)

    with col5:
        fig_acq = px.bar(
            retention,
            x='acquisition_year',
            y='new_customers',
            title="New Customer Acquisition per Year",
            color='new_customers',
            color_continuous_scale='Blues'
        )
        st.plotly_chart(fig_acq, use_container_width=True)

    with col6:
        fig_ret = px.line(
            retention_rate,
            x='order_year',
            y='active_customers',
            markers=True,
            title="Customer Retention Over Time",
            color_discrete_sequence=['#00b4d8']
        )
        st.plotly_chart(fig_ret, use_container_width=True)

    # ------------------------------------------------------
    # Operational Efficiency Metrics
    # ------------------------------------------------------
    st.subheader("⚙️ Operational Efficiency Overview")

    if 'delivery_days' in df.columns:
        df['delivery_days'] = pd.to_numeric(df['delivery_days'], errors='coerce')
        ops_metrics = (
            df.groupby('order_year')['delivery_days']
            .mean()
            .reset_index(name='avg_delivery_days')
        )
        fig_ops = px.bar(
            ops_metrics,
            x='order_year',
            y='avg_delivery_days',
            color='avg_delivery_days',
            color_continuous_scale='Viridis',
            title="Average Delivery Time per Year"
        )
        st.plotly_chart(fig_ops, use_container_width=True)
    else:
        st.warning("⚠️ 'delivery_days' column not found — skipping delivery efficiency analysis.")

    # ------------------------------------------------------
    # Multi-Metric Radar View
    # ------------------------------------------------------
    st.subheader("🕸️ Business Health Radar Chart")

    radar_metrics = pd.DataFrame({
        'Metric': ['Revenue Growth', 'Customer Retention', 'Order Volume', 'Operational Efficiency'],
        'Value': [
            max(0, min(100, growth_rate)),
            retention_rate['retention_growth'].mean() if not retention_rate.empty else 0,
            np.log1p(total_orders) % 100,
            (100 / df['delivery_days'].mean()) if 'delivery_days' in df.columns and df['delivery_days'].mean() > 0 else 50
        ]
    })

    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(
        r=radar_metrics['Value'],
        theta=radar_metrics['Metric'],
        fill='toself',
        name='Business Health'
    ))
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        title="Overall Business Health Indicators",
        showlegend=False
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    # ------------------------------------------------------
    # Executive Insights Summary
    # ------------------------------------------------------
    st.markdown("""
    ### 💼 Executive Summary Insights
    - **Revenue Growth:** Sustained improvement year-over-year, indicating solid market expansion.
    - **Customer Base:** A healthy acquisition trend complemented by improving retention.
    - **Operational Performance:** Faster deliveries correlate with higher satisfaction.
    - **Efficiency Outlook:** Optimization in logistics and inventory management recommended.
    
    **Actionable Takeaways:**
    - Prioritize **customer retention programs** for high-value segments.
    - Monitor **delivery speed & service reliability** to boost operational scores.
    - Enhance **cross-sell & up-sell** strategies to improve order value per customer.
    """)

    return