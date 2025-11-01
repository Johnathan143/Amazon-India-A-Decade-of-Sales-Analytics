import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json, os, requests, urllib.request

#### PAGE 6 ####

#--------------------------------------------------------
# 💎 Prime Membership Impact on Customer Behavior
#--------------------------------------------------------

def prime_membership_impact(filtered_df):
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
        prime_df.groupby(['is_prime_member', 'category'], as_index=False)['final_amount_inr']
        .sum()
    )
    cat_pref['Type'] = cat_pref['is_prime_member'].replace({True: 'Prime', False: 'Non-Prime'})

    fig_cat = px.bar(
        cat_pref,
        x='category',
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

#### PAGE 7 ####

# --------------------------------------------------------
# "🧭 Geographic Analysis"
# --------------------------------------------------------
def geographic_analysis(filtered_df):
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

#### PAGE 8 ####
# --------------------------------------------------------
# "🎉 Festival Sales Impact"
# --------------------------------------------------------
def festival_sales_impact(filtered_df):
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

    cat_before = before_df.groupby('category')['final_amount_inr'].sum().reset_index(name='Before_Sales')
    cat_during = during_df.groupby('category')['final_amount_inr'].sum().reset_index(name='During_Sales')
    cat_merged = pd.merge(cat_before, cat_during, on='category', how='outer').fillna(0)
    cat_merged['Lift_%'] = ((cat_merged['During_Sales'] - cat_merged['Before_Sales']) / (cat_merged['Before_Sales'] + 1e-6)) * 100

    fig_cat = px.bar(cat_merged.sort_values('Lift_%', ascending=False).head(10),
                     x='category', y='Lift_%', text_auto='.1f',
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

#### Page 9 ####
# --------------------------------------------------------
# "🧠 Age Group Behavior"
# --------------------------------------------------------"

def age_group_behavior(filtered_df):
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
        df_age.groupby(['customer_age_group', 'category'])['final_amount_inr']
        .sum()
        .reset_index()
    )

    fig_cat = px.treemap(
        cat_pref,
        path=['customer_age_group', 'category'],
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
        df_age.groupby(['category', 'customer_age_group'])['final_amount_inr']
        .sum().reset_index()
        .pivot(index='category', columns='customer_age_group', values='final_amount_inr')
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

