import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from mlxtend.frequent_patterns import fpgrowth, association_rules
from mlxtend.preprocessing import TransactionEncoder

# 🌐 Streamlit Configuration
st.set_page_config(
    page_title="Customer Intelligence",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 🌟 Custom Elegant Theme Styling
# 🌟 Ultra-Modern Stylish Theme
st.markdown("""
    <style>
        /* BASE FONT & PAGE BG */
        html, body, [class*="css"] {
            font-family: 'Segoe UI', sans-serif;
            background: linear-gradient(to right, #f8fafc, #e2e8f0);
            color: #1f2937;
        }

        /* SIDEBAR ENHANCEMENTS */
        section[data-testid="stSidebar"] {
            background: linear-gradient(to bottom, #0f172a, #1e293b);
            color: white;
            padding-top: 1rem;
        }
        section[data-testid="stSidebar"] h1,
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3 {
            color: #ffffff;
        }

        /* METRIC CARDS with gradient + hover */
        div[data-testid="metric-container"] {
            background: linear-gradient(145deg, #ffffff, #f1f5f9);
            padding: 16px 12px;
            border-radius: 14px;
            box-shadow: 0 4px 14px rgba(0, 0, 0, 0.08);
            transition: all 0.3s ease;
        }
        div[data-testid="metric-container"]:hover {
            transform: translateY(-3px);
            box-shadow: 0 10px 20px rgba(0,0,0,0.12);
        }

        /* HEADINGS */
        h1 {
            font-size: 42px !important;
            font-weight: 800;
            color: #0f172a;
        }
        h2 {
            font-size: 30px !important;
            font-weight: 700;
            color: #1e293b;
        }
        h3 {
            font-size: 24px !important;
            font-weight: 600;
            color: #334155;
        }

        /* PARAGRAPH TEXT */
        .stMarkdown p {
            font-size: 17px;
            color: #475569;
        }

        /* MODERN BUTTONS */
        .stButton>button {
            background: linear-gradient(to right, #3b82f6, #6366f1);
            border: none;
            color: white;
            padding: 0.5em 1.5em;
            border-radius: 8px;
            font-size: 16px;
            font-weight: 600;
            transition: 0.3s ease-in-out;
        }
        .stButton>button:hover {
            background: linear-gradient(to right, #2563eb, #7c3aed);
            transform: scale(1.03);
        }

        /* SCROLLBAR - neon styled */
        ::-webkit-scrollbar {
            width: 8px;
        }
        ::-webkit-scrollbar-thumb {
            background: linear-gradient(#3b82f6, #6366f1);
            border-radius: 12px;
        }

        /* SMOOTH FADE IN ANIMATION */
        .element-container {
            animation: fadeIn 0.5s ease-in-out;
        }
        @keyframes fadeIn {
            from {opacity: 0; transform: translateY(10px);}
            to {opacity: 1; transform: translateY(0);}
        }
    </style>
""", unsafe_allow_html=True)


# ⏳ Data Load Function
@st.cache_data
def load_data():
    df = pd.read_excel("Online Retail.xlsx")
    df.drop_duplicates(inplace=True)
    df.dropna(subset=["CustomerID"], inplace=True)
    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]
    df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["InvoiceMonth"] = df["InvoiceDate"].dt.to_period("M").astype(str)
    return df

# 📦 Load Data
with st.spinner("Loading Data..."):
    df = load_data()

# 🧭 Sidebar Navigation
st.sidebar.title("📊 Customer Dashboard")
section = st.sidebar.radio("Navigate to", [
    "🏠 Home", "Overview", "EDA", "RFM Clustering", 
    "Product Segmentation", "Market Basket", "CLTV & Churn"
])


# 🏠 Home Screen
if section == "🏠 Home":
    # 🎯 Beautiful Project Intro
    st.markdown("""
        <div style='text-align:center;'>
            <h1 style='font-size: 40px;'>🛍️ 🚀 Smarter Decisions Begin with Smarter Customer Analytics</h1>
            <p style='font-size: 20px; color: #555;'>Discover key insights on customer behavior, segmentation, product performance, and revenue trends.</p>
        </div>
    """, unsafe_allow_html=True)

    # 🔄 Lottie Loader Function
    from streamlit_lottie import st_lottie
    import requests

    def load_lottieurl(url):
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()

    # 🎞️ First Main Animation
    lottie_url = "https://assets2.lottiefiles.com/packages/lf20_jcikwtux.json"
    lottie_json = load_lottieurl(lottie_url)
    if lottie_json:
        st_lottie(lottie_json, height=300, key="welcome_anim")
    else:
        st.image("https://cdn-icons-png.flaticon.com/512/1170/1170678.png", width=180)

    # 🎞️ Second Highlight Animation
    hero_anim = load_lottieurl("https://assets2.lottiefiles.com/private_files/lf30_nsqfz4bn.json")
    if hero_anim:
        st_lottie(hero_anim, height=300, key="hero")
    else:
        st.image("https://cdn-icons-png.flaticon.com/512/2331/2331970.png", width=200)

    st.markdown("---")

    # 💡 Project Overview
    st.markdown("## 💡 What This Dashboard Offers")
    st.markdown("""
    - 🎯 **Customer Segmentation** using RFM & clustering  
    - 📊 **Sales Trends** & Product Patterns  
    - 🛍️ **Market Basket Analysis** (Association Rules)  
    - 💰 **Customer Lifetime Value (CLTV)** & Churn Detection  
    - 📦 **Product Segmentation** based on revenue and quantity  
    """)

    # 🔢 Metrics Summary
    st.markdown("### 🔢 Quick Dataset Summary")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🧾 Invoices", df["InvoiceNo"].nunique())
    col2.metric("👥 Customers", df["CustomerID"].nunique())
    col3.metric("💰 Revenue", f"£{df['TotalPrice'].sum():,.0f}")
    col4.metric("📦 Products", df["Description"].nunique())

    # 🖼️ Key Analysis Cards
    st.markdown("### 📸 Explore Key Analysis Sections")
    colA, colB, colC = st.columns(3)
    with colA:
        st.image("https://cdn-icons-png.flaticon.com/512/2332/2332677.png", width=100)
        st.markdown("**RFM Segmentation**")
    with colB:
        st.image("https://cdn-icons-png.flaticon.com/512/3534/3534066.png", width=100)
        st.markdown("**Market Basket Rules**")
    with colC:
        st.image("https://cdn-icons-png.flaticon.com/512/2721/2721615.png", width=100)
        st.markdown("**CLTV & Churn**")

    st.markdown("---")

    # 👉 Footer
    st.markdown("<p style='text-align: center; font-size: 16px;'>🔎 Navigate through the sidebar to explore deep insights into your customers →</p>", unsafe_allow_html=True)


# OVERVIEW 
if section == "Overview":
    st.markdown("## 📊 Business Overview")

    # 🔢 Metric Cards
    col1, col2, col3, col4 = st.columns(4)
    total_invoices = df["InvoiceNo"].nunique()
    unique_customers = df["CustomerID"].nunique()
    total_revenue = df["TotalPrice"].sum()
    total_products = df["Description"].nunique()

    with col1:
        st.markdown(
            f"<div style='background:linear-gradient(to right, #00b4db, #0083b0);"
            f"padding:20px;border-radius:12px;color:white;text-align:center;'>"
            f"<h3>{total_invoices}</h3><p>Total Invoices</p></div>",
            unsafe_allow_html=True
        )
    with col2:
        st.markdown(
            f"<div style='background:linear-gradient(to right, #f7971e, #ffd200);"
            f"padding:20px;border-radius:12px;color:white;text-align:center;'>"
            f"<h3>{unique_customers}</h3><p>Unique Customers</p></div>",
            unsafe_allow_html=True
        )
    with col3:
        st.markdown(
            f"<div style='background:linear-gradient(to right, #43cea2, #185a9d);"
            f"padding:20px;border-radius:12px;color:white;text-align:center;'>"
            f"<h3>£{total_revenue:,.0f}</h3><p>Total Revenue</p></div>",
            unsafe_allow_html=True
        )
    with col4:
        st.markdown(
            f"<div style='background:linear-gradient(to right, #ff5f6d, #ffc371);"
            f"padding:20px;border-radius:12px;color:white;text-align:center;'>"
            f"<h3>{total_products}</h3><p>Unique Products</p></div>",
            unsafe_allow_html=True
        )

    st.markdown("### 🌍 Top 10 Countries by Revenue")
    top_countries = df.groupby("Country")["TotalPrice"].sum().sort_values(ascending=False).head(10)
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    sns.barplot(x=top_countries.values, y=top_countries.index, palette="crest", ax=ax1)
    ax1.set_xlabel("Revenue (£)")
    ax1.set_ylabel("Country")
    ax1.set_title("Top Countries by Revenue", fontsize=14)
    st.pyplot(fig1)

    st.subheader("Monthly Revenue Trend")
    monthly = df.groupby("InvoiceMonth")["TotalPrice"].sum()
    st.line_chart(monthly)

    # -------------------------------
    # ✨ Executive Summary Visuals
    # -------------------------------
    st.markdown("---")
    st.markdown("## ✨ Executive Summary Visuals")

    # 🌍 Choropleth Full-Width
    st.markdown("#### 🌍 Revenue by Country")
    country_sales = df.groupby("Country")["TotalPrice"].sum().reset_index()
    fig_choropleth = px.choropleth(
        country_sales,
        locations="Country",
        locationmode="country names",
        color="TotalPrice",
        color_continuous_scale="Sunsetdark",
        title="🌐 Revenue Distribution by Country"
    )
    fig_choropleth.update_layout(margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig_choropleth, use_container_width=True)

    # 📦 + 🕐 Side-by-Side Charts
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 📦 Monthly Order Volume")
        monthly_orders = df.copy()
        monthly_orders["Month"] = df["InvoiceDate"].dt.to_period("M").astype(str)
        monthly_orders = monthly_orders.groupby("Month")["InvoiceNo"].nunique().reset_index()
        monthly_orders.columns = ["Month", "TotalOrders"]

        fig_area = px.area(
            monthly_orders,
            x="Month",
            y="TotalOrders",
            title="📦 Orders Per Month",
            markers=True,
            color_discrete_sequence=["#4F46E5"]
        )
        fig_area.update_layout(
            margin=dict(l=10, r=10, t=40, b=10),
            xaxis_title="Month",
            yaxis_title="Total Orders",
            plot_bgcolor="white"
        )
        st.plotly_chart(fig_area, use_container_width=True)

    with col2:
        st.markdown("#### 🕐 Peak Shopping Hours")
        df["Hour"] = df["InvoiceDate"].dt.hour
        hour_sales = df.groupby("Hour")["TotalPrice"].sum().reset_index()
        fig_hour = px.bar(
            hour_sales,
            x="Hour",
            y="TotalPrice",
            text_auto=True,
            color="TotalPrice",
            color_continuous_scale="Blues",
            title="💸 Revenue by Hour of Day"
        )
        fig_hour.update_layout(
            xaxis_title="Hour of Day",
            yaxis_title="Revenue (£)",
            plot_bgcolor="white",
            margin=dict(l=10, r=10, t=40, b=10)
        )
        st.plotly_chart(fig_hour, use_container_width=True)



# ----------------------------
# 2. EDA
# ----------------------------
if section == "EDA":
    st.markdown("## 📊 Exploratory Data Analysis – Customer & Sales Patterns")

    # Preprocessing
    df["DayOfWeek"] = df["InvoiceDate"].dt.day_name()
    df["Hour"] = df["InvoiceDate"].dt.hour
    df["Month"] = df["InvoiceDate"].dt.to_period("M").astype(str)

    # ---------- Summary KPIs ----------
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🧾 Total Orders", f"{df['InvoiceNo'].nunique():,}")
    col2.metric("👥 Unique Customers", f"{df['CustomerID'].nunique():,}")
    col3.metric("📦 Unique Products", f"{df['Description'].nunique():,}")
    col4.metric("💰 Total Revenue", f"£{df['TotalPrice'].sum():,.0f}")

    st.markdown("---")

    # ---------- Pie Chart: Top Products ----------
    st.markdown("### 🥧 Top 10 Products by Quantity Sold")
    top_products = df["Description"].value_counts().head(10)
    fig1, ax1 = plt.subplots()
    ax1.pie(top_products.values, labels=top_products.index, autopct='%1.1f%%', startangle=90)
    ax1.axis("equal")
    st.pyplot(fig1)

    st.markdown("---")

    # ---------- Bar Chart: Revenue by Product ----------
    st.markdown("### 💸 Top Products by Revenue")
    top_rev = df.groupby("Description")["TotalPrice"].sum().sort_values(ascending=False).head(10)
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    sns.barplot(x=top_rev.values, y=top_rev.index, palette="rocket", ax=ax2)
    ax2.set_xlabel("Revenue (£)")
    ax2.set_ylabel("Product")
    ax2.set_title("Top Revenue Generating Products")
    st.pyplot(fig2)

    st.markdown("---")

    # ---------- Heatmap: Order Count by Day & Hour ----------
    st.markdown("### ⏱️ Orders by Day and Hour (Heatmap)")
    heatmap_data = df.groupby(["DayOfWeek", "Hour"])["InvoiceNo"].nunique().unstack().fillna(0)
    days_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    heatmap_data = heatmap_data.reindex(days_order)
    fig3, ax3 = plt.subplots(figsize=(12, 5))
    sns.heatmap(heatmap_data, cmap="YlOrBr", linewidths=0.3, annot=True, fmt=".0f", ax=ax3)
    ax3.set_title("Order Volume by Day & Hour")
    st.pyplot(fig3)

    st.markdown("---")


    # ---------- TREEMAP: Revenue by Product ----------
    st.markdown("### 🌳 Revenue Contribution by Product (Treemap)")

    import plotly.express as px
    top_treemap = df.groupby("Description")["TotalPrice"].sum().sort_values(ascending=False).head(20).reset_index()
    fig7 = px.treemap(
        top_treemap,
        path=["Description"],
        values="TotalPrice",
        color="TotalPrice",
        color_continuous_scale="Blues",
        title="Top 20 Products by Revenue"
    )
    st.plotly_chart(fig7, use_container_width=True)

    st.markdown("---")

    # ---------- STACKED BAR: Orders by Country & Weekday ----------
    st.markdown("### 🌍 Weekly Order Distribution by Country")

    top_countries = df["Country"].value_counts().head(5).index
    stacked_df = df[df["Country"].isin(top_countries)].groupby(["Country", "DayOfWeek"])["InvoiceNo"].nunique().reset_index()
    fig8 = px.bar(
        stacked_df,
        x="DayOfWeek",
        y="InvoiceNo",
        color="Country",
        barmode="stack",
        category_orders={"DayOfWeek": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]},
        title="Orders per Weekday (Top 5 Countries)"
    )
    st.plotly_chart(fig8, use_container_width=True)

    st.markdown("---")

   # ---------- INTERACTIVE BUBBLE CHART ----------
    st.markdown("### 💬 Revenue vs Quantity (Hover to See Product Names)")

    # Group data
    bubble_df = df.groupby("Description").agg({
        "Quantity": "sum",
        "TotalPrice": "sum",
        "UnitPrice": "mean"
    }).reset_index()

    # Filter to remove extreme values
    bubble_df = bubble_df[(bubble_df["Quantity"] < 5000) & (bubble_df["TotalPrice"] > 100)]
    top_bubbles = bubble_df.sort_values("TotalPrice", ascending=False).head(100)

    # Product selector (optional)
    selected_products = st.multiselect(
        "🔍 Select products to highlight (optional):",
        options=top_bubbles["Description"].tolist(),
        default=[]
    )

    # Filter based on selection
    if selected_products:
        top_bubbles = top_bubbles[top_bubbles["Description"].isin(selected_products)]

    # Only draw if not empty
    if not top_bubbles.empty:
        fig9 = px.scatter(
            top_bubbles,
            x="Quantity",
            y="TotalPrice",
            size="UnitPrice",
            color="UnitPrice",
            hover_name="Description",
            size_max=40,
            color_continuous_scale="Teal",
            title="Product Revenue vs Quantity (Bubble = Avg Price)"
        )
        fig9.update_traces(
            marker=dict(opacity=0.7, line=dict(width=1, color='DarkSlateGrey'))
        )
        st.plotly_chart(fig9, use_container_width=True)
    else:
        st.info("No products match your selection.")


# ----------------------------
# 3. RFM Clustering
# ----------------------------
if section == "RFM Clustering":
    st.markdown("## 🧮 RFM-Based Customer Segmentation")

    # Calculate RFM metrics
    reference_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)
    rfm = df.groupby("CustomerID").agg({
        "InvoiceDate": lambda x: (reference_date - x.max()).days,
        "InvoiceNo": "nunique",
        "TotalPrice": "sum"
    }).reset_index()
    rfm.columns = ["CustomerID", "Recency", "Frequency", "Monetary"]

    # Standardize RFM values
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm[["Recency", "Frequency", "Monetary"]])

    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=4, random_state=42)
    rfm["Cluster"] = kmeans.fit_predict(rfm_scaled)

    # Cluster Summary
    st.markdown("### 📦 RFM Cluster Summary")
    cluster_summary = rfm.groupby("Cluster")[["Recency", "Frequency", "Monetary"]].mean().round(1)

    col1, col2, col3, col4 = st.columns(4)
    for i, col in enumerate([col1, col2, col3, col4]):
        if i in cluster_summary.index:
            rec, freq, mon = cluster_summary.loc[i]
            col.markdown(
                f"<div style='background:linear-gradient(to right, #6a11cb, #2575fc);"
                f"padding:20px;border-radius:12px;color:white;text-align:center;'>"
                f"<h4>Cluster {i}</h4>"
                f"<p>Recency: {rec:.0f} days</p>"
                f"<p>Frequency: {freq:.1f}</p>"
                f"<p>Monetary: £{mon:,.0f}</p>"
                f"</div>", unsafe_allow_html=True
            )

    # Preview Table
    with st.expander("📋 View RFM Table"):
        st.dataframe(rfm.head(10))

    # PCA for 2D visualization
    pca = PCA(n_components=2)
    rfm_pca = pca.fit_transform(rfm_scaled)
    rfm["PCA1"], rfm["PCA2"] = rfm_pca[:, 0], rfm_pca[:, 1]

    st.markdown("### 🔵 PCA Scatterplot of Clusters")

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(
        data=rfm, x="PCA1", y="PCA2", hue="Cluster",
        palette="Set2", s=100, edgecolor="black"
    )
    ax.set_title("Customer Segments (PCA Reduced)", fontsize=14)
    ax.set_xlabel("PCA Component 1")
    ax.set_ylabel("PCA Component 2")
    st.pyplot(fig)


# ----------------------------
# 4. Product Segmentation
# ----------------------------
if section == "Product Segmentation":
    st.markdown("## 🧾 Product Segmentation by Behavior")

    # Step 1: Aggregate product features
    product_df = df.groupby('Description').agg({
        'Quantity': 'sum',
        'InvoiceNo': 'nunique',
        'UnitPrice': 'mean',
        'TotalPrice': 'sum'
    }).reset_index()

    product_df = product_df[(product_df["Quantity"] > 0) & (product_df["TotalPrice"] > 0)]

    # Step 2: Scale & Cluster
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans

    features = ['Quantity', 'InvoiceNo', 'UnitPrice', 'TotalPrice']
    scaler = StandardScaler()
    product_scaled = scaler.fit_transform(product_df[features])

    kmeans = KMeans(n_clusters=4, random_state=42)
    product_df['Cluster'] = kmeans.fit_predict(product_scaled)

    # Step 3: Cluster Summary
    st.markdown("### 📊 Product Cluster Overview")
    summary = product_df.groupby("Cluster")[features].mean().round(2)
    st.dataframe(summary)

    # Optional: Cluster Filter
    cluster_option = st.selectbox("🔍 Select a Product Cluster to Explore:", options=sorted(product_df['Cluster'].unique()))

    filtered_products = product_df[product_df['Cluster'] == cluster_option]

    # Step 4: Revenue vs Quantity Scatterplot
    import plotly.express as px
    st.markdown("### 💬 Products in Selected Cluster (Bubble = Price)")

    fig1 = px.scatter(
        filtered_products,
        x="Quantity",
        y="TotalPrice",
        size="UnitPrice",
        hover_name="Description",
        color="UnitPrice",
        color_continuous_scale="Viridis",
        title=f"Cluster {cluster_option}: Product Spread",
        size_max=35
    )
    st.plotly_chart(fig1, use_container_width=True)

    # Step 5: Top Products in the Cluster
    st.markdown("### 🏆 Top 10 Products in This Cluster (by Revenue)")
    top10 = filtered_products.sort_values("TotalPrice", ascending=False).head(10)
    st.dataframe(top10[["Description", "Quantity", "InvoiceNo", "UnitPrice", "TotalPrice"]])


# ----------------------------
# 5. Market Basket Analysis
# ----------------------------
if section == "Market Basket":
    st.markdown("## 🛒 Market Basket Analysis")
    st.markdown("Discover product combinations customers often buy together.")

    # Step 1: Clean data (remove refunds)
    mb_df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]
    transactions = mb_df.groupby("InvoiceNo")["Description"].apply(list).tolist()

    # Step 2: Encode transactions
    from mlxtend.preprocessing import TransactionEncoder
    from mlxtend.frequent_patterns import fpgrowth, association_rules

    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    basket = pd.DataFrame(te_ary, columns=te.columns_)

    # Step 3: Filters
    col1, col2 = st.columns(2)
    min_support = col1.slider("📊 Minimum Support", 0.005, 0.05, 0.01, step=0.005)
    min_confidence = col2.slider("✅ Minimum Confidence", 0.3, 0.9, 0.5, step=0.05)

    # Step 4: Run FP-Growth
    frequent_items = fpgrowth(basket, min_support=min_support, use_colnames=True)
    rules = association_rules(frequent_items, metric="confidence", min_threshold=min_confidence)

    if rules.empty:
        st.warning("No strong association rules found with current filters. Try lowering support or confidence.")
    else:
        rules["Rule"] = rules["antecedents"].apply(lambda x: ', '.join(x)) + " ➜ " + \
                        rules["consequents"].apply(lambda x: ', '.join(x))

        top_rules = rules.sort_values("lift", ascending=False).head(10)

        # Step 5: Rule Summary
        st.markdown("### 📋 Top 10 Association Rules")
        st.dataframe(top_rules[["Rule", "support", "confidence", "lift"]].round(3), use_container_width=True)

        # Step 6: Bar Chart of Lift
        import matplotlib.pyplot as plt
        import seaborn as sns

        st.markdown("### 📈 Rule Strength by Lift")

        fig1, ax1 = plt.subplots(figsize=(10, 5))
        sns.barplot(data=top_rules, y="Rule", x="lift", palette="viridis", ax=ax1)
        ax1.set_title("Top Rules by Lift Score")
        st.pyplot(fig1)

        # Step 6.5: Heatmap of Support vs Confidence (colored by Lift)
        st.markdown("### 🔥 Heatmap of Rule Density (Support vs Confidence)")

        import plotly.express as px

        # Use top 100 rules for performance
        heat_rules = rules.sort_values("lift", ascending=False).head(100)

        heatmap_df = heat_rules[["support", "confidence", "lift"]].round(3)

        fig_heat = px.density_heatmap(
            heatmap_df,
            x="confidence",
            y="support",
            z="lift",
            nbinsx=20,
            nbinsy=20,
            color_continuous_scale="YlOrRd",
            title="Support vs Confidence Heatmap (Lift as Color)"
        )
        fig_heat.update_layout(height=400)
        st.plotly_chart(fig_heat, use_container_width=True)

        # Step 6.6: Bubble Chart of Rules (Lift vs Confidence, Size = Support)
        st.markdown("### 💬 Bubble Chart: Confidence vs Lift (Bubble Size = Support)")

        bubble_df = rules.copy()
        bubble_df["Rule"] = bubble_df["antecedents"].apply(lambda x: ', '.join(x)) + " ➜ " + \
                             bubble_df["consequents"].apply(lambda x: ', '.join(x))
        bubble_df = bubble_df[["Rule", "support", "confidence", "lift"]].round(3)

        fig_bubble = px.scatter(
            bubble_df,
            x="confidence",
            y="lift",
            size="support",
            hover_name="Rule",
            color="lift",
            size_max=40,
            color_continuous_scale="Tealgrn",
            title="Association Rules Bubble Chart"
        )
        fig_bubble.update_traces(marker=dict(opacity=0.7, line=dict(width=1, color='DarkSlateGrey')))
        fig_bubble.update_layout(height=500)
        st.plotly_chart(fig_bubble, use_container_width=True)

        # Step 7: Network Graph
        st.markdown("### 🕸️ Product Association Network (Top 20 Rules)")
        import networkx as nx
        import plotly.graph_objects as go

        net_rules = rules.sort_values("lift", ascending=False).head(20)

        G = nx.DiGraph()
        for _, row in net_rules.iterrows():
            for antecedent in row['antecedents']:
                for consequent in row['consequents']:
                    G.add_edge(antecedent, consequent, weight=row['lift'], label=f"{row['confidence']:.2f}")

        pos = nx.spring_layout(G, k=0.5, seed=42)

        edge_x, edge_y = [], []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x += [x0, x1, None]
            edge_y += [y0, y1, None]

        edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=1, color='#888'),
                                hoverinfo='none', mode='lines')

        node_x, node_y, node_text = [], [], []
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(node)

        node_trace = go.Scatter(
            x=node_x, y=node_y, mode='markers+text', text=node_text,
            textposition="top center", hoverinfo='text',
            marker=dict(
                showscale=True, colorscale='YlGnBu', size=20,
                color=[len(list(G.neighbors(n))) for n in G.nodes()],
                colorbar=dict(
                    thickness=10,
                    title=dict(text="# Connections"),
                    xanchor='left'
                )
            )
        )

        fig_net = go.Figure(data=[edge_trace, node_trace],
                            layout=go.Layout(
                                title=dict(
                                    text="🕸️ Product Association Network (Top 20 Rules)",
                                    font=dict(size=18),
                                    x=0.5
                                ),
                                showlegend=False,
                                hovermode='closest',
                                margin=dict(b=20, l=5, r=5, t=40)
                            ))

        if G.number_of_nodes() > 0:
            st.plotly_chart(fig_net, use_container_width=True)
        else:
            st.info("No associations to show in network with current filters.")


    
    
# ----------------------------
# 6. CLTV & Churn
# ----------------------------
if section == "CLTV & Churn":
    st.markdown("## 💡 Customer Lifetime Value (CLTV) & Churn Analysis")

    # Step 1: Build features for each customer
    cust_df = df.groupby("CustomerID").agg({
        "InvoiceDate": [lambda x: (x.max() - x.min()).days, "nunique", "max"],
        "InvoiceNo": "nunique",
        "Quantity": "sum",
        "TotalPrice": "sum"
    }).reset_index()

    cust_df.columns = ["CustomerID", "TenureDays", "ActiveDays", "LastPurchaseDate", "InvoiceCount", "TotalQuantity", "TotalRevenue"]

    # Step 2: Compute CLTV (we're using Total Revenue as proxy)
    cust_df["CLTV"] = cust_df["TotalRevenue"]
    cust_df["CLTV_Segment"] = pd.qcut(cust_df["CLTV"], 4, labels=["Low", "Medium", "High", "Very High"])

    # Step 3: Flag Churned Customers (>90 days since last purchase)
    latest_date = df["InvoiceDate"].max()
    cust_df["Churned"] = cust_df["LastPurchaseDate"].apply(lambda x: 1 if (latest_date - x).days > 90 else 0)

    # Step 4: KPIs Summary
    st.markdown("### 📊 CLTV Segments Summary")
    summary = cust_df.groupby("CLTV_Segment").agg({
        "CustomerID": "count",
        "TotalRevenue": "mean",
        "TenureDays": "mean",
        "InvoiceCount": "mean",
        "Churned": "mean"
    }).round(2).rename(columns={
        "CustomerID": "NumCustomers",
        "TotalRevenue": "AvgRevenue",
        "TenureDays": "AvgTenure",
        "InvoiceCount": "AvgOrders",
        "Churned": "ChurnRate"
    })
    st.dataframe(summary, use_container_width=True)


    # Step 6: 💔 Interactive Churn Bar Chart
    st.markdown("### Churn Status of Customers")

    churn_counts = cust_df["Churned"].value_counts().rename({0: "Active", 1: "Churned"}).reset_index()
    churn_counts.columns = ["Status", "Count"]

    fig_churn = px.bar(
        churn_counts,
        x="Status",
        y="Count",
        color="Status",
        color_discrete_map={"Active": "green", "Churned": "red"},
        text="Count",
        title="💔 Churned vs Active Customers",
        height=400
    )
    fig_churn.update_traces(textposition="outside")
    fig_churn.update_layout(
        xaxis_title="Customer Status",
        yaxis_title="Number of Customers",
        showlegend=False,
        plot_bgcolor="white"
    )

    st.plotly_chart(fig_churn, use_container_width=True)


    # Step 7: Top CLTV Customers
    st.markdown("### 🏆 Top 10 Most Valuable Customers")
    top_cltv = cust_df.sort_values("CLTV", ascending=False).head(10)
    st.dataframe(top_cltv[["CustomerID", "CLTV", "InvoiceCount", "TotalQuantity", "TenureDays", "Churned"]])
    # Donut chart of CLTV segments
    st.markdown("### 🧁 CLTV Segment Breakdown")
    segment_counts = cust_df["CLTV_Segment"].value_counts().sort_index()

    import plotly.express as px
    fig_donut = px.pie(
        names=segment_counts.index,
        values=segment_counts.values,
        hole=0.5,
        color_discrete_sequence=px.colors.sequential.Teal
    )
    fig_donut.update_traces(textinfo="percent+label")
    fig_donut.update_layout(title_text="Customer Distribution by CLTV Segment")
    st.plotly_chart(fig_donut, use_container_width=True)

    # Churn rate by CLTV Segment (line)
    st.markdown("### 📉 Churn Rate by CLTV Segment")

    churn_seg = cust_df.groupby("CLTV_Segment")["Churned"].mean().reset_index()
    fig_line = px.line(
        churn_seg,
        x="CLTV_Segment",
        y="Churned",
        markers=True,
        title="Churn Rate by CLTV Segment",
        color_discrete_sequence=["crimson"]
    )
    fig_line.update_yaxes(title="Churn Rate")
    st.plotly_chart(fig_line, use_container_width=True)

    # Bubble Chart: Revenue vs Quantity (bubble = Tenure)
    st.markdown("### 🎯 CLTV Bubble Chart: Revenue vs Quantity (Bubble = Tenure)")

    bubble = cust_df.copy()
    fig_bubble = px.scatter(
        bubble,
        x="TotalQuantity",
        y="TotalRevenue",
        size="TenureDays",
        color="CLTV_Segment",
        hover_data=["CustomerID", "InvoiceCount"],
        title="Customer Value by Quantity, Revenue & Tenure",
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    fig_bubble.update_layout(height=500)
    st.plotly_chart(fig_bubble, use_container_width=True)
