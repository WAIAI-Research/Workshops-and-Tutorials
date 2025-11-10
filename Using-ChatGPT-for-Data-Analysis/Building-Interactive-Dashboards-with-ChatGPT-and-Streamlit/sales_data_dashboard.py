import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load data
@st.cache_data
def load_data():
    customers = pd.read_csv("customers.csv")
    orders = pd.read_csv("orders.csv", parse_dates=["Order Date", "Ship Date"])
    returns = pd.read_csv("returns.csv")
    return customers, orders, returns

customers_df, orders_df, returns_df = load_data()

# Merge for full analysis
merged_df = orders_df.merge(customers_df, on="Customer ID", how="left")
merged_df = merged_df.merge(returns_df, on="Order ID", how="left")

# Sidebar filters
st.sidebar.header("Filter Options")
selected_region = st.sidebar.multiselect("Region", merged_df["Region"].dropna().unique())
selected_category = st.sidebar.multiselect("Category", merged_df["Category"].dropna().unique())
date_range = st.sidebar.date_input("Order Date Range", [
    merged_df["Order Date"].min(), merged_df["Order Date"].max()
])

# Filter data
filtered_df = merged_df.copy()
if selected_region:
    filtered_df = filtered_df[filtered_df["Region"].isin(selected_region)]
if selected_category:
    filtered_df = filtered_df[filtered_df["Category"].isin(selected_category)]
filtered_df = filtered_df[
    (filtered_df["Order Date"] >= pd.to_datetime(date_range[0])) &
    (filtered_df["Order Date"] <= pd.to_datetime(date_range[1]))
]

# Title and filter summary
st.title("📊 Customer Orders Dashboard")
st.markdown(f"**Filters:** Region={selected_region or 'All'}, Category={selected_category or 'All'}, "
            f"Date Range={date_range[0]} to {date_range[1]}")

# Key metrics
st.metric("Total Sales", f"${filtered_df['Sales'].sum():,.2f}")
st.metric("Total Profit", f"${filtered_df['Profit'].sum():,.2f}")
st.metric("Total Orders", filtered_df["Order ID"].nunique())

# Visualizations
st.subheader("📅 Sales Over Time")
sales_over_time = filtered_df.groupby("Order Date")["Sales"].sum().reset_index()
st.line_chart(sales_over_time.rename(columns={"Order Date": "index"}).set_index("index"))

st.subheader("📍 Sales by Region")
region_sales = filtered_df.groupby("Region")["Sales"].sum().sort_values()
st.bar_chart(region_sales)

st.subheader("📦 Returns Overview")
returns_count = filtered_df["Returned"].value_counts(dropna=False)
st.bar_chart(returns_count)

# Sample data
st.subheader("🔍 Sample Data")
st.dataframe(filtered_df.head(5))

# Summary table
st.subheader("📋 Data Summary")
summary_data = []
for col in filtered_df.columns:
    summary_data.append({
        "Column": col,
        "Data Type": filtered_df[col].dtype,
        "Missing Values": filtered_df[col].isnull().sum(),
        "Unique Values": filtered_df[col].nunique()
    })
summary_df = pd.DataFrame(summary_data)
st.dataframe(summary_df)

# Dynamic Visualization
st.subheader("📊 Custom Visualization")
viz_type = st.selectbox("Choose Chart Type", ["Scatterplot", "Lineplot", "Barplot"])
x_col = st.selectbox("Select X-axis", filtered_df.select_dtypes(include=["number", "object", "category"]).columns)
y_col = st.selectbox("Select Y-axis", filtered_df.select_dtypes(include=["number"]).columns)

fig, ax = plt.subplots()
if viz_type == "Scatterplot":
    sns.scatterplot(data=filtered_df, x=x_col, y=y_col, ax=ax)
elif viz_type == "Lineplot":
    sns.lineplot(data=filtered_df, x=x_col, y=y_col, ax=ax)
elif viz_type == "Barplot":
    sns.barplot(data=filtered_df, x=x_col, y=y_col, ax=ax)
st.pyplot(fig)

# Export filtered data
st.subheader("💾 Export Filtered Data")
csv = filtered_df.to_csv(index=False).encode("utf-8")
st.download_button("Download Filtered Data as CSV", data=csv, file_name="filtered_data.csv", mime="text/csv")
