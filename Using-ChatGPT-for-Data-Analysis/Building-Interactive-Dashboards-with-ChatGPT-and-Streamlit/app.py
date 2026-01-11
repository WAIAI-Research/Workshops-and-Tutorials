# app.py (enhanced + interactive charts via Plotly)
import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="Customer Orders Dashboard", layout="wide")

@st.cache_data(show_spinner=False)
def load_csv(uploaded_file, fallback_path: str) -> pd.DataFrame:
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)
    return pd.read_csv(fallback_path)

def convert_dates(orders: pd.DataFrame) -> pd.DataFrame:
    orders = orders.copy()
    orders["Order Date"] = pd.to_datetime(orders["Order Date"], format="%m/%d/%y", errors="coerce")
    orders["Ship Date"] = pd.to_datetime(orders["Ship Date"], format="%m/%d/%y", errors="coerce")
    return orders

def build_model(customers: pd.DataFrame, orders: pd.DataFrame, returns: pd.DataFrame) -> pd.DataFrame:
    returns_flag = returns.copy()
    returns_flag["Returned_Flag"] = 1

    orders_ret = orders.merge(
        returns_flag[["Order ID", "Returned_Flag"]],
        on="Order ID",
        how="left",
    )
    orders_ret["Returned_Flag"] = orders_ret["Returned_Flag"].fillna(0).astype(int)

    merged_line = orders_ret.merge(customers, on="Customer ID", how="left")
    return merged_line

def order_level_totals(merged_line: pd.DataFrame) -> pd.DataFrame:
    dims = [c for c in ["Customer ID", "Segment", "Region", "City", "State", "Ship Mode", "Order Date", "Ship Date"]
            if c in merged_line.columns]
    out = (
        merged_line.groupby(["Order ID"], as_index=False)
        .agg(
            **{d: (d, "first") for d in dims},
            Sales=("Sales", "sum"),
            Profit=("Profit", "sum"),
            Returned=("Returned_Flag", "max"),
        )
    )
    return out

def money0(x): 
    return f"${x:,.0f}"

def pct1(x):
    return f"{x:.1%}"

def summarize_df(df: pd.DataFrame, descriptions: dict) -> pd.DataFrame:
    rows = []
    for col in df.columns:
        s = df[col]
        rows.append({
            "Column": col,
            "Data Type": str(s.dtype),
            "Description": descriptions.get(col, ""),
            "Missing Values": int(s.isna().sum()),
            "Unique Values": int(s.nunique(dropna=True)),
        })
    return pd.DataFrame(rows)

COLUMN_DESCRIPTIONS = {
    "Customer ID": "Unique identifier for each customer",
    "Segment": "Customer segment (Consumer, Corporate, Home Office)",
    "Country": "Customer country",
    "City": "Customer city",
    "State": "Customer state",
    "Postal Code": "Customer postal/ZIP code",
    "Region": "Sales region",
    "Row ID": "Unique row identifier for each order line",
    "Order ID": "Unique order identifier",
    "Order Date": "Date the order was placed",
    "Ship Date": "Date the order was shipped",
    "Ship Mode": "Shipping method",
    "Product ID": "Product identifier",
    "Category": "Product category",
    "Sub-Category": "Product sub-category",
    "Product Name": "Product name",
    "Sales": "Sales revenue for the order line",
    "Quantity": "Quantity ordered",
    "Discount": "Discount applied to the order line",
    "Profit": "Profit for the order line",
    "Returned": "Return status text (if present in returns file)",
    "Returned_Flag": "1 if order returned, else 0",
}

st.sidebar.title("Data Inputs")
use_uploads = st.sidebar.toggle("Upload CSVs instead of local files", value=False)

customers_file = st.sidebar.file_uploader("customers.csv", type=["csv"]) if use_uploads else None
orders_file = st.sidebar.file_uploader("orders.csv", type=["csv"]) if use_uploads else None
returns_file = st.sidebar.file_uploader("returns.csv", type=["csv"]) if use_uploads else None

CUSTOMERS_PATH = "customers.csv"
ORDERS_PATH = "orders.csv"
RETURNS_PATH = "returns.csv"

try:
    customers = load_csv(customers_file, CUSTOMERS_PATH)
    orders_raw = load_csv(orders_file, ORDERS_PATH)
    returns = load_csv(returns_file, RETURNS_PATH)
except Exception:
    st.error("Could not load CSVs. Include them in the repo root or use uploads.")
    st.stop()

orders = convert_dates(orders_raw)
merged_line = build_model(customers, orders, returns)

st.sidebar.title("Filters")

min_date = merged_line["Order Date"].min()
max_date = merged_line["Order Date"].max()
if pd.isna(min_date) or pd.isna(max_date):
    st.error("Order Date is missing or could not be parsed. Please check your orders.csv date format (MM/DD/YY).")
    st.stop()

date_range = st.sidebar.date_input(
    "Order Date range",
    value=(min_date.date(), max_date.date()),
)

all_segments = sorted([s for s in merged_line["Segment"].dropna().unique().tolist()])
all_regions = sorted([r for r in merged_line["Region"].dropna().unique().tolist()])
all_categories = sorted([c for c in merged_line["Category"].dropna().unique().tolist()])

seg_sel = st.sidebar.multiselect("Segments", options=all_segments, default=all_segments)
reg_sel = st.sidebar.multiselect("Regions", options=all_regions, default=all_regions)
cat_sel = st.sidebar.multiselect("Product Categories", options=all_categories, default=all_categories)

f = merged_line.copy()

start = pd.to_datetime(date_range[0])
end = pd.to_datetime(date_range[1]) + pd.Timedelta(days=1)
f = f[(f["Order Date"] >= start) & (f["Order Date"] < end)]

if seg_sel:
    f = f[f["Segment"].isin(seg_sel)]
else:
    f = f.iloc[0:0]

if reg_sel:
    f = f[f["Region"].isin(reg_sel)]
else:
    f = f.iloc[0:0]

if cat_sel:
    f = f[f["Category"].isin(cat_sel)]
else:
    f = f.iloc[0:0]

f_order = order_level_totals(f)

st.title("Customer Orders & Returns Dashboard")

total_sales = float(f["Sales"].sum()) if len(f) else 0.0
total_profit = float(f["Profit"].sum()) if len(f) else 0.0
unique_customers = int(f["Customer ID"].nunique()) if len(f) else 0
unique_orders = int(f["Order ID"].nunique()) if len(f) else 0
return_rate = float(f_order["Returned"].mean()) if len(f_order) else np.nan

st.subheader("Key metrics (based on current filters)")
st.markdown(
    "\n".join([
        f"- **Total Sales:** {money0(total_sales)}",
        f"- **Total Profit:** {money0(total_profit)}",
        f"- **Unique Customers:** {unique_customers:,}",
        f"- **Unique Orders:** {unique_orders:,}",
        f"- **Return Rate (order-level):** {pct1(return_rate) if pd.notna(return_rate) else '—'}",
    ])
)

st.divider()

tab_sales, tab_returns, tab_geo, tab_vizbuilder, tab_data = st.tabs(
    ["Sales", "Returns", "Geography", "Build a Chart", "Data"]
)

filters_text = (
    f"Dates: {date_range[0]} → {date_range[1]} | "
    f"Segments: {', '.join(seg_sel) if seg_sel else 'None'} | "
    f"Regions: {', '.join(reg_sel) if reg_sel else 'None'} | "
    f"Categories: {', '.join(cat_sel) if cat_sel else 'None'}"
)

with tab_sales:
    st.subheader("Monthly Sales Over Time by Product Category (Interactive)")
    st.caption(f"**Selected filters** — {filters_text}")

    if len(f) == 0:
        st.info("No data matches the selected filters.")
    else:
        monthly = (
            f.assign(Month=f["Order Date"].dt.to_period("M").dt.to_timestamp())
            .groupby(["Month", "Category"], as_index=False)["Sales"]
            .sum()
        )
        fig = px.line(
            monthly,
            x="Month",
            y="Sales",
            color="Category",
            markers=True,
            title="Monthly Sales by Category",
        )
        fig.update_layout(
            xaxis_title="Month",
            yaxis_title="Total Sales",
            legend_title_text="Category",
            height=450,
            margin=dict(l=20, r=20, t=60, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)

with tab_returns:
    st.subheader("Return Rate by Category (order-level, Interactive)")
    st.caption(f"**Selected filters** — {filters_text}")

    if len(f) == 0:
        st.info("No data matches the selected filters.")
    else:
        order_cat = (
            f.groupby(["Order ID", "Category"], as_index=False)
            .agg(Returned=("Returned_Flag", "max"))
        )
        rr_cat = (
            order_cat.groupby("Category")["Returned"]
            .mean()
            .reset_index(name="Return Rate")
            .sort_values("Return Rate", ascending=False)
        )

        st.dataframe(rr_cat, use_container_width=True)

        fig = px.bar(
            rr_cat,
            x="Category",
            y="Return Rate",
            text=rr_cat["Return Rate"].map(lambda v: f"{v:.1%}"),
            title="Return Rate by Category",
        )
        fig.update_traces(textposition="outside", cliponaxis=False)
        fig.update_layout(
            xaxis_title="Category",
            yaxis_title="Return Rate",
            yaxis_tickformat=".0%",
            height=420,
            margin=dict(l=20, r=20, t=60, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)

with tab_geo:
    st.subheader("Total Sales by Region (Interactive)")
    st.caption(f"**Selected filters** — {filters_text}")

    if len(f) == 0:
        st.info("No data matches the selected filters.")
    else:
        sales_region = (
            f.groupby("Region")["Sales"]
            .sum()
            .reset_index()
            .sort_values("Sales", ascending=False)
        )

        fig = px.bar(
            sales_region,
            x="Region",
            y="Sales",
            text=sales_region["Sales"].map(lambda v: f"${v/1_000_000:.2f}M"),
            title="Total Sales by Region",
        )
        fig.update_traces(textposition="outside", cliponaxis=False)
        fig.update_layout(
            xaxis_title="Region",
            yaxis_title="Total Sales",
            height=450,
            margin=dict(l=20, r=20, t=60, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)

with tab_vizbuilder:
    st.subheader("Build a custom chart (Interactive)")
    st.caption("Choose a chart type and columns to visualize. This uses the filtered dataset from the sidebar.")

    if len(f) == 0:
        st.info("No data matches the selected filters.")
    else:
        dataset_choice = st.radio(
            "Dataset to use",
            ["Line-item (detailed)", "Order-level (one row per order)"],
            horizontal=True
        )
        df = f.copy() if dataset_choice.startswith("Line-item") else f_order.copy()

        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        categorical_cols = [c for c in df.columns if df[c].dtype == "object" or str(df[c].dtype).startswith("category")]
        date_cols = [c for c in df.columns if np.issubdtype(df[c].dtype, np.datetime64)]

        chart_type = st.selectbox("Chart type", ["Scatterplot", "Line", "Bar (aggregate)", "Histogram", "Box plot"])

        if chart_type == "Scatterplot":
            if len(numeric_cols) < 2:
                st.warning("Need at least two numeric columns for a scatterplot.")
            else:
                x = st.selectbox("X (numeric)", numeric_cols, index=0)
                y = st.selectbox("Y (numeric)", numeric_cols, index=1)
                color_by = st.selectbox("Color/group (optional)", ["None"] + categorical_cols)

                fig = px.scatter(
                    df,
                    x=x,
                    y=y,
                    color=None if color_by == "None" else color_by,
                    hover_data=[c for c in ["Order ID", "Customer ID", "Category", "Region", "Segment"] if c in df.columns],
                    title=f"Scatterplot: {y} vs {x}",
                )
                fig.update_layout(height=480, margin=dict(l=20, r=20, t=60, b=20))
                st.plotly_chart(fig, use_container_width=True)

        elif chart_type == "Line":
            x_options = date_cols + numeric_cols
            if not x_options or not numeric_cols:
                st.warning("Need at least one X (date/numeric) and one numeric Y column for a line chart.")
            else:
                x = st.selectbox("X (date or numeric)", x_options, index=0)
                y = st.selectbox("Y (numeric)", numeric_cols, index=0)
                group = st.selectbox("Group (optional)", ["None"] + categorical_cols)

                plot_df = df[[x, y] + ([group] if group != "None" else [])].dropna().sort_values(x)

                fig = px.line(
                    plot_df,
                    x=x,
                    y=y,
                    color=None if group == "None" else group,
                    markers=True,
                    title=f"Line chart: {y} over {x}",
                )
                fig.update_layout(height=480, margin=dict(l=20, r=20, t=60, b=20))
                st.plotly_chart(fig, use_container_width=True)

        elif chart_type == "Bar (aggregate)":
            if not categorical_cols or not numeric_cols:
                st.warning("Need at least one categorical group column and one numeric value column for an aggregated bar chart.")
            else:
                group_col = st.selectbox("Group by (categorical)", categorical_cols, index=0)
                value_col = st.selectbox("Value (numeric)", numeric_cols, index=0)
                agg = st.selectbox("Aggregation", ["sum", "mean", "median", "count"])

                if agg == "count":
                    bar_df = df.groupby(group_col)[value_col].count().reset_index(name=f"{agg}({value_col})")
                else:
                    bar_df = df.groupby(group_col)[value_col].agg(agg).reset_index(name=f"{agg}({value_col})")

                val_name = bar_df.columns[-1]
                bar_df = bar_df.sort_values(val_name, ascending=False).head(25)

                fig = px.bar(
                    bar_df,
                    x=group_col,
                    y=val_name,
                    title=f"Bar chart: {val_name} by {group_col} (top 25)",
                    text=val_name,
                )
                fig.update_traces(textposition="outside", cliponaxis=False)
                fig.update_layout(height=480, margin=dict(l=20, r=20, t=60, b=80))
                st.plotly_chart(fig, use_container_width=True)

        elif chart_type == "Histogram":
            if not numeric_cols:
                st.warning("No numeric columns available for histogram.")
            else:
                col = st.selectbox("Numeric column", numeric_cols, index=0)
                bins = st.slider("Bins", min_value=5, max_value=100, value=30)
                fig = px.histogram(df, x=col, nbins=bins, title=f"Histogram: {col}")
                fig.update_layout(height=480, margin=dict(l=20, r=20, t=60, b=20))
                st.plotly_chart(fig, use_container_width=True)

        elif chart_type == "Box plot":
            if not numeric_cols or not categorical_cols:
                st.warning("Need at least one numeric column and one categorical grouping column for a box plot.")
            else:
                numeric = st.selectbox("Numeric column", numeric_cols, index=0)
                group = st.selectbox("Group by (categorical)", categorical_cols, index=0)
                fig = px.box(
                    df,
                    x=group,
                    y=numeric,
                    points="outliers",
                    title=f"Box plot: {numeric} by {group}",
                )
                fig.update_layout(height=480, margin=dict(l=20, r=20, t=60, b=80))
                st.plotly_chart(fig, use_container_width=True)

with tab_data:
    st.subheader("Initial snapshot (first 5 rows of filtered data)")
    st.caption("Snapshot reflects the **filtered** dataset (line-item level).")
    st.dataframe(f.head(5), use_container_width=True)

    st.subheader("Summary table (filtered data)")
    st.caption("Column name, dtype, description, missing values, unique values.")
    summary = summarize_df(f, COLUMN_DESCRIPTIONS)
    st.dataframe(summary, use_container_width=True)

    st.subheader("Export filtered data")
    st.caption("Exports the filtered **line-item** dataset as a CSV.")
    csv_bytes = f.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download filtered data (CSV)",
        data=csv_bytes,
        file_name="filtered_customer_orders_data.csv",
        mime="text/csv",
    )
