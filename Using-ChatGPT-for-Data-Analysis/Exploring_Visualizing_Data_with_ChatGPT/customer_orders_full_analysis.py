import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =========================================================
# Customer + Orders + Returns Analysis (Full code so far)
# =========================================================
# Files expected at:
#   customers.csv
#   orders.csv
#   returns.csv
#
# If you're running locally, change paths accordingly.
# (In this environment, files are located in /mnt/data/)
# =========================================================

CUSTOMERS_PATH = "/mnt/data/customers.csv"
ORDERS_PATH = "/mnt/data/orders.csv"
RETURNS_PATH = "/mnt/data/returns.csv"

# -----------------------------
# 1) Load datasets
# -----------------------------
customers = pd.read_csv(CUSTOMERS_PATH)
orders = pd.read_csv(ORDERS_PATH)
returns = pd.read_csv(RETURNS_PATH)

print("Loaded shapes:")
print("  customers:", customers.shape)
print("  orders   :", orders.shape)
print("  returns  :", returns.shape)

# -----------------------------
# 2) Summary tables for each file
# -----------------------------
def summarize_df(df: pd.DataFrame, descriptions: dict) -> pd.DataFrame:
    summary_rows = []
    for col in df.columns:
        series = df[col]
        row = {
            "Column": col,
            "Data Type": str(series.dtype),
            "Description": descriptions.get(col, ""),
            "Missing Values": int(series.isna().sum()),
            "Unique Values": int(series.nunique(dropna=True)),
        }
        if pd.api.types.is_numeric_dtype(series):
            row.update({
                "Min": float(series.min()),
                "Max": float(series.max()),
                "Mean": float(series.mean()),
                "Median": float(series.median()),
                "Std Dev": float(series.std()),
            })
        else:
            row.update({"Min": None, "Max": None, "Mean": None, "Median": None, "Std Dev": None})
        summary_rows.append(row)
    return pd.DataFrame(summary_rows)

customer_desc = {
    "Customer ID": "Unique identifier for each customer",
    "Segment": "Customer segment (Consumer, Corporate, Home Office)",
    "Country": "Country where the customer is located",
    "City": "Customer city",
    "State": "Customer state",
    "Postal Code": "Customer postal/ZIP code",
    "Region": "Sales region",
}
orders_desc = {
    "Row ID": "Unique row identifier for each order line",
    "Order ID": "Unique order identifier",
    "Order Date": "Date the order was placed",
    "Ship Date": "Date the order was shipped",
    "Ship Mode": "Shipping method",
    "Customer ID": "Customer identifier",
    "Product ID": "Product identifier",
    "Category": "Product category",
    "Sub-Category": "Product sub-category",
    "Product Name": "Product name",
    "Sales": "Sales revenue for the order line",
    "Quantity": "Quantity ordered",
    "Discount": "Discount applied",
    "Profit": "Profit for the order line",
}
returns_desc = {
    "Returned": "Indicates whether the order was returned",
    "Order ID": "Identifier of the returned order",
}

customers_summary = summarize_df(customers, customer_desc)
orders_summary = summarize_df(orders, orders_desc)
returns_summary = summarize_df(returns, returns_desc)

print("\nCustomers summary (head):")
print(customers_summary.head())
print("\nOrders summary (head):")
print(orders_summary.head())
print("\nReturns summary:")
print(returns_summary)

# -----------------------------
# 3) Convert date columns to datetime
# -----------------------------
orders["Order Date"] = pd.to_datetime(orders["Order Date"], format="%m/%d/%y")
orders["Ship Date"] = pd.to_datetime(orders["Ship Date"], format="%m/%d/%y")

print("\nOrders dtypes after date conversion:")
print(orders.dtypes)

# -----------------------------
# 4) Join data + integrity checks (customers ↔ orders ↔ returns)
# -----------------------------
unique_customers_customers = customers["Customer ID"].nunique()
unique_customers_orders = orders["Customer ID"].nunique()
unique_orders = orders["Order ID"].nunique()

orders_with_customer = orders.merge(customers[["Customer ID"]], on="Customer ID", how="left", indicator=True)
unmatched_order_rows = int((orders_with_customer["_merge"] != "both").sum())

# returns flag (order-level)
returns_flag = returns.assign(Returned_Flag=1)[["Order ID", "Returned_Flag"]]

merged = orders.merge(customers, on="Customer ID", how="left")
merged = merged.merge(returns_flag, on="Order ID", how="left")
merged["Returned_Flag"] = merged["Returned_Flag"].fillna(0).astype(int)

print("\nIntegrity checks:")
print("  Unique customers (customers table):", unique_customers_customers)
print("  Unique customers (orders table)   :", unique_customers_orders)
print("  Unique orders                     :", unique_orders)
print("  Unmatched order rows              :", unmatched_order_rows)
print("  Merged shape                      :", merged.shape)

# -----------------------------
# 5) Monthly sales over time (all months)
#    - overall line chart
# -----------------------------
orders["Month"] = orders["Order Date"].dt.to_period("M").dt.to_timestamp()

monthly_sales = (
    orders.groupby("Month")["Sales"]
    .sum()
    .reset_index()
)

plt.figure()
plt.plot(monthly_sales["Month"], monthly_sales["Sales"])
plt.xlabel("Month")
plt.ylabel("Total Sales")
plt.title("Monthly Sales Over Time (Line Chart)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# -----------------------------
# 6) Profit distribution by category (box plot)
# -----------------------------
plt.figure()
orders.boxplot(column="Profit", by="Category")
plt.title("Profit Distribution by Category (Box Plot)")
plt.suptitle("")
plt.xlabel("Category")
plt.ylabel("Profit")
plt.tight_layout()
plt.show()

# -----------------------------
# 7) Monthly sales over time by category (all months) - wider plot
# -----------------------------
monthly_sales_category = (
    orders.groupby(["Month", "Category"])["Sales"]
    .sum()
    .reset_index()
)
pivot_all = monthly_sales_category.pivot(index="Month", columns="Category", values="Sales")

plt.figure(figsize=(14, 6))
for cat in pivot_all.columns:
    plt.plot(pivot_all.index, pivot_all[cat], label=cat)
plt.xlabel("Month")
plt.ylabel("Total Sales")
plt.title("Monthly Sales Over Time by Product Category (Line Chart)")
plt.legend(title="Category")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# -----------------------------
# 8) Monthly sales by category (Jul 2016 – Jul 2017) + download
# -----------------------------
start_date = pd.Timestamp("2016-07-01")
end_date = pd.Timestamp("2017-07-31")
filtered = orders[(orders["Month"] >= start_date) & (orders["Month"] <= end_date)]

monthly_sales_cat_window = (
    filtered.groupby(["Month", "Category"])["Sales"]
    .sum()
    .reset_index()
)
pivot_window = monthly_sales_cat_window.pivot(index="Month", columns="Category", values="Sales")

plt.figure(figsize=(14, 6))
for cat in pivot_window.columns:
    plt.plot(pivot_window.index, pivot_window[cat], label=cat)
plt.xlabel("Month")
plt.ylabel("Total Sales")
plt.title("Monthly Sales by Product Category (July 2016 – July 2017)\n(Line Chart)")
plt.legend(title="Category")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Save window chart
out_monthly_window = "/mnt/data/monthly_sales_by_category_jul2016_jul2017.png"
plt.figure(figsize=(14, 6))
for cat in pivot_window.columns:
    plt.plot(pivot_window.index, pivot_window[cat], label=cat)
plt.xlabel("Month")
plt.ylabel("Total Sales")
plt.title("Monthly Sales by Product Category (July 2016 – July 2017)\n(Line Chart)")
plt.legend(title="Category")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(out_monthly_window, dpi=150)
plt.close()

# Save profit distribution boxplot
out_profit_box = "/mnt/data/profit_distribution_by_category.png"
plt.figure(figsize=(10, 6))
orders.boxplot(column="Profit", by="Category")
plt.title("Profit Distribution by Category\n(Box Plot)")
plt.suptitle("")
plt.xlabel("Category")
plt.ylabel("Profit")
plt.tight_layout()
plt.savefig(out_profit_box, dpi=150)
plt.close()

print("\nSaved charts:")
print(" ", out_monthly_window)
print(" ", out_profit_box)

# -----------------------------
# 9) Return rate by category (order-level) + plot
# -----------------------------
returns_only = returns.copy()
returns_only["Returned_Flag"] = 1

orders_ret = orders.merge(returns_only[["Order ID", "Returned_Flag"]], on="Order ID", how="left")
orders_ret["Returned_Flag"] = orders_ret["Returned_Flag"].fillna(0)

order_level = (
    orders_ret.groupby(["Order ID", "Category"], as_index=False)
    .agg(Returned=("Returned_Flag", "max"))
)

return_rate_by_category = (
    order_level.groupby("Category")["Returned"]
    .mean()
    .reset_index(name="Return Rate")
)

plt.figure(figsize=(8, 5))
plt.bar(return_rate_by_category["Category"], return_rate_by_category["Return Rate"])
plt.xlabel("Category")
plt.ylabel("Return Rate")
plt.title("Return Rate by Product Category (Bar Chart)")
plt.tight_layout()
plt.show()

print("\nReturn rate by category:")
print(return_rate_by_category.sort_values("Return Rate", ascending=False))

# -----------------------------
# 10) Technology return rate by sub-category (order-level) + colored plot + labels + download
# -----------------------------
tech = orders_ret[orders_ret["Category"] == "Technology"]

tech_order_level = (
    tech.groupby(["Order ID", "Sub-Category"], as_index=False)
    .agg(Returned=("Returned_Flag", "max"))
)

tech_return_rate = (
    tech_order_level.groupby("Sub-Category")["Returned"]
    .mean()
    .reset_index(name="Return Rate")
    .sort_values("Return Rate", ascending=False)
)

# sanity check table
sanity_check = (
    tech_order_level.groupby("Sub-Category")
    .agg(Total_Orders=("Order ID", "nunique"), Returned_Orders=("Returned", "sum"))
    .reset_index()
)

colors = ["#d62728", "#ff7f0e", "#2ca02c", "#1f77b4"]  # distinct colors
plt.figure(figsize=(10, 6))
bars = plt.bar(tech_return_rate["Sub-Category"], tech_return_rate["Return Rate"], color=colors[:len(tech_return_rate)])

for bar, rate in zip(bars, tech_return_rate["Return Rate"]):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f"{rate:.1%}", ha="center", va="bottom", fontsize=10)

plt.xlabel("Technology Sub-Category")
plt.ylabel("Return Rate")
plt.title("Return Rate by Technology Sub-Category\n(With Percentage Labels)")
plt.xticks(rotation=30, ha="right")
plt.ylim(0, tech_return_rate["Return Rate"].max() * 1.3)
plt.tight_layout()
plt.show()

out_tech_returns = "/mnt/data/technology_return_rate_by_subcategory.png"
plt.figure(figsize=(10, 6))
bars = plt.bar(tech_return_rate["Sub-Category"], tech_return_rate["Return Rate"], color=colors[:len(tech_return_rate)])
for bar, rate in zip(bars, tech_return_rate["Return Rate"]):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f"{rate:.1%}", ha="center", va="bottom", fontsize=10)
plt.xlabel("Technology Sub-Category")
plt.ylabel("Return Rate")
plt.title("Return Rate by Technology Sub-Category\n(With Percentage Labels)")
plt.xticks(rotation=30, ha="right")
plt.ylim(0, tech_return_rate["Return Rate"].max() * 1.3)
plt.tight_layout()
plt.savefig(out_tech_returns, dpi=150)
plt.close()

print("\nTechnology return rate by sub-category:")
print(tech_return_rate)
print("\nSanity check counts:")
print(sanity_check)
print("\nSaved chart:", out_tech_returns)

# -----------------------------
# 11) Total sales by region (color-coded) + download
# -----------------------------
orders_with_region = orders.merge(customers[["Customer ID", "Region"]], on="Customer ID", how="left")

sales_by_region = (
    orders_with_region.groupby("Region")["Sales"]
    .sum()
    .reset_index()
    .sort_values("Sales", ascending=False)
)

region_colors = {"West": "#1f77b4", "East": "#2ca02c", "Central": "#ff7f0e", "South": "#d62728"}
colors_region = [region_colors.get(r, "#7f7f7f") for r in sales_by_region["Region"]]

plt.figure(figsize=(8, 5))
bars = plt.bar(sales_by_region["Region"], sales_by_region["Sales"], color=colors_region)
for bar, value in zip(bars, sales_by_region["Sales"]):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f"${value/1_000_000:.2f}M", ha="center", va="bottom", fontsize=10)
plt.xlabel("Region")
plt.ylabel("Total Sales")
plt.title("Total Sales by Region\n(Color-Coded Bar Chart)")
plt.tight_layout()
plt.show()

out_sales_region = "/mnt/data/total_sales_by_region_color_coded.png"
plt.figure(figsize=(8, 5))
bars = plt.bar(sales_by_region["Region"], sales_by_region["Sales"], color=colors_region)
for bar, value in zip(bars, sales_by_region["Sales"]):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f"${value/1_000_000:.2f}M", ha="center", va="bottom", fontsize=10)
plt.xlabel("Region")
plt.ylabel("Total Sales")
plt.title("Total Sales by Region\n(Color-Coded Bar Chart)")
plt.tight_layout()
plt.savefig(out_sales_region, dpi=150)
plt.close()

print("\nSales by region:")
print(sales_by_region)
print("\nSaved chart:", out_sales_region)

# -----------------------------
# 12) Monthly sales by category (all months) with Q4 shading + download
# -----------------------------
plt.figure(figsize=(14, 6))
for cat in pivot_all.columns:
    plt.plot(pivot_all.index, pivot_all[cat], label=cat)

years = pivot_all.index.year.unique()
for year in years:
    q4_start = pd.Timestamp(f"{year}-10-01")
    q4_end = pd.Timestamp(f"{year}-12-31")
    plt.axvspan(q4_start, q4_end, alpha=0.12)

plt.xlabel("Month")
plt.ylabel("Total Sales")
plt.title("Monthly Sales Over Time by Product Category\nShaded regions indicate Q4 (Oct–Dec) seasonality")
plt.legend(title="Category")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

out_q4 = "/mnt/data/monthly_sales_by_category_with_q4_shading.png"
plt.figure(figsize=(14, 6))
for cat in pivot_all.columns:
    plt.plot(pivot_all.index, pivot_all[cat], label=cat)
for year in years:
    q4_start = pd.Timestamp(f"{year}-10-01")
    q4_end = pd.Timestamp(f"{year}-12-31")
    plt.axvspan(q4_start, q4_end, alpha=0.12)
plt.xlabel("Month")
plt.ylabel("Total Sales")
plt.title("Monthly Sales Over Time by Product Category\nShaded regions indicate Q4 (Oct–Dec) seasonality")
plt.legend(title="Category")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(out_q4, dpi=150)
plt.close()

print("\nSaved chart:", out_q4)

# -----------------------------
# 13) Most profitable segment + Consumer total profit
# -----------------------------
orders_with_segment = orders.merge(customers[["Customer ID", "Segment"]], on="Customer ID", how="left")
profit_by_segment = (
    orders_with_segment.groupby("Segment")["Profit"]
    .sum()
    .reset_index()
    .sort_values("Profit", ascending=False)
)
print("\nTotal profit by segment:")
print(profit_by_segment)

consumer_profit = orders_with_segment.loc[orders_with_segment["Segment"] == "Consumer", "Profit"].sum()
print("\nConsumer segment total profit:", consumer_profit)

# -----------------------------
# 14) Average Order Value (AOV) by city/state + specific city AOV (Saginaw)
#     (Order-level AOV = average of order totals)
# -----------------------------
orders_loc = orders.merge(customers[["Customer ID", "City", "State"]], on="Customer ID", how="left")

order_level_sales_loc = (
    orders_loc.groupby(["Order ID", "City", "State"], as_index=False)["Sales"]
    .sum()
)

aov_by_city = (
    order_level_sales_loc.groupby("City")["Sales"]
    .mean()
    .reset_index(name="Average Order Value")
    .sort_values("Average Order Value", ascending=False)
)

aov_by_state = (
    order_level_sales_loc.groupby("State")["Sales"]
    .mean()
    .reset_index(name="Average Order Value")
    .sort_values("Average Order Value", ascending=False)
)

print("\nTop 10 cities by AOV:")
print(aov_by_city.head(10))

print("\nTop 10 states by AOV:")
print(aov_by_state.head(10))

saginaw_aov = order_level_sales_loc.loc[order_level_sales_loc["City"] == "Saginaw", "Sales"].mean()
print("\nSaginaw AOV:", saginaw_aov)
