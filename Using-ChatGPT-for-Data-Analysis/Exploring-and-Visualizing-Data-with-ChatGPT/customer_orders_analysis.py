import pandas as pd

# Load the provided CSV files
customers_df = pd.read_csv("/mnt/data/customers.csv")
orders_df = pd.read_csv("/mnt/data/orders.csv")
returns_df = pd.read_csv("/mnt/data/returns.csv")

# Function to summarize a DataFrame
def summarize_dataframe(df, description_map):
    summary = []
    for col in df.columns:
        dtype = df[col].dtype
        missing = df[col].isnull().sum()
        unique = df[col].nunique()
        desc = description_map.get(col, "No description provided.")

        col_summary = {
            "Column": col,
            "Data Type": dtype,
            "Description": desc,
            "Missing Values": missing,
            "Unique Values": unique
        }

        if pd.api.types.is_numeric_dtype(df[col]):
            col_summary.update({
                "Min": df[col].min(),
                "Max": df[col].max(),
                "Mean": df[col].mean(),
                "Median": df[col].median(),
                "Std Dev": df[col].std()
            })
        else:
            col_summary.update({
                "Min": None, "Max": None, "Mean": None, "Median": None, "Std Dev": None
            })

        summary.append(col_summary)
    return pd.DataFrame(summary)

# Column descriptions
customer_descriptions = {
    "Customer ID": "Unique identifier for each customer.",
    "Segment": "Customer segment (e.g., Consumer, Corporate, Home Office).",
    "Country": "Customer's country.",
    "City": "Customer's city.",
    "State": "Customer's state.",
    "Postal Code": "Customer's postal code.",
    "Region": "Geographical region."
}

orders_descriptions = {
    "Row ID": "Unique row number.",
    "Order ID": "Unique identifier for each order.",
    "Order Date": "Date the order was placed.",
    "Ship Date": "Date the order was shipped.",
    "Ship Mode": "Shipping method chosen.",
    "Customer ID": "Unique identifier for each customer.",
    "Product ID": "Unique identifier for each product.",
    "Category": "Product category.",
    "Sub-Category": "Product sub-category.",
    "Product Name": "Name of the product.",
    "Sales": "Revenue from the sale.",
    "Quantity": "Number of units sold.",
    "Discount": "Discount applied to the sale.",
    "Profit": "Profit from the sale."
}

returns_descriptions = {
    "Returned": "Indicates if the order was returned.",
    "Order ID": "Unique identifier for each order."
}

# Create summaries
customers_summary = summarize_dataframe(customers_df, customer_descriptions)
orders_summary = summarize_dataframe(orders_df, orders_descriptions)
returns_summary = summarize_dataframe(returns_df, returns_descriptions)

# Convert date columns in orders_df
orders_df["Order Date"] = pd.to_datetime(orders_df["Order Date"], errors='coerce')
orders_df["Ship Date"] = pd.to_datetime(orders_df["Ship Date"], errors='coerce')

# Convert to categorical
categorical_candidates = [
    "Segment", "Country", "City", "State", "Region",
    "Ship Mode", "Category", "Sub-Category", "Product Name", "Returned"
]

for col in customers_df.columns:
    if customers_df[col].dtype == 'object' and col in categorical_candidates:
        customers_df[col] = customers_df[col].astype('category')

for col in orders_df.columns:
    if orders_df[col].dtype == 'object' and col in categorical_candidates:
        orders_df[col] = orders_df[col].astype('category')

for col in returns_df.columns:
    if returns_df[col].dtype == 'object' and col in categorical_candidates:
        returns_df[col] = returns_df[col].astype('category')

# Combine the first 5 and last 5 rows for display
customers_sample = pd.concat([customers_df.head(5), customers_df.tail(5)])
orders_sample = pd.concat([orders_df.head(5), orders_df.tail(5)])
returns_sample = pd.concat([returns_df.head(5), returns_df.tail(5)])

# Merge orders with customers
merged_df = orders_df.merge(customers_df, on="Customer ID", how="left", indicator=True)

# Unmatched check and counts
unmatched_customers = merged_df[merged_df['_merge'] == 'left_only'].shape[0]
unique_customers = merged_df["Customer ID"].nunique()
unique_orders = merged_df["Order ID"].nunique()