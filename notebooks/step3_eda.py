"""
================================================================================
SHOPPER SPECTRUM - STEP 3: EXPLORATORY DATA ANALYSIS (EDA)
Customer Segmentation and Product Recommendation in E-Commerce
================================================================================

Run this script in Jupyter Notebook or as a .py file after completing Step 2.
All visualizations will be saved to outputs/eda_charts/ folder.

Author: Your Name
Date: 2026-07-04
"""

# ============================================================
# IMPORTS
# ============================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set style for better-looking charts
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

print("=" * 70)
print("📊 STEP 3: EXPLORATORY DATA ANALYSIS (EDA)")
print("=" * 70)

# ============================================================
# 3.1 LOAD CLEANED DATA
# ============================================================
print("\n📥 Loading cleaned dataset...")
df = pd.read_csv('data/online_retail_cleaned.csv')
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
print(f"✅ Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")

# Create output directory for charts
import os
os.makedirs('outputs/eda_charts', exist_ok=True)

# ============================================================
# 3.2 BASIC DATA OVERVIEW
# ============================================================
print("\n" + "=" * 70)
print("📋 3.2 BASIC DATA OVERVIEW")
print("=" * 70)

print("\n📌 Dataset Shape:", df.shape)
print("\n📌 Column Names:", df.columns.tolist())
print("\n📌 Data Types:")
print(df.dtypes)
print("\n📌 Numeric Summary:")
print(df.describe())
print("\n📌 Date Range:")
print(f"   From: {df['InvoiceDate'].min()}")
print(f"   To:   {df['InvoiceDate'].max()}")
print(f"   Duration: {(df['InvoiceDate'].max() - df['InvoiceDate'].min()).days} days")

# ============================================================
# 3.3 COUNTRY-WISE SALES ANALYSIS
# ============================================================
print("\n" + "=" * 70)
print("🌍 3.3 COUNTRY-WISE SALES ANALYSIS")
print("=" * 70)

# Sales by country (top 15)
country_sales = df.groupby('Country').agg({
    'TotalAmount': 'sum',
    'InvoiceNo': 'nunique',
    'CustomerID': 'nunique'
}).round(2)
country_sales.columns = ['TotalRevenue', 'NumOrders', 'NumCustomers']
country_sales = country_sales.sort_values('TotalRevenue', ascending=False)

print("\n📊 Top 15 Countries by Revenue:")
print(country_sales.head(15))

# Save to CSV
country_sales.to_csv('outputs/country_sales_summary.csv')
print("\n💾 Saved: outputs/country_sales_summary.csv")

# --- Chart 1: Top 10 Countries by Revenue ---
fig, ax = plt.subplots(figsize=(12, 6))
top10_countries = country_sales.head(10)
bars = ax.barh(range(len(top10_countries)), top10_countries['TotalRevenue'], color='steelblue')
ax.set_yticks(range(len(top10_countries)))
ax.set_yticklabels(top10_countries.index)
ax.invert_yaxis()
ax.set_xlabel('Total Revenue (£)', fontsize=12)
ax.set_title('Top 10 Countries by Total Revenue', fontsize=14, fontweight='bold')
ax.grid(axis='x', alpha=0.3)
# Add value labels
for i, bar in enumerate(bars):
    width = bar.get_width()
    ax.text(width + width*0.01, bar.get_y() + bar.get_height()/2, 
            f'£{width:,.0f}', ha='left', va='center', fontsize=9)
plt.tight_layout()
plt.savefig('outputs/eda_charts/01_country_revenue.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Saved: outputs/eda_charts/01_country_revenue.png")

# --- Chart 2: Country Revenue Pie Chart (Top 10 + Others) ---
fig, ax = plt.subplots(figsize=(10, 8))
top10_rev = top10_countries['TotalRevenue']
others_rev = country_sales.iloc[10:]['TotalRevenue'].sum()
labels = list(top10_countries.index) + ['Others']
sizes = list(top10_rev.values) + [others_rev]
colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
wedges, texts, autotexts = ax.pie(sizes, labels=labels, autopct='%1.1f%%', 
                                   colors=colors, startangle=90)
ax.set_title('Revenue Distribution by Country', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/eda_charts/02_country_revenue_pie.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Saved: outputs/eda_charts/02_country_revenue_pie.png")

# ============================================================
# 3.4 TOP SELLING PRODUCTS
# ============================================================
print("\n" + "=" * 70)
print("🏆 3.4 TOP SELLING PRODUCTS")
print("=" * 70)

# Top products by quantity sold
product_qty = df.groupby('Description').agg({
    'Quantity': 'sum',
    'TotalAmount': 'sum',
    'InvoiceNo': 'nunique'
}).round(2)
product_qty.columns = ['TotalQuantity', 'TotalRevenue', 'NumOrders']
product_qty = product_qty.sort_values('TotalQuantity', ascending=False)

print("\n📊 Top 15 Products by Quantity Sold:")
print(product_qty.head(15))

# Save to CSV
product_qty.to_csv('outputs/product_summary.csv')
print("\n💾 Saved: outputs/product_summary.csv")

# --- Chart 3: Top 10 Products by Quantity ---
fig, ax = plt.subplots(figsize=(12, 7))
top10_qty = product_qty.head(10)
bars = ax.barh(range(len(top10_qty)), top10_qty['TotalQuantity'], color='coral')
ax.set_yticks(range(len(top10_qty)))
ax.set_yticklabels([desc[:40] + '...' if len(desc) > 40 else desc for desc in top10_qty.index])
ax.invert_yaxis()
ax.set_xlabel('Total Quantity Sold', fontsize=12)
ax.set_title('Top 10 Products by Quantity Sold', fontsize=14, fontweight='bold')
ax.grid(axis='x', alpha=0.3)
for i, bar in enumerate(bars):
    width = bar.get_width()
    ax.text(width + width*0.01, bar.get_y() + bar.get_height()/2, 
            f'{int(width):,}', ha='left', va='center', fontsize=9)
plt.tight_layout()
plt.savefig('outputs/eda_charts/03_top_products_quantity.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Saved: outputs/eda_charts/03_top_products_quantity.png")

# --- Chart 4: Top 10 Products by Revenue ---
product_rev = product_qty.sort_values('TotalRevenue', ascending=False)
fig, ax = plt.subplots(figsize=(12, 7))
top10_rev_prod = product_rev.head(10)
bars = ax.barh(range(len(top10_rev_prod)), top10_rev_prod['TotalRevenue'], color='mediumseagreen')
ax.set_yticks(range(len(top10_rev_prod)))
ax.set_yticklabels([desc[:40] + '...' if len(desc) > 40 else desc for desc in top10_rev_prod.index])
ax.invert_yaxis()
ax.set_xlabel('Total Revenue (£)', fontsize=12)
ax.set_title('Top 10 Products by Revenue', fontsize=14, fontweight='bold')
ax.grid(axis='x', alpha=0.3)
for i, bar in enumerate(bars):
    width = bar.get_width()
    ax.text(width + width*0.01, bar.get_y() + bar.get_height()/2, 
            f'£{width:,.0f}', ha='left', va='center', fontsize=9)
plt.tight_layout()
plt.savefig('outputs/eda_charts/04_top_products_revenue.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Saved: outputs/eda_charts/04_top_products_revenue.png")

# ============================================================
# 3.5 MONTHLY SALES TREND
# ============================================================
print("\n" + "=" * 70)
print("📈 3.5 MONTHLY SALES TREND")
print("=" * 70)

# Create year-month column
df['YearMonth'] = df['InvoiceDate'].dt.to_period('M')

# Monthly aggregation
monthly_sales = df.groupby('YearMonth').agg({
    'TotalAmount': 'sum',
    'InvoiceNo': 'nunique',
    'CustomerID': 'nunique',
    'Quantity': 'sum'
}).round(2)
monthly_sales.columns = ['Revenue', 'Orders', 'Customers', 'Quantity']

print("\n📊 Monthly Sales Summary:")
print(monthly_sales)

# Save to CSV
monthly_sales.to_csv('outputs/monthly_sales_summary.csv')
print("\n💾 Saved: outputs/monthly_sales_summary.csv")

# --- Chart 5: Monthly Revenue Trend ---
fig, ax = plt.subplots(figsize=(14, 6))
months = [str(m) for m in monthly_sales.index]
ax.plot(months, monthly_sales['Revenue'], marker='o', linewidth=2.5, markersize=8, color='navy')
ax.fill_between(months, monthly_sales['Revenue'], alpha=0.3, color='lightblue')
ax.set_xlabel('Month', fontsize=12)
ax.set_ylabel('Revenue (£)', fontsize=12)
ax.set_title('Monthly Revenue Trend', fontsize=14, fontweight='bold')
ax.tick_params(axis='x', rotation=45)
ax.grid(True, alpha=0.3)
# Add value labels
for i, v in enumerate(monthly_sales['Revenue']):
    ax.text(i, v + v*0.02, f'£{v:,.0f}', ha='center', va='bottom', fontsize=8, rotation=45)
plt.tight_layout()
plt.savefig('outputs/eda_charts/05_monthly_revenue_trend.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Saved: outputs/eda_charts/05_monthly_revenue_trend.png")

# --- Chart 6: Monthly Orders Trend ---
fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(months, monthly_sales['Orders'], marker='s', linewidth=2.5, markersize=8, color='darkgreen')
ax.fill_between(months, monthly_sales['Orders'], alpha=0.3, color='lightgreen')
ax.set_xlabel('Month', fontsize=12)
ax.set_ylabel('Number of Orders', fontsize=12)
ax.set_title('Monthly Orders Trend', fontsize=14, fontweight='bold')
ax.tick_params(axis='x', rotation=45)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('outputs/eda_charts/06_monthly_orders_trend.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Saved: outputs/eda_charts/06_monthly_orders_trend.png")

# ============================================================
# 3.6 DAILY TRANSACTION PATTERN
# ============================================================
print("\n" + "=" * 70)
print("📅 3.6 DAILY TRANSACTION PATTERN")
print("=" * 70)

# Daily aggregation
df['Date'] = df['InvoiceDate'].dt.date
daily_sales = df.groupby('Date').agg({
    'TotalAmount': 'sum',
    'InvoiceNo': 'nunique'
}).round(2)
daily_sales.columns = ['Revenue', 'Orders']

print(f"\n📊 Daily transactions: {len(daily_sales)} unique days")
print(f"   Avg daily revenue: £{daily_sales['Revenue'].mean():,.2f}")
print(f"   Avg daily orders: {daily_sales['Orders'].mean():.1f}")

# --- Chart 7: Daily Revenue Line Chart ---
fig, ax = plt.subplots(figsize=(16, 6))
dates = pd.to_datetime(daily_sales.index)
ax.plot(dates, daily_sales['Revenue'], linewidth=1, alpha=0.7, color='purple')
ax.set_xlabel('Date', fontsize=12)
ax.set_ylabel('Daily Revenue (£)', fontsize=12)
ax.set_title('Daily Revenue Over Time', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
# Add 7-day rolling average
daily_sales['Revenue_MA7'] = daily_sales['Revenue'].rolling(window=7, min_periods=1).mean()
ax.plot(dates, daily_sales['Revenue_MA7'], linewidth=2.5, color='red', label='7-Day Moving Avg')
ax.legend()
plt.tight_layout()
plt.savefig('outputs/eda_charts/07_daily_revenue.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Saved: outputs/eda_charts/07_daily_revenue.png")

# ============================================================
# 3.7 REVENUE DISTRIBUTION
# ============================================================
print("\n" + "=" * 70)
print("💰 3.7 REVENUE DISTRIBUTION")
print("=" * 70)

# Transaction-level revenue
transaction_revenue = df.groupby('InvoiceNo')['TotalAmount'].sum()
print(f"\n📊 Transaction Revenue Stats:")
print(transaction_revenue.describe())

# --- Chart 8: Transaction Revenue Distribution ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Histogram
axes[0].hist(transaction_revenue, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
axes[0].set_xlabel('Transaction Value (£)', fontsize=11)
axes[0].set_ylabel('Frequency', fontsize=11)
axes[0].set_title('Distribution of Transaction Values', fontsize=12, fontweight='bold')
axes[0].axvline(transaction_revenue.mean(), color='red', linestyle='--', label=f'Mean: £{transaction_revenue.mean():.2f}')
axes[0].axvline(transaction_revenue.median(), color='green', linestyle='--', label=f'Median: £{transaction_revenue.median():.2f}')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Box plot
axes[1].boxplot(transaction_revenue, vert=True, patch_artist=True, 
                boxprops=dict(facecolor='lightcoral', alpha=0.7))
axes[1].set_ylabel('Transaction Value (£)', fontsize=11)
axes[1].set_title('Transaction Value Box Plot', fontsize=12, fontweight='bold')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/eda_charts/08_transaction_revenue_dist.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Saved: outputs/eda_charts/08_transaction_revenue_dist.png")

# ============================================================
# 3.8 CUSTOMER SPENDING DISTRIBUTION
# ============================================================
print("\n" + "=" * 70)
print("👤 3.8 CUSTOMER SPENDING DISTRIBUTION")
print("=" * 70)

customer_spend = df.groupby('CustomerID')['TotalAmount'].sum()
print(f"\n📊 Customer Spending Stats:")
print(customer_spend.describe())

# --- Chart 9: Customer Spending Distribution ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Histogram
axes[0].hist(customer_spend, bins=50, color='gold', edgecolor='black', alpha=0.7)
axes[0].set_xlabel('Total Customer Spend (£)', fontsize=11)
axes[0].set_ylabel('Number of Customers', fontsize=11)
axes[0].set_title('Distribution of Customer Lifetime Value', fontsize=12, fontweight='bold')
axes[0].axvline(customer_spend.mean(), color='red', linestyle='--', label=f'Mean: £{customer_spend.mean():.2f}')
axes[0].axvline(customer_spend.median(), color='green', linestyle='--', label=f'Median: £{customer_spend.median():.2f}')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Box plot
axes[1].boxplot(customer_spend, vert=True, patch_artist=True,
                boxprops=dict(facecolor='plum', alpha=0.7))
axes[1].set_ylabel('Total Customer Spend (£)', fontsize=11)
axes[1].set_title('Customer Spend Box Plot', fontsize=12, fontweight='bold')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/eda_charts/09_customer_spend_dist.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Saved: outputs/eda_charts/09_customer_spend_dist.png")

# ============================================================
# 3.9 MOST ACTIVE CUSTOMERS
# ============================================================
print("\n" + "=" * 70)
print("⭐ 3.9 MOST ACTIVE CUSTOMERS")
print("=" * 70)

customer_stats = df.groupby('CustomerID').agg({
    'TotalAmount': 'sum',
    'InvoiceNo': 'nunique',
    'Quantity': 'sum'
}).round(2)
customer_stats.columns = ['TotalSpend', 'NumOrders', 'TotalQuantity']
customer_stats = customer_stats.sort_values('TotalSpend', ascending=False)

print("\n📊 Top 15 Customers by Total Spend:")
print(customer_stats.head(15))

# Save to CSV
customer_stats.to_csv('outputs/customer_stats_summary.csv')
print("\n💾 Saved: outputs/customer_stats_summary.csv")

# --- Chart 10: Top 10 Customers by Spend ---
fig, ax = plt.subplots(figsize=(12, 6))
top10_cust = customer_stats.head(10)
bars = ax.barh(range(len(top10_cust)), top10_cust['TotalSpend'], color='teal')
ax.set_yticks(range(len(top10_cust)))
ax.set_yticklabels([f'Customer {int(cid)}' for cid in top10_cust.index])
ax.invert_yaxis()
ax.set_xlabel('Total Spend (£)', fontsize=12)
ax.set_title('Top 10 Customers by Total Spend', fontsize=14, fontweight='bold')
ax.grid(axis='x', alpha=0.3)
for i, bar in enumerate(bars):
    width = bar.get_width()
    ax.text(width + width*0.01, bar.get_y() + bar.get_height()/2, 
            f'£{width:,.0f}', ha='left', va='center', fontsize=9)
plt.tight_layout()
plt.savefig('outputs/eda_charts/10_top_customers.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Saved: outputs/eda_charts/10_top_customers.png")

# ============================================================
# 3.10 QUANTITY & PRICE DISTRIBUTIONS
# ============================================================
print("\n" + "=" * 70)
print("📦 3.10 QUANTITY & PRICE DISTRIBUTIONS")
print("=" * 70)

# --- Chart 11: Quantity Distribution ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Quantity histogram (capped at 100 for visibility)
qty_capped = df[df['Quantity'] <= 100]['Quantity']
axes[0].hist(qty_capped, bins=50, color='salmon', edgecolor='black', alpha=0.7)
axes[0].set_xlabel('Quantity per Line Item', fontsize=11)
axes[0].set_ylabel('Frequency', fontsize=11)
axes[0].set_title('Quantity Distribution (≤100)', fontsize=12, fontweight='bold')
axes[0].grid(True, alpha=0.3)

# Unit price histogram (capped at £50 for visibility)
price_capped = df[df['UnitPrice'] <= 50]['UnitPrice']
axes[1].hist(price_capped, bins=50, color='lightgreen', edgecolor='black', alpha=0.7)
axes[1].set_xlabel('Unit Price (£)', fontsize=11)
axes[1].set_ylabel('Frequency', fontsize=11)
axes[1].set_title('Unit Price Distribution (≤£50)', fontsize=12, fontweight='bold')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/eda_charts/11_qty_price_dist.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Saved: outputs/eda_charts/11_qty_price_dist.png")

# ============================================================
# 3.11 DAY OF WEEK & HOUR PATTERNS
# ============================================================
print("\n" + "=" * 70)
print("⏰ 3.11 DAY OF WEEK & HOUR PATTERNS")
print("=" * 70)

df['DayOfWeek'] = df['InvoiceDate'].dt.day_name()
df['Hour'] = df['InvoiceDate'].dt.hour

# Day of week sales
dow_sales = df.groupby('DayOfWeek')['TotalAmount'].sum().reindex([
    'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'
])
print("\n📊 Sales by Day of Week:")
print(dow_sales.round(2))

# Hourly sales
hourly_sales = df.groupby('Hour')['TotalAmount'].sum()
print("\n📊 Sales by Hour (Top 5):")
print(hourly_sales.sort_values(ascending=False).head())

# --- Chart 12: Day of Week Sales ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

dow_sales.plot(kind='bar', ax=axes[0], color='cornflowerblue', rot=45)
axes[0].set_title('Sales by Day of Week', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Total Revenue (£)', fontsize=11)
axes[0].grid(True, alpha=0.3)

hourly_sales.plot(kind='bar', ax=axes[1], color='orange', rot=0)
axes[1].set_title('Sales by Hour of Day', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Total Revenue (£)', fontsize=11)
axes[1].set_xlabel('Hour', fontsize=11)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/eda_charts/12_dow_hour_sales.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Saved: outputs/eda_charts/12_dow_hour_sales.png")

# ============================================================
# 3.12 SUMMARY STATISTICS EXPORT
# ============================================================
print("\n" + "=" * 70)
print("📊 3.12 SUMMARY STATISTICS")
print("=" * 70)

summary = {
    'Metric': [
        'Total Transactions (Rows)',
        'Unique Customers',
        'Unique Products',
        'Unique Invoices',
        'Countries',
        'Date Range (Days)',
        'Total Revenue (£)',
        'Average Order Value (£)',
        'Average Items per Order',
        'Top Country',
        'Top Product (by Qty)',
        'Top Product (by Revenue)',
        'Most Active Customer',
        'Peak Sales Month',
        'Peak Sales Day'
    ],
    'Value': [
        f"{len(df):,}",
        f"{df['CustomerID'].nunique():,}",
        f"{df['StockCode'].nunique():,}",
        f"{df['InvoiceNo'].nunique():,}",
        f"{df['Country'].nunique()}",
        f"{(df['InvoiceDate'].max() - df['InvoiceDate'].min()).days}",
        f"£{df['TotalAmount'].sum():,.2f}",
        f"£{df.groupby('InvoiceNo')['TotalAmount'].sum().mean():.2f}",
        f"{df.groupby('InvoiceNo')['Quantity'].sum().mean():.1f}",
        f"{country_sales.index[0]} (£{country_sales.iloc[0]['TotalRevenue']:,.2f})",
        f"{product_qty.index[0]} ({int(product_qty.iloc[0]['TotalQuantity']):,})",
        f"{product_rev.index[0]} (£{product_rev.iloc[0]['TotalRevenue']:,.2f})",
        f"Customer {int(customer_stats.index[0])} (£{customer_stats.iloc[0]['TotalSpend']:,.2f})",
        f"{monthly_sales['Revenue'].idxmax()} (£{monthly_sales['Revenue'].max():,.2f})",
        f"{dow_sales.idxmax()} (£{dow_sales.max():,.2f})"
    ]
}

summary_df = pd.DataFrame(summary)
print("\n📋 Business Summary:")
print(summary_df.to_string(index=False))

summary_df.to_csv('outputs/eda_summary.csv', index=False)
print("\n💾 Saved: outputs/eda_summary.csv")

# ============================================================
# COMPLETION
# ============================================================
print("\n" + "=" * 70)
print("🎉 STEP 3: EDA COMPLETE!")
print("=" * 70)
print("\n📁 All charts saved to: outputs/eda_charts/")
print("📁 All data summaries saved to: outputs/")
print("\n📊 Charts generated:")
print("   01_country_revenue.png")
print("   02_country_revenue_pie.png")
print("   03_top_products_quantity.png")
print("   04_top_products_revenue.png")
print("   05_monthly_revenue_trend.png")
print("   06_monthly_orders_trend.png")
print("   07_daily_revenue.png")
print("   08_transaction_revenue_dist.png")
print("   09_customer_spend_dist.png")
print("   10_top_customers.png")
print("   11_qty_price_dist.png")
print("   12_dow_hour_sales.png")
print("\n🚀 Ready for Step 4: RFM Analysis & Customer Segmentation")
