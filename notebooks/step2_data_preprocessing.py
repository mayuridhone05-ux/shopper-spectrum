"""
================================================================================
SHOPPER SPECTRUM - STEP 2: DATA PREPROCESSING
Customer Segmentation and Product Recommendation in E-Commerce
================================================================================

This script performs complete data cleaning on the Online Retail dataset.
Run this in Jupyter Notebook or as a .py file.

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
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', 50)

print("=" * 70)
print("🔧 STEP 2: DATA PREPROCESSING")
print("=" * 70)

# ============================================================
# 2.1 LOAD DATASET
# ============================================================
print("\n📥 Loading dataset...")
# Adjust path as needed for your system
df = pd.read_csv('data/online_retail.csv', encoding='latin1')
print(f"✅ Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")

# ============================================================
# 2.2 INITIAL EXPLORATION
# ============================================================
print("\n🔍 Initial data check:")
print(f"   Missing CustomerID: {df['CustomerID'].isnull().sum():,}")
print(f"   Cancelled invoices: {df['InvoiceNo'].astype(str).str.startswith('C').sum():,}")
print(f"   Negative quantities: {(df['Quantity'] <= 0).sum():,}")
print(f"   Negative/zero prices: {(df['UnitPrice'] <= 0).sum():,}")
print(f"   Blank descriptions: {df['Description'].isnull().sum():,}")
print(f"   Duplicate rows: {df.duplicated().sum():,}")

# ============================================================
# 2.3 DATA CLEANING PIPELINE
# ============================================================
original_shape = df.shape
print(f"\n📊 Original: {original_shape[0]:,} rows")

df_clean = df.copy()

# 2.3.1 Parse dates
df_clean['InvoiceDate'] = pd.to_datetime(df_clean['InvoiceDate'], format='%d-%m-%Y %H:%M')

# 2.3.2 Remove missing CustomerID
df_clean = df_clean.dropna(subset=['CustomerID'])

# 2.3.3 Remove cancelled invoices
df_clean = df_clean[~df_clean['InvoiceNo'].astype(str).str.startswith('C')]

# 2.3.4 Remove negative/zero quantities
df_clean = df_clean[df_clean['Quantity'] > 0]

# 2.3.5 Remove negative/zero prices
df_clean = df_clean[df_clean['UnitPrice'] > 0]

# 2.3.6 Remove non-product StockCodes
non_product_codes = ['POST', 'DOT', 'D', 'C2', 'M', 'BANK CHARGES', 'CRUK', 'PADS']
df_clean = df_clean[~df_clean['StockCode'].isin(non_product_codes)]

# 2.3.7 Remove blank descriptions
df_clean = df_clean.dropna(subset=['Description'])
df_clean = df_clean[df_clean['Description'].str.strip() != '']

# 2.3.8 Remove duplicates
df_clean = df_clean.drop_duplicates()

# 2.3.9 Convert CustomerID to integer
df_clean['CustomerID'] = df_clean['CustomerID'].astype(int)

# 2.3.10 Create TotalAmount
df_clean['TotalAmount'] = df_clean['Quantity'] * df_clean['UnitPrice']

# ============================================================
# 2.4 CLEANING SUMMARY
# ============================================================
print("\n" + "=" * 70)
print("📊 CLEANING SUMMARY")
print("=" * 70)
print(f"   Original rows:      {original_shape[0]:>10,}")
print(f"   Final rows:           {len(df_clean):>10,}")
print(f"   Rows removed:         {original_shape[0] - len(df_clean):>10,}")
print(f"   Retention rate:       {len(df_clean)/original_shape[0]*100:>9.2f}%")

# ============================================================
# 2.5 VERIFY CLEAN DATA
# ============================================================
print("\n✅ Verification:")
print(f"   Missing CustomerID: {df_clean['CustomerID'].isnull().sum()}")
print(f"   Cancelled invoices: {df_clean['InvoiceNo'].astype(str).str.startswith('C').sum()}")
print(f"   Negative quantities: {(df_clean['Quantity'] <= 0).sum()}")
print(f"   Negative prices: {(df_clean['UnitPrice'] <= 0).sum()}")
print(f"   Unique customers: {df_clean['CustomerID'].nunique():,}")
print(f"   Unique products: {df_clean['StockCode'].nunique():,}")
print(f"   Date range: {df_clean['InvoiceDate'].min()} to {df_clean['InvoiceDate'].max()}")

# ============================================================
# 2.6 SAVE CLEANED DATA
# ============================================================
df_clean.to_csv('data/online_retail_cleaned.csv', index=False)
print("\n💾 Cleaned data saved to: data/online_retail_cleaned.csv")
print("\n🎉 STEP 2 COMPLETE! Ready for Step 3: EDA")
