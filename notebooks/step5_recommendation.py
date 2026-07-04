"""
================================================================================
SHOPPER SPECTRUM - STEP 5: PRODUCT RECOMMENDATION SYSTEM
Item-Based Collaborative Filtering using Cosine Similarity
================================================================================

This script builds an item-based collaborative filtering recommender:
  1. Create CustomerID × Description purchase matrix
  2. Compute item-item cosine similarity
  3. Build recommendation function (top 5 similar products)
  4. Test with sample products
  5. Save similarity matrix and product list for Streamlit

Run this after Step 4 (RFM & Segmentation) is complete.
"""

# ============================================================
# IMPORTS
# ============================================================
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

print("=" * 70)
print("🛒 STEP 5: PRODUCT RECOMMENDATION SYSTEM")
print("=" * 70)

# ============================================================
# 5.1 LOAD CLEANED DATA
# ============================================================
print("\n📥 Loading cleaned dataset...")
df = pd.read_csv('data/online_retail_cleaned.csv')
print(f"✅ Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")

import os
os.makedirs('models', exist_ok=True)
os.makedirs('outputs', exist_ok=True)

# ============================================================
# 5.2 CREATE USER-ITEM MATRIX
# ============================================================
print("\n" + "=" * 70)
print("📊 5.2 BUILDING USER-ITEM MATRIX")
print("=" * 70)

# Pivot: Customers (rows) × Products (columns), values = Quantity purchased
print("   Creating CustomerID × Description matrix...")
user_item_matrix = df.pivot_table(
    index='CustomerID',
    columns='Description',
    values='Quantity',
    aggfunc='sum',
    fill_value=0
)

print(f"   Matrix shape: {user_item_matrix.shape}")
print(f"   Customers: {user_item_matrix.shape[0]:,}")
print(f"   Products:  {user_item_matrix.shape[1]:,}")
print(f"   Sparsity:  {(user_item_matrix == 0).sum().sum() / (user_item_matrix.shape[0] * user_item_matrix.shape[1]) * 100:.2f}%")

# ============================================================
# 5.3 COMPUTE ITEM-ITEM COSINE SIMILARITY
# ============================================================
print("\n" + "=" * 70)
print("🔗 5.3 COMPUTING ITEM-ITEM COSINE SIMILARITY")
print("=" * 70)

# Transpose so products are rows, then compute cosine similarity
print("   Calculating cosine similarity between products...")
item_similarity = cosine_similarity(user_item_matrix.T)

# Create DataFrame with product names as index and columns
similarity_df = pd.DataFrame(
    item_similarity,
    index=user_item_matrix.columns,
    columns=user_item_matrix.columns
)

print(f"   Similarity matrix shape: {similarity_df.shape}")
print("   ✅ Cosine similarity computed successfully")

# ============================================================
# 5.4 BUILD RECOMMENDATION FUNCTION
# ============================================================
print("\n" + "=" * 70)
print("🔧 5.4 BUILDING RECOMMENDATION FUNCTION")
print("=" * 70)

def get_recommendations(product_name, top_n=5):
    """
    Recommend top N similar products based on cosine similarity.

    Parameters:
    -----------
    product_name : str
        Exact or partial product description
    top_n : int
        Number of recommendations to return (default: 5)

    Returns:
    --------
    list of tuples: [(product_name, similarity_score), ...]
    """
    # Try exact match first
    if product_name in similarity_df.index:
        matched_product = product_name
    else:
        # Try partial match (case-insensitive contains)
        matches = similarity_df.index[
            similarity_df.index.str.contains(product_name, case=False, na=False)
        ]
        if len(matches) == 0:
            return None  # No match found
        matched_product = matches[0]
        print(f"   ⚠️  Fuzzy match: '{product_name}' → '{matched_product}'")

    # Get similarity scores for the matched product, exclude self
    sim_scores = similarity_df[matched_product].drop(matched_product)

    # Sort descending and get top N
    top_products = sim_scores.sort_values(ascending=False).head(top_n)

    return [(prod, round(score, 4)) for prod, score in top_products.items()]

print("   ✅ Recommendation function defined")

# ============================================================
# 5.5 TEST WITH SAMPLE PRODUCTS
# ============================================================
print("\n" + "=" * 70)
print("🧪 5.5 TESTING RECOMMENDATIONS")
print("=" * 70)

# Get a few popular products to test with
popular_products = df.groupby('Description')['Quantity'].sum().sort_values(ascending=False).head(10)
print("\n📊 Top 10 most popular products (by quantity):")
for i, (prod, qty) in enumerate(popular_products.items(), 1):
    print(f"   {i}. {prod[:50]}{'...' if len(prod) > 50 else ''} ({qty:,})")

# Test recommendations for 3 sample products
test_products = popular_products.index[:3].tolist()

for prod in test_products:
    print(f"\n🎯 Testing: '{prod[:60]}{'...' if len(prod) > 60 else ''}'")
    recs = get_recommendations(prod, top_n=5)
    if recs:
        for i, (rec_prod, score) in enumerate(recs, 1):
            print(f"   {i}. {rec_prod[:60]}{'...' if len(rec_prod) > 60 else ''} (score: {score})")
    else:
        print("   ❌ No recommendations found")

# ============================================================
# 5.6 SAVE MODELS FOR STREAMLIT
# ============================================================
print("\n" + "=" * 70)
print("💾 5.6 SAVING MODELS FOR STREAMLIT APP")
print("=" * 70)

# Save similarity matrix
joblib.dump(similarity_df, 'models/product_similarity.pkl')
print("💾 Saved: models/product_similarity.pkl")

# Save product list (sorted alphabetically for dropdown/search)
product_list = sorted(similarity_df.index.tolist())
joblib.dump(product_list, 'models/product_names.pkl')
print(f"💾 Saved: models/product_names.pkl ({len(product_list):,} products)")

# Also save as a simple text file for reference
with open('outputs/product_list.txt', 'w', encoding='utf-8') as f:
    for p in product_list:
        f.write(p + "\n")
print("💾 Saved: outputs/product_list.txt")

# ============================================================
# 5.7 SIMILARITY MATRIX HEATMAP (Sample)
# ============================================================
print("\n" + "=" * 70)
print("🔥 5.7 GENERATING SIMILARITY HEATMAP (Top 20 Products)")
print("=" * 70)

import matplotlib.pyplot as plt
import seaborn as sns

# Pick top 20 products by total quantity for visualization
top20_products = popular_products.head(20).index.tolist()
sim_subset = similarity_df.loc[top20_products, top20_products]

fig, ax = plt.subplots(figsize=(14, 12))
sns.heatmap(sim_subset, cmap='coolwarm', center=0, square=True,
            linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax)
ax.set_title('Product Similarity Matrix (Top 20 Products)', fontsize=14, fontweight='bold')
# Truncate labels for readability
labels = [p[:25] + '...' if len(p) > 25 else p for p in top20_products]
ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
ax.set_yticklabels(labels, rotation=0, fontsize=8)
plt.tight_layout()
plt.savefig('outputs/eda_charts/19_product_similarity_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Saved: outputs/eda_charts/19_product_similarity_heatmap.png")

# ============================================================
# 5.8 FINAL SUMMARY
# ============================================================
print("\n" + "=" * 70)
print("🎉 STEP 5 COMPLETE!")
print("=" * 70)
print("\n📁 Files generated:")
print("   models/product_similarity.pkl")
print("   models/product_names.pkl")
print("   outputs/product_list.txt")
print("   outputs/eda_charts/19_product_similarity_heatmap.png")
print("\n🚀 Ready for Step 6: Streamlit Application")
