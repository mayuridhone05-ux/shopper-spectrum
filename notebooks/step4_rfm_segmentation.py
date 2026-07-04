"""
================================================================================
SHOPPER SPECTRUM - STEP 4: RFM ANALYSIS & CUSTOMER SEGMENTATION
Customer Segmentation and Product Recommendation in E-Commerce
================================================================================

This script performs:
  1. RFM (Recency, Frequency, Monetary) feature engineering
  2. RFM distribution analysis and correlation heatmap
  3. Log-transformation and standardization
  4. Elbow method + Silhouette score for optimal k
  5. KMeans clustering (k=4)
  6. Auto-labeling clusters: High-Value, Regular, Occasional, At-Risk
  7. 3D cluster visualization
  8. Model persistence for Streamlit app

Run this after Step 3 (EDA) is complete.
"""

# ============================================================
# IMPORTS
# ============================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import joblib
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
pd.set_option('display.max_columns', None)

print("=" * 70)
print("🎯 STEP 4: RFM ANALYSIS & CUSTOMER SEGMENTATION")
print("=" * 70)

# ============================================================
# 4.1 LOAD CLEANED DATA
# ============================================================
print("\n📥 Loading cleaned dataset...")
df = pd.read_csv('data/online_retail_cleaned.csv')
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
print(f"✅ Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")

import os
os.makedirs('outputs/eda_charts', exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('outputs', exist_ok=True)

# ============================================================
# 4.2 CALCULATE RFM VALUES
# ============================================================
print("\n" + "=" * 70)
print("📊 4.2 CALCULATING RFM VALUES")
print("=" * 70)

# Reference date = day after the last transaction
reference_date = df['InvoiceDate'].max() + timedelta(days=1)
print(f"   Reference date: {reference_date.date()}")

# Recency: days since last purchase
# Frequency: number of unique invoices (transactions)
# Monetary: total amount spent
rfm = df.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (reference_date - x.max()).days,
    'InvoiceNo': 'nunique',
    'TotalAmount': 'sum'
}).round(2)

rfm.columns = ['Recency', 'Frequency', 'Monetary']
rfm = rfm.reset_index()

print(f"\n📋 RFM Table Shape: {rfm.shape}")
print("\n📊 RFM Descriptive Statistics:")
print(rfm[['Recency', 'Frequency', 'Monetary']].describe())

# Save RFM table
rfm.to_csv('outputs/rfm_table.csv', index=False)
print("\n💾 Saved: outputs/rfm_table.csv")

# ============================================================
# 4.3 RFM DISTRIBUTIONS
# ============================================================
print("\n" + "=" * 70)
print("📈 4.3 RFM DISTRIBUTIONS")
print("=" * 70)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Recency
axes[0].hist(rfm['Recency'], bins=50, color='steelblue', edgecolor='black', alpha=0.7)
axes[0].axvline(rfm['Recency'].mean(), color='red', linestyle='--', label=f"Mean: {rfm['Recency'].mean():.1f}")
axes[0].axvline(rfm['Recency'].median(), color='green', linestyle='--', label=f"Median: {rfm['Recency'].median():.1f}")
axes[0].set_xlabel('Recency (days)', fontsize=11)
axes[0].set_ylabel('Number of Customers', fontsize=11)
axes[0].set_title('Recency Distribution', fontsize=12, fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Frequency
axes[1].hist(rfm['Frequency'], bins=50, color='coral', edgecolor='black', alpha=0.7)
axes[1].axvline(rfm['Frequency'].mean(), color='red', linestyle='--', label=f"Mean: {rfm['Frequency'].mean():.1f}")
axes[1].axvline(rfm['Frequency'].median(), color='green', linestyle='--', label=f"Median: {rfm['Frequency'].median():.1f}")
axes[1].set_xlabel('Frequency (transactions)', fontsize=11)
axes[1].set_ylabel('Number of Customers', fontsize=11)
axes[1].set_title('Frequency Distribution', fontsize=12, fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Monetary
axes[2].hist(rfm['Monetary'], bins=50, color='mediumseagreen', edgecolor='black', alpha=0.7)
axes[2].axvline(rfm['Monetary'].mean(), color='red', linestyle='--', label=f"Mean: £{rfm['Monetary'].mean():.2f}")
axes[2].axvline(rfm['Monetary'].median(), color='green', linestyle='--', label=f"Median: £{rfm['Monetary'].median():.2f}")
axes[2].set_xlabel('Monetary (£)', fontsize=11)
axes[2].set_ylabel('Number of Customers', fontsize=11)
axes[2].set_title('Monetary Distribution', fontsize=12, fontweight='bold')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/eda_charts/13_rfm_distributions.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Saved: outputs/eda_charts/13_rfm_distributions.png")

# ============================================================
# 4.4 RFM CORRELATION HEATMAP
# ============================================================
print("\n" + "=" * 70)
print("🔥 4.4 RFM CORRELATION HEATMAP")
print("=" * 70)

fig, ax = plt.subplots(figsize=(8, 6))
corr = rfm[['Recency', 'Frequency', 'Monetary']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, fmt='.3f',
            square=True, linewidths=1, ax=ax)
ax.set_title('RFM Correlation Heatmap', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/eda_charts/14_rfm_correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Saved: outputs/eda_charts/14_rfm_correlation_heatmap.png")

# ============================================================
# 4.5 LOG TRANSFORM SKEWED FEATURES
# ============================================================
print("\n" + "=" * 70)
print("🔄 4.5 LOG TRANSFORMATION")
print("=" * 70)

rfm_log = rfm.copy()
rfm_log['Frequency_log'] = np.log1p(rfm_log['Frequency'])
rfm_log['Monetary_log'] = np.log1p(rfm_log['Monetary'])
# Recency is usually less skewed; keep as-is or optionally log-transform
# We keep Recency original for better interpretability

print("   Applied np.log1p() to Frequency and Monetary")
print("\n📊 Log-transformed stats:")
print(rfm_log[['Recency', 'Frequency_log', 'Monetary_log']].describe())

# ============================================================
# 4.6 STANDARDIZE RFM
# ============================================================
print("\n" + "=" * 70)
print("⚖️  4.6 STANDARDIZATION")
print("=" * 70)

features_for_clustering = ['Recency', 'Frequency_log', 'Monetary_log']
X = rfm_log[features_for_clustering].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"   Features used: {features_for_clustering}")
print(f"   Scaled shape: {X_scaled.shape}")
print(f"   Mean: {X_scaled.mean(axis=0).round(4)}")
print(f"   Std:  {X_scaled.std(axis=0).round(4)}")

# Save scaler for Streamlit
joblib.dump(scaler, 'models/scaler.pkl')
print("💾 Saved: models/scaler.pkl")

# ============================================================
# 4.7 ELBOW METHOD
# ============================================================
print("\n" + "=" * 70)
print("🦴 4.7 ELBOW METHOD")
print("=" * 70)

inertias = []
silhouettes = []
K_range = range(2, 11)

for k in K_range:
    kmeans_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans_temp.fit(X_scaled)
    inertias.append(kmeans_temp.inertia_)
    silhouettes.append(silhouette_score(X_scaled, kmeans_temp.labels_))
    print(f"   k={k}: Inertia={kmeans_temp.inertia_:.2f}, Silhouette={silhouette_score(X_scaled, kmeans_temp.labels_):.4f}")

# --- Chart: Elbow Curve ---
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(K_range, inertias, marker='o', linewidth=2.5, markersize=8, color='navy')
ax.set_xlabel('Number of Clusters (k)', fontsize=12)
ax.set_ylabel('Inertia (WCSS)', fontsize=12)
ax.set_title('Elbow Method for Optimal k', fontsize=14, fontweight='bold')
ax.set_xticks(list(K_range))
ax.grid(True, alpha=0.3)
# Annotate k=4
ax.axvline(x=4, color='red', linestyle='--', alpha=0.7, label='k=4 (selected)')
ax.legend()
plt.tight_layout()
plt.savefig('outputs/eda_charts/15_elbow_curve.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Saved: outputs/eda_charts/15_elbow_curve.png")

# ============================================================
# 4.8 SILHOUETTE SCORE
# ============================================================
print("\n" + "=" * 70)
print("👤 4.8 SILHOUETTE SCORE ANALYSIS")
print("=" * 70)

# --- Chart: Silhouette Scores ---
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(K_range, silhouettes, marker='s', linewidth=2.5, markersize=8, color='darkgreen')
ax.set_xlabel('Number of Clusters (k)', fontsize=12)
ax.set_ylabel('Silhouette Score', fontsize=12)
ax.set_title('Silhouette Score vs Number of Clusters', fontsize=14, fontweight='bold')
ax.set_xticks(list(K_range))
ax.grid(True, alpha=0.3)
ax.axvline(x=4, color='red', linestyle='--', alpha=0.7, label='k=4 (selected)')
ax.legend()
plt.tight_layout()
plt.savefig('outputs/eda_charts/16_silhouette_scores.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Saved: outputs/eda_charts/16_silhouette_scores.png")

best_k = list(K_range)[np.argmax(silhouettes)]
print(f"\n📌 Best silhouette score at k={best_k} ({max(silhouettes):.4f})")
print(f"📌 Proceeding with k=4 as per project specification")

# ============================================================
# 4.9 RUN KMEANS CLUSTERING
# ============================================================
print("\n" + "=" * 70)
print("🤖 4.9 KMEANS CLUSTERING (k=4)")
print("=" * 70)

kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
rfm_log['Cluster'] = kmeans.fit_predict(X_scaled)
rfm['Cluster'] = rfm_log['Cluster']

print(f"   Cluster distribution:")
print(rfm['Cluster'].value_counts().sort_index())

# Save model for Streamlit
joblib.dump(kmeans, 'models/kmeans_model.pkl')
print("\n💾 Saved: models/kmeans_model.pkl")

# ============================================================
# 4.10 CLUSTER PROFILE ANALYSIS
# ============================================================
print("\n" + "=" * 70)
print("📋 4.10 CLUSTER PROFILE ANALYSIS")
print("=" * 70)

cluster_profile = rfm.groupby('Cluster').agg({
    'Recency': ['mean', 'median'],
    'Frequency': ['mean', 'median'],
    'Monetary': ['mean', 'median'],
    'CustomerID': 'count'
}).round(2)

cluster_profile.columns = ['Recency_Mean', 'Recency_Median', 'Frequency_Mean', 'Frequency_Median',
                           'Monetary_Mean', 'Monetary_Median', 'Count']
cluster_profile = cluster_profile.reset_index()
print("\n📊 Cluster Profiles (Raw RFM):")
print(cluster_profile.to_string(index=False))

# ============================================================
# 4.11 AUTO-LABEL CLUSTERS
# ============================================================
print("\n" + "=" * 70)
print("🏷️  4.11 AUTO-LABELING CLUSTERS")
print("=" * 70)

# Compute normalized scores for ranking (0-1 scale within each metric)
profile = rfm.groupby('Cluster').agg({
    'Recency': 'mean',
    'Frequency': 'mean',
    'Monetary': 'mean',
    'CustomerID': 'count'
}).reset_index()

# For Recency: lower is better → invert
profile['R_score'] = 1 - (profile['Recency'] - profile['Recency'].min()) / (profile['Recency'].max() - profile['Recency'].min() + 1e-9)
# For Frequency: higher is better
profile['F_score'] = (profile['Frequency'] - profile['Frequency'].min()) / (profile['Frequency'].max() - profile['Frequency'].min() + 1e-9)
# For Monetary: higher is better
profile['M_score'] = (profile['Monetary'] - profile['Monetary'].min()) / (profile['Monetary'].max() - profile['Monetary'].min() + 1e-9)

# Composite score
profile['Composite'] = profile['R_score'] + profile['F_score'] + profile['M_score']

# Sort by composite score descending
profile_sorted = profile.sort_values('Composite', ascending=False)
print("\n📊 Cluster Ranking (by composite RFM score):")
print(profile_sorted[['Cluster', 'Recency', 'Frequency', 'Monetary', 'Composite']].to_string(index=False))

# Assign labels based on ranking and characteristics
labels_map = {}
clusters = profile_sorted['Cluster'].tolist()

# Highest composite = High-Value
labels_map[clusters[0]] = 'High-Value'
# Lowest composite = At-Risk
labels_map[clusters[3]] = 'At-Risk'

# Between High-Value and At-Risk: distinguish Regular vs Occasional
# Regular has higher F and M than Occasional
mid1 = clusters[1]
mid2 = clusters[2]

if profile[profile['Cluster'] == mid1]['Frequency'].values[0] > profile[profile['Cluster'] == mid2]['Frequency'].values[0]:
    labels_map[mid1] = 'Regular'
    labels_map[mid2] = 'Occasional'
else:
    labels_map[mid1] = 'Occasional'
    labels_map[mid2] = 'Regular'

rfm['Segment'] = rfm['Cluster'].map(labels_map)
rfm_log['Segment'] = rfm_log['Cluster'].map(labels_map)

print("\n🏷️  Final Segment Mapping:")
for cid, label in sorted(labels_map.items()):
    count = rfm[rfm['Cluster'] == cid].shape[0]
    print(f"   Cluster {cid} → {label:12s} ({count:,} customers)")

# ============================================================
# 4.12 CLUSTER PROFILE CHART
# ============================================================
print("\n" + "=" * 70)
print("📊 4.12 CLUSTER PROFILE VISUALIZATION")
print("=" * 70)

segment_order = ['High-Value', 'Regular', 'Occasional', 'At-Risk']
segment_colors = {'High-Value': '#2ecc71', 'Regular': '#3498db', 'Occasional': '#f39c12', 'At-Risk': '#e74c3c'}

# Mean RFM by segment
segment_means = rfm.groupby('Segment')[['Recency', 'Frequency', 'Monetary']].mean().reindex(segment_order)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Recency
segment_means['Recency'].plot(kind='bar', ax=axes[0], color=[segment_colors[s] for s in segment_order], rot=0)
axes[0].set_title('Average Recency by Segment', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Days Since Last Purchase', fontsize=11)
axes[0].grid(True, alpha=0.3)

# Frequency
segment_means['Frequency'].plot(kind='bar', ax=axes[1], color=[segment_colors[s] for s in segment_order], rot=0)
axes[1].set_title('Average Frequency by Segment', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Number of Transactions', fontsize=11)
axes[1].grid(True, alpha=0.3)

# Monetary
segment_means['Monetary'].plot(kind='bar', ax=axes[2], color=[segment_colors[s] for s in segment_order], rot=0)
axes[2].set_title('Average Monetary by Segment', fontsize=12, fontweight='bold')
axes[2].set_ylabel('Total Spend (£)', fontsize=11)
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/eda_charts/17_cluster_profiles.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Saved: outputs/eda_charts/17_cluster_profiles.png")

# ============================================================
# 4.13 3D CLUSTER VISUALIZATION
# ============================================================
print("\n" + "=" * 70)
print("🎲 4.13 3D CLUSTER SCATTER PLOT")
print("=" * 70)

fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')

for segment in segment_order:
    subset = rfm[rfm['Segment'] == segment]
    ax.scatter(subset['Recency'], subset['Frequency'], subset['Monetary'],
               c=segment_colors[segment], label=segment, s=30, alpha=0.6, edgecolors='k', linewidth=0.3)

ax.set_xlabel('Recency (days)', fontsize=11)
ax.set_ylabel('Frequency (orders)', fontsize=11)
ax.set_zlabel('Monetary (£)', fontsize=11)
ax.set_title('3D Customer Segments (RFM)', fontsize=14, fontweight='bold')
ax.legend(title='Segment', loc='upper left')
plt.tight_layout()
plt.savefig('outputs/eda_charts/18_3d_clusters.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Saved: outputs/eda_charts/18_3d_clusters.png")

# ============================================================
# 4.14 SAVE FINAL OUTPUTS
# ============================================================
print("\n" + "=" * 70)
print("💾 4.14 SAVING FINAL OUTPUTS")
print("=" * 70)

# Save RFM with segments
rfm.to_csv('outputs/rfm_table.csv', index=False)
print("💾 Saved: outputs/rfm_table.csv")

# Save cluster profiles
cluster_profiles = rfm.groupby('Segment').agg({
    'Recency': ['mean', 'min', 'max'],
    'Frequency': ['mean', 'min', 'max'],
    'Monetary': ['mean', 'min', 'max'],
    'CustomerID': 'count'
}).round(2)
cluster_profiles.columns = ['R_Mean', 'R_Min', 'R_Max', 'F_Mean', 'F_Min', 'F_Max',
                            'M_Mean', 'M_Min', 'M_Max', 'Count']
cluster_profiles = cluster_profiles.reset_index()
cluster_profiles.to_csv('outputs/cluster_profiles.csv', index=False)
print("💾 Saved: outputs/cluster_profiles.csv")

# Save segment mapping for Streamlit
segment_mapping = pd.DataFrame([
    {'Cluster': k, 'Segment': v} for k, v in labels_map.items()
])
segment_mapping.to_csv('outputs/segment_mapping.csv', index=False)
print("💾 Saved: outputs/segment_mapping.csv")

# ============================================================
# 4.15 FINAL SUMMARY
# ============================================================
print("\n" + "=" * 70)
print("🎉 STEP 4 COMPLETE!")
print("=" * 70)
print("\n📁 Files generated:")
print("   outputs/eda_charts/13_rfm_distributions.png")
print("   outputs/eda_charts/14_rfm_correlation_heatmap.png")
print("   outputs/eda_charts/15_elbow_curve.png")
print("   outputs/eda_charts/16_silhouette_scores.png")
print("   outputs/eda_charts/17_cluster_profiles.png")
print("   outputs/eda_charts/18_3d_clusters.png")
print("   outputs/rfm_table.csv")
print("   outputs/cluster_profiles.csv")
print("   outputs/segment_mapping.csv")
print("   models/kmeans_model.pkl")
print("   models/scaler.pkl")
print("\n🚀 Ready for Step 5: Product Recommendation System")
