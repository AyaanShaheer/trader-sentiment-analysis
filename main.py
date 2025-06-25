import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# ===============================
# SECTION 1: Load and Preprocess
# ===============================
fear_greed_df = pd.read_csv('fear_greed_index.csv')
historical_data_df = pd.read_csv('historical_data.csv')

fear_greed_df['date'] = pd.to_datetime(fear_greed_df['date']).dt.date
historical_data_df['date'] = pd.to_datetime(historical_data_df['Timestamp IST'], format='%d-%m-%Y %H:%M', errors='coerce')
historical_data_df = historical_data_df.dropna(subset=['date'])
historical_data_df['date'] = historical_data_df['date'].dt.date

# Merge datasets
merged_df = pd.merge(historical_data_df, fear_greed_df[['date', 'classification', 'value']], on='date', how='left')

# Convert numeric fields
merged_df['Closed PnL'] = pd.to_numeric(merged_df['Closed PnL'], errors='coerce')
merged_df['Size USD'] = pd.to_numeric(merged_df['Size USD'], errors='coerce')

# ===============================
# SECTION 2: Summary Statistics
# ===============================
summary = merged_df.groupby('classification').agg({
    'Closed PnL': ['mean', 'sum', 'count'],
    'Size USD': ['mean', 'sum'],
    'Account': pd.Series.nunique
})
summary.columns = ['Avg PnL', 'Total PnL', 'Trade Count', 'Avg Size USD', 'Total Volume USD', 'Unique Traders']
summary = summary.reset_index()
print("\nSummary by Market Sentiment:\n")
print(summary)

# ===============================
# SECTION 3: Clean Boxplot
# ===============================
sentiment_order = ['Extreme Fear', 'Fear', 'Neutral', 'Greed', 'Extreme Greed']
filtered_df = merged_df[(merged_df['Closed PnL'] > -10000) & (merged_df['Closed PnL'] < 20000)]

plt.figure(figsize=(10, 6))
sns.boxplot(
    data=filtered_df,
    x='classification',
    y='Closed PnL',
    order=sentiment_order,
    palette='coolwarm',
    showmeans=True
)
plt.axhline(0, color='black', linestyle='--', linewidth=1)
plt.title('PnL Distribution by Market Sentiment (Filtered)')
plt.xlabel('Market Sentiment')
plt.ylabel('Closed PnL ($)')
plt.xticks(rotation=15)
plt.grid(True)
plt.tight_layout()
plt.show()

# ===============================
# SECTION 4: Barplot - Volume
# ===============================
plt.figure(figsize=(10, 6))
sns.barplot(data=merged_df, x='classification', y='Size USD', estimator=sum, order=sentiment_order, palette='viridis')
plt.title('Total Trade Volume by Sentiment')
plt.xlabel('Market Sentiment')
plt.ylabel('Total Trade Volume (USD)')
plt.xticks(rotation=15)
plt.tight_layout()
plt.show()

# ===============================
# SECTION 5: Sentiment vs PnL
# ===============================
corr_df = merged_df[['value', 'Closed PnL']].dropna()
print("\nCorrelation between Sentiment Value and Closed PnL:\n")
print(corr_df.corr())

plt.figure(figsize=(8, 6))
sns.scatterplot(data=corr_df, x='value', y='Closed PnL', alpha=0.3)
sns.regplot(data=corr_df, x='value', y='Closed PnL', scatter=False, color='red')
plt.title("Sentiment Index vs Closed PnL")
plt.xlabel("Fear-Greed Index")
plt.ylabel("Closed PnL")
plt.grid(True)
plt.tight_layout()
plt.show()

# ===============================
# SECTION 6: Daily Time Series
# ===============================
daily_pnl = merged_df.groupby('date')['Closed PnL'].sum().reset_index()
daily_sentiment = fear_greed_df[['date', 'value']]
time_df = pd.merge(daily_pnl, daily_sentiment, on='date', how='left')

fig, ax1 = plt.subplots(figsize=(14, 6))
ax1.plot(time_df['date'], time_df['Closed PnL'], color='green', label='Daily Total PnL')
ax1.set_ylabel('Total Closed PnL', color='green')
ax1.tick_params(axis='y', labelcolor='green')

ax2 = ax1.twinx()
ax2.plot(time_df['date'], time_df['value'], color='blue', label='Sentiment Index')
ax2.set_ylabel('Fear-Greed Index', color='blue')
ax2.tick_params(axis='y', labelcolor='blue')

plt.title('Daily Total PnL vs Fear-Greed Index')
fig.tight_layout()
plt.show()

# ===============================
# SECTION 7: Trader Profiling
# ===============================
trader_df = merged_df.groupby('Account').agg({
    'Closed PnL': 'sum',
    'Size USD': 'mean',
    'value': 'mean',
    'classification': pd.Series.nunique,
    'date': 'nunique'
}).reset_index()

trader_df.columns = ['Account', 'Total PnL', 'Avg Trade Size', 'Avg Sentiment Score', 'Sentiment Diversity', 'Active Days']

# Select features for clustering
features = ['Total PnL', 'Avg Trade Size', 'Avg Sentiment Score', 'Active Days']
X = trader_df[features]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply KMeans
kmeans = KMeans(n_clusters=4, random_state=42, n_init='auto')
trader_df['Cluster'] = kmeans.fit_predict(X_scaled)

# Visualize clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(data=trader_df, x='Avg Trade Size', y='Total PnL', hue='Cluster', palette='tab10')
plt.title("Trader Clusters based on Behavior")
plt.xlabel("Avg Trade Size ($)")
plt.ylabel("Total PnL ($)")
plt.grid(True)
plt.tight_layout()
plt.show()

# Print cluster behavior summary
print("\nTrader Cluster Summary:\n")
print(trader_df.groupby('Cluster')[features].mean())
