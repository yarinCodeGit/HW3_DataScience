"""
===========================================================================
Holon Institute of Technology (HIT)
Course: Introduction to Data Science
Assignment 1: Exploratory Data Analysis (EDA)

Author: Yarin Refael Sapir
Dataset: Tech Stock Financial Dataset (2020 - Today)
===========================================================================
"""

import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Define the file path (Ensure the CSV is in the same directory)
file_path = 'tech_stock_prices_2020_to_today.csv'

# ==========================================
# Section 3: Meta-Analysis & Data Structure
# ==========================================
print("==========================================")
print(" SECTION 3: META-ANALYSIS & STRUCTURE")
print("==========================================")

# 3.1 File Analysis
file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
print(f"\n--- 3.1 File Analysis ---")
print(f"File Format: CSV")
print(f"File Size: {file_size_mb:.2f} MB")

# 3.2 Data Structure
df = pd.read_csv(file_path)
print(f"\n--- 3.2 Data Structure ---")
print(f"Dataset Shape: {df.shape}")
print("Column Names:", df.columns.tolist())
print("\nData Types:")
print(df.dtypes)


# ==========================================
# Section 4: Data Quality & Pre-processing
# ==========================================
print("\n==========================================")
print(" SECTION 4: DATA QUALITY & CLEANING")
print("==========================================")

print("\n--- 4.1 & 4.2 & 4.3 & 4.4 Data Quality Check (Before Cleaning) ---")
missing_percent = (df.isnull().sum() / len(df)) * 100
print("Missing Values (%):\n", missing_percent[missing_percent > 0].sort_values(ascending=False))
print(f"\nTotal full duplicates: {df.duplicated().sum()}")
print("\nCardinality (Unique Values):\n", df.nunique())

print("\n--- Applying Data Cleaning ---")
# 1. Convert 'Date' to datetime object
df['Date'] = pd.to_datetime(df['Date'])

# 2. Drop columns with no variance (Cardinality = 1)
columns_to_drop = ['Dividends', 'Stock Splits']
df = df.drop(columns=columns_to_drop)

# 3. Fill missing values with column medians
columns_to_fill = ['P/E Ratio', 'Market Cap', 'Price/Sales Ratio', 'Price/Book Ratio',
                   'Dividend Yield', 'Daily Return', '20-Day MA']
for col in columns_to_fill:
    df[col] = df[col].fillna(df[col].median())

print("Data cleaning completed: Date converted, empty columns dropped, missing values filled with medians.")

# Verification
print("\n--- Verification After Cleaning ---")
print(f"Date column data type: {df['Date'].dtype}")
print(f"Total missing values remaining: {df.isnull().sum().sum()}")


# ==========================================
# Section 5: Univariate Analysis
# ==========================================
print("\n==========================================")
print(" SECTION 5: UNIVARIATE ANALYSIS")
print("==========================================")

# --- 5.1 Numerical Variables ---
print("\n--- 5.1 Numerical Variables: Descriptive Statistics ---")
num_cols = ['Open', 'Volume', 'Daily Return']

for col in num_cols:
    print(f"\nStats for {col}:")
    desc = df[col].describe()

    # Calculating MAD (Median Absolute Deviation)
    mad_val = (df[col] - df[col].median()).abs().median()
    iqr_val = desc['75%'] - desc['25%']
    skewness = df[col].skew()

    print(f"Mean: {desc['mean']:.2f} | Median: {desc['50%']:.2f} | Std: {desc['std']:.2f}")
    print(f"Min: {desc['min']:.2f} | Max: {desc['max']:.2f}")
    print(f"Quantiles -> 25%: {desc['25%']:.2f} | 75%: {desc['75%']:.2f}")
    print(f"IQR: {iqr_val:.2f} | MAD: {mad_val:.2f}")
    print(f"Skewness (Bias): {skewness:.2f}")

print("\n--- Outlier Detection (Demonstrated on 'Volume') ---")
col_outlier = 'Volume'

# Method 1: Z-Score (> 3 standard deviations)
z_scores = (df[col_outlier] - df[col_outlier].mean()) / df[col_outlier].std()
outliers_z = df[abs(z_scores) > 3]
print(f"1. Outliers using Z-Score: {len(outliers_z)} rows")

# Method 2: IQR (1.5 * IQR rule)
Q1 = df[col_outlier].quantile(0.25)
Q3 = df[col_outlier].quantile(0.75)
IQR = Q3 - Q1
outliers_iqr = df[(df[col_outlier] < (Q1 - 1.5 * IQR)) | (df[col_outlier] > (Q3 + 1.5 * IQR))]
print(f"2. Outliers using IQR: {len(outliers_iqr)} rows")

# Method 3: MAD (Median Absolute Deviation threshold)
median_val = df[col_outlier].median()
mad_stat = (df[col_outlier] - median_val).abs().median()
mad_threshold = 3 * (mad_stat / 0.6745)
outliers_mad = df[abs(df[col_outlier] - median_val) > mad_threshold]
print(f"3. Outliers using MAD: {len(outliers_mad)} rows")


# --- 5.2 Categorical Variables ---
print("\n--- 5.2 Categorical Variables ('Ticker') ---")
ticker_counts = df['Ticker'].value_counts()
ticker_pct = df['Ticker'].value_counts(normalize=True) * 100

# 1. Mode and its percentage
mode_val = ticker_counts.index[0]
mode_pct = ticker_pct.iloc[0]
print(f"1. Mode: '{mode_val}' (Note: Dataset is perfectly balanced).")
print(f"   It takes exactly {mode_pct:.2f}% of the data.")

# 2. Top K most common variables
K = 3
top_k = ticker_pct.head(K)
print(f"\n2. Top K={K} most common variables:")
for ticker, pct in top_k.items():
    print(f"   - {ticker}: {pct:.2f}%")

# 3. Smallest number of values to describe P percent of the data
P = 80
cumulative_pct = ticker_pct.cumsum()
num_cats_for_p = (cumulative_pct <= P).sum() + 1
vals_for_p = ticker_pct.index[:num_cats_for_p].tolist()
print(f"\n3. Smallest number of values to describe P={P}% of the data is {num_cats_for_p}.")
print(f"   These values are: {vals_for_p}")

print("\n4. Rare categories:")
print("   NO rare categories. All values are informative (10% each).")


# ==========================================
# Section 6: Correlations and Relationships
# ==========================================
print("\n==========================================")
print(" SECTION 6: CORRELATIONS & VISUALIZATIONS")
print("==========================================")

sns.set_theme(style="whitegrid")

# --- 6.1 Numerical Correlations ---
print("\n--- 6.1 Numerical-Numerical Correlations ---")
corr_cols = ['Open', 'Close', 'Volume', 'P/E Ratio', 'Market Cap', 'Daily Return']
df_corr = df[corr_cols]

pearson_corr = df_corr.corr(method='pearson')
spearman_corr = df_corr.corr(method='spearman')
kendall_corr = df_corr.corr(method='kendall')
print("Pearson, Spearman, and Kendall Correlations calculated.")

# --- 6.2 Categorical-Numerical (Binning) ---
print("\n--- 6.2 Categorical-Numerical (Binning) ---")
df['Volume_Binned'] = pd.qcut(df['Volume'], q=3, labels=['Low', 'Medium', 'High'])
freq_table = pd.crosstab(df['Ticker'], df['Volume_Binned'], normalize='index') * 100
print("Frequency Table (Row Percentages) for Ticker vs. Volume_Binned:")
print(freq_table.round(2))

# --- 6.3 Visualizations (Graphs) ---
print("\n--- 6.3 Generating Graphs (Close windows to continue) ---")

# 1. Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(pearson_corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Heatmap: Pearson Correlation Matrix')
plt.tight_layout()
plt.show()

# 2. Scatterplot
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='Open', y='Close', hue='Ticker', alpha=0.5, s=15)
plt.title('Scatterplot: Open vs. Close Prices')
plt.xlabel('Open Price')
plt.ylabel('Close Price')
plt.tight_layout()
plt.show()

# 3. Histogram
plt.figure(figsize=(8, 6))
sns.histplot(df['Daily Return'], bins=50, kde=True, color='purple')
plt.title('Histogram: Distribution of Daily Returns')
plt.xlabel('Daily Return')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

# 4. Bar Chart
plt.figure(figsize=(10, 6))
avg_volume = df.groupby('Ticker')['Volume'].mean().sort_values(ascending=False)
sns.barplot(x=avg_volume.index, y=avg_volume.values, hue=avg_volume.index, palette='viridis', legend=False)
plt.title('Bar Chart: Average Trading Volume per Ticker')
plt.xlabel('Ticker')
plt.ylabel('Average Volume')
plt.tight_layout()
plt.show()

# 5. Boxplot
plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x='Ticker', y='Daily Return', hue='Ticker', palette='Set2', legend=False)
plt.title('Boxplot: Daily Returns Distribution per Ticker')
plt.xlabel('Ticker')
plt.ylabel('Daily Return')
plt.tight_layout()
plt.show()

# 6. Violin Plot
plt.figure(figsize=(12, 6))
sns.violinplot(data=df, x='Ticker', y='P/E Ratio', hue='Ticker', palette='muted', legend=False)
plt.title('Violin Plot: P/E Ratio Distribution per Ticker')
plt.xlabel('Ticker')
plt.ylabel('P/E Ratio')
plt.tight_layout()
plt.show()

# 7. Pie Chart
plt.figure(figsize=(8, 8))
ticker_counts = df['Ticker'].value_counts()
plt.pie(ticker_counts, labels=ticker_counts.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette('pastel'))
plt.title('Pie Chart: Data Distribution by Ticker')
plt.tight_layout()
plt.show()

# 8. Pairplot
sample_df = df.sample(n=1000, random_state=42)
sns.pairplot(sample_df[['Open', 'Volume', 'Daily Return', 'Ticker']], hue='Ticker', diag_kind='kde', palette='tab10')
plt.suptitle('Pairplot: Numerical Relationships (Sampled Data)', y=1.02)
plt.show()


# ==========================================
# Section 7: Index Analysis (5%)
# ==========================================
print("\n==========================================")
print(" SECTION 7: INDEX ANALYSIS")
print("==========================================")

# Setting 'Date' as the official index of the DataFrame
df_indexed = df.set_index('Date')

print("\n--- 7.1 Index Properties ---")

# 1. Is the index unique?
is_unique = df_indexed.index.is_unique
print(f"1. Is the index unique? {is_unique}")
if not is_unique:
    print("   Explanation: Not unique because multiple companies (Tickers) trade on the exact same date.")

# 2. Is it time-based?
is_time_based = isinstance(df_indexed.index, pd.DatetimeIndex)
print(f"\n2. Is it time-based? {is_time_based}")

# 3. Does the analysis change over time?
print("\n3. Does the analysis change over time? True")
print("   Explanation: Stock prices and volumes are highly dynamic and fluctuate continuously over time.")

# 4. Is it sorted?
is_sorted = df_indexed.index.is_monotonic_increasing
print(f"\n4. Is the index sorted? {is_sorted}")

if not is_sorted:
    df_indexed = df_indexed.sort_index()
    print("   -> Note: The dataset has now been sorted chronologically by the new index.")

import scipy.stats as stats

# ==========================================
# Section 9: BONUS (+10%)
# ==========================================
print("\n==========================================")
print(" SECTION 9: BONUS (+10%)")
print("==========================================")

# --- 9.1 Feature Engineering ---
print("\n--- 9.1 Feature Engineering ---")
# 1. Daily Spread: Measures the daily price volatility (High - Low)
df_indexed['Daily Spread'] = df_indexed['High'] - df_indexed['Low']

# 2. Price Change: The difference between Close and Open price
df_indexed['Price Change'] = df_indexed['Close'] - df_indexed['Open']

# 3. Time Features: Extracting Month and Day of Week from the index
df_indexed['Month'] = df_indexed.index.month
df_indexed['DayOfWeek'] = df_indexed.index.day_name()

print("Successfully engineered new features: 'Daily Spread', 'Price Change', 'Month', 'DayOfWeek'.")


# --- 9.2 Time Dependency & External Info ---
print("\n--- 9.2 Time Dependency Analysis ---")
print("Analyzing the 'Weekend Effect' (Monday vs. Friday returns) using our new features:")
avg_return_by_day = df_indexed.groupby('DayOfWeek')['Daily Return'].mean()
print(avg_return_by_day[['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']].round(4))


# --- 9.3 Hypothesis Testing ---
print("\n--- 9.3 Hypothesis Testing (T-Test) ---")
print("Hypothesis: Is there a statistically significant difference in the daily trading volume between AAPL and MSFT?")

# Drop NaNs just in case to ensure the statistical test runs smoothly
aapl_vol = df_indexed[df_indexed['Ticker'] == 'AAPL']['Volume'].dropna()
msft_vol = df_indexed[df_indexed['Ticker'] == 'MSFT']['Volume'].dropna()

# Perform Welch's t-test (assumes unequal variances)
t_stat, p_value = stats.ttest_ind(aapl_vol, msft_vol, equal_var=False)

print(f"T-statistic: {t_stat:.2f}")
print(f"P-value: {p_value:.4e}")

if p_value < 0.05:
    print("Conclusion: Reject the Null Hypothesis. There IS a significant difference in trading volumes.")
else:
    print("Conclusion: Fail to reject the Null Hypothesis. No significant difference found.")

print("\nAll required analysis and graphs generated successfully. Project complete!")