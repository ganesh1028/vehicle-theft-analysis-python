import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis
from statsmodels.stats.weightstats import ztest
from scipy.stats import ttest_ind, ttest_1samp, chi2_contingency
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv("C:/Users/DELL/OneDrive/Desktop/-Predictive- python Analysis-and-Visualization-of-Stolen-vehicle-Patterns-Using-Real-World-Data--main/pythonproject/cleaned_stolen_vehicles.csv", encoding='utf-8-sig')
#Dataset Overview
print("Dataset Overview:")
print("\nDataset Shape:", df.shape)
print("Column Names:\n", df.columns.tolist())
print(f"Shape: {df.shape}")
print("\nFirst 5 rows:",df.head())
print("\nMissing values:\n", df.isnull().sum())
print("\nCleaned dataset info:\n")
print(df.info())
df.to_csv('cleaned_stolen_vehicles.csv', index=False)
df = df.drop_duplicates()
numeric_cols = df.select_dtypes(include='number').columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
#Fill categorical columns with mode
categorical_cols = df.select_dtypes(include='object').columns
for col in categorical_cols:
    if df[col].isnull().any():
        df[col] = df[col].fillna(df[col].mode()[0])
#Rename columns for consistency
df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
# Convert 'date_stolen' to datetime
if 'date_stolen' in df.columns:
    df['date_stolen'] = pd.to_datetime(df['date_stolen'], errors='coerce')
#Final check
print("\nCleaned dataset info:\n")
print(df.info())
df.to_csv('cleaned_stolen_vehicles.csv', index=False)

# Numerical Summary
numerical = df.select_dtypes(include=[np.number])
print("\nNumerical Summary:")
print(numerical.describe().T)

#Advanced Stats: Skewness & Kurtosis
print("\nSkewness & Kurtosis:")
for col in numerical.columns:
    print(f"\n{col}")
    print(f"Skewness : {skew(df[col].dropna()):.2f}")
    print(f"Kurtosis : {kurtosis(df[col].dropna()):.2f}")

#Central Tendency + Spread
print("\nCentral Tendency + Spread:")
for col in numerical.columns:
    col_data = df[col].dropna()
    print(f"\n{col}")
    print(f"Mean      : {col_data.mean():.2f}")
    print(f"Median    : {col_data.median():.2f}")
    print(f"Mode      : {col_data.mode().iloc[0] if not col_data.mode().empty else 'N/A'}")
    print(f"Min       : {col_data.min()}")
    print(f"Max       : {col_data.max()}")
    print(f"Range     : {col_data.max() - col_data.min()}")
    print(f"Std Dev   : {col_data.std():.2f}")

#Numerical Summary
numerical = df.select_dtypes(include=['int64', 'float64'])
print("\nNumerical Summary:")
print(numerical.describe())
#Categorical Summary
categorical = df.select_dtypes(include=['object'])
print("\nCategorical Summary:")
for col in categorical.columns:
    print(f"\n{col}")
    print(f"Unique Values : {df[col].nunique()}")
    print(f"Top Value     : {df[col].mode()[0] if not df[col].mode().empty else 'N/A'}")
    print(f"Top Frequency : {df[col].value_counts().iloc[0] if not df[col].value_counts().empty else 'N/A'}")
#Missing Data Overview
missing = df.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False)
print("\nMissing Value Report:")
if not missing.empty:
    print(missing)
else:
    print("No missing values found!")
# Correlation Matrix (for numerical features)
print("\nCorrelation Matrix:\n")
print(numerical.corr().round(2))

#Outlier Detection using IQR Method (for model_year)
print("\nOutlier Detection (IQR Method) for Model Year:")
col = 'model_year'
Q1 = df[col].quantile(0.25)
Q3 = df[col].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
print(f"\nColumn: {col}")
print(f"Outliers Detected: {outliers.shape[0]}")
print(f"Lower Bound: {lower_bound:.2f}, Upper Bound: {upper_bound:.2f}")

# ──────────────────────────────────────────────
# Statistical Tests
# ──────────────────────────────────────────────

# 7. Custom Z-Test implementation
def z_test(sample, popmean):
    sample_mean = np.mean(sample)
    sample_std = np.std(sample, ddof=1)
    n = len(sample)
    z = (sample_mean - popmean) / (sample_std / np.sqrt(n))
    p = 2 * (1 - norm.cdf(abs(z)))  # Two-tailed test
    return z, p

# 8. T-Test: Difference in model year between Trailers and Roadbikes
trailers = df[df['vehicle_type'] == 'Trailer']['model_year'].dropna()
roadbikes = df[df['vehicle_type'] == 'Roadbike']['model_year'].dropna()

if len(trailers) > 1 and len(roadbikes) > 1:  # Need at least 2 samples for t-test
    t_stat, p_val = ttest_ind(trailers, roadbikes, equal_var=False)
    print("\nT-Test: Difference in model year between Trailers and Roadbikes")
    print(f"T-statistic = {t_stat:.2f}")
    print(f"P-value     = {p_val:.4f}")
    
    if p_val < 0.05:
        print("Result: Statistically significant difference in means (reject H0)")
    else:
        print("Result: No statistically significant difference (fail to reject H0)")
else:
    print("\nInsufficient data for T-Test between Trailers and Roadbikes")

# 9. Chi-Square Test: Is vehicle_type independent of color?
if len(df['vehicle_type'].unique()) > 1 and len(df['color'].unique()) > 1:
    contingency_table = pd.crosstab(df['vehicle_type'], df['color'])
    if contingency_table.size > 0:  # Check if table has data
        chi2, p_val, dof, expected = chi2_contingency(contingency_table)
        print("\nChi-Square Test: Is vehicle_type independent of color?")
        print(f"Chi2 Statistic = {chi2:.2f}")
        print(f"P-value        = {p_val:.4f}")
    else:
        print("\nNot enough data for Chi-Square Test")
else:
    print("\nNot enough categories for Chi-Square Test")

# ──────────────────────────────────────────────
# Visualizations
# ──────────────────────────────────────────────

# 10. Line Plot - Theft Trends Over Time (if date_stolen is properly formatted)
# Convert date_stolen to datetime if needed
plt.figure(figsize=(10, 5))
df['date_stolen'] = pd.to_datetime(df['date_stolen'])
df = df.sort_values('date_stolen')
df['date_stolen'].value_counts().sort_index().plot(kind='line', color='yellow')
plt.grid(True, linestyle='--', alpha=0.5)
plt.title("Daily Vehicle Thefts Over Time")
plt.xlabel("Date")
plt.ylabel("Number of Thefts")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 11. Bar Plot - Top 10 Most Stolen Vehicle Types
plt.figure(figsize=(10, 5))
df['vehicle_type'].value_counts().head(10).plot(kind='bar', color='grey')
plt.title("Top 10 Most Stolen Vehicle Types")
plt.xlabel("Vehicle Type")
plt.ylabel("Number of Thefts")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 12. Histogram - Distribution of Model Years
plt.figure(figsize=(10, 6))
plt.hist(df['model_year'], bins=20, color='orange', edgecolor='red')
plt.xlabel("Model Year")
plt.ylabel("Frequency")
plt.title("Distribution of Stolen Vehicle Model Years")
plt.tight_layout()
plt.show()

# 13. Pie Chart - Proportion of Thefts by Vehicle Type
vehicle_counts = df['vehicle_type'].value_counts().head(6)
plt.figure(figsize=(7, 4))
plt.pie(vehicle_counts, labels=vehicle_counts.index, autopct='%1.1f%%', startangle=140)
plt.title("Theft Distribution by Vehicle Type")
plt.axis('equal')
plt.tight_layout()
plt.show()

# 14. Box Plot - Model Year Distribution by Vehicle Type
plt.figure(figsize=(8, 6))
top_types = df['vehicle_type'].value_counts().head(10).index
sns.boxplot(x='vehicle_type', y='model_year', data=df[df['vehicle_type'].isin(top_types)],palette="Set3")
plt.title("Model Year Distribution by Vehicle Type")
plt.xlabel("Vehicle Type")
plt.ylabel("Model Year")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 15. Heatmap - Correlation Matrix of Numerical Variables
plt.figure(figsize=(8, 6))
corr_matrix = df.select_dtypes(include='number').corr()
sns.heatmap(corr_matrix, annot=True, linewidths=0.5, cmap="Spectral")
plt.title("Correlation Heatmap of Numerical Features")
plt.tight_layout()
plt.show()

# 16. Count Plot - Top Vehicle Colors
plt.figure(figsize=(10, 5))
top_colors = df['color'].value_counts().head(10).index
color_palette = {
    "Silver": "#C0C0C0",
    "White": "#FFFFFF",
    "Black": "#000000",
    "Blue": "#0000FF",
    "Red": "#FF0000",
    "Grey": "#808080",
    "Green": "#008000",
    "Gold": "#FFD700",
    "Brown": "#A52A2A",
    "Yellow": "#FFFF00"
}
palette = [color_palette[color] for color in top_colors]

sns.countplot(
    y='color',
    data=df[df['color'].isin(top_colors)],
    order=top_colors,
    palette=palette, 
    edgecolor='black'
)

plt.title("Top 10 Colors of Stolen Vehicles (True Colors)")
plt.xlabel("Count")
plt.ylabel("Color")
plt.tight_layout()
plt.show()



