# Importing necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import streamlit as st

# Load the dataset
data_path = "cosmetic_p.csv"
data = pd.read_csv(data_path)

# Basic Exploration
def explore_data(df):
    print("Data Info:")
    print(df.info())
    print("\nFirst Five Rows:")
    print(df.head())
    print("\nSummary Statistics:")
    print(df.describe())

# Clean and preprocess the dataset
def preprocess_data(df):
    # Dropping duplicates if any
    df = df.drop_duplicates()

    # Handling missing values (if applicable)
    if df.isnull().sum().sum() > 0:
        df = df.fillna(method='ffill')

    return df

# Analyze ingredient frequency
def analyze_ingredients(df):
    ingredient_series = df['ingredients'].str.split(', ').explode()
    ingredient_counts = ingredient_series.value_counts()
    
    print("\nTop 10 Ingredients:")
    print(ingredient_counts.head(10))

    # Plot
    plt.figure(figsize=(12, 6))
    sns.barplot(x=ingredient_counts.head(10).values, y=ingredient_counts.head(10).index, palette="viridis")
    plt.title('Top 10 Most Common Ingredients')
    plt.xlabel('Frequency')
    plt.ylabel('Ingredients')
    plt.show()

# Analyze price vs rank
def analyze_price_rank(df):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='price', y='rank', data=df, hue='Label', palette='deep', alpha=0.7)
    plt.title('Price vs Rank by Product Category')
    plt.xlabel('Price ($)')
    plt.ylabel('Rank')
    plt.legend(title='Category')
    plt.show()

# Perform clustering on products
def cluster_products(df):
    features = df[['price', 'rank']]
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    kmeans = KMeans(n_clusters=3, random_state=42)
    df['cluster'] = kmeans.fit_predict(scaled_features)

    # Plot clusters
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='price', y='rank', hue='cluster', data=df, palette='Set2', style='cluster', s=100)
    plt.title('Product Clusters Based on Price and Rank')
    plt.xlabel('Price ($)')
    plt.ylabel('Rank')
    plt.legend(title='Cluster')
    plt.show()

# Predict ingredient popularity trends using a mock ML model
def predict_ingredient_trends(df):
    ingredient_series = df['ingredients'].str.split(', ').explode()
    ingredient_counts = ingredient_series.value_counts()
    
    # Mock prediction logic (trend = count * random factor)
    ingredient_counts_trend = ingredient_counts * 1.05  # Simulate a 5% growth trend
    
    print("\nPredicted Top 10 Ingredients for Future Trends:")
    print(ingredient_counts_trend.head(10))

# Create an interactive dashboard using Streamlit
def interactive_dashboard(df):
    st.title("Sephora Skincare Analysis")

    st.header("Dataset Overview")
    st.write(df.head())

    st.header("Top 10 Ingredients")
    ingredient_series = df['ingredients'].str.split(', ').explode()
    ingredient_counts = ingredient_series.value_counts()
    st.bar_chart(ingredient_counts.head(10))

    st.header("Price vs Rank by Product Category")
    sns.scatterplot(x='price', y='rank', data=df, hue='Label', palette='deep', alpha=0.7)
    plt.title('Price vs Rank by Product Category')
    plt.xlabel('Price ($)')
    plt.ylabel('Rank')
    st.pyplot()

    st.header("Product Clusters")
    cluster_products(df)

# Store data in SQL database and perform queries
def store_and_query_data(df):
    # Connect to SQLite database
    conn = sqlite3.connect("sephora_skincare.db")
    
    # Store the dataframe in a SQL table
    df.to_sql("products", conn, if_exists="replace", index=False)
    
    # Example queries
    print("\nTop 5 Most Expensive Products:")
    query1 = """
    SELECT name, brand, price FROM products
    ORDER BY price DESC
    LIMIT 5;
    """
    print(pd.read_sql(query1, conn))

    print("\nAverage Price by Category:")
    query2 = """
    SELECT Label, AVG(price) as avg_price FROM products
    GROUP BY Label
    ORDER BY avg_price DESC;
    """
    print(pd.read_sql(query2, conn))

    conn.close()

# Main execution
if __name__ == "__main__":
    explore_data(data)
    data = preprocess_data(data)
    analyze_ingredients(data)
    analyze_price_rank(data)
    cluster_products(data)
    predict_ingredient_trends(data)
    store_and_query_data(data)
    # Uncomment the following line to run the Streamlit dashboard
    # interactive_dashboard(data)

