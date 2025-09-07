import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
import numpy as np

# Set style for better visualizations
plt.style.use('seaborn')
sns.set_style("whitegrid")

# Task 1: Load and Explore the Dataset
def load_and_explore_data():
    try:
        # Load Iris dataset
        iris = load_iris()
        df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
        df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
        
        # Display first few rows
        print("First 5 rows of the dataset:")
        print(df.head())
        print("\nDataset Info:")
        print(df.info())
        
        # Check for missing values
        print("\nMissing Values:")
        print(df.isnull().sum())
        
        return df
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        return None

# Task 2: Basic Data Analysis
def analyze_data(df):
    print("\nBasic Statistics:")
    print(df.describe())
    
    # Group by species and calculate mean for each numerical column
    print("\nMean values by species:")
    grouped_means = df.groupby('species').mean()
    print(grouped_means)
    
    # Findings
    print("\nFindings from Analysis:")
    print("- Setosa has the smallest average sepal length but largest average sepal width")
    print("- Virginica has the largest average petal length and width")
    print("- Versicolor shows intermediate values for most measurements")

# Task 3: Data Visualization
def create_visualizations(df):
    # Create a figure with subplots
    fig = plt.figure(figsize=(15, 10))
    
    # 1. Line Plot (Petal Length trend across samples for each species)
    plt.subplot(2, 2, 1)
    for species in df['species'].unique():
        species_data = df[df['species'] == species]['petal length (cm)']
        plt.plot(species_data, label=species)
    plt.title('Petal Length Trend Across Samples')
    plt.xlabel('Sample Index')
    plt.ylabel('Petal Length (cm)')
    plt.legend()
    
    # 2. Bar Plot (Mean sepal length by species)
    plt.subplot(2, 2, 2)
    mean_sepal_length = df.groupby('species')['sepal length (cm)'].mean()
    mean_sepal_length.plot(kind='bar')
    plt.title('Average Sepal Length by Species')
    plt.xlabel('Species')
    plt.ylabel('Sepal Length (cm)')
    
    # 3. Histogram (Sepal width distribution)
    plt.subplot(2, 2, 3)
    sns.histplot(data=df, x='sepal width (cm)', bins=20)
    plt.title('Distribution of Sepal Width')
    plt.xlabel('Sepal Width (cm)')
    plt.ylabel('Count')
    
    # 4. Scatter Plot (Sepal length vs Petal length)
    plt.subplot(2, 2, 4)
    sns.scatterplot(data=df, x='sepal length (cm)', y='petal length (cm)', hue='species', size='species')
    plt.title('Sepal Length vs Petal Length')
    plt.xlabel('Sepal Length (cm)')
    plt.ylabel('Petal Length (cm)')
    
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('iris_visualizations.png')
    plt.show()

def main():
    # Load and explore
    df = load_and_explore_data()
    if df is None:
        return
    
    # Analyze data
    analyze_data(df)
    
    # Create visualizations
    create_visualizations(df)

if __name__ == "__main__":
    main()
