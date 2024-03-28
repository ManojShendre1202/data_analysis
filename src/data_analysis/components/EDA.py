import pandas as pd
from data_analysis.utils.exception import CustomException
from data_analysis.utils.logger import logging
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os


def load_and_preprocess_data(data_path):
    """Loads and preprocesses the raw data.

    Args:
        data_path (str): Path to the CSV file.

    Returns:
        pandas.DataFrame: The processed DataFrame.
    """
    try:
        logging.info("Processing CSV file")
        data = pd.read_csv(data_path)
        data.drop(columns=['model'], inplace=True)
        return data
    except Exception as error:
        raise CustomException(f"Error in loading and preprocessing the data: {error}")


def analyze_data(data):
    """Analyzes the DataFrame, including info, unique values, and nulls.

    Args:
        data (pandas.DataFrame): The DataFrame to analyze.
    """
    try:
        num_cols = data.select_dtypes(exclude='object')
        cat_cols = data.select_dtypes(include='object')

        # Write data shape to file
        with open("C:/datascienceprojects/data_analysis/results/data_details.txt", "w") as f:
            f.write("--- Data Shape ---\n")
            f.write(f"Rows: {data.shape[0]}, Columns: {data.shape[1]}\n\n")

        # Write head to file
            f.write("--- Head (First 5 Rows) ---\n")
            f.write(data.head(5).to_string() + "\n\n")

            # Descriptive statistics
            f.write("--- Descriptive Statistics ---\n")
            f.write(data.describe().to_string() + "\n\n")

            # Write categorical columns to file
            f.write("--- Categorical Columns ---\n")
            for col in data.select_dtypes(include='object'):
                f.write(f"{col}: {len(data[col].unique())} unique values\n")
            f.write("\n")

            # Write null values to file
            f.write("--- Null Values ---\n")
            f.write(data.isnull().sum().to_string() + "\n")
    except (TypeError, ValueError, IOError) as e:  # More specific errors
        raise CustomException(f"Error in 'analyze_data': {e}") 


def detect_and_handle_outliers(data, outlier_cols):
    """Detects, handles outliers, and logs results.

    Args:
        data (pandas.DataFrame): The DataFrame to process.
        outlier_cols (list): Columns to check for outliers.

    Returns:
        pandas.DataFrame: DataFrame with outliers handled.
    """
    try:
        logging.info("Checking for Outliers")
        for col in outlier_cols:
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1

            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            # Handle outliers: Here, we replace with mean for simplicity
            data[col] = np.clip(data[col], lower_bound, upper_bound)

            # Append outlier handling details to file
            with open("C:/datascienceprojects/data_analysis/results/data_details.txt", "a") as f:
                f.write("--- Outlier Handling ---\n")
                f.write(f"{col} - Lower Bound: {lower_bound}\n")
                f.write(f"{col} - Upper Bound: {upper_bound}\n\n")
        return data

    except Exception as e:
        raise CustomException(f"Error in removing outliers: {e}")


def visualize_data(data, plot_dir):
    """Creates histograms, bar plots, scatter plots, and pie charts.

    Args:
        data (pandas.DataFrame): The DataFrame to visualize.
        plot_dir (str):  The directory to save the plots. 
    """
    try:
        logging.info("Visualizing Data")
        create_histogram(data, plot_dir)
        create_barplot(data, plot_dir)
        create_scatterplot(data, plot_dir)
        create_piecharts(data, plot_dir)
    except Exception as e:
        raise CustomException(f"Error in visualizing the data: {e}")
    
def create_scatterplot(data, plot_dir):
    """Creates scatter plots for numerical columns."""
    try:
        logging.info("Creating scatter plots for numerical columns")
        fig, axs = plt.subplots(2, 2, figsize=(12, 8))
        axs[0, 0].scatter(data['price'].iloc[:10000], data['mileage'].iloc[:10000], s=6)
        axs[0, 0].set_title('Price vs KM Travelled')
        axs[0, 0].set_xlabel('Price')
        axs[0, 0].set_ylabel('KM Travelled')

        axs[0, 1].scatter(data['price'].iloc[:10000], data['mpg'].iloc[:10000], s=6)
        axs[0, 1].set_title('Price vs Miles per Gallon')
        axs[0, 1].set_xlabel('Price')
        axs[0, 1].set_ylabel('MPG')

        axs[1, 0].scatter(data['mileage'].iloc[:10000], data['mpg'].iloc[:10000], s=6)
        axs[1, 0].set_title('KM Travelled vs Miles per Gallon')
        axs[1, 0].set_xlabel('KM Travelled')
        axs[1, 0].set_ylabel('MPG')

        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, 'scatterplots.png'))
        plt.close(fig)
    except Exception as e:
        raise CustomException(f"Error in creating scatter plots: {e}")


def create_histogram(data, plot_dir):
    """Creates histograms for numerical columns using Seaborn."""
    try:
        logging.info("Creating histograms for numerical columns with Seaborn")
        num_cols = data.select_dtypes(exclude='object')
        n_bins = 20
        ncols = len(num_cols.columns)

        fig, axes = plt.subplots(nrows=int((ncols + 1) / 2), ncols=2, figsize=(8, 5))
        axes_flat = axes.ravel()

        for i, col in enumerate(num_cols.columns):
            sns.histplot(data=data, x=col, ax=axes_flat[i], bins=n_bins, color='green', fill=False)  # Adjust color as needed
            axes_flat[i].set_title(col)

        if ncols % 2 != 0:
            axes_flat[-1].axis('off')

        fig.tight_layout()
        plt.savefig(os.path.join(plot_dir, 'histograms_seaborn.png'))  # Updated filename
        plt.close(fig) 
    except Exception as e:
        logging.error(f"Error creating histograms with Seaborn: {e}")


def create_barplot(data, plot_dir):
    """Creates bar plots for categorical columns."""
    try:
        logging.info("Creating Barplots plots")
        fig, axs = plt.subplots(2, 2, figsize=(12, 8))

        # Plot 1: KM Travelled vs Fuel Type
        sns.barplot(x='mileage', y='fuelType', data=data, ax=axs[0, 0])
        axs[0, 0].set_title('KM Travelled vs Fuel Type')
        axs[0, 0].set_xlabel('KM Travelled')
        axs[0, 0].set_ylabel('Fuel Type')

        # Plot 2: Miles per Gallon vs Fuel Type
        sns.barplot(x='mpg', y='fuelType', data=data, ax=axs[0, 1])
        axs[0, 1].set_title('Miles per Gallon vs Fuel Type')
        axs[0, 1].set_xlabel('Miles per Gallon')
        axs[0, 1].set_ylabel('Fuel Type')

        # Plot 3: Miles per Gallon vs Transmission
        sns.barplot(x='mpg', y='transmission', data=data, ax=axs[1, 0])
        axs[1, 0].set_title('Miles per Gallon vs Transmission')
        axs[1, 0].set_xlabel('Miles per Gallon')
        axs[1, 0].set_ylabel('Transmission')

        # Plot 4: KM Driven vs Transmission
        sns.barplot(x='mileage', y='transmission', data=data, ax=axs[1, 1])
        axs[1, 1].set_title('KM Driven vs Transmission')
        axs[1, 1].set_xlabel('KM Driven')
        axs[1, 1].set_ylabel('Transmission')

        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, 'barplot.png'))
        plt.close(fig)
    except Exception as e:
        raise CustomException(f"Error in creating scatter plots: {e}")


def create_piecharts(data, plot_dir):
    """
    Creates pie charts for categorical distributions.

    Args:
        data (pandas.DataFrame): The DataFrame to visualize.
        plot_dir (str):  The directory to save the plots. 

    Returns:
        None

    Raises:
        CustomException: If there is an error in creating the pie charts.

    """
    try:
        logging.info("Creating pie charts for categorical distributions")
        for col in ['Manufacturer', 'fuelType', 'transmission']:
            if col in data.columns:  # Validate column presence
                cat_counts = data[col].value_counts()

                # Select appropriate explode values based on data
                explode = np.zeros(len(cat_counts))  # Default no explosion
                if len(cat_counts) > 2:
                    explode[0] = 0.2  # Explode largest slice if more than 2 categories

                plt.figure(figsize=(6, 6))
                plt.pie(cat_counts, labels=cat_counts.index, autopct='%1.1f%%', 
                        startangle=140, explode=explode, shadow=True)
                plt.title(f'Distribution of {col}')
                plt.axis('equal')
                plt.savefig(os.path.join(plot_dir, f'{col.lower()}_pie.png'))
                plt.close()
    except Exception as e:
        raise CustomException(f"Error in creating pie charts: {e}")


