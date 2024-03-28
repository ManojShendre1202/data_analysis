from data_analysis.components.EDA import *
import os
def main():
    logging.info("Starting EDA Process")
    data_path = "C:/datascienceprojects/data_analysis/data/raw.csv"
    plot_dir = "C:/datascienceprojects/data_analysis/results/plots"

    # Create plot directory if it doesn't exist
    os.makedirs(plot_dir, exist_ok=True)

    data = load_and_preprocess_data(data_path)
    analyze_data(data)
    data = detect_and_handle_outliers(data, ['price','mileage','mpg','engineSize','tax'])
    visualize_data(data, plot_dir)

if __name__ == "__main__":
    main()