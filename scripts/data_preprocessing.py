import pandas as pd
import os

def preprocess_data(input_file: str, output_dir: str):
    # Load the raw data
    df = pd.read_csv(input_file)
    
    # Debugging output to inspect raw data
    print("Raw DataFrame:\n", df.head())

    # Ensure 'close' column is available
    if "Rate" in df.columns:
        # Map 'Rate' to 'close' column
        df.rename(columns={"Rate": "close"}, inplace=True)
    elif "close" not in df.columns:
        # If 'Rate' and 'close' are both missing, generate synthetic 'close' values
        print("Warning: No 'Rate' column found. Generating synthetic 'close' column.")
        # Create a synthetic 'close' column (e.g., using a rolling mean or a default value)
        df['close'] = df['Open'].fillna(df['High']).fillna(df['Low'])  # Example logic
    
    # Convert 'close' column to numeric safely
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    
    # Check for any NaN values in 'close' column
    if df["close"].isnull().any():
        print("Warning: There are NaN values in the 'close' column. Filling NaNs with previous value.")
        df["close"].fillna(method="ffill", inplace=True)  # Forward fill to avoid NaNs in time-series
    
    # Feature engineering: Add additional useful indicators
    df["returns"] = df["close"].pct_change()
    df["rolling_mean"] = df["close"].rolling(window=10).mean()
    df["rolling_std"] = df["close"].rolling(window=10).std()
    df.dropna(inplace=True)  # Remove rows with NaN values (after calculating rolling stats)
    
    # Debugging output to inspect processed data
    print("Processed DataFrame:\n", df.head())
    
    # Save processed data
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "nzdusd_processed.csv")
    df.to_csv(output_file, index=False)
    print(f"Processed data saved to {output_file}")

if __name__ == "__main__":
    preprocess_data("data/raw/nzdusd_raw.csv", "data/processed")
