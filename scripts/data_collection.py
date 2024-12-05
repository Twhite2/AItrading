import requests
import pandas as pd
import os
from datetime import datetime, timezone

# Define the API key and endpoint
API_KEY = "1b028d9a1a0e2a4b5f0bbdb2c0e36c79"
BASE_URL = "https://api.forexrateapi.com/v1/latest"

def fetch_live_data(symbol: str, output_dir: str):
    # Extract base and target currencies
    base_currency, target_currency = symbol.split("/")  # e.g., USD/NZD into USD and NZD
    
    # Set up request parameters
    params = {
        "api_key": API_KEY,                  # API key as per the dashboard
        "base": base_currency,              # Base currency, e.g., USD
        "currencies": f"{base_currency},{target_currency}"  # Target currencies, e.g., USD,NZD
    }

    # Make the API request
    response = requests.get(BASE_URL, params=params)

    # Debugging output
    print("Request URL:", response.url)
    print("Status Code:", response.status_code)
    print("Response Text:", response.text)

    # Check if the request was successful
    if response.status_code != 200:
        print("Error fetching data:", response.json())
        return

    # Parse the JSON response
    data = response.json()

    # Validate the response
    if not data.get("rates"):
        print("Error in response data:", data)
        return

    # Extract and process data
    rates = data["rates"]
    timestamp = data.get("timestamp", None)

    # Convert timestamp to a timezone-aware UTC date
    date = (
        datetime.fromtimestamp(timestamp, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
        if timestamp else "Unknown"
    )

    df = pd.DataFrame(rates.items(), columns=["Currency", "Rate"])
    df["Base"] = base_currency
    df["Date"] = date

    # Save the data to a CSV file with a consistent name
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, "nzdusd_raw.csv")  # Ensure compatibility with preprocessing
    df.to_csv(file_path, index=False)
    print(f"Data saved to {file_path}")

if __name__ == "__main__":
    # Example usage
    fetch_live_data("NZD/USD", "data/raw")  # Updated to match the required symbol format
