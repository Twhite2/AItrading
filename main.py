from scripts.data_collection import fetch_live_data
from scripts.data_preprocessing import preprocess_data
from scripts.decision_maker import make_decision

def main():
    # Step 1: Fetch live data
    fetch_live_data("NZD/USD", "data/raw")

    # Step 2: Preprocess data
    preprocess_data("data/raw/nzdusd_raw.csv", "data/processed")

    # Step 3: Make trading decision
    input_file = "data/processed/nzdusd_processed.csv"
    decision = make_decision("data/processed/nzdusd_processed.csv", "mrzlab630/lora-alpaca-trading-candles")
    print("Trading Decision:", decision)

if __name__ == "__main__":
    main()
