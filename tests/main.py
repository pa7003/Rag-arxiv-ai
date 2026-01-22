import argparse
from src.logging_config import setup_logging
from src.data_loader import load_dataset
from src.preprocessing import preprocess_dataframe

def main():
    setup_logging()

    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, required=True)
    args = parser.parse_args()

    print("Run via notebook for full pipeline demo.")

if __name__ == "__main__":
    main()
