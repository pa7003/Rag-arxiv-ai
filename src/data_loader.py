import pandas as pd
import logging

logger = logging.getLogger(__name__)

def load_dataset(path):
    try:
        df = pd.read_csv(path)
        logger.info(f"Loaded dataset with {len(df)} rows")
        return df
    except Exception:
        logger.error("Dataset loading failed", exc_info=True)
        raise
