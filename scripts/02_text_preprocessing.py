#!/usr/bin/env python
"""
Text Preprocessing Script

Preprocesses bug report text: cleaning, POS-aware lemmatization, feature engineering.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import logging
import time
from utils.config import (
    CLEANED_DATA_PATH,
    PREPROCESSED_DATA_PATH,
    CUSTOM_STOPWORDS,
    USE_POS_LEMMATIZATION
)
from data.loader import load_raw_data, save_data
from data.preprocessor import TextPreprocessor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main execution function."""
    logger.info("="*70)
    logger.info("TEXT PREPROCESSING")
    logger.info("="*70)
    
    # Load cleaned data
    logger.info(f"\nLoading data from {CLEANED_DATA_PATH}")
    data = load_raw_data(CLEANED_DATA_PATH)
    
    # Initialize preprocessor
    logger.info(f"\nInitializing preprocessor...")
    logger.info(f"  POS-aware lemmatization: {USE_POS_LEMMATIZATION}")
    logger.info(f"  Custom stopwords: {len(CUSTOM_STOPWORDS)}")
    
    preprocessor = TextPreprocessor(
        custom_stopwords=list(CUSTOM_STOPWORDS),
        use_pos_lemmatization=USE_POS_LEMMATIZATION
    )
    
    # Show stopword summary
    summary = preprocessor.get_stopword_summary()
    logger.info(f"\nStopword Configuration:")
    logger.info(f"  Total stopwords: {summary['total_stopwords']}")
    logger.info(f"  Sample: {summary['sample_stopwords'][:10]}...")
    
    # Preprocess
    start_time = time.time()
    data_processed = preprocessor.preprocess_dataframe(data, show_progress=True)
    elapsed_time = time.time() - start_time
    
    # Save
    save_data(data_processed, PREPROCESSED_DATA_PATH)
    
    # Summary
    logger.info("\n" + "="*70)
    logger.info("PREPROCESSING SUMMARY")
    logger.info("="*70)
    logger.info(f"Input rows: {len(data)}")
    logger.info(f"Output rows: {len(data_processed)}")
    logger.info(f"Rows removed: {len(data) - len(data_processed)}")
    logger.info(f"Processing time: {elapsed_time/60:.2f} minutes")
    logger.info(f"\nText Length Statistics:")
    logger.info(f"  Mean: {data_processed['text_length'].mean():.1f} words")
    logger.info(f"  Median: {data_processed['text_length'].median():.1f} words")
    logger.info(f"  Min: {data_processed['text_length'].min():.0f} words")
    logger.info(f"  Max: {data_processed['text_length'].max():.0f} words")
    
    # Show sample
    logger.info(f"\nSample Preprocessing Results:")
    sample = data_processed.head(3)
    for idx, row in sample.iterrows():
        logger.info(f"\nBug {idx + 1}:")
        logger.info(f"  Original: {row['short_description'][:80]}...")
        logger.info(f"  Processed: {row['text_processed'][:80]}...")
        logger.info(f"  Length: {row['text_length']} words")
    
    logger.info("\n" + "="*70)
    logger.info("✅ TEXT PREPROCESSING COMPLETE")
    logger.info("="*70)
    logger.info(f"Preprocessed data saved to: {PREPROCESSED_DATA_PATH}")


if __name__ == "__main__":
    main()