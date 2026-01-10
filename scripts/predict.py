
#!/usr/bin/env python
"""
Prediction Script

Makes predictions on new bug reports using trained model.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd
import logging
import argparse

from utils.config import MODEL_DIR, PREPROCESSED_DATA_PATH
from models.train import ModelTrainer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def predict_single_bug(short_desc: str, long_desc: str, 
                       component: str, product: str, model_path: Path) -> dict:
    """
    Predict severity for a single bug report.
    
    Parameters
    ----------
    short_desc : str
        Short description
    long_desc : str
        Long description
    component : str
        Component name
    product : str
        Product name
    model_path : Path
        Path to trained model
        
    Returns
    -------
    dict
        Prediction results
    """
    # Load model
    model_data = ModelTrainer.load_model(model_path)
    model = model_data['model']
    label_encoder = model_data['label_encoder']

    # Preprocess text (simplified - in production, use TextPreprocessor)
    from data.preprocessor import TextPreprocessor
    preprocessor = TextPreprocessor()

    combined_text = preprocessor.combine_text(short_desc, long_desc)
    processed_text = preprocessor.preprocess(combined_text)
    text_length = len(processed_text.split())

    # Create DataFrame
    bug_df = pd.DataFrame({
        'text_processed': [processed_text],
        'component_name': [component],
        'product_name': [product],
        'text_length': [text_length]
    })
    
    # Predict
    prediction_encoded = model.predict(bug_df)[0]
    prediction_proba = model.predict_proba(bug_df)[0]
    
    # Decode
    prediction = label_encoder.inverse_transform([prediction_encoded])[0]
    confidence = prediction_proba.max()
    
    # Get all class probabilities
    all_probs = {
        label: prob
        for label, prob in zip(label_encoder.classes_, prediction_proba)
    }
    
    return {
        'predicted_severity': prediction,
        'confidence': confidence,
        'all_probabilities': all_probs,
        'processed_text_length': text_length
    }

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Predict bug severity')
    parser.add_argument('--short-desc', type=str, help='Short description')
    parser.add_argument('--long-desc', type=str, default='', help='Long description')
    parser.add_argument('--component', type=str, default='General', help='Component name')
    parser.add_argument('--product', type=str, default='FIREFOX', help='Product name')
    parser.add_argument('--model', type=str, default=None, help='Model path')

    args = parser.parse_args()

    # Determine model path
    if args.model:
        model_path = Path(args.model)
    else:
        model_path = MODEL_DIR / "best_model_random_forest_tuned.pkl"
    
    if not model_path.exists():
        logger.error(f"Model not found: {model_path}")
        return
    
    # Interactive mode if no description provided
    if not args.short_desc:
        logger.info("Interactive Prediction Mode")
        logger.info("="*70)
        
        short_desc = input("Short description: ")
        long_desc = input("Long description (optional): ")
        component = input("Component (default: General): ") or "General"
        product = input("Product (default: FIREFOX): ") or "FIREFOX"
    else:
        short_desc = args.short_desc
        long_desc = args.long_desc
        component = args.component
        product = args.product
    
    # Predict
    logger.info("\nMaking prediction...")
    result = predict_single_bug(short_desc, long_desc, component, product, model_path)
    
    # Display results
    logger.info("\n" + "="*70)
    logger.info("PREDICTION RESULTS")
    logger.info("="*70)
    logger.info(f"Predicted Severity: {result['predicted_severity'].upper()}")
    logger.info(f"Confidence: {result['confidence']:.2%}")
    logger.info(f"\nAll Class Probabilities:")
    
    sorted_probs = sorted(
        result['all_probabilities'].items(),
        key=lambda x: x[1],
        reverse=True
    )
    
    for label, prob in sorted_probs:
        bar = '█' * int(prob * 50)
        logger.info(f"  {label:12s}: {prob:.2%} {bar}")
    
    # Recommendation
    if result['confidence'] < 0.6:
        logger.warning("\n⚠️  Low confidence - recommend human review")
    
    if result['predicted_severity'] in ['blocker', 'critical']:
        logger.warning("⚠️  High severity - always verify with human review")


if __name__ == "__main__":
    main()

    
