"""
Interactive prediction script for bug severity classification.
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

# CRITICAL: Import FeatureCombiner BEFORE loading model
from utils.custom_transformers import FeatureCombiner
from models.train import ModelTrainer
from data.preprocessor import TextPreprocessor


def print_prediction_results(prediction: str, probabilities: np.ndarray, label_encoder):
    """
    Pretty print prediction results.
    
    Parameters
    ----------
    prediction : str
        Predicted severity
    probabilities : np.ndarray
        Class probabilities
    label_encoder : LabelEncoder
        Label encoder for class names
    """
    print("\n" + "=" * 70)
    print("PREDICTION RESULTS")
    print("=" * 70)
    
    confidence = probabilities.max()
    
    print(f"\nPredicted Severity: {prediction.upper()}")
    print(f"Confidence: {confidence:.1%}")
    
    print("\nAll Class Probabilities:")
    
    # Sort by probability
    class_probs = list(zip(label_encoder.classes_, probabilities))
    class_probs.sort(key=lambda x: x[1], reverse=True)
    
    for class_name, prob in class_probs:
        bar_length = int(prob * 40)
        bar = "█" * bar_length
        print(f"  {class_name:12s}: {prob:6.2%} {bar}")
    
    # Warnings
    if prediction in ['blocker', 'critical']:
        print("\n⚠️  High severity - recommend human review")
    
    if confidence < 0.60:
        print("\n⚠️  Low confidence - consider manual triage")
    
    print("=" * 70 + "\n")


def get_user_input():
    """
    Get bug information from user.
    
    Returns
    -------
    dict
        Bug information
    """
    print("\n" + "=" * 70)
    print("Interactive Prediction Mode")
    print("=" * 70)
    
    short_desc = input("\nShort description: ").strip()
    long_desc = input("Long description: ").strip()
    component = input("Component (default: General): ").strip() or "General"
    product = input("Product (default: FIREFOX): ").strip() or "FIREFOX"
    
    return {
        'short_description': short_desc,
        'long_description': long_desc,
        'component_name': component,
        'product_name': product
    }


def predict_bug_severity(
    short_desc: str,
    long_desc: str,
    component: str = "General",
    product: str = "FIREFOX",
    model_path: Path = None,
    verbose: bool = True
):
    """
    Predict severity for a single bug.
    
    Parameters
    ----------
    short_desc : str
        Short bug description
    long_desc : str
        Long bug description
    component : str
        Component name
    product : str
        Product name
    model_path : Path, optional
        Path to trained model
    verbose : bool
        Whether to print results
        
    Returns
    -------
    tuple
        (prediction, confidence, all_probabilities)
    """
    # Default model path
    if model_path is None:
        model_path = Path(__file__).parent.parent / "models" / "best_model_random_forest_tuned.pkl"
    
    # Load model
    if verbose:
        print("\nLoading model...")
    
    model_data = ModelTrainer.load_model(model_path)
    model = model_data['model']
    label_encoder = model_data['label_encoder']
    
    # Preprocess text
    if verbose:
        print("Preprocessing text...")
    
    preprocessor = TextPreprocessor()
    combined_text = preprocessor.combine_text(short_desc, long_desc)
    text_processed = preprocessor.preprocess(combined_text)
    text_length = len(text_processed.split())
    
    # Create feature DataFrame
    bug_df = pd.DataFrame({
        'text_processed': [text_processed],
        'component_name': [component],
        'product_name': [product],
        'text_length': [text_length]
    })
    
    # Predict
    if verbose:
        print("Making prediction...\n")
    
    prediction_encoded = model.predict(bug_df)[0]
    probabilities = model.predict_proba(bug_df)[0]
    
    # Decode prediction
    prediction = label_encoder.inverse_transform([prediction_encoded])[0]
    confidence = probabilities.max()
    
    # Print results
    if verbose:
        print_prediction_results(prediction, probabilities, label_encoder)
    
    return prediction, confidence, probabilities


def main():
    """Main interactive prediction loop."""
    print("\n" + "=" * 70)
    print("Bug Severity Prediction System")
    print("=" * 70)
    print("\nThis system predicts bug severity using ML.")
    print("Enter bug details to get a severity prediction.\n")
    
    while True:
        try:
            # Get user input
            bug_info = get_user_input()
            
            # Predict
            predict_bug_severity(
                short_desc=bug_info['short_description'],
                long_desc=bug_info['long_description'],
                component=bug_info['component_name'],
                product=bug_info['product_name']
            )
            
            # Continue?
            again = input("\nPredict another bug? (y/n): ").strip().lower()
            if again not in ['y', 'yes']:
                break
                
        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except FileNotFoundError as e:
            print(f"\n❌ Error: {e}")
            print("Make sure the trained model exists in models/ folder")
            break
        except Exception as e:
            print(f"\n❌ Error during prediction: {e}")
            import traceback
            traceback.print_exc()
            break
    
    print("\nThank you for using the Bug Severity Prediction System!")


if __name__ == "__main__":
    main()