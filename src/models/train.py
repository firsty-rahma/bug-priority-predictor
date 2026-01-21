"""
Model training utilities.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report, make_scorer
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import logging
from pathlib import Path
from typing import Dict, Tuple, Any
import pickle
import sys
import warnings

# Add src to path for imports
src_path = Path(__file__).parent.parent
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from utils.custom_transformers import FeatureCombiner

logger = logging.getLogger(__name__)

class ModelTrainer:
    """Train and evaluate bug severity classification models."""

    def __init__(self, random_state: int = 42):
        """
        Initialize ModelTrainer.
        
        Parameters
        ----------
        random_state : int
            Random seed for reproducibility
        """
        self.random_state = random_state
        self.label_encoder = LabelEncoder()
        self.best_model = None
        self.best_params = None
        self.best_score = None
    
    def prepare_data(self, X:pd.DataFrame, y:pd.Series, test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
        """
        Prepare data for training: encode labels and split.
        
        Parameters
        ----------
        X : pd.DataFrame
            Features
        y : pd.Series
            Target variable (severity categories)
        test_size : float
            Proportion of data for testing
            
        Returns
        -------
        tuple
            (X_train, X_test, y_train, y_test)
        """
        logger.info("Preparing data for training")

        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        logger.info(f"Label encoding: {dict(enumerate(self.label_encoder.classes_))}")

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded,
            test_size=test_size,
            random_state=self.random_state,
            stratify=y_encoded
        )

        logger.info(f"Train set: {len(X_train)} samples")
        logger.info(f"Test set: {len(X_test)} samples")
        
        return X_train, X_test, y_train, y_test
    
    def create_pipeline(self, feature_combiner:FeatureCombiner, classifier:Any, 
                        use_smote: bool = True, smote_neighbors = 3) -> ImbPipeline:
        """
        Create sklearn pipeline with optional SMOTE.
        
        Parameters
        ----------
        feature_combiner : FeatureCombiner
            Custom feature engineering transformer
        classifier : sklearn estimator
            Classification model
        use_smote : bool
            Whether to use SMOTE oversampling
            
        Returns
        -------
        Pipeline
            Configured pipeline
        """
        steps = [('feature_combiner', feature_combiner)]
        if use_smote:
            steps.append(('smote', SMOTE(random_state=self.random_state, k_neighbors=smote_neighbors)))

        steps.append(('classifier', classifier))
        return ImbPipeline(steps)
    
    def train_with_cv(
        self,
        pipeline: ImbPipeline,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        cv_folds: int = 5
    ) -> Dict[str, float]:
        """
        Train model with cross-validation.
        
        Parameters
        ----------
        pipeline : ImbPipeline
            Training pipeline
        X_train : pd.DataFrame
            Training features
        y_train : np.ndarray
            Training labels
        cv_folds : int
            Number of CV folds
            
        Returns
        -------
        dict
            Cross-validation scores
        """
        from sklearn.model_selection import cross_val_score
        
        logger.info(f"Training with {cv_folds}-fold cross-validation")
        
        cv = StratifiedKFold(
            n_splits=cv_folds,
            shuffle=True,
            random_state=self.random_state
        )
        
        # F1-macro score (better for imbalanced data)
        scores = cross_val_score(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring='f1_macro',
            n_jobs=-1
        )
        
        results = {
            'mean': scores.mean(),
            'std': scores.std(),
            'scores': scores
        }
        
        logger.info(f"CV F1-Macro: {results['mean']:.4f} (±{results['std']:.4f})")
        
        return results
    
    def hyperparameter_tuning(self, pipeline: ImbPipeline, param_grid: Dict[str, Any],
                              X_train: pd.DataFrame, y_train: np.ndarray, 
                              cv_folds: int = 5) -> GridSearchCV:
        """
        Perform hyperparameter tuning with grid search.
        
        Parameters
        ----------
        pipeline : ImbPipeline
            Training pipeline
        param_grid : dict
            Parameter grid for search
        X_train : pd.DataFrame
            Training features
        y_train : np.ndarray
            Training labels
        cv_folds : int
            Number of CV folds
            
        Returns
        -------
        GridSearchCV
            Fitted grid search object
        """
        logger.info("Starting hyperparameter tuning")
        logger.info(f"Parameter grid: {param_grid}")

        cv = StratifiedKFold(
            n_splits=cv_folds,
            shuffle=True,
            random_state=self.random_state
        )

        grid_search = GridSearchCV(pipeline, param_grid, cv=cv,
                                   scoring=make_scorer(f1_score, average='macro'),
                                   n_jobs=-1, verbose=1, return_train_score=True, error_score='raise')
        grid_search.fit(X_train, y_train)

        self.best_model = grid_search.best_estimator_
        self.best_params = grid_search.best_params_
        self.best_score = grid_search.best_score_

        logger.info(f"Best parameters: {self.best_params}")
        logger.info(f"Best CV F1-Macro: {self.best_score:.4f}")

        return grid_search
    
    def evaluate(self, model: Any, X_test: pd.DataFrame, y_test: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate model on test set.
        
        Parameters
        ----------
        model : estimator
            Trained model
        X_test : pd.DataFrame
            Test features
        y_test : np.ndarray
            Test labels (encoded)
            
        Returns
        -------
        dict
            Evaluation metrics
        """
        logger.info("Evaluating model on test set")

        # Predict
        y_pred = model.predict(X_test)

        # Decode labels
        y_test_decoded = self.label_encoder.inverse_transform(y_test)
        y_pred_decoded = self.label_encoder.inverse_transform(y_pred)

        # Calculate metrics
        f1_macro = f1_score(y_test_decoded, y_pred_decoded, average='macro')
        f1_weighted = f1_score(y_test_decoded, y_pred_decoded, average='weighted')

        logger.info(f"Test F1-Macro: {f1_macro:.4f}")
        logger.info(f"Test F1-Weighted: {f1_weighted:.4f}")

        # Classification report
        report = classification_report(
            y_test_decoded,
            y_pred_decoded,
            output_dict=True
        )
        
        results = {
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'classification_report': report,
            'y_test': y_test_decoded,
            'y_pred': y_pred_decoded
        }
        
        return results
    
    def save_model(self, model: Any, filepath: Path, 
                   include_encoder: bool = True) -> None:
        """
        Save trained model to disk.
        
        Parameters
        ----------
        model : estimator
            Trained model
        filepath : Path
            Output file path
        include_encoder : bool
            Whether to include label encoder
        """
        filepath.parent.mkdir(parents=True, exist_ok=True)

        if include_encoder:
            model_data = {
                'model': model,
                'label_encoder': self.label_encoder,
                'label_mapping': {i: label for i, label in enumerate(self.label_encoder.classes_)}
            }
        else:
            model_data = model
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"✅ Model saved to: {filepath}")
    
    @staticmethod
    def load_model(filepath: Path) -> Dict[str, Any]:
        """
        Load trained model from disk.
        
        Parameters
        ----------
        filepath : Path
            Model file path
            
        Returns
        -------
        dict or model
            Loaded model (and encoder if saved together)
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found {filepath}")
        
        # CRITICAL FIX: Import FeatureCombiner before unpickling
        # This ensures pickle can find the class definition
        from utils.custom_transformers import FeatureCombiner

        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)

        logger.info(f"✅ Model loaded from: {filepath}")
        return model_data