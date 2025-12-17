import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
import warnings, os

# Set the data file first
script_dir = os.path.dirname(os.path.abspath(__file__))
data_file = os.path.join(script_dir, "../data/bugs_preprocessed.csv")

# Load the preprocessed data
data = pd.read_csv(data_file)

# prepare features and target
X = data['text_processed']
y = data['severity_category']

# train-test split
X_train_text