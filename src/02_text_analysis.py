import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import re

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')  # Open Multilingual WordNet
nltk.download('averaged_perceptron_tagger_eng')  # For POS tagging

# Load the data
data = pd.read_csv('../data/bugs_cleaned.csv')
print(f"Columns: {data.shape[1]}")
print(f"Shape: {data.shape}")
data.head()


# Verify missing values
print("Missing values:")
data.isnull().sum()

# Check duplicate data
data.duplicated(subset = ['short_description', 'long_description']).sum()

# Check duplicated data
data[data.duplicated(subset = ['short_description', 'long_description'])].iloc[:2]

# Since the component_name columns have different values in the duplicated table, 
# I think I can leave it as it is
# Ok, let's check deeper the missing value at the short_description and long_description

# Check the row with empty value/zero or more spaces/empty string on the short_description
short_conditional = data['short_description'].isna() | data['short_description'].str.contains(r'^\s*$', na=False) | (data['short_description'] == '')
data[short_conditional]

# Get the index
short_conditional_index = data[short_conditional].index.tolist()

# Check rows with empty value/zero or more spaces/empty string on the long_description
long_conditional = data['long_description'].isna() | data['long_description'].str.contains(r'^\s*$', na=False) | (data['long_description'] == '')

# Get the index
long_conditional_index = data[long_conditional].index.tolist()

# Check if there are rows with empty values in both long_description and short_description
both_empty_description_index = list(set(short_conditional_index) & set(long_conditional_index))

# Then, I'll drop the row with the index (just an index)
data_cleaned = data.drop(both_empty_description_index[0])

# Before I clean the text data, I will combine those two columns first
# (and I will fill null values at the long_description column with an empty string)

data_cleaned['text_combined'] = data_cleaned['short_description'] + ' ' + data_cleaned['long_description'].fillna('')

# lower all text data
text_combined_lower = data_cleaned['text_combined'].str.lower()

# remove special characters and split the words
text_tokens = text_combined_lower.str.replace(r'[^a-z\s]','', regex=True).str.split()
text_tokens

# Word frequency analysis
# Overall word frequency
all_words = [word for tokens in text_tokens for word in tokens]
word_freq_raw = Counter(all_words)
print(f"\nTotal unique words: {len(word_freq_raw)}")
print(f"\nTotal words: {len(all_words)}")
print(f"\nTop 20 most common words:")
for word, count in word_freq_raw.most_common(30):
    print(f"{word:15s} : {count:5d}")

# Word frequency by severity category
print("Word frequency by severity category (Top 10):")
for severity in data_cleaned['severity_category'].unique():
    severity_mask = data_cleaned['severity_category'] == severity
    severity_words = [word for idx, tokens in text_tokens[severity_mask].items() for word in tokens]
    severity_freq = Counter(severity_words)
    print(f"\n{severity.upper()} (n={severity_mask.sum()}):")
    for word, count in severity_freq.most_common(10):
        print(f"{word:15s} : {count:5d}")

# Remove stopwords (since so many stopwords there)
stop_words = set(stopwords.words('english'))
print(f"Standard English stopwords: {len(stop_words)}")

# Since there are some special characters in the stopwords
# I will clean those words
cleaned_stop_words = {re.sub(r'[^a-z]','',s) for s in stop_words}

# Custom stopwords
custom_stopwords = {'O', 'always', 'firefox', 'mozilla', 'gecko', 'bugzilla','bug', 'issue', 'error', 'problem', 'http', 'r', 'o', 'u', 'v', 'x', 'result', 'window', 'windows', 'use', 'uses', 'build', 'line', 'result', 'results', 'file', 'file', 'enus', 'page'}

print(f"\nCustom domain-specific stopwords being added:")
print(f"  {sorted(custom_stopwords)}")
cleaned_stop_words.update(custom_stopwords)

stop_words.update(custom_stopwords)
print(f"\nTotal stopwords after adding custom: {len(stop_words)}")

# Removal process from stopwords
text_tokens_no_stop = text_tokens.apply(lambda x: [word for word in x if word not in cleaned_stop_words])

# Word frequency after stopword removal
all_words_no_stop = [word for tokens in text_tokens_no_stop for word in tokens]
word_freq_no_stop = Counter(all_words_no_stop)

print(f"\nAfter stopword removal:")
print(f"Total unique words: {len(word_freq_no_stop)}")
print(f"Total words: {len(all_words_no_stop)}")
print(f"Reduction: {len(all_words) - len(all_words_no_stop)} words removed")
print(f"\nTop 20 most common words (after stopword removal):")
for word, count in word_freq_no_stop.most_common(30):
    print(f"{word:15s} : {count:5d}")

# Helper function to convert NLTK POS tags to WordNet POS tags
# I need this since it can make a lemmatization for verb, noun, etc.

def get_wordnet_pos(treebank_tag):
    """Convert Treebank POS tag to WordNet POS tag"""
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN  # Default to noun

# Advanced lemmatization with POS tagging
def lemmatize_with_pos(tokens):
    """Lemmatize tokens using POS tags for better accuracy"""
    lemmatizer = WordNetLemmatizer()

    # Get POS tags
    pos_tags = nltk.pos_tag(tokens)

    # Lemmatize with appropriate POS
    lemmatized = [
        lemmatizer.lemmatize(word, get_wordnet_pos(pos)) 
        for word, pos in pos_tags
    ]
    return lemmatized

# Apply advanced lemmatization
text_tokens_lemmatized_pos = text_tokens_no_stop.apply(lemmatize_with_pos)

# Word frequency after lemmatization
all_words_lemmatized = [word for tokens in text_tokens_lemmatized_pos for word in tokens]
word_freq_lemmatized = Counter(all_words_lemmatized)

print(f"\nAfter lemmatization:")
print(f"Total unique words: {len(word_freq_lemmatized)}")
print(f"Total words: {len(all_words_lemmatized)}")
print(f"Reduction: {len(word_freq_no_stop) - len(word_freq_lemmatized)} unique words merged")
print(f"\nTop 20 most common words (after lemmatization):")
for word, count in word_freq_lemmatized.most_common(20):
    print(f"  {word:15s} : {count:5d}")

# Analyze word frequency by severity category after lemmatization
severity_word_freq = {}
severity_totals = {}
for severity in data_cleaned['severity_category'].unique():
    severity_mask = data_cleaned['severity_category'] == severity
    severity_words = [word for idx, tokens in text_tokens_lemmatized_pos[severity_mask].items() for word in tokens]
    severity_word_freq[severity] = Counter(severity_words)
    severity_totals[severity] = len(severity_words)

# Total word number after lemmatized
overall_total = sum(word_freq_lemmatized.values())

# Counting word distribution by severity
print("\nWord distribution by severity:")
for severity in sorted(severity_totals.keys()):
    count = data_cleaned[data_cleaned['severity_category'] == severity].shape[0]
    words = severity_totals[severity]
    avg_words = words / count if count > 0 else 0
    print(f"  {severity:12s}: {count:4d} bugs, {words:6d} total words, {avg_words:.1f} avg words/bug")

# Find distinctive words for each severity
# Find the ratio between words that appear significantly more in a category
# and overall
distinctive_results = {}
for severity in sorted(data_cleaned['severity_category'].unique()):
    print(f"\n{severity.upper()}:")

    severity_total = severity_totals[severity]
    distinctive_words = []
    for word, count in severity_word_freq[severity].most_common(100):
        if count >= 5:# Minimum threshold
            # Calculate relative frequency
            severity_ratio = count / severity_total if severity_total > 0 else 0
            overall_ratio = word_freq_lemmatized[word] / overall_total if overall_total > 0 else 0
            distinctiveness = severity_ratio / overall_ratio if overall_ratio > 0 else 0

            # Only include words that are notably more common in this category
            if distinctiveness > 1.5:
                distinctive_words.append((word, count, distinctiveness))
    # Sort by distinctiveness score
    distinctive_words.sort(key = lambda x: x[2], reverse = True)
    distinctive_results[severity] = distinctive_words
    print(f"  Top 15 distinctive words:")
    for i, (word, count, ratio) in enumerate(distinctive_words[:15], 1):
        print(f"    {i:2d}. {word:15s} : count={count:4d}, distinctiveness={ratio:.2f}x")

    if len(distinctive_words) == 0:
        print("    (No highly distinctive words found)")

# Text length analysis and describe the text length statistically
data_cleaned['text_length_raw'] = text_tokens.apply(len)
data_cleaned['text_length_processed'] = text_tokens_lemmatized_pos.apply(len)

print("\nText length by severity category:")
length_stats = data_cleaned.groupby('severity_category')[['text_length_raw', 'text_length_processed']].describe()
print(length_stats)

print("\nText length reduction by preprocessing:")
total_reduction = data_cleaned['text_length_raw'].sum() - data_cleaned['text_length_processed'].sum()
reduction_pct = (total_reduction / data_cleaned['text_length_raw'].sum()) * 100
print(f"Total words removed: {total_reduction} ({reduction_pct:.1f}%)")


# Save tokens as space-separated strings
data_cleaned['text_processed'] = text_tokens_lemmatized_pos.apply(lambda x: ' '.join(x))

# Save preprocessed data
data_cleaned.to_csv('../data/bugs_preprocessed.csv', index=False)
print(f"Saved preprocessed data: ../data/bugs_preprocessed.csv")
print(f"Shape: {data_cleaned.shape}")
print(f"Columns: {list(data_cleaned.columns)}")

# Save word frequencies for reference
freq_df = pd.DataFrame([
    {'word': word, 'frequency': count, 'rank': idx+1}
    for idx, (word, count) in enumerate(word_freq_lemmatized.most_common(500))
])
freq_df.to_csv('../data/word_frequencies.csv', index=False)
print(f"Saved word frequencies: ../data/word_frequencies.csv")

# Summary
print("\nSummary:")
print(f"  • Original data: {data_cleaned.shape[0]+1} rows")
print(f"  • After cleaning: {data_cleaned.shape[0]} rows")
print(f"  • Total unique words (raw): {len(word_freq_raw)}")
print(f"  • After stopword removal: {len(word_freq_no_stop)}")
print(f"  • After lemmatization: {len(word_freq_lemmatized)}")
print(f"  • Custom stopwords added: {len(custom_stopwords)}")
print(f"  • Word reduction: {len(word_freq_raw) - len(word_freq_lemmatized)} ({((len(word_freq_raw) - len(word_freq_lemmatized))/len(word_freq_raw)*100):.1f}%)")

