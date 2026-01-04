#!/usr/bin/env python
# coding: utf-8

# In[1]:


import scipy as sp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import re


# In[2]:


# Download required NLTK data
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')  # Open Multilingual WordNet
nltk.download('averaged_perceptron_tagger_eng')  # For POS tagging (optional but recommended)


# In[3]:


# Load the data
data = pd.read_csv('../data/bugs_cleaned.csv')
print(f"Columns: {data.shape[1]}")
print(f"Shape: {data.shape}")
data.head()


# In[4]:


# Verify missing values
print("Missing values:")
data.isnull().sum()


# In[5]:


# Check duplicate data
data.duplicated(subset = ['short_description', 'long_description']).sum()


# In[6]:


# Check duplicated data
data[data.duplicated(subset = ['short_description', 'long_description'])].iloc[:2]


# In[7]:


# Since the component_name columns have different values in the duplicated table, 
# I think I can leave it as it is
# Ok, let's check deeper the missing value at the short_description and long_description

# Check the row with empty value/zero or more spaces/empty string on the short_description
short_conditional = data['short_description'].isna() | data['short_description'].str.contains(r'^\s*$', na=False) | (data['short_description'] == '')
data[short_conditional]


# In[8]:


# Get the index
short_conditional_index = data[short_conditional].index.tolist()
print(short_conditional_index)


# In[9]:


# Check rows with empty value/zero or more spaces/empty string on the long_description
long_conditional = data['long_description'].isna() | data['long_description'].str.contains(r'^\s*$', na=False) | (data['long_description'] == '')
data[long_conditional]


# In[10]:


# Get the index
long_conditional_index = data[long_conditional].index.tolist()
print(long_conditional_index)


# In[11]:


# Check if there are rows with empty values in both long_description and short_description
both_empty_description_index = list(set(short_conditional_index) & set(long_conditional_index))
print(both_empty_description_index)


# In[12]:


# Then, I'll drop the row with the index
data_cleaned = data.drop(both_empty_description_index[0])
data_cleaned


# In[13]:


# Before I clean the text data, I will combine those two columns first
# (and I will fill null values at the long_description column with an empty string)

data_cleaned['text_combined'] = data_cleaned['short_description'] + ' ' + data_cleaned['long_description'].fillna('')
data_cleaned['text_combined']


# In[14]:


# lower all text data
text_combined_lower = data_cleaned['text_combined'].str.lower()
text_combined_lower


# In[15]:


# remove special characters and split the words
text_tokens = text_combined_lower.str.replace(r'[^a-z\s]','', regex=True).str.split()
text_tokens


# In[16]:


# Word frequency analysis
# Overall word frequency
all_words = [word for tokens in text_tokens for word in tokens]
word_freq_raw = Counter(all_words)
print(f"\nTotal unique words: {len(word_freq_raw)}")
print(f"\nTotal words: {len(all_words)}")
print(f"\nTop 20 most common words:")
for word, count in word_freq_raw.most_common(30):
    print(f"{word:15s} : {count:5d}")


# In[17]:


# Word frequency by severity category
print("Word frequency by severity category (Top 10):")
for severity in data_cleaned['severity_category'].unique():
    severity_mask = data_cleaned['severity_category'] == severity
    severity_words = [word for idx, tokens in text_tokens[severity_mask].items() for word in tokens]
    severity_freq = Counter(severity_words)
    print(f"\n{severity.upper()} (n={severity_mask.sum()}):")
    for word, count in severity_freq.most_common(10):
        print(f"{word:15s} : {count:5d}")


# In[18]:


# Remove stopwords (since so many stopwords there)
import string

stop_words = set(stopwords.words('english'))
print(f"Standard English stopwords: {len(stop_words)}")

# Since there are some special characters in the stopwords
# I will clean those words
cleaned_stop_words = {re.sub(r'[^a-z]','',s) for s in stop_words}

# Custom stopwords
custom_stopwords = {'always', 'firefox', 'mozilla', 'gecko', 'bugzilla','bug', 'bugs', 'os', 'com', 
                    'http', 'window', 'windows', 'use', 'uses', 'used', 'using','build', 'builds', 'built', 'building',
                    'line', 'lines', 'result', 'results', 'resulting', 'resulted', 'file', 'files', 'filed', 'enus', 
                    'page', 'pages','xa', 'nt', 'rv', 'new', 'news', 'newly', 'newer', 'reproduce', 'reproduced', 'reproducing', 'reproduces', 
                    'see', 'saw', 'sees', 'seeing', 'seen', 'step', 'steps', 'stepped', 'stepping','reproducible', 'reproducibly', 
                    'report', 'reported', 'reports', 'reporting', 'reporter', 'testcases', 'testcase', 'expect', 'expected', 'expects',
                    'today', 'yesterday', 'current', 'currently', 'latest', 'recent', 'recently', 'get', 'gets', 'got', 'gotten', 'getting',
                    'try', 'tries','tried', 'trying', 'open', 'opens', 'opened', 'opening', 'close', 'closes', 'closed', 'closing', 
                    'click','clicks', 'clicked', 'clicking'}

# Since I found a single alphabet in the processed text after testing,
# I added this function to add all alphabet in lowercase
lowercase_alphabet_stopwords = string.ascii_lowercase
custom_stopwords.update(lowercase_alphabet_stopwords)

print(f"\nCustom domain-specific stopwords being added:")
print(f"  {sorted(custom_stopwords)}")
cleaned_stop_words.update(custom_stopwords)

print(f"\nTotal stopwords after adding custom: {len(cleaned_stop_words)}")


# In[19]:


# Removal process
text_tokens_no_stop = text_tokens.apply(lambda x: [word for word in x if word not in cleaned_stop_words])


# In[20]:


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


# In[21]:


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


# In[22]:


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


# In[23]:


# Analyze word frequency by severity category
severity_word_freq = {}
severity_totals = {}
for severity in data_cleaned['severity_category'].unique():
    severity_mask = data_cleaned['severity_category'] == severity
    severity_words = [word for idx, tokens in text_tokens_lemmatized_pos[severity_mask].items() for word in tokens]
    severity_word_freq[severity] = Counter(severity_words)
    severity_totals[severity] = len(severity_words)


# In[24]:


# Total word number after lemmatized
overall_total = sum(word_freq_lemmatized.values())


# In[25]:


# Counting word distribution by severity
print("\nWord distribution by severity:")
for severity in sorted(severity_totals.keys()):
    count = data_cleaned[data_cleaned['severity_category'] == severity].shape[0]
    words = severity_totals[severity]
    avg_words = words / count if count > 0 else 0
    print(f"  {severity:12s}: {count:4d} bugs, {words:6d} total words, {avg_words:.1f} avg words/bug")


# In[26]:


# Find distinctive words for each severity
# Find the ratio between words that appear significantly more in a cateory
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



# In[27]:


# Find words that are common across all categories
common_across_all = []
for word in word_freq_lemmatized.most_common(50):
    word_name = word[0]
    # Check if word appears in top 30 of all categories
    appears_in_all = all(
        word_name in [w[0] for w in severity_word_freq[sev].most_common(30)]
        for sev in data_cleaned['severity_category'].unique()
    )
    if appears_in_all:
        common_across_all.append(word_name)

print(f"\nWords appearing frequently in all severity categories:")
print(f"(These are candidates for custom stopwords)")
for word in common_across_all[:15]:
    print(f"  - {word}")

# I will re-run script after this


# In[28]:


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


# In[29]:


# Make a viz
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Text length distribution by severity
sn.boxplot(data=data_cleaned, x='severity_category', y='text_length_processed', ax=axes[0, 0])
axes[0, 0].set_title('Text Length Distribution by Severity Category', fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel('Severity Category')
axes[0, 0].set_ylabel('Number of Words (After Processing)')
axes[0, 0].tick_params(axis='x', rotation=45)

# Top 20 words (all severities)
top_words = word_freq_lemmatized.most_common(20)
words, counts = zip(*top_words)
axes[0, 1].barh(range(len(words)), counts, color='steelblue')
axes[0, 1].set_yticks(range(len(words)))
axes[0, 1].set_yticklabels(words)
axes[0, 1].set_xlabel('Frequency')
axes[0, 1].set_title('Top 20 Words After Processing', fontsize=14, fontweight='bold')
axes[0, 1].invert_yaxis()

# Compare text length in raw data and processed data
axes[1, 0].scatter(data_cleaned['text_length_raw'], 
                   data_cleaned['text_length_processed'], 
                   alpha=0.3, s=10)
axes[1, 0].plot([0, data_cleaned['text_length_raw'].max()], 
                [0, data_cleaned['text_length_raw'].max()], 
                'r--', label='y=x')
axes[1, 0].set_xlabel('Original Text Length')
axes[1, 0].set_ylabel('Processed Text Length')
axes[1, 0].set_title('Text Length: Before vs After Processing', fontsize=14, fontweight='bold')
axes[1, 0].legend()

# Severity category distribution
severity_counts = data_cleaned['severity_category'].value_counts().sort_index()
axes[1, 1].bar(range(len(severity_counts)), severity_counts.values, color='coral')
axes[1, 1].set_xticks(range(len(severity_counts)))
axes[1, 1].set_xticklabels(severity_counts.index, rotation=45)
axes[1, 1].set_ylabel('Count')
axes[1, 1].set_title('Severity Category Distribution', fontsize=14, fontweight='bold')
# Add count labels on bars
for i, v in enumerate(severity_counts.values):
    axes[1, 1].text(i, v + 50, str(v), ha='center', va='bottom')

plt.tight_layout()
plt.savefig('../figures/text_analysis_complete.png', dpi=300, bbox_inches='tight')
print("Visualization saved: ../figures/text_analysis_complete.png")
plt.show()


# In[30]:


# Additional: Distinctive words visualization for top 3 severities
fig2, axes2 = plt.subplots(1, 3, figsize=(18, 6))

top_severities = ['critical', 'major', 'blocker']
for idx, severity in enumerate(top_severities):
    if severity in distinctive_results and len(distinctive_results[severity]) > 0:
        top_distinctive = distinctive_results[severity][:15]
        if top_distinctive:
            words_dist, counts_dist, ratios_dist = zip(*top_distinctive)

            axes2[idx].barh(range(len(words_dist)), ratios_dist, color='indianred')
            axes2[idx].set_yticks(range(len(words_dist)))
            axes2[idx].set_yticklabels(words_dist)
            axes2[idx].set_xlabel('Distinctiveness Score')
            axes2[idx].set_title(f'Distinctive Words: {severity.upper()}', fontsize=12, fontweight='bold')
            axes2[idx].invert_yaxis()

plt.tight_layout()
plt.savefig('../figures/distinctive_words.png', dpi=300, bbox_inches='tight')
print("Visualization saved: ../figures/distinctive_words.png")
plt.show()


# In[31]:


# Convert tokens to strings
data_cleaned['text_processed'] = text_tokens_lemmatized_pos.apply(
    lambda x: ' '.join(x) if isinstance(x, list) and len(x) > 0 else ''
)

# Check before cleaning
print(f"\nBefore cleaning:")
print(f"  Total rows: {len(data_cleaned)}")
print(f"  NaN in text_processed: {data_cleaned['text_processed'].isna().sum()}")

# Check rows that would be dropped
problematic_mask = (
    data_cleaned['text_processed'].isna() | 
    (data_cleaned['text_processed'].str.strip() == '')
)

if problematic_mask.sum() > 0:
    print(f"\n  ⚠️ Found {problematic_mask.sum()} problematic rows:")
    print("\n  Sample problematic rows:")
    print(data_cleaned[problematic_mask][['bug_id', 'short_description', 'long_description', 'text_processed']].head())

    # Show their severity distribution
    print("\n  Severity distribution of dropped rows:")
    print(data_cleaned[problematic_mask]['severity_category'].value_counts())

# Drop rows with empty or NaN text_processed
data_cleaned = data_cleaned[
    data_cleaned['text_processed'].notna() & 
    (data_cleaned['text_processed'].str.strip() != '')
].copy()

print(f"\nAfter cleaning:")
print(f"  Total rows: {len(data_cleaned)}")
print(f"  Rows dropped: {problematic_mask.sum()}")
print(f"  NaN in text_processed: {data_cleaned['text_processed'].isna().sum()}")
print(f"  Empty strings: {(data_cleaned['text_processed'] == '').sum()}")

# Verify no issues remain
assert data_cleaned['text_processed'].isna().sum() == 0, "Still have NaN!"
assert (data_cleaned['text_processed'] == '').sum() == 0, "Still have empty strings!"
print("\n  ✅ All checks passed!")

# Save cleaned data
data_cleaned.to_csv('../data/bugs_preprocessed.csv', index=False)
print(f"\nSaved: ../data/bugs_preprocessed.csv ({len(data_cleaned)} rows)")

# Save word frequencies for reference
freq_df = pd.DataFrame([
    {'word': word, 'frequency': count, 'rank': idx+1}
    for idx, (word, count) in enumerate(word_freq_lemmatized.most_common(500))
])
freq_df.to_csv('../data/word_frequencies.csv', index=False)
print(f"Saved word frequencies: ../data/word_frequencies.csv")

print("\n" + "="*60)
print("TEXT PREPROCESSING COMPLETE!")
print("="*60)


# In[32]:


# ============================================================
# SUMMARY
# ============================================================
print("\n" + "="*60)
print("WEEK 2 TEXT PREPROCESSING - COMPLETE!")
print("="*60)
print("\nSummary:")
print(f"  • Original data: {data_cleaned.shape[0]+1} rows")
print(f"  • After cleaning: {data_cleaned.shape[0]} rows")
print(f"  • Total unique words (raw): {len(word_freq_raw)}")
print(f"  • After stopword removal: {len(word_freq_no_stop)}")
print(f"  • After lemmatization: {len(word_freq_lemmatized)}")
print(f"  • Custom stopwords added: {len(custom_stopwords)}")
print(f"  • Word reduction: {len(word_freq_raw) - len(word_freq_lemmatized)} ({((len(word_freq_raw) - len(word_freq_lemmatized))/len(word_freq_raw)*100):.1f}%)")

print("\nOutput files:")
print("  ✓ ../data/bugs_preprocessed.csv")
print("  ✓ ../data/word_frequencies.csv")
print("  ✓ ../figures/text_analysis_complete.png")
print("  ✓ ../figures/distinctive_words.png")


# In[ ]:




