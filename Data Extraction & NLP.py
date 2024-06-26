#!/usr/bin/env python
# coding: utf-8

# # Assignment Submitted by:
# # FARSANA

# # Data Extraction using BeautifulSoup

# In[1]:


import pandas as pd
import requests
from bs4 import BeautifulSoup
import os

def extract_article(url):
    # Send a GET request to the URL
    response = requests.get(url)
    
    # Check if the request was successful
    if response.status_code == 200:
        # Parse the HTML content of the webpage
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Find the title of the article
        title = soup.find('title').get_text().strip()
        
        # Find the main content of the article within div elements with the specified class
        article_content = ""
        div_elements = soup.find_all('div', class_='td-post-content tagdiv-type')
        for div in div_elements:
            # Extract text from each div element and append to article_content
            article_content += div.get_text().strip() + "\n\n"
        
        # Combine title and article content
        article_text = f"{title}\n\n{article_content}"
        
        return article_text, url
    else:
        print(f"Failed to fetch URL: {url}")
        return None, None

def save_article_to_file(article_text, filename):
    # Create a new directory to store text files if it doesn't exist
    if not os.path.exists('extracted_articles'):
        os.makedirs('extracted_articles')
    
    # Save the article text to a text file
    with open(f'extracted_articles/{filename}.txt', 'w', encoding='utf-8') as file:
        file.write(article_text)

# Read URLs from the Excel file and store them in a list
url_list = []
successful_urls = []
df = pd.read_excel('Input.xlsx')
for index, row in df.iterrows():
    url = row['URL']
    article_text, extracted_url = extract_article(url)
    if article_text:
        successful_urls.append(extracted_url)
        save_article_to_file(article_text, f"{url_id}")
        print(f"Article for URL ID {index} extracted and saved successfully.")
    else:
        print(f"Failed to extract article for URL ID {index}")

print("List of successful URLs:")
print(successful_urls)


# # Data Analysis

# ## Pre-processing the data

# ### Removing Stopwords

# In[1]:


import os

# Set the path to the folder containing the extracted articles
extracted_folder_path = "extracted_articles"

# Set the path to the folder containing stopwords
stopwords_folder_path = "Stopwords"

# Set the path to the folder where you want to save the processed text
processed_folder_path = "processed_articles"

# Function to load stopwords from files in the stopwords folder
def load_stopwords(stopwords_folder_path):
    stop_words = set()
    for file_name in os.listdir(stopwords_folder_path):

        if '.ipynb_checkpoints' in file_name:
            continue
        with open(os.path.join(stopwords_folder_path, file_name), 'r') as file:
            custom_stopwords = file.read().splitlines()
            stop_words.update(custom_stopwords)
    return stop_words

# Function to remove stopwords from a given text
def remove_stopwords(text, stop_words):
    tokens = text.split()
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
    return ' '.join(filtered_tokens)

# Load stopwords
stop_words = load_stopwords(stopwords_folder_path)

# Function to process each file in the extracted folder
def process_files(folder_path, processed_folder_path, stop_words):
    # Create processed folder if it does not exist
    if not os.path.exists(processed_folder_path):
        os.makedirs(processed_folder_path)
    
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        with open(file_path, 'r', encoding='utf-8') as file:  # Specify encoding as UTF-8
            text = file.read()
            processed_text = remove_stopwords(text, stop_words)
        # Write processed text to a new file in the processed folder
        processed_file_path = os.path.join(processed_folder_path, file_name)
        with open(processed_file_path, 'w', encoding='utf-8') as file:  # Specify encoding as UTF-8
            file.write(processed_text)

# Call process_files function to remove stopwords from all files in the extracted folder
process_files(extracted_folder_path, processed_folder_path, stop_words)


# In[2]:


import os

# Function to read words from a file and return them as a list
def read_words_from_file(file_path):
    words = []
    with open(file_path, 'r') as file:
        for line in file:
            # Remove any leading/trailing whitespace and add the word to the list
            words.append(line.strip())
    return words

# Directory paths for Stopwords and MasterDictionary folders
stopwords_dir = "Stopwords"
master_dict_dir = "MasterDictionary"

# Initialize lists to store stopwords and master dictionary words
stopwords = []
positive_words = []
negative_words = []

# Accessing files in the Stopwords folder
for URL_ID in os.listdir(stopwords_dir):
    file_path = os.path.join(stopwords_dir, URL_ID)
    if os.path.isfile(file_path):
        # Read words from the file and add them to the stopwords list
        stopwords.extend(read_words_from_file(file_path))

# Accessing files in the MasterDictionary folder
for URL_ID in os.listdir(master_dict_dir):
    file_path = os.path.join(master_dict_dir, URL_ID)
    if os.path.isfile(file_path):
        # Determine if the file contains positive or negative words based on its name
        if "positive" in URL_ID.lower():
            positive_words.extend(read_words_from_file(file_path))
        elif "negative" in URL_ID.lower():
            negative_words.extend(read_words_from_file(file_path))


# In[3]:


# Convert lists to sets for efficient comparison
stopwords_set = set(stopwords)
positive_set = set(positive_words)
negative_set = set(negative_words)

# Find words that are unique to each set
unique_stopwords = stopwords_set - positive_set - negative_set
unique_positive = positive_set - stopwords_set
unique_negative = negative_set - stopwords_set

# Convert sets back to lists
unique_stopwords_list = list(unique_stopwords)
unique_positive_list = list(unique_positive)
unique_negative_list = list(unique_negative)


# # Extracting Derived variables

# In[4]:


import os
import nltk
from nltk.tokenize import word_tokenize

# Directory path for processed articles
processed_articles_dir = "processed_articles"

# Initialize a dictionary to store tokens for each file
tokens_per_file = {}

# Accessing files in the processed_articles folder
for URL_ID in os.listdir(processed_articles_dir):
    file_path = os.path.join(processed_articles_dir, URL_ID)
    if os.path.isfile(file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
            tokens = word_tokenize(text)
            tokens_per_file[URL_ID] = tokens


# In[48]:


import pandas as pd

# Function to calculate positive score
def calculate_positive_score(tokens, positive_words):
    return sum(1 for token in tokens if token in positive_words)

# Function to calculate negative score
def calculate_negative_score(tokens, negative_words):
    return -1 * sum(1 for token in tokens if token in negative_words)

# Function to calculate polarity score
def calculate_polarity_score(positive_score, negative_score):
    return (positive_score - negative_score) / ((positive_score + negative_score) + 0.000001)

# Function to calculate subjectivity score
def calculate_subjectivity_score(positive_score, negative_score, total_words):
    return (positive_score + negative_score) / (total_words + 0.000001)

# Initialize lists to store results
results = []

# Iterate over each URL_ID and its corresponding list of tokens
for i, (URL_ID, tokens) in enumerate(tokens_per_file.items()):
    # Calculate positive score
    positive_score = calculate_positive_score(tokens, unique_positive_list)
    
    # Calculate negative score
    negative_score = calculate_negative_score(tokens, unique_negative_list)
    
    # Calculate polarity score
    polarity_score = calculate_polarity_score(positive_score, negative_score)
    
    # Calculate subjectivity score
    subjectivity_score = calculate_subjectivity_score(positive_score, negative_score, len(tokens))
    
    # Append results to the list
    url = successful_urls[i] if i < len(successful_urls) else ''
    results.append({
        "URL_ID": URL_ID.replace('.txt', ''),
        "URL": url,
        "Positive Score": positive_score,
        "Negative Score": negative_score,
        "Polarity Score": polarity_score,
        "Subjectivity Score": subjectivity_score
    })

# Convert results to pandas DataFrame for easy manipulation
results_df = pd.DataFrame(results)


# In[49]:


results_df


# ## Analysis of readability

# In[50]:


import os
import pandas as pd
import re
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords

# Function to count syllables in a word
def count_syllables(word):
    vowels = 'aeiouy'
    word = word.lower()
    count = 0
    if word[0] in vowels:
        count += 1
    for index in range(1, len(word)):
        if word[index] in vowels and word[index - 1] not in vowels:
            count += 1
    if word.endswith('e'):
        count -= 1
    if word.endswith('es') or word.endswith('ed'):
        count -= 1
    if count == 0:
        count = 1
    return count

# Navigate to the directory
directory = "processed_articles"

# Load input Excel file
input_df = pd.read_excel("Input.xlsx")  

# Define personal pronouns regex pattern
personal_pronouns_pattern = re.compile(r'\b(?:I|we|my|ours|us)\b', flags=re.IGNORECASE)

# Define stopwords
stop_words = set(stopwords.words('english'))

# Create an empty DataFrame to store the metrics
metrics_df = pd.DataFrame(columns=['URL_ID', 'Avg_Sentence_Length', 'Percentage_Complex_Words', 'Fog_Index',
                                    'Avg_Words_Per_Sentence', 'Complex_Word_Count', 'Word_Count', 
                                    'Syllable_Count_Per_Word', 'Personal_Pronouns', 'Average_Word_Length'])

for URL_ID in os.listdir(directory):
    if URL_ID.endswith(".txt"):  
        filepath = os.path.join(directory, URL_ID)
        with open(filepath, "r", encoding="utf-8") as file:
            text = file.read()
            # Tokenize sentences and words
            sentences = sent_tokenize(text)
            words = word_tokenize(text)
            # Clean words by removing stopwords and punctuation
            cleaned_words = [word.lower() for word in words if word.lower() not in stop_words and word.isalpha()]
            # Calculate metrics
            avg_sentence_length = len(words) / len(sentences)
            percentage_complex_words = sum(1 for word in cleaned_words if count_syllables(word) > 2) / len(cleaned_words)
            fog_index = 0.4 * (avg_sentence_length + percentage_complex_words)
            avg_words_per_sentence = len(words) / len(sentences)
            complex_word_count = sum(1 for word in cleaned_words if count_syllables(word) > 2)
            word_count = len(cleaned_words)
            syllable_count_per_word = sum(count_syllables(word) for word in cleaned_words) / len(cleaned_words)
            personal_pronouns_count = len(re.findall(personal_pronouns_pattern, text))
            average_word_length = sum(len(word) for word in cleaned_words) / len(cleaned_words)
            # Append metrics into the DataFrame
            metrics_df = metrics_df.append({'URL_ID': URL_ID.replace(".txt",""),
                                            'Avg_Sentence_Length': avg_sentence_length,
                                            'Percentage_Complex_Words': percentage_complex_words,
                                            'Fog_Index': fog_index,
                                            'Avg_Words_Per_Sentence': avg_words_per_sentence,
                                            'Complex_Word_Count': complex_word_count,
                                            'Word_Count': word_count,
                                            'Syllable_Count_Per_Word': syllable_count_per_word,
                                            'Personal_Pronouns': personal_pronouns_count,
                                            'Average_Word_Length': average_word_length},
                                           ignore_index=True)

# Merge the metrics DataFrame with the existing results DataFrame based on the 'URL_ID' column
final_df = pd.merge(results_df, metrics_df, on='URL_ID', how='outer')
# final_df.drop(columns=["URL_ID"], inplace=True)

"""
op_df = pd.merge(final_df, input_df, on='URL_ID', how='outer')"""
"""# Save the merged DataFrame to a CSV file
final_df.to_csv("merged_results.csv", index=False)"""


# In[51]:


final_df


# In[52]:


# Save the merged DataFrame to a CSV file
final_df.to_csv("Output Data.csv", index=False)

