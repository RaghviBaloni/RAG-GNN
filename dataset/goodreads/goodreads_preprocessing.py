import os
import json
import pickle
import re
import collections
import random
#from bedrock import Bedrock

import boto3
# import config
from collections import Counter
from scipy.sparse import lil_matrix
from scipy.sparse import csr_matrix
import numpy as np 
# import nltk

#from bedrock import Bedrock
# nltk.download('stopwords')
# from nltk.corpus import stopwords
import urllib, urllib.request
#!pip3 install matplotlib
from tqdm import tqdm
# import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np

def generate_markdown_link(header):
    # Removing the '#' and any leading/trailing spaces from the header
    clean_header = header.strip('#').strip()

    # Generate the anchor link by replacing spaces with hyphens, converting to lowercase,
    # and handling special characters like underscores appropriately
    anchor_link = clean_header.replace(' ', '-').replace('_', '-')
    anchor_link = ''.join(char.lower() for char in anchor_link if char.isalnum() or char in '-')

    # Form the markdown link
    markdown_link = f"[{clean_header}](#{anchor_link})"
    return markdown_link

# Examples
example1 = "## canonical edge types for fastgtn"



print(generate_markdown_link(example1))

import json
from tqdm import tqdm

# File paths
input_file_path = 'goodsreads_data/goodreads_books.json'  # Adjust this path as needed

data = []
target_keys = ['similar_books', 'description', 'authors', 'publisher', 'book_id', 'title_without_series', 'title', 'format', 'popular_shelves', 'edition_information', 'series', 'language_code', 'country_code', 'edition_information']

try:
    # Read the entire JSON file
    with open(input_file_path, 'r') as file:
        full_data = json.load(file)

    # Process each record
    for record in tqdm(full_data):
        tmp_clean = {k: record[k] for k in target_keys if k in record}
        data.append(tmp_clean)

except json.JSONDecodeError as e:
    print(f"Error decoding JSON: {e}")
except Exception as e:
    print(f"An error occurred: {e}")

# read book genres
book_genres_dict = {}

with open('goodsreads_data/goodreads_book_genres_initial.json') as f:
    readin = f.readlines()
    for line in tqdm(readin):
        tmp = json.loads(line)
        assert tmp['book_id'] not in book_genres_dict
        book_genres_dict[tmp['book_id']] = [k for k in tmp['genres']]

print(len(book_genres_dict))

filtered_book_genres_dict = {}
for book_id in book_genres_dict.keys():
    if (len(book_genres_dict[book_id])==1):
        filtered_book_genres_dict[book_id] = book_genres_dict[book_id][0]

set(list(filtered_book_genres_dict.values()))

filter_data_v2 = []
for struct in data:
    if struct['book_id'] in filtered_book_genres_dict:
        filter_data_v2.append(struct)

print (len(filter_data_v2))

filter_data_v5= [book for book in filter_data_v2 if book['description'] != '']
print (len(filter_data_v5))

# add genres into filter_data

for i in tqdm(range(len(filter_data_v5))):
    assert filter_data_v5[i]['book_id'] in filtered_book_genres_dict
    filter_data_v5[i]['genres'] = filtered_book_genres_dict[filter_data_v5[i]['book_id']]

genres=[book['genres'] for book in filter_data_v5]
print (Counter(genres))

import random

def sample_books_by_genre(book_list, total_sample_size, genre_ratios):
    # Categorize books by genre
    genre_categories = {genre: [] for genre in genre_ratios}
    for book in book_list:
        genre = book['genres']
        if genre in genre_categories:
            genre_categories[genre].append(book)

    # Calculate number of samples per genre
    total_ratios = sum(genre_ratios.values())
    genre_samples = {genre: int(total_sample_size * (count / total_ratios)) for genre, count in genre_ratios.items()}

    # Sample books from each genre
    sampled_books = []
    for genre, books in genre_categories.items():
        sampled_books.extend(random.sample(books, min(genre_samples[genre], len(books))))

    return sampled_books


genre_ratios = {'non-fiction': 1200, 'fiction': 1200, 'romance': 1200}
filter_data_v6 = sample_books_by_genre(filter_data_v5, 3600, genre_ratios)

len(filter_data_v6)

random.shuffle(filter_data_v6)

def clean_similar_books(book_list):
    # Extract all book IDs from the list of dictionaries
    all_book_ids = {book['book_id'] for book in book_list}

    # Iterate through each book and clean up the similar_books list
    for book in book_list:
        # Filter out similar_book IDs that are not in all_book_ids
        book['similar_books'] = [id for id in book['similar_books'] if id in all_book_ids]
#         book['popular_shelves'] = [shelf for shelf in book['popular_shelves'] if int(shelf['count']) > 1]

    return book_list


cleaned_books = clean_similar_books(filter_data_v6)

# save the first stage processed data
pickle.dump(cleaned_books, open('./dataset/goodreads/books_filtered.pkl','wb'))

