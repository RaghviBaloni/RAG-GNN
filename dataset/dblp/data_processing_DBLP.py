# %%
import os
import json
import pickle
import random
from copy import deepcopy
from tqdm import tqdm
import ijson
from collections import defaultdict
import json
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
from scipy.sparse import csr_matrix
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
from decimal import Decimal
import pandas as pd

from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth, association_rules

# %%
os.system("pip3 install mlxtend")

# %% [markdown]
# ## Read DBLP

# %%
# with open('./dblp.v12.json') as f:
#     data = json.load(f)
os.system("pip3 install ijson")

# %%
def abstract_formating(inverted_index):

    flattened = [(word, idx) for word, indices in inverted_index.items() for idx in indices]

    # Sort by index
    flattened.sort(key=lambda x: x[1])

    # Reconstruct the sentence
    abstract = ""
    for word, index in flattened:
        if abstract and not abstract[-1].isalnum():
            abstract += word
        else:
            abstract += (" " if abstract else "") + word

    return abstract

# %%
filename = "dblp.v12.json"
simp_data = {}

no_ref = 0
no_fos = 0
no_authors = 0
no_ven = 0
no_abs = 0

with open(filename, 'r') as file:
    # Assuming the JSON is an array of objects
    objects = ijson.items(file, 'item')
    for p in tqdm(objects):
        # Process each object here
        tmp_dict = {}
        tmp_dict['title'] = p['title']
        tmp_dict['year'] = p['year']
        flag = 1
        if 'authors' in p:
            tmp_dict['authors'] = p['authors']  
            
        else:
            no_authors = no_authors + 1
            flag = 0
    
        if 'references' in p:
            tmp_dict['references'] = p['references']  
        else:
            no_ref = no_ref + 1
            flag = 0

        if 'fos' in p:
            tmp_dict['fos'] = p['fos']  
        else:
            no_fos = no_fos + 1
            flag = 0

        if 'venue' in p:
            tmp_dict['venue'] = p['venue']  
        else:
            no_ven = no_ven + 1
            flag = 0
        
        if 'indexed_abstract' in p:
            abstract_dict= p['indexed_abstract']['InvertedIndex']
            abstract=abstract_formating(abstract_dict)
            tmp_dict['abstract'] = abstract
        else:
            no_abs = no_abs + 1
            flag = 0
        if (flag):    
            simp_data[p['id']] = tmp_dict


# %%
# # Apply the Apriori algorithm
# frequent_itemsets = apriori(df, min_support=0.9, use_colnames=True)

# # Generate the association rules
# rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)

# # Print the results
# print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])

# %%
machine_learning_labels = set(['Artificial intelligence', 'Machine learning', 'Computer vision', 
                               'Natural language processing'])
networking_security_labels = set(['Computer network'])
theoretical_cs_labels = set(['Theoretical computer science'])

# Initialize paper label as None
paper_label = None


for key,item in simp_data.items():
    fos_names = {fos['name'] for fos in item['fos']}
    if fos_names & machine_learning_labels:
        paper_label = 'machine_learning'
        item['paper_label'] = paper_label
    elif fos_names & networking_security_labels:
        paper_label = 'computer_networking'
        item['paper_label'] = paper_label
    elif fos_names & theoretical_cs_labels:
        paper_label = 'theoretical_computer_science'
        item['paper_label'] = paper_label
    else:
        item['paper_label'] = None

# %%
#8373
#6522

simp_data[8763]

# %%


for key,value in simp_data.items():
    value['id'] = key
    simp_data[key] = value
    
simp_data_2 = [value for value in simp_data.values()]

# %%
paper_labels_list = []
for key,value in simp_data.items():
    paper_labels_list.append(value['paper_label'])

# %%
from collections import Counter
Counter(paper_labels_list)

# %%
simp_data[1688]

# %%
import random

def sample_books_by_genre(book_list, total_sample_size, genre_ratios,class_label):
    # Categorize books by genre
    genre_categories = {genre: [] for genre in genre_ratios}
    for book in book_list:
        genre = book[class_label]
        if genre in genre_categories:
            genre_categories[genre].append(book)

    # Calculate number of samples per genre
    total_ratios = sum(genre_ratios.values())
    genre_samples = {genre: int(total_sample_size * (count / total_ratios)) for genre, count in genre_ratios.items()}

    # Sample books from each genre
    sampled_books = []
    random.seed(123)
    for genre, books in genre_categories.items():
        sampled_books.extend(random.sample(books, min(genre_samples[genre], len(books))))

    return sampled_books


genre_ratios = {'machine_learning': 2000, 'computer_networking': 2000, 'theoretical_computer_science': 2000}
filter_data_v6 = sample_books_by_genre(simp_data_2, 6000, genre_ratios,'paper_label')

# %%
random.seed(123)
random.shuffle(filter_data_v6)

# %%
def clean_similar_books(book_list,similar_tag,id_label):
    # Extract all book IDs from the list of dictionaries
    all_book_ids = {book[id_label] for book in book_list}

    # Iterate through each book and clean up the similar_books list
    for book in book_list:
        # Filter out similar_book IDs that are not in all_book_ids
        book[similar_tag] = [id for id in book[similar_tag] if id in all_book_ids]
#         book['popular_shelves'] = [shelf for shelf in book['popular_shelves'] if int(shelf['count']) > 1]

    return book_list


cleaned_books = clean_similar_books(filter_data_v6,'references','id')


# %%
pickle.dump(cleaned_books, open('dblp_v2.pkl','wb'))
filter_data_v7 = pickle.load(open('dblp_v2.pkl','rb'))


