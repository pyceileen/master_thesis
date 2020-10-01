# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 11:11:55 2020

@author: Pei-yuChen
"""
#%%
import unidecode
from word2number import w2n
import re  
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer 
#from nltk.corpus import stopwords
# Expand contractions
from util.dictionaries import contractions_dict # import my own script


# =============================================================================
# stopwords_list = stopwords.words('english')
# deselect_stop_words = ['again','against','any','all','can','down','further',
#                         'have','just','no','nor','not','off','on','only',
#                         'out','over','such','should','up','very','will',
#                         'd','m','o','s','t','y']
# for i in deselect_stop_words:
#     stopwords_list.remove(i)
#      #print(stopwords_list)
# =============================================================================

stopwords_list = ['the', 'a', 'an']    
#%% prepare cleaning functions

# Remove HTML tags (no need for this dataset)
    
# Lowercase all texts
def lower_case(doc):
    doc = doc.lower()    
    return doc

# Convert Accented Characters (no need, already did in excel)
def remove_accented(doc):
    """
    remove accented characters e.g. “” ‘’, should be straight "" ''
    """
    doc = unidecode.unidecode(doc)
    return doc    

# Expand contractions
def expand_contractions(doc):
    """
    expand shortened words, e.g. don't -> do not
    """
    # add whitespace in front of special characters, so they don't affect expanding
    pat = re.compile(r"([()!?:,.\"])")
    doc = pat.sub(" \\1 ", doc)
    
    for w in doc.split():
        if w in contractions_dict:
            doc = doc.replace(w, contractions_dict[w])
    return " ".join(doc.split())

# Remove stopwords
def remove_stopwords(doc):
    doc = [w for w in word_tokenize(doc) if not w in stopwords_list]
    doc = " ".join(doc)
    return doc

# Lemmatization
def lemmatization(doc):
    lemmatizer = WordNetLemmatizer() 
    doc = [lemmatizer.lemmatize(w) for w in word_tokenize(doc)] 
    doc = " ".join(doc)
    return doc

# Convert number words to numeric form, remove with special character 
def convert_num(doc):
    s = doc.split()
    #s = word_tokenize(doc)
    temp = []
    for w in s:
        try:
            temp += [str(w2n.word_to_num(w))]
        except ValueError:
            temp += [w] 
    doc = ' '.join(temp)
    return doc

def fix_ps(doc):
    doc = doc.replace('p . s .', 'ps')
    
    dictionary = {"jan":"january", "feb":"february", 
                  "mar":"march", "apr":"april", 
                  "may":"may", "jun":"june", 
                  "jul":"july", "aug":"august", 
                  "sep":"september", "oct":"october", 
                  "nov":"november", "dec":"december", } 

    for word, initial in dictionary.items():
        doc = doc.replace(word, initial)    
    return doc

# Remove special characters, punctuation
def remove_special(doc):
    doc = re.sub("[^a-zA-Zé ]", "",doc) # remove non a-z, A-Z
    return doc

# Remove extra white spaces   
def remove_whitespace(doc):
    doc = doc.strip() # remove head and tail extra space
    return " ".join(doc.split()) # remove extra space in the middle of sentence

#%%
def text_wrapper(doc, lowercase=True, accented=False, 
                       contractions=True, ps=True, num=True, 
                       special=True, 
                       whitespace=True, stopwords=True, lemma=False):
    """ 
    Adaptable functions
    """
    if lowercase == True: 
        cleaned_doc = lower_case(doc)
    if accented == True:
        cleaned_doc = remove_accented(cleaned_doc)
    if contractions == True:
        cleaned_doc = expand_contractions(cleaned_doc)
    if stopwords == True: 
        cleaned_doc = remove_stopwords(cleaned_doc)
    if lemma == True: 
        cleaned_doc = lemmatization(cleaned_doc)
    if ps == True:
        cleaned_doc = fix_ps(cleaned_doc)
    if num == True: 
        cleaned_doc = convert_num(cleaned_doc)
    if special == True:
        cleaned_doc = remove_special(cleaned_doc)
    if whitespace == True: 
        cleaned_doc = remove_whitespace(cleaned_doc)
    
    return(cleaned_doc)
    

'''
for neural network: no lemma
for svm, lemma
'''

def text_preprocessing(doc, sent_tok=False):
    if sent_tok==True:
        filtered_sentences = []
        for sentence in sent_tokenize(doc):
            clean = text_wrapper(sentence)
            filtered_sentences.append(clean)
        return (filtered_sentences)
    else:
        clean = text_wrapper(doc)
        return (clean)












