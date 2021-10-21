import re
import string
from collections import Counter
from itertools import chain
import json

import nltk
import pandas as pd
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
from pandas.core.series import Series


def is_word_in(word:str, series:pd.Series):
    return word in [item for sublist in series for item in sublist]

def join_sarcasm_tag(words:list):
    if not all(x in words for x in ['/', 's']):
        return words
    joined_words = []
    words_iter = enumerate(words)
    for index, word in words_iter:
        if index + 1 < len(words) and word == '/' and words[index + 1] == 's':
            joined_words.append('/s')
            next(words_iter)
        else:
            joined_words.append(word)
    return joined_words

def get_counter(series):
  flat_list = [item for sublist in series for item in sublist]
  c = Counter(flat_list)
  return c

def get_wordnet_pos(POS):
    """Map POS tag to first character lemmatize() accepts"""
    tag = POS[0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(POS, wordnet.NOUN)

def preprocess_text(sentences:Series, stem:bool, lemm:bool):

    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('wordnet')

    sentences = sentences.apply(str.lower)

    sentences = sentences.apply(lambda x: x.replace('\\s', '/s'))

    # Tokenization
    sentences = sentences.apply(nltk.tokenize.TweetTokenizer(strip_handles=True, reduce_len=True).tokenize)

    # Some tokens still contains whitespace, split those
    sentences = sentences.apply(lambda lst: [tk for tk in chain.from_iterable(token.split(' ') for token in lst) if tk != ''])

    # Standardize dates
    dates_re = r"(\b(0?[1-9]|[12]\d|30|31)[^\w\d\r\n:](0?[1-9]|1[0-2])[^\w\d\r\n:](\d{4}|\d{2})\b)|(\b(0?[1-9]|1[0-2])[^\w\d\r\n:](0?[1-9]|[12]\d|30|31)[^\w\d\r\n:](\d{4}|\d{2})\b)"
    sentences = sentences.apply(lambda x: [token if not re.findall(dates_re, token) else '$DATE$' for token in x])

    emoticon_re = r"(\:\w+\:|\<[\/\\]?3|[\(\)\\\D|\*\$][\-\^]?[\:\;\=]|[\:\;\=B8][\-\^]?[3DOPp\@\$\*\\\)\(\/\|])(?=\s|[\!\.\?]|$)"

    # Split token with *delimiter*
    delimiters = "-", "_"
    regexPattern = '|'.join(map(re.escape, delimiters))
    sentences = sentences.apply(lambda lst: [tk for tk in chain.from_iterable(re.split(regexPattern, token) if not re.findall(emoticon_re, token) else token for token in lst) if tk != ''])

    # Remove digits
    sentences = sentences.apply(lambda lst: [new_token for new_token in (''.join(c for c in token if not c.isdigit()) for token in lst) if new_token != ''])

    # Reconstruct reddit /s tag
    sentences = sentences.apply(join_sarcasm_tag)

    # Remove not negation stopwords
    s_word = ['y', "that'll", 'off', "a", 's', 't', 'can', 'will', 'just', "about", "above", "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'ma', "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "could", "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from", "further", "had", "has", "have", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "it", "it's", "its", "itself", "let's", "me", "more", "most", "my", "myself", "nor", "of", "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "she", "she'd", "she'll", "she's", "should", "so", "some", "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "we", "we'd", "we'll", "we're", "we've", "were", "what", "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", "would", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves" ]
    neg_s_word = ['not', 'don', "don't", 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]
    sentences = sentences.apply(lambda x: [item for item in x if item not in s_word])

    # Remove punctuation
    punctuation = string.punctuation
    sentences = sentences.apply(lambda x: [item for item in x if ''.join(sorted(set(item), key=item.index)) not in punctuation or re.findall(emoticon_re, item)])

    # Contruct POS tagging
    from nltk.corpus import wordnet

    wordlist = {item for sublist in sentences for item in sublist}
    taggedwords = {word : tag for word, tag in nltk.pos_tag(wordlist)}

    if lemm:
        # Lemmatization
        wnl = WordNetLemmatizer()
        sentences = sentences.apply(lambda comment: [wnl.lemmatize(word, get_wordnet_pos(taggedwords[word])) for word in comment])

    def apply_negation(tokens:list, neg_s_word:list) -> list:
        for i, tk in enumerate(tokens):
            original_tk = tk.replace('NOT_', '')
            if i + 1 < len(tokens) and original_tk in neg_s_word:
                tokens[i + 1] = 'NOT_' + tokens[i + 1]
                tokens[i] = ''
        tokens = [tk for tk in tokens if tk != '']
        return tokens

    # Reduce negation tokens
    sentences = sentences.apply(apply_negation, neg_s_word=neg_s_word)

    # Remove one char token
    sentences = sentences.apply(lambda x: [item for item in x if len(item) != 1])

    if stem:
        # Stemmization
        ls = SnowballStemmer(language='english')
        sentences = sentences.apply(lambda comment: [ls.stem(word) for word in comment])

    return sentences

def encode_subreddit(subs:Series, json_encoding:str):
    with open(json_encoding, 'r') as file:
        dictionary = json.load(file)
    subs = subs.apply(str.lower).apply(lambda x: dictionary.get(x, 0))
    return subs