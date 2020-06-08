import string
from selectolax.parser import HTMLParser
from nltk.tokenize.regexp import regexp_tokenize
import pkg_resources
from symspellpy.symspellpy import SymSpell
import spacy
from nltk.corpus import stopwords


# pattern for text extraction, using by regexp_tokenize
pattern = r'''(?x)             # set flag to allow verbose regexps
         [a-zA-Z]+             # words only
'''

# load SymSpell dictionary for spelling correction
sym_spell = SymSpell(max_dictionary_edit_distance=0, prefix_length=7)
dictionary_path = pkg_resources.resource_filename("symspellpy", "frequency_dictionary_en_82_765.txt")
sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)

# load Spacy model object for lemmatization
nlp = spacy.load('en', parse=False, tag=False, entity=False)

# list of stopwords
cachedStopWords = stopwords.words("english") + list(string.ascii_lowercase) + ['-PRON-']

#####################################################################################################
# remove html syntax
def get_text_selectolax(html):
    tree = HTMLParser(html)

    if tree.body is None:
        return None

    for tag in tree.css('script'):
        tag.decompose()
    for tag in tree.css('style'):
        tag.decompose()

    text = tree.body.text(separator='\n')
    return text


#####################################################################################################
# extract ONLY words using regexp_tokenize
def extract_words(text):
    text_cleaned = regexp_tokenize(text, pattern=pattern)
    text_cleaned = " ".join(text_cleaned)
    return text_cleaned


#####################################################################################################
# correct spelling using SymSpell
def correct_spelling(input_term):
    return sym_spell.word_segmentation(input_term).corrected_string


#####################################################################################################
# lemmatization using Spacy
def lemmatize_text(text):
    text_cleaned = nlp(text)
    text_cleaned_list = [word.lemma_ for word in text_cleaned]
    return text_cleaned_list

#####################################################################################################
def clean_text(text):

    # removing HTML tags
    text_cleaned = get_text_selectolax(text)

    # covert everything to lower-case
    text_cleaned = text_cleaned.lower()

    # extract ONLY words
    text_cleaned = extract_words(text_cleaned )

    # correct spelling errors
    text_cleaned = correct_spelling(text_cleaned)

    # lemmatize
    text_cleaned = lemmatize_text(text_cleaned)

    # remove stopwords
    text_cleaned = ' '.join([word for word in text_cleaned if word not in cachedStopWords])

    return text_cleaned