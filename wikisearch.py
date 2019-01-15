import wikipedia as wk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import difflib

wk.set_lang('en')

question_list = [
    "Do you know about",
    # "What about",
    "Try to find about"
]

threshold = 0.5


def preprocess_sentence(ps, sentence):
    tokens = word_tokenize(sentence)
    stems = [ps.stem(w) for w in tokens]
    sentence_in_stems = ""
    for stem in stems:
        sentence_in_stems += " " + stem
    return stems, sentence_in_stems


def wikipedia_search(word):
    ps = PorterStemmer()
    question_inits = [preprocess_sentence(ps, q) for q in question_list]
    word_init, word_init_sentence = preprocess_sentence(ps, word)
    perform_search = False
    search_array = []
    for q in question_inits:
        if difflib.SequenceMatcher(None, word_init_sentence, q[1]).ratio() > threshold:
            for one_word in q[0][::-1]:
                if one_word in word_init:
                    ind = word_init.index(one_word)
                    search_array = word_init[ind+1:]
                    perform_search = True
                    break
        if perform_search:
            break
    if not perform_search or not search_array:
        return ""

    search_string = ""
    for elem in search_array:
        search_string += " " + elem
    results = wk.search(search_string)
    if results:
        page = wk.page(results[0])
        msg = page.title + "\n" + page.summary
    else:
        msg = ""
    return msg
