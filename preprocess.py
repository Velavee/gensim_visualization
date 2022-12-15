import re
import gensim
from gensim.utils import simple_preprocess

def sent_to_words(titles):
    for title in titles:
        title = re.sub('\S*@\S*\s?', '', title)  # remove emails
        title = re.sub('\s+', ' ', title)  # remove newline chars
        title = re.sub("\'", "", title)  # remove single quotes
        title = gensim.utils.simple_preprocess(str(title), deacc=True) 
        yield(title)