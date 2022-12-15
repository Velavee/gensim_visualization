# https://www.machinelearningplus.com/nlp/topic-modeling-visualization-how-to-present-results-lda-models/
import sys
import re, numpy as np, pandas as pd
from pprint import pprint

# Gensim
import gensim, spacy, logging, warnings
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
import matplotlib.pyplot as plt

# NLTK Stop words
from nltk.corpus import stopwords
stop_words = stopwords.words('english')

warnings.filterwarnings('ignore',category=DeprecationWarning)
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

def sent_to_words(titles):
    for title in titles:
        title = re.sub('\S*@\S*\s?', '', title)  # remove emails
        title = re.sub('\s+', ' ', title)  # remove newline chars
        title = re.sub("\'", "", title)  # remove single quotes
        title = gensim.utils.simple_preprocess(str(title), deacc=True) 
        yield(title)

def format_topics_sentences(ldamodel, corpus, texts):
    sent_topics_df = pd.DataFrame()

    # Get main topic in each document
    for i, row_list in enumerate(ldamodel[corpus]):
        row = row_list[0] if ldamodel.per_word_topics else row_list
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # Get the dominant topic, Perc Contribution, and keywords for each doc
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0: # => dominant topic
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
            else:
                break
            sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

            # Add original text to the end of the output
            contents = pd.Series(texts)
            sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
            return(sent_topics_df)


df = pd.read_json('testomatTests.json')
print(df.shape)
df.head()

data = df.title.values.tolist()
data_words = list(sent_to_words(data))

# Build the bigram and trigram models
bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100)
trigram = gensim.models.Phrases(bigram[data_words], threshold=100)
bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)

def process_words(texts, stop_words=stop_words, allowed_postags=['NOUN','ADJ','VERB','ADV']):
    texts = [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]
    texts = [bigram_mod[doc] for doc in texts]
    texts = [trigram_mod[bigram_mod[doc]] for doc in texts]
    texts_out = []
    nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
    for title in texts:
        doc = nlp(" ".join(title))
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    texts_out = [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]
    return texts_out

data_ready = process_words(data_words)

# Create Dictionary
id2word = corpora.Dictionary(data_ready)

# Create Corpus: Term Document Frequency
corpus = [id2word.doc2bow(text) for text in data_ready]

# Build LDA model
lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                            id2word=id2word,
                                            num_topics=4,
                                            random_state=100,
                                            update_every=1,
                                            chunksize=10,
                                            passes=10,
                                            alpha='symmetric',
                                            iterations=100,
                                            per_word_topics=True)

pprint(lda_model.print_topics())

df_topic_sents_keywords = format_topics_sentences(ldamodel=lda_model, corpus=corpus, texts=data_ready)

# Format
df_dominant_topic = df_topic_sents_keywords.reset_index()
df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']
df_dominant_topic.head(10)

doc_lens = [len(d) for d in df_dominant_topic.Text]

# Plot
plt.figure(figsize=(16,7), dpi=160)
plt.hist(doc_lens, bins=10, color='navy')
plt.text(40, 200, 'Mean    : ' + str(round(np.mean(doc_lens))))
plt.text(40, 190, 'Median   : ' + str(round(np.median(doc_lens))))
plt.text(40, 180, 'Stdev    : ' + str(round(np.std(doc_lens))))
plt.text(40, 170, '1%ile    : ' + str(round(np.std(doc_lens))))
plt.text(40, 160, '99%ile   : ' + str(round(np.quantile(doc_lens, q=0.99))))

plt.gca().set(xlim=(0, 10), ylabel='Number of Documents', xlabel='Document Word Count')
plt.tick_params(size=16)
plt.xticks(np.linspace(0,50,11))
plt.title('Distribution of Document Word Counts', fontdict=dict(size=22))
plt.savefig('word_count.png')
