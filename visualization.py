import matplotlib.pyplot as plt
import numpy as np

def create_word_distrib(dominant_topic):
    doc_lens = [len(d) for d in dominant_topic.Text]

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