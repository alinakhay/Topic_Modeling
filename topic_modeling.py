"""
Using this same process you can also train a classifier for sentiment analysis, with the sentiment tags included in the dataset that we didn't use in this tutorial.

For topic modeling we will use Gensim.


We pick up halfway through the classifier tutorial. We leave our text as a list of words, since Gensim accepts that as input.
Then, we create a Gensim dictionary from the data using the bag of words model:
"""

from gensim import corpora
texts = [process_text(text) for text in texts]
dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]

# After that, we're ready to go. It's important to note that here we're just using the review texts, and not the topics that come with the dataset. Using this dictionary, we train an LDA model, instructing Gensim to find three topics in the data:

from gensim import models
model = models.ldamodel.LdaModel(corpus, num_topics=3, id2word=dictionary, passes=15)

topics = model.print_topics(num_words=3)
for topic in topics:
    print(topic)


# And that's it! The code will print out the mixture of the most representative words for three topics:

# (0, '0.038*"bathroom" + 0.024*"shower" + 0.022*"clean"')
# (1, '0.104*"room" + 0.021*"bed" + 0.019*"small"')
# (2, '0.031*"bed" + 0.014*"air" + 0.014*"noisi"')

# Interestingly, the algorithm identified words that look a lot like keywords for our original Facilities, Comfort and Cleanliness topics.