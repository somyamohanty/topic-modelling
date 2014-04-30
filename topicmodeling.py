#!/usr/bin/env python

import scipy as sp
import csv
import datetime
# import nltk.stem
import gensim
import string
import os.path
import sys
from collections import Counter
from dateutil import parser
from pattern.vector import stem, PORTER, LEMMA
from operator import itemgetter

def stop_word_list():
    stoplist=set()
    for line in open('stopwords.txt','rb'):
        line=line.strip()
        stoplist.add(line.lower())
    return stoplist

def read_articles(f):
    print "Reading file: %s" % f
    
    articles = []

    with open(f, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter='\t')
        reader.next()
        for row in reader:
            if row != []:
                a = {}
                a["title"] = row[2]
                a["abstract"] = row[3]
                
                articles.append(a)
    
    print "Completed Read: %s" % f
    return articles

def process_articles(articles, stoplist):
    print "Cleaning Articles: Special Characters, Stemming, Stopwords"

    remove_list = string.ascii_letters + string.digits

    cleanArticles = []

    for a in articles:
        #html entities
        a = gensim.utils.decode_htmlentities(a)

        #Remove Unicode
        temp = a.decode("utf-8")
        temp = temp.encode('ascii', errors='ignore')

        #Split    
        temp = temp.split()
        cleanArticle = []
        for w in temp:
            if w in stoplist:
                continue        
        
            #Remove Special Chars
            w = ''.join([l for l in w if l in remove_list])
        
            #Lowercase
            w = w.lower()
            if w.startswith('rt'):
                continue
            if w != '':
                w = stem(w, stemmer=LEMMA)
                cleanArticle.append(w)


        cleanArticles.append(cleanArticle)

    print "Cleaned Articles"

    return cleanArticles

def process_articles_gensim(articles):
    print "Cleaning Articles: Special Characters, Stemming, Stopwords"

    cleanArticles = []

    for a in articles:
        cleanArticles.append(gensim.utils.lemmatize(a))

    print "Cleaned Articles"

    return cleanArticles

### END INPUT PROCESSING STUFF ###


### TOPIC MODELING ###
def put_in_dict_corpus(filename, documents):
    print "Rare words cleanup"

    texts = [[word for word in document] for document in documents]
    all_tokens = sum(texts, [])
    tokens_once = set(word for word in set(all_tokens) if all_tokens.count(word) == 1)
    texts = [[word for word in text if word not in tokens_once] for text in texts]

    print "Rare words cleanup done"

    print "Creating Dictionary"
    #create or load dictionary
    if not os.path.isfile(filename+'.dict'):
        dictionary = gensim.corpora.Dictionary(texts) 
        dictionary.compactify() 
        dictionary.save(filename+'.dict') 
        print "Created %s.dict" % (filename)
    else:
        dictionary = gensim.corpora.Dictionary.load(filename+'.dict')
        print "Loaded %s.dict" % (filename)

    print "Creating Corpus"

    #create or load corpus
    if not os.path.isfile(filename+'_corpus.mm'):
        corpus = [dictionary.doc2bow(text) for text in texts]
        gensim.corpora.MmCorpus.serialize(filename+'_corpus.mm', corpus)
    else:
        corpus = gensim.corpora.MmCorpus(filename+'_corpus.mm')
        print "Loaded %s_corpus.mm" % (filename)

    print "Created Dictionary and Corpus"
    return dictionary, corpus

def tf_idf(filename, corpus):

    print "Creating TF-IDF model and Corpus"

    if not os.path.isfile(filename+'.tfidf'):
        model = gensim.models.TfidfModel(corpus)
        model.save(filename+'.tfidf')
        print "Created %s.tfidf" % (filename)
    else:
        model = gensim.models.TfidfModel.load(filename+'.tfidf')
        print "Loaded %s.tfidf" % (filename)

    if not os.path.isfile(filename+'_corpus_tfidf.mm'):
        corpus_tfidf = model[corpus]
        gensim.corpora.MmCorpus.serialize(filename+'_corpus_tfidf.mm', corpus_tfidf)
    else:
        corpus_tfidf = gensim.corpora.MmCorpus(filename+'_corpus_tfidf.mm')
        print "Loaded %s_corpus_tfidf.mm" % (filename)

    print "Created/Loaded TF-IDF model and Corpus"

    return model, corpus_tfidf

def lda(filename, corpus, dictionary, no_topics):
    print "Creating LDA model"

    if not os.path.isfile(filename+'.lda'):
        model = gensim.models.ldamodel.LdaModel(corpus=corpus, num_topics= no_topics,id2word=dictionary, passes=200, update_every=1,chunksize=10000)
        model.save(filename+'.lda')
        print "Created %s.lda" % (filename)
    else:
        model = gensim.models.ldamodel.LdaModel.load(filename+'.lda')
        print "Loaded %s.lda" % (filename)
    
    print "Created/Loaded LDA model"    
    return model

def lsi(filename, corpus, dictionary, no_topics):
    print "Creating LSI model"

    if not os.path.isfile(filename+'.lsi'):
        model = gensim.models.lsimodel.LsiModel(corpus=corpus, num_topics= no_topics,id2word=dictionary)
        model.save(filename+'.lsi')
        print "Created %s.lsi" % (filename)
    else:
        model = gensim.models.lsimodel.LsiModel.load(filename+'.lsi')
        print "Loaded %s.lsi" % (filename)
    
    print "Created/Loaded LSI model"    
    return model

def hdp(filename, corpus, dictionary):
    print "Creating HDP model"

    if not os.path.isfile(filename+'.hdp'):
        model = gensim.models.hdpmodel.HdpModel(corpus=corpus, id2word=dictionary)
        model.save(filename+'.hdp')
        print "Created %s.hdp" % (filename)
    else:
        model = gensim.models.hdpmodel.HdpModel.load(filename+'.hdp')
        print "Loaded %s.hdp" % (filename)
    
    print model.show_topics(topics=1, topn=20)
    print "Created/Loaded HDP model"    
    return model

### END TOPIC MODELING ###

def show_top_topics(topics_words_dict, topic_id, num_topics):
    topic_list = topics_words_dict[topic_id]
    topic_list_sorted = sorted(topic_list,key=lambda tup: tup[1], reverse=True)
    topic_str = ", ".join("(word: %s, score: %s)" % tup for tup in topic_list_sorted[:num_topics])
    return topic_str



### MAIN ###
def main(argv):
    total = len(argv)

    if total < 3:
        print "Utilization: python topicmodelling.py <input_csv_file> <saved_models_filename> <no_topics>"
        exit(0)

    i_file = str(argv[1])

    filename = str(argv[2])

    no_topics = argv[3]

    if os.path.isfile(i_file):
        
        #Read tweets from csv

        articles = read_articles(i_file)
        
        #Pull out the titles/abstracts of each
        # articleDocs = [x["title"] for x in articles]
        articleDocs = [x["abstract"] for x in articles]
        
        #stopword list
        stopwords = stop_word_list()

        #Process them to clean bag of words
        #articleDocs = process_articles(articleDocs,stopwords)
        articleDocs = process_articles_gensim(articleDocs)
        
        #create gensim dict,corpus with the tweet bodies
        dictionary, corpus = put_in_dict_corpus(filename, articleDocs)
        

        #create TF-IDF model and corpus
        model, corpus_tfidf = tf_idf(filename, corpus)

        # Running LDA and printing results
        myLDA = lda(filename, corpus_tfidf, dictionary, int(no_topics))

        LDAtopics = myLDA.show_topics(topics=int(no_topics),topn=20,formatted=False)
        i = 0
        lda_topics_words = {}
        for t in LDAtopics:
            print "LDA Topic %i" % (i)
            print "======="
            topic_score = []
            for w in t:
                topic_score.append((w[1], w[0]))
                print "%-*s %f" % (15, w[1], w[0])
            lda_topics_words[i] = topic_score
            i += 1

        #Running LSI and printing results
        myLSI = lsi(filename, corpus_tfidf, dictionary, int(no_topics))
        LSItopics = myLSI.show_topics(num_words=20,formatted=False)
        i = 0
        lsi_topics_words = {}
        for t in LSItopics:
            print "LSI Topic %i" % (i)
            print "======="
            topic_score = []
            for w in t:
                topic_score.append((w[1], w[0]))
                print "%-*s %f" % (15, w[1], w[0])
            lsi_topics_words[i] = topic_score
            i += 1

        # #Running HDP and printing results
        # myHDP = hdp(filename, corpus_tfidf, dictionary)
        # myHDP.print_topics(topics=20, topn=10)
        
        topic_doc_count_lda = {}
        topic_doc_count_lsi = {}

        for i in range(0, len(articles)):
            print '\n'
            print "="*80

            print "Document Title: ", articles[i]['title']
            print "Document Abstract: ", articles[i]['abstract']
            print "Processed Document: ", articleDocs[i]
            # print "TF_IDF Corpus: ", str(corpus_tfidf[i])
            # print 'Matrix format: ' + str(corpus[i])
            # print 'Topic probability mixture: ' + str(myLDA[corpus[i]])

            print "LDA:"
            print 'Topic probability mixture: ' + str(myLDA[corpus_tfidf[i]])

            prob_topic_no_lda = sorted(myLDA[corpus_tfidf[i]],key=lambda tup: tup[1], reverse=True)
            for j,each_lda in enumerate(prob_topic_no_lda[:2]):
                j += 1
                print '%d Maximally probable topic LDA: topic #%s' % (j, str(each_lda[0]))
                #print 'Topic words: %s' % myLDA.print_topic(each_lda[0],4)
                print 'Topic words: %s' % show_top_topics(lda_topics_words, each_lda[0], 4)

                if str(each_lda[0]) in topic_doc_count_lda:
                    topic_doc_count_lda[str(each_lda[0])] += 1
                else:
                    topic_doc_count_lda[str(each_lda[0])] = 1

            print "-"*80
            
            print "LSI:"
            if myLSI[corpus[i]] != []:
                print 'Topic probability mixture: ' + str(myLSI[corpus_tfidf[i]])
                prob_topic_no_lsi = sorted(myLSI[corpus_tfidf[i]],key=lambda tup: tup[1], reverse=True)
                for k,each_lsi in enumerate(prob_topic_no_lsi[:2]):
                    k += 1
                    print '%d Maximally probable topic LSI: topic #%s' % (k, str(each_lsi[0]))
                    #print 'Topic words: %s' % myLSI.print_topic(each_lsi[0],4)
                    print 'Topic words: %s' % show_top_topics(lsi_topics_words, each_lsi[0], 4)

                    if str(each_lsi[0]) in topic_doc_count_lsi:
                        topic_doc_count_lsi[str(each_lsi[0])] += 1
                    else:
                        topic_doc_count_lsi[str(each_lsi[0])] = 1
            else:
                print "No LSI model due to blank tfidf-corpus"

            print "="*80

        print "="*80
        print "="*80

        print "LDA documents/topic"
        for k,v in topic_doc_count_lda.items():
            print "Topic %s in LDA models has %s number of documents" % (str(k),str(v))

        print "\n"
        print "LSI documents/topic"
        for k,v in topic_doc_count_lsi.items():
            print "Topic %s in LSI models has %s number of documents" % (str(k),str(v))
        
if __name__ == "__main__":
    main(sys.argv)
### END MAIN ###