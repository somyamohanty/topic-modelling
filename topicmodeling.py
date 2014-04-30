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
    # stemmer = nltk.stem.SnowballStemmer('english')
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

    print "Created TF-IDF model and Corpus"

    return model, corpus_tfidf

def lda(filename, corpus, dictionary):
    print "Creating LDA model"

    if not os.path.isfile(filename+'.lda'):
        model = gensim.models.ldamodel.LdaModel(corpus=corpus, num_topics= 20,id2word=dictionary, passes=250, update_every=1,chunksize=10000)
        model.save(filename+'.lda')
        print "Created %s.lda" % (filename)
    else:
        model = gensim.models.ldamodel.LdaModel.load(filename+'.lda')
        print "Loaded %s.lda" % (filename)
    
    print "Created LDA model"    
    return model

def lsi(filename, corpus, dictionary):
    print "Creating LSI model"

    if not os.path.isfile(filename+'.lsi'):
        model = gensim.models.lsimodel.LsiModel(corpus=corpus, num_topics= 20,id2word=dictionary)
        model.save(filename+'.lsi')
        print "Created %s.lsi" % (filename)
    else:
        model = gensim.models.lsimodel.LsiModel.load(filename+'.lsi')
        print "Loaded %s.lsi" % (filename)
    
    print "Created LSI model"    
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
    print "Created HDP model"    
    return model

### END TOPIC MODELING ###

### MAIN ###
def main(argv):
    total = len(argv)

    if total < 2:
        print "Utilization: python topicmodelling.py <input_csv_file> <saved_models_filename>"
        exit(0)

    i_file = str(argv[1])

    filename = str(argv[2])

    if os.path.isfile(i_file):
        
        #Read tweets from csv
        articles = read_articles(i_file)
        
        #Pull out the titles/abstracts of each
        #articleDocs = [x["title"] for x in articles]
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
        myLDA = lda(filename, corpus_tfidf, dictionary)
        LDAtopics = myLDA.show_topics(topics=20,topn=20,formatted=False)
        i = 0
        for t in LDAtopics:
            i += 1
            print "LDA Topic %i" % (i)
            print "======="
            for w in t:
                print "%-*s %f" % (15, w[1], w[0])

        #Running LSI and printing results
        myLSI = lsi(filename, corpus_tfidf, dictionary)
        LSItopics = myLSI.show_topics(num_words=25,formatted=False)
        i = 0
        for t in LSItopics:
            i += 1
            print "LSI Topic %i" % (i)
            print "======="
            for w in t:
                print "%-*s %f" % (15, w[1], w[0])

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
            prob_topic_no_lda = max(myLDA[corpus_tfidf[i]],key=itemgetter(1))[0]
            print 'Topic probability mixture: ' + str(myLDA[corpus_tfidf[i]])
            print 'Maximally probable topic LDA: topic #' + str(prob_topic_no_lda)
            print 'Topic words: ', myLDA.print_topic(prob_topic_no_lda)

            if str(prob_topic_no_lda) in topic_doc_count_lda:
                topic_doc_count_lda[str(prob_topic_no_lda)] += 1
            else:
                topic_doc_count_lda[str(prob_topic_no_lda)] = 1

            print "-"*80
            
            print "LSI:"
            if myLSI[corpus[i]] != []:
                prob_topic_no_lsi = max(myLSI[corpus_tfidf[i]],key=itemgetter(1))[0]
                print 'Topic probability mixture: ' + str(myLSI[corpus_tfidf[i]])
                print 'Maximally probable topic LSI: topic #' + str(prob_topic_no_lsi)
                print 'Topic words: ', myLSI.print_topic(prob_topic_no_lsi)

                if str(prob_topic_no_lsi) in topic_doc_count_lsi:
                    topic_doc_count_lsi[str(prob_topic_no_lsi)] += 1
                else:
                    topic_doc_count_lsi[str(prob_topic_no_lsi)] = 1
            else:
                print "No LSI model due to blank tfidf-corpus"

            print "="*80

        print topic_doc_count_lda, topic_doc_count_lsi

if __name__ == "__main__":
    main(sys.argv)
### END MAIN ###