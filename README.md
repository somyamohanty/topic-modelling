#Topic Modelling
Topic modelling for scientific articles using gensim LDA, LSI and HDP implementations. Utilizes titles and abstracts of scientific articles to obtain document/topic distributions of the input data-set. Currently LDA performs the best with alpha='auto' and number of topics between 30-50.

## Requirements
1. Gensim 
2. Numpy
3. Sci-kit
4. Plotly (For plotting graphs of topic distributions)

## Utilization

'''
python topicmodeling.py <path_to_input> <path_to_saved_models_output> <number_of_topics>
'''


