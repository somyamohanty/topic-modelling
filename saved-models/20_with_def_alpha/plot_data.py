import plotly
import json


py = plotly.plotly("somyamohanty", "94wksibtod")

with open('20_def_alpha_plot_data.json', 'rb') as f:
    data_dist = json.loads(f.readline())

f.close()

x_lda = data_dist['lda']['x']
y_lda = data_dist['lda']['y']
label_lda = data_dist['lda']['labels']

layout = {
    'title': 'LDA Topic-Document Distribution - ' + str(20) + ' topics',
    'xaxis': {'title': 'Topic'},
    'yaxis': {'title': 'Number of Document'},
    }

py.iplot([{
    'x':x_lda, 
    'y':y_lda, 
    'type': 'bar', 
    'name': 'Number of Docs',
    'text': label_lda,
    }], 
    layout=layout, filename='lda_default_alpha'+str(20), fileopt='overwrite', width=1000, height=650)

lda_plot_dict = {
    'x':x_lda,
    'y':y_lda,
    'labels':label_lda
}


x_lsi = data_dist['lsi']['x']
y_lsi = data_dist['lsi']['y']
label_lsi = data_dist['lsi']['labels']

layout = {
    'title': 'LSI Topic-Document Distribution - ' + str(20) + ' topics',
    'xaxis': {'title': 'Topic'},
    'yaxis': {'title': 'Number of Document'},
    }

py.iplot([{
    'x':x_lsi, 
    'y':y_lsi, 
    'type': 'bar', 
    'name': 'Number of Docs', 
    'text': label_lsi,
    }], 
    layout=layout, filename='lsi_def_'+str(20), fileopt='overwrite', width=1000, height=650)
