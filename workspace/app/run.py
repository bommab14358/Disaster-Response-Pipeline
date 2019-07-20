import json
import plotly
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    '''
    Objective: To split the message text into words and process the words into a model interpretable format

    Input -
    text: Message as a text item

    Output -
    clean_tokens: Array containing all the words that make up the text with its inflected forms removed and case lowered
    '''
    # Define regular expression to identify URLs
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    # Identify all the URLs within the text and replace the url with text 'urlplaceholder'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    # Convert text into an array of words
    tokens = nltk.word_tokenize(text)
    # Convert the words into lowercase and remove the inflection forms from the words
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    # Return the array of processed words
    return clean_tokens


# load data
engine = create_engine('sqlite:///data/DisasterResponse.db')
df = pd.read_sql_table('data', engine)

# load model
model = joblib.load("models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    '''
    Objective:
    1. To showcase genres
    2. To showcase the distribution of Categories by their frequency
    '''
    # Extract genre distribution
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    # Extract category names
    labels = df.drop(['id', 'message', 'original', 'genre'], axis=1).columns
    # Calculate frequency of labels as column sum of each category
    label_freq = []
    for i in labels:
        label_freq.append([i, df[i].sum()])

    # Creating arrays for category frequency and category names in descreasing order of label frequency
    label_freq = pd.DataFrame(label_freq, columns = ['category', 'frequency']).sort_values(['frequency'], ascending = False)
    label_freq.index = label_freq.category
    label_freq = label_freq['frequency']
    labels = list(label_freq.index)

    # create visuals
    graphs = [
        {
            'data': [
                {'x':labels,'y':label_freq,'type':'bar','row':1, 'col':1},
                {'x':genre_names,'y':genre_counts,'type':'bar', 'xaxis':'x2', 'yaxis':'y2','row':2, 'col':1},
            ],

            'layout': {
                'autosize':False,
                'width':1200,
                'height':1000,
                'showlegend':False,
                'grid':{'rows':2, 'columns':1, 'pattern':'independent'},
                'title':'Distribution of messages by Category and Genre',
                
                'yaxis': {'title': "Count of messages"},
                'xaxis': {'title': "Category Name", 'tickangle':45, 'tickfont':{'size':10}},
                
                'xaxis2':{'title':'Genre'},
                'yaxis2':{'title':'Count of messages'},
                }
            }
    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    # encode plotly graphs in JSON
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '')

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file.
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
