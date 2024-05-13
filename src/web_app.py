#libraries
import dash
from dash import html, dcc, Input, Output, State
import pandas as pd
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import string

# Load the model and vectorizer
model = joblib.load("./Amazon-Customer-Sentiment-Review/artifacts/model1.pkl")

#initialise dash app
app = dash.Dash(__name__)
server = app.server

input_ids = ['rating', 'date', 'variation', 'verified_reviews']

variation_options = [
    {'label': 'Charcoal Fabric', 'value': 'charcoal_fabric'},
    {'label': 'Walnut Finish', 'value': 'walnut_finish'},
    {'label': 'Heather Gray Fabric', 'value': 'heather_gray_fabric'},
    {'label': 'Sandstone Fabric', 'value': 'sandstone_fabric'},
    {'label': 'Oak Finish', 'value': 'oak_finish'},
    {'label': 'Black', 'value': 'black'},
    {'label': 'White', 'value': 'white'},
    {'label': 'Black Spot', 'value': 'black_spot'}, 
    {'label': 'White Spot', 'value': 'white_spot'}, 
    {'label': 'Black Show', 'value': 'black_show'}, 
    {'label': 'White Show', 'value': 'white_show'}, 
    {'label': 'Black Plus', 'value': 'black_plus'}, 
    {'label': 'White Plus', 'value': 'white_plus'}, 
    {'label': 'Configuration: Fire TV Stick', 'value': 'fire_tv_stick'},
    {'label': 'Black Dot', 'value': 'black_dot'},  
    {'label': 'White Dot', 'value': 'white_dot'}, 
]

app.layout = html.Div(
    className='app-container', 
    style = {'background-color' : '#1d7874'},
    children=[
        html.H1('Amazon Alexa Review',
            style={ 'padding': '10px 20px',
                        'margin-top': '20px',
                        'font-family': 'calibri',
                        'font-weight': '600',
                        'font-size':'2em',
                        'color': '#071e22',
                        'width': '500px',
                        'text-align': 'center'}),
            html.Div( 
            className='content',
            style = {'background-color' : '#071e22',
                    'color': 'white',
                    'width':'200',
                    'font-family':'calibri',
                    'padding':'50px'},
            children=[
                html.Div([
                    html.Label('Rating', style={'display': 'block', 
                                                'margin-left': '25px',
                                                'font-size':'20px'}),
                    dcc.Slider(
                        id='rating-slider',
                        min=1,
                        max=5,
                        step=1,
                        value=3,  # Set default value here
                        marks={i: str(i) for i in range(1, 6)},
                    )
                ], className='rating-container',
                style={'width':'500px'}), 
                html.Div([
                    html.Label('Date', style={'display': 'block',
                                                'margin-left': '25px',
                                                'font-size':'20px'}),
                    dcc.DatePickerSingle(
                        id='date-picker',
                        date=None,
                        style={'margin-left': '25px'}
                    )
                ], className='date-container'), 
                html.Div([
                    html.Div([
                        html.Label('Variation', style={'display': 'block',
                                                        'margin-top':'20px',
                                                'margin-left': '25px',
                                                'color': 'white',
                                                'font-size':'20px'}),
                        dcc.Dropdown(
                            id='variation-dropdown',
                            options=variation_options,
                            value=None,
                            style={'margin-left': '13px', 'width': '465px', 'height': '50px', 'font-size':'22px', 'color': '#071e22'}
                        )
                    ], className='variation-container'),  
                    html.Div([
                        html.Label('Review', style={'display':'block',
                                                    'margin-top':'20px',
                                                    'margin-left': '25px',
                                                'font-size':'20px'}),
                        dcc.Textarea(
                            id='review-text',
                            value='',
                            style={'margin-left':'25px',
                                    'width':'460px',
                                    'height': '150px',
                                    'font-size':'large',
                                    'color':'#071e22'}
                        )
                    ], className='review-container'),  
                    html.Button('Submit', 
                                id='submit-val', 
                                n_clicks=0,
                                style={'background-color': '#1d7874',
                                        'color': 'white',
                                        'padding': '10px',
                                        'border': '1px solid transparent',
                                        'border-radius' : '10px',
                                        'margin-top':'10px',
                                        'margin-left':'25px'}),
                    html.Div(id='output-div')
                ], className='review-section'),  
            ]
        )
    ]
)

@app.callback(
    Output(component_id='output-div', component_property='children'),
    Input(component_id='submit-val', component_property='n_clicks'),
    State('review-text', 'value')
)
def update_output(n_clicks, review_text):
    if n_clicks > 0 and review_text:  # Check if review text is not empty
        # Preprocess the review text
        processed_text = process(review_text)

        # Vectorize the processed text
        vectorized_text = vectorizer.transform([processed_text])

        # Make predictions
        predicted_sentiment = model.predict(vectorized_text)[0]

        # Interpret results
        if predicted_sentiment == -1:
            sentiment = "Negative Sentiment"
        elif predicted_sentiment == 0:
            sentiment = "Neutral Sentiment"
        else:
            sentiment = "Positive Sentiment"  

        return html.Div([
            html.H3("Predicted Sentiment:"),
            html.P(sentiment)
        ])
    else:
        return None  

def negate_sequence(text):
    negation = False
    delims = "?.,!:;"
    result = []
    words = text.split()
    prev = None
    pprev = None
    for word in words:
        stripped = word.strip(delims).lower()
        negated = "not_" + stripped if negation else stripped
        result.append(negated)
        if prev:
            bigram = prev + " " + negated
            result.append(bigram)
            if pprev:
                trigram = pprev + " " + bigram
                result.append(trigram)
            pprev = prev
        prev = negated

        if any(neg in word for neg in ["not", "n't", "no"]):
            negation = not negation

        if any(c in word for c in delims):
            negation = False

    return result

def process(text):
    # Convert text to lower case
    text = text.lower()
    
    # Remove all punctuation in text
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove HTML code or URL links
    text = re.sub(r'^https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
    
    # Fix abbreviated words
    text = contractions.fix(text)
    
    # Tokenize and handle negation
    tokens = negate_sequence(text)
    
    lemmatizer = WordNetLemmatizer()
    
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    
    lemmatized_tokens = []
    
    for token in tokens:
        if token in stop_words:
            continue
        
        # Lemmatization
        lemma = lemmatizer.lemmatize(token)
        lemmatized_tokens.append(lemma)
        
    processed_text = ' '.join(lemmatized_tokens)
    
    return processed_text

if __name__ == '__main__':
    app.run_server(debug = True)
