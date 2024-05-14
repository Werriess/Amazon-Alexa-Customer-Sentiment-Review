import dash
from dash import html, dcc, Input, Output, State
import pandas as pd
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import string
import contractions
import dash_table


model = joblib.load("artifacts/model2.pkl")
vectorizer = joblib.load("artifacts/vectorizer2.pkl")

app = dash.Dash(__name__)
server = app.server

df = pd.read_csv("data/validation_data.csv")

columns = [{"name": i, "id": i} for i in df.columns]

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
    {'label': 'White Dot', 'value': 'white_dot'}
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
                        value=3, 
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
                html.Div([
                    html.H3('Amazon Reviews'),
                    html.Div(
                        dcc.Loading(
                            id='loading-table',
                            type='circle',
                            children=[
                                html.Div(
                                    dash_table.DataTable(
                                        id='review-table',
                                        columns=columns,
                                        data=df.to_dict('records'),
                                        page_size=10,  
                                        style_table={'overflowX': 'auto'},  
                                        style_cell={'minWidth': '150px', 'width': '150px', 'maxWidth': '150px'},  
                                        style_header={'backgroundColor': 'rgb(30, 30, 30)', 'color': 'white'},  
                                        style_data={'backgroundColor': 'rgb(50, 50, 50)', 'color': 'white'},  
                                        row_selectable='single',  
                                        selected_rows=[]  
                                    )
                                )
                            ]
                        )
                    )
                ], style={'overflowY': 'auto', 'height': '250px', 'margin-top': '20px', 'margin-left': '25px'})
            ]
        )
    ]
)

@app.callback(
    Output(component_id='rating-slider', component_property='value'),
    Output(component_id='date-picker', component_property='date'),
    Output(component_id='variation-dropdown', component_property='value'),
    Output(component_id='review-text', component_property='value'),
    Input(component_id='review-table', component_property='selected_rows'),
    State(component_id='review-table', component_property='data')
)
def update_inputs(selected_rows, data):
    if selected_rows:
        selected_row_index = selected_rows[0]
        selected_row = data[selected_row_index]
        rating = selected_row['rating']
        date = selected_row['date']
        variation = selected_row['variation']
        review = selected_row['verified_reviews']
        return rating, date, variation, review
    else:
        return 3, None, None, ''

@app.callback(
    Output(component_id='output-div', component_property='children'),
    Input(component_id='submit-val', component_property='n_clicks'),
    State('review-text', 'value')
)
def update_output(n_clicks, review_text):
    if n_clicks > 0 and review_text:  
        processed_text = process(review_text)

        vectorized_text = vectorizer.transform([processed_text])

        predicted_sentiment = model.predict(vectorized_text)[0]

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
    text = text.lower()
    
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    def remove_emojis(text):
        emoji_pattern = re.compile("["
                        u"\U0001F600-\U0001F64F"  
                        u"\U0001F300-\U0001F5FF"  
                        u"\U0001F680-\U0001F6FF"  
                        u"\U0001F1E0-\U0001F1FF"  
                        u"\U00002500-\U00002BEF"  
                        u"\U00002702-\U000027B0"
                        u"\U00002702-\U000027B0"
                        u"\U000024C2-\U0001F251"
                        u"\U0001f926-\U0001f937"
                        u"\U00010000-\U0010ffff"
                        u"\u2640-\u2642"
                        u"\u2600-\u2B55"
                        u"\u200d"
                        u"\u23cf"
                        u"\u23e9"
                        u"\u231a"
                        u"\ufe0f"  
                        u"\u3030"
                        "]+", flags=re.UNICODE)
        return emoji_pattern.sub(r'', text)
    
    text = remove_emojis(text)
    text = re.sub(r'^https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
    
    text = contractions.fix(text)
    
    tokens = negate_sequence(text)
    
    lemmatizer = WordNetLemmatizer()
    
    stop_words = set(stopwords.words('english'))
    
    lemmatized_tokens = []
    
    for token in tokens:
        if token in stop_words:
            continue
        
        lemma = lemmatizer.lemmatize(token)
        lemmatized_tokens.append(lemma)
        
    processed_text = ' '.join(lemmatized_tokens)
    
    return processed_text

if __name__ == '__main__':
    app.run_server(debug = True)
