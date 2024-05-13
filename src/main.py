import string
import re
import contractions
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

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

print(process("I didn't perform"))
