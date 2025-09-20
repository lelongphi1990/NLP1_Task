1)	Folder Structure:

flask_nlp_task /
├── Task.py
├── templates/
│               └── index.html
│               └── result.html
└── static/
                └── style.css

2)	Installation (if you haven't already):

pip install Flask NLTK spacy textblob
python -m spacy download en_core_web_sm # Download SpaCy's small English model
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('averaged_perceptron_tagger'); nltk.download('wordnet'); nltk.download('vader_lexicon')" # Download NLTK data

3)	Task.py : #This project is a web-based Multi-Language Natural Language Processing (NLP) Analyzer built using the Flask framework in Python. It allows users to input text via a simple web form and then automatically processes and analyzes that text, supporting both English (EN) and Vietnamese (VI).

import os
from flask import Flask, render_template, request

# --- NLP Libraries ---
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import spacy

# --- Language Detection ---
from langdetect import detect, DetectorFactory
DetectorFactory.seed = 0 # Ensure consistent results for langdetect

# --- Vietnamese NLP ---
try:
    from underthesea import word_tokenize as vn_word_tokenize
    from underthesea import pos_tag as vn_pos_tag
    from underthesea import ner as vn_ner
    VN_NLP_AVAILABLE = True
except ImportError:
    print("underthesea not found or failed to load. Vietnamese NLP will be limited.")
    VN_NLP_AVAILABLE = False


# --- Flask App Setup ---
app = Flask(__name__)

# --- NLP Model Loading (Load once globally for efficiency) ---
# SpaCy for English NER and POS tagging
try:
    nlp_spacy_en = spacy.load("en_core_web_sm")
except OSError:
    print("SpaCy model 'en_core_web_sm' not found. Please run: python -m spacy download en_core_web_sm")
    exit()

# NLTK resources (ensure they are downloaded by running the command above)
lemmatizer_en = WordNetLemmatizer()
vader_analyzer_en = SentimentIntensityAnalyzer()


# --- NLP Analysis Function ---
def perform_nlp_analysis(text):
    """
    Performs comprehensive NLP analysis on the given text,
    detecting language and applying appropriate tools.
    """
    if not text or not text.strip():
        app.logger.warning("perform_nlp_analysis received empty or whitespace-only text.")
        return {'lang': 'unknown'} # Indicate empty, no specific error for analysis

    results = {}
    
    # 1. Language Detection
    try:
        lang = detect(text)
        results['lang'] = lang
    except Exception as e:
        app.logger.warning(f"Could not detect language: {e}. Defaulting to English.")
        lang = 'en'
        results['lang'] = 'unknown (defaulting to English)'

    # Initialize placeholders for results if not applicable
    results['word_count'] = 0
    results['sentence_count'] = 0
    results['first_50_tokens'] = []
    results['unique_lemmas'] = []
    results['named_entities'] = []
    results['pos_tags'] = []

    if lang == 'en':
        app.logger.info("Performing English NLP analysis.")
        try:
            doc_spacy = nlp_spacy_en(text)

            # --- NLTK Tokenization and Basic Stats ---
            tokens = word_tokenize(text.lower())
            sentences = sent_tokenize(text)
            results['word_count'] = len(tokens)
            results['sentence_count'] = len(sentences)

            # Programmatic Modification 1: Remove specific token (e.g., 'the') from display list
            modified_tokens_display = [t for t in tokens if t != 'the']
            results['first_50_tokens'] = modified_tokens_display[:min(50, len(modified_tokens_display))]

            # --- Stop Words and Lemmatization ---
            stop_words = set(stopwords.words('english'))
            filtered_tokens = [
                token for token in tokens
                if token.isalpha() and token not in stop_words
            ]
            lemmas = [lemmatizer_en.lemmatize(token) for token in filtered_tokens]
            results['unique_lemmas'] = list(set(lemmas))[:min(20, len(lemmas))]

            # --- SpaCy Named Entity Recognition (NER) ---
            raw_entities = [{'text': ent.text, 'label': ent.label_} for ent in doc_spacy.ents]
            modified_entities = []
            for entity in raw_entities:
                if entity['label'] == 'CARDINAL':
                    continue
                if entity['text'].lower() == 'google':
                    entity['label'] = 'TECH_COMPANY'
                modified_entities.append(entity)
            results['named_entities'] = modified_entities

            # --- SpaCy Part-of-Speech Tagging (POS) ---
            raw_pos_tags = [{'token': token.text, 'pos': token.pos_} for token in doc_spacy if not token.is_space]
            modified_pos_tags = []
            forbidden_pos = {'PUNCT', 'SYM', 'X'}
            for pos_tag in raw_pos_tags:
                if pos_tag['pos'] in forbidden_pos:
                    continue
                if pos_tag['pos'] == 'PROPN':
                    pos_tag['pos'] = 'NAME'
                modified_pos_tags.append(pos_tag)
            results['pos_tags'] = modified_pos_tags[:min(30, len(modified_pos_tags))]


        except Exception as e:
            app.logger.error(f"Error during English NLP analysis: {e}", exc_info=True)
            results['error_details'] = f"English NLP error: {str(e)}"
            return None # Indicate critical error

    elif lang == 'vi':
        app.logger.info("Performing Vietnamese NLP analysis.")
        if not VN_NLP_AVAILABLE:
            results['error_details'] = "Vietnamese NLP tools (underthesea) not available."
            return None

        try:
            # --- Tokenization ---
            tokens = vn_word_tokenize(text, format="text").split(" ")
            # Underthesea tokenizes words like 'tôi_là_sinh_viên' then split by space
            # We want individual parts if possible for word count, or keep as multi-word tokens
            # For simplicity, let's count these as individual tokens for now
            
            results['first_50_tokens'] = tokens[:min(50, len(tokens))]
            results['word_count'] = len(tokens)
            # underthesea doesn't have a direct sent_tokenize, but a simple split by common punctuation can work
            sentences = text.split('.') # Basic sentence splitting, might need refinement
            results['sentence_count'] = len([s for s in sentences if s.strip()])


            # --- POS Tagging ---
            pos_tags_raw = vn_pos_tag(text) # Returns list of (token, pos_tag) tuples
            # Filter out punctuation and format
            modified_pos_tags = []
            for token, pos in pos_tags_raw:
                if pos not in ['P', 'CH', 'F']: # Common punctuation/symbol tags in Vietnamese
                    modified_pos_tags.append({'token': token, 'pos': pos})
            results['pos_tags'] = modified_pos_tags[:min(30, len(modified_pos_tags))]


            # --- Named Entity Recognition ---
            ner_raw = vn_ner(text) # Returns list of (token, pos_tag, ner_tag) tuples
            # underthesea's NER tags can be B-PER, I-PER, B-LOC, I-LOC etc.
            # We'll reformat them into a more user-friendly list of unique entities
            current_entity_text = ""
            current_entity_label = ""
            modified_entities = []

            for token, _, ner_tag in ner_raw:
                if ner_tag.startswith('B-'): # Beginning of an entity
                    if current_entity_text: # If there was a previous entity, add it
                        modified_entities.append({'text': current_entity_text, 'label': current_entity_label})
                    current_entity_text = token
                    current_entity_label = ner_tag[2:] # Remove 'B-'
                elif ner_tag.startswith('I-') and ner_tag[2:] == current_entity_label: # Inside an entity
                    current_entity_text += " " + token
                else: # Not part of an entity or new entity type starts
                    if current_entity_text:
                        modified_entities.append({'text': current_entity_text, 'label': current_entity_label})
                    current_entity_text = ""
                    current_entity_label = ""
            if current_entity_text: # Add the last entity if any
                modified_entities.append({'text': current_entity_text, 'label': current_entity_label})
            
            # Example modification: change specific entity label
            for entity in modified_entities:
                if entity['text'].lower() == 'Việt Nam':
                    entity['label'] = 'COUNTRY_VN'
            results['named_entities'] = modified_entities


        except Exception as e:
            app.logger.error(f"Error during Vietnamese NLP analysis: {e}", exc_info=True)
            results['error_details'] = f"Vietnamese NLP error: {str(e)}"
            return None # Indicate critical error

    else:
        results['error_details'] = f"Language '{lang}' is not supported for full NLP analysis."
        return None # Language not supported

    return results

# --- Flask Routes ---
@app.route('/', methods=['GET'])
def index():
    # Only render the input form on GET request
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    user_input_text = request.form.get('user_text', '')

    if not user_input_text.strip():
        # Redirect back to index with an error message
        # Flash messages are good for this, but for simplicity, we'll redirect
        # and re-render the index with an error query param or similar if needed.
        # For this version, we'll just show the error on the result page.
        analysis_results = {'error_details': "Please enter some text for analysis."}
        return render_template('result.html',
                               user_input=user_input_text,
                               analysis=analysis_results)


    analysis_results = perform_nlp_analysis(user_input_text)

    # Check for critical analysis failures or specific errors in results
    if 'error_details' in analysis_results:
        # If there's an error_details key, something went wrong during analysis
        # We can display this error on the result page
        return render_template('result.html',
                               user_input=user_input_text,
                               analysis=analysis_results) # Pass the error for display

    # If all is well, render the result page
    return render_template('result.html',
                           user_input=user_input_text,
                           analysis=analysis_results)

if __name__ == '__main__':
    app.run(debug=True)



4)   templates/index.html
#HTML
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Multi-Language NLP Analyzer</title>
    <link rel="stylesheet" href="{{ url_for('static', filename='style.css') }}">
</head>
<body>
    <div class="container">
        <h1>Multi-Language NLP Text Analyzer (EN/VI)</h1>
        <p>Enter text below to get comprehensive NLP analysis.</p>

        <form method="POST" action="/analyze">
            <label for="user_text">Enter your text here:</label>
            <textarea id="user_text" name="user_text" rows="10" placeholder="Type your text for analysis..." required></textarea>
            <button type="submit">Analyze Text</button>
        </form>
    </div>
</body>

5)   templates/result.html
#HTML
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial=1.0">
    <title>Analysis Result</title>
    <link rel="stylesheet" href="{{ url_for('static', filename='style.css') }}">
</head>
<body>
    <div class="container">
        <h1>NLP Analysis Result</h1>

        {# Display any error messages from the analysis #}
        {% if analysis.error_details %}
            <p class="error-message">Error: {{ analysis.error_details }}</p>
            <p><a href="/">Go back to input form</a></p>
        {% else %}
            <h3>Your Input:</h3>
            <p class="user-input">{{ user_input }}</p>

            <p><strong>Detected Language:</strong> {{ analysis.lang | upper }}</p>

            <h2>Analysis:</h2>
            <div class="analysis-results">
                <h3>General Statistics:</h3>
                <p><strong>Word Count:</strong> {{ analysis.word_count }}</p>
                <p><strong>Sentence Count:</strong> {{ analysis.sentence_count }}</p>
                <p><strong>First 50 Tokens:</strong> 
                    {% if analysis.first_50_tokens %}
                        <span class="token-list">{{ analysis.first_50_tokens|join(', ') }}</span>
                    {% else %}
                        No tokens found.
                    {% endif %}
                </p>

                <h3>Key Terms (English only for Lemmas):</h3>
                <p><strong>Unique Lemmas (up to 20):</strong> 
                    {% if analysis.unique_lemmas %}
                        <span class="lemma-list">{{ analysis.unique_lemmas|join(', ') }}</span>
                    {% else %}
                        Not applicable for this language or no lemmas found.
                    {% endif %}
                </p>

                <h3>Named Entities:</h3>
                {% if analysis.named_entities %}
                    <ul class="entity-list">
                        {% for entity in analysis.named_entities %}
                            <li><strong>{{ entity.text }}</strong> ({{ entity.label }})</li>
                        {% endfor %}
                    </ul>
                {% else %}
                    <p>No named entities found.</p>
                {% endif %}

                <h3>Part-of-Speech Tags:</h3>
                {% if analysis.pos_tags %}
                    <ul class="pos-list">
                        {% for tag in analysis.pos_tags %}
                            <li>{{ tag.token }} <span>({{ tag.pos }})</span></li>
                        {% endfor %}
                    </ul>
                {% else %}
                    <p>No POS tags generated.</p>
                {% endif %}

            </div>
            <p><a href="/">Analyze another text</a></p>
        {% endif %}
    </div>
</body>
</html>

6) static/style.css

body {
    font-family: Arial, sans-serif;
    margin: 0;
    padding: 20px;
    background-color: #f4f4f4;
    color: #333;
    display: flex;
    justify-content: center;
    align-items: flex-start;
    min-height: 100vh;
}

.container {
    background-color: #fff;
    padding: 30px;
    border-radius: 8px;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
    width: 100%;
    max-width: 900px;
}

h1, h2, h3 {
    color: #0056b3;
    text-align: center;
    margin-bottom: 20px;
}

p {
    margin-bottom: 10px;
    line-height: 1.6;
}

form label {
    display: block;
    margin-bottom: 8px;
    font-weight: bold;
}

textarea {
    width: calc(100% - 20px);
    padding: 10px;
    margin-bottom: 20px;
    border: 1px solid #ccc;
    border-radius: 4px;
    font-size: 16px;
    resize: vertical;
}

button {
    background-color: #007bff;
    color: white;
    padding: 12px 20px;
    border: none;
    border-radius: 4px;
    cursor: pointer;
    font-size: 18px;
    width: 100%;
    transition: background-color 0.3s ease;
}

button:hover {
    background-color: #0056b3;
}

.user-input {
    background-color: #e9ecef;
    padding: 15px;
    border-radius: 5px;
    margin-bottom: 20px;
    font-style: italic;
    white-space: pre-wrap; /* Preserves whitespace and line breaks */
    border-left: 5px solid #007bff;
    word-wrap: break-word; /* Ensure long words break */
}

.analysis-results ul {
    list-style-type: disc;
    margin-left: 20px;
    margin-bottom: 15px;
}

.analysis-results ul.pos-list {
    display: flex;
    flex-wrap: wrap;
    list-style: none;
    padding: 0;
}

.analysis-results ul.pos-list li {
    background-color: #e6f7ff;
    border: 1px solid #b3e0ff;
    border-radius: 3px;
    padding: 5px 8px;
    margin: 3px;
    font-size: 0.9em;
}

.analysis-results ul.pos-list li span {
    font-weight: bold;
    color: #0056b3;
}

.error-message {
    color: red;
    background-color: #ffebe8;
    border: 1px solid #ffccc7;
    padding: 10px;
    border-radius: 5px;
    margin-bottom: 20px;
    text-align: center;
}

.positive {
    color: green;
    font-weight: bold;
}

.negative {
    color: red;
    font-weight: bold;
}

.neutral {
    color: orange;
    font-weight: bold;
}

.sentiment-score {
    font-weight: bold;
    color: #4CAF50; /* A pleasant green */
}

a {
    display: block;
    text-align: center;
    margin-top: 20px;
    color: #007bff;
    text-decoration: none;
}

a:hover {
    text-decoration: underline;
}