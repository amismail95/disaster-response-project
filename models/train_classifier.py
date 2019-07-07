import nltk
nltk.download(['punkt', 'wordnet'])
import pandas as pd
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
import pickle
import sys


def load_data(database_filepath):
    """
    Load the necessary data.

    Args:
    database_filepath: file path for the database.

    Returns:
    X: features of the data.
    y: labels of the data.
    """
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('ProjectTableIsmail', engine)
    X = df['message']
    y = df.iloc[:, 4:]
    categories = list(df.columns[4:])
    return X, y, categories


def tokenize(text):
    """
    Make tokens from the text of the data.

    Args:
    text: messages from the features.

    Returns:
    clean_tokens: A list of tokenized text.
    """
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for token in tokens:
        clean_tok = lemmatizer.lemmatize(token).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    """
    Build a model using pipelines of CountVectorizer, TfidfTransformer, and MultiOutputClassifier.

    Args: None.

    Returns:
    pipeline: the model built with the pipeline.
    """
    pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer=tokenize)),
    ('tfid', TfidfTransformer()),
    ('clf', MultiOutputClassifier(MultinomialNB()))])

    return pipeline




def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate the model using classification_report.

    Args:
    model: the model used to train the data.
    X_test: features of the training set.
    Y_test: labels of the training sets.
    category_names: column names for the features.

    Returns: None
    """
    y_pred = model.predict(X_test)
    y_pred_data = pd.DataFrame(y_pred, columns=Y_test.columns)
    for column in y_pred_data:
        print('Scores for {}'.format(column))
        print('\n')
        print(classification_report(Y_test[column], y_pred_data[column]))


def save_model(model, model_filepath):
    """
    Saves the model as a pickle file.

    Args:
    model: the model used to train the data.
    model_filepath: the file path for the saved model.

    Returns:
    None
    """

    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
