import joblib 
from logger import get_logger 
from src.data_ingestion import load_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from src.config import preprocessed_data_path, model_saving_path
from sklearn.metrics import classification_report, accuracy_score

logger = get_logger("train") 

def train_model(X_train, y_train):
    logger.info("Training RandomForest Classifier")
    model = RandomForestClassifier(n_estimators = 100, class_weight = {0: 2.533, 1: 4.222, 2: 0.422})
    model.fit(X_train, y_train) 
    return model 

def main(df):

    df = df[df["Cleaned_Text"].notna()]

    # Extract features and labels 
    X = df['Cleaned_Text']
    y = df['Sentiment']
    logger.info("Extracted features and labels")

    # TF-IDF Vectorization
    vectorizer = TfidfVectorizer(max_features = 1000) 
    x_vectorized = vectorizer.fit_transform(X) 
    logger.info("Vectorized text into TF-IDF") 

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        x_vectorized, y, test_size = 0.2, shuffle = True
    )
    logger.info("Splitted the data into train and test") 

    # Train the model 
    model = train_model(X_train, y_train) 

    # Evaluate the model
    y_pred = model.predict(X_test)
    logger.info("Evaluating the model")
    print(classification_report(y_test, y_pred)) 

    # Calculate Accuracy
    accuracy = accuracy_score(y_test, y_pred) 
    logger.info(f"Accuracy of the model is {accuracy}") 
    print(f"Accuracy: {accuracy}") 

    # Save the model 
    joblib.dump(model, model_saving_path)
    logger.info(f"Saved the model to {model_saving_path}") 

if __name__ == "__main__":
    df = load_data(preprocessed_data_path)
    logger.info("Loaded the preprocessed data")
    main(df)  