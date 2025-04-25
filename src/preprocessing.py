import re
import nltk
import string
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from src.data_ingestion import load_data
from config import raw_data_path, preprocessed_data_path, processed_features_path

# Download NLTK resources
nltk.download('wordnet')
nltk.download('stopwords')

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", "", text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)

def preprocess_dataframe(df):
    # Drop unnecessary columns
    df.drop(columns=["Id", "ProductId", "UserId", "ProfileName", "Time"], inplace=True)

    # Drop rows with missing values
    df.dropna(subset=["Text", "Summary", "Score"], inplace=True)

    # Create helpfulness ratio
    df["HelpfulnessRatio"] = df.apply(
        lambda row: row["HelpfulnessNumerator"] / row["HelpfulnessDenominator"]
        if row["HelpfulnessDenominator"] > 0 else 0,
        axis=1
    )

    # Clean text
    df["Cleaned_Text"] = df["Text"].apply(clean_text)
    df["Cleaned_Summary"] = df["Summary"].apply(clean_text)

    # Encode sentiment
    df["Sentiment"] = df["Score"].apply(lambda x: 0 if x <= 2 else 1 if x == 3 else 2)

    return df

def extract_features_and_labels(df):
    tfidf = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
    X = tfidf.fit_transform(df["Cleaned_Text"])
    y = df["Sentiment"]
    return X, y

def save_features(X, y, path):
    df_features = pd.DataFrame(X.toarray())
    df_features["Sentiment"] = y.values
    df_features.to_csv(path, index=False)

def main():
    df = load_data(path=raw_data_path)
    df = preprocess_dataframe(df)
    df.to_csv(preprocessed_data_path, index=False)

    X, y = extract_features_and_labels(df)
    save_features(X, y, processed_features_path)

if __name__ == "__main__":
    main()
