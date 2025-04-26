import re
import nltk
import string
from logger import get_logger 
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from src.data_ingestion import load_data
from src.config import raw_data_path, preprocessed_data_path

# Download NLTK resources
nltk.download('wordnet')
nltk.download('stopwords')

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

logger = get_logger("preprocess") 

def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", "", text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)

def preprocess_dataframe(df):
    logger.info("Dropped Unnecessary Columns")
    # Drop unnecessary columns
    df.drop(columns=["Id", "ProductId", "UserId", "ProfileName", "Time"], inplace=True)

    # Drop rows with missing values
    logger.info("Dropped Rows with missing values") 
    df.dropna(subset=["Text", "Summary", "Score"], inplace=True)

    # Create helpfulness ratio
    df["HelpfulnessRatio"] = df.apply(
        lambda row: row["HelpfulnessNumerator"] / row["HelpfulnessDenominator"]
        if row["HelpfulnessDenominator"] > 0 else 0,
        axis=1
    )

    # Clean text
    logger.info("Cleaning Text & Summary")
    df["Cleaned_Text"] = df["Text"].apply(clean_text)
    df["Cleaned_Summary"] = df["Summary"].apply(clean_text)

    df["Cleaned_Text"] = df["Cleaned_Text"].fillna("")
    df["Cleaned_Summary"] = df["Cleaned_Summary"].fillna("")

    # Encode sentiment 
    logger.info("Encoding Sentiment (1, 2) - Negative(0), (3) - Neutral(1), (4, 5) - Positive(2)") 
    df["Sentiment"] = df["Score"].apply(lambda x: 0 if x <= 2 else 1 if x == 3 else 2)

    return df

def main():
    df = load_data(path=raw_data_path)
    df = preprocess_dataframe(df)
    df.to_csv(preprocessed_data_path, index=False)
    logger.info("Saved the preprocessed data") 

if __name__ == "__main__":
    main()
