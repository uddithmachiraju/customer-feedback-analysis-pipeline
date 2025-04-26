import joblib
from logger import get_logger
from sklearn.metrics import classification_report, accuracy_score
from src.data_ingestion import load_data
from src.config import eval_path, model_saving_path

logger = get_logger("evaluate")

def evaluate_model(model, X_eval, y_eval):
    logger.info("Predicting on evaluation data")
    y_pred = model.predict(X_eval)

    logger.info("Generating classification report")
    print(classification_report(y_eval, y_pred))

    accuracy = accuracy_score(y_eval, y_pred)
    logger.info(f"Evaluation Accuracy: {accuracy}")
    print(f"Accuracy: {accuracy}")

def main():
    # Load the evaluation dataset
    eval_df = load_data(eval_path)
    logger.info("Loaded evaluation dataset")

    # Separate features and labels
    X_eval = eval_df.drop(columns=["Sentiment"])
    y_eval = eval_df["Sentiment"]
    logger.info("Separated features and labels")

    # Load the saved model
    model = joblib.load(model_saving_path)
    logger.info("Loaded trained model")

    # Evaluate the model
    evaluate_model(model, X_eval, y_eval)

if __name__ == "__main__":
    main()
