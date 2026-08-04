"""Evaluate the saved facial recognition model without retraining.

Use this on every device so accuracy matches the shared saved weights.
"""
from models.model import Model
from preprocess import get_datasets

BEST_MODEL_PATH = "results/section6_final_model.keras"

if __name__ == "__main__":
    print("* Loading facial recognition model from {}".format(BEST_MODEL_PATH))
    model = Model.load_model(BEST_MODEL_PATH)
    print("* Loading datasets")
    _, _, test_dataset = get_datasets()
    print("* Evaluating facial recognition model on test set")
    model.evaluate(test_dataset)
    print("* Confusion matrix")
    print(model.get_confusion_matrix(test_dataset))
