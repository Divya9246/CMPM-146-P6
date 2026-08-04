"""Evaluate the locked FER2013 model without retraining.

Use this on every device so accuracy stays the same as the saved weights.
"""
from models.model import Model
from preprocess import get_datasets

BEST_MODEL_PATH = "results/best_basic_model.keras"

if __name__ == "__main__":
    print("* Loading locked model from {}".format(BEST_MODEL_PATH))
    model = Model.load_model(BEST_MODEL_PATH)
    print("* Loading datasets")
    _, _, test_dataset = get_datasets()
    print("* Evaluating locked model on test set")
    model.evaluate(test_dataset)
    print("* Confusion matrix")
    print(model.get_confusion_matrix(test_dataset))
