from typing import Dict, List

import numpy as np
import pandas as pd
import seaborn as sn
from matplotlib import pyplot as plt
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import ComplementNB
from sklearn.neighbors import KNeighborsClassifier


def get_data(path: str = "") -> List[pd.DataFrame]:
    """
    function to read data from csv
    :param path: string path to folder containing log2.csv (default value if CWD)
    :return: list of dataframes containing data and class labels respectively
    """
    X = pd.read_csv("log2.csv")
    y = X[["Action"]]
    X = X.drop("Action", axis=1)
    return [X, y]


def visualize(X: pd.DataFrame, y: pd.DataFrame) -> None:
    """
    function to visualize proportion of class sizes in the dataset
    :param X: dataframe containing data
    :param y: dataframe containing class labels corresponding to X
    :return: None
    """
    y["Action"].value_counts().plot.pie(explode=(0.02, 0.04, 0.05, 0.09), title="Proportion of classes in dataset")
    plt.savefig("Figures/proportions")

    for i, column in enumerate(X.columns):
        fig, ax = plt.subplots(1, 2)

        ax[0].hist(
            (
                X[y["Action"] == "allow"][column],
                X[y["Action"] == "deny"][column],
                X[y["Action"] == "drop"][column],
                X[y["Action"] == "reset-both"][column],
            )
        )
        ax[0].set_xlabel(column)
        ax[0].set_ylabel("Frequency")

        ax[1].boxplot(
            (
                X[y["Action"] == "allow"][column],
                X[y["Action"] == "deny"][column],
                X[y["Action"] == "drop"][column],
                X[y["Action"] == "reset-both"][column],
            )
        )
        ax[1].set_xlabel("Action")
        ax[1].set_ylabel(column)

        X[column].hist(by=y["Action"])

        ax[0].legend(["allow", "deny", "drop", "reset-both"])
        ax[1].set_xticklabels(["allow", "deny", "drop", "reset-both"])
        fig.suptitle("Distribution of classes among attributes")
        plt.savefig("Figures/boxplots")


def cross_validate(estimator: BaseEstimator, X: pd.DataFrame, y: pd.DataFrame, num_splits: int, save_name: str) -> None:
    """
    function to perform cross validation and call error_profile at the end to generate an error report for a sklearn
    model
    :param estimator: SkLearn classification model
    :param X: dataframe containing data
    :param y: dataframe containing class labels corresponding to X
    :param num_splits: number of folds for k-fold cross validation
    :param save_name: save name for error profile plots (file extension will be appended)
    :return: None
    """
    splitter = StratifiedKFold(n_splits=num_splits, shuffle=True, random_state=0)

    predictions = {"test": [], "train": []}
    y_true = {"test": [], "train": []}

    for train_index, test_index in splitter.split(X, y):
        estimator.fit(X.iloc[train_index, :], y.iloc[train_index, 0])
        test_pred = estimator.predict(X.iloc[test_index, :])
        train_pred = estimator.predict(X.iloc[train_index, :])

        predictions["train"].append(train_pred)
        predictions["test"].append(test_pred)

        y_true["train"].append(np.array(y.iloc[train_index])[:, 0])
        y_true["test"].append(np.array(y.iloc[test_index])[:, 0])

    error_profile(y_true, predictions, model_type=save_name)


def fit_and_test(X, y) -> None:
    """
    function to fit and test numerous models for the given data
    :param X: dataframe containing data
    :param y: dataframe containing class labels corresponding to X
    :return: None
    """
    models = {
        "tree2": RandomForestClassifier(n_estimators=1, n_jobs=-1, class_weight="balanced", random_state=0),
        "tree1": RandomForestClassifier(n_estimators=1, n_jobs=-1, random_state=0, criterion="entropy"),
        "random_forest_10": RandomForestClassifier(
            n_estimators=10, n_jobs=-1, class_weight="balanced", criterion="gini"
        ),
        "random_forest_100": RandomForestClassifier(n_estimators=100, n_jobs=-1, criterion="entropy"),
        "knn_1": KNeighborsClassifier(n_neighbors=1, n_jobs=-1, metric="hamming"),
        "knn_5": KNeighborsClassifier(n_neighbors=5, n_jobs=-1, metric="hamming"),
        "knn_15": KNeighborsClassifier(n_neighbors=15, n_jobs=-1, metric="hamming"),
        "cnb": ComplementNB(),
    }

    for model_name in models.keys():
        cross_validate(estimator=models[model_name], X=X, y=y, num_splits=5, save_name=model_name)


def error_profile(y_true: Dict[str, List[np.ndarray]], y_pred: Dict[str, List[np.ndarray]], model_type: str) -> None:
    """
    function to generate the error profile based on true labels and predicted labels for a classification problem
    :param y_true: dictionary containing true labels for training and testing of each fold
    :param y_pred: dictionary containing predicted labels for training and testing of each fold
    :param model_type: name of model to use to save error profile plots (file extensions will be appended)
    :return: None
    """
    num_folds = len(y_pred["train"])

    acc = {"train": [], "test": []}
    test_predictions = np.array([])
    test_labels = np.array([])

    for k in range(num_folds):
        y_train_true = y_true["train"][k]
        y_train_pred = y_pred["train"][k]
        y_test_true = y_true["test"][k]
        y_test_pred = y_pred["test"][k]

        # Accuracies
        train_acc = np.sum(np.equal(y_train_true, y_train_pred)) / np.shape(y_train_true)[0]
        test_acc = np.sum(np.equal(y_test_true, y_test_pred)) / np.shape(y_test_true)[0]
        acc["train"].append(train_acc)
        acc["test"].append(test_acc)

        test_labels = np.concatenate((test_labels, y_test_true))
        test_predictions = np.concatenate((test_predictions, y_test_pred))

    pd.DataFrame(acc).plot().set_title("Accuracies for " + model_type)
    plt.xlabel("Cross validation fold")
    plt.ylabel("Accuracy (max = 1)")
    plt.xticks(list(range(num_folds)))
    plt.tight_layout()
    plt.savefig("Figures/" + model_type + "_acc")

    classes = np.unique(test_labels)

    # Confusion matrix
    # we only care for the confusion matrix of the testing set
    conf_mat = confusion_matrix(test_labels, test_predictions)
    fig, ax = plt.subplots(1, 2, sharey="all", figsize=(16, 9))
    sn.heatmap(
        conf_mat,
        cmap="Oranges",
        annot=True,
        xticklabels=classes,
        yticklabels=classes,
        ax=ax[0],
    )
    ax[0].set_title("Confusion matrix")
    conf_mat2 = np.array(conf_mat)
    np.fill_diagonal(conf_mat2, -1)
    sn.heatmap(
        conf_mat2,
        cmap="Oranges",
        annot=True,
        xticklabels=classes,
        yticklabels=classes,
        ax=ax[1],
    )
    ax[1].set_title("Confusion matrix (ignoring diagonal)")
    fig.suptitle("Confusion matrices for " + model_type)
    plt.savefig("Figures/" + model_type + "cfx_mat")

    # Evaluate metrics for each class
    metrics = {}
    total = np.sum(conf_mat)
    for class_num in range(np.shape(conf_mat)[0]):
        class_metrics = {}
        tp = conf_mat[class_num, class_num]
        fn = np.sum(conf_mat[class_num, :]) - tp
        fp = np.sum(conf_mat[:, class_num]) - tp
        tn = total - tp - fn - fp

        class_metrics["sens"] = tp / (tp + fn)  # specificity (recall)
        class_metrics["spes"] = tn / (tn + fp)  # sensitivity
        class_metrics["ppv"] = tp / (tp + fp)  # positive predictive value (precision)
        class_metrics["npv"] = tn / (tn + fn)  # negative predictive value
        class_metrics["F1"] = (2 * tp) / (2 * tp + fn + fp)  # F1 score
        class_metrics["auc"] = roc_auc_score(  # Area under ROC
            test_labels == classes[class_num], test_predictions == classes[class_num]
        )

        metrics[classes[class_num]] = class_metrics

    print("-" * 100)
    print("## Error profile for " + model_type)
    print("Cross validated accuracy = {}%".format(np.mean(acc["test"]) * 100))
    print(pd.DataFrame(metrics).to_markdown())
    print("-" * 100)


if __name__ == "__main__":
    data, labels = get_data()
    visualize(data, labels)
    fit_and_test(data, labels)
