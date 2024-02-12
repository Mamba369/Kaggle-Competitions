from pandas import DataFrame

DEFAULT_SEED = 47


def lazy_predict(train_df: DataFrame) -> DataFrame:
    from lazypredict.Supervised import LazyClassifier
    from sklearn.model_selection import train_test_split

    # Splitting the data into features and target
    x_train = train_df.drop("Survived", axis=1)
    y_train = train_df["Survived"]

    # Splitting the dataset into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(
        x_train, y_train, test_size=0.1, random_state=DEFAULT_SEED
    )
    # Initialize LazyClassifier to fit all models
    clf = LazyClassifier(verbose=0, ignore_warnings=False, custom_metric=None)
    # Fit and compare all models
    models, _ = clf.fit(x_train, x_test, y_train, y_test)
    return models


def set_random_seed(seed: int = DEFAULT_SEED) -> None:
    import random
    import numpy as np

    # Define random seed to ensure repeatability of results
    random.seed(seed)
    np.random.seed(seed)
