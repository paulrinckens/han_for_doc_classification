import numpy as np
import pandas as pd


def load(path):
    df = pd.read_csv(path,
                     encoding="utf-8",
                     delimiter=";",
                     quotechar="'").rename(
        columns={
            "Text": "text",
            "Label": "label"
        })
    train, dev, test = split_df(df, 'label', 0.8, 0.1, 0.1)
    train_x = list(train["text"])
    train_y_dummies = pd.get_dummies(train["label"])
    class_names = list(train_y_dummies.columns.values)
    train_y = np.array(train_y_dummies)
    dev_x = list(dev["text"])
    dev_y = np.array(pd.get_dummies(dev["label"]))
    test_x = list(test["text"])
    test_y = np.array(pd.get_dummies(test["label"]))
    print(class_names)
    return train_x, train_y, dev_x, dev_y, test_x, test_y, class_names


def split_df(dataframe, column_name, training_split, validation_split,
             test_split):
    """
    Splits a pandas dataframe into trainingset, validationset and testset in specified ratio.
    All sets are balanced, which means they have the same ratio for each category as the full set.
    Input:   dataframe        - Pandas Dataframe, should include a column for data and one for categories
             column_name      - Name of dataframe column which contains the categorical output values
             training_split   - from ]0,1[, default = 0.6
             validation_split - from ]0,1[, default = 0.2
             test_split       - from ]0,1[, default = 0.2
                                Sum of all splits need to be 1
    Output:  train            - Pandas DataFrame of trainset
             validation       - Pandas DataFrame of validationset
             test             - Pandas DataFrame of testset
    """
    if training_split + validation_split + test_split != 1.0:
        raise ValueError('Split paramter sum should be 1.0')

    total = len(dataframe.index)

    train = dataframe.reset_index().groupby(column_name).apply(
        lambda x: x.sample(frac=training_split)).reset_index(
        drop=True).set_index('index')
    train = train.sample(frac=1)
    temp_df = dataframe.drop(train.index)
    validation = temp_df.reset_index().groupby(column_name).apply(
        lambda x: x.sample(frac=validation_split / (
                test_split + validation_split))).reset_index(
        drop=True).set_index('index')
    validation = validation.sample(frac=1)
    test = temp_df.drop(validation.index)
    test = test.sample(frac=1)

    print('Total: ', len(dataframe))
    print('Training: ', len(train), ', Percentage: ',
          len(train) / len(dataframe))
    print('Validation: ', len(validation), ', Percentage: ',
          len(validation) / len(dataframe))
    print('Test:', len(test), ', Percentage: ', len(test) / len(dataframe))

    return train, validation, test
