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
    if training_split + validation_split + test_split != 1.0:
        raise ValueError('Split paramter sum should be 1.0')

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
