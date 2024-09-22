import numpy as np
import pandas as pd
from collections import Counter
from DecisionTree.DecisionTree import DecisionTreeID3

def q2():
    train = pd.read_csv('car/train.csv')
    test = pd.read_csv('car/test.csv')
    X_train = train.drop(columns=['label'])
    y_train = train['label']

    X_test = test.drop(columns=['label'])
    y_test = test['label']
    for type in ['IG', 'ME', 'GI']:
        print('')
        for i in range(1, 7):
            tree = DecisionTreeID3(type, i)

            tree.fit(X_train, y_train)

            train_pred = tree.predict(X_train)
            test_pred = tree.predict(X_test)

            train_correct = (train_pred == y_train).sum()
            train_total = len(y_train)
            train_accuracy = train_correct / train_total
            print(f"Test Accuracy with {type} for depth of {i} on training data: {train_accuracy:.4f}")

            test_correct = (test_pred == y_test).sum()
            test_total = len(y_test)
            test_accuracy = test_correct / test_total

            
            print(f"Test Accuracy with {type} for depth of {i} on test data: {test_accuracy:.4f}")

def q3():
    train = pd.read_csv('bank/train.csv')
    test = pd.read_csv('bank/test.csv')
    X_train = train.drop(columns=['y'])
    y_train = train['y']

    X_test = test.drop(columns=['y'])
    y_test = test['y']
    print("Without replacing the unknowns")
    for type in ['IG', 'ME', 'GI']:
        print('')
        for i in range(1, 17):
            tree = DecisionTreeID3(type, i)

            tree.fit(X_train, y_train)

            train_pred = tree.predict(X_train)
            test_pred = tree.predict(X_test)

            train_correct = (train_pred == y_train).sum()
            train_total = len(y_train)
            train_accuracy = train_correct / train_total
            print(f"Test Accuracy with {type} for depth of {i} on training data: {train_accuracy:.4f}")

            test_correct = (test_pred == y_test).sum()
            test_total = len(y_test)
            test_accuracy = test_correct / test_total

            
            print(f"Test Accuracy with {type} for depth of {i} on test data: {test_accuracy:.4f}")

    def fill_unknown_with_mode(df):
        for column in df.columns:
            counts = Counter(df[column])
            sorted_counts = sorted(counts.items(), key=lambda item: item[1], reverse=True)

            if sorted_counts[0][0] != 'unknown':
                df[column].replace('unknown', sorted_counts[0][0], inplace=True) 
            else:
                df[column].replace('unknown', sorted_counts[1][0], inplace=True) 
        return df


    print("With replacing the unknowns")
    train = fill_unknown_with_mode(train)
    test = fill_unknown_with_mode(test)

    X_train = train.drop(columns=['y'])
    y_train = train['y']

    X_test = test.drop(columns=['y'])
    y_test = test['y']
    for type in ['IG', 'ME', 'GI']:
        print('')
        for i in range(1, 17):
            tree = DecisionTreeID3(type, i)

            tree.fit(X_train, y_train)

            train_pred = tree.predict(X_train)
            test_pred = tree.predict(X_test)

            train_correct = (train_pred == y_train).sum()
            train_total = len(y_train)
            train_accuracy = train_correct / train_total
            print(f"Test Accuracy with {type} for depth of {i} on training data: {train_accuracy:.4f}")

            test_correct = (test_pred == y_test).sum()
            test_total = len(y_test)
            test_accuracy = test_correct / test_total

            
            print(f"Test Accuracy with {type} for depth of {i} on test data: {test_accuracy:.4f}")

q2()
q3()
