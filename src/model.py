from sklearn import linear_model
from sklearn.metrics import accuracy_score
import pandas as pd
import time

import tensorflow as tf
from tensorflow.keras import layers

def train_and_predict_ridge_classification(train_df,test_df):
    reg = linear_model.RidgeClassifier(alpha = 0.55)
    reg.fit(train_df[train_df.columns[2:]],train_df["target"])
    test_predictions = reg.predict(test_df[test_df.columns[1:]])
    test_predictions_df = pd.DataFrame([pd.Series(test_predictions)]).transpose()
    test_predictions_df.columns = ["target"]
    test_predictions_df["id"] = test_df["id"]
    return test_predictions_df


if __name__== "__main__":
    train_df = pd.read_csv("data/train.csv")
    test_df = pd.read_csv("data/test.csv")


    # # train on ridge classification
    # test_predictions_df = train_and_predict_ridge_classification(df,test_df)
    # test_predictions_df.to_csv("output/test_predictions.csv" + str(int(time.time())), index = False)

    # train on decision trees


    # train on fully connected neural network
    model = tf.keras.Sequential()
    model.add(layers.Dense(150,activation='relu', input_shape=(300,)))
    model.add(layers.Dense(75,activation='relu'))
    model.add(layers.Dense(1))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.01),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=["accuracy"]
    )
    labels = train_df["target"].values
    print(labels.shape)
    model.fit(train_df[train_df.columns[2:]].values,train_df["target"].values,epochs=10, batch_size=32)
    test_predictions = model.predict(test_df[test_df.columns[1:]].values)
    test_predictions = test_predictions.reshape((test_predictions.shape[0],))
    test_predictions_df = pd.DataFrame([pd.Series(test_predictions)]).transpose()
    test_predictions_df.columns = ["target"]
    test_predictions_df["id"] = test_df["id"]
    # test_predictions_df = train_and_predict_ridge_classification(df,test_df)
    test_predictions_df.to_csv("output/test_predictions_fc_neural_network.csv" + str(int(time.time())), index = False)

    # print(model.summary())
