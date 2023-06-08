import pandas
import pandas as pd


import tensorflow as tf
from keras.models import Sequential
from keras.layers import Activation, Dense


from matplotlib import pyplot as plt

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier


import numpy as np
from utils import split_train_test



def preprocess_data(X: pd.DataFrame, y: pd.Series):
    X['booking_day'] = pd.to_datetime(X['booking_datetime']).dt.dayofyear
    X['check_in'] = pd.to_datetime(X['checkin_date']).dt.dayofyear
    X['check_out'] = pd.to_datetime(X['checkout_date']).dt.dayofyear
    #X['hotel_county_code_one_hot'] = pd.Categorical(X['hotel_country_code']).cat.codes
    X['h_customer_one_hot'] = pd.factorize(X['h_customer_id'])[0]
    X['hotel_county_code_one_hot'] = pd.factorize(X['hotel_country_code'])[0]
    X['accommodation_type_name_one_hot'] = pd.factorize(X['accommadation_type_name'])[0]
    X['charge_option_one_hot'] = pd.factorize(X['charge_option'])[0]
    X['nationality_one_hot'] = pd.factorize(X['guest_nationality_country_name'])[0]
    X['language_one_hot'] = pd.factorize(X['language'])[0]
    X['payment_type_one_hot'] = pd.factorize(X['original_payment_type'])[0]
    X['cancellation_policy_one_hot'] = pd.factorize(X['cancellation_policy_code'])[0]
    X['is_first_booking'] = pd.factorize(X['is_first_booking'])[0]
    X['cancellation_policy_code'] = pd.factorize(X['cancellation_policy_code'])[0]

    ls = {'request_nonesmoke':0.5, 'request_latecheckin':0.5, 'request_highfloor':0.5, 'request_largebed':0.5, 'request_twinbeds':0.5, 'request_airport':0.5, 'request_earlycheckin':0.5}
    X.fillna(ls, inplace=True)

    bk_id = X.pop("h_booking_id")
    X.drop(columns=["hotel_country_code", "accommadation_type_name", "charge_option", "guest_nationality_country_name", "language", "original_payment_type", "cancellation_policy_code", "booking_datetime", "checkin_date", "checkout_date",  "hotel_brand_code", "hotel_chain_code", "original_payment_method", "hotel_id", "hotel_area_code", "hotel_live_date", "h_customer_id", "customer_nationality", "origin_country_code", "original_payment_currency", "is_user_logged_in"], inplace=True)
    X = (X - X.min()) / (X.max() - X.min())


    """
    y[y.notnull() == False] = -1
    y[y != -1] = 1
    """

    y = y.notnull().astype(int)

    #X = np.float64(X)
    #y = np.float64(y)

    return X, y, bk_id


def generate_model():
    model = Sequential()
    model.add(tf.keras.layers.Dense(28, activation='relu'))
    model.add(tf.keras.layers.Dense(15, activation='relu'))
    model.add(tf.keras.layers.Dense(15, activation='relu'))
    model.add(tf.keras.layers.Dense(2, activation='sigmoid'))
    model.compile(
    loss=tf.keras.losses.binary_crossentropy,
    optimizer=tf.keras.optimizers.Adam(lr=0.03),
    metrics=[
        tf.keras.metrics.BinaryAccuracy(name='accuracy'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall')
    ])

    return model


def encode(y: np.array):
    y_t = np.zeros(shape=(len(y), 2))
    y_t[:, 0] = y
    y_t[:, 1] = 1 - y
    return y_t

def decode(y: np.array):
    return (y[:, 0] > y[:, 1])*1

def prepare_data(df: pd.DataFrame):
    y = df.pop("cancellation_datetime")
    X = df
    X, y, id = preprocess_data(X, y)

    X_tr, y_tr, X_tst, y_tst = split_train_test(X, y)


    id_tr = id[y.index]
    id_tst = id[y_tst.index]

    y_tr = encode(y_tr)
    y_tst = encode(y_tst)

    return X_tr, y_tr, X_tst, y_tst, id_tr, id_tst


def predict_and_test(X, model, id, test_num):
    pred = model.predict(X)

    pred = decode(pred)
    pd_df = pd.DataFrame({"id":id, "vals": pred})
    pd_df.to_csv(index=False, path_or_buf=f'pred_tst_{test_num}.csv')

def evaluate(X, y, model):
    pred = model.predict(X)
    y = decode(y)
    pred = decode(pred)
    print("correctly classified:", np.sum(y != pred)/len(y))

if __name__ == "__main__":
    np.random.seed(0)
    df = pandas.read_csv("agoda_cancellation_train.csv")

    X_tr, y_tr, X_tst, y_tst, id_tr, id_tst = prepare_data(df)

    model = generate_model()
    model.fit(X_tr, y_tr, epochs=100)#, verbose=True, steps_per_epoch=200)

    evaluate(X_tst, y_tst, model)

    X = pandas.read_csv("Agoda_Test_1.csv")
    X, y, id = preprocess_data(X, pd.Series())
    predict_and_test(X, model, id, 1)




