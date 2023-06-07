import pandas
import pandas as pd
"""
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Activation, Dense
"""
from ff import FF
from matplotlib import pyplot as plt

import numpy as np
from utils import split_train_test



def preprocess_data(X: pd.DataFrame, y: pd.Series):
    X['booking_day'] = pd.to_datetime(df['booking_datetime']).dt.dayofyear
    X['check_in'] = pd.to_datetime(df['checkin_date']).dt.dayofyear
    X['check_out'] = pd.to_datetime(df['checkout_date']).dt.dayofyear
    #X['hotel_county_code_one_hot'] = pd.Categorical(X['hotel_country_code']).cat.codes
    X['hotel_county_code_one_hot'] = pd.factorize(X['hotel_country_code'])[0]
    X['accommodation_type_name_one_hot'] = pd.factorize(X['accommadation_type_name'])[0]
    X['charge_option_one_hot'] = pd.factorize(X['charge_option'])[0]
    X['nationality_one_hot'] = pd.factorize(X['guest_nationality_country_name'])[0]
    X['language_one_hot'] = pd.factorize(X['language'])[0]
    X['payment_type_one_hot'] = pd.factorize(X['original_payment_type'])[0]
    X['cancellation_policy_one_hot'] = pd.factorize(X['cancellation_policy_code'])[0]
    X['is_first_booking'] = pd.factorize(X['is_first_booking'])[0]

    ls = {'request_nonesmoke':0.5, 'request_latecheckin':0.5, 'request_highfloor':0.5, 'request_largebed':0.5, 'request_twinbeds':0.5, 'request_airport':0.5, 'request_earlycheckin':0.5}
    X.fillna(ls, inplace=True)

    X.drop(columns=["hotel_country_code", "accommadation_type_name", "charge_option", "guest_nationality_country_name", "language", "original_payment_type", "cancellation_policy_code", "booking_datetime", "checkin_date", "checkout_date",  "hotel_brand_code", "hotel_chain_code", "original_payment_method","h_booking_id", "hotel_id", "hotel_area_code", "hotel_live_date", "h_customer_id", "customer_nationality", "origin_country_code", "original_payment_currency", "is_user_logged_in"], inplace=True)
    X.dropna(inplace=True)

    y = y.notnull().astype(int).fillna(0)

    X = np.float64(X)
    y = np.float64([y])

    return X, y


def generate_model():
    model = Sequential()
    model.add(tf.keras.Input(shape=(26,))) #256*256
    model.add(tf.keras.layers.Dense(10, activation='relu'))
    model.add(tf.keras.layers.Dense(5, activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='relu'))
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model

layers_sizes = [26, 10, 1] # flexible, but should be [784,...,10]
epochs = 10
eta = 0.1
batch_size = 20


if __name__ == "__main__":
    np.random.seed(0)
    df = pandas.read_csv("agoda_cancellation_train.csv")
    y = df.pop("cancellation_datetime")
    X = df
    X_tr, y_tr, X_tst, y_tst = split_train_test(X, y)
    X_tr, y_tr = preprocess_data(X_tr, y_tr)
    X_tst, y_tst = preprocess_data(X_tst, y_tst)

    """
    model = generate_model()
    model.fit(X_tr, y_tr, epochs=100000)#, verbose=True)  # , steps_per_epoch=50)
    print()
    """
    net = FF(layers_sizes)
    steps, test_acc = net.sgd(X_tr.T, y_tr.T, epochs, eta, batch_size, X_tst.T, y_tst.T)

    ## plotting learning curve and visualizing some examples from test set

    plt.plot(steps, test_acc, 'r')
    plt.xlabel('steps')
    plt.ylabel('acc')
    plt.show()