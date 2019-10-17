import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, TimeDistributed, LSTM, Input
from tensorflow.keras.optimizers import Adam

def create_generator(timesteps, features):
    model = Sequential()
    model.add(LSTM(100, activation='relu',input_shape=(timesteps, features), return_sequences=True))
    model.add(LSTM(100, activation='relu',input_shape=(timesteps, features), return_sequences=True))
    model.add(TimeDistributed(Dense(features)))
    model.compile(loss='binary_crossentropy', optimizer='adam')
    return model

def create_discriminator(timesteps, features):
    model = Sequential()
    model.add(LSTM(100, activation='relu',input_shape=(timesteps, features), return_sequences=True))
    model.add(LSTM(100, activation='relu',input_shape=(timesteps, features), return_sequences=False))
    model.add(Dense(units=1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam')
    return model

def create_gan(discriminator, generator, timesteps, features):
    discriminator.trainable=False
    z = Input(shape=(timesteps, features))
    x = generator(z)
    gan_output = discriminator(x)
    gan = Model(inputs=z, outputs=gan_output)
    gan.compile(loss='binary_crossentropy', optimizer='adam')
    return gan