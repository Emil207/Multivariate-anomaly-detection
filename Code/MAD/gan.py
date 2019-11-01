import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, TimeDistributed, LSTM, Input
from tensorflow.keras.optimizers import Adam

def create_generator(timesteps, features, units):
    model = Sequential()
    model.add(LSTM(units, activation='relu',input_shape=(timesteps, features), return_sequences=True))
    model.add(LSTM(units, activation='relu',input_shape=(timesteps, features), return_sequences=True))
    model.add(TimeDistributed(Dense(features)))
    model.compile(loss='binary_crossentropy', optimizer='adam')
    return model

def create_discriminator(timesteps, features, units):
    model = Sequential()
    model.add(LSTM(units, activation='relu',input_shape=(timesteps, features), return_sequences=True))
    model.add(LSTM(units, activation='relu',input_shape=(timesteps, features), return_sequences=False))
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

def training(X_train, generator, discriminator, gan, samples, timesteps, features, train_writer, epochs=20):
    
    for e in range(1, epochs+1):
            
        # ---------------------
        # Generate and get data
        # ---------------------
                    
        # Generate fake samples
        noise = np.random.normal(0, 1, [samples, timesteps, features])
        G_z = generator.predict(noise) 
        # Get the sampled real series
        X = X_train
            
        # Labels for fake data
        fake = np.zeros(samples)
        # Labels for real data
        valid = 0.9 * np.ones(samples) 
            
        # ------------------------
        # Train the discriminator
        # ------------------------
            
        discriminator.trainable = True
        d_loss_real = discriminator.train_on_batch(X, valid)
        d_loss_fake = discriminator.train_on_batch(G_z, fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            
        # ------------------
        #  Train Generator
        # ------------------
            
        discriminator.trainable = False
            
        # Tricking the noised input of the generator as real data
        noise = np.random.normal(0,1, [samples, timesteps, features])
            
        # Training the GAN by alternating the training of the discriminator          
        # and training the chained GAN model with discriminator’s weights freezed.
        g_loss = gan.train_on_batch(noise, valid)
         
        # -------------------------------
        #  Write to Tensorboard and plot
        # -------------------------------
        
        with train_writer.as_default():
                tf.summary.scalar('d_loss', d_loss, step=e)
                tf.summary.scalar('g_loss', g_loss, step=e)        
       # if e % (epochs/10) == 0:
       #     plot_generated_series(generator, noise, timesteps, features)

    
def plot_generated_series(generator, noise, timesteps, features, dim=(10,10)):
    
    generated_series = generator.predict(noise)
    plt.figure(figsize=(10,10))
    
    for i in range(generated_series.shape[0]):
        plt.subplot(dim[0], dim[1], i+1)
        plt.plot(generated_series[i])
        plt.axis('off')
    plt.tight_layout()   