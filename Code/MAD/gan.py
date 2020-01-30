import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, TimeDistributed, LSTM, Input, Bidirectional
from tensorflow.keras.optimizers import Adam
tf.keras.backend.set_floatx('float64')

def create_generator(timesteps, features, units, layers):
    model = Sequential()
    for layer in range(layers):
        model.add(Bidirectional(LSTM(units, activation='relu',input_shape=(timesteps, features), return_sequences=True)))
    model.add(TimeDistributed(Dense(features)))
    #model.compile(loss='binary_crossentropy', optimizer='adam')
    return model

def create_discriminator(timesteps, features, units, layers, optimizer):
    model = Sequential()
    if layers > 1:
        for layer in range(layers - 1):
            model.add(LSTM(units, activation='relu',input_shape=(timesteps, features), 
                                         return_sequences=True))
    model.add(LSTM(units, activation='relu',input_shape=(timesteps, features), 
                                 return_sequences=False))
    model.add(Dense(units=1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=optimizer)
    return model

def create_gan(discriminator, generator, timesteps, features, optimizer):
    discriminator.trainable=False
    z = Input(shape=(timesteps, features))
    x = generator(z)
    gan_output = discriminator(x)
    gan = Model(inputs=z, outputs=gan_output)
    gan.compile(loss='binary_crossentropy', optimizer=optimizer)
    return gan

def training(samples, generator, discriminator, gan, train_writer, epochs=20, batch_size=16):
    
    nbr_samples = batch_size
    #nbr_samples = samples.shape[0]
    steps = samples.shape[1]
    features = samples.shape[2]
    
    fake = np.zeros(nbr_samples)        # Labels for fake data
    valid = 0.9 * np.ones(nbr_samples)  # Labels for real data
    
    for e in range(1, epochs+1):
        
        # ---------------------
        # Generate and get data
        # ---------------------
                    
        # Generate fake samples
        noise = np.random.normal(0, 1, [nbr_samples, steps, features])
        G_z = generator.predict(noise) 
        # Get the sampled real series
        idx = np.random.randint(0, samples.shape[0], batch_size)
        X = samples[idx]
                  
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
        noise = np.random.normal(0,1, [nbr_samples, steps, features])
            
        # Training the GAN by alternating the training of the discriminator          
        # and training the chained GAN model with discriminator’s weights freezed.
        g_loss = gan.train_on_batch(noise, valid)
         
        # -------------------------------
        #  Write to Tensorboard
        # -------------------------------
        
        with train_writer.as_default():
                tf.summary.scalar('d_loss', d_loss, step=e)
                tf.summary.scalar('g_loss', g_loss, step=e)  
