import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, TimeDistributed, LSTM, Input, Bidirectional
from tensorflow.keras.optimizers import Adam

def wasserstein_loss(y_true, y_pred):
    return tf.keras.backend.mean(y_true * y_pred)

def create_generator(timesteps, features, units, layers):
    model = Sequential()
    for layer in range(layers):
        model.add(Bidirectional(LSTM(units, activation='relu',input_shape=(timesteps, features), return_sequences=True)))
    model.add(TimeDistributed(Dense(features)))
    return model

def create_discriminator(timesteps, features, units, layers, optimizer):
    model = Sequential()
    if layers > 1:
        for layer in range(layers - 1):
            model.add(LSTM(units, activation='relu',input_shape=(timesteps, features), 
                                         return_sequences=True))
    model.add(LSTM(units, activation='relu',input_shape=(timesteps, features), 
                                 return_sequences=False))
    model.add(Dense(units=1, activation=None))
    model.compile(loss=wasserstein_loss, optimizer=optimizer)
    return model

def create_gan(discriminator, generator, timesteps, features, optimizer):
    z = Input(shape=(timesteps, features))
    x = generator(z)
    gan_output = discriminator(x)
    gan = Model(inputs=z, outputs=gan_output)
    gan.compile(loss=wasserstein_loss, optimizer=optimizer)
    return gan

def training(samples, generator, discriminator, gan, train_writer, epochs=20, batch_size=16, n_critic=5, clip_value=0.05):
    
    nbr_samples = batch_size
    timesteps = samples.shape[1]
    features = samples.shape[2]
    
    fake = np.ones(nbr_samples)        # Labels for fake data
    valid = -np.ones(nbr_samples)      # Labels for real data
    
    for e in range(1, epochs+1):
        
        for _ in range(n_critic):
            # ---------------------
            # Generate and get data
            # ---------------------

            # Generate fake samples
            noise = np.random.normal(0, 1, [nbr_samples, timesteps, features])
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
            
            # Clip critic weights
            for l in discriminator.layers:
                weights = l.get_weights()
                weights = [np.clip(w, -clip_value, clip_value) for w in weights]
                l.set_weights(weights)
            
        # ------------------
        #  Train Generator
        # ------------------
            
        discriminator.trainable = False
            
        # Tricking the noised input of the generator as real data
        #noise = np.random.normal(0,1, [nbr_samples, timesteps, features])
            
        # Training the GAN by alternating the training of the discriminator          
        # and training the chained GAN model with discriminator’s weights freezed.
        g_loss = gan.train_on_batch(noise, valid)
         
        # -------------------------------
        #  Write to Tensorboard
        # -------------------------------
        
        with train_writer.as_default():
                tf.summary.scalar('d_loss', d_loss, step=e)
                tf.summary.scalar('g_loss', g_loss, step=e)  
                #tf.summary.scalar('d_loss_fake', d_loss, step=e)
                #tf.summary.scalar('d_loss_real', d_loss, step=e)


