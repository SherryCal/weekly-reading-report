from __future__ import print_function, division
from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import RMSprop
from keras.layers.advanced_activations import LeakyReLU
import matplotlib.pyplot as plt
import sys
import numpy as np
import keras.backend as K

class WGAN():
    def __init__(self):
        self.img_rows = 28
        self.img_cols = 28
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 100
        self.n_critic = 5
        self.clip_value = 0.01
        optimizer = RMSprop(lr=0.0002)
        self.critic = self.build_critic
        self.critic.compile(loss=self.wasserstein_loss, optimizer=optimizer, metrics=['accuracy'])
        self.generator = self.build_generator()
        z = Input(shape=(self.latent_dim,))
        img = self.generator(z)
        self.critic.trainable = False
        valid = self.critic(img)
        self.combined = Model(z, valid)
        self.combined.compile(loss=self.wasserstein_loss, optimizer=optimizer, metrics=['accuracy'])
    def wasserstein_loss(self, y_true, y_pred):
        return K.mean(y_true * y_pred)
    def build_generator(self):
        model = Sequential()
        model.add(Dense(128*7*7, activation="relu", input_dim=self.latent_dim))
        model.add(Reshape((7, 7, 128)))
        model.add(UpSampling2D())
        model.add(Conv2D(128, kernel_size=4, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(UpSampling2D())
        model.add(Conv2D(64, kernel_size=4, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(Conv2D(self.channels, kernel_size=4, padding="same"))
        model.add(Activation("tanh"))
        model.summary()
        noise = Input(shape=(self.latent_dim,))
        img = model(noise)
        return Model(noise, img)
    @property
    def build_critic(self):
        model = Sequential()
        model.add(Conv2D(16,kernel_size=3,strides=2,input_shape=self.img_shape,padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(32, kernel_size=3,strides=2, padding="same"))
        model.add(ZeroPadding2D(padding=((0,1),(0,1))))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dropout(0.25))
        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation(LeakyReLU(alpha=0.2)))
        model.add(Dropout(0.25))
        model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation(LeakyReLU(alpha=0.2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(1))
        model.summary()
        img = Input(shape=self.img_shape)
        validity=model(img)
        return Model(img,validity)
    def train(self,epochs,batch_size=128,sample_interval=50):
        (X_train, _), (_, _) = mnist.load_data()
        X_train = (X_train.astype(np.float32)-127.5)/127.5
        X_train = np.expand_dims(X_train, axis=3)
        valid = -np.ones((batch_size, 1))
        fake = np.ones((batch_size, 1))
        for epoch in range(epochs):
            for _ in range(self.n_critic):
                idx = np.random.randint(0, X_train.shape[0], batch_size)
                imgs = X_train[idx]
                noise = np.random.normal(0,1,(batch_size,self.latent_dim))
                gen_imgs = self.generator.predict(noise)
                d_loss_real = self.critic.train_on_batch(imgs,valid)
                d_loss_fake = self.critic.train_on_batch(gen_imgs, fake)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
                for l in self.critic.layers:
                    weights = l.get_weights()
                    weights = [np.clip(w, -self.clip_value, self.clip_value) for w in weights]
                    l.set_weights(weights)
            g_loss = self.combined.train_on_batch(noise,valid)
            print("%d [D loss: %f] [G loss: %f]" % (epoch, d_loss[0], g_loss[0]))
            if epoch % sample_interval == 0:
                self.sample_images(epoch)
    def sample_images(self,epoch):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        gen_imgs = self.generator.predict(noise)
        gen_imgs = 0.5 * gen_imgs + 0.5
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
                axs[i, j].axis('off')
                cnt += 1
        fig.savefig("images/mnist_%d.png" % epoch)
        plt.close()


if __name__ == '__main__':
    wgan=WGAN()
    wgan.train(epochs=3000, batch_size=10, sample_interval=3)

