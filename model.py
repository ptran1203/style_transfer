import tensorflow as tf
import keras
import numpy as np
import datetime
import matplotlib.pyplot as plt
import utils
import keras.backend as K

from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers import (
    Input, Dense, Reshape,
    Flatten, Dropout,
    BatchNormalization, Activation,
    Lambda, Layer, Add, Concatenate,
    Average, UpSampling2D,
    MaxPooling2D, AveragePooling2D,
    GlobalMaxPooling2D, GlobalAveragePooling2D,
)
from keras.models import Sequential, Model, model_from_json
from keras.optimizers import Adam
from keras.applications.vgg16 import VGG16


class AdaptiveInstanceNorm(Layer):
    def __init__(self, epsilon=1e-3):
        super(AdaptiveInstanceNorm, self).__init__()
        self.epsilon = epsilon


    def call(self, inputs):
        x, style = inputs
        N, H, W, C = x.shape

        axis = [1, 2]
        x_mean = K.mean(x, axis=axis, keepdims=True)
        x_std = K.std(x, axis=axis, keepdims=True)

        style_mean = K.mean(style, axis=axis, keepdims=True)
        style_std = K.std(style, axis=axis, keepdims=True)

        norm = (x - x_mean) * (1 / (x_std + self.epsilon))

        return norm * (style_std + self.epsilon) + style_mean


    def compute_output_shape(self, input_shape):
        return input_shape[0]


class StyleTransferModel:
    UP_DECONV = 1
    UP_NEAREAST = 2

    def __init__(self, rst, lr):
        self.rst = rst
        self.lr = lr
        img_shape = (self.rst, self.rst, 3)

        # ===== Build the model ===== #
        self.encoder = self.build_encoder()
        content_img = Input(shape=img_shape)
        style_img = Input(shape=img_shape)

        content_feat = self.encoder(content_img)
        style_feat = self.encoder(style_img)

        combined_feat = AdaptiveInstanceNorm()([content_feat, style_feat])
        self.decoder = self.build_decoder((8, 8, 512))

        gen_img = self.decoder(combined_feat)
        gen_feat = self.encoder(gen_img)

        self.combined_model = Model(inputs=[content_img, style_img],
                                    outputs=gen_feat)
        self.combined_model.add_loss(K.mean(K.square(combined_feat - gen_feat)))
        self.combined_model.compile(optimizer=Adam(self.lr), loss="mse")

        self.transfer_model = Model(inputs=[content_img, style_img],
                                    outputs=gen_img)


    def build_encoder(self):
        input_shape = (self.rst, self.rst, 3)
        model = VGG16(
            include_top=False,
            weights='imagenet',
            input_tensor=Input(input_shape),
            input_shape=input_shape,
        )
        model.trainable = False
        for layer in model.layers:
            layer.trainable = False

        return model

    def decode_block(self, x, filters, kernel_size,
                    activation, batch_norm=False,
                    upsampling_mode=UP_NEAREAST):

        if upsampling_mode == self.UP_NEAREAST:
            x = Conv2DTranspose(filters, kernel_size=kernel_size,
                                strides=2, padding='same',
                                activation=activation)

            if batch_norm:
                x = BatchNormalization()(x)
        else:
            # de-convolution
            x = Conv2D(filters, kernel_size=kernel_size, strides=2,
                        padding='same', activation=activation)
            if batch_norm:
                x = BatchNormalization()(x)
            x = UpSampling2D(size=(2, 2), interpolation='nearest')(x)

        return x


    def build_decoder(self, input_shape, upsampling_mode=UP_NEAREAST):
        feat = Input(input_shape)
        init_feat_height = self.rst // 32
        init_channel = 256
        kernel_size = 5
        up_iterations = int(np.log(32 / np.log(init_feat_height)))

        x = self.decode_block(feat, 512, kernel_size=kernel_size,
                              activation='relu',
                              upsampling_mode=upsampling_mode)

        for i in range(up_iterations - 1):
            x = self.decode_block(x, init_channel, kernel_size=kernel_size,
                                  activation='relu',
                                  upsampling_mode=upsampling_mode)

        style_image = Conv2D(3, kernel_size=kernel_size, strides=1,
                   activation='tanh', padding='same')(x)

        model = Model(inputs=input_tensor, outputs=style_image, name='decoder')
        return model


    @staticmethod
    def init_hist():
        return {
            "loss": [],
            "val_loss": []
        }


    def train(self, data_gen, epochs, augment_factor=0):
        history = self.init_hist()
        print("Train on {} samples".format(len(data_gen.x)))

        for e in range(epochs):
            start_time = datetime.datetime.now()
            print("Train epochs {}/{} - ".format(e + 1, epochs), end="")

            batch_loss = self.init_hist()
            for content_img, style_img in data_gen.next_batch(augment_factor):
                style_feat = self.encoder.predict(style_img)
                loss = self.combined_model.train_on_batch([content_img, style_img],
                                                          style_feat)
                batch_loss['loss'].append(loss)

        # evaluate
        # batch_loss['val_loss'] = 

        mean_loss = np.mean(np.array(batch_loss['loss']))
            mean_val_loss = np.mean(np.array(batch_loss['val_loss']))

            history['loss'].append(mean_loss)
            history['val_loss'].append(mean_val_loss)

            print("Loss: {}, Val Loss: {} - {}".format(
                mean_loss, mean_val_loss,
                datetime.datetime.now() - start_time
            ))

        self.history = history
        return history


    def plot_history(self):
        plt.plot(self.history['loss'], label='train loss')
        plt.plot(self.history['val_loss'], label='val loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.title('Segmentation model')
        plt.legend()
        plt.show()

    def save_weight(self):
        self.ae_model.save_weights(self.base_dir + '/decoder.h5')

    def load_weight(self):
        self.ae_model.load_weights(self.base_dir + '/decoder.h5')


    def generate(self, content_imgs, style_imgs):
        return self.transfer_model.predict([content_imgs, style_imgs])


    def show_sample(self, content_img, style_img):
        gen_img = self.generate(content_img, style_img)

        plt.figure(figsize=(18, 10))
        plt.subplot(131)
        plt.imshow(content_img[0])
        plt.subplot(132)
        plt.imshow(style_img[0])
        plt.subplot(133)
        plt.imshow(gen_img[0])
        plt.show()
