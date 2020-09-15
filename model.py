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
from keras.applications.vgg19 import VGG19
from keras.applications.vgg16 import VGG16

DEFAULT_STYLE_LAYERS = [
    'block1_conv1', 'block2_conv1',
    'block3_conv1', 'block4_conv1',
]
DEFAULT_LAST_LAYER = 'block4_conv1'
class AdaptiveInstanceNorm(Layer):
    def __init__(self, epsilon=1e-3):
        super(AdaptiveInstanceNorm, self).__init__()
        self.epsilon = epsilon


    def call(self, inputs):
        x, style = inputs
        axis = [1, 2]
        x_mean = K.mean(x, axis=axis, keepdims=True)
        x_std = K.std(x, axis=axis, keepdims=True)

        style_mean = K.mean(style, axis=axis, keepdims=True)
        style_std = K.std(style, axis=axis, keepdims=True)

        norm = (x - x_mean) * (1 / (x_std + self.epsilon))

        return norm * (style_std + self.epsilon) + style_mean


    def compute_output_shape(self, input_shape):
        return input_shape[0]


class Reduction(Layer):
    def __init__(self):
        super(Reduction, self).__init__()

    def call(self, inputs):
        return tf.reduce_sum(inputs, axis=0)

class StyleTransferModel:
    UP_DECONV = 1
    UP_NEAREAST = 2

    def __init__(self, base_dir, rst, lr,
                style_layer_names=DEFAULT_STYLE_LAYERS,
                last_layer=DEFAULT_LAST_LAYER,
                skip_conts=DEFAULT_STYLE_LAYERS,
                show_interval=25,
                style_loss_weight=1):
        self.base_dir = base_dir
        self.rst = rst
        self.lr = lr
        self.style_layer_names = style_layer_names
        self.last_layer = last_layer
        self.skip_conts = skip_conts
        self.show_interval = show_interval
        img_shape = (self.rst, self.rst, 3)

        # ===== Build the model ===== #
        self.encoder = self.build_encoder()
        self.style_layers = self.build_style_layers()
        content_img = Input(shape=img_shape)
        style_img = Input(shape=img_shape)

        content_feat = self.encoder(content_img)
        style_feat = self.encoder(style_img)

        combined_feat = AdaptiveInstanceNorm()([content_feat, style_feat])
        self.init_rst = K.int_shape(combined_feat)[1]
        self.decoder = self.build_decoder((self.init_rst, self.init_rst, 512))

        gen_img = self.decoder(combined_feat)
        gen_feat = self.encoder(gen_img)

        # style loss
        

        self.transfer_model = Model(inputs=[content_img, style_img],
                                    outputs=gen_img)
        self.transfer_model.add_loss(K.mean(K.square(combined_feat - gen_feat)))
        self.transfer_model.add_loss(style_loss_weight*self.compute_style_loss(gen_img, style_img))
        self.transfer_model.compile(optimizer=Adam(self.lr),
                                    loss=["mse"],
                                    loss_weights=[0])


    def compute_style_loss(self, gen_img, style_img):
        gen_feats = self.style_layers(gen_img)
        style_feats = self.style_layers(style_img)
        style_loss = []
        axis = [1, 2] # instance norm
        for i in range(len(style_feats)):
            gmean = K.mean(gen_feats[i], axis=axis)
            gstd = K.std(gen_feats[i], axis=axis)

            smean = K.mean(style_feats[i], axis=axis)
            sstd = K.std(style_feats[i], axis=axis)

            style_loss.append(
                K.mean(K.square(gmean - smean)) +
                K.mean(K.square(gstd - sstd))
            )

        return Reduction()(style_loss)


    def build_style_layers(self):
        return Model(
            inputs=self.encoder.inputs,
            outputs=[self.encoder.get_layer(l).get_output_at(0) \
                for l in self.style_layer_names]
        )


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

        return Model(
            inputs=model.inputs,
            outputs=model.get_layer(self.last_layer).get_output_at(0)
        )


    def conv_block(self, x, filters, kernel_size,
                    activation, batch_norm=False,
                    upsampling_mode=UP_NEAREAST,
                    conv_layers=1, skip_cont=None):


        for i in range(conv_layers):
            x = Conv2D(filters, kernel_size=kernel_size, strides=1,
                        padding='same', activation=activation)(x)

        # if skip_cont is not None:
            # x = Add()([x, skip_cont])
        if batch_norm:
            x = BatchNormalization()(x)
        x = UpSampling2D(size=(2, 2), interpolation='nearest')(x)

        return x


    def iterations(self):
        i = 0
        init_rst = self.init_rst
        while init_rst != self.rst:
            i += 1
            init_rst *= 2
        
        return i


    def build_decoder(self, input_shape, upsampling_mode=UP_NEAREAST):
        feat = Input(input_shape)
        init_channel = 256
        kernel_size = 3
        up_iterations = self.iterations()

        x = self.conv_block(feat, 512, kernel_size=kernel_size,
                              activation='relu',
                              upsampling_mode=upsampling_mode,
                              conv_layers=2,
                              skip_cont=self.encoder.get_layer(self.skip_conts[0]).get_output_at(0))

        for i in range(1, up_iterations):
            x = self.conv_block(x, init_channel, kernel_size=kernel_size,
                                  activation='relu',
                                  upsampling_mode=upsampling_mode,
                                  conv_layers=3,
                                  skip_cont=self.encoder.get_layer(self.skip_conts[i]).get_output_at(0))
            init_channel //= 2

        x = Conv2D(init_channel, kernel_size=kernel_size, strides=1,
                   activation='relu', padding='same')(x)
        x = Conv2D(init_channel, kernel_size=kernel_size, strides=1,
                   activation='relu', padding='same')(x)
        style_image = Conv2D(3, kernel_size=1, strides=1,
                   activation='tanh', padding='same')(x)

        model = Model(inputs=feat, outputs=style_image, name='decoder')
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
                loss = self.transfer_model.train_on_batch([content_img, style_img],
                                                          style_img)
                batch_loss['loss'].append(loss)

            # evaluate
            # batch_loss['val_loss'] = 

            mean_loss = np.mean(np.array(batch_loss['loss']))
            mean_val_loss = 0#np.mean(np.array(batch_loss['val_loss']))

            history['loss'].append(mean_loss)
            history['val_loss'].append(mean_val_loss)

            print("Loss: {}, Val Loss: {} - {}".format(
                mean_loss, mean_val_loss,
                datetime.datetime.now() - start_time
            ))

            if e % self.show_interval == 0:
                idx = np.random.randint(0, 400)
                cimg, simg = data_gen.x[idx:idx+1], data_gen.y[idx:idx+1]
                self.show_sample(cimg, simg)

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
        self.transfer_model.save_weights(self.base_dir + '/transfer_model.h5')


    def load_weight(self):
        self.transfer_model.load_weights(self.base_dir + '/transfer_model.h5')


    def generate(self, content_imgs, style_imgs):
        return self.transfer_model.predict([content_imgs, style_imgs])


    def show_sample(self, content_img, style_img):
        gen_img = self.generate(content_img, style_img)
        utils.show_images(np.concatenate([content_img, style_img, gen_img]))