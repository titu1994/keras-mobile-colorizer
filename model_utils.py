from keras.layers import Conv2D, Input, Reshape, RepeatVector, concatenate, UpSampling2D, Flatten, Conv2DTranspose
from keras.models import Model

from keras import backend as K
from keras.optimizers import Adam

# shows the minimum value of the AB channels
def y_true_min(yt, yp):
    return K.min(yt)

# shows the maximum value of the RGB AB channels
def y_true_max(yt, yp):
    return K.max(yt)

# shows the minimum value of the predicted AB channels
def y_pred_min(yt, yp):
    return K.min(yp)

# shows the maximum value of the predicted AB channels
def y_pred_max(yt, yp):
    return K.max(yp)

def generate_mobilenet_model(lr=1e-3, img_size=128):
    '''
    Creates a Colorizer model. Note the difference from the report
    - https://github.com/baldassarreFe/deep-koalarization/blob/master/report.pdf

    I use a long skip connection network to speed up convergence and
    boost the output quality.
    '''
    # encoder model
    encoder_ip = Input(shape=(img_size, img_size, 1))
    encoder1 = Conv2D(64, (3, 3), padding='same', activation='relu', strides=(2, 2))(encoder_ip)
    encoder = Conv2D(128, (3, 3), padding='same', activation='relu')(encoder1)
    encoder2 = Conv2D(128, (3, 3), padding='same', activation='relu', strides=(2, 2))(encoder)
    encoder = Conv2D(256, (3, 3), padding='same', activation='relu')(encoder2)
    encoder = Conv2D(256, (3, 3), padding='same', activation='relu', strides=(2, 2))(encoder)
    encoder = Conv2D(512, (3, 3), padding='same', activation='relu')(encoder)
    encoder = Conv2D(512, (3, 3), padding='same', activation='relu')(encoder)
    encoder = Conv2D(256, (3, 3), padding='same', activation='relu')(encoder)

    # input fusion
    # Decide the image shape at runtime to allow prediction on
    # any size image, even if training is on 128x128
    batch, height, width, channels = K.int_shape(encoder)

    mobilenet_features_ip = Input(shape=(1000,))
    fusion = RepeatVector(height * width)(mobilenet_features_ip)
    fusion = Reshape((height, width, 1000))(fusion)
    fusion = concatenate([encoder, fusion], axis=-1)
    fusion = Conv2D(256, (1, 1), padding='same', activation='relu')(fusion)

    # decoder model
    decoder = Conv2D(128, (3, 3), padding='same', activation='relu')(fusion)
    decoder = UpSampling2D()(decoder)
    decoder = concatenate([decoder, encoder2], axis=-1)
    decoder = Conv2D(64, (3, 3), padding='same', activation='relu')(decoder)
    decoder = Conv2D(64, (3, 3), padding='same', activation='relu')(decoder)
    decoder = UpSampling2D()(decoder)
    decoder = concatenate([decoder, encoder1], axis=-1)
    decoder = Conv2D(32, (3, 3), padding='same', activation='relu')(decoder)
    decoder = Conv2D(2, (3, 3), padding='same', activation='tanh')(decoder)
    decoder = UpSampling2D((2, 2))(decoder)

    model = Model([encoder_ip, mobilenet_features_ip], decoder, name='Colorizer')
    model.compile(optimizer=Adam(lr), loss='mse', metrics=[y_true_max,
                                                         y_true_min,
                                                         y_pred_max,
                                                         y_pred_min])

    print("Colorization model built and compiled")
    return model


if __name__ == '__main__':
    model = generate_mobilenet_model()
    model.summary()

    from keras.utils.vis_utils import plot_model
    plot_model(model, to_file='skip_model.png', show_shapes=True)
