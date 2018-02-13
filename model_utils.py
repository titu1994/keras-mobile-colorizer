from keras.layers import Conv2D, Input, Reshape, RepeatVector, concatenate, UpSampling2D, Flatten, Conv2DTranspose
from keras.models import Model

from keras import backend as K
from keras.losses import mean_squared_error
from keras.optimizers import Adam

mse_weight = 1.0 #1e-3

# set these to zeros to prevent learning
perceptual_weight = 1. / (2. * 128. * 128.) # scaling factor
attention_weight = 1.0 # 1.0


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


def gram_matrix(x):
    assert K.ndim(x) == 4

    with K.name_scope('gram_matrix'):
        if K.image_data_format() == "channels_first":
            batch, channels, width, height = K.int_shape(x)
            features = K.batch_flatten(x)
        else:
            batch, width, height, channels = K.int_shape(x)
            features = K.batch_flatten(K.permute_dimensions(x, (0, 3, 1, 2)))

        gram = K.dot(features, K.transpose(features)) # / (channels * width * height)
    return gram


def l2_norm(x):
    return K.sqrt(K.sum(K.square(x)))


def attention_vector(x):
    if K.image_data_format() == "channels_first":
        batch, channels, width, height = K.int_shape(x)
        filters = K.batch_flatten(K.permute_dimensions(x, (1, 0, 2, 3)))  # (channels, batch*width*height)
    else:
        batch, width, height, channels = K.int_shape(x)
        filters = K.batch_flatten(K.permute_dimensions(x, (3, 0, 1, 2)))  # (channels, batch*width*height)

    filters = K.mean(K.square(filters), axis=0)  # (batch*width*height,)
    filters = filters / l2_norm(filters)  # (batch*width*height,)
    return filters


def total_loss(y_true, y_pred):
    mse_loss = mse_weight * mean_squared_error(y_true, y_pred)
    perceptual_loss = perceptual_weight * K.sum(K.square(gram_matrix(y_true) - gram_matrix(y_pred)))
    attention_loss = attention_weight * l2_norm(attention_vector(y_true) - attention_vector(y_pred))

    return mse_loss + perceptual_loss + attention_loss


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
    #decoder = Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same', activation='relu')(decoder)
    decoder = concatenate([decoder, encoder2], axis=-1)
    decoder = Conv2D(64, (3, 3), padding='same', activation='relu')(decoder)
    decoder = Conv2D(64, (3, 3), padding='same', activation='relu')(decoder)
    decoder = UpSampling2D()(decoder)
    #decoder = Conv2DTranspose(64, (4, 4), strides=(2, 2), padding='same', activation='relu')(decoder)
    decoder = concatenate([decoder, encoder1], axis=-1)
    decoder = Conv2D(32, (3, 3), padding='same', activation='relu')(decoder)
    decoder = Conv2DTranspose(2, (4, 4), strides=(2, 2), padding='same', activation='tanh')(decoder)
    # decoder = Conv2D(2, (3, 3), padding='same', activation='tanh')(decoder)
    # decoder = UpSampling2D((2, 2))(decoder)

    model = Model([encoder_ip, mobilenet_features_ip], decoder, name='Colorizer')
    model.compile(optimizer=Adam(lr), loss=total_loss, metrics=[y_true_max,
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
