import os

from data_utils import train_generator, val_batch_generator
from model_utils import generate_mobilenet_model
from train_utils import TensorBoardBatch

from keras.callbacks import ModelCheckpoint


nb_train_images = 60000  # there are 82783 images in MS-COCO, set this to how many samples you want to train on.
batch_size = 125

model = generate_mobilenet_model(lr=1e-3)
model.summary()

# continue training if weights are available
if os.path.exists('weights/model.h5'):
    model.load_weights('weights/model.h5')

# use Batchwise TensorBoard callback
tensorboard = TensorBoardBatch(batch_size=batch_size)
checkpoint = ModelCheckpoint('weights/mobilenet_model.h5', monitor='loss', verbose=1,
                             save_best_only=True, save_weights_only=True)
callbacks = [checkpoint, tensorboard]


model.fit_generator(generator=train_generator(batch_size),
                    steps_per_epoch=nb_train_images // batch_size,
                    epochs=100,
                    verbose=1,
                    callbacks=callbacks,
                    validation_data=val_batch_generator(batch_size),
                    validation_steps=1
                    )
