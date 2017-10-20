import os
from data_utils import train_generator, val_batch_generator
from model_utils import generate_model

from keras.callbacks import ModelCheckpoint, TensorBoard
import tensorflow as tf

nb_train_images = 60000  # there are 82783 images in MS-COCO, set this to how many samples you want to train on.
batch_size = 125

model = generate_model(lr=1e-3)
model.summary()

# continue training if weights are available
if os.path.exists('weights/model.h5'):
    model.load_weights('weights/model.h5')

'''
Below is a modification to the TensorBoard callback to perform 
batchwise writing to the tensorboard, instead of only at the end
of the batch.
'''
class TensorBoardBatch(TensorBoard):
    def __init__(self, *args, **kwargs):
        super(TensorBoardBatch, self).__init__(*args)

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}

        for name, value in logs.items():
            if name in ['batch', 'size']:
                continue
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value.item()
            summary_value.tag = name
            self.writer.add_summary(summary, batch)

        self.writer.flush()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        for name, value in logs.items():
            if name in ['batch', 'size']:
                continue
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value.item()
            summary_value.tag = name
            self.writer.add_summary(summary, epoch * self.batch_size)

        self.writer.flush()

# use above Batchwise TensorBoard callback
tensorboard = TensorBoardBatch(batch_size=batch_size)
checkpoint = ModelCheckpoint('weights/model.h5', monitor='loss', verbose=1,
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
