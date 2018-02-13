import os
import numpy as np

from keras.preprocessing.image import img_to_array, load_img
from data_utils import prepare_input_image_batch, postprocess_output, resize
from model_utils import generate_mobilenet_model


IMAGE_FOLDER_PATH = r"D:\Yue\Documents\Datasets\MSCOCO\val\valset\\"
batch_size = 10
image_size = 256

model = generate_mobilenet_model(img_size=image_size)
model.load_weights('weights/mobilenet_model_v2.h5')

X = []
files = os.listdir(IMAGE_FOLDER_PATH)

files = files[:100]
for i, filename in enumerate(files):
    img = img_to_array(load_img(os.path.join(IMAGE_FOLDER_PATH, filename))) / 255.
    img = resize(img, (image_size, image_size, 3)) * 255.  # resize needs floats to be in 0-1 range, preprocess needs in 0-255 range
    X.append(img)

    if i % (len(files) // 20) == 0:
        print("Loaded %0.2f percentage of images from directory" % (i / float(len(files)) * 100))

X = np.array(X, dtype='float32')
print("Images loaded. Shape = ", X.shape)

X_lab, X_features = prepare_input_image_batch(X, batchsize=batch_size)
predictions = model.predict([X_lab, X_features], batch_size, verbose=1)

postprocess_output(X_lab, predictions, image_size=image_size)

