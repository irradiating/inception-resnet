# coding: utf8

from keras.applications import InceptionResNetV2
from keras.applications import InceptionV3
from keras.models import Model
from keras.layers import Dense, Flatten
from keras.utils import plot_model
import os, cv2
import numpy as np
from dataset_helper import dataset_list, dataset_smalllist_prepare, labelfile_get_by_index


def inception_resnet_v2_model_detector():
    detector = InceptionResNetV2(include_top=False, weights="imagenet", input_shape=(139,139,3), pooling="none")
    #print detector.summary()
    #plot_model(model, to_file="inceptionv2.png", show_shapes=True)
    return detector

def inception_v3_model_detector():
    #print help(InceptionV3)
    detector = InceptionV3(include_top=False, weights="imagenet", input_shape=(139,139,3), pooling=None)
    #print model.summary()
    #plot_model(model, to_file="inception3.png", show_shapes=True)
    return detector


def inception_train_faces():
    batch_size = 20
    epochs = 200

    if os.path.exists(weights_filename):
        print "loading existing weights"
        model.load_weights(weights_filename)

    print "loading dataset"
    datalist = dataset_list(faces_path)
    for train, test, validate in datalist:
        samples, labels = dataset_smalllist_prepare(train)
        samples_test, labels_test = dataset_smalllist_prepare(test)
        samples_validate, labels_validate = dataset_smalllist_prepare(validate)
        print

        if labels.shape[1] != labels_test.shape[1]:
            continue
        elif labels.shape[1] != labels_validate.shape[1]:
            continue

        print "training model"
        model.fit(samples, labels, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(samples_validate, labels_validate))
        print "saving weights"
        model.save_weights(weights_filename, overwrite=True)

        try:
            print "evaluating model"
            loss, accuracy = model.evaluate(samples_test, labels_test, verbose=1)
            print "model loss: %s, accuracy: %s" %(loss, accuracy)
        except:
            continue

def use_model(model, samples):

    samples = np.array(samples, dtype=np.float32)

    samples = samples.reshape(samples.shape[0], image_shape[0], image_shape[1], image_shape[2])
    samples /= 255

    prediction = model.predict(samples)

    index = np.argmax(prediction, axis=1)
    confidence = prediction[0][index]

    #return np.argmax(prediction, axis=1)
    return confidence[0], index[0]

def cycle_recognize_move(model, path="/opt/Project/dataset/camface"):
    if os.path.exists(weights_filename):
        print "loading saved weights"
        model.load_weights(weights_filename)

    for short_name in os.listdir(path):
        full_name = os.path.join(path, short_name)

        img = cv2.imread(full_name)
        try:
            img = cv2.resize(img, (139,139))
        except:
            continue

        confidence, prediction = use_model(model, [img])
        name = labelfile_get_by_index(prediction)

        print name, confidence

        if confidence >= 0.9:
            name = "%s_09" %(name)
        elif confidence >= 0.8 and confidence < 0.9:
            name = "%s_08" %(name)
        elif confidence >= 0.7 and confidence < 0.8:
            name = "%s_07" %(name)
        else:
            name = "unconfident"

        dstdir = os.path.join(path, name)

        if not os.path.exists(dstdir):
            os.mkdir(dstdir)

        dst_filename = os.path.join(dstdir, short_name)
        os.rename(full_name, dst_filename)



def make_precompiled_model():

    detector = inception_resnet_v2_model_detector()

    # make detector layers not trainable
    for layer in detector.layers:
        # print layer
        layer.trainable = False

    x = Flatten()(detector.output)
    x = Dense(4096, activation="relu")(x)
    x = Dense(num_labels, activation="softmax")(x)

    model = Model(detector.input, x)

    model.compile(optimizer="rmsprop", loss="categorical_crossentropy")

    print model.summary()

    return model




if __name__ == "__main__":
    weights_filename = "inception_resnet_v2.hdf"
    faces_path = "/opt/Project/dataset/faces"
    num_labels = 188

    model = make_precompiled_model()

    inception_train_faces()