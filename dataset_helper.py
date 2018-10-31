# coding: utf8

from keras.utils import np_utils
import numpy as np
import pandas as pd
import os, sys, time, datetime, cv2, random, shutil


def numpy_save(samples, labels, name):
    np.savez_compressed(name, samples, labels)

def numpy_restore(filename):
    if os.path.exists(filename):
        data = np.load(filename)
        samples = data["arr_0"]
        labels = data["arr_1"]
        return samples, labels

def labelfile_append(name="pupkin", filename="/opt/Project/dataset/labelfile.csv"):
    if not os.path.exists(filename):
        raise Exception("%s does not exist" %(filename))

    strtime = datetime.datetime.strftime(datetime.datetime.now(), "%Y%m%d%H%M%S")

    set = pd.read_csv(filename, sep=":")

    set = set.append([{"name":name,"time":strtime}], ignore_index=True)

    set.to_csv(filename, sep=":", index=False)

    max_index = set.last_valid_index()

    print name, max_index, np_utils.to_categorical(max_index, 5000)

def labelfile_get_by_name(name="pupkin", filename="/opt/Project/dataset/labelfile.csv"):
    if not os.path.exists(filename):
        raise Exception("%s does not exist" %(filename))

    set = pd.read_csv(filename, sep=":")
    max_index = set.last_valid_index()

    for i in range(max_index+1):
        if set.loc[i].values[0] == name:
            return i, np_utils.to_categorical(i, 5000)

    #print index, np_utils.to_categorical(index, 5000)

def labelfile_get_by_index(index=2, filename="/opt/Project/dataset/labelfile.csv"):
    if not os.path.exists(filename):
        raise Exception("%s does not exist" %(filename))

    set = pd.read_csv(filename, sep=":")
    return set.loc[index].values[0]

def initial_make_small_dir_chunks(root_dset_path):
    trainbig_path = os.path.join(root_dset_path, "trainbig")
    train_path = os.path.join(root_dset_path, "train")
    test_path = os.path.join(root_dset_path, "test")
    validate_path = os.path.join(root_dset_path, "validate")

    for client_shortname in os.listdir(trainbig_path):
        source_path_fullname = os.path.join(trainbig_path, client_shortname)
        dst_path_fullname = os.path.join(validate_path, client_shortname)
        counter = 0
        for image_short_name in os.listdir(source_path_fullname):
            if counter == 3000:
                break
            image_source_full_name = os.path.join(source_path_fullname, image_short_name)

            try:
                cv2.imread(image_source_full_name)
            except:
                os.remove(image_source_full_name)
                continue

            newshortname = "%s_%s.png" %(client_shortname, counter)
            image_dst_full_name = os.path.join(dst_path_fullname, newshortname)

            os.rename(image_source_full_name, image_dst_full_name)

            counter += 1

def dataset_dump_expand():
    if os.path.exists("/opt/Project/dataset/labelfile.csv"):
        filename = "/opt/Project/dataset/labelfile.csv"
    elif os.path.exists("/opt/Projects/dataset/labelfile.csv"):
        filename = "/opt/Projects/dataset/labelfile.csv"

    other_path = "/opt/Projects/dataset/faces/other"
    dst_path = "/opt/Projects/dataset/faces"

    for item in os.walk(other_path):
        other_short_contents = item[2]

    # dirlist = ['alicia', 'angelina', 'batrudinov', 'blur', 'bryce', 'chernishevich', 'chuprak', 'deltoro', 'duglas', 'ella', 'famke', 'han', 'janice', 'jessa', 'jorjeva', 'junk', 'kalashnikov', 'kayden', 'klimovskiy', 'kostuk', 'lili', 'mia', 'monique', 'olsen', 'other', 'peaks', 'penya', 'perec', 'peta', 'pratt', 'priemka', 'rachel', 'riley', 'romi', 'rud', 'sisoeva', 'strange', 'tori', 'vershinina', 'vicenko', 'yanov', 'yaskovich']
    # for dir in dirlist:
    #     file.write( "%s:\n" %(dir) )
    # file = open(filename, "r")
    # file.close()

    for counter in range(178, 1001):
        move_path = os.path.join(dst_path, str(counter))

        for inner_couner in range(100):
            short_name = random.choice( other_short_contents )

            src_name = os.path.join(other_path, short_name)
            dst_name = os.path.join(move_path, short_name)

            shutil.copy2(src=src_name, dst=dst_name)





def dataset_assemble(path="/opt/Project/dataset/employee/"):

    for setname in ["test", "validate"]:
        set_path = os.path.join(path, setname)
        set_save = os.path.join(path, "%s.npz" %(setname))

        samples = []
        labels = []
        for label in os.listdir(set_path):
            labelencode, labelonehot = labelfile_get_by_name(label)
            images_label_dir = os.path.join(set_path, label)
            for img_short_name in os.listdir(images_label_dir):
                img_full_name = os.path.join(images_label_dir, img_short_name)
                img = cv2.imread(img_full_name)

                samples.append(img)
                labels.append(labelencode)

        samples = np.array(samples, dtype=np.uint8)
        samples = samples.reshape(samples.shape[0], 512, 512, 3)
        samples = samples.astype("float32")
        samples /= 255

        labels_onehot = np_utils.to_categorical(labels)

        numpy_save(samples, labels_onehot, set_save)

def dataset_smalllist_prepare(samplelist):
    samples = []
    labels = []

    for label, samplename in samplelist:
        if os.path.exists(samplename):
            #print label, samplename
            sample = cv2.imread(samplename)
            sample = cv2.resize(sample, (139,139))
            samples.append(sample)
            labels.append(label)

    samples = np.array(samples, dtype=np.float32)
    samples /= 255
    labels = np_utils.to_categorical(labels)
    print samples.shape, labels.shape

    return samples, labels

def dataset_list(path="/opt/Project/dataset/faces", items=10):
    if os.path.exists("/opt/Project/dataset/faces"):
        path = "/opt/Project/dataset/faces"
    elif os.path.exists("/opt/Projects/dataset/faces"):
        path = "/opt/Projects/dataset/faces"

    items = 10

    workdict = {}
    finallist = []

    for item in os.walk(path):
        dir, _, namelist = item
        #print dir, namelist

        if not namelist:
            continue

        workdict[dir] = namelist

    for item in range(items):

        train = []
        test = []
        validate = []

        for dir in workdict.keys():
            try:
                key = dir.split("/")[5]
                codekey = labelfile_get_by_name(key)[0]
                #print codekey
                namelist = workdict[dir]

            except:
                continue

            # if key == "junk" or key == "other" or key == "blur":
            #     test_counter = 100
            #     train_counter = 500
            # else:
            #     test_counter = 50
            #     train_counter = 200

            length = len(namelist)
            if not length:
                continue
            else:
                test_counter = length * 2 / 100
                train_counter = length * 6 / 100


            for counter in range(test_counter):
                try:
                    name = os.path.join( dir, namelist.pop() )
                    #if os.path.exists(name):
                        #print name, codekey
                    test.append( (codekey, name) )

                    name = os.path.join( dir, namelist.pop() )
                    #if os.path.exists(name):
                        #print name, codekey
                    validate.append( (codekey, name) )

                    #print len(test), len(validate)
                except:
                    break

            for counter in range(train_counter):
                try:
                    name = os.path.join(dir, namelist.pop())
                    # if os.path.exists(name):
                    #     print name, codekey
                    train.append( (codekey, name) )

                    #print len(train)
                except:
                    break

        print item, len(train), len(test), len(validate)
        if train and test and validate:
            finallist.append([train, test, validate])

    finallist.reverse()
    return finallist

def dataset_load(path="/opt/Project/dataset/faces/"):
    train_set = os.path.join(path, "train.npz")
    samples, labels = numpy_restore(train_set)
    # print samples.shape
    # print labels.shape, "\n"

    test_set = os.path.join(path, "test.npz")
    samples_test, labels_test = numpy_restore(test_set)
    # print samples_test.shape
    # print labels_test.shape, "\n"

    validate_set = os.path.join(path, "validate.npz")
    samples_validate, labels_validate = numpy_restore(validate_set)
    # print samples_validate.shape
    # print labels_validate.shape, "\n"

    return samples, labels, samples_test, labels_test, samples_validate, labels_validate

def resize_and_rename(source="/opt/Project/dataset/faces", dest="/mnt/nfs/webhost/Projects/dataset/faces"):
    for subdir in os.listdir(source):


        sourcedir = os.path.join(source, subdir)
        destdir = os.path.join(dest, subdir)

        if not os.path.isdir(destdir):
            os.mkdir(destdir)

        for short_name in os.listdir(sourcedir):
            full_name = os.path.join(sourcedir, short_name)

            timeform = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")
            newname = "%s/%s.jpg" %( destdir, timeform )

            img = cv2.imread(full_name)
            img = cv2.resize(img, (128,128))
            cv2.imwrite(newname, img)


if __name__ == "__main__":
    faces_path = "/opt/Project/dataset/faces/"
    faces_128_path = "/opt/Project/dataset/faces_128/"

    #initial_make_small_dir_chunks(faces_path)
    #dataset_assemble(faces_path)
    #dataset_load()

    # list = dataset_list(faces_128_path)
    # print len(list)
    # for train, test, validate in list:
    #     print len(train), len(test), len(validate)

    dataset_dump_expand()