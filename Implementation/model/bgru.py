## -*- coding: utf-8 -*-
'''
This python file is used to train four class focus data in blstm model

'''

import os
from preprocess_dl_Input_version5 import *
import tensorflow as tf
import tensorflow.keras.metrics as metrics
from convert_array import make_dim_same_length

RANDOMSEED = 2018  # for reproducibility
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"


def build_model(maxlen, vector_dim, layers, dropout):
    print('Build model...')

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Masking(mask_value=0.0, input_shape=(maxlen, vector_dim)))

    for i in range(1, layers):
        model.add(tf.keras.layers.Bidirectional(tf.keras.layers.GRU(units=256, activation='tanh', recurrent_activation='hard_sigmoid', return_sequences=True)))
        model.add(tf.keras.layers.Dropout(dropout))

    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.GRU(units=256, activation='tanh', recurrent_activation='hard_sigmoid')))
    model.add(tf.keras.layers.Dropout(dropout))

    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='adamax',
                  metrics=[metrics.Accuracy(),metrics.TruePositives(), metrics.FalsePositives(),
                           metrics.FalseNegatives(), metrics.Precision(), metrics.Recall()])
    # 'fbeta_score' not a metric in keras

    model.summary()

    return model


def make_or_restore_model(maxlen, vector_dim, layers, dropout):
    # Either restore the latest model, or create a fresh one
    # if there is no checkpoint available.
    checkpoints = [checkpoint_dir + "/" + name for name in os.listdir(checkpoint_dir)]
    if checkpoints:
        latest_checkpoint = max(checkpoints, key=os.path.getctime)
        print("Restoring from", latest_checkpoint)
        return tf.keras.models.load_model(latest_checkpoint)
    print("Creating a new model")
    return build_model(maxlen, vector_dim, layers, dropout)


def main(traindataSet_path, testdataset_path, realtestpath, weightpath, resultpath, batch_size, maxlen, vector_dim, layers, dropout, num_val_samples):
    print("\n Loading data...")
    train_dataset, val_dataset, test_dataset, all_test_samples = get_data(traindataSet_path,
                                                                          maxlen,
                                                                          vector_dim,
                                                                          batch_size,
                                                                          num_val_samples)

    print("\n Training")
    model = run_training(dropout, layers, maxlen, vector_dim, 1, train_dataset, val_dataset)

    print("\n Evaluate")
    result = model.evaluate(test_dataset)
    report_metrics(model.metrics_names, result,  all_test_samples, resultpath)


def run_training(dropout, layers, maxlen, vector_dim, epochs, train_dataset, val_dataset):
    strategy = tf.distribute.MirroredStrategy()
    print("Number of devices: {}".format(strategy.num_replicas_in_sync))
    # Open a strategy scope.
    with strategy.scope():
        # Everything that creates variables should be under the strategy scope.
        # In general this is only model construction & `compile()`.
        model = make_or_restore_model(maxlen, vector_dim, layers, dropout)

    callbacks = [
        # This callback saves a SavedModel every epoch
        # We include the current epoch in the folder name.
        # tf.keras.callbacks.TensorBoard(log_dir='./logs'),
        # tf.keras.callbacks.ModelCheckpoint(
        #     filepath=checkpoint_dir + "/ckpt-{epoch}", save_freq="epoch"
        # )
    ]
    model.fit(train_dataset,
              epochs=epochs,
              callbacks=callbacks,
              validation_data=val_dataset,
              verbose=2)

    return model


def get_data(traindataSet_path, maxlen, vector_dim, batch_size, num_val_samples):
    print("Training dataset")
    x_train = []
    y_train = []
    for filename in os.listdir(traindataSet_path):
        if (filename.endswith(".pkl") is False):
            continue
        print(filename)
        f = open(os.path.join(traindataSet_path, filename), "rb")
        dataset_file, labels_file, funcs_file, filenames_file, testcases_file = pickle.load(f)
        f.close()
        x_train += dataset_file
        y_train += labels_file

    bin_labels = []
    for label in y_train:
        bin_labels.append(multi_labels_to_two(label))
    y_train = bin_labels

    print("Test dataset")
    x_test = []
    y_test = []
    for filename in os.listdir(traindataSet_path):
        if (filename.endswith(".pkl") is False):
            continue
        print(filename)
        f = open(os.path.join(traindataSet_path, filename), "rb")
        datasetfile, labelsfile, funcsfiles, filenamesfile, testcasesfile = pickle.load(f)
        f.close()
        x_test += datasetfile
        y_test += labelsfile

    bin_labels = []
    for label in y_test:
        bin_labels.append(multi_labels_to_two(label))
    y_test = bin_labels

    x_train_np = make_dim_same_length(x_train, maxlen, vector_dim)
    y_train_np = np.array(y_train)
    x_test_np = make_dim_same_length(x_test, maxlen, vector_dim)
    y_test_np = np.array(y_test)

    # Reserve num_val_samples samples for validation
    x_val_np = x_train_np[-num_val_samples:]
    y_val_np = y_train_np[-num_val_samples:]
    x_train_np = x_train_np[:-num_val_samples]
    y_train_np = y_train_np[:-num_val_samples]

    return (
        tf.data.Dataset.from_tensor_slices((x_train_np, y_train_np)).batch(batch_size),
        tf.data.Dataset.from_tensor_slices((x_val_np, y_val_np)).batch(batch_size),
        tf.data.Dataset.from_tensor_slices((x_test_np, y_test_np)).batch(batch_size),
        len(x_test_np)
    )


def testrealdata(realtestpath, weightpath, batch_size, maxlen, vector_dim, layers, dropout):
    model = build_model(maxlen, vector_dim, layers, dropout)
    model.load_weights(weightpath)
    
    print("Loading data...")
    for filename in os.listdir(realtestpath):
        print(filename)
        f = open(realtestpath+filename, "rb")
        realdata = pickle.load(f,encoding="latin1")
        f.close()
    
        labels = model.predict(x = realdata[0],batch_size = 1)
        for i in range(len(labels)):
            if labels[i][0] >= 0.5:
                print(realdata[1][i])


def report_metrics(metric_names, result, all_test_samples, resultpath):
    metric_values = dict(zip(metric_names, result))
    print(metric_values)

    # score, tp, fp, fn, precision, recall, f_score = result[0]
    #score, accuracy, tp, fp, fn, precision, recall = result[0]
    # metrics = [metrics.Accuracy(), metrics.TruePositives(), metrics.FalsePositives(),
    #            metrics.FalseNegatives(), metrics.Precision(), metrics.Recall()])
    # f = open("TP_index_blstm.pkl", 'wb')
    # pickle.dump(result[1], f)
    # f.close()

    # f_TP = open("./result_analyze/BGRU/TP_filenames.txt", "ab+")
    # for i in range(len(result[1])):
    #     TP_index = result[1][i]
    #     f_TP.write(str(filenames[TP_index]) + '\n')
    #
    # f_FP = open("./result_analyze/BGRU/FP_filenames.txt", "ab+")
    # for j in range(len(result[2])):
    #     FP_index = result[2][j]
    #     f_FP.write(str(filenames[FP_index]) + '\n')
    #
    # f_FN = open("./result_analyze/BGRU/FN_filenames.txt", "a+")
    # for k in range(len(result[3])):
    #     FN_index = result[3][k]
    #     f_FN.write(str(filenames[FN_index]) + '\n')
    score = metric_values['loss']
    tp = metric_values['true_positives']
    recall = metric_values['recall']
    fp = metric_values['false_positives']
    precision = metric_values['precision']
    fn = metric_values['false_negatives']
    #accuracy = metric_values['accuracy']
    # score, tp, fp, fn, precision, recall, f_score = result[0]
    # {'loss': 0.07849834859371185, 'true_positives': 0.0, 'recall': 0.0, 'false_positives': 0.0, 'precision': 0.0,
    #  'false_negatives': 0.0, 'accuracy': 0.0}
    tn = all_test_samples - tp - fp - fn
    fwrite = open(resultpath, 'a')
    fwrite.write('metric_values: ' + ' ' + str(metric_values) + '\n')
    fwrite.write('cdg_ddg: ' + ' ' + str(all_test_samples) + '\n')
    fwrite.write("TP:" + str(tp) + ' FP:' + str(fp) + ' FN:' + str(fn) + ' TN:' + str(tn) + '\n')
    fpr = float(fp) / (fp + tn)
    fwrite.write('FPR: ' + str(fpr) + '\n')
    fnr = float(fn) / (tp + fn)
    fwrite.write('FNR: ' + str(fnr) + '\n')
    accuracy = float(tp + tn) / (all_test_samples)
    fwrite.write('Accuracy: ' + str(accuracy) + '\n')
    precision = float(tp) / (tp + fp)
    fwrite.write('precision: ' + str(precision) + '\n')
    recall = float(tp) / (tp + fn)
    fwrite.write('recall: ' + str(recall) + '\n')
    f_score = (2 * precision * recall) / (precision + recall)
    fwrite.write('fbeta_score: ' + str(f_score) + '\n')
    # fwrite.write('train_time:' + str(train_time) + '  ' + 'test_time:' + str(test_time) + '\n')
    fwrite.write('--------------------\n')
    fwrite.close()


# def main(traindataSet_path, testdataSet_path, realtestpath, weightpath, resultpath, batch_size, maxlen, vector_dim,
#          layers, dropout):
#     print("Loading data...")
#
#     model = build_model(maxlen, vector_dim, layers, dropout)
#
#     # model.load_weights(weightpath)  #load weights of trained model
#
#     print("Train...")
#     dataset = []
#     labels = []
#     testcases = []
#     for filename in os.listdir(traindataSet_path):
#         if not (filename.endswith(".pkl") is True):
#             continue
#         print(filename)
#         f = open(os.path.join(traindataSet_path, filename), "rb")
#         dataset_file, labels_file, funcs_file, filenames_file, testcases_file = pickle.load(f)
#         f.close()
#         dataset += dataset_file
#         labels += labels_file
#     print(len(dataset), len(labels))
#
#     bin_labels = []
#     for label in labels:
#         bin_labels.append(multi_labels_to_two(label))
#     labels = bin_labels
#
#     np.random.seed(RANDOMSEED)
#     np.random.shuffle(dataset)
#     np.random.seed(RANDOMSEED)
#     np.random.shuffle(labels)
#
#     train_generator = generator_of_data(dataset, labels, batch_size, maxlen, vector_dim)
#     all_train_samples = len(dataset)
#     steps_epoch = int(all_train_samples / batch_size)
#     print("start")
#     t1 = time.time()
#
#     model.fit_generator(train_generator, steps_per_epoch=steps_epoch, epochs=10)
#     t2 = time.time()
#     train_time = t2 - t1
#
#     model.save_weights(weightpath)
#
#     # model.load_weights(weightpath)
#     print("Test1...")
#     dataset = []
#     labels = []
#     testcases = []
#     filenames = []
#     funcs = []
#     for filename in os.listdir(traindataSet_path):
#         if (filename.endswith(".pkl") is False):
#             continue
#         print(filename)
#         f = open(os.path.join(traindataSet_path, filename), "rb")
#         datasetfile, labelsfile, funcsfiles, filenamesfile, testcasesfile = pickle.load(f)
#         f.close()
#         dataset += datasetfile
#         labels += labelsfile
#         testcases += testcasesfile
#         funcs += funcsfiles
#         filenames += filenamesfile
#     print(len(dataset), len(labels), len(testcases))
#
#     bin_labels = []
#     for label in labels:
#         bin_labels.append(multi_labels_to_two(label))
#     labels = bin_labels
#
#     batch_size = 1
#     test_generator = generator_of_data(dataset, labels, batch_size, maxlen, vector_dim)
#     all_test_samples = len(dataset)
#     steps_epoch = int(math.ceil(all_test_samples / batch_size))
#
#     t1 = time.time()
#     result = model.evaluate_generator(test_generator, steps=steps_epoch)
#     t2 = time.time()
#     test_time = t2 - t1
#     report_metrics()
#
#     dict_testcase2func = {}
#     for i in testcases:
#         if not i in dict_testcase2func:
#             dict_testcase2func[i] = {}
#     TP_indexs = result[1]
#     for i in TP_indexs:
#         if funcs[i] == []:
#             continue
#         for func in funcs[i]:
#             if func in dict_testcase2func[testcases[i]].keys():
#                 dict_testcase2func[testcases[i]][func].append("TP")
#             else:
#                 dict_testcase2func[testcases[i]][func] = ["TP"]
#     FP_indexs = result[1]
#     for i in FP_indexs:
#         if funcs[i] == []:
#             continue
#         for func in funcs[i]:
#             if func in dict_testcase2func[testcases[i]].keys():
#                 dict_testcase2func[testcases[i]][func].append("FP")
#             else:
#                 dict_testcase2func[testcases[i]][func] = ["FP"]
#     f = open(resultpath + "_dict_testcase2func.pkl", 'wb')
#     pickle.dump(dict_testcase2func, f)
#     f.close()

if __name__ == "__main__":
    num_val_samples = 2
    batchSize = 32
    vectorDim = 30
    maxLen = 500
    layers = 2
    dropout = 0.2

    checkpoint_dir = "./ckpt"
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    #traindataSetPath = "./dl_input_shuffle/cdg_ddg/train/"
    traindataSetPath = "../source2slice/test_data/4/dl_input_shuffle/train/"
    #testdataSetPath = "./dl_input_shuffle/cdg_ddg/test/"
    testdataSetPath = "../source2slice/test_data/4/dl_input_shuffle/test/"
    realtestdataSetPath = "data/"
    weightPath = './model/BGRU'
    resultPath = "./result/BGRU/BGRU.txt"
    main(traindataSetPath, testdataSetPath, realtestdataSetPath, weightPath, resultPath, batchSize, maxLen, vectorDim, layers, dropout, num_val_samples)
    #testrealdata(realtestdataSetPath, weightPath, batchSize, maxLen, vectorDim, layers, dropout)
