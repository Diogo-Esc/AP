#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Aprendizagem Profunda, TP1
Diogo Escaldeira 50054
João Azevedo 53389
"""
import os
from requests import options 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np
from datetime import datetime
from tp1_utils import load_data, overlay_masks, compare_masks
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, MaxPooling2D, Flatten, Dense, Dropout, Activation
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.models import Model
from tensorflow.python.keras.layers import GlobalAveragePooling2D
import matplotlib.pyplot as plt

#Treinar a Rede
def train_network(log_dir, optimizer, model, metric, loss_func, epochs, batch_size, t_X, t_Y, v_X, v_Y):
    model.compile(loss = loss_func, optimizer = optimizer, metrics = [metric])
    model.summary()
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir = log_dir, histogram_freq = 1)
    history = model.fit(t_X, t_Y, validation_data = (v_X, v_Y), batch_size = batch_size, epochs = epochs, callbacks = [tensorboard_callback])

    return history, model

#Cria uma rede multiClass, Density - Densidade da camada dense, Activation - Funcao de activacao da ultima layer
def createmultiClassmodel(density, activation):
    inputs = Input(shape=(64, 64, 3), name = 'inputs')
    layer = Conv2D(64, (1, 1), padding = "same", activation = "relu")(inputs)
    layer = BatchNormalization(axis = -1)(layer)
    layer = MaxPooling2D(pool_size = (2, 2))(layer)
    layer = Conv2D(64, (3, 3), padding = "same", activation = "relu")(layer)
    layer = BatchNormalization(axis = -1)(layer)
    layer = MaxPooling2D(pool_size = (2, 2))(layer)
    layer = Conv2D(64, (3, 3), padding = "same", activation = "relu")(layer)
    layer = MaxPooling2D(pool_size = (2, 2))(layer)
    features = Flatten(name = 'features')(layer)
    layer = Dense(density, activation = "relu")(features)
    layer = BatchNormalization()(layer)
    layer = Dropout(0.25)(layer)
    layer = Dense(10, activation = activation)(layer)

    return Model(inputs = inputs, outputs = layer)

#Cria um modelo, Activation - Funcao de activacao da ultima layer
def createmultimodel(activation):
    inputs = Input(shape=(64, 64, 3), name='inputs')
    layer = Conv2D(64, (1, 1), padding="same", activation="relu")(inputs)
    layer = BatchNormalization(axis=-1)(layer)
    layer = MaxPooling2D(pool_size=(2, 2))(layer)
    layer = Conv2D(64, (3, 3), padding="same", activation="relu")(layer)
    layer = BatchNormalization(axis=-1)(layer)
    layer = MaxPooling2D(pool_size=(2, 2))(layer)
    layer = Conv2D(64, (3, 3), padding="same", activation="relu")(layer)
    layer = MaxPooling2D(pool_size=(2, 2))(layer)

    features = Flatten(name='features')(layer)
    layer = Dense(32, activation="relu")(features)
    layer = BatchNormalization()(layer)
    layer = Dropout(0.25)(layer)
    layer = Dense(10, activation=activation)(layer)
    return Model(inputs=inputs, outputs=layer)

#MultiClass - Teste final a rede multi-class
def test_multiclass():
    log_dir = "logs/fit/multiclass/" + datetime.now().strftime("%Y%m%d-%H%M%S")

    t_X, v_X, t_Y, v_Y = train_test_split(X_train, Y_train_classes, test_size = 0.125, stratify = Y_train_classes, shuffle = True)

    opt = Adam(amsgrad=True)

    multiClassModel = createmultimodel('softmax')

    history, model = train_network(log_dir, opt, multiClassModel, "accuracy", "categorical_crossentropy", 25, 20, t_X, t_Y, v_X, v_Y)

    test_eval = model.evaluate(X_test, Y_test_classes)
    
    print("Test evaluation: ", test_eval)

#MultiLabel - Teste final a rede multi-label
def test_multilabel():
    log_dir = "logs/fit/multilabel/" + datetime.now().strftime("%Y%m%d-%H%M%S")


    t_X, v_X, t_Y, v_Y = train_test_split(X_train, Y_train_labels, test_size = 0.125, stratify = Y_train_labels, shuffle = True)

    opt = Adam(amsgrad=True)

    multiLabelModel = createmultimodel('sigmoid')

    history, model = train_network(log_dir, opt, multiLabelModel, "binary_accuracy", "binary_crossentropy", 25, 20, t_X, t_Y, v_X, v_Y)

    test_eval = model.evaluate(X_test, Y_test_labels)
    
    print("Test evaluation: ", test_eval)

#Cria um modelo EfficientNetB0, Activation - Funcao de activacao da ultima layer
def create_model_efficientnetb0(activation):
    core = tf.keras.applications.EfficientNetB0(include_top=False, input_shape=(64, 64, 3))

    for layer in core.layers:
        layer.trainable = False

    inputs = core.output
    layer = GlobalAveragePooling2D()(inputs)
    features = Flatten(name='features')(layer)
    layer = Dense(32, activation="relu")(features)
    layer = BatchNormalization()(layer)
    layer = Dropout(0.25)(layer)
    layer = Dense(10, activation=activation)(layer)
    return Model(inputs=core.input, outputs=layer)

#EfficientNetB0 MultiClass - Teste final a rede efficientnetb0 multi-class
def test_efficientnetb0_multiclass():
    log_dir = "logs/fit/efficientnetmulticlass/" + datetime.now().strftime("%Y%m%d-%H%M%S")


    X_enb0 = tf.keras.applications.efficientnet.preprocess_input(data['train_X']*255)

    t_X, v_X, t_Y, v_Y = train_test_split(X_enb0, Y_train_classes, test_size = 0.125, stratify = Y_train_classes, shuffle = True)

    opt = Adam(amsgrad=True)

    multiClassModel = create_model_efficientnetb0('softmax')

    history, model = train_network(log_dir, opt, multiClassModel, "accuracy", "categorical_crossentropy", 25, 20, t_X, t_Y, v_X, v_Y)

    test_eval = model.evaluate(X_test, Y_test_classes)
    
    print("Test evaluation: ", test_eval)

#EfficientNetB0 MultiLabel - Teste final a rede efficientnetb0 multi-label
def test_efficientnetb0_multiclass():
    log_dir = "logs/fit/efficientnetmultilabel/" + datetime.now().strftime("%Y%m%d-%H%M%S")


    X_enb0 = tf.keras.applications.efficientnet.preprocess_input(data['train_X']*255)

    t_X, v_X, t_Y, v_Y = train_test_split(X_enb0, Y_train_labels, test_size = 0.125, stratify = Y_train_labels, shuffle = True)

    opt = Adam(amsgrad=True)

    multiLabelModel = create_model_efficientnetb0('sigmoid')

    history, model = train_network(log_dir, opt, multiLabelModel, "accuracy", "categorical_crossentropy", 25, 20, t_X, t_Y, v_X, v_Y)

    test_eval = model.evaluate(X_test, Y_test_labels)
    
    print("Test evaluation: ", test_eval)

#Cria uma rede segmentation, Activation - Funcao de activacao da ultima layer, Bool - Indica se vai ter uma camada densa, 
# Conv - Indica o tamanho das camadas de convolucao
def create_model_segm(activation, bool, convs):      
    inputs = Input(shape = (64, 64, 3), name = 'inputs')
    layer = Conv2D(convs, (1, 1), padding = "same")(inputs)
    layer = Activation("relu")(layer)
    layer = BatchNormalization(axis = -1)(layer)
    layer = MaxPooling2D(pool_size = (2, 2))(layer)
    layer = Conv2D(convs, (3, 3), padding = "same")(layer)
    layer = Activation("relu")(layer)
    layer = BatchNormalization(axis = -1)(layer)
    layer = MaxPooling2D(pool_size = (2, 2))(layer)
    layer = Conv2D(convs, (3, 3), padding = "same")(layer)
    layer = Activation("relu")(layer)
    layer = Conv2D(convs, (3, 3), padding = "same")(layer)
    layer = Activation("relu")(layer)
    layer = MaxPooling2D(pool_size = (2, 2))(layer)

    if(bool == True):
        features = Flatten(name = 'features')(layer)
        layer = Dense(32, activation = "relu")(features)
        layer = BatchNormalization()(layer)
        layer = Dropout(0.25)(layer)
    else:
         layer = Flatten(name = 'features')(layer)

    layer = Dense(4096)(layer)
    layer = Activation(activation)(layer)

    return Model(inputs = inputs, outputs = layer)

#Segmentation - Teste final a rede segmentation
def test_seg():
    log_dir = "logs/fit/segmentation/finalNET/" + datetime.now().strftime("%Y%m%d-%H%M%S")

    Y_train_masks = data['train_masks']
    Y_test_masks = data['test_masks']

    opt = Adam(amsgrad = True)

    Y_train_masks = np.reshape(Y_train_masks, (np.shape(Y_train_masks)[0], -1))  
    Y_test_masks = np.reshape(Y_test_masks, (np.shape(Y_test_masks)[0], -1)) 

    t_X, v_X, t_Y, v_Y = train_test_split(X_train, Y_train_masks, test_size = 0.125, shuffle=True)
    
    model = create_model_segm("sigmoid", False, 32)

    history, model = train_network(log_dir, opt, model, "binary_accuracy", "binary_crossentropy", 10, 10, t_X, t_Y, v_X, v_Y)

    test_eval = model.evaluate(X_test, Y_test_masks)
    print("Test evaluation: ", test_eval)

    pre = np.reshape(model.predict(X_test),  (500, 64, 64, 1))
    compare_masks("compare_masks.png", data['test_masks'], pre)
    overlay_masks("overlay_masks.png", X_test, pre)

#Segmentation - Validar o tamanho das camadas de convolucao 32vs64
def test_seg_convs():
    log_dir1 = "logs/fit/segmentation/convsize/32/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir2 = "logs/fit/segmentation/convsize/64/" + datetime.now().strftime("%Y%m%d-%H%M%S")

    Y_train_masks = data['train_masks']  
    Y_test_masks = data['test_masks'] 

    Y_train_masks = np.reshape(Y_train_masks, (np.shape(Y_train_masks)[0], -1))  
    Y_test_masks = np.reshape(Y_test_masks, (np.shape(Y_test_masks)[0], -1))  

    opt = Adam(amsgrad = True)

    t_X, v_X, t_Y, v_Y = train_test_split(X_train, Y_train_masks, test_size = 0.125, shuffle=True)

    model32 = create_model_segm("sigmoid", False, 32)
    model64 = create_model_segm("sigmoid", False, 64)

    history1, model1 = train_network(log_dir1, opt, model32, "binary_accuracy", "binary_crossentropy", 10, 10, t_X, t_Y, v_X, v_Y)
        
    history2, model2 = train_network(log_dir2, opt, model64, "binary_accuracy", "binary_crossentropy", 10, 10, t_X, t_Y, v_X, v_Y)

    y1 = history1.history['val_binary_accuracy'] 
    y2 = history2.history['val_binary_accuracy'] 

    test_eval1 = model1.evaluate(X_test, Y_test_masks)
    print("Test evaluation 32: ", test_eval1)

    test_eval2 = model2.evaluate(X_test, Y_test_masks)
    print("Test evaluation 64: ", test_eval2)

    plt.plot(list(range(1, 11)), y1, color = 'b', label='32')
    plt.plot(list(range(1, 11)), y2, color = 'g', label='64')
    plt.grid(linestyle = '--')
    plt.legend(loc = 'lower right')
    plt.suptitle('ConvSize Validation')
    plt.xlabel('Epochs')
    plt.ylabel('Validation Accuracy')
    plt.savefig('SEG_convSize_validation', dpi = 200)
    plt.show()
    plt.close()

#Segmentation - Validar o otimizador AdamAmsgradvsSGDvsRMSprop
def test_seg_opt():
    log_dir1 = "logs/fit/segmentation/optimizers/adam/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir2 = "logs/fit/segmentation/optimizers/sgd/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir3 = "logs/fit/segmentation/optimizers/rmsprop/" + datetime.now().strftime("%Y%m%d-%H%M%S")

    opt1 = Adam(amsgrad = True)
    opt2 = SGD()
    opt3 = RMSprop()

    Y_train_masks = data['train_masks']  
    Y_test_masks = data['test_masks'] 

    Y_train_masks = np.reshape(Y_train_masks, (np.shape(Y_train_masks)[0], -1))  
    Y_test_masks = np.reshape(Y_test_masks, (np.shape(Y_test_masks)[0], -1))  

    t_X, v_X, t_Y, v_Y = train_test_split(X_train, Y_train_masks, test_size = 0.125, shuffle=True)

    model = create_model_segm("sigmoid", False, 32)

    history1, model1 = train_network(log_dir1, opt1, model, "binary_accuracy", "binary_crossentropy", 10, 10, t_X, t_Y, v_X, v_Y)

    history2, model2 = train_network(log_dir2, opt2, model, "binary_accuracy", "binary_crossentropy", 10, 10, t_X, t_Y, v_X, v_Y)
    
    history3, model3 = train_network(log_dir3, opt3, model, "binary_accuracy", "binary_crossentropy", 10, 10, t_X, t_Y, v_X, v_Y)

    y1 = history1.history['val_binary_accuracy'] 
    y2 = history2.history['val_binary_accuracy'] 
    y3 = history3.history['val_binary_accuracy'] 
    
    test_eval1 = model1.evaluate(X_test, Y_test_masks)
    print("Test evaluation Adam: ", test_eval1)

    test_eval2 = model2.evaluate(X_test, Y_test_masks)
    print("Test evaluation SGD: ", test_eval2)

    test_eval3 = model3.evaluate(X_test, Y_test_masks)
    print("Test evaluation RMSprop: ", test_eval3)

    plt.plot(list(range(1, 11)), y1, color = 'b', label='Adam')
    plt.plot(list(range(1, 11)), y2, color = 'g', label='SVC')
    plt.plot(list(range(1, 11)), y3, color = 'r', label='RMSprop')
    plt.grid(linestyle = '--')
    plt.legend(loc = 'lower right')
    plt.suptitle('Seg Optimizers Validation')
    plt.xlabel('Epochs')
    plt.ylabel('Validation Accuracy')
    plt.savefig('SEG_optvalidation', dpi = 200)
    plt.show()
    plt.close()

#Segmentation - Validar o numero de epochs necessarios para atingir um pico de accuracy
def test_seg_epochs():
    log_dir = "logs/fit/segmentation/epochs/" + datetime.now().strftime("%Y%m%d-%H%M%S")

    Y_train_masks = data['train_masks']
    Y_test_masks = data['test_masks']

    opt = Adam(amsgrad = True)

    Y_train_masks = np.reshape(Y_train_masks, (np.shape(Y_train_masks)[0], -1))  
    Y_test_masks = np.reshape(Y_test_masks, (np.shape(Y_test_masks)[0], -1)) 

    t_X, v_X, t_Y, v_Y = train_test_split(X_train, Y_train_masks, test_size = 0.125, shuffle=True)
    
    modelA = create_model_segm("sigmoid", False, 32)

    history, model = train_network(log_dir, opt, modelA, "binary_accuracy", "binary_crossentropy", 30, 10, t_X, t_Y, v_X, v_Y)

    y = history.history['val_binary_accuracy'] 

    test_eval = model.evaluate(X_test, Y_test_masks)
    print("Test evaluation: ", test_eval)

    plt.plot(list(range(1, 31)), y, color = 'b')
    plt.grid(linestyle = '--')
    plt.suptitle('Epochs Validation')
    plt.xlabel('Epochs')
    plt.ylabel('Validation Accuracy')
    plt.savefig('SEG_epochsvalidation', dpi = 200)
    plt.show()
    plt.close()

#Segmentation - Validar se e benefico uma rede densa no final, seguido de batchnormalization, e dropout
def test_seg_withdense_nodense():
    log_dirN = "logs/fit/segmentation/last_dense_layer/With_DenseLayer/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dirS = "logs/fit/segmentation/last_dense_layer/No_DenseLayer/" + datetime.now().strftime("%Y%m%d-%H%M%S")

    Y_train_masks = data['train_masks']
    Y_test_masks = data['test_masks']  

    opt = Adam(amsgrad = True)

    Y_train_masks = np.reshape(Y_train_masks, (np.shape(Y_train_masks)[0], -1)) 
    Y_test_masks = np.reshape(Y_test_masks, (np.shape(Y_test_masks)[0], -1)) 

    t_X, v_X, t_Y, v_Y = train_test_split(X_train, Y_train_masks, test_size = 0.125, shuffle=True)
    
    modelN = create_model_segm("sigmoid", False, 32)
    modelN._name = 'NoDenseModel'

    modelY = create_model_segm('sigmoid', True, 32)
    modelY._name = 'DenseModel'

    historyN, modelN = train_network(log_dirN, opt, modelN, "binary_accuracy", "binary_crossentropy", 10, 10, t_X, t_Y, v_X, v_Y)

    historyY, modelS = train_network(log_dirS, opt, modelY, "binary_accuracy", "binary_crossentropy", 10, 10, t_X, t_Y, v_X, v_Y)

    yN = historyN.history['val_binary_accuracy'] 
    yY = historyY.history['val_binary_accuracy']
    
    test_evalN = modelN.evaluate(X_test, Y_test_masks)
    print("Test evaluation No Dense: ", test_evalN)

    test_evalS = modelS.evaluate(X_test, Y_test_masks)
    print("Test evaluation Dense: ", test_evalS)    

    plt.plot(list(range(1, 11)), yN, color = 'b', label='No Dense')
    plt.plot(list(range(1, 11)), yY, color = 'g', label='Dense')
    plt.grid(linestyle = '--')
    plt.legend(loc = 'lower right')
    plt.suptitle('Dense_NoDense Validation')
    plt.xlabel('Epochs')
    plt.ylabel('Validation Accuracy')
    plt.savefig('dense_nodense', dpi = 200)
    plt.show()
    plt.close()

#Segmentation - Validar a melhor funcao de ativacao final
def test_seg_finalactivfunc():
    log_dirsoft = "logs/fit/segmentation/lastactivfunc/softmax/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dirsigm = "logs/fit/segmentation/lastactivfunc/sigmoid/" + datetime.now().strftime("%Y%m%d-%H%M%S")

    Y_train_masks = data['train_masks']  
    Y_test_masks = data['test_masks']  

    opt = Adam(amsgrad = True)

    Y_train_masks = np.reshape(Y_train_masks, (np.shape(Y_train_masks)[0], -1)) 
    Y_test_masks = np.reshape(Y_test_masks, (np.shape(Y_test_masks)[0], -1)) 

    t_X, v_X, t_Y, v_Y = train_test_split(X_train, Y_train_masks, test_size = 0.125, shuffle=True)

    modelsoft = create_model_segm("softmax", False, 32)
    modelsoft._name = 'softmax__model'
    modelsigm = create_model_segm("sigmoid", False, 32)
    modelsigm._name = 'sigmoid_model'

    historysoft, modelsoft = train_network(log_dirsoft, opt, modelsoft, "binary_accuracy", "binary_crossentropy", 10, 10, t_X, t_Y, v_X, v_Y)
                    
    historysigm, modelsigm = train_network(log_dirsigm, opt, modelsigm, "binary_accuracy", "binary_crossentropy", 10, 10, t_X, t_Y, v_X, v_Y)

    ysoft = historysoft.history['val_binary_accuracy'] 
    ysigm = historysigm.history['val_binary_accuracy']

    test_evalsoft = modelsoft.evaluate(X_test, Y_test_masks)
    print("Test evaluation Soft: ", test_evalsoft)

    test_evalsigm = modelsigm.evaluate(X_test, Y_test_masks)
    print("Test evaluation Sigm: ", test_evalsigm)

    plt.plot(list(range(1, 11)), ysoft, color = 'b', label='Softmax')
    plt.plot(list(range(1, 11)), ysigm, color = 'g', label='Sigmoid')
    plt.grid(linestyle = '--')
    plt.legend(loc = 'lower right')
    plt.suptitle('SoftmaxvsSigmoid Validation')
    plt.xlabel('Epochs')
    plt.ylabel('Validation Accuracy')
    plt.savefig('softVSsigm', dpi = 200)
    plt.show()
    plt.close()

#Segmentation - Validar o learning rate ideal para o otimizador escolhido para o problema
def test_segmentation_lr():
    log_dir1 = "logs/fit/segmentation/lr/0.001/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir2 = "logs/fit/segmentation/lr/0.002/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir3 = "logs/fit/segmentation/lr/0.003/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir4 = "logs/fit/segmentation/lr/0.004/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir5 = "logs/fit/segmentation/lr/0.005/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir6 = "logs/fit/segmentation/lr/0.01/" + datetime.now().strftime("%Y%m%d-%H%M%S")

    opt1 = Adam(amsgrad = True, learning_rate = 0.001)
    opt2 = Adam(amsgrad = True, learning_rate = 0.002)
    opt3 = Adam(amsgrad = True, learning_rate = 0.003)
    opt4 = Adam(amsgrad = True, learning_rate = 0.004)
    opt5 = Adam(amsgrad = True, learning_rate = 0.005)
    opt6 = Adam(amsgrad = True, learning_rate = 0.01)

    Y_train_masks = data['train_masks']  
    Y_test_masks = data['test_masks']  

    Y_train_masks = np.reshape(Y_train_masks, (np.shape(Y_train_masks)[0], -1)) 
    Y_test_masks = np.reshape(Y_test_masks, (np.shape(Y_test_masks)[0], -1)) 

    t_X, v_X, t_Y, v_Y = train_test_split(X_train, Y_train_masks, test_size = 0.125, shuffle=True)

    model = create_model_segm("sigmoid", False, 32)
    model._name = 'LRvalidation'

    history1, model1 = train_network(log_dir1, opt1, model,  "binary_accuracy", "binary_crossentropy", 10, 10, t_X, t_Y, v_X, v_Y)

    history2, model2 = train_network(log_dir2, opt2, model,  "binary_accuracy", "binary_crossentropy", 10, 10, t_X, t_Y, v_X, v_Y)
    
    history3, model3 = train_network(log_dir3, opt3, model,  "binary_accuracy", "binary_crossentropy", 10, 10, t_X, t_Y, v_X, v_Y)

    history4, model4 = train_network(log_dir4, opt4, model,  "binary_accuracy", "binary_crossentropy", 10, 10, t_X, t_Y, v_X, v_Y)

    history5, model5 = train_network(log_dir5, opt5, model,  "binary_accuracy", "binary_crossentropy", 10, 10, t_X, t_Y, v_X, v_Y)

    history6, model6 = train_network(log_dir6, opt6, model,  "binary_accuracy", "binary_crossentropy", 10, 10, t_X, t_Y, v_X, v_Y)

    test_eval1 = model1.evaluate(X_test, Y_test_masks)
    test_eval2 = model2.evaluate(X_test, Y_test_masks)
    test_eval3 = model3.evaluate(X_test, Y_test_masks)
    test_eval4 = model4.evaluate(X_test, Y_test_masks)
    test_eval5 = model5.evaluate(X_test, Y_test_masks)
    test_eval6 = model6.evaluate(X_test, Y_test_masks)
   
    print("Test evaluation 0.001: ", test_eval1)
    print("Test evaluation 0.002: ", test_eval2)
    print("Test evaluation 0.003: ", test_eval3)
    print("Test evaluation 0.004: ", test_eval4)
    print("Test evaluation 0.005: ", test_eval5) 
    print("Test evaluation 0.01: ", test_eval6)

    plt.plot(list(range(1, 11)), history1.history['val_binary_accuracy'], color = 'b', label='0.001 lr')
    plt.plot(list(range(1, 11)), history2.history['val_binary_accuracy'], color = 'g', label='0.002 lr')
    plt.plot(list(range(1, 11)), history3.history['val_binary_accuracy'], color = 'r', label='0.003 lr')
    plt.plot(list(range(1, 11)), history4.history['val_binary_accuracy'], color = 'y', label='0.004 lr')
    plt.plot(list(range(1, 11)), history5.history['val_binary_accuracy'], color = 'k', label='0.005 lr')
    plt.plot(list(range(1, 11)), history6.history['val_binary_accuracy'], color = 'c', label='0.01 lr')
    plt.grid(linestyle = '--')
    plt.legend(loc = 'lower right')
    plt.suptitle('Learning Rate Validation')
    plt.xlabel('Epochs')
    plt.ylabel('Validation Accuracy')
    plt.savefig('segm_lr', dpi = 200)
    plt.show()
    plt.close()

#Multiclass - Validar a densidade da camada densa 32vs64vs128vs256vs512
def test_multiclass_dense():
    log_dir32 = "logs/fit/multiclass/dense/32/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir64 = "logs/fit/multiclass/dense/64/" + datetime.now().strftime("%Y%m%d-%H%M%S")

    t_X, v_X, t_Y, v_Y = train_test_split(X_train, Y_train_classes, test_size = 0.125, stratify = Y_train_classes, shuffle = True)

    opt = Adam(amsgrad = True)

    model32 = createmultiClassmodel(32, 'softmax')
    model32._name = "Model.Dense.32"

    model64 = createmultiClassmodel(64, 'softmax')
    model64._name = "Model.Dense.64"


    history32, model32 = train_network(log_dir32, opt, model32, "accuracy", "categorical_crossentropy", 25, 20, t_X, t_Y, v_X, v_Y)

    history64, model64 = train_network(log_dir64, opt, model64, "accuracy", "categorical_crossentropy", 25, 20, t_X, t_Y, v_X, v_Y)
    
    test_eval32 = model32.evaluate(X_test, Y_test_classes)
    test_eval64 = model64.evaluate(X_test, Y_test_classes)
    
    print("Test evaluation 32: ", test_eval32)
    print("Test evaluation 64: ", test_eval64)

    plt.plot(list(range(1, 26)), history32.history['val_accuracy'], color = 'b', label='32 Dense')
    plt.plot(list(range(1, 26)), history64.history['val_accuracy'], color = 'g', label='64 Dense')
    plt.grid(linestyle = '--')
    plt.legend(loc = 'lower right')
    plt.suptitle('32vs64 Dense Validation')
    plt.xlabel('Epochs')
    plt.ylabel('Validation Accuracy')
    plt.savefig('32vs64Dense', dpi = 200)
    plt.show()
    plt.close()

#Multiclass - Validar o learning rate ideal para o otimizador escolhido para o problema
def test_multiclass_lr():
    log_dir1 = "logs/fit/multiclass/lr/0.001/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir2 = "logs/fit/multiclass/lr/0.002/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir3 = "logs/fit/multiclass/lr/0.003/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir4 = "logs/fit/multiclass/lr/0.004/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir5 = "logs/fit/multiclass/lr/0.005/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir6 = "logs/fit/multiclass/lr/0.01/" + datetime.now().strftime("%Y%m%d-%H%M%S")

    optAdam_amsgrad1 = Adam(amsgrad = True, learning_rate = 0.001)
    optAdam_amsgrad2 = Adam(amsgrad = True, learning_rate = 0.002)
    optAdam_amsgrad3 = Adam(amsgrad = True, learning_rate = 0.003)
    optAdam_amsgrad4 = Adam(amsgrad = True, learning_rate = 0.004)
    optAdam_amsgrad5 = Adam(amsgrad = True, learning_rate = 0.005)
    optAdam_amsgrad6 = Adam(amsgrad = True, learning_rate = 0.01)

    t_X, v_X, t_Y, v_Y = train_test_split(X_train, Y_train_classes, test_size = 0.125, stratify = Y_train_classes, shuffle = True)

    model = createmultiClassmodel(32, 'softmax')
    model._name = "LRvalidation"

    history1, model1 = train_network(log_dir1, optAdam_amsgrad1, model, "accuracy", "categorical_crossentropy", 25, 20, t_X, t_Y, v_X, v_Y)

    history2, model2 = train_network(log_dir2, optAdam_amsgrad2, model, "accuracy", "categorical_crossentropy", 25, 20, t_X, t_Y, v_X, v_Y)
    
    history3, model3 = train_network(log_dir3, optAdam_amsgrad3, model, "accuracy", "categorical_crossentropy", 25, 20, t_X, t_Y, v_X, v_Y)

    history4, model4 = train_network(log_dir4, optAdam_amsgrad4, model, "accuracy", "categorical_crossentropy", 25, 20, t_X, t_Y, v_X, v_Y)

    history5, model5 = train_network(log_dir5, optAdam_amsgrad5, model, "accuracy", "categorical_crossentropy", 25, 20, t_X, t_Y, v_X, v_Y)

    history6, model6 = train_network(log_dir6, optAdam_amsgrad6, model, "accuracy", "categorical_crossentropy", 25, 20, t_X, t_Y, v_X, v_Y)

    test_eval1 = model1.evaluate(X_test, Y_test_classes)
    test_eval2 = model2.evaluate(X_test, Y_test_classes)
    test_eval3 = model3.evaluate(X_test, Y_test_classes)
    test_eval4 = model4.evaluate(X_test, Y_test_classes)
    test_eval5 = model5.evaluate(X_test, Y_test_classes)
    test_eval6 = model6.evaluate(X_test, Y_test_classes)
   
    print("Test evaluation 0.001: ", test_eval1)
    print("Test evaluation 0.002: ", test_eval2)
    print("Test evaluation 0.003: ", test_eval3)
    print("Test evaluation 0.004: ", test_eval4)
    print("Test evaluation 0.005: ", test_eval5) 
    print("Test evaluation 0.01: ", test_eval6)

    plt.plot(list(range(1, 26)), history1.history['val_accuracy'], color = 'b', label='0.001 lr')
    plt.plot(list(range(1, 26)), history2.history['val_accuracy'], color = 'g', label='0.002 lr')
    plt.plot(list(range(1, 26)), history3.history['val_accuracy'], color = 'r', label='0.003 lr')
    plt.plot(list(range(1, 26)), history4.history['val_accuracy'], color = 'y', label='0.004 lr')
    plt.plot(list(range(1, 26)), history5.history['val_accuracy'], color = 'k', label='0.005 lr')
    plt.plot(list(range(1, 26)), history6.history['val_accuracy'], color = 'c', label='0.01 lr')
    plt.grid(linestyle = '--')
    plt.legend(loc = 'lower right')
    plt.suptitle('Learning Rate Validation')
    plt.xlabel('Epochs')
    plt.ylabel('Validation Accuracy')
    plt.savefig('multic_lr', dpi = 200)
    plt.show()
    plt.close()

data = load_data()

X_train = data['train_X']  
X_test = data['test_X']  
 
Y_train_classes = data['train_classes']
Y_test_classes = data['test_classes']

Y_train_labels = data['train_labels'] 
Y_test_labels = data['test_labels']

test_seg()