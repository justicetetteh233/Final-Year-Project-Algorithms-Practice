"""
Created on Mon Mar 6 16:50:44 2023

@author: Oyelade
"""

# -*- coding: utf-8 -*-
"""
Created on Teu Jan 18 16:50:44 2021

@author: Oyelade
"""
#from __future__ import print_function
import tensorflow as tf
from tensorflow.python.keras.regularizers import l1, l2, L1L2
#from tensorflow import keras
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, Dropout, Flatten, Concatenate, Reshape, Activation
#from tensorflow.python.keras.layers import Input, Dense, Conv2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, Dropout, Flatten, Concatenate, Reshape, Activation
from tensorflow.keras.optimizers import SGD, Adam, Adadelta, RMSprop, Adagrad, Adamax
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.python.keras.backend import set_session
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from time import time
import numpy as np
import os


class RootCNN(object):
    REGULARIZER_RATES=0.0002
       
    def __init__(self, root_params=None):
        self.num_classes = root_params["num_classes"]
        self.class_names = root_params["class_names"]
        self.img_width=root_params["img_size"]["width"] 
        self.img_height=root_params["img_size"]["height"]
        self.activations=root_params["activations"]
        self.epoch=root_params["cnn_epoch"]
        self.batch_size=root_params["batch_size"]
        self.log=root_params["log_mode"]
        self.models_path=root_params["models_path"]
        self.model_filename=root_params["model_filename"]
        self.checkpoint_path=root_params["checkpoint_path"]
        self.checkpoint_epoch=root_params["checkpoint_epoch"]
        self.fromCheckpoint=root_params["fromCheckpoint"]        
        self.path_save_result=root_params["save_results_dir"]
        self.show=root_params["show"]        
        self.num_channels=root_params["num_channels"]
        self.input_source=root_params["input_source"]
        self.K=root_params["K"]
        self.session=None
    
    def cnn_input(self, model_type=0, is_zeropad=None):
        channel='channels_first'
        #K.set_image_data_format('channels_last')
        if model_type==0:
            input_shape=(None, self.num_channels, self.img_width, self.img_height) if self.K.image_data_format() ==channel  else (None, self.img_width, self.img_height, self.num_channels)
            inputs =ZeroPadding2D((1,1), 
                                  input_shape=
                                  (self.num_channels, self.img_width, self.img_height)
                                  if self.K.image_data_format() == channel 
                                  else (self.img_width, self.img_height, self.num_channels)) if is_zeropad else input_shape
        else:
            input_shape=(self.num_channels, self.img_width, self.img_height) if self.K.image_data_format() == channel else (self.img_width, self.img_height, self.num_channels)
            inputs = ZeroPadding2D(padding=(1, 1))(input_shape) if is_zeropad else input_shape
        return inputs, input_shape
        
    def _2Dconvolution__(self, convo_number=None, cf=None, ck=None, activation=None, regularizer=None, is_first=False, input_pad=None,is_zeropad=False):
        regularizer=self.get_regularizer()         
        if is_first and not(is_zeropad):
            convo2d=Conv2D(int(ck), (int(cf),int(cf)), strides=(1,1), padding='same', 
                           activation=self.get_activation_function(func=activation), 
                           input_shape=input_pad,  
                       name='conv_'+str(convo_number), kernel_regularizer=regularizer)         
        return convo2d
    
    def _2Dpool__(self, pnumber=None, pfilter=None, ptype=None):
        if ptype=="Max": #data_format="channels_first",
            pool = MaxPooling2D(pool_size=(int(pfilter), int(pfilter)), strides=(2,2), padding='same',  name='pool_'+str(pnumber))        
        else:
            pool = AveragePooling2D(pool_size=(int(pfilter), int(pfilter)), strides=(2,2), padding='same', name='pool_'+str(pnumber))
        return pool
    
    def flaten(self):
        return Flatten()
    
    def architecture_summary(self, model=None, input_shape=None):
        model.build(input_shape)
        return model.summary()
        
    def fully_dense(self, activation=None, dropout=None, model=None):
        regularizer=self.get_regularizer('L1')         
        model.add(Dropout(rate=float(dropout)))     
        model.add(Dense(self.num_classes, name='loss_classifier_0', kernel_regularizer=regularizer))
        model.add(Activation(activation, name='class_prob'))
        return model                    
    
    def get_regularizer(self, regularizer=None):
        if regularizer=="L1" :
            regularizer=l1(self.REGULARIZER_RATES)
        elif regularizer=="L2":
            regularizer=l2(self.REGULARIZER_RATES)
        else: 
            regularizer=L1L2(self.REGULARIZER_RATES)
        return regularizer
    
    def get_activation_function(self, func=None):        
        if func=='leakyrelu':
            return tf.keras.layers.LeakyReLU(alpha=0.1)
        elif func=='parametricrelu':
            return tf.keras.layers.PReLU() #alpha=0.1
        else:
            return func
        
    def load_trained_model(self, x_train, y_train, x_eval, y_eval): 
        if self.fromCheckpoint:
            checkpoint_dir = os.path.dirname(self.checkpoint_path)
            latest = tf.train.latest_checkpoint(checkpoint_dir)
            self.model.load_weights('./outputs/checkpoints/mammo/trainedmodelmammo.h5')
            #self.get_fc_features__(model=self.model, x_train=x_train, y_train=y_train)
            
        loss_train, accuracy_train, loss_eval, accuracy_eval,time_total_train=self.trainmodel(model=self.model, x_train=x_train, y_train=y_train, x_eval=x_eval, y_eval=y_eval)
        return loss_train, accuracy_train, loss_eval, accuracy_eval, time_total_train
    
    def config(self):
        if self.K.backend() == 'tensorflow':
            config = tf.compat.v1.ConfigProto() 
            config.gpu_options.per_process_gpu_memory_fraction = 0.333
            session = tf.compat.v1.Session(config=config) 
            #session = tf.compat.v1.Session(graph=tf.Graph())
            self.session=session
            set_session(session)
            # Using the Winograd non-fused algorithms provides a small performance boost.
            os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'
            os.environ['KMP_DUPLICATE_LIB_OK']='True'
            #To not use GPU, a good solution is to not allow the environment to see any GPUs by setting the environmental variable CUDA_VISIBLE_DEVICES.
            os.environ["CUDA_VISIBLE_DEVICES"]="1"
            os.environ["OMP_NUM_THREADS"] = "NUM_PARALLEL_EXEC_UNITS"
            os.environ["KMP_BLOCKTIME"] = "30"
            os.environ["KMP_SETTINGS"] = "1"
            os.environ["KMP_AFFINITY"]= "granularity=fine,verbose,compact,1,0"
            
    def compile_model(self, model=None, optimizer=None, learning_rate=None, lossfunc=None):
        momentum=0.9 #0.0,  0.5,  0.9,  0.99
        if optimizer == 'SGD':
            optim = SGD(lr=learning_rate, decay=1e-6, momentum=momentum, nesterov=True)
        elif optimizer == 'RMSprop':
            optim = tf.keras.optimizers.RMSprop(learning_rate=learning_rate,rho=0.9,momentum=momentum,epsilon=1e-07, centered=False, name="RMSprop")
        elif optimizer == 'Adagrad':
            optim = tf.keras.optimizers.Adagrad(learning_rate=learning_rate, initial_accumulator_value=0.1,epsilon=1e-07,name="Adagrad")
        elif optimizer == 'Adam': #adam
            optim=Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-8) 
        elif optimizer == 'Adadelta': #Adadelta
            optim=Adadelta(lr=learning_rate, rho=0.95, epsilon=1e-07, name="Adadelta")
        elif optimizer == 'Adamax': #Adamax
            optim=Adamax(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-07) #, name="Adamax"
        elif optimizer == 'Momentum': #Momentum
            optim=SGD(lr=learning_rate, decay=1e-6, momentum=momentum, nesterov=False)
        else: #optimizer == 'Nestrov': #Nestrov
            optim= SGD(lr=learning_rate, decay=1e-6, momentum=0.0, nesterov=True)
        
        model.compile(optimizer=optim, loss=lossfunc, metrics=['accuracy']) #'categorical_crossentropy'        
        return model
    
    def trainmodel(self, model=None, x_train=None, y_train=None, x_eval=None, y_eval=None):
        time_total_train = time()
        checkpoint_path=self.checkpoint_path+'cp-{epoch:04d}.ckpt'#.format(self.epoch)
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, verbose=0, save_weights_only=True,
        save_freq='epoch', period=20)
        #model = ResumableModel(model, save_every_epochs=1, custom_objects=None, to_path=self.checkpoint_path)
        print(x_train.shape, y_train.shape)
        ml = model.fit(x_train, y_train, 
                       epochs=self.epoch, 
                       initial_epoch=self.checkpoint_epoch,
                       validation_data=(x_eval, y_eval),
                       #steps_per_epoch=int(x_train.shape[0]//self.batch_size),
                       batch_size=self.batch_size,  
                       #callbacks=[cp_callback],
                       verbose=self.log)
        print(ml)
        print("test loss, test acc:", ml.history)
        print("Evaluate on test data")
        results = model.evaluate(x_eval, y_eval, batch_size=self.batch_size)
        print("evaluation result", results)
        time_total_train = round(time() - time_total_train, 4)
        
        loss_train = None #ml.history["loss"] #ml["loss"] # 
        accuracy_train =  None # ml.history["accuracy"] #  ml["acc"] # 
        loss_eval =  None #ml.history["val_loss"] #ml["val_loss"] #
        accuracy_eval =  None #ml.history["val_accuracy"]   #ml["val_acc"] #
        #self.save_model(model)
        return loss_train, accuracy_train, loss_eval, accuracy_eval, time_total_train,
    
    def save_model(self, model=None):
        pathf=self.path_save_result+self.model_filename
        pathf_h5=self.path_save_result+self.model_filename+'.h5'
        model.save(pathf)
        model.save(pathf_h5)
        #tf.keras.models.save_model()
        
    def predict(self, x=None, y=None):
        if x==None and y==None:
            x_test, y_test=self.input_source.get_test_data() 
        else:
            x_test, y_test=x, y
        time_predict, avg_pred, y_test, y_pred, x_test=self.predictmodel(model=self.model, x_test=x_test, y_test=y_test)
        return time_predict, avg_pred, y_test, y_pred, x_test
    
    def predictmodel(self, model=None, x_test=None, y_test=None):
        time_predict = time()
        y_pred = model.predict(x_test)
        avg_pred=tf.keras.losses.categorical_crossentropy(y_pred, y_test)
        time_predict = round(time() - time_predict, 8)
        return time_predict, avg_pred, y_test, y_pred, x_test
    
    def extract_vector_features(self, model=None, x_train=None, y_train=None, x_eval=None, y_eval=None, layer_name=None):
        checkpoint_dir = os.path.dirname(self.checkpoint_path)
        latest = tf.train.latest_checkpoint(checkpoint_dir)
        model.load_weights(latest) #self.checkpoint_path+'cp-0025.ckpt')#
        
        #Create a new model 
        model =tf.keras.models.Model(inputs=model.inputs, outputs=model.get_layer(name=layer_name).output,) 
        layer_shape=model.get_layer(name=layer_name).output_shape
        #layer_shape ---> 8, 8, 512 for histo;  5, 5, 1024 for mammo
        train_features = np.zeros(shape=(x_train.shape[0], layer_shape[1], layer_shape[2], layer_shape[3])) #because this is the shape of the last layer: 'pool_23''
        train_features = model.predict(x_train)
        train_features = np.reshape(train_features, (x_train.shape[0], layer_shape[1] * layer_shape[2] * layer_shape[3]))
        test_features = np.zeros(shape=(x_eval.shape[0], layer_shape[1], layer_shape[2], layer_shape[3]))
        test_features = model.predict(x_eval)
        test_features = np.reshape(test_features, (x_eval.shape[0], layer_shape[1] * layer_shape[2] * layer_shape[3]))
        print(train_features.shape)
        print(train_features.shape[0])
        print(self.checkpoint_path)
        #print(testy)
        
        #save the extracted features
        np.save(self.checkpoint_path+"train_features.npy", train_features)
        np.save(self.checkpoint_path+"train_labels.npy", y_train)
        np.save(self.checkpoint_path+"validation_features.npy", test_features)
        np.save(self.checkpoint_path+"validation_labels.npy", y_eval)        
        return None
    
    def optimize_vector_features(self,x_train=None, y_train=None, x_eval=None, y_eval=None):
        #print(self.checkpoint_path)
        train_features = np.load(self.checkpoint_path+"train_features.npy")
        train_labels = np.load(self.checkpoint_path+"train_labels.npy")
        validation_labels = np.load(self.checkpoint_path+"validation_labels.npy")
        validation_features = np.load(self.checkpoint_path+"validation_features.npy")
        whole_feat=np.concatenate((train_features,validation_features))
        whole_label=np.concatenate((train_labels,validation_labels))        
        lb, ub=whole_feat.shape[1]//8, whole_feat.shape[1]
        return lb, ub, train_features, y_train, validation_features, y_eval# train_labels, validation_labels
    
    def get_extracted_features(self, model=None, x_train=None, y_train=None):
        '''
        checkpoint_dir = os.path.dirname(self.checkpoint_path)
        latest = tf.train.latest_checkpoint(checkpoint_dir)
        model.load_weights(self.checkpoint_path+'cp-0025.ckpt')#latest)            
        '''
        #Create a new model 
        old_model=model
        model =tf.keras.models.Model(inputs=model.inputs, #name="pool_5"
                                     outputs=model.get_layer(index=2).output,)             
        train_features = np.zeros(shape=(x_train.shape[0], 5, 5, 1024)) #because this is the shape of the last layer: 'pool_23''
        train_features = model.predict(x_train)            
        print(x_train.shape[0])
        print(len(y_train))
        print(train_features.shape)
        print(train_features.shape[0])            
        train_features = np.reshape(train_features, (x_train.shape[0], 5 * 5 * 1024))            
        np.save(self.checkpoint_path+"train_features.npy", train_features)
        np.save(self.checkpoint_path+"train_labels.npy", y_train)
        
        pca = PCA(n_components=100, whiten=True)
        X_pca = pca.fit_transform(train_features)
        #print(X_pca)
        print("Original number of features:", train_features.shape)
        print("Reduced number of features:", X_pca.shape)
        return old_model
         
    '''
    def _get_average_error__(self, individual=None, X_data=None, y_data=None):
        t1 = time()
        weights = [rand(*w.shape) for w in self.model_rnn.get_weights()]
        ws = []
        cur_point = 0
        for wei in weights:
            ws.append(reshape(individual[cur_point:cur_point + len(wei.reshape(-1))], wei.shape))
            cur_point += len(wei.reshape(-1))

        self.model_rnn.set_weights(ws)
        y_pred = self.model_rnn.predict(X_data)
        # print("GAE time: {}".format(time() - t1))

        # return [mean_squared_error(y_pred, y_data), mean_absolute_error(y_pred, y_data)]
        return tf.keras.losses.categorical_crossentropy(y_pred, y_data)

    def _objective_function__(self, solution=None):
        weights = [rand(*w.shape) for w in self.model_rnn.get_weights()]
        ws = []
        cur_point = 0
        for wei in weights:
            ws.append(reshape(solution[cur_point:cur_point + len(wei.reshape(-1))], wei.shape))
            cur_point += len(wei.reshape(-1))

        self.model_rnn.set_weights(ws)
        y_pred = self.model_rnn.predict(self.X_train)
        return tf.keras.losses.categorical_crossentropy(y_pred, self.y_train)
    '''
    
    def predict_single_img(image):
        image = image.resize((200, 200))
        '''
        image = img_to_array(image)
        image = np.expand_dims(image, 0)
        image = imagenet_utils.preprocess_input(image)
        image = image / 255
        pred = np.argmax(model.predict(image))
        return labels[pred]
        '''