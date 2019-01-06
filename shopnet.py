import tensorflow as tf
import tensorflow.keras.backend as K 
# import keras.backend as K
# tf.enable_eager_execution()
# from keras import optimizers
import os, h5py
import fire
import numpy as np
from misc import get_logger, Option
opt = Option('./config.json')

import pickle

def get_word_idx_size() :
    return len(pickle.load(open(opt.word_path, 'rb')))

class ShopNet :
    def __init__(self) :
        self.logger = get_logger('ShopNet')

        self.N_IMG_FEAT = 2048
        self.max_len = opt.max_len
        self.voca_size = get_word_idx_size() + 1 #500424+1 이어야함 #500458+1 #96778+1 #opt.max_embd_words + 1
        self.embd_size = opt.embd_size

        self.C_idx = dict()
        self.C_idx['b'] = {c:c-1 for c in range(1, 57+1)}
        self.C_idx['m'] = {c:c-1 for c in range(1, 552+1)}
        self.C_idx['s'] = {c:c-2 for c in range(2, 3190+1)}
        self.C_idx['d'] = {c:c-2 for c in range(2, 404+1)}

        self.N_Cb = 57
        self.N_Cm = 552
        self.N_Cs = 3190-1
        self.N_Cd = 404-1

    def model_image(self, trainable=True, load=False) :
        if load :
            load_path = opt.model_root+'image/model.h5'
            model = tf.keras.models.load_model(load_path)
        else :
            # with tf.variable_scope('image_classifier') :
            #     pf = 'img'

            #     inputs_img = tf.keras.Input(shape=(self.N_IMG_FEAT,), name=pf+'IN')

            #     count='1'
            #     x = tf.keras.layers.Dense(100, trainable=trainable, name=pf+'DS'+count)(inputs_img)
            #     x = tf.keras.layers.BatchNormalization(trainable=trainable, name=pf+'BN'+count)(x)
            #     x = tf.keras.layers.Activation('elu', name=pf+'A'+count)(x)


            #     with tf.name_scope('classify') :
            #         b = tf.keras.layers.Dense(self.N_Cb, activation='softmax', name=pf+'b')(x)
            #         m = tf.keras.layers.Dense(self.N_Cm, activation='softmax', name=pf+'m')(x)
            #         s = tf.keras.layers.Dense(self.N_Cs, activation='softmax', name=pf+'s')(x)
            #         d = tf.keras.layers.Dense(self.N_Cd, activation='softmax', name=pf+'d')(x)



            with tf.variable_scope('image_classifier') :
                pf = 'img'

                inputs_img = tf.keras.Input(shape=(self.N_IMG_FEAT,), name=pf+'IN')

                count='1'
                x = tf.keras.layers.Dense(1024, trainable=trainable, name=pf+'DS'+count)(inputs_img)
                x = tf.keras.layers.BatchNormalization(trainable=trainable, name=pf+'BN'+count)(x)
                x = tf.keras.layers.Activation('elu', name=pf+'A'+count)(x)
                x_res = tf.keras.layers.Dropout(0.5, name=pf+'DR'+count)(x)

                count='2'
                x = tf.keras.layers.Dense(1024, trainable=trainable, name=pf+'DS'+count)(x_res)
                x = tf.keras.layers.BatchNormalization(trainable=trainable, name=pf+'BN'+count)(x)
                x = tf.keras.layers.Activation('elu', name=pf+'A'+count)(x)
                x = tf.keras.layers.Dropout(0.25, name=pf+'DR'+count)(x)

                count='3'
                x = tf.keras.layers.Dense(1024, trainable=trainable, name=pf+'DS'+count)(x)
                x = tf.keras.layers.concatenate([x, x_res], axis=-1, name=pf+'CON'+count)
                x = tf.keras.layers.BatchNormalization(trainable=trainable, name=pf+'BN'+count)(x)
                x = tf.keras.layers.Activation('relu', name=pf+'A'+count)(x)
                x = tf.keras.layers.Dropout(0.25, name=pf+'DR'+count)(x)


                with tf.name_scope('classify') :
                    b = tf.keras.layers.Dense(self.N_Cb, activation='softmax', name=pf+'b')(x)
                    m = tf.keras.layers.Dense(self.N_Cm, activation='softmax', name=pf+'m')(x)
                    s = tf.keras.layers.Dense(self.N_Cs, activation='softmax', name=pf+'s')(x)
                    d = tf.keras.layers.Dense(self.N_Cd, activation='softmax', name=pf+'d')(x)

            model = tf.keras.Model(inputs=[inputs_img], outputs=[b, m, s, d])

        return model

    def model_text(self, trainable = True, load=False) : 
        if load :
            load_path = opt.model_root+'text/model.h5'
            model = tf.keras.models.load_model(load_path)
            # model = self.model_text(load=False)
            # latest = 'ensemble/text/cp-0001-1.704.ckpt'
            # model.load_weights(latest)
            # print("Restore saved weights on {}.".format(latest))

        else :
            # with tf.name_scope('text_classifier') :
            #     pf = 'GAP'

            #     inputs_text = tf.keras.Input(shape=(opt.max_len,), name=pf+'IN')

            #     embd = tf.keras.layers.Embedding(self.voca_size, self.embd_size, trainable=trainable, name=pf+'EM', embeddings_initializer='glorot_uniform')
            #     x = embd(inputs_text)
            
            #     count = '1'
            #     # x = tf.keras.layers.LSTM(opt.embd_size, return_sequences=False, recurrent_dropout=0.25, trainable=trainable)(x)
            #     x = tf.keras.layers.GlobalAveragePooling1D(name=pf+'AP'+count)(x)
            #     x = tf.keras.layers.BatchNormalization(trainable=trainable, name=pf+'BN'+count)(x)
            #     x = tf.keras.layers.Activation('elu', name=pf+'A'+count)(x)
            #     x = tf.keras.layers.Dropout(0.25, name=pf+'DR'+count)(x)


            #     count = '3'
            #     x = tf.keras.layers.Dense(200, trainable=trainable, name=pf+'DS'+count)(x)
            #     x = tf.keras.layers.BatchNormalization(trainable=trainable, name=pf+'BN'+count)(x)
            #     x = tf.keras.layers.Activation('elu', name=pf+'A'+count)(x)
            #     x = tf.keras.layers.Dropout(0.5, name=pf+'DR'+count)(x) #0.25=>0.5

            #     with tf.name_scope('classify') :
            #         b = tf.keras.layers.Dense(self.N_Cb, activation='softmax', name=pf+'b')(x)
            #         m = tf.keras.layers.Dense(self.N_Cm, activation='softmax', name=pf+'m')(x)
            #         s = tf.keras.layers.Dense(self.N_Cs, activation='softmax', name=pf+'s')(x)
            #         d = tf.keras.layers.Dense(self.N_Cd, activation='softmax', name=pf+'d')(x)

            # model = tf.keras.Model(inputs=[inputs_text], outputs=[b, m, s, d])

            with tf.name_scope('text_classifier') :
                pf = 'GAP'

                inputs_text = tf.keras.Input(shape=(opt.max_len,), name=pf+'IN')

                embd = tf.keras.layers.Embedding(self.voca_size, self.embd_size, trainable=trainable, name=pf+'EM', embeddings_initializer='glorot_uniform')
                x = embd(inputs_text)
            
                count = '1'
                # x = tf.keras.layers.LSTM(opt.embd_size, return_sequences=False, recurrent_dropout=0.25, trainable=trainable)(x)
                x = tf.keras.layers.GlobalAveragePooling1D(name=pf+'AP'+count)(x)
                x = tf.keras.layers.BatchNormalization(name=pf+'BN'+count)(x)
                x = tf.keras.layers.Activation('elu', name=pf+'A'+count)(x)
                x = tf.keras.layers.Dropout(0.25, name=pf+'DR'+count)(x)

                count = '2'
                x = tf.keras.layers.Dense(512, trainable=trainable, name=pf+'DS'+count)(x)
                x = tf.keras.layers.BatchNormalization(trainable=trainable, name=pf+'BN'+count)(x)
                x = tf.keras.layers.Activation('elu', name=pf+'A'+count)(x)
                x = tf.keras.layers.Dropout(0.5, name=pf+'DR'+count)(x) #0.25=>0.5

                count = '3'
                x = tf.keras.layers.Dense(2048, trainable=trainable, name=pf+'DS'+count)(x)
                x = tf.keras.layers.BatchNormalization(trainable=trainable, name=pf+'BN'+count)(x)
                x = tf.keras.layers.Activation('elu', name=pf+'A'+count)(x)
                x = tf.keras.layers.Dropout(0.5, name=pf+'DR'+count)(x) #0.25=>0.5

                count = '4'
                x = tf.keras.layers.Dense(512, trainable=trainable, name=pf+'DS'+count)(x)
                x = tf.keras.layers.BatchNormalization(trainable=trainable, name=pf+'BN'+count)(x)
                x = tf.keras.layers.Activation('elu', name=pf+'A'+count)(x)
                x = tf.keras.layers.Dropout(0.5, name=pf+'DR'+count)(x) #0.25=>0.5

                with tf.name_scope('classify') :
                    b = tf.keras.layers.Dense(self.N_Cb, activation='softmax', name=pf+'b')(x)
                    m = tf.keras.layers.Dense(self.N_Cm, activation='softmax', name=pf+'m')(x)
                    s = tf.keras.layers.Dense(self.N_Cs, activation='softmax', name=pf+'s')(x)
                    d = tf.keras.layers.Dense(self.N_Cd, activation='softmax', name=pf+'d')(x)

            model = tf.keras.Model(inputs=[inputs_text], outputs=[b, m, s, d])
        return model

    def ensemble_model(self, models, load=False) :
        if load :
            load_path = opt.model_root+'ensemble/model.h5'
            model = tf.keras.models.load_model(load_path)
        else :
            for i in range(len(models)) :
                model = models[i]
                with tf.name_scope('model_'+str(i)) :
                    for layer in model.layers[1:-4] :
                        layer.trainable = False

            inputs = [model.input for model in models]
            concate_layers = [model.layers[-6].output for model in models]
            # print(concate_layers)

            x = tf.keras.layers.concatenate(concate_layers, axis=-1)        
            x = tf.keras.layers.Dense(1024)(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Activation('elu')(x)
            # x = tf.keras.layers.Dropout(0.25)(x)

            with tf.name_scope('classify') :
                b = tf.keras.layers.Dense(self.N_Cb, activation='softmax', name='b')(x)
                m = tf.keras.layers.Dense(self.N_Cm, activation='softmax', name='m')(x)
                s = tf.keras.layers.Dense(self.N_Cs, activation='softmax', name='s')(x)
                d = tf.keras.layers.Dense(self.N_Cd, activation='softmax', name='d')(x)

            model = tf.keras.Model(inputs=inputs, outputs=[b,m,s,d])

        return model


    
    def model_image1(self, trainable=True, load=False) :
        if load :
            load_path = opt.model_root+'image/model.h5'
            model = tf.keras.models.load_model(load_path)
            # model = self.model_image(load=False)
            # latest = 'ensemble/image/cp-0001-4.166.ckpt'
            # model.load_weights(latest)
            # model.save('img_model_fromMac.h5')
            # print("Restore saved weights on {}.".format(latest))

        else :
            with tf.variable_scope('image_classifier') :
                pf = 'img'

                inputs_img = tf.keras.Input(shape=(self.N_IMG_FEAT,), name=pf+'IN')

                count='1'
                x = tf.keras.layers.Dense(100, trainable=trainable, name=pf+'DS'+count)(inputs_img)
                x = tf.keras.layers.BatchNormalization(trainable=trainable, name=pf+'BN'+count)(x)
                x = tf.keras.layers.Activation('elu', name=pf+'A'+count)(x)
                # x_res = tf.keras.layers.Dropout(0.5, name=pf+'DR'+count)(x)

                # count='2'
                # x = tf.keras.layers.Dense(, trainable=trainable, name=pf+'DS'+count)(x_res)
                # x = tf.keras.layers.BatchNormalization(trainable=trainable, name=pf+'BN'+count)(x)
                # x = tf.keras.layers.Activation('elu', name=pf+'A'+count)(x)
                # x = tf.keras.layers.Dropout(0.25, name=pf+'DR'+count)(x)

                # count='3'
                # x = tf.keras.layers.Dense(1024, trainable=trainable, name=pf+'DS'+count)(x)
                # x = tf.keras.layers.concatenate([x, x_res], axis=-1, name=pf+'CON'+count)
                # x = tf.keras.layers.BatchNormalization(trainable=trainable, name=pf+'BN'+count)(x)
                # x = tf.keras.layers.Activation('relu', name=pf+'A'+count)(x)
                # x = tf.keras.layers.Dropout(0.25, name=pf+'DR'+count)(x)


                with tf.name_scope('classify') :
                    b = tf.keras.layers.Dense(self.N_Cb, activation='softmax', name=pf+'b')(x)
                    m = tf.keras.layers.Dense(self.N_Cm, activation='softmax', name=pf+'m')(x)
                    s = tf.keras.layers.Dense(self.N_Cs, activation='softmax', name=pf+'s')(x)
                    d = tf.keras.layers.Dense(self.N_Cd, activation='softmax', name=pf+'d')(x)

            model = tf.keras.Model(inputs=[inputs_img], outputs=[b, m, s, d])

        return model

    def model_image2(self, trainable=True, load=False) :
        if load :
            model = tf.keras.models.load_model("./ensemble/image/model.h5")
            # model = self.model_image(load=False)
            # latest = 'ensemble/image/cp-0008-4.215.ckpt'
            # model.load_weights(latest)
            # print("Restore saved weights on {}.".format(latest))

        else :
            with tf.variable_scope('image_classifier') :
                pf = 'img'

                inputs_img = tf.keras.Input(shape=(self.N_IMG_FEAT,), name=pf+'IN')

                count='1'
                x = tf.keras.layers.Dense(512, trainable=trainable, name=pf+'DS'+count)(inputs_img)
                x = tf.keras.layers.BatchNormalization(trainable=trainable, name=pf+'BN'+count)(x)
                x = tf.keras.layers.Activation('relu', name=pf+'A'+count)(x)
                x_res = tf.keras.layers.Dropout(0.2, name=pf+'DR'+count)(x)

                count='2'
                x = tf.keras.layers.Dense(512, trainable=trainable, name=pf+'DS'+count)(x_res)
                x = tf.keras.layers.BatchNormalization(trainable=trainable, name=pf+'BN'+count)(x)
                x = tf.keras.layers.Activation('relu', name=pf+'A'+count)(x)
                x = tf.keras.layers.Dropout(0.2, name=pf+'DR'+count)(x)

                count='3'
                x = tf.keras.layers.Dense(512, trainable=trainable, name=pf+'DS'+count)(x)
                x = tf.keras.layers.concatenate([x, x_res], axis=-1, name=pf+'CON'+count)
                x = tf.keras.layers.BatchNormalization(trainable=trainable, name=pf+'BN'+count)(x)
                x = tf.keras.layers.Activation('relu', name=pf+'A'+count)(x)
                x = tf.keras.layers.Dropout(0.2, name=pf+'DR'+count)(x)


                with tf.name_scope('classify') :
                    b = tf.keras.layers.Dense(self.N_Cb, activation='softmax', name=pf+'b')(x)
                    m = tf.keras.layers.Dense(self.N_Cm, activation='softmax', name=pf+'m')(x)
                    s = tf.keras.layers.Dense(self.N_Cs, activation='softmax', name=pf+'s')(x)
                    d = tf.keras.layers.Dense(self.N_Cd, activation='softmax', name=pf+'d')(x)

            model = tf.keras.Model(inputs=[inputs_img], outputs=[b, m, s, d])

        return model


    def model_text1(self, trainable = True, load=False) : 
        if load :
            model = tf.keras.models.load_model("./ensemble/text/model.h5")
            # model = self.model_text(load=False)
            # latest = 'ensemble/text/cp-0001-1.704.ckpt'
            # model.load_weights(latest)
            # print("Restore saved weights on {}.".format(latest))

        else :
            with tf.name_scope('text_classifier') :
                pf = 'GAP'

                inputs_text = tf.keras.Input(shape=(opt.max_len,), name=pf+'IN')

                embd = tf.keras.layers.Embedding(self.voca_size, self.embd_size, trainable=trainable, name=pf+'EM', embeddings_initializer='glorot_uniform')
                x = embd(inputs_text)
            
                count = '1'
                # x = tf.keras.layers.LSTM(opt.embd_size, return_sequences=False, recurrent_dropout=0.25, trainable=trainable)(x)
                x = tf.keras.layers.GlobalAveragePooling1D(name=pf+'AP'+count)(x)
                x = tf.keras.layers.Dropout(0.25, name=pf+'DR'+count)(x)

                count = '2'
                x = tf.keras.layers.Dense(1024, trainable=trainable, name=pf+'DS'+count)(x)
                x = tf.keras.layers.BatchNormalization(trainable=trainable, name=pf+'BN'+count)(x)
                x = tf.keras.layers.Activation('elu', name=pf+'A'+count)(x)
                x = tf.keras.layers.Dropout(0.5, name=pf+'DR'+count)(x) #0.25=>0.5

                count = '3'
                x = tf.keras.layers.Dense(1024, trainable=trainable, name=pf+'DS'+count)(x)
                x = tf.keras.layers.BatchNormalization(trainable=trainable, name=pf+'BN'+count)(x)
                x = tf.keras.layers.Activation('elu', name=pf+'A'+count)(x)
                x = tf.keras.layers.Dropout(0.5, name=pf+'DR'+count)(x) #0.25=>0.5

                with tf.name_scope('classify') :
                    b = tf.keras.layers.Dense(self.N_Cb, activation='softmax', name=pf+'b')(x)
                    m = tf.keras.layers.Dense(self.N_Cm, activation='softmax', name=pf+'m')(x)
                    s = tf.keras.layers.Dense(self.N_Cs, activation='softmax', name=pf+'s')(x)
                    d = tf.keras.layers.Dense(self.N_Cd, activation='softmax', name=pf+'d')(x)

            model = tf.keras.Model(inputs=[inputs_text], outputs=[b, m, s, d])
        return model

    def load_models(self, paths) :
        return [tf.keras.models.load_model(path) for path in paths]


    def train(self, datafile=opt.data_root + 'train/data.h5py', case = 'image', load=True, batch_size=opt.batch_size, save_dir=opt.model_root, lr=opt.lr,loss_weights=opt.loss_weights,num_epochs=opt.num_epochs) :
        def generator(ds, batch_size, case='image', raise_stop_event=False):
            """
            data generator for training and validation.
            providing dataset by size of batch_size.
            yeild [1.img feature, 2. text feature], [3.onehot of category b,
                                                        4.onehot of category m,
                                                        5.onehot of category s,
                                                        6.onehot of category d]
            refer sample_generator of kakao basecode.
            """
            left, limit = 0, ds['img_feat'].shape[0]
            while True:
                right = min(left + batch_size, limit)
                
                X_img = ds['img_feat'][left:right]
                X_text = ds['uni'][left:right]
                # X_text_brand = ds['uni_brand'][left:right]

                onehots = dict()
                for kind in ['b', 'm', 's', 'd'] :
                    Y = ds[kind][left:right]
                    # onehots[kind] = [tf.one_hot(self.C_idx[kind][x], depth=len(self.C_idx[kind])).numpy() 
                    #             if x!=-1 else tf.zeros((len(self.C_idx[kind]),)).numpy() for x in Y]
                    size = right-left
                    onehots[kind] = np.zeros((size, len(self.C_idx[kind])))
                    for i in range(size) :
                        if Y[i] != -1 :
                            onehots[kind][i,self.C_idx[kind][Y[i]]] = 1  

                # yield [X_img, X_text, X_text_brand], [onehots['b'], onehots['m'],onehots['s'],onehots['d']]

                if case == 'image' :
                    yield [X_img], [onehots['b'], onehots['m'],onehots['s'],onehots['d']]
                elif case == 'text':
                    yield [X_text], [onehots['b'], onehots['m'],onehots['s'],onehots['d']]
                elif case == 'ensemble' :
                    yield [X_img, X_text], [onehots['b'], onehots['m'],onehots['s'],onehots['d']]



                left = right
                if left == limit:
                    left = 0
                    if raise_stop_event:
                        raise StopIteration

        def acc_igm1(y_true, y_pred) :
            """
            custom accuracy function which only considers known-labels(not -1)
            """
            known = K.equal(K.max(y_true, axis=-1), 1)
            correct = K.equal(K.argmax(y_true, axis=-1), K.argmax(y_pred, axis=-1))
            true = K.all(K.stack([correct, known], axis=0), axis=0)
            # print(np.sum(true)/np.sum(known))
            return K.sum(K.cast(true, 'int32'))/K.sum(K.cast(known, 'int32'))
            # return np.sum(true)/np.sum(known)

        # construct model
        if case == 'image' :
            model = self.model_image(load=load)
        elif case == 'text' :
            model = self.model_text(load=load)
        elif case == 'ensemble' :
            models = self.load_models([opt.model_root+'image/model.h5',opt.model_root+'text/model.h5'])
            model = self.ensemble_model(models, load=load)
        else :
            assert False, 'wrong input. case must be one of ["image", "text", "ensemble"]'


        # model = tf.keras.models.load_model("./mymodel1/my_model.h5")
        # model = self.model()

        # model.compile(optimizer=tf.train.MomentumOptimizer(0.001, momentum=0.9, use_nesterov=True), 
        # model.compile(optimizer=tf.keras.optimizers.Adam(lr = 0.0001, amsgrad=True), 
        model.compile(optimizer=tf.train.AdamOptimizer(lr), 
                        loss='categorical_crossentropy', 
                        metrics=[acc_igm1],
                        loss_weights=loss_weights)
        
        # get train dataset
        data = h5py.File(datafile, 'r')
        assert data.get('train'), "datafile must have 'train' group in data."
        assert data.get('dev'), "datafile must have 'dev' group in data."
        m = data['train']['img_feat'].shape[0]
        n_batch = int((m-1)/batch_size) + 1
        train_generator = generator(data['train'], batch_size, case = case)

        m_val = data['dev']['img_feat'].shape[0]
        n_batch_val = int((m_val-1)/batch_size) + 1
        val_generator = generator(data['dev'], batch_size, case = case)

        ## checkpoint callback
        ckpt_path = save_dir + case + '/' + 'cp-{epoch:04d}-{val_loss:.3f}.ckpt'
        ckpt_dir = os.path.dirname(ckpt_path)
        if not os.path.isdir(ckpt_dir) :
            os.makedirs(ckpt_dir)


        cp_callback = tf.keras.callbacks.ModelCheckpoint(ckpt_path,
                                                save_weights_only=True,
                                                verbose=1,
                                                period=2,
                                                save_best_only=False)

        # train model
        model.summary()
        print("Start training ShopNet('^')v {} on {:,} train-data.".format(case, m))
        history = model.fit_generator(train_generator, steps_per_epoch=n_batch, epochs=num_epochs, 
                                    validation_data=val_generator, validation_steps=n_batch_val, 
                                    callbacks=[cp_callback])      

        # save model

        model_path = save_dir + case + '/model.h5'
        model.save(model_path)

        open(save_dir + 'history.pk', 'wb').write(pickle.dumps(history.history, 2))




    def predict(self, model_path, datafile, datakind="train", writefile = './predict_result.tsv') :
        """
        predict using trained model
        """

        # model = self.model()
        # model.load_weights(model_dir + 'cp-0004-1.628.ckpt')
        
        # model_path = model_dir + 'ensemble_model.h5'
        # model_path = model_dir + 'ensemble/model.h5'

        model = tf.keras.models.load_model(model_path) # with tf eager execution
        # model = tf.keras.models.load_model(model_path, custom_objects={'acc_igm1' : acc_igm1})
        print(model.summary())

        # datafile = opt.data_root + datakind + '/data.h5py'
        ds = h5py.File(datafile, 'r')['dev']

        m = ds['img_feat'].shape[0]
        step = 100000
        result = {}

        # to change index of max probability to category label.
        idx_C = dict()
        for kind in ['b','m','s','d'] :
            idx_C[kind] = {idx:C for C,idx in self.C_idx[kind].items()}

        for i in range(0,m,step) :
            pids = ds['pid'][i:min(m, i+step)]
            img_feat = ds['img_feat'][i:min(m, i+step)]
            text_feat = ds['uni'][i:min(m, i+step)]
            # text_brand_feat = ds['uni_brand'][i:min(m, i+step)]

            prob_b, prob_m, prob_s, prob_d = model.predict([img_feat, text_feat])
            # prob_b, prob_m, prob_s, prob_d = model.predict([img_feat])
            c_b = [idx_C['b'].get(int(each)) for each in np.argmax(prob_b, axis=-1)]
            c_m = [idx_C['m'].get(int(each)) for each in np.argmax(prob_m, axis=-1)]
            c_s = [idx_C['s'].get(int(each)) for each in np.argmax(prob_s, axis=-1)]
            c_d = [idx_C['d'].get(int(each)) for each in np.argmax(prob_d, axis=-1)]

            for each, pid in enumerate(pids) :
                result[pid.decode('utf8')] = [c_b[each], c_m[each], c_s[each], c_d[each]]
            # result.extend(list(zip(pids, c_b, c_m, c_s, c_d)))

            if datakind == "train" :
                true_b = ds['b'][i:min(m, i+step)]
                true_m = ds['m'][i:min(m, i+step)]
                true_s = ds['s'][i:min(m, i+step)]
                true_d = ds['d'][i:min(m, i+step)]

                show = 10
                print('textfeat', pids[:show])
                print('textfeat', text_feat[:show])
                print('true b', true_b[:show])
                print('pred b', c_b[:show])
                print('true m', true_m[:show])
                print('pred m', c_m[:show])
                print('true s', true_s[:show])
                print('pred s', c_s[:show])
                print('true d', true_d[:show])
                print('pred d', c_d[:show])

                print('result', list(result.items())[:3])

                acc_b = sum(true_b==c_b) / len(true_b)
                acc_m = sum(true_m==c_m) / len(true_m)
                acc_s = sum((true_s==c_s)&(true_s!=-1)) / sum(true_s!=-1)
                acc_d = sum((true_d==c_d)&(true_d!=-1)) / sum(true_d!=-1)

                print('-'*50)
                print('accuracy(only known) : ', acc_b,acc_m,acc_s, acc_d)
                print('-'*50)

                # Show wrong predicted cases
                print('what\'re wrongs?')
                wrong = (true_b!=c_b) + (true_m!=c_m) + ((true_s!=c_s) & (true_s!=-1)) + ((true_d!=c_d) & (true_d!=-1))
                print('{} wrong cases in {}'.format(sum(wrong), len(true_b)))
                for j in range(min(show,len(wrong))) :
                    print('pids : {}, true: {},{},{},{}, predict: {},{},{},{}'.format(pids[wrong][j].decode('utf8'), true_b[wrong][j], true_m[wrong][j], true_s[wrong][j], true_d[wrong][j], np.array(c_b)[wrong][j], np.array(c_m)[wrong][j], np.array(c_s)[wrong][j],np.array(c_d)[wrong][j]))
                    print('text', set(text_feat[wrong][j]))

                return


        print('Successfully predict for {} data!'.format(m))
        self.write_prediction(result, datakind, writefile)

    def write_prediction(self, result, datakind, writefile) :
        ORDERED_LIST = []
        pid_order = []
        if datakind == "dev" :
            ORDERED_LIST = opt.dev_data_list
        elif datakind == "test" :
            ORDERED_LIST = opt.test_data_list

        for data_path in ORDERED_LIST:
            h = h5py.File(data_path, 'r')[datakind]
            pid_order.extend(h['pid'][::])
        self.logger.info("Writing {} pids' classified results".format(len(pid_order)))

        tpl = '{pid}\t{b}\t{m}\t{s}\t{d}'

        with open(writefile, 'w') as fout:
            for b_pid in pid_order : 
                pid = b_pid.decode('utf8')
                if result.get(pid) == None :
                    b,m,s,d = -1,-1,-1,-1
                    self.logger.warning("EMPTY PID")
                else :
                    b,m,s,d = result.get(pid)
            # for pid, b, m, s, d in result:
                # print(pid.decode('utf8'), b, m, s, d)
                ans = tpl.format(pid=pid, b=b, m=m, s=s, d=d)
                fout.write(ans)
                fout.write('\n')
        
        print('predict results are written on {}'.format(writefile))


if __name__ == '__main__':
    shopnet = ShopNet()
    fire.Fire({'train': shopnet.train,
               'predict': shopnet.predict})











