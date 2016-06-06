# yet another version of the IDSIA network
# based on code from keras tutorial 
# http://keras.io/getting-started/sequential-model-guide/
from keras.models import Model, Sequential, model_from_json
from keras.layers import Dense, Activation, Flatten, Input
from keras.layers import Convolution2D, MaxPooling2D, UpSampling2D, merge, ZeroPadding2D
from keras import backend as K
from keras.optimizers import SGD
from generate_data import *
import multiprocessing
import sys
import matplotlib
import matplotlib.pyplot as plt
# loosing independence of backend for 
# custom loss function
import theano
import theano.tensor as T

rng = np.random.RandomState(7)
train_samples = 150 # under 210 mean its not using all images
val_samples = 10
learning_rate = 0.01

doTrain = int(sys.argv[1])

patchSize = 572 #140
patchSize_out = 564 #132


# need to define a custom loss, because all pre-implementations
# seem to assume that scores over patch add up to one which
# they clearly don't and shouldn't
def unet_crossentropy_loss(y_true, y_pred):
    epsilon = 1.0e-4
    y_pred_clipped = T.clip(y_pred, epsilon, 1.0-epsilon)
    loss_vector = -T.mean(y_true * T.log(y_pred_clipped) + (1-y_true) * T.log(1-y_pred_clipped), axis=1)
    average_loss = T.mean(loss_vector)
    return average_loss

def unet_block_down(input, nb_filter):
    # first convolutional block consisting of 2 conv layers plus activation, then maxpool.
    # All are valid area, not same
    conv1 = Convolution2D(nb_filter=nb_filter, nb_row=3, nb_col=3, subsample=(1,1),
                             init="normal", border_mode="valid")(input)
    act1 = Activation("relu")(conv1)
    conv2 = Convolution2D(nb_filter=nb_filter, nb_row=3, nb_col=3, subsample=(1,1),
                             init="normal", border_mode="valid")(act1)
    act2 = Activation("relu")(conv2)
    # now downsamplig with maxpool
    pool1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), border_mode="valid")(act2)
    return (act2, pool1)

def unet_block_up(input, nb_filter, down_block_out):
    print "This is unet_block_up"
    print "input ", input._keras_shape
    # upsampling
    up_sampled = UpSampling2D(size=(2,2))(input)
    print "upsampled ", up_sampled._keras_shape
    # up-convolution
    conv_up = Convolution2D(nb_filter=nb_filter, nb_row=2, nb_col=2, subsample=(1,1),
                             init="normal", border_mode="same")(up_sampled)
    print "up-convolution ", conv_up._keras_shape
    # concatenation with cropped high res output
    # this is too large and needs to be cropped
    print "to be merged with ", down_block_out._keras_shape

    padding_1 = int((down_block_out._keras_shape[2] - conv_up._keras_shape[2])/2)
    padding_2 = int((down_block_out._keras_shape[3] - conv_up._keras_shape[3])/2)
    print "padding: ", (padding_1, padding_2)
    conv_up_padded = ZeroPadding2D(padding=(padding_1, padding_2))(conv_up)

    merged = merge([conv_up_padded, down_block_out], mode='concat', concat_axis=1)
    print "merged ", merged._keras_shape
    # two 3x3 convolutions with ReLU
    # first one halves the feature channels
    conv1 = Convolution2D(nb_filter=nb_filter, nb_row=3, nb_col=3, subsample=(1,1),
                             init="normal", border_mode="valid")(merged)
    act1 = Activation("relu")(conv1)
    print "conv1 ", act1._keras_shape
    conv2 = Convolution2D(nb_filter=nb_filter, nb_row=3, nb_col=3, subsample=(1,1),
                             init="normal", border_mode="valid")(act1)
    act2 = Activation("relu")(conv2)
    print "conv2 ", act2._keras_shape
    
    return act2
    

if doTrain:
    # input data should be large patches as prediction is also over large patches
    print 
    print "=== building network ==="

    print "== BLOCK 1 =="
    input = Input(shape=(1, patchSize, patchSize))
    print "input ", input._keras_shape
    block1_act, block1_pool = unet_block_down(input=input, nb_filter=64)
    print "block1 ", block1_pool._keras_shape

    print "== BLOCK 2 =="
    block2_act, block2_pool = unet_block_down(input=block1_pool, nb_filter=128)
    print "block2 ", block2_pool._keras_shape

    print "== BLOCK 3 =="
    block3_act, block3_pool = unet_block_down(input=block2_pool, nb_filter=256)
    print "block3 ", block3_pool._keras_shape

    print "== BLOCK 4 =="
    block4_act, block4_pool = unet_block_down(input=block3_pool, nb_filter=512)
    print "block4 ", block4_pool._keras_shape

    print "== BLOCK 5 =="
    print "Pooled output is just for size check, not actually used from this block"
    block5_act, block5_pool = unet_block_down(input=block4_pool, nb_filter=1024)
    print "block5 ", block5_pool._keras_shape

    print "=============="
    print

    print "== BLOCK 4 UP =="
    block4_up = unet_block_up(input=block5_act, nb_filter=512, down_block_out=block4_act)
    print "block4 up", block4_up._keras_shape
    print

    print "== BLOCK 3 UP =="
    block3_up = unet_block_up(input=block4_up, nb_filter=256, down_block_out=block3_act)
    print "block3 up", block3_up._keras_shape
    print

    print "== BLOCK 2 UP =="
    block2_up = unet_block_up(input=block3_up, nb_filter=128, down_block_out=block2_act)
    print "block2 up", block2_up._keras_shape

    print
    print "== BLOCK 1 UP =="
    block1_up = unet_block_up(input=block2_up, nb_filter=64, down_block_out=block1_act)
    print "block1 up", block1_up._keras_shape

    print "== 1x1 convolution =="
    conv_end = Convolution2D(nb_filter=1, nb_row=1, nb_col=1, subsample=(1,1),
                             init="normal", border_mode="valid")(block1_up)
    output = Activation("sigmoid")(conv_end)
    print "output ", output._keras_shape
    output_flat = Flatten()(output)
    print "output flat ", output_flat._keras_shape
    model = Model(input=input, output=output_flat)

    sgd = SGD(lr=learning_rate, decay=0, momentum=0.0, nesterov=False)
    #model.compile(loss='mse', optimizer=sgd)
    model.compile(loss=unet_crossentropy_loss, optimizer=sgd)
    data_val = generate_experiment_data_patch_prediction(purpose='validate', nsamples=val_samples, patchSize=patchSize, outPatchSize=patchSize_out)
   
    data_x_val = data_val[0].astype(np.float32)
    data_x_val = np.reshape(data_x_val, [-1, 1, patchSize, patchSize])
    data_y_val = data_val[1].astype(np.float32)

    # start pool for data
    print "Starting worker."
    pool = multiprocessing.Pool(processes=1)
    futureData = pool.apply_async(stupid_map_wrapper, [[generate_experiment_data_patch_prediction,'train', train_samples, patchSize, patchSize_out]])
    
    best_val_loss_so_far = 1000
    
    for epoch in xrange(10000000):
        print "Waiting for data."
        data = futureData.get()
        
        data_x = data[0].astype(np.float32)
        data_x = np.reshape(data_x, [-1, 1, patchSize, patchSize])
        data_y = data[1].astype(np.float32)
        
        print "got new data"
        futureData = pool.apply_async(stupid_map_wrapper, [[generate_experiment_data_patch_prediction,'train', train_samples, patchSize, patchSize_out]])
 
        model.fit(data_x, data_y, batch_size=1, nb_epoch=1)
        
        validation_loss = model.evaluate(data_x_val, data_y_val, batch_size=1)
        print "validation loss ", validation_loss
        
        json_string = model.to_json()
        open('unet_keras.json', 'w').write(json_string)
        model.save_weights('unet_keras_weights.h5', overwrite=True) 
        
        if validation_loss < best_val_loss_so_far:
            best_val_loss_so_far = validation_loss
            print "NEW BEST MODEL"
            json_string = model.to_json()
            open('unet_keras_best.json', 'w').write(json_string)
            model.save_weights('unet_keras_best_weights.h5', overwrite=True) 

else:
    model = model_from_json(open('unet_keras.json').read())
    model.load_weights('unet_keras_weights.h5')
    
    sgd = SGD(lr=0.01, decay=0, momentum=0.0, nesterov=False)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=sgd)
    
    image = mahotas.imread('ac3_input_0141.tif')
    image = image[:patchSize,:patchSize]
    image = image / 255.0
    image = image - 0.5
    data = np.reshape(image, (1,1,patchSize,patchSize))
    probs = model.predict(x=data, batch_size = 1)
    probs = np.reshape(probs, (patchSize_out,patchSize_out))
    
    #plt.imshow(image); plt.figure()
    plt.imshow(1-probs); plt.show()
        
