from keras.models import Sequential
from keras.layers import Conv2D, Input, BatchNormalization
# from keras.layers.advanced_activations import LeakyReLU
from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD, Adam
import prepare_data as pd
import numpy
import math
import tensorflow as tf
import asl.active_shift2d_op as active_shift2d_op

def psnr(target, ref):
    # assume RGB image
    target_data = numpy.array(target, dtype=float)
    ref_data = numpy.array(ref, dtype=float)

    diff = ref_data - target_data
    diff = diff.flatten('C')

    rmse = math.sqrt(numpy.mean(diff ** 2.))

    return 20 * math.log10(255. / rmse)


def model():
   
    '''
    .Input("x: T")		// [ batch, in_channels, in_rows, in_cols]
    .Input("shift: T")	// [ 2, in_channels]
    .Output("output: T")
    .Attr("T: {float, double}")
    .Attr("strides: list(int)")
    .Attr("paddings: list(int)")
    .Attr("normalize: bool = false")
    .Attr("data_format: { 'NHWC', 'NCHW' } = 'NCHW' ")
    '''
    # lrelu = LeakyReLU(alpha=0.1)
    arr1 = numpy.random.random((8,1,9,9)) 
    shift1 = numpy.random.random((2,1))
    arr2 = numpy.random.random((8,1,3,3)) 
    shift2 = numpy.random.random((2,1))
    arr3 = numpy.random.random((8,1,5,5)) 
    shift3 = numpy.random.random((2,1))   

    #config = tf.ConfigProto()
    #config.gpu_options.allow_grouwth = True

    a1 = tf.constant(arr1, dtype=tf.float32)
    s1 = tf.constant(shift1, dtype = numpy.float32)
    a2 = tf.constant(arr2, dtype=tf.float32)
    s2 = tf.constant(shift2, dtype = numpy.float32)
    a3 = tf.constant(arr3, dtype=tf.float32)
    s3 = tf.constant(shift3, dtype = numpy.float32)

    SRCNN = Sequential()
    '''
    SRCNN.add(Conv2D(nb_filter=128, nb_row=9, nb_col=9, init='glorot_uniform',
                     activation='relu', border_mode='valid', bias=True, input_shape=(32, 32, 1)))
    SRCNN.add(Conv2D(nb_filter=64, nb_row=3, nb_col=3, init='glorot_uniform',
                     activation='relu', border_mode='same', bias=True))
    # SRCNN.add(BatchNormalization())
    SRCNN.add(Conv2D(nb_filter=1, nb_row=5, nb_col=5, init='glorot_uniform',
                     activation='linear', border_mode='valid', bias=True))
    '''
    #SRCNN.add(active_shift2d_op.active_shift2d_op(data, shift, grad, strides, paddings, normalize, data_format))
    SRCNN.add(active_shift2d_op.active_shift2d_op(a1,s1,strides=[1,1,1,1],paddings=[0,0,0,0]))
    SRCNN.add(active_shift2d_op.active_shift2d_op(a2,s2,strides=[1,1,1,1],paddings=[0,0,0,0]))
    SRCNN.add(active_shift2d_op.active_shift2d_op(a3,s3,strides=[1,1,1,1],paddings=[0,0,0,0]))



    adam = Adam(lr=0.0003)
    SRCNN.compile(optimizer=adam, loss='mean_squared_error', metrics=['mean_squared_error'])
    return SRCNN




def train():
    srcnn_model = model()
    print(srcnn_model.summary())
    data, label = pd.read_training_data("./crop_train.h5")
    val_data, val_label = pd.read_training_data("./test.h5")

    checkpoint = ModelCheckpoint("SRCNN_check.h5", monitor='val_loss', verbose=1, save_best_only=True,
                                 save_weights_only=False, mode='min')
    callbacks_list = [checkpoint]

    srcnn_model.fit(data, label, batch_size=128, validation_data=(val_data, val_label),
                    callbacks=callbacks_list, shuffle=True, nb_epoch=200, verbose=0)
    # srcnn_model.load_weights("m_model_adam.h5")


if __name__ == "__main__":
    train()
