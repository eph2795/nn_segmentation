
# coding: utf-8

# In[1]:

from collections import OrderedDict
from lasagne.layers import InputLayer
from lasagne.layers import DenseLayer
from lasagne.layers import NonlinearityLayer
from lasagne.layers import Pool2DLayer as PoolLayer
from lasagne.layers import Conv2DLayer as ConvLayer
from lasagne.layers import BatchNormLayer, batch_norm
from lasagne.nonlinearities import elu, softmax
import theano.tensor as T
import lasagne.layers
import theano
from lasagne.init import GlorotNormal
from lasagne.layers import (InputLayer, ConcatLayer, Pool2DLayer, ReshapeLayer, DimshuffleLayer, NonlinearityLayer,
DropoutLayer, Deconv2DLayer, batch_norm)
from sklearn.utils import shuffle

random_state_ = 42

# In[ ]:

# def iterate_minibatches(X, y, batchsize):
#     X_shuffled, y_shuffled = shuffle(X, y, random_state=random_state_)
#     N = X.shape[0]
#     batches_number = N // batchsize + (N % batchsize != 0)
#     return [(X_shuffled[i * batchsize:min((i + 1) * batchsize, N), :, :, :],
#              y_shuffled[i * batchsize:min((i + 1) * batchsize, N)]) for i in range(batches_number)]


def iterate_minibatches(ids, batchsize):
#    shuffled_ids = shuffle(ids, random_state=random_state_)
    shuffled_ids = shuffle(ids)
    N = shuffled_ids.size
    batches_number = N // batchsize + (N % batchsize != 0)
    return [shuffled_ids[i * batchsize:min((i + 1) * batchsize, N)] for i in range(batches_number)]


# In[2]:

def build_UNet(n_input_channels=3, BATCH_SIZE=None, num_output_classes=2,
               pad='same', nonlinearity=elu,
               input_dim=(128, 128), base_n_filters=64, do_dropout=False, weights=None):
    net = OrderedDict()
    net['input'] = InputLayer(
        (BATCH_SIZE, n_input_channels, input_dim[0], input_dim[1])
    )

    net['contr_1_1'] = batch_norm(
        ConvLayer(
            net['input'], num_filters=base_n_filters, filter_size=3, 
            nonlinearity=nonlinearity, pad=pad, W=GlorotNormal(),
        )
    )
    net['contr_1_2'] = batch_norm(
        ConvLayer(
            net['contr_1_1'], num_filters=base_n_filters, filter_size=3,
            nonlinearity=nonlinearity, pad=pad, W=GlorotNormal()
        )
    )
    net['pool1'] = Pool2DLayer(net['contr_1_2'], pool_size=2)

    net['contr_2_1'] = batch_norm(
        ConvLayer(
            net['pool1'], num_filters=base_n_filters*2, filter_size=3,
            nonlinearity=nonlinearity, pad=pad, W=GlorotNormal()
        )
    )
    net['contr_2_2'] = batch_norm(
        ConvLayer(
            net['contr_2_1'], num_filters=base_n_filters*2, filter_size=3, 
            nonlinearity=nonlinearity, pad=pad, W=GlorotNormal()
        )
    )
    net['pool2'] = Pool2DLayer(net['contr_2_2'], pool_size=2)

    net['contr_3_1'] = batch_norm(
        ConvLayer(
            net['pool2'], num_filters=base_n_filters*4, filter_size=3, 
            nonlinearity=nonlinearity, pad=pad, W=GlorotNormal()
        )
    )
    net['contr_3_2'] = batch_norm(
        ConvLayer(
            net['contr_3_1'], num_filters=base_n_filters*4, filter_size=3, 
            nonlinearity=nonlinearity, pad=pad, W=GlorotNormal()
        )
    )
    net['pool3'] = Pool2DLayer(net['contr_3_2'], pool_size=2)

    net['contr_4_1'] = batch_norm(
        ConvLayer(
            net['pool3'], num_filters=base_n_filters*8, filter_size=3, 
            nonlinearity=nonlinearity, pad=pad, W=GlorotNormal()
        )
    )
    net['contr_4_2'] = batch_norm(
        ConvLayer(
            net['contr_4_1'], num_filters=base_n_filters*8, filter_size=3, 
            nonlinearity=nonlinearity, pad=pad, W=GlorotNormal()
        )
    )
    l = net['pool4'] = Pool2DLayer(net['contr_4_2'], pool_size=2)

    if do_dropout:
        l = DropoutLayer(l, p=0.4)

    net['encode_1'] = batch_norm(
        ConvLayer(
            l, num_filters=base_n_filters*16, filter_size=3, 
            nonlinearity=nonlinearity, pad=pad, W=GlorotNormal()
        )
    )
    net['encode_2'] = batch_norm(
        ConvLayer(
            net['encode_1'], num_filters=base_n_filters*16, filter_size=3, 
            nonlinearity=nonlinearity, pad=pad, W=GlorotNormal()
        )
    )
    net['upscale1'] = batch_norm(
        Deconv2DLayer(
            net['encode_2'], num_filters=base_n_filters*16, filter_size=2, 
            stride=2, crop="valid", nonlinearity=nonlinearity, W=GlorotNormal()
        )
    )

    net['concat1'] = ConcatLayer(
        [net['upscale1'], net['contr_4_2']], 
        cropping=(None, None, "center", "center")
    )
    net['expand_1_1'] = batch_norm(
        ConvLayer(
            net['concat1'], num_filters=base_n_filters*8, filter_size=3,
            nonlinearity=nonlinearity, pad=pad, W=GlorotNormal()
        )
    )
    net['expand_1_2'] = batch_norm(
        ConvLayer(
            net['expand_1_1'], num_filters=base_n_filters*8, filter_size=3,
            nonlinearity=nonlinearity, pad=pad, W=GlorotNormal()
        )
    )
    net['upscale2'] = batch_norm(
        Deconv2DLayer(
            net['expand_1_2'], num_filters=base_n_filters*8, filter_size=2,
            stride=2, crop="valid", nonlinearity=nonlinearity, W=GlorotNormal()
        )
    )

    net['concat2'] = ConcatLayer(
        [net['upscale2'], net['contr_3_2']], 
        cropping=(None, None, "center", "center")
    )
    net['expand_2_1'] = batch_norm(
        ConvLayer(
            net['concat2'], num_filters=base_n_filters*4, filter_size=3,
            nonlinearity=nonlinearity, pad=pad, W=GlorotNormal()
        )
    )
    net['expand_2_2'] = batch_norm(
        ConvLayer(
            net['expand_2_1'], num_filters=base_n_filters*4, filter_size=3,
            nonlinearity=nonlinearity, pad=pad, W=GlorotNormal()
        )
    )
    net['upscale3'] = batch_norm(
        Deconv2DLayer(
            net['expand_2_2'], num_filters=base_n_filters*4, filter_size=2, 
            stride=2, crop="valid", nonlinearity=nonlinearity, W=GlorotNormal()
        )
    )

    net['concat3'] = ConcatLayer(
        [net['upscale3'], net['contr_2_2']], 
        cropping=(None, None, "center", "center")
    )
    net['expand_3_1'] = batch_norm(
        ConvLayer(
            net['concat3'], num_filters=base_n_filters*2, filter_size=3, 
            nonlinearity=nonlinearity, pad=pad, W=GlorotNormal()
        )
    )
    net['expand_3_2'] = batch_norm(
        ConvLayer(
            net['expand_3_1'], num_filters=base_n_filters*2, filter_size=3, 
            nonlinearity=nonlinearity, pad=pad, W=GlorotNormal()
        )
    )
    net['upscale4'] = batch_norm(
        Deconv2DLayer(
            net['expand_3_2'], num_filters=base_n_filters*2, filter_size=2,
            stride=2, crop="valid", nonlinearity=nonlinearity, W=GlorotNormal()
        )
    )

    net['concat4'] = ConcatLayer(
        [net['upscale4'], net['contr_1_2']], 
        cropping=(None, None, "center", "center")
    )
    net['expand_4_1'] = batch_norm(
        ConvLayer(
            net['concat4'], num_filters=base_n_filters, filter_size=3,
            nonlinearity=nonlinearity, pad=pad, W=GlorotNormal()
        )
    )
    net['expand_4_2'] = batch_norm(
        ConvLayer(
            net['expand_4_1'], num_filters=base_n_filters, filter_size=3, 
            nonlinearity=nonlinearity, pad=pad, W=GlorotNormal()
        )
    )

    net['output_segmentation'] = ConvLayer(
        net['expand_4_2'], num_filters=num_output_classes, filter_size=1, nonlinearity=None
    )
    net['dimshuffle'] = DimshuffleLayer(net['output_segmentation'], (1, 0, 2, 3))
    net['reshapeSeg'] = ReshapeLayer(net['dimshuffle'], (num_output_classes, -1))
    net['dimshuffle2'] = DimshuffleLayer(net['reshapeSeg'], (1, 0))
    net['output_flattened'] = NonlinearityLayer(net['dimshuffle2'], nonlinearity=lasagne.nonlinearities.softmax)

    if weights is not None:
        lasagne.layers.set_all_param_values(net['output_flattened'], weights)
        
    return net


# In[ ]:



