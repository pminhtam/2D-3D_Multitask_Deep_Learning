from keras import backend as K
from keras.models import Model
from keras.optimizers import SGD
from keras.layers import Input

from deephar.activations import channel_softmax_2d
from deephar.models.blocks import build_context_aggregation

from deephar.utils import *
from deephar.layers import *

from deephar.objectives import elasticnet_loss_on_valid_joints


def action_top(x, name=None):
    x = global_max_min_pooling(x)
    x = Activation('softmax', name=name)(x)
    return x


def build_act_pred_block(x, num_out, name=None, last=False, include_top=True):

    num_features = K.int_shape(x)[-1]

    ident = x
    x = act_conv_bn(x, int(num_features/2), (1, 1))
    x = act_conv_bn(x, num_features, (3, 3))
    x = add([ident, x])

    ident = x
    x1 = act_conv_bn(x, num_features, (3, 3))
    x = max_min_pooling(x1, (2, 2))
    action_hm = act_conv(x, num_out, (3, 3))
    y = action_hm
    if include_top:
        y = action_top(y)

    if not last:
        action_hm = UpSampling2D((2, 2))(action_hm)
        action_hm = act_conv_bn(action_hm, num_features, (3, 3))
        x = add([ident, x1, action_hm])

    return x, y


def build_pose_model(num_joints, num_actions, num_temp_frames=None, pose_dim=2,
        name=None, include_top=True, network_version='v1'):

    y = Input(shape=(num_temp_frames, num_joints, pose_dim))
    p = Input(shape=(num_temp_frames, num_joints, 1))

    ## Pose information
    mask = Lambda(lambda x: K.tile(x, [1, 1, 1, pose_dim]))(p)
    x = Lambda(lambda x: x[0] * x[1])([y, mask])

    if network_version == 'v1':
        a = conv_bn_act(x, 8, (3, 1))
        b = conv_bn_act(x, 16, (3, 3))
        c = conv_bn_act(x, 24, (3, 5))
        x = concatenate([a, b, c])
        a = conv_bn(x, 56, (3, 3))
        b = conv_bn(x, 32, (1, 1))
        b = conv_bn(b, 56, (3, 3))
        x = concatenate([a, b])
        x = max_min_pooling(x, (2, 2))
    elif network_version == 'v2':
        a = conv_bn_act(x, 12, (3, 1))
        b = conv_bn_act(x, 24, (3, 3))
        c = conv_bn_act(x, 36, (3, 5))
        x = concatenate([a, b, c])
        a = conv_bn(x, 112, (3, 3))
        b = conv_bn(x, 64, (1, 1))
        b = conv_bn(b, 112, (3, 3))
        x = concatenate([a, b])
        x = max_min_pooling(x, (2, 2))
    else:
        raise Exception('Unkown network version "{}"'.format(network_version))

    x, y1 = build_act_pred_block(x, num_actions, name='y1',
            include_top=include_top)
    x, y2 = build_act_pred_block(x, num_actions, name='y2',
            include_top=include_top)
    x, y3 = build_act_pred_block(x, num_actions, name='y3',
            include_top=include_top)
    _, y4 = build_act_pred_block(x, num_actions, name='y4',
            include_top=include_top, last=True)
    x = [y1, y2, y3, y4]

    model = Model(inputs=[y, p], outputs=x, name=name)

    return model


def build_visual_model(num_joints, num_actions, num_features,
        num_temp_frames=None, name=None, include_top=True):

    inp = Input(shape=(num_temp_frames, num_joints, num_features))
    x = conv_bn(inp, 256, (1, 1))
    x = MaxPooling2D((2, 2))(x)
    x, y1 = build_act_pred_block(x, num_actions, name='y1',
            include_top=include_top)
    x, y2 = build_act_pred_block(x, num_actions, name='y2',
            include_top=include_top)
    x, y3 = build_act_pred_block(x, num_actions, name='y3',
            include_top=include_top)
    _, y4 = build_act_pred_block(x, num_actions, name='y4',
            include_top=include_top, last=True)
    model = Model(inp, [y1, y2, y3, y4], name=name)

    return model



def build_merge_model(
        num_actions,
        input_shape,
        num_frames,
        num_joints,
        pose_dim=2,
        depth_maps=8,
        num_context_per_joint=2,
        pose_net_version='v1',
        output_poses=False,
        weighted_merge=True,
        ar_pose_weights=None,
        ar_visual_weights=None,
        full_trainable=False):


    y_in = Input(shape=(num_frames,num_joints,pose_dim),dtype=float)
    p_in = Input(shape=(num_frames,num_joints,1),dtype=float)
    hs_in = Input(shape=(num_frames,32,32,num_joints),dtype=float)
    xb1_in = Input(shape=(num_frames,32,32,576),dtype=float)


    outputs = []
    if output_poses:
        outputs.append(y_in)
        outputs.append(p_in)

    model_pose = build_pose_model(num_joints, num_actions, num_frames,
            pose_dim=pose_dim, include_top=False, name='PoseAR',
            network_version=pose_net_version)
    # model_pose.trainable = False
    if ar_pose_weights is not None:
        model_pose.load_weights(ar_pose_weights)
    out_pose = model_pose([y_in, p_in])

    f = kronecker_prod(hs_in, xb1_in)
    num_features = K.int_shape(f)[-1]
    model_vis = build_visual_model(num_joints, num_actions, num_features,
            num_temp_frames=num_frames, include_top=False, name='GuidedVisAR')
    # model_vis.trainable = False
    if ar_visual_weights is not None:
        model_vis.load_weights(ar_visual_weights)
    out_vis = model_vis(f)

    for i in range(len(out_pose)):
        outputs.append(action_top(out_pose[i], name='p%d' % (i+1)))

    for i in range(len(out_vis)):
        outputs.append(action_top(out_vis[i], name='v%d' % (i+1)))

    p = out_pose[-1]
    v = out_vis[-1]

    def _heatmap_weighting(inp):
        num_filters = K.int_shape(inp)[-1]
        conv = SeparableConv2D(num_filters, (1, 1),
                use_bias=False)
        x = conv(inp)
        w = conv.get_weights()
        w[0].fill(1.)
        w[1].fill(0)
        for i in range(num_filters):
            w[1][0, 0, i, i] = 1.
        conv.set_weights(w)

        return x

    if weighted_merge:
        p = _heatmap_weighting(p)
        v = _heatmap_weighting(v)

    m = add([p, v])
    outputs.append(action_top(m, name='m'))
    outputs.append(action_top(p, name='mmm'))
    outputs.append(action_top(v, name='mmmvvv'))
    # """
    # model = Model(inputs = [model_pose.input, model_vis.input], outputs)
    model = Model(inputs = [y_in, p_in, hs_in, xb1_in], outputs = outputs)


    return model


def compile(model, lr=0.001, momentum=0.95, loss_weights=None,
        pose_predicted=False):

    if pose_predicted:
        losses = []
        losses.append(elasticnet_loss_on_valid_joints)
        losses.append('binary_crossentropy')
        for i in range(len(model.output) - 2):
            losses.append('categorical_crossentropy')

        model.compile(loss=losses,
                optimizer=SGD(lr=lr, momentum=momentum, nesterov=True),
                loss_weights=loss_weights)
    else:
        model.compile(loss='categorical_crossentropy',
                optimizer=SGD(lr=lr, momentum=momentum, nesterov=True),
                metrics=['acc'], loss_weights=loss_weights)

