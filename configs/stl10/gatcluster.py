model_name = "gatcluster"
num_workers = 4
device = 0

num_train = 5

num_cluster = 10
batch_size = 1000
target_sub_batch_size = 200
train_batch_size = batch_size
train_sub_batch_size = 32
num_trans_aug = 1
num_repeat = 8
fea_dim = 10
att_conv_dim = num_cluster
att_size = 6

max_iters = 6000

data_train = dict(
    type="stl10_gray",
    root_folder="./datasets/stl10",
    split="train+test",
    download=True,
    ims_per_batch=batch_size,
    shuffle=True,
    aspect_ratio_grouping=False,
    train=True,
    show=False,
    num_trans_aug=num_trans_aug,
)

data_test = dict(
    type="stl10_gray",
    root_folder="./datasets/stl10",
    split="train+test",
    download=True,
    shuffle=False,
    ims_per_batch=50,
    aspect_ratio_grouping=False,
    train=False,
    show=False,
)

model = dict(
    type="gattcluster",
    feature=dict(
        type="convnet",
        input_channel=1,
        conv_layers=[[64, 64, 64], 'max_pooling', [128, 128, 128], 'max_pooling', [256, 256, 256], 'max_pooling',
                    [fea_dim]],
        kernels=[[3, 3, 3], 2, [3, 3, 3], 2, [3, 3, 3], 2, [1]],
        strides=[[1, 1, 1], 2, [1, 1, 1], 2, [1, 1, 1], 2, [1]],
        pads=   [[0, 0, 0], 0, [0, 0, 0], 0, [0, 0, 0], 0, [0]],
        num_block=4,
        fc_input_neurons=None,
        fc_layers=[],
        batch_norm=True,
        transpose=False,
        return_pool_idx=False,
        last_conv_activation="relu",
        last_fc_activation=None,
        use_ave_pool=False,
        use_last_conv_bn=True,
    ),

    gaussian_att_cluster_head=dict(
        classifier_ori=dict(
            type="mlp",
            num_neurons=[fea_dim, num_cluster, num_cluster],
            last_activation="softmax"
        ),
        feature_conv=dict(
            type="convnet",
            input_channel=fea_dim,
            conv_layers=[[fea_dim]],
            kernels=[[1]],
            strides=[[1]],
            pads=[[0]],
            num_block=1,
            fc_input_neurons=None,
            fc_layers=[],
            batch_norm=True,
            transpose=False,
            return_pool_idx=False,
            last_conv_activation=None,
            last_fc_activation=None,
            use_ave_pool=False,
            use_last_conv_bn=True,
        ),
        theta_mlp=dict(
            type="mlp",
            num_neurons=[att_size*att_size*att_conv_dim, 3],
            last_activation="sigmoid"
        ),
        att_conv=dict(
            type="convnet",
            input_channel=fea_dim,
            conv_layers=[[num_cluster]],
            kernels=[[1]],
            strides=[[1]],
            pads=[[0]],
            num_block=1,
            fc_input_neurons=None,
            fc_layers=[],
            batch_norm=True,
            transpose=False,
            return_pool_idx=False,
            last_conv_activation="relu",
            last_fc_activation=None,
            use_ave_pool=False,
            use_last_conv_bn=True,
        ),
        classifier_att=dict(
            type="mlp",
            num_neurons=[fea_dim, num_cluster, num_cluster],
            last_activation="softmax"
        ),

        loss="bce",
        balance_scores=True,
        num_cluster=num_cluster,
        batch_size=batch_size,
        sub_batch_size=train_sub_batch_size,
        ignore_label=-1,
        fea_height=att_size,
        fea_width=att_size,
        loss_weight=dict(loss_sim=5, loss_ent=3, loss_rel=1, loss_att=5),
        lamda=0.05,
        num_att_map=1,
    ),

    weight=None,
)

solver = dict(
    type="adam",
    max_iter=max_iters,
    base_lr=0.001,
    bias_lr_factor=1,
    weight_decay=0,
    weight_decay_bias=0,
    checkpoint_period=5,
    target_sub_batch_size=target_sub_batch_size,
    train_batch_size=train_batch_size,
    train_sub_batch_size=train_sub_batch_size,
    num_repeat=num_repeat,
    sim_loss=True,
    ent_loss=True,
    rel_loss=True,
    att_loss=True,
)

results = dict(
    output_dir="./results/stl10/{}".format(model_name),
)