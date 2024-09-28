import tensorflow as tf
from tensorflow.keras.models import Model  # layers, Sequential, optimizers, losses, metrics, datasets
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Embedding, Input
from tensorflow.keras.layers import GlobalAveragePooling1D,Dropout
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Add
import tensorflow.keras.backend as K
from tensorflow.keras import models, layers
from tensorflow.keras import regularizers
from tensorflow.keras import optimizers, losses, metrics, datasets
from tensorflow.keras.applications import EfficientNetB4, EfficientNetB7
# from keras.applications.efficientnet import EfficientNetB4, EfficientNetB7
from bgi.layers.attention import AttentionWithContext

class CausalSelfAttention(tf.keras.layers.Layer):
  def __init__(self, **kwargs):
    super().__init__()
    self.mha = tf.keras.layers.MultiHeadAttention(**kwargs)
    # Use Add instead of + so the keras mask propagates through.
    self.add = tf.keras.layers.Add() 
    self.layernorm = tf.keras.layers.LayerNormalization()

  def call(self, x):
    attn = self.mha(query=x, value=x, use_causal_mask=False)
    x = self.add([x, attn])
    print(x)
    return self.layernorm(x)
  
def make_spatial_RNA_image_model(multi_max_features: list = [40000],
                                 mult_feature_names: list = ['Gene'],
                                 embedding_dims=128,
                                 activation='softmax',
                                 head_1=128,
                                 head_2=128,
                                 head_3=128,
                                 drop_rate=0.05,
                                 include_attention: bool = False,
                                 model_type: int = 0
                                 ):
    assert len(multi_max_features) == len(mult_feature_names)

    # RNA encoder
    x_feature_inputs = []
    x_value_inputs = []
    features = []
    inputs = []
    if include_attention == True:
        max_length, name = multi_max_features[0], mult_feature_names[0]
        
        feature_input = Input(shape=(None,), name='Input-{}-Feature'.format(name))
        value_input = Input(shape=(None,), name='Input-{}-Value'.format(name), dtype='float')
        x_feature_inputs.append(feature_input)
        x_value_inputs.append(value_input)

        embedding = Embedding(max_length, embedding_dims, input_length=None, name='{}-Embedding'.format(name))(
            feature_input)

        sparse_value = tf.expand_dims(value_input, 2, name='{}-Expend-Dims'.format(name))
        sparse_value = BatchNormalization(name='{}-BN-1'.format(name))(sparse_value)
        x = tf.multiply(embedding, sparse_value, name='{}-Multiply'.format(name))
        # x = BatchNormalization(name='{}-BN-2'.format(name))(x)

        # Attention
        weight_output,a = AttentionWithContext()(x)
        x = K.tanh(K.sum(weight_output, axis=1))
        x = BatchNormalization(name='{}-BN-3'.format(name))(x)
        features.append(x)

        image_value_input = Input(shape=(multi_max_features[1], multi_max_features[1], 3), name='Image-Input-{}-Value'.format(mult_feature_names[1]), dtype='float')
        x_value_inputs.append(image_value_input)

        inputs.append(x_feature_inputs)
        inputs.append(x_value_inputs)

    else:
        max_length, name = multi_max_features[0], mult_feature_names[0]
        value_input = Input(shape=(max_length,), name='Input-{}-Value'.format(name), dtype='float')
        x_value_inputs.append(value_input)
        sparse_value = BatchNormalization(name='{}-BN-1'.format(name))(value_input)
        x = Dense(head_1, name='{}-projection-0'.format(name), activation='relu')(sparse_value)
        x = BatchNormalization(name='{}-BN-3'.format(name))(x)

        features.append(x)

        image_value_input = Input(shape=(multi_max_features[1], multi_max_features[1], 3), name='Image-Input-{}-Value'.format(mult_feature_names[1]), dtype='float')
        x_value_inputs.append(image_value_input)

        inputs.append(x_value_inputs)
        print(inputs)

    dropout0 = Dropout(rate=drop_rate)(features[0])
    output0 = Dense(head_1, name='projection-0', activation='relu')(dropout0)

    ####
    # Image encoder
    max_length, name = multi_max_features[1], mult_feature_names[1]
    
    if include_attention:
        if model_type == 0:
            # VGG16
            image_network = models.Sequential([
                layers.Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(max_length, max_length, 3), data_format="channels_last", strides=(2, 2)),
                layers.Conv2D(64, (3, 3), activation='relu', padding='same', data_format="channels_last", strides=(2, 2)),
                BatchNormalization(),

                layers.MaxPooling2D(pool_size=(2, 2), padding='same'),
                layers.Conv2D(128, (3, 3), activation='relu', padding='same', data_format="channels_last", strides=(2, 2)),
                layers.Conv2D(128, (3, 3), activation='relu', padding='same', data_format="channels_last", strides=(2, 2)),
                BatchNormalization(),

                layers.MaxPooling2D(pool_size=(2, 2), padding='same'),
                layers.Conv2D(256, (3, 3), activation='relu', padding='same', data_format="channels_last", strides=(2, 2)),
                layers.Conv2D(256, (3, 3), activation='relu', padding='same', data_format="channels_last", strides=(2, 2)),
                layers.Conv2D(256, (3, 3), activation='relu', padding='same', data_format="channels_last", strides=(2, 2)),
                BatchNormalization(),

                layers.MaxPooling2D(pool_size=(2, 2), padding='same'),
                layers.Conv2D(512, (3, 3), activation='relu', padding='same', data_format="channels_last", strides=(2, 2)),
                layers.Conv2D(512, (3, 3), activation='relu', padding='same', data_format="channels_last", strides=(2, 2)),
                layers.Conv2D(512, (3, 3), activation='relu', padding='same', data_format="channels_last", strides=(2, 2)),
                BatchNormalization(),
                
                layers.MaxPooling2D(pool_size=(2, 2), padding='same'),
                layers.Conv2D(512, (3, 3), activation='relu', padding='same', data_format="channels_last", strides=(2, 2)),
                layers.Conv2D(512, (3, 3), activation='relu', padding='same', data_format="channels_last", strides=(2, 2)),
                layers.Conv2D(512, (3, 3), activation='relu', padding='same', data_format="channels_last", strides=(2, 2)),
                BatchNormalization(),
                layers.MaxPooling2D(pool_size=(2, 2), padding='same'),
                layers.Flatten(),
                Dropout(rate=drop_rate),

                Dense(4096, activation='relu'),
                BatchNormalization(),
                Dense(2048, activation='relu'),
                BatchNormalization(),
                Dense(head_1, activation='relu')
            ])
            output1 = image_network(image_value_input)
        else:
            # EfficientNet B7
            base_model2 = EfficientNetB7(
                input_shape=(multi_max_features[1], multi_max_features[1], 3),
                include_top=False,
                weights='model_weights/efficientnetb7_notop.h5',
            )
            base_model2.build(input_shape=(multi_max_features[1], multi_max_features[1], 3))
            base_model2.layers.pop()
            base_model = Model(base_model2.input, base_model2.layers[-1].output)
            
            image_network = models.Sequential([
                base_model,
                layers.MaxPooling2D(pool_size=(2, 2), padding='valid'),
                layers.Flatten(),
                Dense(head_1, name='{}-projection-0'.format(name), activation='relu')
            ])
            print(image_network.summary())
            output1 = image_network(image_value_input)

    else:
        if model_type == 0:
            # CNNs
            image_network = models.Sequential([
                layers.Conv2D(32, (3, 3), activation='relu', padding='valid', input_shape=(max_length, max_length, 3), data_format="channels_last", strides=(2, 2)),
                BatchNormalization(),
                layers.MaxPooling2D(pool_size=(2, 2), padding='valid'),
                layers.Conv2D(64, (3, 3), activation='relu', padding='valid', data_format="channels_last", strides=(2, 2)),
                BatchNormalization(),
                layers.MaxPooling2D(pool_size=(2, 2), padding='valid'),
                layers.Conv2D(128, (3, 3), activation='relu', padding='valid', data_format="channels_last", strides=(2, 2)),
                layers.Flatten(),
                Dropout(rate=drop_rate),
                Dense(head_1, name='{}-projection-0'.format(name), activation='relu')
            ])
            output1 = image_network(image_value_input)
        else:
            # EfficientNet B4
            base_model2 = EfficientNetB4(
                input_shape=(multi_max_features[1], multi_max_features[1], 3),
                include_top=False,
                weights='model_weights/efficientnetb4_notop.h5',
            )
            base_model2.build(input_shape=(multi_max_features[1], multi_max_features[1], 3))
            base_model2.layers.pop()
            base_model = Model(base_model2.input, base_model2.layers[-1].output)
            
            image_network = models.Sequential([
                base_model,
                layers.MaxPooling2D(pool_size=(2, 2), padding='valid'),
                layers.Flatten(),
                Dense(head_1, name='{}-projection-0'.format(name), activation='relu')
            ])
            print(image_network.summary())
            output1 = image_network(image_value_input)

    return tf.keras.Model(inputs=inputs, outputs=[output0, output1])

def multi_embedding_attention_transfer(supvised_train: bool = False,
                                    scan_train: bool = False,
                                    multi_max_features: list = [40000],
                                    mult_feature_names: list = ['Gene'],
                                    embedding_dims=128,
                                    num_classes=5,
                                    activation='softmax',
                                    head_1=128,
                                    head_2=128,
                                    head_3=128,
                                    drop_rate=0.05,
                                    include_attention: bool = False,
                                    use_bias=True,
                                    combine_omics: bool = True,
                                    model_type: int = 0
                                    ):
    assert len(multi_max_features) == len(mult_feature_names)

    # 特征索引
    x_feature_inputs = []
    # 特征值
    x_value_inputs = []
    # 特征向量
    embeddings = []
    features = []
    weight_output_all = []
    if include_attention == True:
        for max_length, name in zip(multi_max_features, mult_feature_names):
            # 输入
            feature_input = Input(shape=(None,), name='Input-{}-Feature'.format(name))
            value_input = Input(shape=(None,), name='Input-{}-Value'.format(name), dtype='float')
            x_feature_inputs.append(feature_input)
            x_value_inputs.append(value_input)

            # 向量
            embedding = Embedding(max_length, embedding_dims, input_length=None, name='{}-Embedding'.format(name))(
                feature_input)

            # 向量 * 特征值
            sparse_value = tf.expand_dims(value_input, 2, name='{}-Expend-Dims'.format(name))
            sparse_value = BatchNormalization(name='{}-BN-1'.format(name))(sparse_value)
            x = tf.multiply(embedding, sparse_value, name='{}-Multiply'.format(name))
            # x = BatchNormalization(name='{}-BN-2'.format(name))(x)

            # # Attention
            weight_output,a = AttentionWithContext()(x)
            x = K.tanh(K.sum(weight_output, axis=1))

            x = BatchNormalization(name='{}-BN-3'.format(name))(x)

            features.append(x)
            #weight_output_all.append(a)
        inputs = []
        inputs.append(x_feature_inputs)
        inputs.append(x_value_inputs)

    else:
        for max_length, name in zip(multi_max_features, mult_feature_names):
            # 输入

            value_input = Input(shape=(max_length,), name='Input-{}-Value'.format(name), dtype='float')

            x_value_inputs.append(value_input)

            # 向量 * 特征值
            sparse_value = BatchNormalization(name='{}-BN-1'.format(name))(value_input)

            x = Dense(head_1, name='{}-projection-0'.format(name), activation='relu')(sparse_value)


            x = BatchNormalization(name='{}-BN-3'.format(name))(x)

            features.append(x)
        inputs = []
        inputs.append(x_value_inputs)

    if combine_omics:
        if len(features) > 1:
        #feature = concatenate(features)
            if model_type == 0:
                feature = Add()([features[0],features[1]])
            elif model_type == 1:
                cross_attention = CausalSelfAttention(num_heads=head_2, key_dim=256, dropout=drop_rate) # FIXME
                features[0] = tf.expand_dims(features[0], axis=1)
                features[1] = tf.expand_dims(features[1], axis=1)

                feature_attn = cross_attention(tf.concat([features[0], features[1]], 1))
                feature = tf.math.reduce_sum(feature_attn, axis=1)
        else:
            feature = features[0]
    
        dropout = Dropout(rate=drop_rate)(feature)
        output = Dense(head_1, name='projection-1', activation='relu')(dropout)
        
        return tf.keras.Model(inputs=inputs, outputs=output)
    
    else:
        dropout0 = Dropout(rate=drop_rate)(features[0])
        dropout1 = Dropout(rate=drop_rate)(features[1])

        output0 = Dense(head_1, name='projection-0', activation='relu')(dropout0)
        output1 = Dense(head_1, name='projection-1', activation='relu')(dropout1)

        return tf.keras.Model(inputs=inputs, outputs=[output0, output1])


def multi_embedding_attention_transfer(supvised_train: bool = False,
                                    scan_train: bool = False,
                                    multi_max_features: list = [40000],
                                    mult_feature_names: list = ['Gene'],
                                    embedding_dims=128,
                                    num_classes=5,
                                    activation='softmax',
                                    head_1=128,
                                    head_2=128,
                                    head_3=128,
                                    drop_rate=0.05,
                                    include_attention: bool = False,
                                    use_bias=True,
                                    combine_omics: bool = True,
                                    model_type: int = 0
                                    ):
    assert len(multi_max_features) == len(mult_feature_names)

    # 特征索引
    x_feature_inputs = []
    # 特征值
    x_value_inputs = []
    # 特征向量
    embeddings = []
    features = []
    weight_output_all = []
    if include_attention == True:
        for max_length, name in zip(multi_max_features, mult_feature_names):
            # 输入
            feature_input = Input(shape=(None,), name='Input-{}-Feature'.format(name))
            value_input = Input(shape=(None,), name='Input-{}-Value'.format(name), dtype='float')
            x_feature_inputs.append(feature_input)
            x_value_inputs.append(value_input)

            # 向量
            embedding = Embedding(max_length, embedding_dims, input_length=None, name='{}-Embedding'.format(name))(
                feature_input)

            # 向量 * 特征值
            sparse_value = tf.expand_dims(value_input, 2, name='{}-Expend-Dims'.format(name))
            print("Before")
            print(sparse_value)
            sparse_value = BatchNormalization(name='{}-BN-1'.format(name))(sparse_value)
            print("After")
            print(sparse_value)
            x = tf.multiply(embedding, sparse_value, name='{}-Multiply'.format(name))
            # x = BatchNormalization(name='{}-BN-2'.format(name))(x)

            # # Attention
            weight_output,a = AttentionWithContext()(x)
            x = K.tanh(K.sum(weight_output, axis=1))

            x = BatchNormalization(name='{}-BN-3'.format(name))(x)

            features.append(x)
            #weight_output_all.append(a)
        inputs = []
        inputs.append(x_feature_inputs)
        inputs.append(x_value_inputs)

    else:
        for max_length, name in zip(multi_max_features, mult_feature_names):
            # 输入

            value_input = Input(shape=(max_length,), name='Input-{}-Value'.format(name), dtype='float')

            x_value_inputs.append(value_input)

            # 向量 * 特征值
            sparse_value = BatchNormalization(name='{}-BN-1'.format(name))(value_input)

            x = Dense(head_1, name='{}-projection-0'.format(name), activation='relu')(sparse_value)


            x = BatchNormalization(name='{}-BN-3'.format(name))(x)

            features.append(x)
        inputs = []
        inputs.append(x_value_inputs)
    # Concatenate
    print(f"Before add modalities input {features}")
    print(f"Before add modalities input 0 {features[0]}")
    print(f"Before add modalities input 0 {features[1]}")
    print(f"Before add modalities features {len(features)}")

    if combine_omics:
        if len(features) > 1:
        #feature = concatenate(features)
            if model_type == 0:
                feature = Add()([features[0],features[1]])
            elif model_type == 1:
                cross_attention = CausalSelfAttention(num_heads=head_2, key_dim=256, dropout=drop_rate) # FIXME
                features[0] = tf.expand_dims(features[0], axis=1)
                features[1] = tf.expand_dims(features[1], axis=1)

                feature_attn = cross_attention(tf.concat([features[0], features[1]], 1))
                feature = tf.math.reduce_sum(feature_attn, axis=1)
        else:
            feature = features[0]
    
        dropout = Dropout(rate=drop_rate)(feature)
        output = Dense(head_1, name='projection-1', activation='relu')(dropout)
        
        return tf.keras.Model(inputs=inputs, outputs=output)
    
    else:
        dropout0 = Dropout(rate=drop_rate)(features[0])
        dropout1 = Dropout(rate=drop_rate)(features[1])

        output0 = Dense(head_1, name='projection-0', activation='relu')(dropout0)
        output1 = Dense(head_1, name='projection-1', activation='relu')(dropout1)

        return tf.keras.Model(inputs=inputs, outputs=[output0, output1])
    

def multi_embedding_attention_transfer_1(supvised_train: bool = False,
                                    scan_train: bool = False,
                                    multi_max_features: list = [40000],
                                    mult_feature_names: list = ['Gene'],
                                    embedding_dims=128,
                                    num_classes=5,
                                    activation='softmax',
                                    head_1=128,
                                    head_2=128,
                                    head_3=128,
                                    drop_rate=0.05,
                                    include_attention: bool = False,
                                    use_bias=True,
                                    ):
    assert len(multi_max_features) == len(mult_feature_names)

    # 特征索引
    x_feature_inputs = []
    # 特征值
    x_value_inputs = []
    # 特征向量
    embeddings = []
    features = []
    weight_output_all = []
    if include_attention == True:
        for max_length, name in zip(multi_max_features, mult_feature_names):
            # 输入
            feature_input = Input(shape=(None,), name='Input-{}-Feature'.format(name))
            value_input = Input(shape=(None,), name='Input-{}-Value'.format(name), dtype='float')
            x_feature_inputs.append(feature_input)
            x_value_inputs.append(value_input)

            # 向量
            embedding = Embedding(max_length, embedding_dims, input_length=None, name='{}-Embedding'.format(name))(
                feature_input)

            # 向量 * 特征值
            sparse_value = tf.expand_dims(value_input, 2, name='{}-Expend-Dims'.format(name))
            sparse_value = BatchNormalization(name='{}-BN-1'.format(name))(sparse_value)
            x = tf.multiply(embedding, sparse_value, name='{}-Multiply'.format(name))
            # x = BatchNormalization(name='{}-BN-2'.format(name))(x)

            # # Attention
            weight_output,a = AttentionWithContext()(x)
            x = K.tanh(K.sum(weight_output, axis=1))

            x = BatchNormalization(name='{}-BN-3'.format(name))(x)

            features.append(x)
            weight_output_all.append(a)
        inputs = []
        inputs.append(x_feature_inputs)
        inputs.append(x_value_inputs)

    else:
        for max_length, name in zip(multi_max_features, mult_feature_names):
            # 输入

            value_input = Input(shape=(max_length,), name='Input-{}-Value'.format(name), dtype='float')

            x_value_inputs.append(value_input)

            # 向量 * 特征值
            sparse_value = BatchNormalization(name='{}-BN-1'.format(name))(value_input)

            x = Dense(head_1, name='{}-projection-0'.format(name), activation='relu')(sparse_value)


            x = BatchNormalization(name='{}-BN-3'.format(name))(x)

            features.append(x)
        inputs = []
        inputs.append(x_value_inputs)
    # Concatenate
    if len(features) > 1:
    #feature = concatenate(features)
        feature = Add()([features[0],features[1]])
    else:
        feature = features[0]
    dropout = Dropout(rate=drop_rate)(feature)
    output = Dense(head_1, name='projection-1', activation='relu')(dropout)

    # inputs = []
    # inputs.append(x_feature_inputs)
    # inputs.append(x_value_inputs)
    return tf.keras.Model(inputs=inputs, outputs=[output,weight_output_all])

def multi_embedding_attention_transfer_explainability(supvised_train: bool = False,
                                    scan_train: bool = False,
                                    multi_max_features: list = [40000],
                                    mult_feature_names: list = ['Gene'],
                                    embedding_dims=128,
                                    num_classes=5,
                                    activation='softmax',
                                    head_1=128,
                                    head_2=128,
                                    head_3=128,
                                    drop_rate=0.05,
                                    include_attention: bool = False,
                                    use_bias=True,
                                    combine_omics: bool = True,
                                    model_type: int = 0,
                                    only_RNA: bool = False
                                    ):
    assert len(multi_max_features) == len(mult_feature_names)

    # 特征索引
    x_feature_inputs = []
    # 特征值
    x_value_inputs = []
    # 特征向量
    embeddings = []
    features = []
    weight_output_all = []
    if include_attention == True:
        for max_length, name in zip(multi_max_features, mult_feature_names):
            # 输入
            feature_input = Input(shape=(None,), name='Input-{}-Feature'.format(name))
            value_input = Input(shape=(None,), name='Input-{}-Value'.format(name), dtype='float')
            x_feature_inputs.append(feature_input)
            x_value_inputs.append(value_input)

            # 向量
            embedding = Embedding(max_length, embedding_dims, input_length=None, name='{}-Embedding'.format(name))(
                feature_input)

            # 向量 * 特征值
            sparse_value = tf.expand_dims(value_input, 2, name='{}-Expend-Dims'.format(name))
            sparse_value = BatchNormalization(name='{}-BN-1'.format(name))(sparse_value)
            x = tf.multiply(embedding, sparse_value, name='{}-Multiply'.format(name))
            # x = BatchNormalization(name='{}-BN-2'.format(name))(x)

            # # Attention
            weight_output,a = AttentionWithContext()(x)
            x = K.tanh(K.sum(weight_output, axis=1))

            x = BatchNormalization(name='{}-BN-3'.format(name))(x)

            features.append(x)
            weight_output_all.append(a)
        inputs = []
        inputs.append(x_feature_inputs)
        inputs.append(x_value_inputs)

    else:
        for max_length, name in zip(multi_max_features, mult_feature_names):
            # 输入

            value_input = Input(shape=(max_length,), name='Input-{}-Value'.format(name), dtype='float')

            x_value_inputs.append(value_input)

            # 向量 * 特征值
            sparse_value = BatchNormalization(name='{}-BN-1'.format(name))(value_input)

            x = Dense(head_1, name='{}-projection-0'.format(name), activation='relu')(sparse_value)


            x = BatchNormalization(name='{}-BN-3'.format(name))(x)

            features.append(x)
        inputs = []
        inputs.append(x_value_inputs)

    if combine_omics:
        # Concatenate
        if len(features) > 1:
        #feature = concatenate(features)
            if model_type == 0:
                feature = Add()([features[0],features[1]])
            elif model_type == 1:
                cross_attention = CausalSelfAttention(num_heads=head_2, key_dim=256, dropout=drop_rate) # FIXME
                features[0] = tf.expand_dims(features[0], axis=1)
                features[1] = tf.expand_dims(features[1], axis=1)

                feature_attn = cross_attention(tf.concat([features[0], features[1]], 1))
                feature = tf.math.reduce_sum(feature_attn, axis=1)
        else:
            feature = features[0]
            
        dropout = Dropout(rate=drop_rate)(feature)
        output = Dense(head_1, name='projection-1', activation='relu')(dropout)
        
        return tf.keras.Model(inputs=inputs, outputs=[output, weight_output_all])
    
    else:
        dropout0 = Dropout(rate=drop_rate)(features[0])
        dropout1 = Dropout(rate=drop_rate)(features[1])

        output0 = Dense(head_1, name='projection-0', activation='relu')(dropout0)
        output1 = Dense(head_1, name='projection-1', activation='relu')(dropout1)

        return tf.keras.Model(inputs=inputs, outputs=[output0, output1, weight_output_all])


def multi_embedding_attention_transfer_explainability_modalities(supvised_train: bool = False,
                                    scan_train: bool = False,
                                    multi_max_features: list = [40000],
                                    mult_feature_names: list = ['Gene'],
                                    embedding_dims=128,
                                    num_classes=5,
                                    activation='softmax',
                                    head_1=128,
                                    head_2=128,
                                    head_3=128,
                                    drop_rate=0.05,
                                    include_attention: bool = False,
                                    use_bias=True,
                                    combine_omics: bool = True,
                                    model_type: int = 0,
                                    only_RNA: bool = False
                                    ):
    assert len(multi_max_features) == len(mult_feature_names)

    # 特征索引
    x_feature_inputs = []
    # 特征值
    x_value_inputs = []
    # 特征向量
    embeddings = []
    features = []
    weight_output_all = []
    if include_attention == True:
        for max_length, name in zip(multi_max_features, mult_feature_names):
            # 输入
            feature_input = Input(shape=(None,), name='Input-{}-Feature'.format(name))
            value_input = Input(shape=(None,), name='Input-{}-Value'.format(name), dtype='float')
            x_feature_inputs.append(feature_input)
            x_value_inputs.append(value_input)

            # 向量
            embedding = Embedding(max_length, embedding_dims, input_length=None, name='{}-Embedding'.format(name))(
                feature_input)

            # 向量 * 特征值
            sparse_value = tf.expand_dims(value_input, 2, name='{}-Expend-Dims'.format(name))
            sparse_value = BatchNormalization(name='{}-BN-1'.format(name))(sparse_value)
            x = tf.multiply(embedding, sparse_value, name='{}-Multiply'.format(name))
            # x = BatchNormalization(name='{}-BN-2'.format(name))(x)

            # # Attention
            weight_output,a = AttentionWithContext()(x)
            x = K.tanh(K.sum(weight_output, axis=1))

            x = BatchNormalization(name='{}-BN-3'.format(name))(x)

            features.append(x)
            weight_output_all.append(a)
        inputs = []
        inputs.append(x_feature_inputs)
        inputs.append(x_value_inputs)

    else:
        for max_length, name in zip(multi_max_features, mult_feature_names):
            # 输入

            value_input = Input(shape=(max_length,), name='Input-{}-Value'.format(name), dtype='float')

            x_value_inputs.append(value_input)

            # 向量 * 特征值
            sparse_value = BatchNormalization(name='{}-BN-1'.format(name))(value_input)

            x = Dense(head_1, name='{}-projection-0'.format(name), activation='relu')(sparse_value)


            x = BatchNormalization(name='{}-BN-3'.format(name))(x)

            features.append(x)
        inputs = []
        inputs.append(x_value_inputs)

    if combine_omics:
        if model_type == 0:
            dropout0 = Dropout(rate=drop_rate)(features[0])
            dropout1 = Dropout(rate=drop_rate)(features[1])
            output0 = Dense(head_1, name='projection-0', activation='relu')(dropout0)
            output1 = Dense(head_1, name='projection-1', activation='relu')(dropout1)
        
            return tf.keras.Model(inputs=inputs, outputs=[output0, output1, weight_output_all])
        else:
            feature = features[0]
            dropout = Dropout(rate=drop_rate)(feature)
            output = Dense(head_1, name='projection-1', activation='relu')(dropout)
            
            return tf.keras.Model(inputs=inputs, outputs=[output, weight_output_all])
    
    else:
        dropout0 = Dropout(rate=drop_rate)(features[0])
        dropout1 = Dropout(rate=drop_rate)(features[1])

        output0 = Dense(head_1, name='projection-0', activation='relu')(dropout0)
        output1 = Dense(head_1, name='projection-1', activation='relu')(dropout1)

        return tf.keras.Model(inputs=inputs, outputs=[output0, output1, weight_output_all])


class EncoderHead(tf.keras.Model):

    def __init__(self, hidden_size=128, dropout=0.05):
        super(EncoderHead, self).__init__()
        # self.num_classes = num_classes
        self.feature_fc1 = tf.keras.layers.Dense(units=hidden_size, activation='relu')
        self.feature_fc2 = tf.keras.layers.Dense(units=hidden_size, activation='relu')
        self.feature_bn1 = tf.keras.layers.BatchNormalization()
        self.feature_bn2 = tf.keras.layers.BatchNormalization()
        self.feature_dropout1 = tf.keras.layers.Dropout(rate=dropout)
        self.feature_dropout2 = tf.keras.layers.Dropout(rate=dropout)

    def call(self, input):
        x = input
        feature_output = self.feature_fc1(x)
        feature_output = self.feature_bn1(feature_output)
        feature_output = self.feature_dropout1(feature_output)
        feature_output = self.feature_fc2(feature_output)

        return feature_output