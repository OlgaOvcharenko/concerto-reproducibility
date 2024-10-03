import datetime
import itertools
import tensorflow as tf
import numpy as np
import os
import scanpy as sc
import argparse
import pandas as pd
import copy
from collections import Counter
from scipy.sparse import issparse
import scipy
import sys
#sys.path.append("./Concerto-main/")
from bgi.utils.data_utils import *
from bgi.models.DeepSingleCell import *
from bgi.metrics.clustering_metrics import *
from bgi.losses.contrastive_loss import simclr_loss
import re
import h5py
import time

def set_seeds(seed=10):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)


def contrastive_loss(logits) :
    return tf.math.reduce_mean(
        tf.keras.metrics.sparse_categorical_crossentropy(
            y_true=tf.range(logits.shape[0]), y_pred=logits, from_logits=True
        )
    )


def clip_loss(text_embeds, image_embeds, logit_scale) :
    # normalized features
    image_embeds = image_embeds / tf.norm(tensor=image_embeds, ord="euclidean", axis=-1, keepdims=True)
    text_embeds = text_embeds / tf.norm(tensor=text_embeds, ord="euclidean", axis=-1, keepdims=True)

    # cosine similarity as logits
    logit_scale = tf.math.exp(logit_scale)
    logits_per_text = tf.matmul(text_embeds, image_embeds, transpose_b=True) * logit_scale
    similarity = logits_per_text

    caption_loss = contrastive_loss(similarity)
    image_loss = contrastive_loss(tf.transpose(similarity))
    return (caption_loss + image_loss) / 2.0

def create_single_cell_network(mult_feature_name: str, tf_path: str, super_parameters=None):
    if super_parameters is None:
        super_parameters = {'batch_size12': 32,
                            'batch_size13': 32,
                            'epoch_pretrain': 3,
                            'lr': 1e-4,
                            'drop_rate': 0.1, 
                            'attention_t': True, 
                            'attention_s': False, 
                            'heads': 128,
                            'combine_omics': False,
                            'model_type': 1} 
    
    f = np.load(os.path.join(tf_path, 'vocab_size.npz'))
    vocab_size = int(f['vocab size'])
    encode_network = single_embedding_attention_transfer(max_length=vocab_size,
                                                        name=mult_feature_name,
                                                        embedding_dims=128,
                                                        include_attention=super_parameters['attention_t'],
                                                        drop_rate=super_parameters['drop_rate'],
                                                        head_1=super_parameters["heads"],
                                                        head_2=super_parameters["heads"],
                                                        head_3=super_parameters["heads"],
                                                        combine_omics=super_parameters['combine_omics'],
                                                        model_type=super_parameters['model_type'])
    return encode_network


def cellbind_train_multimodal(mod1a_tf_path: str, mod2_tf_path: str, mod1b_tf_path: str, mod3_tf_path: str, weight_path: str, mod1_network, mod2_network, mod3_network, super_parameters=None):
    train_log_dir = 'logs_tensorboard/gradient_tape/' + f'{super_parameters["model_type"]}_multi_{super_parameters["data"]}_{super_parameters["batch_size12"]}_{super_parameters["batch_size13"]}_{super_parameters["epoch_pretrain"]}_{super_parameters["lr"]}_{super_parameters["drop_rate"]}_{super_parameters["attention_s"]}_{super_parameters["attention_t"]}_{super_parameters["heads"]}' + '/train'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)

    set_seeds(np.random.randint(0, 10))   
    if not os.path.exists(weight_path):
        os.makedirs(weight_path)
    if super_parameters is None:
        super_parameters = {'batch_size12': 32,
                            'batch_size13': 32, 
                            'epoch_pretrain': 3,
                            'lr': 1e-4,
                            'drop_rate': 0.1, 
                            'attention_t': True, 
                            'attention_s': False, 
                            'heads': 128,
                            'combine_omics': False,
                            'model_type': 1} 
    
    tf_list_1a = [f for f in os.listdir(os.path.join(mod1a_tf_path)) if 'tfrecord' in f]
    train_source_list_mod1a = []
    train_source_list_mod2 = []
    for i in tf_list_1a:
        train_source_list_mod1a.append(os.path.join(mod1a_tf_path, i))
        train_source_list_mod2.append(os.path.join(mod2_tf_path, i))

    tf_list_1b = [f for f in os.listdir(os.path.join(mod1b_tf_path)) if 'tfrecord' in f]
    train_source_list_mod1b = []
    train_source_list_mod3 = []
    for i in tf_list_1b:
        train_source_list_mod1b.append(os.path.join(mod1b_tf_path, i))
        train_source_list_mod3.append(os.path.join(mod3_tf_path, i))
    
    # Params
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    total_update_steps = 300 * super_parameters['epoch_pretrain']
    lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(super_parameters['lr'], total_update_steps, super_parameters['lr'] * 1e-2, power=1)
    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=lr_schedule)
    temperature = tf.Variable(np.log(1/0.07), trainable=True, dtype='float32')

    tf_step = 0
    for epoch in range(super_parameters['epoch_pretrain']):
        for mod1a_file, mod2_file, mod1b_file, mod3_file in itertools.zip_longest(train_source_list_mod1a, train_source_list_mod2, train_source_list_mod1b, train_source_list_mod3):
            # FIXME
            train_db_mod1a = create_classifier_dataset_multi([mod1a_file],
                                                           batch_size=super_parameters['batch_size12'],
                                                           is_training=True,
                                                           data_augment=False,
                                                           shuffle_size=10000,
                                                           seed=epoch
                                                           )
            train_db_mod2 = create_classifier_dataset_multi([mod2_file], 
                                                            batch_size=super_parameters['batch_size12'],
                                                            is_training=True,
                                                            data_augment=False,
                                                            shuffle_size=10000,
                                                            seed=epoch
                                                            )
            
            # If one ds has more tf records than another
            if mod1b_file is None and mod3_file is None:
                print("Files")
                f_i = np.random.randint(low=0, high=len(train_source_list_mod1b))
                mod1b_file_tmp = [train_source_list_mod1b[f_i]]
                mod3_file_tmp = [train_source_list_mod3[f_i]]
            else:
                mod1b_file_tmp = mod1b_file
                mod3_file_tmp = mod3_file

            train_db_mod1b = create_classifier_dataset_multi([mod1b_file_tmp],
                                                             batch_size=super_parameters['batch_size13'],
                                                             is_training=True,
                                                             data_augment=False,
                                                             shuffle_size=10000,
                                                             seed=epoch
                                                            )
            train_db_mod3 = create_classifier_dataset_multi([mod3_file_tmp], 
                                                            batch_size=super_parameters['batch_size13'],
                                                            is_training=True,
                                                            data_augment=False,
                                                            shuffle_size=10000,
                                                            seed=epoch
                                                            )

            train_loss.reset_states()

            step = 0
            it1a, it2, it1b, it3 = iter(train_db_mod1a), iter(train_db_mod2), iter(train_db_mod1b), iter(train_db_mod3)
            opit1a, opit2, opit1b, opit3 = it1a.get_next_as_optional(), it2.get_next_as_optional(), it1b.get_next_as_optional(), it3.get_next_as_optional()
            while opit1a.has_value() or opit2.has_value():
                step += 1

                # Oversample for bigger dataset (batches)
                if (not opit1b.has_value()) or (not opit3.has_value()):
                    print("Entered")
                    f_i = np.random.randint(low=0, high=len(train_source_list_mod1b))
                    train_db_mod1b = create_classifier_dataset_multi([train_source_list_mod1b[f_i]],
                                                           batch_size=super_parameters['batch_size13'],
                                                           is_training=True,
                                                           data_augment=False,
                                                           shuffle_size=10000,
                                                           seed=epoch
                                                           )
                    train_db_mod3 = create_classifier_dataset_multi([train_source_list_mod3[f_i]], 
                                                                    batch_size=super_parameters['batch_size13'],
                                                                    is_training=True,
                                                                    data_augment=False,
                                                                    shuffle_size=10000,
                                                                    seed=epoch
                                                                    )
                    it1b, it3 = iter(train_db_mod1b), iter(train_db_mod3)
                    opit1b, opit3 = it1b.get_next_as_optional(), it3.get_next_as_optional()
                    print(opit1b.has_value())
                    print(opit3.has_value())

                source_features_mod1a, source_values_mod1a, _, _ = opit1a.get_value()
                source_features_mod2, source_values_mod2, _, _ = opit2.get_value()
                source_features_mod1b, source_values_mod1b, _, _ = opit1b.get_value()
                source_features_mod3, source_values_mod3, _, _ = opit3.get_value()

                opit1a = it1a.get_next_as_optional()
                opit2 = it2.get_next_as_optional()
                opit1b = it1b.get_next_as_optional()
                opit3 = it3.get_next_as_optional()
                


    #             with tf.GradientTape() as tape:
    #                 if super_parameters["combine_omics"]:
    #                         z1 = mod1_network([[source_features_mod1, source_features_mod2],
    #                                         [source_values_mod1, source_values_mod2]], training=True)
    #                         z2 = mod2_network([[source_features_mod1, source_features_mod2],
    #                                         [source_values_mod1, source_values_mod2]], training=True)
    #                         ssl_loss = clip_loss(z1, z2, temperature=0.1)
    #                         loss = ssl_loss
                        
    #                 elif not super_parameters["combine_omics"]:
    #                     raise Exception("Not implemented")
    #                     # res_en = encode_network([[source_features_RNA, source_features_protein],
    #                     #                 [source_values_RNA, source_values_protein]], training=True)
    #                     # res_dec = decode_network([source_values_RNA, source_values_protein], training=True)
    #                     # zt_1, zt_2 = res_en[0], res_en[1]
    #                     # zs_1, zs_2 = res_dec[0], res_dec[1]

    #                     # # TT
    #                     # loss_TT = clip_loss(zt_1, zt_2, temperature)

    #                     # # SS
    #                     # loss_SS = clip_loss(zs_1, zs_2, temperature)

    #                     # # TS
    #                     # loss_TS = clip_loss(zt_1, zs_2, temperature)
                        
    #                     # # ST
    #                     # loss_ST = clip_loss(zt_2, zs_1, temperature)
    #                     # loss = loss_TT + loss_TS + loss_ST + loss_SS
                        
    #                 train_loss(loss)

    #             if super_parameters["combine_omics"]:
    #                 variables = [mod1_network.trainable_variables, mod2_network.trainable_variables, [temperature]]
    #             elif super_parameters["model_type"] in [2]:
    #                 pass
    #                 # variables = [encode_network.trainable_variables, [temperature]]

    #             grads = tape.gradient(loss, variables)
    #             for grad, var in zip(grads, variables):
    #                 optimizer.apply_gradients(zip(grad, var))

    #             if step > 0 and step % 100 == 0:
    #                 template = 'Epoch {}, step {}, simclr loss: {:0.4f}.'
    #                 print(template.format(epoch + 1, str(step), train_loss.result()))
                    
    #             # Tensorboard
    #             with train_summary_writer.as_default():
    #                 tf.summary.scalar('loss', train_loss.result(), step=tf_step)
    #             tf_step += 1

    #     encode_network.save_weights(
    #         weight_path + f'multi_weight_encoder_{super_parameters["data"]}_{super_parameters["batch_size"]}_model_{super_parameters["combine_omics"]}_{super_parameters["model_type"]}_epoch_{epoch+1}_{super_parameters["lr"]}_{super_parameters["drop_rate"]}_{super_parameters["attention_t"]}_{super_parameters["attention_s"]}_{super_parameters["heads"]}.h5')
    #     decode_network.save_weights(
    #         weight_path + f'multi_weight_decoder_{super_parameters["data"]}_{super_parameters["batch_size"]}_model_{super_parameters["combine_omics"]}_{super_parameters["model_type"]}_epoch_{epoch+1}_{super_parameters["lr"]}_{super_parameters["drop_rate"]}_{super_parameters["attention_t"]}_{super_parameters["attention_s"]}_{super_parameters["heads"]}.h5')

    # print(weight_path + f'multi_weight_encoder_{super_parameters["data"]}_{super_parameters["batch_size"]}_model_{super_parameters["combine_omics"]}_{super_parameters["model_type"]}_epoch_{super_parameters["epoch_pretrain"]}_{super_parameters["lr"]}_{super_parameters["drop_rate"]}_{super_parameters["attention_t"]}_{super_parameters["attention_s"]}_{super_parameters["heads"]}.h5')
    print('finished')
