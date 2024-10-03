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
        train_source_list_mod1b.append(os.path.join(mod1a_tf_path, i))
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
            train_db_mod1b = create_classifier_dataset_multi([mod1b_file],
                                                           batch_size=super_parameters['batch_size13'],
                                                           is_training=True,
                                                           data_augment=False,
                                                           shuffle_size=10000,
                                                           seed=epoch
                                                           )
            train_db_mod3 = create_classifier_dataset_multi([mod3_file], 
                                                            batch_size=super_parameters['batch_size13'],
                                                            is_training=True,
                                                            data_augment=False,
                                                            shuffle_size=10000,
                                                            seed=epoch
                                                            )
            
            # If one ds has more tf records than another
            if train_db_mod1b is None and train_db_mod3 is None:
                train_db_mod1b = create_classifier_dataset_multi([train_source_list_mod1b[np.random.randint(low=0, high=len(train_source_list_mod1b), size=1)]],
                                                           batch_size=super_parameters['batch_size13'],
                                                           is_training=True,
                                                           data_augment=False,
                                                           shuffle_size=10000,
                                                           seed=epoch
                                                           )
                train_db_mod3 = create_classifier_dataset_multi([train_source_list_mod3[np.random.randint(low=0, high=len(train_source_list_mod3), size=1)]], 
                                                                batch_size=super_parameters['batch_size13'],
                                                                is_training=True,
                                                                data_augment=False,
                                                                shuffle_size=10000,
                                                                seed=epoch
                                                                )
                print("Files")
                print(train_source_list_mod3[np.random.randint(low=0, high=len(train_source_list_mod3), size=1)])
                print(train_db_mod1b)
                print(train_db_mod3)

            train_loss.reset_states()
            
            step = 0
            it1a, it2, it1b, it3 = iter(train_db_mod1a), iter(train_db_mod2), iter(train_db_mod1b), iter(train_db_mod3)
            opit1a, opit2, opit1b, opit3 = it1a.get_next_as_optional(), it2.get_next_as_optional(), it1b.get_next_as_optional(), it3.get_next_as_optional()
            # for (source_features_mod1a, source_values_mod1a, _, _), \
            #     (source_features_mod2, source_values_mod2, _, _), \
            #     (source_features_mod1b, source_values_mod1b, _, _), \
            #     (source_features_mod3, source_values_mod3, _, _) \
            #         in (itertools.zip_longest(train_db_mod1a, train_db_mod2, train_db_mod1b, train_db_mod3)):
            #     step += 1

            while tf.get_static_value(opit1a.has_value()) or tf.get_static_value(opit2.has_value()):
                step += 1

                opit1a = it1a.get_next_as_optional()
                opit2 = it2.get_next_as_optional()
                opit1b = it1b.get_next_as_optional()
                opit3 = it3.get_next_as_optional()

                source_features_mod1a, source_values_mod1a, _, _ = opit1a.get_value()
                source_features_mod2, source_values_mod2, _, _ = opit2.get_value()
                source_features_mod1b, source_values_mod1b, _, _ = opit1b.get_value()
                source_features_mod3, source_values_mod3, _, _ = opit3.get_value()

                # Oversample for bigger dataset (batches)
                if not (tf.get_static_value(opit1b.has_value()) or tf.get_static_value(opit3.has_value())):
                    print(tf.get_static_value(opit1b.has_value()))
                    print(tf.get_static_value(opit3.has_value()))
                    train_db_mod1b = create_classifier_dataset_multi([train_source_list_mod1b[np.random.randint(low=0, high=len(train_source_list_mod1b), size=1)]],
                                                           batch_size=super_parameters['batch_size13'],
                                                           is_training=True,
                                                           data_augment=False,
                                                           shuffle_size=10000,
                                                           seed=epoch
                                                           )
                    train_db_mod3 = create_classifier_dataset_multi([train_source_list_mod3[np.random.randint(low=0, high=len(train_source_list_mod3), size=1)]], 
                                                                    batch_size=super_parameters['batch_size13'],
                                                                    is_training=True,
                                                                    data_augment=False,
                                                                    shuffle_size=10000,
                                                                    seed=epoch
                                                                    )
                    it1b, it3 = iter(train_db_mod1b), iter(train_db_mod3)
                    opit1b, opit3 = it1b.get_next_as_optional(), it3.get_next_as_optional()
                    print(tf.get_static_value(opit1b.has_value()))
                    print(tf.get_static_value(opit3.has_value()))

                # source_features_mod1a, source_values_mod1a = opit1a.get_value()

# iterator = iter(dataset)
# print(iterator.get_next())

#                 print("Batches")
                # print(source_features_mod1a)
                # print(source_features_mod2)
                # print(source_features_mod1b)
                # print(source_features_mod3)
                # exit()

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

def concerto_train_spatial_multimodal(mult_feature_names:list, RNA_tf_path: str, staining_tf_path: str, weight_path: str, super_parameters=None):
    train_log_dir = 'logs_tensorboard/gradient_tape/' + f'{super_parameters["model_type"]}_{super_parameters["mask"]}_multi_{super_parameters["data"]}_{super_parameters["batch_size"]}_{super_parameters["epoch_pretrain"]}_{super_parameters["lr"]}_{super_parameters["drop_rate"]}_{super_parameters["attention_s"]}_{super_parameters["attention_t"]}_{super_parameters["heads"]}' + '/train'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)

    set_seeds(np.random.randint(0, 10))   
    if not os.path.exists(weight_path):
        os.makedirs(weight_path)
    if super_parameters is None:
        super_parameters = {'batch_size': 32, 
                            'epoch_pretrain': 3,
                            'lr': 1e-4,
                            'drop_rate': 0.1, 
                            'attention_t': True, 
                            'attention_s': False, 
                            'heads': 128,
                            'combine_omics': False,
                            'model_type': 1} 
    
    f = int(np.load(os.path.join(RNA_tf_path, 'vocab_size.npz'))['vocab size'])
    vocab_size_RNA = int(f)

    f = np.load(os.path.join(staining_tf_path, 'vocab_size.npz'))
    vocab_size_staining = int(f['rows'])
    
    encode_network = make_spatial_RNA_image_model(multi_max_features=[vocab_size_RNA, vocab_size_staining],
                                                  mult_feature_names=mult_feature_names,
                                                  embedding_dims=128,
                                                  include_attention=True,
                                                  drop_rate=super_parameters['drop_rate'],
                                                  head_1=super_parameters["heads"],
                                                  head_2=super_parameters["heads"],
                                                  head_3=super_parameters["heads"],
                                                  model_type=super_parameters['model_type_image'])

    decode_network = make_spatial_RNA_image_model(multi_max_features=[vocab_size_RNA, vocab_size_staining],
                                                  mult_feature_names=mult_feature_names,
                                                  embedding_dims=128,
                                                  include_attention=False,
                                                  drop_rate=super_parameters['drop_rate'],
                                                  head_1=super_parameters["heads"],
                                                  head_2=super_parameters["heads"],
                                                  head_3=super_parameters["heads"],
                                                  model_type=super_parameters['model_type_image'])

    tf_list_1 = [f for f in os.listdir(os.path.join(RNA_tf_path)) if 'tfrecord' in f]
    train_source_list_RNA = []
    train_source_list_staining = []
    for i in tf_list_1:
        train_source_list_RNA.append(os.path.join(RNA_tf_path, i))
        train_source_list_staining.append(os.path.join(staining_tf_path, i))

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    total_update_steps = 300 * super_parameters['epoch_pretrain']
    lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(super_parameters['lr'], total_update_steps,
                                                                super_parameters['lr'] * 1e-2, power=1)
    
    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=lr_schedule)
    temperature = tf.Variable(np.log(1/0.07), trainable=True, dtype='float32')

    tf_step = 0
    for epoch in range(super_parameters['epoch_pretrain']):
        for RNA_file, staining_file in zip(train_source_list_RNA, train_source_list_staining):
            train_loss.reset_states()

            train_db_RNA = create_classifier_spatial_RNA_dataset_multi([RNA_file],
                                                           batch_size=super_parameters['batch_size'],
                                                           is_training=True,
                                                           data_augment=False,
                                                           shuffle_size=10000,
                                                           seed=epoch
                                                           )

            train_db_staining = create_classifier_dataset_spatial_multi([staining_file],
                                                               batch_size=super_parameters['batch_size'],
                                                               is_training=True,
                                                               shuffle_size=10000,
                                                               seed=epoch
                                                               )
            
            step = 0
            for (source_features_RNA, source_values_RNA, _, _, _), \
                (_, source_image_raw_staining, source_radius_staining) \
                    in (zip(train_db_RNA, train_db_staining)):
                step += 1


                # TODO Add preprocessing of mask
                batch_masks = np.zeros(source_image_raw_staining.shape, dtype=int)
                if super_parameters['mask'] == 1:
                    radius = source_radius_staining.numpy().reshape((super_parameters['batch_size'],))
                    for im, r in enumerate(radius):
                        arr = np.arange(-int(source_image_raw_staining.shape[1]/2), int(source_image_raw_staining.shape[2]/2)) ** 2
                        batch_masks[im,:,:, 0] = np.add.outer(arr, arr) < r ** 2
                        batch_masks[im,:,:, 1] = np.add.outer(arr, arr) < r ** 2
                        batch_masks[im,:,:, 2] = np.add.outer(arr, arr) < r ** 2
                        

                    batch_masks = tf.convert_to_tensor(batch_masks, dtype=tf.uint8)
                    source_image_raw_staining = tf.math.multiply(source_image_raw_staining, batch_masks)

                    # print(batch_masks)
                    # print(source_image_raw_staining)

                with tf.GradientTape() as tape:
                    if super_parameters["combine_omics"]:
                            raise Exception("combine_omics: can not combine omics for image, unlike in Concerto.")
                            
                    elif not super_parameters["combine_omics"]:
                        if super_parameters["model_type"] == 1:
                            res_en = encode_network([[source_features_RNA,],
                                            [source_values_RNA, source_image_raw_staining]], training=True)

                            zt_1, zt_2 = res_en[0], res_en[1]

                            loss = clip_loss(zt_1, zt_2, temperature)

                        elif super_parameters["model_type"] == 2:
                            res_dec = decode_network([source_values_RNA, source_image_raw_staining], training=True)
                            res_en = encode_network([[source_features_RNA,],
                                            [source_values_RNA, source_image_raw_staining]], training=True)
                            zt_1, zt_2 = res_en[0], res_en[1]
                            zs_1, zs_2 = res_dec[0], res_dec[1]

                            # TT
                            loss_TT = clip_loss(zt_1, zt_2, temperature)

                            # SS
                            loss_SS = clip_loss(zs_1, zs_2, temperature)

                            # TS
                            loss_TS = clip_loss(zt_1, zs_2, temperature)
                            
                            # ST
                            loss_ST = clip_loss(zt_2, zs_1, temperature)
                            
                            loss = loss_TT + loss_TS + loss_ST + loss_SS
                        
                    train_loss(loss)

                if super_parameters["combine_omics"]:
                    variables = [encode_network.trainable_variables, decode_network.trainable_variables]
                elif super_parameters["model_type"] in [1, 4]:
                    variables = [encode_network.trainable_variables, [temperature]]
                else:
                    variables = [encode_network.trainable_variables, decode_network.trainable_variables, [temperature]]

                grads = tape.gradient(loss, variables)
                for grad, var in zip(grads, variables):
                    optimizer.apply_gradients(zip(grad, var))

                if step > 0 and step % 5 == 0:
                    template = 'Epoch {}, step {}, simclr loss: {:0.4f}.'
                    print(template.format(epoch + 1,
                                          str(step),
                                          train_loss.result()))
                    
                # Tensorboard
                with train_summary_writer.as_default():
                    tf.summary.scalar('loss', train_loss.result(), step=tf_step)
                tf_step += 1

        encode_network.save_weights(
            weight_path + f'multi_weight_encoder_{super_parameters["data"]}_{super_parameters["mask"]}_{super_parameters["batch_size"]}_model_{super_parameters["combine_omics"]}_{super_parameters["model_type"]}_epoch_{epoch+1}_{super_parameters["lr"]}_{super_parameters["drop_rate"]}_{super_parameters["attention_t"]}_{super_parameters["attention_s"]}_{super_parameters["heads"]}.h5')
        decode_network.save_weights(
            weight_path + f'multi_weight_decoder_{super_parameters["data"]}_{super_parameters["mask"]}_{super_parameters["batch_size"]}_model_{super_parameters["combine_omics"]}_{super_parameters["model_type"]}_epoch_{epoch+1}_{super_parameters["lr"]}_{super_parameters["drop_rate"]}_{super_parameters["attention_t"]}_{super_parameters["attention_s"]}_{super_parameters["heads"]}.h5')

    print(weight_path + f'multi_weight_encoder_{super_parameters["data"]}_{super_parameters["mask"]}_{super_parameters["batch_size"]}_model_{super_parameters["combine_omics"]}_{super_parameters["model_type"]}_epoch_{super_parameters["epoch_pretrain"]}_{super_parameters["lr"]}_{super_parameters["drop_rate"]}_{super_parameters["attention_t"]}_{super_parameters["attention_s"]}_{super_parameters["heads"]}.h5')
    return print('finished')

def concerto_test_spatial_multimodal(mult_feature_names, model_path: str, 
                                     RNA_tf_path: str, staining_tf_path: str, 
                                     n_cells_for_sample=None,super_parameters=None,
                                     saved_weight_path=None, only_image=False):
    if super_parameters is None:
        super_parameters = {'batch_size': 32, 'epoch': 1, 'lr': 1e-5, 'drop_rate': 0.1, 'combine_omics': False}
    
    f = int(np.load(os.path.join(RNA_tf_path, 'vocab_size.npz'))['vocab size'])
    vocab_size_RNA = int(f)

    f = np.load(os.path.join(staining_tf_path, 'vocab_size.npz'))
    vocab_size_staining = int(f['rows'])

    batch_size = super_parameters['batch_size']
    
    encode_network = make_spatial_RNA_image_model(multi_max_features=[vocab_size_RNA, vocab_size_staining],
                                                  mult_feature_names=mult_feature_names,
                                                  embedding_dims=128,
                                                  include_attention=super_parameters['attention_t'],
                                                  drop_rate=super_parameters['drop_rate'],
                                                  head_1=super_parameters["heads"],
                                                  head_2=super_parameters["heads"],
                                                  head_3=super_parameters["heads"],
                                                  model_type=super_parameters['model_type_image'])

    tf_list_1 = [f for f in os.listdir(os.path.join(RNA_tf_path)) if 'tfrecord' in f]
    train_source_list_RNA = []
    train_source_list_staining = []
    for i in tf_list_1:
        train_source_list_RNA.append(os.path.join(RNA_tf_path, i))
        train_source_list_staining.append(os.path.join(staining_tf_path, i))

    if saved_weight_path is None:
        weight_id_list = []
        weight_list = [f for f in os.listdir(model_path) if f.endswith('h5')]
        for id in weight_list:
            id_1 = re.findall('.*epoch(.*).h.*', id)  # f1
            weight_id_list.append(int(id_1[0]))
        encode_network.load_weights(model_path + 'weight_encoder_epoch{}.h5'.format(max(weight_id_list)),
                                    by_name=True)

    else:
        encode_network.load_weights(saved_weight_path, by_name=True)
        print('load saved weight')

    source_data_batch = []
    source_data_feature = []
    source_data_feature_2 = []
    RNA_id_all = []
    for RNA_file, staining_file in zip(train_source_list_RNA, train_source_list_staining):
        train_db_RNA = create_classifier_spatial_RNA_dataset_multi([RNA_file],
                                                           batch_size=super_parameters['batch_size'],
                                                           is_training=False,
                                                           data_augment=False,
                                                           shuffle_size=10000
                                                           )

        train_db_staining = create_classifier_dataset_spatial_multi([staining_file],
                                                            batch_size=super_parameters['batch_size'],
                                                            is_training=False,
                                                            shuffle_size=10000
                                                            )



        step = 0
        for (source_features_RNA, source_values_RNA, _, _, _), \
                (_, source_image_raw_staining, source_radius_staining) \
                    in (zip(train_db_RNA, train_db_staining)):
            if step == 0:
                if super_parameters["combine_omics"]:
                    # TODO
                    raise Exception("Not implemented")
                    # encode_output, attention_output = encode_network([[source_features_RNA,],
                    #                         [source_values_RNA, source_image_raw_staining]], training=False)
                    
                    # break

                else:
                    encode_output1, encode_output2 = encode_network([[source_features_RNA,],
                                            [source_values_RNA, source_image_raw_staining]], training=False)

                    if only_image:
                        encode_output = encode_output1
                    else:
                        encode_output = tf.concat([encode_output1, encode_output2], axis=1)
                    
                    break

        dim = encode_output.shape[1]
        if n_cells_for_sample is None:            
            feature_len = 1000000
        else:            
            n_cells_for_sample_1 = n_cells_for_sample//8
            feature_len = n_cells_for_sample_1 // batch_size * batch_size
        
        source_data_feature_1 = np.zeros((feature_len, dim))
        source_data_batch_1 = np.zeros((feature_len))

        source_data_feature_1_2 = np.zeros((feature_len, dim))

        RNA_id = []
        all_samples = 0
        for (source_features_RNA, source_values_RNA, source_batch_RNA, source_id_RNA, _), \
                (_, source_image_raw_staining, source_radius_staining) \
                    in (zip(train_db_RNA, train_db_staining)):
            if all_samples  >= feature_len:
                print("Entered if break")
                break

            batch_masks = np.zeros(source_image_raw_staining.shape, dtype=int)
            if super_parameters['mask'] == 1:
                radius = source_radius_staining.numpy().reshape((super_parameters['batch_size'],))
                for im, r in enumerate(radius):
                    arr = np.arange(-int(source_image_raw_staining.shape[1]/2), int(source_image_raw_staining.shape[2]/2)) ** 2
                    batch_masks[im,:,:, 0] = np.add.outer(arr, arr) < r ** 2
                    batch_masks[im,:,:, 1] = np.add.outer(arr, arr) < r ** 2
                    batch_masks[im,:,:, 2] = np.add.outer(arr, arr) < r ** 2
                    

                batch_masks = tf.convert_to_tensor(batch_masks, dtype=tf.uint8)
                source_image_raw_staining = tf.math.multiply(source_image_raw_staining, batch_masks)


            if super_parameters["combine_omics"]:
                raise Exception("Not implemented")
                # if only_image:
                #     encode_output, attention_output = encode_network([[source_features_RNA],
                #                                                 [source_values_RNA]],
                #                                                 training=False)
                # else:
                #     encode_output, attention_output = encode_network([[source_features_RNA, source_features_protein],
                #                                                     [source_values_RNA, source_values_protein]],
                #                                                     training=False)

            else:
                encode_output1, encode_output2 = encode_network([[source_features_RNA,],
                                            [source_values_RNA, source_image_raw_staining]], training=False)

                if only_image:
                    encode_output = encode_output1
                else:
                    encode_output = tf.concat([encode_output1, encode_output2], axis=1)

            encode_output = tf.nn.l2_normalize(encode_output, axis=-1)
            source_data_feature_1[all_samples:all_samples + len(source_id_RNA), :] = encode_output
            source_data_batch_1[all_samples:all_samples + len(source_id_RNA)] = source_batch_RNA

            if only_image:
                encode_output2 = tf.nn.l2_normalize(encode_output2, axis=-1)
                source_data_feature_1_2[all_samples:all_samples + len(source_id_RNA), :] = encode_output2

            RNA_id.extend(list(source_id_RNA.numpy().astype('U')))
            all_samples += len(source_id_RNA)

        source_data_feature.extend(source_data_feature_1[:all_samples])
        source_data_batch.extend(source_data_batch_1[:all_samples])
        if only_image:
            source_data_feature_2.extend(source_data_feature_1_2[:all_samples])
        RNA_id_all.extend(RNA_id[:all_samples])

    source_data_feature = np.array(source_data_feature).astype('float32')
    if only_image:
        source_data_feature_2 = np.array(source_data_feature_2).astype('float32')
    source_data_batch = np.array(source_data_batch).astype('int32')
    if only_image:
        return source_data_feature, source_data_feature_2, source_data_batch, RNA_id_all
    else:    
        return source_data_feature, source_data_batch, RNA_id_all


def concerto_train_multimodal_tt(mult_feature_names:list, RNA_tf_path: str, Protein_tf_path: str, weight_path: str, super_parameters=None):
    train_log_dir = 'logs_tensorboard/gradient_tape/' + f'multi_{super_parameters["data"]}_{super_parameters["epoch_pretrain"]}_{super_parameters["lr"]}_{super_parameters["drop_rate"]}_{super_parameters["attention_s"]}_{super_parameters["attention_t"]}_{super_parameters["heads"]}' + '/train'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)

    set_seeds(np.random.randint(0, 10))   
    if not os.path.exists(weight_path):
        os.makedirs(weight_path)
    if super_parameters is None:
        super_parameters = {'batch_size': 31, 
                            'epoch_pretrain': 3,
                            'lr': 1e-4,
                            'drop_rate': 0.1, 
                            'attention_t': True, 
                            'attention_s': False, 
                            'heads': 128} 
    # dirname = os.getcwd()
    # f = np.load(ref_tf_path + './vocab_size.npz')
    f = np.load(os.path.join(RNA_tf_path, 'vocab_size.npz'))
    vocab_size_RNA = int(f['vocab size'])
    f = np.load(os.path.join(Protein_tf_path, 'vocab_size.npz'))
    vocab_size_Protein = int(f['vocab size'])
    encode_network = multi_embedding_attention_transfer(multi_max_features=[vocab_size_RNA,vocab_size_Protein],
                                                        mult_feature_names=mult_feature_names,
                                                        embedding_dims=128,
                                                        include_attention=super_parameters['attention_t'],
                                                        drop_rate=super_parameters['drop_rate'],
                                                        head_1=super_parameters["heads"],
                                                        head_2=super_parameters["heads"],
                                                        head_3=super_parameters["heads"])

    encode_network2 = multi_embedding_attention_transfer(multi_max_features=[vocab_size_RNA,vocab_size_Protein],
                                                        mult_feature_names=mult_feature_names,
                                                        embedding_dims=128,
                                                        include_attention=super_parameters['attention_t'],
                                                        drop_rate=super_parameters['drop_rate'],
                                                        head_1=super_parameters["heads"],
                                                        head_2=super_parameters["heads"],
                                                        head_3=super_parameters["heads"])

    # tf_list_1 = os.listdir(os.path.join(ref_tf_path))
    tf_list_1 = [f for f in os.listdir(os.path.join(RNA_tf_path)) if 'tfrecord' in f]
    train_source_list_RNA = []
    train_source_list_Protein = []
    for i in tf_list_1:
        train_source_list_RNA.append(os.path.join(RNA_tf_path, i))
        train_source_list_Protein.append(os.path.join(Protein_tf_path, i))

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_cls_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_cls_accuracy')
    test_cls_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_cls_accuracy')
    total_update_steps = 300 * super_parameters['epoch_pretrain']
    lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(super_parameters['lr'], total_update_steps,
                                                                super_parameters['lr'] * 1e-2, power=1)
    opt_simclr = tf.keras.optimizers.legacy.Adam(learning_rate=lr_schedule)
    
    tf_step = 0
    for epoch in range(super_parameters['epoch_pretrain']):
        for RNA_file, Protein_file in zip(train_source_list_RNA, train_source_list_Protein):
            train_db_RNA = create_classifier_dataset_multi([RNA_file],
                                                           batch_size=super_parameters['batch_size'],
                                                           is_training=False,
                                                           data_augment=False,
                                                           shuffle_size=10000,
                                                           )
            train_db_Protein = create_classifier_dataset_multi([Protein_file],
                                                               batch_size=super_parameters['batch_size'],
                                                               is_training=False,
                                                               data_augment=False,
                                                               shuffle_size=10000,
                                                               )
            train_loss.reset_states()
            train_cls_accuracy.reset_states()
            test_cls_accuracy.reset_states()
            step = 0
            for (source_features_RNA, source_values_RNA,
                 source_batch_RNA, source_id_RNA), \
                (source_features_protein, source_values_protein,
                 source_batch_Protein, source_id_Protein) \
                    in (zip(train_db_RNA, train_db_Protein)):
                step += 1

                with tf.GradientTape() as tape:
                    z1 = encode_network([[source_features_RNA, source_features_protein],
                                         [source_values_RNA, source_values_protein]], training=True)
                    z2 = encode_network2([[source_features_RNA, source_features_protein],
                                         [source_values_RNA, source_values_protein]], training=True)
                    ssl_loss = simclr_loss(z1, z2, temperature=0.1)
                    loss = ssl_loss
                    train_loss(loss)

                variables = [encode_network.trainable_variables,
                             encode_network2.trainable_variables,
                             ]
                grads = tape.gradient(loss, variables)
                for grad, var in zip(grads, variables):
                    opt_simclr.apply_gradients(zip(grad, var))

                if step > 0 and step % 5 == 0:
                    template = 'Epoch {}, step {}, simclr loss: {:0.4f}.'
                    print(template.format(epoch + 1,
                                          str(step),
                                          train_loss.result()))
                    
                # Tensorboard
                with train_summary_writer.as_default():
                    tf.summary.scalar('loss', train_loss.result(), step=tf_step)
                tf_step += 1

        encode_network.save_weights(
            weight_path + f'multi_weight_encoder_{super_parameters["data"]}_epoch_{epoch+1}_{super_parameters["lr"]}_{super_parameters["drop_rate"]}_{super_parameters["attention_t"]}_{super_parameters["attention_s"]}_{super_parameters["heads"]}.h5')
        encode_network2.save_weights(
            weight_path + f'multi_weight_decoder_{super_parameters["data"]}_epoch_{epoch+1}_{super_parameters["lr"]}_{super_parameters["drop_rate"]}_{super_parameters["attention_t"]}_{super_parameters["attention_s"]}_{super_parameters["heads"]}.h5')

    return print('finished')

def concerto_train_multimodal_ss(mult_feature_names:list, RNA_tf_path: str, Protein_tf_path: str, weight_path: str, super_parameters=None):
    train_log_dir = 'logs_tensorboard/gradient_tape/' + f'multi_{super_parameters["data"]}_{super_parameters["epoch_pretrain"]}_{super_parameters["lr"]}_{super_parameters["drop_rate"]}_{super_parameters["attention_s"]}_{super_parameters["attention_t"]}_{super_parameters["heads"]}' + '/train'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)

    set_seeds(np.random.randint(0, 10))   
    if not os.path.exists(weight_path):
        os.makedirs(weight_path)
    if super_parameters is None:
        super_parameters = {'batch_size': 31, 
                            'epoch_pretrain': 3,
                            'lr': 1e-4,
                            'drop_rate': 0.1, 
                            'attention_t': True, 
                            'attention_s': False, 
                            'heads': 128} 
    # dirname = os.getcwd()
    # f = np.load(ref_tf_path + './vocab_size.npz')
    f = np.load(os.path.join(RNA_tf_path, 'vocab_size.npz'))
    vocab_size_RNA = int(f['vocab size'])
    f = np.load(os.path.join(Protein_tf_path, 'vocab_size.npz'))
    vocab_size_Protein = int(f['vocab size'])
    decode_network = multi_embedding_attention_transfer(multi_max_features=[vocab_size_RNA,vocab_size_Protein],
                                                        mult_feature_names=mult_feature_names,
                                                        embedding_dims=128,
                                                        include_attention=super_parameters['attention_s'],
                                                        drop_rate=super_parameters['drop_rate'],
                                                        head_1=super_parameters["heads"],
                                                        head_2=super_parameters["heads"],
                                                        head_3=super_parameters["heads"])

    decode_network2 = multi_embedding_attention_transfer(multi_max_features=[vocab_size_RNA,vocab_size_Protein],
                                                        mult_feature_names=mult_feature_names,
                                                        embedding_dims=128,
                                                        include_attention=super_parameters['attention_s'],
                                                        drop_rate=super_parameters['drop_rate'],
                                                        head_1=super_parameters["heads"],
                                                        head_2=super_parameters["heads"],
                                                        head_3=super_parameters["heads"])

    # tf_list_1 = os.listdir(os.path.join(ref_tf_path))
    tf_list_1 = [f for f in os.listdir(os.path.join(RNA_tf_path)) if 'tfrecord' in f]
    train_source_list_RNA = []
    train_source_list_Protein = []
    for i in tf_list_1:
        train_source_list_RNA.append(os.path.join(RNA_tf_path, i))
        train_source_list_Protein.append(os.path.join(Protein_tf_path, i))

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_cls_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_cls_accuracy')
    test_cls_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_cls_accuracy')
    total_update_steps = 300 * super_parameters['epoch_pretrain']
    lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(super_parameters['lr'], total_update_steps,
                                                                super_parameters['lr'] * 1e-2, power=1)
    opt_simclr = tf.keras.optimizers.legacy.Adam(learning_rate=lr_schedule)
    
    tf_step = 0
    for epoch in range(super_parameters['epoch_pretrain']):
        for RNA_file, Protein_file in zip(train_source_list_RNA, train_source_list_Protein):
            train_db_RNA = create_classifier_dataset_multi([RNA_file],
                                                           batch_size=super_parameters['batch_size'],
                                                           is_training=False,
                                                           data_augment=False,
                                                           shuffle_size=10000,
                                                           )
            train_db_Protein = create_classifier_dataset_multi([Protein_file],
                                                               batch_size=super_parameters['batch_size'],
                                                               is_training=False,
                                                               data_augment=False,
                                                               shuffle_size=10000,
                                                               )
            train_loss.reset_states()
            train_cls_accuracy.reset_states()
            test_cls_accuracy.reset_states()
            step = 0
            for (source_features_RNA, source_values_RNA,
                 source_batch_RNA, source_id_RNA), \
                (source_features_protein, source_values_protein,
                 source_batch_Protein, source_id_Protein) \
                    in (zip(train_db_RNA, train_db_Protein)):
                step += 1

                with tf.GradientTape() as tape:
                    z1 = decode_network([source_values_RNA, source_values_protein], training=True)
                    z2 = decode_network2([source_values_RNA, source_values_protein], training=True)
                    ssl_loss = simclr_loss(z1, z2, temperature=0.1)
                    loss = ssl_loss
                    train_loss(loss)

                variables = [decode_network.trainable_variables,
                             decode_network2.trainable_variables,
                             ]
                grads = tape.gradient(loss, variables)
                for grad, var in zip(grads, variables):
                    opt_simclr.apply_gradients(zip(grad, var))

                if step > 0 and step % 5 == 0:
                    template = 'Epoch {}, step {}, simclr loss: {:0.4f}.'
                    print(template.format(epoch + 1,
                                          str(step),
                                          train_loss.result()))
                    
                # Tensorboard
                with train_summary_writer.as_default():
                    tf.summary.scalar('loss', train_loss.result(), step=tf_step)
                tf_step += 1

        decode_network.save_weights(
            weight_path + f'multi_weight_encoder_{super_parameters["data"]}_epoch_{epoch+1}_{super_parameters["lr"]}_{super_parameters["drop_rate"]}_{super_parameters["attention_t"]}_{super_parameters["attention_s"]}_{super_parameters["heads"]}.h5')
        decode_network2.save_weights(
            weight_path + f'multi_weight_decoder_{super_parameters["data"]}_epoch_{epoch+1}_{super_parameters["lr"]}_{super_parameters["drop_rate"]}_{super_parameters["attention_t"]}_{super_parameters["attention_s"]}_{super_parameters["heads"]}.h5')

    return print('finished')


def concerto_test_1set_attention(model_path:str, ref_tf_path:str, super_parameters=None, n_cells_for_ref=5000):
    set_seeds(np.random.randint(0, 10))
    
    if super_parameters is None:
            super_parameters = {'batch_size': 32, 'epoch': 1, 'lr': 1e-5}
    
    f = np.load(ref_tf_path + 'vocab_size.npz')
    vocab_size = int(f['vocab size'])
    batch_size = super_parameters['batch_size']
    encode_network = multi_embedding_attention_transfer_1(
        multi_max_features=[vocab_size],
        mult_feature_names=['RNA'],
        embedding_dims=128,
        include_attention=True,
        drop_rate=0.1,
        head_1=128,
        head_2=128,
        head_3=128)
    tf_list_1 = [f for f in os.listdir(os.path.join(ref_tf_path)) if 'tfrecord' in f]
    train_source_list = [os.path.join(ref_tf_path, i) for i in tf_list_1]
    # choose last epoch as test model
    weight_id_list = []
#     weight_list = [f for f in os.listdir(model_path) if (f.endswith('h5') and f.startswith('weight') )]
    weight_list = [f for f in os.listdir(model_path) if (f.endswith('h5') and ('encoder' in f) )] # yyyx 1214
    for id in weight_list:
        id_1 = re.findall('.*epoch(.*).h5', id)  # f1
        weight_id_list.append(int(id_1[0]))
    weight_name_ = sorted(list(zip(weight_id_list,weight_list)),key=lambda x:x[0])[-1][1]
    encode_network.load_weights(model_path + weight_name_, by_name=True)
    
    t1 = time.time()
    ref_db = create_classifier_dataset_multi(
        train_source_list,
        batch_size=batch_size, # maybe slow
        is_training=True,
        data_augment=False,
        shuffle_size=10000)
    for step, (target_features, target_values, target_batch, target_id) in enumerate(ref_db):
        output,_ = encode_network([target_features, target_values], training=False)
        break
    t2 = time.time()
    print('load all tf in memory time(s)',t2-t1) # time consumption is huge this step!!!!

    feature_len = n_cells_for_ref//batch_size*batch_size
    print(feature_len, batch_size)
    t2 = time.time()
    dim = output.shape[1]
    source_data_feature_1 = np.zeros((feature_len, dim))
    source_data_batch_1 = np.zeros((feature_len))
    attention_weight = np.zeros((feature_len, vocab_size,1))
    source_id_batch_1 = []
    for step, (target_features, target_values, target_batch, target_id) in enumerate(ref_db):
        if step*batch_size >= feature_len:
            break
        output,attention_output = encode_network([target_features, target_values], training=False)
        output = tf.nn.l2_normalize(output, axis=-1)
        source_data_feature_1[step * batch_size:(step+1) * batch_size, :] = output
        source_data_batch_1[step * batch_size:(step+1) * batch_size] = target_batch
        attention_weight[step * batch_size:(step+1) * batch_size, :,:] = attention_output[-1]
        source_id_batch_1.extend(list(target_id.numpy().astype('U')))

    t3 = time.time()
    print('test time',t3-t2)
    print('source_id_batch_1 len', len(source_id_batch_1))
#     source_id_batch_1 = [i.decode("utf-8") for i in source_id_batch_1]
    return source_data_feature_1, list(source_id_batch_1),attention_weight



def concerto_test_new(model_path:str, ref_tf_path:str, query_tf_path:str, super_parameters=None, n_cells_for_ref=5000):
    if super_parameters is None:
        super_parameters = {'batch_size': 32, 'epoch': 1, 'lr': 1e-5}

    f = np.load(ref_tf_path + 'vocab_size.npz')
    vocab_size = int(f['vocab size'])
    encode_network = multi_embedding_attention_transfer(
        multi_max_features=[vocab_size],
        mult_feature_names=['RNA'],
        embedding_dims=128,
        include_attention=True,
        drop_rate=0.1,
        head_1=128,
        head_2=128,
        head_3=128)

    tf_list_1 = [f for f in os.listdir(os.path.join(ref_tf_path)) if 'tfrecord' in f]
    train_source_list = []
    for i in tf_list_1:
        train_source_list.append(os.path.join(ref_tf_path, i))
    tf_list_2 = [f for f in os.listdir(os.path.join(query_tf_path)) if 'tfrecord' in f]
    train_target_list = []
    for i in tf_list_2:
        train_target_list.append(os.path.join(query_tf_path, i))
    weight_id_list = []
    weight_list = [f for f in os.listdir(model_path) if f.endswith('h5')]

    for id in weight_list:
        id_1 = re.findall('.*epoch(.*).h.*', id)  # f1
        weight_id_list.append(int(id_1[0]))
    encode_network.load_weights(model_path + 'weight_encoder_epoch{}.h5'.format(max(weight_id_list)),by_name=True)
    source_data_batch = []
    source_data_feature = []
    source_data_id = []
    batch_size = super_parameters['batch_size']
    for file in train_source_list:
        print(file)
        feature_len = 0
        ref_db = create_classifier_dataset_multi(
            [file],
            batch_size=1,
            is_training=True,
            data_augment=False,
            shuffle_size=10000)
        for step, (target_features, target_values, target_batch, target_id) in enumerate(ref_db):
            feature_len += 1
            if step == 0:
                output = encode_network([target_features, target_values], training=False)

        dim = output.shape[1]
        source_data_feature_1 = np.zeros((feature_len, dim))
        source_data_batch_1 = np.zeros((feature_len))
        source_id_batch_1 = []
        for step, (target_features, target_values, target_batch, target_id) in enumerate(ref_db):
            output = encode_network([target_features, target_values], training=False)
            output = tf.nn.l2_normalize(output, axis=-1)
            source_data_feature_1[step, :] = output
            source_data_batch_1[step] = target_batch
            source_id_batch_1.append(target_id.numpy()[0])
        source_data_feature.extend(source_data_feature_1)
        source_data_batch.extend(source_data_batch_1)
        source_data_id.extend(source_id_batch_1)

    target_data_batch = []
    target_data_feature = []
    target_data_id = []
    for file in train_target_list:
        print(file)
        feature_len = 0
        query_db = create_classifier_dataset_multi([file],
                                                   batch_size=1,
                                                   is_training=False,
                                                   data_augment=False,
                                                   shuffle_size=10000)
        for step, (target_features, target_values, target_batch, target_id) in enumerate(query_db):
            feature_len += 1
            if step == 0:
                output = encode_network([target_features, target_values], training=False)

        dim = output.shape[1]
        target_data_feature_1 = np.zeros((feature_len, dim))
        target_data_batch_1 = np.zeros((feature_len))
        target_id_batch_1 = []
        for step, (target_features, target_values, target_batch, target_id) in enumerate(query_db):
            output = encode_network([target_features, target_values], training=False)
            output = tf.nn.l2_normalize(output, axis=-1)
            target_data_feature_1[step, :] = output
            target_data_batch_1[step] = target_batch
            target_id_batch_1.append(target_id.numpy()[0])

        target_data_feature.extend(target_data_feature_1)
        target_data_batch.extend(target_data_batch_1)
        target_data_id.extend(target_id_batch_1)

    ref_embedding = np.array(source_data_feature[:n_cells_for_ref])
    query_embedding = np.array(target_data_feature)
    source_data_id_subsample = source_data_id[:n_cells_for_ref]
    source_data_id_subsample = [i.decode("utf-8") for i in source_data_id_subsample]
    target_data_id = [i.decode("utf-8") for i in target_data_id]
    print('query embedding shape', query_embedding.shape)
    print('reference embedding shape', ref_embedding.shape)
    return ref_embedding, query_embedding,source_data_id_subsample,target_data_id # N*dim, 顺序按照adata1， adata2的cell 顺序；

def concerto_test_attention_0117(model_path:str, ref_tf_path:str, query_tf_path:str, super_parameters=None):
    if super_parameters is None:
        super_parameters = {'batch_size': 32, 'epoch': 1, 'lr': 1e-5,'drop_rate': 0.1}

    f = np.load(ref_tf_path + 'vocab_size.npz')
    vocab_size = int(f['vocab size'])
    encode_network = multi_embedding_attention_transfer(
        multi_max_features=[vocab_size],
        mult_feature_names=['RNA'],
        embedding_dims=128,
        include_attention=True,
        drop_rate=super_parameters['drop_rate'],
        head_1=128,
        head_2=128,
        head_3=128)

    tf_list_1 = [f for f in os.listdir(os.path.join(ref_tf_path)) if 'tfrecord' in f]
    train_source_list = []
    for i in tf_list_1:
        train_source_list.append(os.path.join(ref_tf_path, i))
    tf_list_2 = [f for f in os.listdir(os.path.join(query_tf_path)) if 'tfrecord' in f]
    train_target_list = []
    for i in tf_list_2:
        train_target_list.append(os.path.join(query_tf_path, i))
    weight_id_list = []
    weight_list = [f for f in os.listdir(model_path) if f.endswith('h5')]

    for id in weight_list:
        id_1 = re.findall('.*epoch(.*).h.*', id)  # f1
        weight_id_list.append(int(id_1[0]))
    # encode_network.load_weights(model_path + 'weight_encoder_epoch{}.h5'.format(max(weight_id_list)))
    encode_network.load_weights(model_path + 'weight_encoder_epoch{}.h5'.format(max(weight_id_list)), by_name=True) # yyyx0126, 支持多模态模型
    
    source_data_batch = []
    source_data_feature = []
    source_data_id = []
    batch_size = super_parameters['batch_size']
    for file in train_source_list:
        print(file)
        train_size = 0
        ref_db = create_classifier_dataset_multi(
            [file],
            batch_size=batch_size,
            is_training=False,
            data_augment=False,
            shuffle_size=10000)
        for step, (target_features, target_values, target_batch, target_id) in enumerate(ref_db):
            train_size += len(target_id)
            if step == 0:
                output = encode_network([target_features, target_values], training=False)

        dim = output.shape[1]
        source_data_feature_1 = np.zeros((train_size, dim))
        source_data_batch_1 = np.zeros((train_size))
        #source_id_batch_1 = np.zeros((train_size))
        source_id_batch_1 = []
        all_samples = 0
        for step, (target_features, target_values, target_batch, target_id) in enumerate(ref_db):
            output = encode_network([target_features, target_values], training=False)
            output = tf.nn.l2_normalize(output, axis=-1)
            source_data_feature_1[all_samples:all_samples + len(target_id), :] = output
            source_data_batch_1[all_samples:all_samples + len(target_id)] = target_batch
            #source_id_batch_1[all_samples:all_samples + len(target_id)] = target_id.numpy().decode("utf-8")
            source_id_batch_1.extend(list(target_id.numpy().astype('U')))

            all_samples += len(target_id)
        source_data_feature.extend(source_data_feature_1)
        source_data_batch.extend(source_data_batch_1)
        source_data_id.extend(source_id_batch_1)

    target_data_batch = []
    target_data_feature = []
    target_data_id = []
    for file in train_target_list:
        print(file)
        train_size = 0
        query_db = create_classifier_dataset_multi([file],
                                                   batch_size=batch_size,
                                                   is_training=False,
                                                   data_augment=False,
                                                   shuffle_size=10000)
        for step, (target_features, target_values, target_batch, target_id) in enumerate(query_db):
            train_size += len(target_id)
            if step == 0:
                output = encode_network([target_features, target_values], training=False)

        dim = output.shape[1]
        target_data_feature_1 = np.zeros((train_size, dim))
        target_data_batch_1 = np.zeros((train_size))
        #target_id_batch_1 = np.zeros((train_size))
        target_id_batch_1 = []
        all_samples = 0
        for step, (target_features, target_values, target_batch, target_id) in enumerate(query_db):

            output = encode_network([target_features, target_values], training=False)
            output = tf.nn.l2_normalize(output, axis=-1)
            target_data_feature_1[all_samples:all_samples + len(target_id), :] = output
            target_data_batch_1[all_samples:all_samples + len(target_id)] = target_batch
            #target_id_batch_1[all_samples:all_samples + len(target_id)] = target_id.numpy().decode("utf-8")
            target_id_batch_1.extend(list(target_id.numpy().astype('U')))
            all_samples += len(target_id)

        target_data_feature.extend(target_data_feature_1)
        target_data_batch.extend(target_data_batch_1)
        target_data_id.extend(target_id_batch_1)


    ref_embedding = np.array(source_data_feature)
    query_embedding = np.array(target_data_feature)
    source_data_id_subsample = source_data_id
    #source_data_id_subsample = [i.decode("utf-8") for i in source_data_id_subsample]
    #target_data_id = [i.decode("utf-8") for i in target_data_id]
    print('query embedding shape', query_embedding.shape)
    print('reference embedding shape', ref_embedding.shape)
    print('ref id length',len(source_data_id_subsample))
    print('query id length', len(target_data_id))
    return ref_embedding, query_embedding,source_data_id_subsample,target_data_id # N*dim, 顺序按照adata1， adata2的cell 顺序；

def concerto_test_ref_query(model_path:str, ref_tf_path:str, query_tf_path:str, super_parameters=None):
    if super_parameters is None:
        super_parameters = {'batch_size': 32, 'epoch': 1, 'lr': 1e-5,'drop_rate': 0.1}

    f = np.load(ref_tf_path + 'vocab_size.npz')
    vocab_size = int(f['vocab size'])
    encode_network = multi_embedding_attention_transfer(
        multi_max_features=[vocab_size],
        mult_feature_names=['RNA'],
        embedding_dims=128,
        include_attention=True,
        drop_rate=super_parameters['drop_rate'],
        head_1=128,
        head_2=128,
        head_3=128)

    tf_list_1 = [f for f in os.listdir(os.path.join(ref_tf_path)) if 'tfrecord' in f]
    train_source_list = []
    for i in tf_list_1:
        train_source_list.append(os.path.join(ref_tf_path, i))
    tf_list_2 = [f for f in os.listdir(os.path.join(query_tf_path)) if 'tfrecord' in f]
    train_target_list = []
    for i in tf_list_2:
        train_target_list.append(os.path.join(query_tf_path, i))
    weight_id_list = []
    weight_list = [f for f in os.listdir(model_path) if f.endswith('h5')]

    for id in weight_list:
        id_1 = re.findall('.*epoch(.*).h.*', id)  # f1
        weight_id_list.append(int(id_1[0]))
    # encode_network.load_weights(model_path + 'weight_encoder_epoch{}.h5'.format(max(weight_id_list)))
    encode_network.load_weights(model_path + 'weight_encoder_epoch{}.h5'.format(max(weight_id_list)), by_name=True) # yyyx0126, 支持多模态模型
    
    source_data_batch = []
    source_data_feature = []
    source_data_id = []
    batch_size = super_parameters['batch_size']
    for file in train_source_list:
        print(file)
        train_size = 0
        ref_db = create_classifier_dataset_multi(
            [file],
            batch_size=batch_size,
            is_training=False,
            data_augment=False,
            shuffle_size=10000)
        for step, (target_features, target_values, target_batch, target_id) in enumerate(ref_db):
            train_size += len(target_id)
            if step == 0:
                output = encode_network([target_features, target_values], training=False)

        dim = output.shape[1]
        source_data_feature_1 = np.zeros((train_size, dim))
        source_data_batch_1 = np.zeros((train_size))
        #source_id_batch_1 = np.zeros((train_size))
        source_id_batch_1 = []
        all_samples = 0
        for step, (target_features, target_values, target_batch, target_id) in enumerate(ref_db):
            output = encode_network([target_features, target_values], training=False)
            output = tf.nn.l2_normalize(output, axis=-1)
            source_data_feature_1[all_samples:all_samples + len(target_id), :] = output
            source_data_batch_1[all_samples:all_samples + len(target_id)] = target_batch
            #source_id_batch_1[all_samples:all_samples + len(target_id)] = target_id.numpy().decode("utf-8")
            source_id_batch_1.extend(list(target_id.numpy().astype('U')))

            all_samples += len(target_id)
        source_data_feature.extend(source_data_feature_1)
        source_data_batch.extend(source_data_batch_1)
        source_data_id.extend(source_id_batch_1)

    target_data_batch = []
    target_data_feature = []
    target_data_id = []
    for file in train_target_list:
        print(file)
        train_size = 0
        query_db = create_classifier_dataset_multi([file],
                                                   batch_size=batch_size,
                                                   is_training=False,
                                                   data_augment=False,
                                                   shuffle_size=10000)
        for step, (target_features, target_values, target_batch, target_id) in enumerate(query_db):
            train_size += len(target_id)
            if step == 0:
                output = encode_network([target_features, target_values], training=False)

        dim = output.shape[1]
        target_data_feature_1 = np.zeros((train_size, dim))
        target_data_batch_1 = np.zeros((train_size))
        #target_id_batch_1 = np.zeros((train_size))
        target_id_batch_1 = []
        all_samples = 0
        for step, (target_features, target_values, target_batch, target_id) in enumerate(query_db):

            output = encode_network([target_features, target_values], training=False)
            output = tf.nn.l2_normalize(output, axis=-1)
            target_data_feature_1[all_samples:all_samples + len(target_id), :] = output
            target_data_batch_1[all_samples:all_samples + len(target_id)] = target_batch
            #target_id_batch_1[all_samples:all_samples + len(target_id)] = target_id.numpy().decode("utf-8")
            target_id_batch_1.extend(list(target_id.numpy().astype('U')))
            all_samples += len(target_id)

        target_data_feature.extend(target_data_feature_1)
        target_data_batch.extend(target_data_batch_1)
        target_data_id.extend(target_id_batch_1)


    ref_embedding = np.array(source_data_feature)
    query_embedding = np.array(target_data_feature)
    source_data_id_subsample = source_data_id
    #source_data_id_subsample = [i.decode("utf-8") for i in source_data_id_subsample]
    #target_data_id = [i.decode("utf-8") for i in target_data_id]
    print('query embedding shape', query_embedding.shape)
    print('reference embedding shape', ref_embedding.shape)
    print('ref id length',len(source_data_id_subsample))
    print('query id length', len(target_data_id))
    return ref_embedding, query_embedding,source_data_id_subsample,target_data_id # N*dim, 顺序按照adata1， adata2的cell 顺序；


def concerto_test_ref(model_path:str, ref_tf_path:str, super_parameters=None,saved_weight_path=None):
    if super_parameters is None:
        super_parameters = {'batch_size': 128, 'epoch': 1, 'lr': 1e-5,'drop_rate': 0.1, "attention_t": True, "heads": 128}

    f = np.load(ref_tf_path + 'vocab_size.npz')
    vocab_size = int(f['vocab size'])
    encode_network = multi_embedding_attention_transfer(
        multi_max_features=[vocab_size],
        mult_feature_names=['RNA'],
        embedding_dims=128,
        include_attention=super_parameters["attention_t"],
        drop_rate=super_parameters['drop_rate'],
        head_1=super_parameters["heads"],
        head_2=super_parameters["heads"],
        head_3=super_parameters["heads"])

    tf_list_1 = [f for f in os.listdir(os.path.join(ref_tf_path)) if 'tfrecord' in f]
    train_source_list = []
    for i in tf_list_1:
        train_source_list.append(os.path.join(ref_tf_path, i))

        
    if saved_weight_path is None:
        weight_id_list = []
        weight_list = [f for f in os.listdir(model_path) if f.endswith('h5')]
        for id in weight_list:
            id_1 = re.findall('.*epoch(.*).h.*', id)  # f1
            weight_id_list.append(int(id_1[0]))
            encode_network.load_weights(model_path + 'weight_encoder_epoch{}.h5'.format(max(weight_id_list)),
                                    by_name=True)

    else:
        encode_network.load_weights(saved_weight_path, by_name=True)
        print('load saved weight')
    
    source_data_batch = []
    source_data_feature = []
    source_data_id = []
    batch_size = super_parameters['batch_size']
    for file in train_source_list:
        print(file)
        train_size = 0
        ref_db = create_classifier_dataset_multi(
            [file],
            batch_size=batch_size,
            is_training=False,
            data_augment=False,
            shuffle_size=10000)
        for step, (target_features, target_values, target_batch, target_id) in enumerate(ref_db):
            train_size += len(target_id)
            if step == 0:
                output = encode_network([target_features, target_values], training=False)

        dim = output.shape[1]
        source_data_feature_1 = np.zeros((train_size, dim))
        source_data_batch_1 = np.zeros((train_size))
        #source_id_batch_1 = np.zeros((train_size))
        source_id_batch_1 = []
        all_samples = 0
        for step, (target_features, target_values, target_batch, target_id) in enumerate(ref_db):
            output = encode_network([target_features, target_values], training=False)
            output = tf.nn.l2_normalize(output, axis=-1)
            source_data_feature_1[all_samples:all_samples + len(target_id), :] = output
            source_data_batch_1[all_samples:all_samples + len(target_id)] = target_batch
            #source_id_batch_1[all_samples:all_samples + len(target_id)] = target_id.numpy().decode("utf-8")
            source_id_batch_1.extend(list(target_id.numpy().astype('U')))

            all_samples += len(target_id)
        source_data_feature.extend(source_data_feature_1)
        source_data_batch.extend(source_data_batch_1)
        source_data_id.extend(source_id_batch_1)

    ref_embedding = np.array(source_data_feature)    
    print('reference embedding shape', ref_embedding.shape)
    print('ref id length',len(source_data_id))    
    return ref_embedding, source_data_id 


def concerto_test_multimodal_decoder(mult_feature_names:list, model_path: str, RNA_tf_path: str, Protein_tf_path: str, n_cells_for_sample=None,super_parameters=None,
                             saved_weight_path=None):
    if super_parameters is None:
        super_parameters = {'batch_size': 32, 'epoch': 1, 'lr': 1e-5, 'drop_rate': 0.1}
    
    batch_size = super_parameters['batch_size']
    f = np.load(os.path.join(RNA_tf_path, 'vocab_size.npz'))
    vocab_size_RNA = int(f['vocab size'])
    f = np.load(os.path.join(Protein_tf_path, 'vocab_size.npz'))
    vocab_size_Protein = int(f['vocab size'])
    # encode_network = multi_embedding_attention_transfer_explainability(
    #     multi_max_features=[vocab_size_RNA, vocab_size_Protein],
    #     mult_feature_names=['RNA', 'Protein'],
    #     embedding_dims=128,
    #     include_attention=True,
    #     drop_rate=super_parameters['drop_rate'],
    #     head_1=128,
    #     head_2=128,
    #     head_3=128)
    
    decode_network = multi_embedding_attention_transfer(
        multi_max_features=[vocab_size_RNA,vocab_size_Protein],
        mult_feature_names=mult_feature_names,
        embedding_dims=128,
        include_attention=super_parameters['attention_s'],
        drop_rate=super_parameters['drop_rate'],
        head_1=super_parameters["heads"],
        head_2=super_parameters["heads"],
        head_3=super_parameters["heads"])


    tf_list_1 = [f for f in os.listdir(os.path.join(RNA_tf_path)) if 'tfrecord' in f]
    train_source_list_RNA = []
    train_source_list_Protein = []
    for i in tf_list_1:
        train_source_list_RNA.append(os.path.join(RNA_tf_path, i))
        train_source_list_Protein.append(os.path.join(Protein_tf_path, i))

    if super_parameters["epoch"] != 0:
        if saved_weight_path is None:
            weight_id_list = []
            weight_list = [f for f in os.listdir(model_path) if f.endswith('h5')]
            for id in weight_list:
                id_1 = re.findall('.*epoch(.*).h.*', id)  # f1
                weight_id_list.append(int(id_1[0]))
            decode_network.load_weights(model_path + 'weight_decoder_epoch{}.h5'.format(max(weight_id_list)),
                                        by_name=True)

        else:
            decode_network.load_weights(saved_weight_path, by_name=True)
            print('load saved weight')

    source_data_batch = []
    source_data_feature = []
    RNA_id_all = []
    attention_output_RNA_all = []
    attention_output_Protein_all = []
    for RNA_file, Protein_file in zip(train_source_list_RNA, train_source_list_Protein):
        train_size = 0
        train_db_RNA = create_classifier_dataset_multi([RNA_file],
                                                       batch_size=batch_size,
                                                       is_training=False,
                                                       data_augment=False,
                                                       shuffle_size=10000,
                                                       )
        train_db_Protein = create_classifier_dataset_multi([Protein_file],
                                                           batch_size=batch_size,
                                                           is_training=False,
                                                           data_augment=False,
                                                           shuffle_size=10000,
                                                           )
        step = 0
        for (source_features_RNA, source_values_RNA,
             source_batch_RNA, source_id_RNA), \
            (source_features_protein, source_values_protein,
             source_batch_Protein, source_id_Protein) \
                in (zip(train_db_RNA, train_db_Protein)):
            #train_size += len(source_id_RNA)
            if step == 0:
                encode_output = decode_network([source_values_RNA, source_values_protein], training=False)
                break

        dim = encode_output.shape[1]
        if n_cells_for_sample is None:            
            feature_len = 1000000
        else:            
            n_cells_for_sample_1 = n_cells_for_sample//8
            feature_len = n_cells_for_sample_1 // batch_size * batch_size
        
        source_data_feature_1 = np.zeros((feature_len, dim))
        source_data_batch_1 = np.zeros((feature_len))
        # print('feature_len:', feature_len)
        RNA_id = []
        all_samples = 0
        for (source_features_RNA, source_values_RNA,
             source_batch_RNA, source_id_RNA), \
            (source_features_protein, source_values_protein,
             source_batch_Protein, source_id_Protein) \
                in (zip(train_db_RNA, train_db_Protein)):
            if all_samples  >= feature_len:
                break
            decode_output = decode_network([[source_features_RNA, source_features_protein],
                                                              [source_values_RNA, source_values_protein]],
                                                             training=False)

            decode_output = tf.nn.l2_normalize(decode_output, axis=-1)
            source_data_feature_1[all_samples:all_samples + len(source_id_RNA), :] = decode_output
            source_data_batch_1[all_samples:all_samples + len(source_id_RNA)] = source_batch_RNA
            RNA_id.extend(list(source_id_RNA.numpy().astype('U')))
            all_samples += len(source_id_RNA)
            # print('all_samples num:{}'.format(all_samples))

        source_data_feature.extend(source_data_feature_1[:all_samples])
        source_data_batch.extend(source_data_batch_1[:all_samples])
        RNA_id_all.extend(RNA_id[:all_samples])
        
    source_data_feature = np.array(source_data_feature).astype('float32')
    source_data_batch = np.array(source_data_batch).astype('int32')
    
    #np.savez_compressed('./multi_attention.npz', **attention_weight)
    return source_data_feature, source_data_batch, RNA_id_all, None


def concerto_test_multimodal(mult_feature_names, model_path: str, RNA_tf_path: str, Protein_tf_path: str, n_cells_for_sample=None,super_parameters=None,
                             saved_weight_path=None, only_RNA=False):
    if super_parameters is None:
        super_parameters = {'batch_size': 32, 'epoch': 1, 'lr': 1e-5, 'drop_rate': 0.1, 'combine_omics': False}
    
    batch_size = super_parameters['batch_size']
    f = np.load(os.path.join(RNA_tf_path, 'vocab_size.npz'))
    vocab_size_RNA = int(f['vocab size'])
    f = np.load(os.path.join(Protein_tf_path, 'vocab_size.npz'))
    vocab_size_Protein = int(f['vocab size'])
    encode_network = multi_embedding_attention_transfer_explainability(
        multi_max_features=[vocab_size_RNA, vocab_size_Protein],
        mult_feature_names=mult_feature_names,
        embedding_dims=128,
        include_attention=True,
        drop_rate=super_parameters['drop_rate'],
        head_1=super_parameters['heads'],
        head_2=super_parameters['heads'],
        head_3=super_parameters['heads'],
        combine_omics=super_parameters['combine_omics'])

    tf_list_1 = [f for f in os.listdir(os.path.join(RNA_tf_path)) if 'tfrecord' in f]
    train_source_list_RNA = []
    train_source_list_Protein = []
    for i in tf_list_1:
        train_source_list_RNA.append(os.path.join(RNA_tf_path, i))
        train_source_list_Protein.append(os.path.join(Protein_tf_path, i))

    if saved_weight_path is None:
        weight_id_list = []
        weight_list = [f for f in os.listdir(model_path) if f.endswith('h5')]
        for id in weight_list:
            id_1 = re.findall('.*epoch(.*).h.*', id)  # f1
            weight_id_list.append(int(id_1[0]))
        encode_network.load_weights(model_path + 'weight_encoder_epoch{}.h5'.format(max(weight_id_list)),
                                    by_name=True)

    else:
        encode_network.load_weights(saved_weight_path, by_name=True)
        print('load saved weight')

    source_data_batch = []
    source_data_feature = []
    RNA_id_all = []
    attention_output_RNA_all = []
    attention_output_Protein_all = []
    for RNA_file, Protein_file in zip(train_source_list_RNA, train_source_list_Protein):
        train_size = 0
        train_db_RNA = create_classifier_dataset_multi([RNA_file],
                                                       batch_size=batch_size,
                                                       is_training=False,
                                                       data_augment=False,
                                                       shuffle_size=10000,
                                                       )
        train_db_Protein = create_classifier_dataset_multi([Protein_file],
                                                           batch_size=batch_size,
                                                           is_training=False,
                                                           data_augment=False,
                                                           shuffle_size=10000,
                                                           )
        step = 0
        for (source_features_RNA, source_values_RNA,
             source_batch_RNA, source_id_RNA), \
            (source_features_protein, source_values_protein,
             source_batch_Protein, source_id_Protein) \
                in (zip(train_db_RNA, train_db_Protein)):
            #train_size += len(source_id_RNA)
            if step == 0:
                if super_parameters["combine_omics"]:
                    encode_output, attention_output = encode_network([[source_features_RNA, source_features_protein],
                                                                    [source_values_RNA, source_values_protein]],
                                                                    training=False)
                    break

                else:
                    encode_output1, encode_output2, attention_output = encode_network([[source_features_RNA, source_features_protein],
                                                                [source_values_RNA, source_values_protein]],
                                                                training=False)
                    if only_RNA:
                        # FIXME back
                        encode_output = encode_output1
                    else:
                        encode_output = tf.concat([encode_output1, encode_output2], axis=1)
                    break

        dim = encode_output.shape[1]
        if n_cells_for_sample is None:            
            feature_len = 1000000
        else:            
            n_cells_for_sample_1 = n_cells_for_sample//8
            feature_len = n_cells_for_sample_1 // batch_size * batch_size
        
        source_data_feature_1 = np.zeros((feature_len, dim))
        source_data_batch_1 = np.zeros((feature_len))
        attention_output_RNA = np.zeros((feature_len, vocab_size_RNA, 1))
        attention_output_Protein = np.zeros((feature_len, vocab_size_Protein, 1))
        # print('feature_len:', feature_len)
        RNA_id = []
        all_samples = 0
        for (source_features_RNA, source_values_RNA,
             source_batch_RNA, source_id_RNA), \
            (source_features_protein, source_values_protein,
             source_batch_Protein, source_id_Protein) \
                in (zip(train_db_RNA, train_db_Protein)):
            if all_samples  >= feature_len:
                print("Entered if break")
                break

            if super_parameters["combine_omics"]:
                if only_RNA:
                    encode_output, attention_output = encode_network([[source_features_RNA],
                                                                [source_values_RNA]],
                                                                training=False)
                else:
                    encode_output, attention_output = encode_network([[source_features_RNA, source_features_protein],
                                                                    [source_values_RNA, source_values_protein]],
                                                                    training=False)

            else:
                encode_output1, encode_output2, attention_output = encode_network([[source_features_RNA, source_features_protein], 
                                                                                   [source_values_RNA, source_values_protein]],
                                                             training=False)
                if only_RNA:
                    if super_parameters["data"] == "human":
                        encode_output = encode_output2
                    else:
                        encode_output = encode_output1
                else:
                    encode_output = tf.concat([encode_output1, encode_output2], axis=1)

            encode_output = tf.nn.l2_normalize(encode_output, axis=-1)
            source_data_feature_1[all_samples:all_samples + len(source_id_RNA), :] = encode_output
            source_data_batch_1[all_samples:all_samples + len(source_id_RNA)] = source_batch_RNA
            RNA_id.extend(list(source_id_RNA.numpy().astype('U')))
            attention_output_RNA[all_samples:all_samples + len(source_id_RNA), :, :] = attention_output[0]
            attention_output_Protein[all_samples:all_samples + len(source_id_RNA), :, :] = attention_output[1]
            all_samples += len(source_id_RNA)
            # print('all_samples num:{}'.format(all_samples))

        source_data_feature.extend(source_data_feature_1[:all_samples])
        source_data_batch.extend(source_data_batch_1[:all_samples])
        RNA_id_all.extend(RNA_id[:all_samples])
        attention_output_RNA_all.extend(attention_output_RNA[:all_samples])
        attention_output_Protein_all.extend(attention_output_Protein[:all_samples])

    source_data_feature = np.array(source_data_feature).astype('float32')
    source_data_batch = np.array(source_data_batch).astype('int32')
    attention_weight = {'attention_output_RNA': attention_output_RNA_all,
                        'attention_output_Protein': attention_output_Protein_all}
    #np.savez_compressed('./multi_attention.npz', **attention_weight)
    return source_data_feature, source_data_batch, RNA_id_all, attention_weight

def concerto_test_multimodal_modalities(mult_feature_names, model_path: str, RNA_tf_path: str, Protein_tf_path: str, n_cells_for_sample=None,super_parameters=None,
                             saved_weight_path=None, only_RNA=False):
    if super_parameters is None:
        super_parameters = {'batch_size': 32, 'epoch': 1, 'lr': 1e-5, 'drop_rate': 0.1, 'combine_omics': False}
    
    batch_size = super_parameters['batch_size']
    f = np.load(os.path.join(RNA_tf_path, 'vocab_size.npz'))
    vocab_size_RNA = int(f['vocab size'])
    f = np.load(os.path.join(Protein_tf_path, 'vocab_size.npz'))
    vocab_size_Protein = int(f['vocab size'])
    encode_network = multi_embedding_attention_transfer_explainability_modalities(
        multi_max_features=[vocab_size_RNA, vocab_size_Protein],
        mult_feature_names=mult_feature_names,
        embedding_dims=128,
        include_attention=True,
        drop_rate=super_parameters['drop_rate'],
        head_1=super_parameters['heads'],
        head_2=super_parameters['heads'],
        head_3=super_parameters['heads'],
        combine_omics=super_parameters['combine_omics'])

    tf_list_1 = [f for f in os.listdir(os.path.join(RNA_tf_path)) if 'tfrecord' in f]
    train_source_list_RNA = []
    train_source_list_Protein = []
    for i in tf_list_1:
        train_source_list_RNA.append(os.path.join(RNA_tf_path, i))
        train_source_list_Protein.append(os.path.join(Protein_tf_path, i))

    if saved_weight_path is None:
        weight_id_list = []
        weight_list = [f for f in os.listdir(model_path) if f.endswith('h5')]
        for id in weight_list:
            id_1 = re.findall('.*epoch(.*).h.*', id)  # f1
            weight_id_list.append(int(id_1[0]))
        encode_network.load_weights(model_path + 'weight_encoder_epoch{}.h5'.format(max(weight_id_list)),
                                    by_name=True)

    else:
        encode_network.load_weights(saved_weight_path, by_name=True)
        print('load saved weight')

    source_data_batch = []
    source_data_feature, source_data_feature2 = [], []
    RNA_id_all = []
    attention_output_RNA_all = []
    attention_output_Protein_all = []
    for RNA_file, Protein_file in zip(train_source_list_RNA, train_source_list_Protein):
        train_size = 0
        train_db_RNA = create_classifier_dataset_multi([RNA_file],
                                                       batch_size=batch_size,
                                                       is_training=False,
                                                       data_augment=False,
                                                       shuffle_size=10000,
                                                       )
        train_db_Protein = create_classifier_dataset_multi([Protein_file],
                                                           batch_size=batch_size,
                                                           is_training=False,
                                                           data_augment=False,
                                                           shuffle_size=10000,
                                                           )
        step = 0
        for (source_features_RNA, source_values_RNA,
             source_batch_RNA, source_id_RNA), \
            (source_features_protein, source_values_protein,
             source_batch_Protein, source_id_Protein) \
                in (zip(train_db_RNA, train_db_Protein)):
            #train_size += len(source_id_RNA)
            if step == 0:
                if super_parameters["combine_omics"]:
                    encode_output1, encode_output2, attention_output = encode_network([[source_features_RNA, source_features_protein],
                                                                    [source_values_RNA, source_values_protein]],
                                                                    training=False)
                    break
                else:
                    raise Exception("Invalid arguments")

        dim = encode_output1.shape[1]
        if n_cells_for_sample is None:            
            feature_len = 1000000
        else:            
            n_cells_for_sample_1 = n_cells_for_sample//8
            feature_len = n_cells_for_sample_1 // batch_size * batch_size
        
        source_data_feature_1 = np.zeros((feature_len, dim))
        source_data_feature_2 = np.zeros((feature_len, dim))
        source_data_batch_1 = np.zeros((feature_len))
        attention_output_RNA = np.zeros((feature_len, vocab_size_RNA, 1))
        attention_output_Protein = np.zeros((feature_len, vocab_size_Protein, 1))
        # print('feature_len:', feature_len)
        RNA_id = []
        all_samples = 0
        for (source_features_RNA, source_values_RNA,
             source_batch_RNA, source_id_RNA), \
            (source_features_protein, source_values_protein,
             source_batch_Protein, source_id_Protein) \
                in (zip(train_db_RNA, train_db_Protein)):
            if all_samples  >= feature_len:
                print("Entered if break")
                break

            if super_parameters["combine_omics"]:
                encode_output1, encode_output2, attention_output = encode_network([[source_features_RNA, source_features_protein], 
                                                                                   [source_values_RNA, source_values_protein]],
                                                             training=False)

            else:
                raise Exception("Invalid arguments")

            encode_output1 = tf.nn.l2_normalize(encode_output1, axis=-1)
            source_data_feature_1[all_samples:all_samples + len(source_id_RNA), :] = encode_output1
            encode_output2 = tf.nn.l2_normalize(encode_output2, axis=-1)
            source_data_feature_2[all_samples:all_samples + len(source_id_RNA), :] = encode_output2
            source_data_batch_1[all_samples:all_samples + len(source_id_RNA)] = source_batch_RNA
            RNA_id.extend(list(source_id_RNA.numpy().astype('U')))
            attention_output_RNA[all_samples:all_samples + len(source_id_RNA), :, :] = attention_output[0]
            attention_output_Protein[all_samples:all_samples + len(source_id_RNA), :, :] = attention_output[1]
            all_samples += len(source_id_RNA)
            # print('all_samples num:{}'.format(all_samples))

        source_data_feature.extend(source_data_feature_1[:all_samples])
        source_data_feature2.extend(source_data_feature_2[:all_samples])
        source_data_batch.extend(source_data_batch_1[:all_samples])
        RNA_id_all.extend(RNA_id[:all_samples])
        attention_output_RNA_all.extend(attention_output_RNA[:all_samples])
        attention_output_Protein_all.extend(attention_output_Protein[:all_samples])

    source_data_feature = np.array(source_data_feature).astype('float32')
    source_data_feature2 = np.array(source_data_feature2).astype('float32')
    source_data_batch = np.array(source_data_batch).astype('int32')
    attention_weight = {'attention_output_RNA': attention_output_RNA_all,
                        'attention_output_Protein': attention_output_Protein_all}
    #np.savez_compressed('./multi_attention.npz', **attention_weight)
    return source_data_feature, source_data_feature2, source_data_batch, RNA_id_all, attention_weight


def concerto_test_multimodal_project(model_path: str, RNA_tf_path: str, Protein_tf_path: str, super_parameters=None,saved_weight_path = None):
    if super_parameters is None:
        super_parameters = {'batch_size': 32, 'epoch': 1, 'lr': 1e-5, 'drop_rate': 0.1}

    batch_size = super_parameters['batch_size']
    f = np.load(os.path.join(RNA_tf_path, 'vocab_size.npz'))
    vocab_size_RNA = int(f['vocab size'])
    f = np.load(os.path.join(Protein_tf_path, 'vocab_size.npz'))
    vocab_size_Protein = int(f['vocab size'])
    encode_network = multi_embedding_attention_transfer_explainability(
        multi_max_features=[vocab_size_RNA,vocab_size_Protein],
        mult_feature_names=['RNA','Protein'],
        embedding_dims=128,
        include_attention=True,
        drop_rate=super_parameters['drop_rate'],
        head_1=128,
        head_2=128,
        head_3=128)
    encode_network_RNA = multi_embedding_attention_transfer_explainability(
        multi_max_features=[vocab_size_RNA],
        mult_feature_names=['RNA'],
        embedding_dims=128,
        include_attention=True,
        drop_rate=super_parameters['drop_rate'],
        head_1=128,
        head_2=128,
        head_3=128)


    tf_list_1 = [f for f in os.listdir(os.path.join(RNA_tf_path)) if 'tfrecord' in f]
    train_source_list_RNA = []
    train_source_list_Protein = []
    for i in tf_list_1:
        train_source_list_RNA.append(os.path.join(RNA_tf_path, i))
        train_source_list_Protein.append(os.path.join(Protein_tf_path, i))

    weight_id_list = []
    weight_list = [f for f in os.listdir(model_path) if f.endswith('h5')]

    for id in weight_list:
        id_1 = re.findall('.*epoch(.*).h.*', id)  # f1
        weight_id_list.append(int(id_1[0]))
    
    if  saved_weight_path is None:
        encode_network.load_weights(model_path + 'weight_encoder_epoch{}.h5'.format(max(weight_id_list)),
                                    by_name=True)
        encode_network_RNA.load_weights(model_path + 'weight_encoder_epoch{}.h5'.format(max(weight_id_list)),
                                    by_name=True)

    else:
        encode_network.load_weights(saved_weight_path,by_name=True)
        encode_network_RNA.load_weights(saved_weight_path, by_name=True)
        

    source_data_batch = []
    source_data_feature = []
    source_data_feature_RNA = []    
    RNA_id_all = []

    for RNA_file, Protein_file in zip(train_source_list_RNA, train_source_list_Protein):
        print(RNA_file)
        print(Protein_file)
        train_size = 0
        train_db_RNA = create_classifier_dataset_multi([RNA_file],
                                                       batch_size=batch_size,
                                                       is_training=False,
                                                       data_augment=False,
                                                       shuffle_size=10000,
                                                       )
        train_db_Protein = create_classifier_dataset_multi([Protein_file],
                                                           batch_size=batch_size,
                                                           is_training=False,
                                                           data_augment=False,
                                                           shuffle_size=10000,
                                                           )
        step = 0
        for (source_features_RNA, source_values_RNA,
             source_batch_RNA, source_id_RNA), \
            (source_features_protein, source_values_protein,
             source_batch_Protein, source_id_Protein) \
                in (zip(train_db_RNA, train_db_Protein)):
            train_size += len(source_id_RNA)
            if step == 0:
                encode_output, attention_output = encode_network([[source_features_RNA, source_features_protein],
                                                                        [source_values_RNA, source_values_protein]],
                                                                       training=False)

        dim = encode_output.shape[1]
        source_data_feature_1 = np.zeros((train_size, dim))
        source_data_feature_RNA_1 = np.zeros((train_size, dim))        
        source_data_batch_1 = np.zeros((train_size))
        RNA_id = []
        all_samples = 0
        for (source_features_RNA, source_values_RNA,
             source_batch_RNA, source_id_RNA), \
            (source_features_protein, source_values_protein,
             source_batch_Protein, source_id_Protein) \
                in (zip(train_db_RNA, train_db_Protein)):
            encode_output, attention_output = encode_network([[source_features_RNA, source_features_protein],
                                                              [source_values_RNA, source_values_protein]],
                                                             training=False)
            encode_output_RNA, attention_output_ = encode_network_RNA([[source_features_RNA],
                                                              [source_values_RNA]],
                                                             training=False)


            encode_output = tf.nn.l2_normalize(encode_output, axis=-1)
            source_data_feature_1[all_samples:all_samples + len(source_id_RNA), :] = encode_output
            source_data_feature_RNA_1[all_samples:all_samples + len(source_id_RNA), :] = encode_output_RNA            
            source_data_batch_1[all_samples:all_samples + len(source_id_RNA)] = source_batch_RNA
            RNA_id.extend(list(source_id_RNA.numpy().astype('U')))
            all_samples += len(source_id_RNA)
            # print('all_samples num:{}'.format(all_samples))

        source_data_feature.extend(source_data_feature_1)
        source_data_feature_RNA.extend(source_data_feature_RNA_1)        
        source_data_batch.extend(source_data_batch_1)
        RNA_id_all.extend(RNA_id)

    source_data_feature = np.array(source_data_feature).astype('float32')
    source_data_feature_RNA = np.array(source_data_feature_RNA).astype('float32')    
    source_data_batch = np.array(source_data_batch).astype('int32')

    return source_data_feature,source_data_feature_RNA, source_data_batch, RNA_id_all




def knn_classifier(ref_embedding, query_embedding, ref_anndata, column_name,k, num_chunks=100):
    '''
    return :
        target_neighbor: predicted label
        traget_prob: confidence score
    '''
    train_features = tf.transpose(ref_embedding)
    num_test_images = int(query_embedding.shape[0])
    imgs_per_chunk = num_test_images // num_chunks
    if imgs_per_chunk == 0:
        imgs_per_chunk = 10

    print(num_test_images, imgs_per_chunk)
    # ref_anndata = ref_anndata[source_data_id]
    train_labels = ref_anndata.obs[column_name].tolist()
    target_pred_labels = []
    target_pred_prob = []
    for idx in range(0, num_test_images, imgs_per_chunk):
        # get the features for test images
        features = query_embedding[
                   idx: min((idx + imgs_per_chunk), num_test_images), :
                   ]
        # targets = test_labels[idx : min((idx + imgs_per_chunk), num_test_images)]
        similarity = tf.matmul(features, train_features)
        target_distances, target_indices = tf.math.top_k(similarity, k, sorted=True)

        for distances, indices in zip(target_distances, target_indices):
            selected_label = {}
            selected_count = {}
            count = 0
            for distance, index in zip(distances, indices):
                label = train_labels[index]
                weight = distance
                if label not in selected_label:
                    selected_label[label] = 0
                    selected_count[label] = 0
                selected_label[label] += weight
                selected_count[label] += 1
                count += 1

            filter_label_list = sorted(selected_label.items(), key=lambda x: x[1], reverse=True)
            target_pred_labels.append(filter_label_list[0][0])

            prob = selected_label[filter_label_list[0][0]] / selected_count[filter_label_list[0][0]]
            target_pred_prob.append(prob)

    target_neighbor = np.array(target_pred_labels)
    target_prob = np.array(target_pred_prob)

    return target_neighbor, target_prob #返回预测的label和置信度

def knn_classifier_faiss(ref_embedding, query_embedding, ref_anndata, source_data_id, column_name,k, num_chunks=100):
    '''
    return :
        target_neighbor: predicted label
        traget_prob: confidence score
    '''
    import faiss
    
    ref_embedding = ref_embedding.astype('float32')
    query_embedding = query_embedding.astype('float32')
    n, dim = ref_embedding.shape[0], ref_embedding.shape[1]
    index = faiss.IndexFlatIP(dim)
    #index = faiss.index_cpu_to_all_gpus(index)
    index.add(ref_embedding)
    ref_anndata = ref_anndata[source_data_id]
    train_labels = ref_anndata.obs[column_name].tolist()
    target_pred_labels = []
    target_pred_prob = []

    target_distances, target_indices= index.search(query_embedding, k)  # Sample itself is included

    for distances, indices in zip(target_distances, target_indices):
        selected_label = {}
        selected_count = {}
        count = 0
        for distance, index in zip(distances, indices):
            label = train_labels[index]
            weight = distance
            if label not in selected_label:
                selected_label[label] = 0
                selected_count[label] = 0
            selected_label[label] += weight
            selected_count[label] += 1
            count += 1

        filter_label_list = sorted(selected_label.items(), key=lambda x: x[1], reverse=True)
        target_pred_labels.append(filter_label_list[0][0])

        prob = selected_label[filter_label_list[0][0]] / selected_count[filter_label_list[0][0]]
        target_pred_prob.append(prob)

    target_neighbor = np.array(target_pred_labels)
    target_prob = np.array(target_pred_prob)

    return target_neighbor, target_prob #返回预测的label和置信度

# GPU control
def concerto_GPU():
    return


if __name__ == '__main__':
    ref_path = ''
    query_path = ''