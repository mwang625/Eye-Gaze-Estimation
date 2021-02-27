from typing import Dict

import tensorflow as tf

from core import BaseDataSource, BaseModel
import util.gaze

#from  tensorflow.compat.v1.keras.applications.resnet_v2 import ResNet152V2 as RN, preprocess_input
#from  tensorflow.keras.applications.resnet import ResNet101 as RN, preprocess_input
from  tensorflow.keras.applications.resnet50 import  preprocess_input
from  tensorflow.keras.applications import ResNet50 as RN
from tensorflow.keras.layers import Dense,Flatten, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import BatchNormalization, RepeatVector, Input, Reshape

class NewModel(BaseModel):
    """An example neural network architecture."""

    def build_model(self, data_sources: Dict[str, BaseDataSource], mode: str):
        """Build model."""
        data_source = next(iter(data_sources.values()))
        input_tensors = data_source.output_tensors
        # stack left-eye and right-eye to get a 6 channels tensors, placeholder
        # le = input_tensors['left-eye']
        # re = input_tensors['right-eye']

        le = input_tensors['left-eye'] # shape (60 224 3)
        re = input_tensors['right-eye'] # shape (60 224 3)
        face = input_tensors['face'] #shape (224, 224, 3)

        le = preprocess_input(le) # add one dimension for batch
        re = preprocess_input(re) # add one dimension for batch
        face = preprocess_input(face)  # add one dimension for batch
        # RN = resNet50.ResNet50
        # le_conv = RN(weights='imagenet', include_top=False, input_tensor = le)
        le_conv = RN(weights='imagenet', include_top=False, input_tensor = le)
        # re_conv = RN(weights='imagenet', include_top=False, input_tensor = re)
        re_conv = RN(weights='imagenet', include_top=False, input_tensor = re)
        face_conv = RN(weights='imagenet', include_top=False, input_tensor = face)

        le_flatten = Flatten()(le_conv.output)
        re_flatten = Flatten()(re_conv.output)
        face_flatten = Flatten()(face_conv.output)

        # for face
        # face_flatten = Dense(2048, activation='relu', name='face_1')(face_flatten)
        face_flatten = Dense(1000, activation='relu', name='face_3')(face_flatten)

        # for eye
        # le_flatten = Dense(2048, activation='relu', name='le_3')(le_flatten)
        le_flatten = Dense(1000, activation='relu', name='le_4')(le_flatten)

        # re_flatten = Dense(2048, activation='relu', name='le_3')(re_flatten)
        re_flatten = Dense(1000, activation='relu', name='le_4')(re_flatten)

        # for face landmark
        # add face landmarks(coordinates)

        # add head pose into the vector
        # x = tf.concat((le_flatten, re_flatten, face_flatten, lm,  input_tensors['head']), axis = 1)
        x = tf.concat((le_flatten, re_flatten, face_flatten, input_tensors['head']), axis = 1)
        x = BatchNormalization()(x)
        # x = tf.concat((le_flatten, re_flatten, face_flatten), axis = 1)
        x = Dense(1024, activation='relu', name='fc_3')(x)
        # x = Dense(1024, activation='relu', name='fc_4')(x)
        x = Dense(512, activation='relu', name='fc_6')(x)
        x = Dense(128, activation='relu', name='fc_7')(x)
        preds = Dense(2,name='preds')(x)

        # eye_model = Model(lre, preds)
        
        # Define outputs
        loss_terms = {}
        metrics = {}
        if 'gaze' in input_tensors:
            y = input_tensors['gaze']
            with tf.variable_scope('mse'):  # To optimize
                # NOTE: You are allowed to change the optimized loss
                loss_terms['gaze_mse'] = tf.reduce_mean(tf.squared_difference(preds, y))
                # loss_terms['gaze_mse'] = util.gaze.tensorflow_angular_error_from_pitchyaw(preds, y)
            with tf.variable_scope('ang'):  # To evaluate in addition to loss terms
                metrics['gaze_angular'] = util.gaze.tensorflow_angular_error_from_pitchyaw(preds, y)
        return {'gaze': preds}, loss_terms, metrics   # are graphs to be executed
