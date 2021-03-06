#!/usr/bin/env python3
"""Copyright (c) 2020 AIT Lab, ETH Zurich, Seonwook Park

Students and holders of copies of this code, accompanying datasets,
and documentation, are not allowed to copy, distribute or modify
any of the mentioned materials beyond the scope and duration of the
Machine Perception course projects.

That is, no partial/full copy nor modification of this code and
accompanying data should be made publicly or privately available to
current/future students or other parties."""

"""Main script for training a model for gaze estimation."""
import argparse
import multiprocessing
import os

import coloredlogs
import tensorflow as tf
import random
import numpy as np

import logging
logger = logging.getLogger(__name__)

if __name__ == '__main__':

    # Set global log level
    parser = argparse.ArgumentParser(description='Train a gaze estimation model.')
    parser.add_argument('-v', type=str, help='logging level', default='info',
                        choices=['debug', 'info', 'warning', 'error', 'critical'])
    parser.add_argument('--restore', type=str, help='Restore weights from this folder')
    args = parser.parse_args() # contains the variable v and restore
    coloredlogs.install(
        datefmt='%d/%m %H:%M',
        fmt='%(asctime)s %(levelname)s %(message)s',
        level=args.v.upper(),
    )

    # Set identifier to that provided, and restore weights from there
    identifier = None
    if args.restore is not None:
        identifier = os.path.relpath(
            args.restore,
            start=os.path.abspath(os.path.dirname(__file__)) + '/../outputs',
        )
        logger.info('Manually selected folder to restore from: %s' % identifier)

    # Initialize Tensorflow session
    tf.logging.set_verbosity(tf.logging.INFO)
    session_config = tf.ConfigProto(
        inter_op_parallelism_threads=0,
        intra_op_parallelism_threads=multiprocessing.cpu_count(),
        allow_soft_placement=True,
        gpu_options=tf.GPUOptions(allow_growth=True),
    )
    with tf.Session(config=session_config) as session:
        random.seed(5)
        np.random.seed(5)
        # Declare some parameters
        batch_size = 16
        learning_rate = 1e-4
        data_to_retrieve = ['left-eye', 'right-eye', 'face', 'face-landmarks', 'gaze', 'head']  # Available are: left-eye
                                                         #                right-eye
                                                         #                eye-region
                                                         #                face
                                                         #                gaze (required)
                                                         #                head
                                                         #                face-landmarks

        # Define model
        from datasources import HDF5Source
        from models.VGG16 import NewModel
        model = NewModel(
            # Tensorflow session
            # Note: The same session must be used for the model and the data sources.
            session,

            # The learning schedule describes in which order which loss term to optimize
            # at which learning rate.
            #
            # A standard network would have one entry (dict) in this argument where all model
            # parameters are optimized. To do this, you must specify which variables must be
            # optimized and this is done by specifying which prefixes to look for.
            # The prefixes are defined by using `tf.variable_scope`.
            #
            # The loss terms which can be specified depends on model specifications, specifically
            # the `loss_terms` output of `BaseModel::build_model`.
            learning_schedule=[
                {
                    'loss_terms_to_optimize': ['gaze_mse'],
                    'metrics': ['gaze_angular'],
                    'learning_rate': learning_rate,
                },
            ],

            test_losses_or_metrics=['gaze_mse', 'gaze_angular'],

            # Data sources for training and testing.
            train_data={
                'real': HDF5Source(
                    session,
                    batch_size,
                    hdf_path='/cluster/project/infk/hilliges/lectures/mp20/project3/mp20_train.h5',
                    entries_to_use=data_to_retrieve,
                    min_after_dequeue=2000,
                    data_format='NHWC',
                    shuffle=True,
                    staging=True,
                    use_colour= True
                ),
            },
            test_data={
                'real': HDF5Source(
                    session,
                    batch_size,
                    hdf_path='/cluster/project/infk/hilliges/lectures/mp20/project3/mp20_validation.h5',
                    entries_to_use=data_to_retrieve,
                    testing=True,
                    num_threads=2,
                    data_format='NHWC',
                    use_colour=True
                ),
            },

            # We may use a user-defined identifier to resume previous training or
            # perform predictions on a pre-trained model.
            identifier=identifier,
        )

        if args.restore is None:
            # Train this model for a set number of epochs
            model.train(
                # original 20
                num_epochs = 25
            )

        # Evaluate for submission
        model.evaluate_for_submission(
            HDF5Source(
                session,
                batch_size,
                hdf_path='/cluster/project/infk/hilliges/lectures/mp20/project3/mp20_test_students.h5',
                entries_to_use=[k for k in data_to_retrieve if k != 'gaze'],
                testing=True,
                num_threads=1,
                data_format='NHWC',
                use_colour=True
            )
        )

        del model

