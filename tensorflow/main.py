import argparse
import json
import logging
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import mixed_precision


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)06d: %(levelname).1s %(pathname)s:%(lineno)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def add_parser_arguments(parser):
    parser.add_argument(
        "--epochs",
        default=10,
        type=int,
        metavar="N",
        help="number of total epochs (default: 10) to run",
    )
    parser.add_argument(
        "--steps",
        default=70,
        type=int,
        metavar="N",
        help="number of steps (default: 70) per epochs to run",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        default=256,
        type=int,
        metavar="N",
        help="mini-batch size (default: 256) per worker",
    )
    parser.add_argument(
        "--lr",
        "--learning-rate",
        default=0.001,
        type=float,
        metavar="LR",
        help="learning rate",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Run model in mixed precision mode.",
    )
    parser.add_argument(
        "--collective-communication",
        type=str,
        default="auto",
        choices=["auto", "nccl", "ring"],
        help="collective communication strategy for workers.",
    )


def mnist_dataset(batch_size):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    train_dataset = (
        tf.data.Dataset.from_tensor_slices((x_train, y_train))
        .repeat()
        .batch(batch_size)
    )
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(
        batch_size
    )
    return train_dataset, test_dataset


def build_and_compile_cnn_model(learning_rate):
    inputs = tf.keras.Input(shape=(28, 28, 1))
    x = tf.keras.layers.Resizing(32, 32)(inputs)
    x = tf.cast(x, tf.int32)
    outputs = tf.keras.applications.efficientnet_v2.EfficientNetV2S(
        include_top=False,
        weights=None,
        input_shape=(32, 32, 1),
        classes=10,
        include_preprocessing=True,
    )(x)

    model = tf.keras.Model([inputs], [outputs])
    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate),
        metrics=["accuracy"],
    )
    return model


def main():
    tf_config = json.loads(os.environ["TF_CONFIG"])
    num_workers = len(tf_config["cluster"]["worker"])
    logging.info(tf_config)

    parser = argparse.ArgumentParser(
        description="TensorFlow ImageNet Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_parser_arguments(parser)
    args, rest = parser.parse_known_args()

    if args.amp:
        policy = mixed_precision.Policy("mixed_float16")
        mixed_precision.set_global_policy(policy)

    else:
        policy = mixed_precision.Policy("float32")
        mixed_precision.set_global_policy(policy)

    logging.info("Compute dtype: %s" % policy.compute_dtype)
    logging.info("Variable dtype: %s" % policy.variable_dtype)

    # Currently, this is the only strategy which supports NCCL, RING, and AUTO.
    strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy(
        cluster_resolver=tf.distribute.cluster_resolver.TFConfigClusterResolver(),
        communication=args.collective_communication,
    )
    global_batch_size = args.batch_size * num_workers
    multi_worker_train_dataset, multi_worker_test_dataset = mnist_dataset(
        global_batch_size
    )
    with strategy.scope():
        multi_worker_model = build_and_compile_cnn_model(args.lr)

    multi_worker_model.fit(
        multi_worker_train_dataset,
        validation_data=multi_worker_test_dataset,
        epochs=args.epochs,
        steps_per_epoch=args.steps,
    )


if __name__ == "__main__":
    main()
