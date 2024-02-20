import os

import tensorflow as tf
from tensorflow.python.framework import tensor_util

summary_dir = '/scratch/bae9wk/magvit/workdir'
record_path = '/scratch/bae9wk/magvit/workdir/checkpoint_105520'

from flax.training import checkpoints

vq_checkpoint_path = record_path
vq_train_state = checkpoints.restore_checkpoint(vq_checkpoint_path, None)
print(vq_train_state.keys())

#'workdir/events.out.tfevents.1699479805.udc-an36-25.891926.0.v2'
#'workdir/events.out.tfevents.1705692517.udc-an36-19.664773.0.v2'
#'/scratch/bae9wk/magvit/workdir/events.out.tfevents.1705942498.udc-an34-13.742936.0.v2'

# from tensorflow.core.util import event_pb2
# from tensorflow.python.lib.io import tf_record
# from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file

#print_tensors_in_checkpoint_file(record_path, '', True, all_tensor_names=True)
#from tf.io.gfile import Gfile



# def my_summary_iterator(path):
#     for r in tf_record.tf_record_iterator(path):
#         yield event_pb2.Event.FromString(r)

# for filename in os.listdir(summary_dir):
#     path = os.path.join(summary_dir, filename)
#     #record = open(record_path, 'rb').read()
#     #with event_pb2.Event.FromString(record) as event:
#     for event in my_summary_iterator(record_path):
#         for value in event.summary.value:
#             t = tensor_util.MakeNdarray(value.tensor)
#             print(value.tag, event.step, t, type(t))
