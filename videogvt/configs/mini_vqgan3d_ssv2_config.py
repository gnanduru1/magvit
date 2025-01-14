# Copyright 2023 The videogvt Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: disable=line-too-long
r"""Configs for the VQGAN-3D on Something-Something-v2.

"""

import ml_collections

from videogvt.configs import vqgan3d_ucf101_config

SSV2_TRAIN_SIZE = 168913
SSV2_VAL_SIZE = 24777
SSV2_TEST_SIZE = 27157
VARIANT = 'VQGAN/3D'


def get_config(config_str='B'):
  """Returns the base experiment configuration."""
  version, *options = config_str.split('-')

  config = vqgan3d_ucf101_config.get_config(config_str)
  config.experiment_name = f'SSV2_{VARIANT}'

  # Overall
  config.image_size = 224
  #config.num_training_epochs = {'B': 135, 'L': 400}[version]
  config.num_training_epochs = {'B': 80}[version]
  config.batch_size = {'B': 128}[version]

  # Dataset.
  config.dataset_name = 'video_tfrecord_dataset'
  config.dataset_configs = ml_collections.ConfigDict()
  config.dataset_configs.base_dir = f'../mini_ssv2'
  config.dataset_configs.tables = {
      'train': 'train_tfrecord',
      'validation': 'val_tfrecord',
      'test': 'test_tfrecord'
  }
  config.dataset_configs.examples_per_subset = {
      'train': SSV2_TRAIN_SIZE,
      'validation': SSV2_VAL_SIZE,
      'test': SSV2_VAL_SIZE
  }

  config.dataset_configs.camera_name = 'image_aux1'
  config.dataset_configs.num_classes = 174
  config.dataset_configs.frame_rate = 10
  config.dataset_configs.num_frames = 4
  config.dataset_configs.stride = 1
  config.dataset_configs.zero_centering = False  # Range is 0 to 1
  config.dataset_configs.num_eval_clips = 15  # Sample 16 frames out of 30
  config.dataset_configs.shuffle_buffer_size = 8 * config.get_ref('batch_size')
  config.dataset_configs.prefetch_to_device = 2

  # Model: vqvae
  config.vqvae.channel_multipliers = (1, 2, 4)
  # config.vqvae.custom_conv_padding = 'constant'
  config.discriminator.channel_multipliers = (2, 4, 4, 4)

  # Learning rate
  steps_per_epoch = SSV2_TRAIN_SIZE // config.get_ref('batch_size')
  config.lr_configs.steps_per_epoch = steps_per_epoch
  total_steps = config.get_ref('num_training_epochs') * steps_per_epoch
  config.lr_configs.warmup_steps = 1 * steps_per_epoch
  config.lr_configs.steps_per_cycle = total_steps

  config.init_from.checkpoint_path = None

  config.init_from = None
  config.logging.enable_checkpoint = True

  # Evaluation.
  config.eval.enable_inception_score = False
  config.eval.num_examples = SSV2_VAL_SIZE * 10
  config.eval.final_num_repeats = 1
  config.eval.final_num_example_multiplier = 10

  # Standalone evaluation.
  if 'eval' in options:
    config.eval_only = True
    config.eval_from.checkpoint_path = None
    # {
    #     'B': 'gs://magvit/models/bair_3d_base',
    #     'L': 'gs://magvit/models/bair_3d_large',
    # }[version]
    config.eval_from.step = -1
    config.eval_from.legacy_checkpoint = False

  return config
