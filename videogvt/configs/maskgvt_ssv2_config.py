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
r"""Configuration and hyperparameter for the MaskGVT on SSv2 frame prediction.

"""

## TO DO

# Cond frames?
# Checkpoint saving?


import ml_collections
from videogvt.configs import maskgvt_ucf101_config


SSV2_TRAIN_SIZE = 168913
SSV2_VAL_SIZE = 24777
SSV2_TEST_SIZE = 27157
VARIANT = 'MaskGVT/16'


def get_config(config_str='B'):
  """Get the base hyperparameter configuration."""
  version, *options = config_str.split('-')

  config = maskgvt_ucf101_config.get_config(config_str)
  config.experiment_name = f'SSV2_FP_{VARIANT}'

  if 'runlocal' in options:
    config.num_training_epochs = 1

  # Dataset.
  config.dataset_name = 'video_tfrecord_dataset'
  config.dataset_configs = ml_collections.ConfigDict()
  config.dataset_configs.base_dir = '../ssv2'
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

  # Model: MaskGVT
  config.tasks = ('frame_prediction',)
  config.frame_prediction = ml_collections.ConfigDict()
  config.frame_prediction.cond_frames = 1
  config.frame_prediction.cond_latent_frames = 1
  config.frame_prediction.condition_mode = 'cond->input'
  config.frame_prediction.weight_mode = 'mask+refine+recons'

  # VQ Model
  from videogvt.configs import vqgan3d_ssv2_config
  config.vq_model_from.config = vqgan3d_ssv2_config.get_config(f'{version}-eval')
  config.vq_model_from.checkpoint_path = None

  # Learning rate
  steps_per_epoch = SSV2_TRAIN_SIZE // config.get_ref('batch_size')
  config.lr_configs.steps_per_epoch = steps_per_epoch
  total_steps = config.get_ref('num_training_epochs') * steps_per_epoch
  config.lr_configs.steps_per_cycle = total_steps

  config.logging.enable_checkpoint = False

  # Evaluation.
  config.eval.enable_inception_score = False
  config.eval.enable_frechet_distance = True

  config.eval.data_splits = 'train,validation'
  config.eval.num_examples = SSV2_VAL_SIZE * 10
  config.eval.final_num_repeats = 4
  config.eval.final_num_example_multiplier = 10

  # Standalone evaluation.
  if 'eval' in options:
    config.eval_only = True
    config.eval.data_splits = 'validation'
    config.eval_from.checkpoint_path = None#{
    #     'B': 'gs://magvit/models/bair_gvt_base_fp1',
    #     'L': 'gs://magvit/models/bair_gvt_large_fp1',
    # }[version]
    config.eval_from.step = -1

    if 'single' in options:
      config.dataset_configs.num_eval_clips = 1
      config.sampling.mask_bins = 12
      config.sampling.choice_temperature = 400.
      config.sampling.mask_scheduling_method = 'exp'

    config.eval.enable_ssim_psnr = True
    config.eval.enable_lpips = True
    config.eval.results_with_input = True
    config.eval.results_with_condition = False
    config.eval.results_dir = None

  config.init_from = None
  return config
