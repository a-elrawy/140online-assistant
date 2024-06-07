# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

"""
# Training the model

Basic run (on CPU for 50 epochs):
    python examples/asr/asr_ctc/speech_to_text_ctc.py \
        # (Optional: --config-path=<path to dir of configs> --config-name=<name of config without .yaml>) \
        model.train_ds.manifest_filepath="<path to manifest file>" \
        model.validation_ds.manifest_filepath="<path to manifest file>" \
        trainer.devices=1 \
        trainer.accelerator='cpu' \
        trainer.max_epochs=50


Add PyTorch Lightning Trainer arguments from CLI:
    python speech_to_text_ctc.py \
        ... \
        +trainer.fast_dev_run=true

Hydra logs will be found in "$(./outputs/$(date +"%y-%m-%d")/$(date +"%H-%M-%S")/.hydra)"
PTL logs will be found in "$(./outputs/$(date +"%y-%m-%d")/$(date +"%H-%M-%S")/lightning_logs)"

Override some args of optimizer:
    python speech_to_text_ctc.py \
    # (Optional: --config-path=<path to dir of configs> --config-name=<name of config without .yaml>) \
    model.train_ds.manifest_filepath="./an4/train_manifest.json" \
    model.validation_ds.manifest_filepath="./an4/test_manifest.json" \
    trainer.devices=2 \
    trainer.max_epochs=2 \
    model.optim.args.betas=[0.8,0.5] \
    model.optim.args.weight_decay=0.0001

Override optimizer entirely
    python speech_to_text_ctc.py \
    # (Optional: --config-path=<path to dir of configs> --config-name=<name of config without .yaml>) \
    model.train_ds.manifest_filepath="./an4/train_manifest.json" \
    model.validation_ds.manifest_filepath="./an4/test_manifest.json" \
    trainer.devices=2 \
    trainer.max_epochs=2 \
    model.optim.name=adamw \
    model.optim.lr=0.001 \
    ~model.optim.args \
    +model.optim.args.betas=[0.8,0.5]\
    +model.optim.args.weight_decay=0.0005

# Fine-tune a model

For documentation on fine-tuning this model, please visit -
https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/asr/configs.html#fine-tuning-configurations

# Pretrained Models

For documentation on existing pretrained models, please visit -
https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/asr/results.html

"""

import pytorch_lightning as pl
from nemo.collections.asr.parts.submodules.ctc_decoding import CTCDecoding, CTCDecodingConfig
# from nemo.collections.asr.parts.submodules.ctc_decoding import CTCDecoding, CTCDecodingConfig
import nemo.collections.asr as nemo_asr

from nemo.collections.asr.models import EncDecCTCModel
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager
from nemo.collections.asr.metrics.wer import WER


from omegaconf import DictConfig, OmegaConf, open_dict
from pyctcdecode import build_ctcdecoder


@hydra_runner(config_path="../conf", config_name="config")
def main(cfg):
    logging.info(f'Hydra config: {OmegaConf.to_yaml(cfg)}')

    trainer = pl.Trainer(**cfg.trainer)
    exp_manager(trainer, cfg.get("exp_manager", None))
    asr_model = EncDecCTCModel(cfg=cfg.model, trainer=trainer)

    # Initialize the weights of the model from another model, if provided via config
    asr_model.maybe_init_from_pretrained_checkpoint(cfg)

    score_dict = {}
    alphas = [0.1]

    for alpha in alphas:
        ctc_decoding: CTCDecodingConfig = CTCDecodingConfig()
        ctc_decoding.strategy = 'pyctcdecode'
        # ctc_decoding.beam.kenlm_path = 'lm/pure_mgb2.arpa'
        # ctc_decoding.beam.kenlm_path = 'lm/pure_140.arpa'
        ctc_decoding.beam.kenlm_path = f'lm/final{alpha}.arpa'
        ctc_decoding.beam.beam_size = 20
        ctc_decoding.beam.beam_alpha = 0.09
        ctc_decoding.beam.beam_beta = 1.05

        decoding_cls = OmegaConf.structured(CTCDecodingConfig)
        decoding_cls = OmegaConf.create(OmegaConf.to_container(decoding_cls))
        decoding_cfg = OmegaConf.merge(decoding_cls, ctc_decoding)
        with open_dict(asr_model.cfg):
                        asr_model.cfg.decoding = decoding_cfg


        asr_model.decoding = CTCDecoding(
                        decoding_cfg=asr_model.cfg.decoding, vocabulary=asr_model.decoding.vocabulary)

        asr_model.wer = WER(
                    decoding=asr_model.decoding,
                    use_cer=asr_model._cfg.get('use_cer', False),
                    dist_sync_on_step=True,
                    log_prediction=asr_model._cfg.get("log_prediction", False),
                )
    

        if hasattr(cfg.model, 'test_ds') and cfg.model.test_ds.manifest_filepath is not None:
            if asr_model.prepare_test(trainer):
                metrics = trainer.test(asr_model)

        score_dict[alpha] = metrics[0]
        # asr_model.wer.write(f'result0.{alpha}.txt')

    for alpha in alphas:
        print(alpha, " : ", score_dict[alpha])

if __name__ == '__main__':
    main()  # noqa pylint: disable=no-value-for-parameter

