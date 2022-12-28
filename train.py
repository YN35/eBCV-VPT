import uuid
import argparse, os, sys, datetime, glob, importlib, csv
import numpy as np
import time
import torch
import torchvision
import pytorch_lightning as pl

from packaging import version
from omegaconf import OmegaConf
from torch.utils.data import random_split, DataLoader, Dataset, Subset
from functools import partial
from PIL import Image

from pytorch_lightning import seed_everything
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, Callback, LearningRateMonitor
from pytorch_lightning.utilities.distributed import rank_zero_only
from pytorch_lightning.utilities import rank_zero_info

from lib.utils import instantiate_from_config


def get_parser(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="paths to base configs. Loaded from left-to-right. "
             "Parameters can be overwritten or added with command-line options of the form `--key value`.",
    )
    return parser


class SetupCallback(Callback):
    def __init__(self, resume, now, logdir, ckptdir, cfgdir, config, lightning_config):
        super().__init__()
        self.resume = resume
        self.now = now
        self.logdir = logdir
        self.ckptdir = ckptdir
        self.cfgdir = cfgdir
        self.config = config
        self.lightning_config = lightning_config

    def on_keyboard_interrupt(self, trainer, pl_module):
        if trainer.global_rank == 0:
            print("Summoning checkpoint.")
            ckpt_path = os.path.join(self.ckptdir, "last.ckpt")
            trainer.save_checkpoint(ckpt_path)

    def on_pretrain_routine_start(self, trainer, pl_module):
        if trainer.global_rank == 0:
            # Create logdirs and save configs
            os.makedirs(self.logdir, exist_ok=True)
            os.makedirs(self.ckptdir, exist_ok=True)
            os.makedirs(self.cfgdir, exist_ok=True)

            if "callbacks" in self.lightning_config:
                if 'metrics_over_trainsteps_checkpoint' in self.lightning_config['callbacks']:
                    os.makedirs(os.path.join(self.ckptdir, 'trainstep_checkpoints'), exist_ok=True)
            print("Project config")
            print(OmegaConf.to_yaml(self.config))
            OmegaConf.save(self.config,
                           os.path.join(self.cfgdir, "{}-project.yaml".format(self.now)))

            print("Lightning config")
            print(OmegaConf.to_yaml(self.lightning_config))
            OmegaConf.save(OmegaConf.create({"lightning": self.lightning_config}),
                           os.path.join(self.cfgdir, "{}-lightning.yaml".format(self.now)))

        else:
            # ModelCheckpoint callback created log directory --- remove it
            if not self.resume and os.path.exists(self.logdir):
                dst, name = os.path.split(self.logdir)
                dst = os.path.join(dst, "child_runs", name)
                os.makedirs(os.path.split(dst)[0], exist_ok=True)
                try:
                    os.rename(self.logdir, dst)
                except FileNotFoundError:
                    pass


class CUDACallback(Callback):
    # see https://github.com/SeanNaren/minGPT/blob/master/mingpt/callback.py
    def on_train_epoch_start(self, trainer, pl_module):
        # Reset the memory use counter
        torch.cuda.reset_peak_memory_stats(trainer.root_gpu)
        torch.cuda.synchronize(trainer.root_gpu)
        self.start_time = time.time()

    def on_train_epoch_end(self, trainer, pl_module):
        torch.cuda.synchronize(trainer.root_gpu)
        max_memory = torch.cuda.max_memory_allocated(trainer.root_gpu) / 2 ** 20
        epoch_time = time.time() - self.start_time

        try:
            max_memory = trainer.training_type_plugin.reduce(max_memory)
            epoch_time = trainer.training_type_plugin.reduce(epoch_time)

            rank_zero_info(f"Average Epoch time: {epoch_time:.2f} seconds")
            rank_zero_info(f"Average Peak memory {max_memory:.2f}MiB")
        except AttributeError:
            pass


if __name__ == "__main__":
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

    # add cwd for convenience and to make classes in this file available when
    # running as `python train.py`
    # (in particular `train.DataModuleFromConfig`)
    sys.path.append(os.getcwd())

    parser = get_parser()
    parser = Trainer.add_argparse_args(parser)

    opt, unknown = parser.parse_known_args()
    
    cfg = OmegaConf.load(opt.config)
    opt = cfg.pop("opt", OmegaConf.create())
    
    resume_path = opt.modelprm
    if not opt.mode == "train":
        if not os.path.exists(resume_path):
            raise ValueError("Cannot find {}".format(resume_path))
        if os.path.isfile(resume_path):
            paths = resume_path.split("/")
            opt.resume_from_checkpoint = resume_path
        else:
            raise ValueError("Cannot find {}".format(resume_path))
        
    if opt.name:
        name = "_" + opt.name
    else:
        name = "_" + opt.config.split('/')[-1].split('.')[0]
    nowname = now + name
    logdir = os.path.join(opt.logdir, nowname)

    model_dir = os.path.join(logdir, 'model')
    ckptdir = os.path.join(logdir, "checkpoints")
    cfgdir = os.path.join(logdir, "configs")

    seed_everything(opt.seed)

    #try:
    # init and save configs
    lightning_cfg = cfg.pop("lightning", OmegaConf.create())
    # merge trainer cli with config
    trainer_cfg = lightning_cfg.get("trainer", OmegaConf.create())
    # default to ddp
    trainer_cfg['gpus'] = opt.gpus
    if trainer_cfg['gpus']:
        gpuinfo = trainer_cfg["gpus"]
        print(f"Running on GPUs {gpuinfo}")
        cpu = False
        trainer_cfg["accelerator"] = "gpu"
    else:
        print(f"Running on CPU")
        cpu = True
    trainer_opt = argparse.Namespace(**trainer_cfg)
    lightning_cfg.trainer = trainer_cfg

    # model
    model = instantiate_from_config(cfg.model)

    # trainer and callbacks
    trainer_kwargs = dict()

    # default logger configs
    default_logger_cfgs = {
        "wandb": {
            "target": "pytorch_lightning.loggers.WandbLogger",
            "params": {
                "name": nowname,
                "save_dir": logdir,
                "offline": opt.logoffline,
                "id": str(uuid.uuid1()),
            }
        },
        "testtube": {
            "target": "pytorch_lightning.loggers.TestTubeLogger",
            "params": {
                "name": "testtube",
                "save_dir": logdir,
            }
        },
    }
    default_logger_cfg = default_logger_cfgs["wandb"]
    if "logger" in lightning_cfg:
        logger_cfg = lightning_cfg.logger
    else:
        logger_cfg = OmegaConf.create()
    logger_cfg = OmegaConf.merge(default_logger_cfg, logger_cfg)
    trainer_kwargs["logger"] = instantiate_from_config(logger_cfg)

    # modelcheckpoint - use TrainResult/EvalResult(checkpoint_on=metric) to
    # specify which metric is used to determine best models
    default_modelckpt_cfg = {
        "target": "pytorch_lightning.callbacks.ModelCheckpoint",
        "params": {
            "dirpath": ckptdir,
            "filename": "{epoch:06}",
            "verbose": True,
            "save_last": True,
        }
    }
    if hasattr(model, "monitor"):
        print(f"Monitoring {model.monitor} as checkpoint metric.")
        default_modelckpt_cfg["params"]["monitor"] = model.monitor
        default_modelckpt_cfg["params"]["save_top_k"] = 3

    if "modelcheckpoint" in lightning_cfg:
        modelckpt_cfg = lightning_cfg.modelcheckpoint
    else:
        modelckpt_cfg =  OmegaConf.create()
    modelckpt_cfg = OmegaConf.merge(default_modelckpt_cfg, modelckpt_cfg)
    print(f"Merged modelckpt-cfg: \n{modelckpt_cfg}")
    if version.parse(pl.__version__) < version.parse('1.4.0'):
        trainer_kwargs["checkpoint_callback"] = instantiate_from_config(modelckpt_cfg)

    # add callback which sets up log directory
    default_callbacks_cfg = {
        "setup_callback": {
            "target": "train.SetupCallback",
            "params": {
                "resume": resume_path,
                "now": now,
                "logdir": logdir,
                "ckptdir": ckptdir,
                "cfgdir": cfgdir,
                "config": cfg,
                "lightning_config": lightning_cfg,
            }
        },
        "learning_rate_logger": {
            "target": "train.LearningRateMonitor",
            "params": {
                "logging_interval": "step",
                # "log_momentum": True
            }
        },
        "cuda_callback": {
            "target": "train.CUDACallback"
        },
    }
    if version.parse(pl.__version__) >= version.parse('1.4.0'):
        default_callbacks_cfg.update({'checkpoint_callback': modelckpt_cfg})

    if "callbacks" in lightning_cfg:
        callbacks_cfg = lightning_cfg.callbacks
    else:
        callbacks_cfg = OmegaConf.create()

    if 'metrics_over_trainsteps_checkpoint' in callbacks_cfg:
        print(
            'Caution: Saving checkpoints every n train steps without deleting. This might require some free space.')
        default_metrics_over_trainsteps_ckpt_dict = {
            'metrics_over_trainsteps_checkpoint':
                {"target": 'pytorch_lightning.callbacks.ModelCheckpoint',
                    'params': {
                        "dirpath": os.path.join(ckptdir, 'trainstep_checkpoints'),
                        "filename": "{epoch:06}-{step:09}",
                        "verbose": True,
                        'save_top_k': -1,
                        'every_n_train_steps': 10000,
                        'save_weights_only': True
                    }
                    }
        }
        default_callbacks_cfg.update(default_metrics_over_trainsteps_ckpt_dict)

    callbacks_cfg = OmegaConf.merge(default_callbacks_cfg, callbacks_cfg)
    if 'ignore_keys_callback' in callbacks_cfg and hasattr(trainer_opt, 'resume_from_checkpoint'):
        callbacks_cfg.ignore_keys_callback.params['ckpt_path'] = trainer_opt.resume_from_checkpoint
    elif 'ignore_keys_callback' in callbacks_cfg:
        del callbacks_cfg['ignore_keys_callback']

    trainer_kwargs["callbacks"] = [instantiate_from_config(callbacks_cfg[k]) for k in callbacks_cfg]
    trainer_kwargs["plugins"] = list()
    from pytorch_lightning.plugins import DDPPlugin, NativeMixedPrecisionPlugin
    #trainer_kwargs["plugins"].append(DDPPlugin(find_unused_parameters=False))
    trainer_kwargs["plugins"].append(NativeMixedPrecisionPlugin(16, 'cuda', torch.cuda.amp.GradScaler(enabled=opt.fp16)))
    trainer = Trainer.from_argparse_args(trainer_opt, **trainer_kwargs)
    #trainer = Trainer(gpus=1, precision=16, amp_backend="native", strategy="deepspeed_stage_2_offload", benchmark=True, limit_val_batches=0, num_sanity_val_steps=0, accumulate_grad_batches=1)
    trainer.logdir = logdir  ###

    # data
    data = instantiate_from_config(cfg.data)
    # NOTE according to https://pytorch-lightning.readthedocs.io/en/latest/datamodules.html
    # calling these ourselves should not be necessary but it is.
    # lightning still takes care of proper multiprocessing though
    data.prepare_data()
    data.setup()
    print("#### Data #####")
    print(f"test, {data._dataset(False).__class__.__name__}, {len(data._dataset(False))}")
    print(f"train, {data._dataset(True).__class__.__name__}, {len(data._dataset(True))}")

    # # configure learning rate
    # bs, base_lr = cfg.data.params.batch_size, cfg.model.base_learning_rate
    # if not cpu:
    #     ngpu = len(lightning_cfg.trainer.gpus.strip(",").split(','))
    # else:
    #     ngpu = 1
    # if 'accumulate_grad_batches' in lightning_cfg.trainer:
    #     accumulate_grad_batches = lightning_cfg.trainer.accumulate_grad_batches
    # else:
    #     accumulate_grad_batches = 1
    # print(f"accumulate_grad_batches = {accumulate_grad_batches}")
    # lightning_cfg.trainer.accumulate_grad_batches = accumulate_grad_batches
    # if opt.scale_lr:
    #     model.learning_rate = accumulate_grad_batches * ngpu * bs * base_lr
    #     print(
    #         "Setting learning rate to {:.2e} = {} (accumulate_grad_batches) * {} (num_gpus) * {} (batchsize) * {:.2e} (base_lr)".format(
    #             model.learning_rate, accumulate_grad_batches, ngpu, bs, base_lr))
    # else:
    #     model.learning_rate = base_lr
    #     print("++++ NOT USING LR SCALING ++++")
    #     print(f"Setting learning rate to {model.learning_rate:.2e}")


    # allow checkpointing via USR1
    def melk(*args, **kwargs):
        # run all checkpoint hooks
        if trainer.global_rank == 0:
            print("Summoning checkpoint.")
            ckpt_path = os.path.join(ckptdir, "last.ckpt")
            trainer.save_checkpoint(ckpt_path)


    def divein(*args, **kwargs):
        if trainer.global_rank == 0:
            import pudb;
            pudb.set_trace()


    import signal

    signal.signal(signal.SIGUSR1, melk)
    signal.signal(signal.SIGUSR2, divein)

    # run
    print('########## Starting training ##########')
    print('press Ctrl+C to save checkpoint and end training')
    if opt.mode == "train" or opt.mode == "finetune":
        try:
            trainer.fit(model, data)
        except Exception:
            melk()
            raise
    # 学習終了かつ途中終了されなかった場合
    if not opt.no_test and not trainer.interrupted:
        trainer.test(model, data)
    # except Exception:
    #     if opt.debug and trainer.global_rank == 0:
    #         try:
    #             import pudb as debugger
    #         except ImportError:
    #             import pdb as debugger
    #         debugger.post_mortem()
    #     raise
    #finally:
    # move newly created debug project to debug_runs
    if opt.debug and opt.mode == "train" and trainer.global_rank == 0:
        dst, name = os.path.split(logdir)
        dst = os.path.join(dst, "debug_runs", name)
        os.makedirs(os.path.split(dst)[0], exist_ok=True)
        os.rename(logdir, dst)
    if trainer.global_rank == 0:
        print(trainer.profiler.summary())