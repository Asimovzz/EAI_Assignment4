if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import contextlib
import copy
import os
import pathlib
import random
import re

import hydra
import numpy as np
import torch
import torch.distributed as dist
import tqdm
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.common.pytorch_util import optimizer_to
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.model.common.lr_scheduler import get_scheduler
from diffusion_policy.model.diffusion.ema_model import EMAModel
from diffusion_policy.policy.diffusion_unet_image_policy import DiffusionUnetImagePolicy
from diffusion_policy.workspace.base_workspace import BaseWorkspace

OmegaConf.register_new_resolver("eval", eval, replace=True)


class _NullLogger:

    def log(self, *_args, **_kwargs):
        return None


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model


class RobotWorkspace(BaseWorkspace):
    include_keys = ["global_step", "epoch"]
    exclude_keys = ("model_ddp",)

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)

        self.distributed = False
        self.rank = 0
        self.local_rank = 0
        self.world_size = 1
        self.device = None
        self.model_ddp = None

        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # configure model
        self.model: DiffusionUnetImagePolicy = hydra.utils.instantiate(cfg.policy)

        self.ema_model: DiffusionUnetImagePolicy = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)

        # configure training state
        self.optimizer = hydra.utils.instantiate(cfg.optimizer, params=self.model.parameters())

        # configure training state
        self.global_step = 0
        self.epoch = 0

    @property
    def is_main_process(self):
        return self.rank == 0

    def _setup_distributed(self, cfg: OmegaConf):
        use_ddp = bool(cfg.training.get("use_ddp", False))
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        self.distributed = use_ddp and world_size > 1

        if self.distributed:
            self.rank = int(os.environ["RANK"])
            self.local_rank = int(os.environ["LOCAL_RANK"])
            self.world_size = world_size
            dist.init_process_group(backend=cfg.training.get("ddp_backend", "nccl"))
        else:
            self.rank = 0
            self.local_rank = 0
            self.world_size = 1

        seed = int(cfg.training.seed) + self.rank
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        if torch.cuda.is_available():
            torch.cuda.set_device(self.local_rank)
            self.device = torch.device(f"cuda:{self.local_rank}")
        else:
            self.device = torch.device("cpu")

    def _teardown_distributed(self):
        if self.distributed and dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()

    def _reduce_mean(self, value: torch.Tensor):
        if not self.distributed:
            return value
        reduced = value.detach().clone()
        dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
        reduced /= self.world_size
        return reduced

    def run(self):
        cfg = copy.deepcopy(self.cfg)
        seed = cfg.training.seed

        self._setup_distributed(cfg)
        try:
            # configure dataset
            dataset: BaseImageDataset
            dataset = hydra.utils.instantiate(cfg.task.dataset, zarr_path=str(cfg.task.dataset.zarr_path))
            assert isinstance(dataset, BaseImageDataset)
            train_dataloader, train_batch_sampler = create_dataloader(
                dataset,
                distributed=self.distributed,
                rank=self.rank,
                world_size=self.world_size,
                **cfg.dataloader,
            )
            normalizer = dataset.get_normalizer()

            # configure validation dataset
            val_dataset = dataset.get_validation_dataset()
            val_dataloader = None
            if self.is_main_process:
                val_dataloader, _ = create_dataloader(
                    val_dataset,
                    distributed=False,
                    rank=0,
                    world_size=1,
                    **cfg.val_dataloader,
                )

            # resume after dataset construction so current config paths are used.
            if cfg.training.resume:
                resume_ckpt_path = cfg.training.get("resume_ckpt_path", None)
                lastest_ckpt_path = pathlib.Path(resume_ckpt_path) if resume_ckpt_path else self.get_checkpoint_path()
                if not lastest_ckpt_path.is_file():
                    raise FileNotFoundError(f"Resume checkpoint not found: {lastest_ckpt_path}")
                if self.is_main_process:
                    print(f"Resuming from checkpoint {lastest_ckpt_path}")
                self.load_checkpoint(path=lastest_ckpt_path, map_location="cpu")
                if self.is_main_process:
                    print(f"Resumed state: epoch={self.epoch}, global_step={self.global_step}")

            self.model.set_normalizer(normalizer)
            if cfg.training.use_ema:
                self.ema_model.set_normalizer(normalizer)

            # device transfer
            self.model.to(self.device)
            if self.ema_model is not None:
                self.ema_model.to(self.device)
            optimizer_to(self.optimizer, self.device)

            if self.distributed:
                self.model_ddp = DDP(
                    self.model,
                    device_ids=[self.local_rank],
                    output_device=self.local_rank,
                    find_unused_parameters=bool(cfg.training.get("find_unused_parameters", False)),
                )
                if self.ema_model is not None and not cfg.training.resume:
                    self.ema_model.load_state_dict(self.model.state_dict())
            train_model = self.model_ddp if self.model_ddp is not None else self.model

            # configure lr scheduler
            lr_scheduler = get_scheduler(
                cfg.training.lr_scheduler,
                optimizer=self.optimizer,
                num_warmup_steps=cfg.training.lr_warmup_steps,
                num_training_steps=(len(train_dataloader) * cfg.training.num_epochs)
                // cfg.training.gradient_accumulate_every,
                # pytorch assumes stepping LRScheduler every epoch
                # however huggingface diffusers steps it every batch
                last_epoch=self.global_step - 1,
            )

            # configure ema
            ema: EMAModel = None
            if cfg.training.use_ema:
                ema = hydra.utils.instantiate(cfg.ema, model=self.ema_model)

            # configure checkpoint
            if self.is_main_process:
                _ = TopKCheckpointManager(
                    save_dir=os.path.join(self.output_dir, "checkpoints"),
                    **cfg.checkpoint.topk,
                )

            # save batch for sampling
            train_sampling_batch = None

            if cfg.training.debug:
                cfg.training.num_epochs = 2
                cfg.training.max_train_steps = 3
                cfg.training.max_val_steps = 3
                cfg.training.rollout_every = 1
                cfg.training.checkpoint_every = 1
                cfg.training.val_every = 1
                cfg.training.sample_every = 1

            # training loop
            log_path = os.path.join(self.output_dir, "logs.json.txt")
            logger_cm = JsonLogger(log_path) if self.is_main_process else contextlib.nullcontext(_NullLogger())

            with logger_cm as json_logger:
                while self.epoch < cfg.training.num_epochs:
                    if hasattr(train_batch_sampler, "set_epoch"):
                        train_batch_sampler.set_epoch(self.epoch)

                    step_log = dict()

                    # ========= train for this epoch ==========
                    if cfg.training.freeze_encoder:
                        self.model.obs_encoder.eval()
                        self.model.obs_encoder.requires_grad_(False)

                    train_losses = []
                    self.optimizer.zero_grad()
                    with tqdm.tqdm(
                        train_dataloader,
                        desc=f"Training epoch {self.epoch} [rank {self.rank}]",
                        leave=False,
                        mininterval=cfg.training.tqdm_interval_sec,
                        disable=not self.is_main_process,
                    ) as tepoch:
                        for batch_idx, batch in enumerate(tepoch):
                            batch = dataset.postprocess(batch, self.device)
                            if train_sampling_batch is None:
                                train_sampling_batch = batch

                            raw_loss = train_model(batch)
                            loss = raw_loss / cfg.training.gradient_accumulate_every
                            loss.backward()

                            is_last_batch = batch_idx == (len(train_dataloader) - 1)
                            should_step = ((batch_idx + 1) % cfg.training.gradient_accumulate_every == 0) or is_last_batch
                            if should_step:
                                self.optimizer.step()
                                self.optimizer.zero_grad()
                                lr_scheduler.step()
                                if cfg.training.use_ema:
                                    ema.step(self.model)

                            reduced_loss = self._reduce_mean(raw_loss)
                            raw_loss_cpu = reduced_loss.item()
                            if self.is_main_process:
                                tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)
                            train_losses.append(raw_loss_cpu)
                            step_log = {
                                "train_loss": raw_loss_cpu,
                                "global_step": self.global_step,
                                "epoch": self.epoch,
                                "lr": lr_scheduler.get_last_lr()[0],
                            }

                            if not is_last_batch and self.is_main_process:
                                json_logger.log(step_log)

                            self.global_step += 1

                            if (cfg.training.max_train_steps is not None) and batch_idx >= (cfg.training.max_train_steps - 1):
                                break

                    if self.distributed:
                        dist.barrier()

                    # at the end of each epoch
                    if train_losses:
                        step_log["train_loss"] = float(np.mean(train_losses))

                    # ========= eval for this epoch ==========
                    if self.is_main_process:
                        policy = self.model
                        if cfg.training.use_ema:
                            policy = self.ema_model
                        policy.eval()

                        if (self.epoch % cfg.training.val_every) == 0 and val_dataloader is not None:
                            with torch.no_grad():
                                val_losses = []
                                with tqdm.tqdm(
                                    val_dataloader,
                                    desc=f"Validation epoch {self.epoch}",
                                    leave=False,
                                    mininterval=cfg.training.tqdm_interval_sec,
                                ) as tepoch:
                                    for batch_idx, batch in enumerate(tepoch):
                                        batch = dataset.postprocess(batch, self.device)
                                        loss = self.model(batch)
                                        val_losses.append(loss.detach())
                                        if (cfg.training.max_val_steps is not None) and batch_idx >= (
                                            cfg.training.max_val_steps - 1
                                        ):
                                            break
                                if len(val_losses) > 0:
                                    val_loss = torch.stack(val_losses).mean().item()
                                    step_log["val_loss"] = val_loss

                        if (self.epoch % cfg.training.sample_every) == 0 and train_sampling_batch is not None:
                            with torch.no_grad():
                                batch = train_sampling_batch
                                obs_dict = batch["obs"]
                                gt_action = batch["action"]

                                result = policy.predict_action(obs_dict)
                                pred_action = result["action_pred"]
                                mse = torch.nn.functional.mse_loss(pred_action, gt_action)
                                step_log["train_action_mse_error"] = mse.item()

                        policy.train()

                        # end of epoch
                        json_logger.log(step_log)

                        if (self.epoch + 1) % cfg.training.checkpoint_every == 0:
                            save_name = pathlib.Path(self.cfg.task.dataset.zarr_path).stem
                            run_parts = [save_name]
                            for cfg_key in ("setting", "exp_name"):
                                cfg_value = self.cfg.get(cfg_key, None)
                                if cfg_value not in (None, "", "default"):
                                    run_parts.append(str(cfg_value))
                            run_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", "-".join(run_parts)).strip("._-")
                            self.save_checkpoint(f"checkpoints/{run_name}-{seed}/{self.epoch + 1}.ckpt")

                    if self.distributed:
                        dist.barrier()

                    self.epoch += 1
        finally:
            self._teardown_distributed()


class BatchSampler:

    def __init__(
        self,
        data_size: int,
        batch_size: int,
        shuffle: bool = False,
        seed: int = 0,
        drop_last: bool = True,
    ):
        assert drop_last
        self.data_size = data_size
        self.batch_size = batch_size
        self.num_batch = data_size // batch_size
        self.discard = data_size - batch_size * self.num_batch
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def _get_perm(self):
        if self.shuffle:
            rng = np.random.default_rng(self.seed + self.epoch)
            perm = rng.permutation(self.data_size)
        else:
            perm = np.arange(self.data_size)
        if self.discard > 0:
            perm = perm[:-self.discard]
        return perm

    def __iter__(self):
        perm = self._get_perm().reshape(self.num_batch, self.batch_size)
        for i in range(self.num_batch):
            yield perm[i]

    def __len__(self):
        return self.num_batch


class DistributedBatchSampler(BatchSampler):

    def __init__(
        self,
        data_size: int,
        batch_size: int,
        world_size: int,
        rank: int,
        shuffle: bool = False,
        seed: int = 0,
        drop_last: bool = True,
    ):
        super().__init__(
            data_size=data_size,
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
            drop_last=drop_last,
        )
        self.world_size = world_size
        self.rank = rank
        self.global_num_batch = self.num_batch
        self.num_batch = self.global_num_batch // self.world_size
        self.distributed_discard = self.global_num_batch - (self.num_batch * self.world_size)

    def __iter__(self):
        perm = self._get_perm()
        if self.distributed_discard > 0:
            perm = perm[: -(self.distributed_discard * self.batch_size)]
        perm = perm.reshape(self.num_batch * self.world_size, self.batch_size)
        for i in range(self.rank, len(perm), self.world_size):
            yield perm[i]

    def __len__(self):
        return self.num_batch


def create_dataloader(
    dataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool,
    seed: int = 0,
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
):
    if distributed:
        batch_sampler = DistributedBatchSampler(
            len(dataset),
            batch_size,
            world_size=world_size,
            rank=rank,
            shuffle=shuffle,
            seed=seed,
            drop_last=True,
        )
    else:
        batch_sampler = BatchSampler(len(dataset), batch_size, shuffle=shuffle, seed=seed, drop_last=True)

    def collate(x):
        assert len(x) == 1
        return x[0]

    dataloader = DataLoader(
        dataset,
        collate_fn=collate,
        sampler=batch_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(persistent_workers and num_workers > 0),
    )
    return dataloader, batch_sampler


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")),
    config_name=pathlib.Path(__file__).stem,
)
def main(cfg):
    workspace = RobotWorkspace(cfg)
    workspace.run()


if __name__ == "__main__":
    main()
