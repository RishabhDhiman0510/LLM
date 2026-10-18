import argparse
from dataclasses import asdict, dataclass, fields


@dataclass
class ModelConfig:
    vocab_size: int = 8192
    context_length: int = 512
    d_model: int = 320
    n_layers: int = 8
    n_heads: int = 5
    dropout: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ModelConfig":
        return cls(**{f.name: data[f.name] for f in fields(cls) if f.name in data})


@dataclass
class TrainConfig:
    batch_size: int = 32
    grad_accum: int = 8
    max_steps: int = 3500
    lr: float = 1e-3
    min_lr: float = 1e-4
    warmup_steps: int = 300
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    log_interval: int = 10
    eval_interval: int = 250
    eval_batches: int = 40
    ckpt_interval: int = 500
    compile: bool = True
    seed: int = 1337


@dataclass
class FinetuneConfig(TrainConfig):
    grad_accum: int = 1
    max_steps: int = 0  # 0 means epochs * batches per epoch
    epochs: int = 2
    lr: float = 3e-4
    min_lr: float = 3e-5
    warmup_steps: int = 50
    weight_decay: float = 0.0
    dropout: float = 0.1
    eval_interval: int = 200
    eval_batches: int = 20
    ckpt_interval: int = 1000
    compile: bool = False


def add_config_args(parser: argparse.ArgumentParser, config: object, skip: tuple[str, ...] = ()) -> None:
    for field in fields(config):
        if field.name in skip:
            continue
        flag = "--" + field.name.replace("_", "-")
        default = getattr(config, field.name)
        if isinstance(default, bool):
            parser.add_argument(flag, action=argparse.BooleanOptionalAction, default=default)
        else:
            parser.add_argument(flag, type=type(default), default=default)


def config_from_args(cls: type, args: argparse.Namespace, **overrides) -> object:
    values = {f.name: getattr(args, f.name) for f in fields(cls) if hasattr(args, f.name)}
    return cls(**{**values, **overrides})
