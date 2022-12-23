import time
from functools import partial
import os
import argparse
import sys
import json
import torch
import torch.nn.functional as F
import torchvision
import numpy as np
try:
    import timm
    from timm.data import resolve_data_config
    from timm.data.transforms_factory import create_transform
    has_timm = True
except ImportError:
    has_timm = False

try:
    import transformers
    from transformers import AutoFeatureExtractor
    has_huggingface = True
except ImportError:
    has_huggingface = False

try:
    import webdataset as wds
    has_wds = True
except ImportError:
    has_wds = False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_type', type=str, default="image_folder")
    parser.add_argument('--dataset_root', type=str, default="root")
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--model', type=str, default="resnet50")
    parser.add_argument('--library', type=str, default="timm")
    parser.add_argument('--model_class', type=str, default="")
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--out_folder', type=str, default="out")
    parser.add_argument('--out_format', type=str, default="npy")
    parser.add_argument('--batches_per_chunk', type=int, default=100)
    parser.add_argument('--normalize', default=False, action="store_true", help="verbose mode")
    parser.add_argument('--verbose', default=False, action="store_true", help="verbose mode")
    parser.add_argument('--per_shard', default=10_000, type=int, help="Samples per shard")
    parser.add_argument('--distributed', default=False, action="store_true", help="distributed mode")
    parser.add_argument('--dist_env', type=str, default="env://")
    parser.add_argument('--dist_backend', type=str, default="nccl")
    parser.add_argument('--save_images', default=False, action="store_true", help="verbose mode")
    parser.add_argument('--save_features', default=False, action="store_true", help="verbose mode")
    parser.add_argument('--save_ids', default=False, action="store_true", help="verbose mode")
    parser.add_argument('--save_meta', default=False, action="store_true", help="verbose mode")
    parser.add_argument('--save_probas', default=False, action="store_true", help="verbose mode")
    parser.add_argument('--filter_thres', default=0, type=float, help="verbose mode")
    args = parser.parse_args()
    run(args)
   
def log_and_continue(exn):
    """Call in an exception handler to ignore any exception, isssue a warning, and continue."""
    logging.warning(f'Handling webdataset error ({repr(exn)}). Ignoring.')
    return True

def world_info_from_env():
    # from https://github.com/mlfoundations/open_clip/blob/1c8647f14ff1f826b9096962777e39f7c5cd4ba9/src/training/distributed.py
    # Thanks to OpenCLIP authors
    local_rank = 0
    for v in ('SLURM_LOCALID', 'MPI_LOCALRANKID', 'OMPI_COMM_WORLD_LOCAL_RANK', 'LOCAL_RANK'):
        if v in os.environ:
            local_rank = int(os.environ[v])
            break
    global_rank = 0
    for v in ('SLURM_PROCID', 'PMI_RANK', 'OMPI_COMM_WORLD_RANK', 'RANK'):
        if v in os.environ:
            global_rank = int(os.environ[v])
            break
    world_size = 1
    for v in ('SLURM_NTASKS', 'PMI_SIZE', 'OMPI_COMM_WORLD_SIZE', 'WORLD_SIZE'):
        if v in os.environ:
            world_size = int(os.environ[v])
            break

    return local_rank, global_rank, world_size

from webdataset.shardlists import IterableDataset, Composable, ShardSample, SimpleShardSample
class DistPytorchEnv:
    """A class encapsulating the PyTorch node/worker environment."""

    def __init__(self, group=None):
        """Initialize rank/worker information."""
        import socket

        super().__init__()
        self.rank = None
        self.worker = None
        self.group = group
        self.nodeinfo = (socket.gethostname(), os.getpid())
        self.update_env()

    def update_env(self):
        """Update information about node and worker environment.
        This code is written this way because the torch.distributed info is
        available only in the environment where the loader is created.
        This class retains that environment info when it is serialized.
        """
        from webdataset import gopen

        try:
            import torch
            import torch.distributed
        except Exception:
            return

        if self.rank is None:
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                group = self.group or torch.distributed.group.WORLD
                self.rank = torch.distributed.get_rank(group=group), \
                            torch.distributed.get_world_size(group=group)
            else:
                _, rank, world_size = world_info_from_env()
                if world_size > 1:
                    self.rank = (rank, world_size)

        if self.worker is None:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                self.worker = worker_info.id, worker_info.num_workers

        gopen.info["nodeinfo"] = self.nodeinfo
        gopen.info["rank"], gopen.info["size"] = self.rank or (-1, -1)
        gopen.info["worker_id"], gopen.info["num_workers"] = self.worker or (-1, -1)

class DistShardList(IterableDataset, DistPytorchEnv, Composable):
    """An iterable dataset yielding a list of urls.
    This understands the PyTorch distributed and worker APIs and splits shards
    accordingly.
    """

    def __init__(
        self,
        urls,
        epoch_shuffle=False,
        shuffle=False,
        split_by_worker=True,
        split_by_node=True,
        verbose=False,
    ):
        """Create a ShardList.
        :param urls: a list of URLs as a Python list or brace notation string
        :param shuffle: shuffle samples before iterating
        :param split_by_node: split shards by node if True
        :param split_by_worker: split shards by worker if True
        :param group: group used for determining rank/world_size
        If WDS_SHUFFLE is in the environment, it is used for shuffling shards prior
        to splitting; this assigns different shards to different nodes on each epoch.
        """
        super().__init__()

        self.verbose = verbose
        if self.verbose:
            print("PytorchShardList init")
        self.epoch = -1
        self.epoch_shuffle = epoch_shuffle
        self.shuffle = shuffle
        self.split_by_worker = split_by_worker
        self.split_by_node = split_by_node
        if not isinstance(urls, ShardSample):
            urls = SimpleShardSample(urls)
        self.shardsample = urls

    def set_epoch(self, epoch):
        """Set the current epoch. Used for per-node shuffling."""
        self.epoch = epoch - 1

    def __iter__(self):
        """Return an iterator over the shards."""
        self.epoch += 1
        if hasattr(self.shardsample, "set_epoch"):
            self.shardsample.set_epoch(self.epoch)
        self.update_env()
        urls = self.shardsample.sample()
        if self.epoch_shuffle:
            if "WDS_EPOCH" not in os.environ:
                raise ValueError(
                    "when specifying epoch_shuffle, you must provide the epoch in the WDS_EPOCH environment variable"
                )
            epoch = int(os.environ["WDS_EPOCH"])
            if self.verbose:
                print(f"PytorchShardList epochshuffle {epoch}")
            random.Random(epoch).shuffle(urls)
        if self.split_by_node:
            rank, world = self.rank or (0, 1)
            if self.verbose:
                print(f"PytorchShardList rank {rank} of {world}")
            urls = urls[rank::world]
        if self.split_by_worker:
            worker, nworkers = self.worker or (0, 1)
            if self.verbose:
                print(f"PytorchShardList worker {worker} of {nworkers}")
            urls = urls[worker::nworkers]
        if self.shuffle:
            random.Random(self.epoch + 17).shuffle(urls)
        if self.verbose:
            print(f"PytorchShardList got {len(urls)} urls", urls)
        for url in urls:
            yield dict(
                url=url,
                __url__=url,
                __worker__=str(self.worker),
                __rank__=str(self.rank),
                __nodeinfo__=str(self.nodeinfo),
            )
 
def run(args):
    if args.distributed:
        local_rank, rank, world_size = world_info_from_env()
        # env variables used by  webdataset for sharding
        os.environ['LOCAL_RANK'] = str(local_rank)
        os.environ['RANK'] = str(rank)
        os.environ['WORLD_SIZE'] = str(world_size)
        device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
        torch.cuda.set_device(device)
        print(local_rank, rank, world_size)
    else:
        rank = 0
        world_size = 1
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = True
    
    if args.library == "timm":
        assert has_timm
        model = timm.create_model(args.model, pretrained=True)
        config = resolve_data_config({}, model=model)
        transform = create_transform(**config)
    elif args.library == "huggingface":
        assert has_huggingface
        fe = AutoFeatureExtractor.from_pretrained(args.model)
        def prepro(image):
            data = fe(image, return_tensors="pt")
            return data['pixel_values'][0]
        transform = prepro
        model = getattr(transformers, args.model_class).from_pretrained(args.model)
    else:
        raise ValueError(args.library)
    model = model.to(device)
    model.eval()
    
    # with torch.no_grad():
        # sample_input = (torch.randn(64,3,224,224).to(device),)
        # traced_model = torch.jit.trace(model, sample_input, strict=False)
        # model = torch.jit.freeze(traced_model)
    
    def dup(data):
        for sample in data:
            yield {"image": sample["image"], "raw": sample["image"], "json": sample["json"]}

    if args.dataset_type == "webdataset":
        assert has_wds
        # shardlist = wds.SimpleShardList(args.dataset_root)
        shardlist  = DistShardList(
            args.dataset_root,
            epoch_shuffle=False,
            split_by_node=True,
            split_by_worker=True,
            verbose=True,
        )
        dataset = (
            wds.WebDataset(shardlist)
            .decode("pil", handler=wds.ignore_and_continue)
            .rename(image="jpg;png", json="json")
            .then(dup)
            .map_dict(image=transform)
            .to_tuple("raw", "image", "json")
            .batched(args.batch_size)
        )
        dataloader = wds.WebLoader(
            dataset,
            batch_size=None,
            shuffle=False,
            num_workers=args.workers,
        )
    elif args.dataset_type == "image_folder":
        dataset = torchvision.datasets.ImageFolder(args.dataset_root, transform=transform)
        dataloader = torch.utils.data.DataLoader(dataset, num_workers=args.workers, shuffle=False, batch_size=args.batch_size)
    else:
        raise ValueError(args.dataset_type)

    chunk = []
    chunk_id = 0
    nb = 0
    t0 = time.time()
    pattern = f"{args.out_folder}/{rank}_%05d.tar"
    sink = wds.ShardWriter(pattern, maxcount=args.per_shard)
    for X_raw, X, meta in dataloader:
        X = X.to(device)
        with torch.no_grad():
            if args.library == "timm":
                features = model(X)
            elif args.library == "huggingface":
                output = model(X)
                features = output["pooler_output"]
            features = features.view(len(features), -1)
            probas = features.softmax(dim=1)
            maxes, ids = probas.max(dim=1)
        if args.filter_thres:
            mask = (maxes >= args.filter_thres)
        else:
            ids = features.argmax(dim=1)
            mask = None
        features = features.cpu()
        features = features.numpy()
        for i in range(len(X_raw)):
            if mask is not None and mask[i] == False:
                continue
            key = f"{rank}_{i+nb}"
            dic = {"__key__": key}
            if args.save_images:
                dic["image.jpg"] = X_raw[i]
            if args.save_features:
                dic["features.npy"] = features[i]
            if args.save_meta:
                dic["meta.json"] = meta[i]
            if args.save_ids:
                dic["class.cls"] = ids[i].item()
            if args.save_probas:
                dic["proba.pyd"] = maxes[i].item()
            sink.write(dic)
        nb += len(features)
        if rank == 0:
            total = nb*world_size
            throughput = total / (time.time() - t0)
            print(f"Total nb of images processed: {total}. Throughput: {throughput:.2f} images per sec")

if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
