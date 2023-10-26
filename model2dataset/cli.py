import time
from functools import partial
import os
import logging
import argparse
import sys
import json
import torch
import torch.nn.functional as F
import torchvision
import numpy as np
import braceexpand
from omegaconf import OmegaConf
try:
    import webdataset as wds
    has_wds = True
except ImportError:
    has_wds = False

from factory import build_pipeline, PipelineImageTransform

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="config.yaml")
    parser.add_argument('--worker', type=int, default=0)
    parser.add_argument('--nb_workers', type=int, default=1)
    parser.add_argument('--verbose', default=False, action="store_true", help="verbose mode")
    args = parser.parse_args()
    run(args)
   
def log_and_continue(exn):
    """Call in an exception handler to ignore any exception, isssue a warning, and continue."""
    logging.warning(f'Handling webdataset error ({repr(exn)}). Ignoring.')
    return True
 
def run(args):
    torch.backends.cudnn.benchmark = True
    cfg = OmegaConf.load(args.config)
    dataset = cfg.dataset
    pipe = build_pipeline(cfg.pipeline)
    output = cfg.output
    pipe_image_transform = PipelineImageTransform(pipe)
    if dataset.type == "webdataset":
        assert has_wds
        shardlist = wds.SimpleShardList(dataset.root)
        shardlist.urls = [url for i, url in enumerate(shardlist.urls) if i % args.nb_workers == args.worker]
        if args.verbose:
            print(f"Worker: {args.worker}, Total Workers: {args.nb_workers}, Number of shards to process: {len(shardlist.urls)}")
        ds = (
            wds.WebDataset(shardlist)
            .decode("pil", handler=wds.ignore_and_continue)
            .rename(**dataset.entries)
            .compose(pipe_image_transform)
            .to_tuple(*dataset.inputs)
            .batched(dataset.batch_size)
        )
        dataloader = wds.WebLoader(
            ds,
            batch_size=None,
            shuffle=False,
            num_workers=dataset.workers,
        )
    else:
        raise ValueError(dataset.type)
    
    nb_total = 0
    t0 = time.time()
    pattern = f"{output.folder}/{args.worker}_%015d.tar"
    sink = wds.ShardWriter(pattern, maxcount=output.per_shard)
    for data in dataloader:
        nb = len(data[0])
        data_dict = {}
        for k, d in zip(dataset.inputs, data):
            data_dict[k] = d
        out = pipe.predict(data_dict)
        data_dict.update(out)
        for k, v in data_dict.items():
            print(k, len(v))
        for i in range(nb):
            key = f"{args.worker}_{i+nb_total}"
            dic = {"__key__": key}
            for out in output.outputs:
                name, ext = out.split(".")
                dic[out] = data_dict[name][i]
            #print(dic)
            sink.write(dic)
        nb_total += nb
        throughput = nb_total / (time.time() - t0)
        print(f"Total nb of samples processed: {nb_total}. Throughput: {throughput:.2f} samples per sec")

if __name__ == "__main__":
    sys.exit(main())