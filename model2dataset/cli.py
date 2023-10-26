import time
from functools import partial
import numpy as np
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

from factory import build_pipeline, PipelineImageTransform, build_filters

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
    filters = build_filters(cfg.filters) if cfg.filters else None
    
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
    nb_written = 0
    sink = wds.ShardWriter(pattern, maxcount=output.per_shard)
    for data in dataloader:


        # Predicting
        nb = len(data[0])
        data_dict = {}
        for k, d in zip(dataset.inputs, data):
            data_dict[k] = d
        out = pipe.predict(data_dict)
        data_dict.update(out)
        # Filtering
        if filters is not None:
            filtered = []
            for i in range(nb):
                d = {}
                for name in data_dict.keys():
                    d[name] = data_dict[name][i]
                filtered.append(filters.filter(d)==False)
        else:
            filtered = None
        print(np.mean(filtered))
        # Saving
        for i in range(nb):
            if filtered and filtered[i]:
                continue
            key = f"{args.worker}_{i+nb_total}"
            dic = {"__key__": key}
            d = {}
            for name, ext in output.outputs.items():
                dic[name + "." + ext] = data_dict[name][i]
                d[name] = data_dict[name][i]
            sink.write(dic)
            nb_written += 1
        nb_total += nb
        dt = (time.time() - t0)
        processed_thr = nb_total / dt
        written_thr = nb_written / dt
        print(f"Total nb of samples processed: {nb_total}. Total written: {nb_written}. Processed/s: {processed_thr:.2f}. Written/s: {written_thr:.2f}.")

if __name__ == "__main__":
    sys.exit(main())