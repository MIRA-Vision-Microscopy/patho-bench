import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from trident.Concurrency import batch_producer, batch_consumer, cache_batch
from omegaconf import DictConfig, OmegaConf
from trident.IO import collect_valid_slides
from task_runner import run_task
from trident import Processor
from threading import Thread
from queue import Queue
import hydra
import os


@hydra.main(config_path="configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    if cfg.wsi_cache:
        _run_parallel_mode(cfg)
    else:
        _run_sequential_mode(cfg)


def _initialize_processor(cfg: DictConfig):
    return Processor(
        job_dir=os.path.join("experiments", cfg.job_dir),
        wsi_source=cfg.wsi_dir,
        wsi_ext=cfg.wsi_ext,
        wsi_cache=cfg.wsi_cache,
        skip_errors=cfg.skip_errors,
        custom_mpp_keys=cfg.custom_mpp_keys,
        custom_list_of_wsis=cfg.custom_list_of_wsis,
        max_workers=cfg.max_workers,
        reader_type=cfg.reader_type,
        search_nested=cfg.search_nested,
    )


def _run_sequential_mode(cfg: DictConfig):
    processor = _initialize_processor(cfg)
    tasks = ['seg', 'coords', 'feat'] if cfg.task.task == 'all' else [cfg.task.task]
    for task in tasks:
        run_task(processor, task, cfg)


def _run_parallel_mode(cfg: DictConfig):
    queue = Queue(maxsize=1)

    valid_slides = collect_valid_slides(
        wsi_dir=cfg.wsi_dir,
        custom_list_path=cfg.custom_list_of_wsis,
        wsi_ext=cfg.wsi_ext,
        search_nested=cfg.search_nested,
        max_workers=cfg.max_workers
    )

    print(f"[MAIN] Found {len(valid_slides)} valid slides in {cfg.wsi_dir}.")

    warm = valid_slides[:cfg.cache_batch_size]
    warmup_dir = os.path.join(cfg.wsi_cache, "batch_0")
    print(f"[MAIN] Warmup caching batch: {warmup_dir}")
    cache_batch(warm, warmup_dir)
    queue.put(0)

    def processor_factory(wsi_dir):
        cfg_local = OmegaConf.merge(cfg, {"wsi_dir": wsi_dir, "wsi_cache": None, "custom_list_of_wsis": None, "search_nested": False})
        return _initialize_processor(cfg_local)

    def run_task_fn(processor, task_name):
        tasks = ['seg', 'coords', 'feat'] if cfg.task.task == 'all' else [cfg.task.task]
        for task in tasks:
            run_task(processor, task, cfg)

    print("[MAIN] Starting producer and consumer threads.")
    producer = Thread(target=batch_producer, args=(
        queue, valid_slides, cfg.cache_batch_size, cfg.cache_batch_size, cfg.wsi_cache
    ))
    consumer = Thread(target=batch_consumer, args=(
        queue, cfg.task, cfg.wsi_cache, processor_factory, run_task_fn
    ))

    producer.start()
    consumer.start()
    producer.join()
    consumer.join()


if __name__ == "__main__":
    main()
