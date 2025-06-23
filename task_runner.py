from mira_utils.segmentation_utils import segment_full_wsi
from omegaconf import DictConfig
from trident import Processor

def run_task(processor: Processor, task: str, cfg: DictConfig):
    if task == 'seg':
        _run_segmentation(processor, cfg)
    elif task == 'coords':
        _run_coordinate_extraction(processor, cfg)
    elif task == 'feat':
        _run_feature_extraction(processor, cfg)
    else:
        raise ValueError(f"Invalid task: {task}")

def _run_segmentation(processor: Processor, cfg: DictConfig):
    if not cfg.segmenter.model_name:
        segment_full_wsi(processor)
    else:
        from trident.segmentation_models.load import segmentation_model_factory

        segmentation_model = segmentation_model_factory(
            cfg.segmenter.model_name, confidence_thresh=cfg.segmenter.seg_conf_thresh
        )

        artifact_remover_model = None
        if cfg.segmenter.remove_artifacts or cfg.segmenter.remove_penmarks:
            artifact_remover_model = segmentation_model_factory(
                'grandqc_artifact',
                remove_penmarks_only=cfg.segmenter.remove_penmarks and not cfg.segmenter.remove_artifacts
            )

        processor.run_segmentation_job(
            segmentation_model,
            seg_mag=segmentation_model.target_mag,
            holes_are_tissue=not cfg.segmenter.remove_holes,
            artifact_remover_model=artifact_remover_model,
            batch_size=cfg.segmenter.seg_batch_size or cfg.batch_size,
            device=cfg.device
        )

def _run_coordinate_extraction(processor: Processor, cfg: DictConfig):
    processor.run_patching_job(
            target_magnification=cfg.patching.mag,
            patch_size=cfg.patching.patch_size,
            overlap=cfg.patching.overlap,
            saveto=cfg.patching.coords_dir,
            min_tissue_proportion=cfg.patching.min_tissue_proportion
        )


def _run_feature_extraction(processor: Processor, cfg: DictConfig):
    if cfg.features.type == 'slide':
        from trident.slide_encoder_models.load import encoder_factory
        encoder = encoder_factory(cfg.features.slide_encoder.backbone)
        processor.run_slide_feature_extraction_job(
            slide_encoder=encoder,
            coords_dir=cfg.patching.coords_dir or f'{cfg.patching.mag}x_{cfg.patching.patch_size}px_{cfg.patching.overlap}px_overlap',
            device=cfg.device,
            saveas='h5',
            batch_limit=cfg.features.feat_batch_size or cfg.batch_size,
        )
    else:
        from trident.patch_encoder_models.load import encoder_factory
        encoder = encoder_factory(cfg.features.patch_encoder.backbone, weights_path=cfg.features.patch_encoder.ckpt_path)
        processor.run_patch_feature_extraction_job(
            coords_dir=cfg.patching.coords_dir or f'{cfg.patching.mag}x_{cfg.patching.patch_size}px_{cfg.patching.overlap}px_overlap',
            patch_encoder=encoder,
            device=cfg.device,
            saveas='h5',
            batch_limit=cfg.features.feat_batch_size or cfg.batch_size,
        )
