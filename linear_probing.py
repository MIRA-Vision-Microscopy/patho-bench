from patho_bench.ExperimentFactory import ExperimentFactory
from omegaconf import DictConfig, OmegaConf
import hydra
import os



@hydra.main(config_path="configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    job_dir = os.path.join("experiments", cfg.job_dir)
    default_embeddings_dir = os.path.join(
        job_dir,
        f"{cfg.patching.mag}x_{cfg.patching.patch_size}px_{cfg.patching.overlap}px_overlap/features_{cfg.features.patch_encoder.backbone}",
    )

    patch_embeddings_dirs = cfg.probing.patch_embeddings_dirs
    if patch_embeddings_dirs is None:
        patch_embeddings_dirs = [default_embeddings_dir]
    elif isinstance(patch_embeddings_dirs, str):
        patch_embeddings_dirs = [patch_embeddings_dirs]
    else:
        patch_embeddings_dirs = list(patch_embeddings_dirs)

    external_saveto = cfg.probing.external_saveto
    if external_saveto is None and cfg.probing.external_split is not None:
        external_saveto = os.path.join(job_dir, "external_test")

    # Now we can run the experiment
    experiment = ExperimentFactory.linprobe(
                        split = cfg.custom_list_of_wsis,
                        task_config = cfg.probing.config_path,
                        pooled_embeddings_dir= None,
                        saveto = job_dir,
                        combine_slides_per_patient = cfg.probing.combine_slides_per_patient,
                        cost = 1,
                        balanced = cfg.probing.balanced,
                        gpu = cfg.probing.gpu,
                        patch_embeddings_dirs = patch_embeddings_dirs,
                        model_name = cfg.probing.model,
                        bag_size = cfg.probing.bag_size,
                        patch_aggregation = cfg.probing.aggregate,
                        external_split = cfg.probing.external_split,
                        validate_external = cfg.probing.validate_external,
                        preserve_external_folds = cfg.probing.preserve_external_folds,
                        external_saveto = external_saveto,
                    )

    experiment.train()
    experiment.test()
    result = experiment.report_results(metric = 'macro-ovr-auc')

if __name__ == "__main__":
    main()
