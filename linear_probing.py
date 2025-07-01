from patho_bench.ExperimentFactory import ExperimentFactory
from omegaconf import DictConfig, OmegaConf
import hydra
import os



@hydra.main(config_path="configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    job_dir = os.path.join("experiments", cfg.job_dir)
    embeddings_dir = f"{cfg.patching.mag}x_{cfg.patching.patch_size}px_{cfg.patching.overlap}px_overlap/features_{cfg.features.patch_encoder.backbone}"
    

    # Now we can run the experiment
    experiment = ExperimentFactory.linprobe(
                        split = cfg.custom_list_of_wsis,
                        task_config = cfg.probing.config_path,
                        pooled_embeddings_dir= None,
                        saveto =job_dir,
                        combine_slides_per_patient = cfg.probing.combine_slides_per_patient,
                        cost = 1,
                        balanced = False,
                        patch_embeddings_dirs = os.path.join(job_dir, embeddings_dir),
                        model_name = cfg.probing.model,
                        patch_aggregation=cfg.probing.aggregate              
                    )

    experiment.train()
    experiment.test()
    result = experiment.report_results(metric = 'macro-ovr-auc')

if __name__ == "__main__":
    main()
