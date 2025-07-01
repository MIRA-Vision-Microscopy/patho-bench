# Linear Probing of Pathology Foundation Models
This project can be used for linear probing of pathology foundation models. It is largely based on [patho-bench](https://github.com/mahmoodlab/patho-bench), which was developed by the [Mahmood Lab](https://faisal.ai/) at Harvard Medical School and Brigham and Women's Hospital.

[arXiv](https://arxiv.org/pdf/2502.06750) | [HuggingFace](https://huggingface.co/datasets/MahmoodLab/Patho-Bench) | [Blog](https://www.linkedin.com/pulse/announcing-new-open-source-tools-accelerate-ai-pathology-andrew-zhang-loape/?trackingId=pDkifo54SRuJ2QeGiGcXpQ%3D%3D) | [Cite](https://github.com/mahmoodlab/patho-bench?tab=readme-ov-file#how-to-cite)
 | [License](https://github.com/mahmoodlab/patho-bench/blob/main/LICENSE)



## Feature Extraction

For linear probing or pre-trained foundation models, you first have to extract the features of you dataset.

```
python feature_extraction.py --config-name cervical
```

All parameters are configured using Hydra. An overview of all configuration parameters can be found in [HYDRA.md](HYDRA.md).


> [!IMPORTANT]
> If the patches are not stored in a format that inherently provides metadata about their resolution, it has to be provided in a seperate .csv file. The following structure of the csv file is recommended:


| slide_id | wsi | mpp | LABEL | fold_0 |
| -------- | ------- | ------- | ------- | ------- |
| example_wsi_1 | example_wsi_1.svs | 0.25 | 1 | train |
| example_wsi_2 | example_wsi_2.svs | 0.25 | 0 | test |

> [!TIP]
> For the datasets that have been used by the Mahmood lab to test their patho bench framework, these files can be found [here](https://huggingface.co/datasets/MahmoodLab/Patho-Bench/tree/main).


## Linear Probing
After feature extraction, run the linear probing experiment.
```
python linear_probing.py --config-name cervical
```

This requires additional config parameters:
| Name                         | Description                                                                                   | Default     |
|------------------------------|-----------------------------------------------------------------------------------------------|-------------|
| `config_path`                | Path of experiment config.                                                                    | _Required_  |
| `combine_slides_per_patient` | Perform early fusion prior to pooling                                                         | `False`     |
| `aggregate`                  | Aggregation strategy of paches per WSI. `concat` will concatenate all features and `stack` will treat each patch as a seperate instance.  | `concat`    |
| `model`                      | Name of the model to use for pooling. Only needed if pooled_embeddings_dir is empty.          | `Titan`     |

> [!NOTE]  
> Regarding the `combine_slides_per_patient` argument: If True, will perform early fusion by combining patches from all slides in to a single bag prior to pooling. If False, will pool each slide individually and take the mean of the slide-level features. The ideal value of this parameter depends on what pooling model you are using. For example, Titan requires this to be False because it uses spatial information (patch coordinates) during pooling. If a model doesn't use spatial information, you can usually set this to True, but it's best to consult with model documentation.

A config (.yaml) file needs to be provided that summarizes task labels, metrics,  etc. A minimum config file should look like this:

```yaml
task_col: LABEL # Column containing labels for the task
extra_cols: [] # Any extra columns needed to perform the task (e.g. survival tasks)

metrics: # List of one or more performance metrics to report (this is used for automated result compilation when using Patho-Bench in advanced mode)
  - macro-ovr-auc

label_dict: # Dictionary of integer labels to string labels
  0: healthy
  1: diseased

sample_col: slide_id # Column containing the unit of analysis. Use 'case_id' for patient-level tasks and 'slide_id' for slide-level tasks.
```


> [!TIP]
> For the datasets that have been used by the Mahmood lab to test their patho bench framework, the config files can be found [here](https://huggingface.co/datasets/MahmoodLab/Patho-Bench/tree/main).


## Advanced Usage
Patho-Bench offers automated scripts for large-scale benchmarking. These have not been adapted to be compatible with Hydra but can be found [here](https://github.com/mahmoodlab/Patho-Bench/tree/main/advanced_usage).

## Additional evaluations
Patho-Bench supports additional evaluation frameworks:
- `coxnet`  ➡️  Cox proportional hazards model for survival prediction (using pre-pooled features)
- `protonet`  ➡️  Prototyping (using pre-pooled features) (Coming soon!)
- `retrieval`  ➡️  Retrieval (using pre-pooled features)
- `finetune`  ➡️  Supervised finetuning or training from scratch (using patch features)

Details on using these can be found in the [tutorial](https://github.com/mahmoodlab/Patho-Bench/blob/main/tutorial/tutorial.ipynb).