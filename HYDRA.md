## Configuration Parameters

### Generic Arguments

| Name                      | Description                                                                                      | Default     |
|---------------------------|--------------------------------------------------------------------------------------------------|-------------|
| `device`                  | Device to use for processing tasks. One of `cuda`, `mps`, or `cpu`.                              | `cuda`      |
| `job_dir`                 | Directory to store outputs.                                                                      | _Required_  |
| `wsi_dir`                 | Directory containing WSI files (no nesting allowed).                                             | _Required_  |
| `wsi_ext`                 | List of allowed file extensions for WSI files. If empty, allow all common formats.               | `None`      |
| `custom_mpp_keys`         | Custom keys used to store the resolution as MPP in your list of WSI files.                       | `None`      |
| `custom_list_of_wsis`     | Custom list of WSIs specified in a csv file.                                                     | `None`      |
| `search_nested`           | If set, recursively search for whole-slide images in subdirectories of `wsi_dir`.                | `False`     |
| `batch_size`              | Default batch size for all operations if not set individually.                                   | `64`        |
| `wsi_cache`               | Path to a local cache used to speed up access to WSIs stored on slower drives.                   | `None`      |
| `cache_batch_size`        | Maximum number of slides to cache locally at once. Helps control disk usage.                     | `32`        |
| `max_workers`             | Maximum number of workers. Set to 0 to use main process.                                         | `None`      |
| `skip_errors`             | Skip errored slides and continue processing.                                                     | `False`     |


### Reader
| Name                      | Description                                                                                      | Default     |
|---------------------------|--------------------------------------------------------------------------------------------------|-------------|
| `reader_type`             | Force the use of a specific WSI image reader.  Can be `openslide`, `image`, or `cucim`.          | `None`      |


### Tissue Segmentation
| Name                      | Description                                                                                      | Default     |
|---------------------------|--------------------------------------------------------------------------------------------------|-------------|
| `model_name`              | Type of tissue vs. background segmenter. Options are `none`, `hest` or `grandqc`.                | `none`      |
| `seg_conf_thresh`         | Confidence threshold to apply to binarize segmentation predictions. Lower to retain more tissue. | `0.5`       |
| `remove_holes`            | Remove holes from tissue segmentation mask.                                                      | `False`     |
| `remove_artifacts`        | Run an additional model to remove artifacts (including penmarks, blurs, stains, etc.).           | `False`     |
| `remove_penmarks`         | Run an additional model to remove penmarks.                                                      | `False`     |
| `seg_batch_size`          | Batch size for segmentation. If `None`, use `batch_size` argument instead.                       | `None`      |


### Patching
| Name                      | Description                                                                                      | Default     |
|---------------------------|--------------------------------------------------------------------------------------------------|-------------|
| `mag`                     | Magnification for coords/features extraction.  One of `[5, 10, 20, 40, 80]`.                     | `20`        |
| `patch_size`              | Patch size for coords/image extraction.                                                          | `512`       |
| `overlap`                 | Absolute overlap for patching in pixels.                                                         | `0`         |
| `min_tissue_proportion`   | Minimum proportion of the patch under tissue to be kept. Between `0` and `1.0`.                  | `0`         |
| `coords_dir`              | Directory to save/restore tissue coordinates.                                                    | `None`      |


### Feature Extraction
| Name                      | Description                                                                                      | Default     |
|---------------------------|--------------------------------------------------------------------------------------------------|-------------|
| `type`                    | Type of encoder. `patch` or `slide`.                                                             | _Required_  |
| `feat_batch_size`         | Batch size for feature extraction.  If `None`, use `batch_size` argument instead.                | `None`      |
| `patch_encoder`           | Patch encoder. One of `conch_v1`, `uni_v1`, `uni_v2`, `ctranspath`, `phikon`,`resnet50`, `gigapath`, `virchow`, `virchow2`, `hoptimus0`, `hoptimus1`, `phikon_v2`, `conch_v15`, `musk`, `hibou_l`, `kaiko-vits8`, `kaiko-vits16`, `kaiko-vitb8`, `kaiko-vitb16`, `kaiko-vitl14`, `lunit-vits8`, `midnight12k`.                                                                                  | `conch_v15` |
| `patch_encoder_ckpt_path` | Optional local path to a patch encoder checkpoint (.pt, .pth, .bin, or .safetensors).            | `None`      |
| `slide_encoder`           | Slide encoder. One of `threads`, `titan`, `prism`, `gigapath`, `chief`, `madeleine`, `mean-virchow`, `mean-virchow2`, `mean-conch_v1`, `mean-conch_v15`, `mean-ctranspath`, `mean-gigapath`, `mean-resnet50`, `mean-hoptimus0`, `mean-phikon`, `mean-phikon_v2`, `mean-musk`, `mean-uni_v1`, `mean-uni_v2`.                                                                                     | `None`      |
| `slide_encoder_ckpt_path` | Optional local path to a slide encoder checkpoint (.pt, .pth, .bin, or .safetensors).            | `None`      |

[Back to README.md](README.md)



