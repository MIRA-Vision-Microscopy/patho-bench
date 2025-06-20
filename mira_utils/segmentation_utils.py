from __future__ import annotations
from trident.IO import create_lock, remove_lock, is_locked, update_log, mask_to_gdf, overlay_gdf_on_thumbnail
from trident import Processor 
from inspect import signature
import geopandas as gpd
from tqdm import tqdm
import numpy as np
import os

def segment_tissue(wsi, job_dir) -> str:
    wsi._lazy_initialize()
    max_dimension = 1000
    if wsi.width > wsi.height:
        thumbnail_width = max_dimension
        thumbnail_height = int(thumbnail_width * wsi.height / wsi.width)
    else:
        thumbnail_height = max_dimension
        thumbnail_width = int(thumbnail_height * wsi.width / wsi.height)
    thumbnail = wsi.get_thumbnail((thumbnail_width, thumbnail_height))

    # Generate full mask
    predicted_mask = np.ones((wsi.height, wsi.width), dtype=np.uint8) * 255

    # Save thumbnail image
    thumbnail_saveto = os.path.join(job_dir, 'thumbnails', f'{wsi.name}.jpg')
    os.makedirs(os.path.dirname(thumbnail_saveto), exist_ok=True)
    thumbnail.save(thumbnail_saveto)

    # Save geopandas contours
    gdf_saveto = os.path.join(job_dir, 'contours_geojson', f'{wsi.name}.geojson')
    os.makedirs(os.path.dirname(gdf_saveto), exist_ok=True)
    gdf_contours = mask_to_gdf(
        mask=predicted_mask,
        max_nb_holes=0,
        min_contour_area=1000,
        pixel_size=wsi.mpp,
        contour_scale=1
    )
    gdf_contours.set_crs("EPSG:3857", inplace=True)  # used to silent warning // Web Mercator
    gdf_contours.to_file(gdf_saveto, driver="GeoJSON")
    wsi.gdf_contours = gdf_contours
    wsi.tissue_seg_path = gdf_saveto

    # Draw the contours on the thumbnail image
    contours_saveto = os.path.join(job_dir, 'contours', f'{wsi.name}.jpg')
    annotated = np.array(thumbnail)
    overlay_gdf_on_thumbnail(gdf_contours, annotated, contours_saveto, thumbnail_width / wsi.width)

    return gdf_saveto


def segment_full_wsi(processor: Processor) -> str:
        saveto = os.path.join(processor.job_dir, 'contours')
        os.makedirs(saveto, exist_ok=True)

        sig = signature(processor.run_segmentation_job)
        local_attrs = {k: v for k, v in locals().items() if k in sig.parameters}
        processor.save_config(
            saveto=os.path.join(processor.job_dir, '_config_segmentation.json'),
            local_attrs=local_attrs,
            ignore = ['segmentation_model', 'loop', 'valid_slides', 'wsis']
        )

        processor.loop = tqdm(processor.wsis, desc='Segmenting tissue', total = len(processor.wsis))
        for wsi in processor.loop:   
            # Check if contour already exists
            if os.path.exists(os.path.join(saveto, f'{wsi.name}.jpg')) and not is_locked(os.path.join(saveto, f'{wsi.name}.jpg')):
                processor.loop.set_postfix_str(f'{wsi.name} already segmented. Skipping...')
                update_log(os.path.join(processor.job_dir, '_logs_segmentation.txt'), f'{wsi.name}{wsi.ext}', 'Tissue segmented.')
                continue

            # Check if another process has claimed this slide
            if is_locked(os.path.join(saveto, f'{wsi.name}.jpg')):
                processor.loop.set_postfix_str(f'{wsi.name} is locked. Skipping...')
                continue

            try:
                processor.loop.set_postfix_str(f'Segmenting {wsi}')
                create_lock(os.path.join(saveto, f'{wsi.name}.jpg'))
                update_log(os.path.join(processor.job_dir, '_logs_segmentation.txt'), f'{wsi.name}{wsi.ext}', 'LOCKED. Segmenting tissue...')

                # call a function from WSI object to do the work
                gdf_saveto = segment_tissue(wsi, job_dir=processor.job_dir)
                remove_lock(os.path.join(saveto, f'{wsi.name}.jpg'))

                gdf = gpd.read_file(gdf_saveto, rows=1)
                if gdf.empty:
                    update_log(os.path.join(processor.job_dir, '_logs_segmentation.txt'), f'{wsi.name}{wsi.ext}', 'Segmentation returned empty GeoDataFrame.')
                    processor.loop.set_postfix_str(f'Empty GeoDataFrame for {wsi.name}.')
                else:
                    update_log(os.path.join(processor.job_dir,  '_logs_segmentation.txt'), f'{wsi.name}{wsi.ext}', 'Tissue segmented.')

            except Exception as e:
                if isinstance(e, KeyboardInterrupt):
                    remove_lock(os.path.join(saveto, f'{wsi.name}.jpg'))
                if processor.skip_errors:
                    update_log(os.path.join(processor.job_dir, '_logs_segmentation.txt'), f'{wsi.name}{wsi.ext}', f'ERROR: {e}')
                    continue
                else:
                    raise e
                
        # Return the directory where the contours are saved
        return saveto
