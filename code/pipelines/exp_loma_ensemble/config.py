import os

class Config:
    input_dir_root = "/kaggle/input"
    output_dir = "/tmp"
    check_exist_dir = False

    input_csv = "/kaggle/input/image-matching-challenge-2024/sample_submission.csv"
    category_csv = "/kaggle/input/image-matching-challenge-2024/train/categories.csv"
    target_datasets = None

    # Pipeline JSONs
    pipeline_json = "new-pipelines/exp_ver58_loma_ensemble/pipeline.json"
    transparent_pipeline_json = "new-pipelines/exp_ver58_loma_ensemble/transp_pipeline.json"

    # Scene-level fallback: re-run ALIKED+LightGlue when LoMa SfM
    # registers fewer than 30% of images
    fallback_enabled = True
    fallback_min_ratio = 0.3
    fallback_pipeline_json = "new-pipelines/exp_ver58_loma_ensemble/pipeline_fallback.json"

    colmap_mapper_options = {
        "min_model_size": 3,
        "max_num_models": 3,
    }
