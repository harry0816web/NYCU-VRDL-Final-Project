import os

class Config:
    input_dir_root = "../datas/input"
    output_dir = "../datas/output/exp_best_refactored/debug"
    check_exist_dir = False

    input_csv = "../datas/input/image-matching-challenge-2024/local_sample_submission.csv"
    category_csv = "../datas/input/image-matching-challenge-2024/train/categories.csv"
    target_datasets = None

    # Pipeline JSONs (relative to new-pipelines/)
    pipeline_json = "exp_best_refactored/pipeline.json"
    transparent_pipeline_json = "exp_best_refactored/transp_pipeline.json"

    # Fallback: ALIKED+LightGlue re-run when LoMa SfM registers < 30% of images
    fallback_enabled = True
    fallback_min_ratio = 0.3
    fallback_pipeline_json = "exp_best_refactored/pipeline_fallback.json"

    colmap_mapper_options = {
        "min_model_size": 3,
        "max_num_models": 2,
    }
