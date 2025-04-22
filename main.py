import argparse
from src.pipelines.train_pipeline import TrainPipeline

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument(
        "--pipeline",
        type=str,
        choices=['train', 'select'],
        default="train",
        help="실행할 파이프라인을 선택: train 또는 select"
    )
    args = parser.parse_args()

    if args.pipeline == "train":
        from src.pipelines.train_pipeline import TrainPipeline
        pipeline = TrainPipeline(args.config)
    else:
        from src.pipelines.feature_selection_pipeline import FeatureSelectionPipeline
        pipeline = FeatureSelectionPipeline(args.config)
    pipeline.run()
    if args.pipeline == "train":
        pipeline.save_result()