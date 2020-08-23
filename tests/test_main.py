from geneeval import main


class TestMain:
    def test_prepare_can_run(self):
        pass

    def test_evaluate_features_can_run(
        self, benchmark_filepath_manager, features_json_filepath: str
    ) -> None:
        main.evaluate_features(features_json_filepath, include_tasks=None, exclude_tasks=None)

    def test_evaluate_predictions_can_run(
        self, benchmark_filepath_manager, features_json_filepath: str
    ) -> None:
        main.evaluate_features(features_json_filepath, include_tasks=None, exclude_tasks=None)
