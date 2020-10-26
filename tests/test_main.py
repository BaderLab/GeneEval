from geneeval import main


class TestMain:
    def test_prepare_can_run(self):
        pass

    def test_evaluate_features_can_run(
        self, benchmark_filepath_manager, features_json_filepath: str
    ) -> None:
        results = main.evaluate_features(
            features_json_filepath, include_tasks=None, exclude_tasks=None
        )
        for partition in results.values():
            for scores in partition.values():
                for score in scores.values():
                    assert isinstance(score, float)

    def test_evaluate_predictions_can_run(
        self, benchmark_filepath_manager, predictions_filepath: str
    ) -> None:
        results = main.evaluate_predictions(predictions_filepath)
        for partition in results.values():
            for scores in partition.values():
                for score in scores.values():
                    assert isinstance(score, float)
