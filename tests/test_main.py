from geneeval import main


class TestMain:
    def test_prepare_can_run(self):
        pass

    def test_evaluate_features_can_run(self, embeddings_json_filepath: str) -> None:
        main.evaluate_features(embeddings_json_filepath, include_tasks=None, exclude_tasks=None)

    def test_evaluate_predictions_can_run(self, embeddings_json_filepath: str) -> None:
        main.evaluate_features(embeddings_json_filepath, include_tasks=None, exclude_tasks=None)
