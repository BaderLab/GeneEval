import pandas as pd
import pytest
from geneeval import DatasetReader
from geneeval.data import PreprocessedData
from hypothesis import given
from hypothesis.strategies import text
from numpy.testing import assert_array_equal


# TODO: Tests for each type of task.
class TestDatasetReader:
    def test_dataset_reader(
        self,
        features_dataframe: pd.DataFrame,
        benchmark_filepath_manager,
        preprocessed_data: PreprocessedData,
    ) -> None:
        data = DatasetReader(
            features=features_dataframe,
            task="subcellular_localization",
        )

        assert_array_equal(
            # X_train, y_train will contain the 3 train + 1 valid examples
            data.X_train,
            features_dataframe[0:4],
        )
        # It is easier to spot errors if we compare to the text labels as oppose to binarized_labels
        data.lb.inverse_transform(data.y_train) == [
            ("cytoplasmic side", "endoplasmic reticulum membrane"),
            ("vacuole membrane",),
            ("cytoplasmic side", "lipid-anchor"),
            ("vacuole membrane",),
        ],

        assert_array_equal(data.X_test, features_dataframe[4:5])
        data.lb.inverse_transform(data.y_test) == [("endoplasmic reticulum membrane",)]
        assert_array_equal(data.splits.test_fold, [-1, -1, -1, 0])

    @given(task=text())
    def test_dataset_reader_invalid_task_value_error(self, task: str) -> None:
        with pytest.raises(ValueError):
            _ = DatasetReader(features=pd.DataFrame(), task=task)
