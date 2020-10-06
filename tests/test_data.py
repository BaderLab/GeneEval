from geneeval import DatasetReader
import pandas as pd
from numpy.testing import assert_array_equal
import numpy as np


# TODO: Tests for each type of task.
class TestDatasetReader:
    def test_dataset_reader(
        self, benchmark_filepath_manager, features_dataframe: pd.DataFrame
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
        assert_array_equal(data.y_train, np.array([0, 0, 1, 1]))
        assert_array_equal(data.X_test, features_dataframe[4:5])
        assert_array_equal(data.y_test, np.array([1]))
        assert_array_equal(data.splits.test_fold, [-1, -1, -1, 0])
