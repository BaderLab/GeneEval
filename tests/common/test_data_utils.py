import pytest
from hypothesis.strategies import text
from hypothesis import given
from geneeval.common.data_utils import load_features
import pandas as pd
from pathlib import Path


class TestDataUtils:
    @given(filepath=text())
    def test_value_error_load_features(self, filepath: str) -> None:
        filepath = Path(filepath)
        with pytest.raises(ValueError):
            load_features(filepath)

    def test_can_pass_kwargs_read_json(
        self, features_dataframe: pd.DataFrame, features_json_filepath: Path
    ) -> None:
        # Pass a keyword that we know causes a certain error as a check that kwargs is working.
        with pytest.raises(ValueError):
            load_features(features_json_filepath, orient="split")

    def test_can_pass_kwargs_read_csv(
        self, features_dataframe: pd.DataFrame, features_csv_filepath: Path
    ) -> None:
        # Pass a keyword that we know causes a certain error as a check that kwargs is working.
        with pytest.raises(ValueError):
            assert load_features(features_csv_filepath, delim_whitespace=True)

    def test_load_features_json(
        self, features_dataframe: pd.DataFrame, features_json_filepath: Path
    ) -> None:
        pd.testing.assert_frame_equal(
            features_dataframe,
            load_features(features_json_filepath),
        )

    def test_load_features_tsv(
        self, features_dataframe: pd.DataFrame, features_tsv_filepath: Path
    ) -> None:
        pd.testing.assert_frame_equal(
            features_dataframe,
            load_features(features_tsv_filepath),
        )

    def test_load_features_csv(
        self, features_dataframe: pd.DataFrame, features_csv_filepath: Path
    ) -> None:
        pd.testing.assert_frame_equal(
            features_dataframe,
            load_features(features_csv_filepath),
        )

    def test_load_features_txt(
        self, features_dataframe: pd.DataFrame, features_txt_filepath: Path
    ) -> None:
        pd.testing.assert_frame_equal(
            features_dataframe,
            load_features(features_txt_filepath),
        )
