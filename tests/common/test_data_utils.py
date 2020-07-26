import pytest
from hypothesis.strategies import text
from hypothesis import given
from geneeval.common.data_utils import load_embeddings
import pandas as pd


class TestDataUtils:
    @given(filepath=text())
    def test_value_error_load_embeddings(self, filepath: str) -> None:
        with pytest.raises(ValueError):
            load_embeddings(filepath)

    def test_load_embeddings_json(
        self, embeddings_dataframe: pd.DataFrame, embeddings_json_filepath: str
    ) -> None:
        pd.testing.assert_frame_equal(
            embeddings_dataframe, load_embeddings(embeddings_json_filepath),
        )

    def test_load_embeddings_tsv(
        self, embeddings_dataframe: pd.DataFrame, embeddings_tsv_filepath: str
    ) -> None:
        pd.testing.assert_frame_equal(
            embeddings_dataframe, load_embeddings(embeddings_tsv_filepath),
        )

    def test_load_embeddings_csv(
        self, embeddings_dataframe: pd.DataFrame, embeddings_csv_filepath: str
    ) -> None:
        pd.testing.assert_frame_equal(
            embeddings_dataframe, load_embeddings(embeddings_csv_filepath),
        )

    def test_load_embeddings_txt(
        self, embeddings_dataframe: pd.DataFrame, embeddings_txt_filepath: str
    ) -> None:
        pd.testing.assert_frame_equal(
            embeddings_dataframe, load_embeddings(embeddings_txt_filepath),
        )
