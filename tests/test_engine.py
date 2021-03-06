from geneeval.engine import Engine
import pandas as pd


class TestEngine:
    def test_engine_can_run(
        self, benchmark_filepath_manager, features_dataframe: pd.DataFrame
    ) -> None:
        engine = Engine(features_dataframe)
        engine.run()
        # TODO: This check is a little weak. Once we finalize the benchmark, we should add
        # checks for specific keys.
        assert isinstance(engine.results, dict)
        assert engine.results

    # TODO: Write these once the benchmark is finalized.
    def test_engine_can_run_include_tasks(
        self, benchmark_filepath_manager, features_dataframe: pd.DataFrame
    ) -> None:
        pass

    def test_engine_can_run_exclude_tasks(
        self, benchmark_filepath_manager, features_dataframe: pd.DataFrame
    ) -> None:
        pass
