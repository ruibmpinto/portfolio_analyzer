"""
Data export functionality.

Handles exporting analysis results to various formats.
"""

import pandas as pd
from typing import Dict, Any
import json


class DataExporter:
    """
    Exports portfolio analysis results to files.

    Supports CSV, Excel, and JSON formats.
    """

    def export_to_csv(
        self,
        data: pd.DataFrame,
        file_path: str,
        **kwargs) -> None:
        """
        Export DataFrame to CSV file.

        Args:
            data: DataFrame to export
            file_path: Output file path
            **kwargs: Additional arguments for to_csv

        Example:
            >>> exporter = DataExporter()
            >>> exporter.export_to_csv(
            ...     results_df,
            ...     'output/analysis.csv',
            ...     index=False
            ... )
        """
        try:
            data.to_csv(file_path, **kwargs)
        except Exception as e:
            raise Exception(f"Error exporting to CSV: {str(e)}")

    def export_to_excel(
        self,
        sheets: Dict[str, pd.DataFrame],
        file_path: str,
        engine: str = 'openpyxl',
        **writer_kwargs) -> None:
        """
        Export a dict of DataFrames to an Excel workbook.

        Each `(sheet_name, df)` pair becomes one tab in the
        workbook. Empty / None entries are skipped so callers
        can pass conditionally-populated sheets without
        guarding every entry. Single-sheet exports use the
        same call with a one-entry dict.

        Args:
            sheets: Dictionary {sheet_name: DataFrame}.
            file_path: Output file path (.xlsx).
            engine: Excel writer engine. Defaults to
                ``'openpyxl'`` (the only engine this project
                depends on); pass another string to override.
            **writer_kwargs: Passed to ``pd.ExcelWriter``.

        Example:
            >>> exporter.export_to_excel(
            ...     {'Holdings': holdings_df,
            ...      'Metrics': metrics_df},
            ...     'output/report.xlsx')
        """
        try:
            with pd.ExcelWriter(
                    file_path,
                    engine=engine,
                    **writer_kwargs) as writer:
                for sheet_name, df in sheets.items():
                    # Skip slots the caller left empty
                    if df is None:
                        continue
                    if isinstance(df, pd.DataFrame) and df.empty:
                        continue
                    # Tabular sheets carry a default RangeIndex
                    # we don't want in the workbook; matrix-style
                    # sheets (e.g. correlation) carry meaningful
                    # labels that must be preserved.
                    keep_index = not isinstance(df.index, pd.RangeIndex)
                    df.to_excel(
                        writer, sheet_name=sheet_name, index=keep_index)
        except Exception as e:
            raise Exception(
                f"Error exporting to Excel: {str(e)}")

    def export_to_json(
        self,
        data: Dict[str, Any],
        file_path: str,
        indent: int = 2
    ) -> None:
        """
        Export dictionary to JSON file.

        Args:
            data: Dictionary to export
            file_path: Output file path (.json)
            indent: JSON indentation level

        Example:
            >>> metrics = {
            ...     'sharpe': 1.5,
            ...     'beta': 1.2,
            ...     'alpha': 0.05
            ... }
            >>> exporter.export_to_json(
            ...     metrics,
            ...     'output/metrics.json'
            ... )
        """
        try:
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=indent)
        except Exception as e:
            raise Exception(f"Error exporting to JSON: {str(e)}")

    def export_series_to_csv(
        self,
        data: pd.Series,
        file_path: str,
        **kwargs
    ) -> None:
        """
        Export Series to CSV file.

        Args:
            data: Series to export
            file_path: Output file path
            **kwargs: Additional arguments for to_csv
        """
        try:
            data.to_csv(file_path, **kwargs)
        except Exception as e:
            raise Exception(f"Error exporting Series to CSV: {str(e)}")
