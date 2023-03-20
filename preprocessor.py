import pandas as pd
import numpy as np


class Preprocessor:
    def __init__(self, data: pd.DataFrame, label_column: str):
        """Preprocesses data for use

        Args:
            data (pd.DataFrame): Data to preprocess
            label_column (str): Name of the column containing the labels
        """
        self.original_data = data.copy()
        self.data = data.copy()
        self.label_column = label_column

    def _drop_rows(self, rows: list[int]):
        """Drops rows from the data

        Args:
            rows (list[int]): List of row indexes to drop

        Returns:
            None

        """
        self.data.drop(rows, inplace=True)

    def _standardize(self, col: pd.Series) -> pd.Series:
        """Standardizes a column

        Args:
            col (pd.Series): Column to standardize

        Returns:
            pd.Series: Standardized column
        """
        return (col - col.mean()) / col.std()

    def _standardize_cols(self):
        """Standardizes all columns except the label column"""
        for col in self.data.columns:
            if col != self.label_column:
                self.data[col] = self._standardize(self.data[col])

    def _impute(self, col: pd.Series, method: str = "mean") -> pd.Series:
        """Imputes missing values in a column

        Args:
            col (pd.Series): Column to impute
            method (str, optional): Method to use for imputation. Defaults to "mean".

        Returns:
            pd.Series: Imputed column
        """
        if method == "mean":
            return col.fillna(col.mean())
        elif method == "median":
            return col.fillna(col.median())
        elif method == "mode":
            return col.fillna(col.mode()[0])
        else:
            return col

    def _impute_cols(self):
        """Imputes missing values in all columns except the label column"""
        for col in self.data.columns:
            if col != self.label_column:
                if self.data[col].dtype == "object":
                    self.data[col] = self._impute(self.data[col], "mode")
                else:
                    self.data[col] = self._impute(self.data[col], "mean")

    def _label_encode(self, labels: list[int] = [0, 1]) -> pd.Series:
        """Encodes the label column (two classes only: 0 and 1)

        Args:
            labels (list[int], optional): A list of what you want the labels to be. Defaults to [0, 1]

        Returns:
            pd.Series: Encoded label column
        """
        col: pd.Series = self.data[self.label_column]
        encoded = col.astype("category").cat.codes
        _decoder: dict[int, str] = dict(
            enumerate(col.astype("category").cat.categories)
        )
        self._decoder = {labels[k]: v for k, v in _decoder.items()}
        transformation_dict = {0: labels[0], 1: labels[1]}
        return encoded.map(transformation_dict)

    def _get_rows_with_missing_values(self) -> list[int]:
        """Gets the indexes of rows with missing values

        Returns:
            list[int]: List of row  which have missing values
        """
        return self.data[self.data.isna().any(axis=1)].index.tolist()

    def preprocess(
        self,
        drop_rows: list = [],
        drop_na: bool = True,
        standardize: bool = True,
        labels: list[int] = [0, 1],
    ) -> pd.DataFrame:
        """Preprocesses the data

        Args:
            impute_method (str, optional): Method to use for imputation. Defaults to "mean".
            drop_rows (list, optional): List of row indexes to drop. Defaults to [].
            drop_na (bool, optional): Whether to drop rows with missing values. Defaults to True.
            standardize (bool, optional): Whether to standardize the data. Defaults to True.

        Returns:
            pd.DataFrame: Preprocessed data
        """
        self.data = self.original_data.copy()
        if drop_na:
            drop_rows += self._get_rows_with_missing_values()
        self._drop_rows(drop_rows)
        self._impute_cols()
        if standardize:
            self._standardize_cols()
        self.data[self.label_column] = self._label_encode(labels)
        return self.data.copy()

    def get_folds(self, k: int = 10) -> list[pd.DataFrame]:
        """Returns k folds of the data after shuffling

        Args:
            k (int, optional): Number of folds. Defaults to 10.

        Returns:
            list[pd.DataFrame]: List of folds
        """
        data = self.data.copy()
        data = data.sample(frac=1).reset_index(drop=True)
        folds = np.array_split(data, k)
        return folds

    def decode(self, labels: list[int]) -> list[str]:
        """Decodes labels

        Args:
            labels (list[int]): List of labels to decode

        Returns:
            list[str]: List of decoded labels
        """
        return [self._decoder[label] for label in labels]
