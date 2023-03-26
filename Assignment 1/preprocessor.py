import pandas as pd
import numpy as np


class Preprocessor:
    def __init__(self, data: pd.DataFrame, label_column: str):
        """Preprocesses data for use

        Args:
            data (pd.DataFrame): Data to preprocess
            label_column (str): Name of the column containing the labels
        """
        self.original_data: pd.DataFrame = data.copy()
        self.data: pd.DataFrame = data.copy()
        self.label_column: str = label_column

    def _drop_rows(self, rows: list[int]) -> None:
        """Drops rows from the data

        Args:
            rows (list[int]): List of row indexes to drop

        Returns:
            None

        """
        self.data.drop(rows, inplace=True)

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

    def _impute_cols(self) -> None:
        """Imputes missing values in all columns except the label column"""
        for col in self.data.columns:
            if col != self.label_column:
                if self.data[col].dtype == "object":
                    self.data[col] = self._impute(self.data[col], "mode")
                else:
                    self.data[col] = self._impute(self.data[col], "mean")

    def _label_encode(self, labels: list[int] = [0, 1]) -> pd.Series:
        """Encodes the label column

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
        print(self._decoder)
        return encoded.map(transformation_dict)

    def _get_rows_with_missing_values(self) -> list[int]:
        """Gets the indexes of rows with missing values

        Returns:
            list[int]: List of row  which have missing values
        """
        return self.data[self.data.isna().any(axis=1)].index.tolist()

    def get_folds(self, k: int = 100) -> list[pd.DataFrame]:
        """Returns k folds of the data after shuffling

        Args:
            k (int, optional): Number of folds. Defaults to 100.

        Returns:
            list[pd.DataFrame]: List of folds
        """
        data = self.data.copy()
        # data = data.sample(frac=1, random_state=42).reset_index(drop=True)
        folds = np.array_split(data, k)
        return folds

    def create_standardize_dict(self, df: pd.DataFrame):
        """Creates a dictionary of column means and standard deviations"""
        self._standardize_dict: dict[str, (float, float)] = {}
        for col in df.columns:
            if col != self.label_column:
                mean = df[col].mean()
                std = df[col].std()
                self._standardize_dict[col] = (mean, std)

    def _standardize(self, df: pd.DataFrame, col_name: str) -> pd.Series:
        """Standardizes a column

        Args:
            col (str): Column to standardize
            df (pd.DataFrame): Dataframe containing the column

        Returns:
            pd.Series: Standardized column
        """
        mean, std = self._standardize_dict[col_name]
        col = df[col_name]
        return (col - mean) / std

    def standardize_cols(self, df: pd.DataFrame) -> None:
        """Standardizes all columns except the label column

        Args:
            df (pd.DataFrame): Dataframe containing the columns to standardize

        Returns:
            None
        """
        for col in df.columns:
            if col != self.label_column:
                df[col] = self._standardize(df, col)

    def preprocess(
        self,
        drop_rows: list = [],
        drop_na: bool = True,
        standardize: bool = True,
        labels: list[int] = [0, 1],
        n_splits: int = 10,
    ) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
        """Preprocesses the data

        Args:
            drop_rows (list, optional): List of row indexes to drop. Defaults to [].
            drop_na (bool, optional): Whether to drop rows with missing values. Defaults to True.
            standardize (bool, optional): Whether to standardize the data. Defaults to True.
            labels (list[int], optional): A list of what you want the labels to be. Defaults to [0, 1].
            n_splits (int, optional): Number of train-test splits. Defaults to 10.
        Returns:
            list[tuple[pd.DataFrame, pd.DataFrame]]: List of tuples containing train and test data
        """
        self.data = self.original_data.copy()
        if drop_na:
            drop_rows += self._get_rows_with_missing_values()
        self._drop_rows(drop_rows)
        self._impute_cols()
        self.data[self.label_column] = self._label_encode(labels)
        splits = []
        folds = self.get_folds()
        for i in range(n_splits):
            train = pd.concat(folds[:i] + folds[i + 33 :])
            test = pd.concat(folds[i : i + 33])
            if standardize:
                self.create_standardize_dict(train)
                self.standardize_cols(train)
                self.standardize_cols(test)
            train = train.copy()
            test = test.copy()
            splits.append((train, test))
        return splits

    def decode(self, labels: list[int]) -> list[str]:
        """Decodes labels

        Args:
            labels (list[int]): List of labels to decode

        Returns:
            list[str]: List of decoded labels
        """
        return [self._decoder[label] for label in labels]
