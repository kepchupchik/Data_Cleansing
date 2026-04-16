from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


TRUE_VALUES = {"true", "1", "yes", "y", "да", "истина", "t", "on"}
FALSE_VALUES = {"false", "0", "no", "n", "нет", "ложь", "f", "off"}


@dataclass(slots=True)
class ConversionResult:
    converted: pd.Series
    invalid_count: int
    invalid_examples: list[str]
    target_type: str


@dataclass(slots=True)
class FillResult:
    series: pd.Series
    filled_count: int
    remaining_missing: int
    used_groups: tuple[str, ...]


@dataclass(slots=True)
class BulkFillConfig:
    numeric_method: str = "median"
    categorical_method: str = "mode"
    numeric_constant: Any = None
    categorical_constant: Any = None
    fallback_to_global: bool = True


def format_value(value: Any) -> str:
    if isinstance(value, (float, np.floating)):
        return f"{value:.4f}".rstrip("0").rstrip(".")
    if isinstance(value, (pd.Timestamp,)):
        return value.strftime("%Y-%m-%d %H:%M:%S")
    return str(value)


def safe_mode(series: pd.Series) -> Any:
    clean_series = series.dropna()
    if clean_series.empty:
        return np.nan
    mode_values = clean_series.mode(dropna=True)
    if mode_values.empty:
        return np.nan
    return mode_values.iloc[0]


def compute_outlier_records(series: pd.Series, multiplier: float = 1.5) -> tuple[list[dict[str, Any]], dict[str, float]]:
    clean_series = pd.to_numeric(series.dropna(), errors="coerce").dropna()
    if clean_series.empty:
        bounds = {"lower": np.nan, "upper": np.nan, "q1": np.nan, "q3": np.nan, "iqr": np.nan}
        return [], bounds

    q1 = clean_series.quantile(0.25)
    q3 = clean_series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - multiplier * iqr
    upper = q3 + multiplier * iqr

    outliers = clean_series[(clean_series < lower) | (clean_series > upper)].sort_values()
    records = []
    for row_index, value in outliers.items():
        side = "below" if value < lower else "above"
        records.append({"row_index": int(row_index), "value": value, "side": side})

    bounds = {"lower": float(lower), "upper": float(upper), "q1": float(q1), "q3": float(q3), "iqr": float(iqr)}
    return records, bounds


def _compute_fill_value(series: pd.Series, method: str, constant_value: Any = None) -> Any:
    clean_series = series.dropna()
    if method == "constant":
        if constant_value is None or str(constant_value).strip() == "":
            raise ValueError("Для заполнения константой нужно указать непустое значение.")
        return constant_value
    if clean_series.empty:
        raise ValueError("В столбце нет значений, подходящих для заполнения пропусков.")

    if method == "median":
        if not pd.api.types.is_numeric_dtype(series):
            raise ValueError("Медиана доступна только для числовых столбцов.")
        return clean_series.median()
    if method == "mean":
        if not pd.api.types.is_numeric_dtype(series):
            raise ValueError("Среднее доступно только для числовых столбцов.")
        return clean_series.mean()
    if method == "mode":
        result = safe_mode(clean_series)
        if pd.isna(result):
            raise ValueError("Не удалось вычислить моду для заполнения.")
        return result

    raise ValueError(f"Неподдерживаемый метод заполнения: {method}")


def _group_fill_values(
    df: pd.DataFrame,
    target_column: str,
    method: str,
    group_columns: tuple[str, ...],
    constant_value: Any = None,
) -> pd.Series:
    grouped = df.groupby(list(group_columns), dropna=False)[target_column]

    if method == "mean":
        return grouped.transform("mean")
    if method == "median":
        return grouped.transform("median")
    if method == "mode":
        return grouped.transform(lambda values: safe_mode(values))
    if method == "constant":
        value = _compute_fill_value(df[target_column], method, constant_value)
        return pd.Series(value, index=df.index, dtype="object")

    raise ValueError(f"Неподдерживаемый метод группового заполнения: {method}")


def fill_missing_values(
    df: pd.DataFrame,
    target_column: str,
    method: str,
    *,
    constant_value: Any = None,
    group_columns: tuple[str, ...] = (),
    fallback_to_global: bool = True,
) -> FillResult:
    series = df[target_column].copy()
    initial_missing = int(series.isna().sum())
    valid_group_columns = tuple(column for column in group_columns if column and column != target_column and column in df.columns)

    if initial_missing == 0:
        return FillResult(series=series, filled_count=0, remaining_missing=0, used_groups=valid_group_columns)

    if valid_group_columns:
        group_values = _group_fill_values(df, target_column, method, valid_group_columns, constant_value=constant_value)
        series = series.fillna(group_values)

    if series.isna().any() and fallback_to_global:
        fill_value = _compute_fill_value(df[target_column], method, constant_value=constant_value)
        series = series.fillna(fill_value)

    remaining_missing = int(series.isna().sum())
    filled_count = initial_missing - remaining_missing
    return FillResult(
        series=series,
        filled_count=filled_count,
        remaining_missing=remaining_missing,
        used_groups=valid_group_columns,
    )


def bulk_fill_missing(df: pd.DataFrame, config: BulkFillConfig) -> tuple[pd.DataFrame, dict[str, Any]]:
    result = df.copy()
    report: dict[str, Any] = {"filled_columns": {}, "skipped_columns": [], "remaining_missing": {}}

    for column_name in result.columns:
        if result[column_name].isna().sum() == 0:
            continue
        if result[column_name].dropna().empty:
            report["skipped_columns"].append(column_name)
            continue

        is_numeric = pd.api.types.is_numeric_dtype(result[column_name])
        method = config.numeric_method if is_numeric else config.categorical_method
        constant_value = config.numeric_constant if is_numeric else config.categorical_constant

        try:
            fill_result = fill_missing_values(
                result,
                column_name,
                method,
                constant_value=constant_value,
                fallback_to_global=config.fallback_to_global,
            )
            result[column_name] = fill_result.series
            if fill_result.filled_count:
                report["filled_columns"][column_name] = fill_result.filled_count
            if fill_result.remaining_missing:
                report["remaining_missing"][column_name] = fill_result.remaining_missing
        except ValueError:
            report["skipped_columns"].append(column_name)

    return result, report


def _normalize_text_series(series: pd.Series) -> pd.Series:
    normalized = series.astype("string")
    normalized = normalized.str.strip().str.lower()
    return normalized


def convert_series_type(series: pd.Series, target_type: str) -> ConversionResult:
    non_null_mask = series.notna()
    invalid_examples: list[str] = []

    if target_type == "string":
        converted = series.astype("string")
        return ConversionResult(converted=converted, invalid_count=0, invalid_examples=[], target_type=target_type)

    if target_type == "category":
        converted = series.astype("category")
        return ConversionResult(converted=converted, invalid_count=0, invalid_examples=[], target_type=target_type)

    if target_type == "auto":
        converted = series.convert_dtypes()
        return ConversionResult(converted=converted, invalid_count=0, invalid_examples=[], target_type=target_type)

    if target_type == "float":
        converted = pd.to_numeric(series, errors="coerce").astype("Float64")
        invalid_mask = non_null_mask & converted.isna()
    elif target_type == "integer":
        numeric_series = pd.to_numeric(series, errors="coerce")
        invalid_mask = non_null_mask & numeric_series.isna()
        fractional_mask = numeric_series.notna() & ~np.isclose(numeric_series % 1, 0)
        invalid_mask = invalid_mask | fractional_mask
        numeric_series = numeric_series.mask(fractional_mask)
        converted = numeric_series.astype("Int64")
    elif target_type == "datetime":
        converted = pd.to_datetime(series, errors="coerce", dayfirst=True)
        invalid_mask = non_null_mask & converted.isna()
    elif target_type == "boolean":
        normalized = _normalize_text_series(series)
        mapped = pd.Series(pd.NA, index=series.index, dtype="boolean")
        mapped.loc[normalized.isin(TRUE_VALUES)] = True
        mapped.loc[normalized.isin(FALSE_VALUES)] = False
        mapped.loc[~non_null_mask] = pd.NA
        converted = mapped
        invalid_mask = non_null_mask & mapped.isna()
    else:
        raise ValueError(f"Неподдерживаемый целевой тип: {target_type}")

    if invalid_mask.any():
        invalid_examples = [str(value) for value in series.loc[invalid_mask].dropna().astype(str).head(5).tolist()]

    return ConversionResult(
        converted=converted,
        invalid_count=int(invalid_mask.sum()),
        invalid_examples=invalid_examples,
        target_type=target_type,
    )


def infer_series_hints(series: pd.Series) -> list[str]:
    hints: list[str] = []
    clean_series = series.dropna()
    if clean_series.empty:
        return ["Столбец состоит только из пропусков."]

    if pd.api.types.is_numeric_dtype(series):
        hints.append("Похоже на числовой признак.")
    else:
        text_series = clean_series.astype("string")
        numeric_ratio = pd.to_numeric(text_series, errors="coerce").notna().mean()
        datetime_ratio = pd.to_datetime(text_series, errors="coerce", dayfirst=True).notna().mean()
        bool_ratio = _normalize_text_series(text_series).isin(TRUE_VALUES | FALSE_VALUES).mean()

        if numeric_ratio >= 0.8:
            hints.append("Большинство значений можно преобразовать в число.")
        if datetime_ratio >= 0.8:
            hints.append("Большинство значений можно преобразовать в дату/время.")
        if bool_ratio >= 0.8:
            hints.append("Большинство значений можно преобразовать в логический тип.")
        if not hints:
            hints.append("Похоже на категориальный или текстовый признак.")

    if clean_series.nunique(dropna=True) <= 12:
        hints.append("Небольшое число уникальных значений.")
    if clean_series.duplicated().mean() > 0.5:
        hints.append("Много повторяющихся значений.")

    return hints


def summarize_column(df: pd.DataFrame, column_name: str) -> dict[str, Any]:
    series = df[column_name]
    non_null = int(series.notna().sum())
    missing = int(series.isna().sum())
    unique = int(series.nunique(dropna=True))
    summary: dict[str, Any] = {
        "name": column_name,
        "dtype": str(series.dtype),
        "non_null": non_null,
        "missing": missing,
        "unique": unique,
        "hints": infer_series_hints(series),
    }

    if pd.api.types.is_numeric_dtype(series):
        clean_series = pd.to_numeric(series, errors="coerce").dropna()
        summary["role"] = "numeric"
        summary["min"] = float(clean_series.min()) if not clean_series.empty else np.nan
        summary["max"] = float(clean_series.max()) if not clean_series.empty else np.nan
        summary["mean"] = float(clean_series.mean()) if not clean_series.empty else np.nan
        summary["median"] = float(clean_series.median()) if not clean_series.empty else np.nan
    else:
        value_counts = series.dropna().astype(str).value_counts().head(5)
        summary["role"] = "categorical"
        summary["top_values"] = value_counts.to_dict()

    return summary


def build_column_description(df: pd.DataFrame, column_name: str) -> str:
    summary = summarize_column(df, column_name)
    lines = [
        f"Столбец: {summary['name']}",
        f"Тип: {summary['dtype']}",
        f"Непустых значений: {summary['non_null']}",
        f"Пропусков: {summary['missing']}",
        f"Уникальных значений: {summary['unique']}",
        "",
        "Подсказки:",
    ]
    lines.extend(f"- {hint}" for hint in summary["hints"])

    if summary["role"] == "numeric":
        lines.extend(
            [
                "",
                f"Минимум: {format_value(summary['min'])}",
                f"Максимум: {format_value(summary['max'])}",
                f"Среднее: {format_value(summary['mean'])}",
                f"Медиана: {format_value(summary['median'])}",
            ]
        )
    else:
        lines.append("")
        lines.append("Частые значения:")
        for key, value in summary.get("top_values", {}).items():
            lines.append(f"- {key}: {value}")

    return "\n".join(lines)


def build_dataset_overview_text(df: pd.DataFrame) -> str:
    rows, columns = df.shape
    total_missing = int(df.isna().sum().sum())
    duplicates = int(df.duplicated().sum())
    memory_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
    numeric_columns = [column for column in df.columns if pd.api.types.is_numeric_dtype(df[column])]
    categorical_columns = [column for column in df.columns if not pd.api.types.is_numeric_dtype(df[column])]

    lines = [
        "Обзор набора данных",
        "",
        f"Строк: {rows}",
        f"Столбцов: {columns}",
        f"Пропусков: {total_missing}",
        f"Дубликатов строк: {duplicates}",
        f"Использование памяти: {memory_mb:.2f} МБ",
        "",
        f"Числовых столбцов: {len(numeric_columns)}",
        f"Категориальных/текстовых столбцов: {len(categorical_columns)}",
    ]

    if numeric_columns:
        lines.append("")
        lines.append("Числовые столбцы:")
        lines.extend(f"- {column}" for column in numeric_columns[:10])

    if categorical_columns:
        lines.append("")
        lines.append("Категориальные/текстовые столбцы:")
        lines.extend(f"- {column}" for column in categorical_columns[:10])

    return "\n".join(lines)
