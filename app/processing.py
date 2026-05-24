"""Очистка данных, расчёт маркетинговых метрик по каналам/кампаниям.

Поддерживает любой набор доступных столбцов: метрики рассчитываются только
если у нас есть все необходимые для них исходные данные. Безопасно обрабатывает
пропуски, нечисловые значения и деление на ноль.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from app.data_mapping import NUMERIC_ROLES, ROLE_LABELS_RU


AGE_BINS = [0, 17, 24, 34, 44, 54, 64, 200]
AGE_LABELS = ["0-17", "18-24", "25-34", "35-44", "45-54", "55-64", "65+"]

# Имя столбца с объединённым ключом «канал + кампания», когда выбраны оба.
# Пользователю показывается label, само имя — техническое и не должно утекать в UI.
_COMBO_KEY = "__channel_campaign__"


@dataclass
class AnalysisResult:
    """Полный результат анализа."""

    mapping: dict
    group_col: str
    group_label: str
    channel_col: str | None
    campaign_col: str | None
    cleaned: pd.DataFrame
    metrics: pd.DataFrame
    metric_labels: dict
    available_metrics: list
    recommendations: list[str]
    warnings: list[str]
    age_table: pd.DataFrame | None = None
    month_table: pd.DataFrame | None = None
    rows_loaded: int = 0
    rows_dropped: int = 0
    rows_original: int = 0
    dropped_rows: pd.DataFrame = field(default_factory=pd.DataFrame)
    extra_category_col: str | None = None
    extra_category_label: str | None = None
    extra_summary: pd.DataFrame | None = None
    extra_metric: str | None = None  # ключевая метрика для extra-графика
    age_metric: str | None = None


# ---------- очистка ----------------------------------------------------------

_BOOL_TRUE_TOKENS = {"yes", "y", "true", "t", "да", "д", "истина", "+"}
_BOOL_FALSE_TOKENS = {"no", "n", "false", "f", "нет", "н", "ложь", "-"}


def _safe_to_numeric(series: pd.Series) -> pd.Series:
    """Конвертация значений в числа: запятые → точки, пробелы убираем, прочее → NaN."""
    if pd.api.types.is_bool_dtype(series):
        return series.astype(float)
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(float)
    cleaned = (
        series.astype(str)
        .str.strip()
        .str.replace(" ", "", regex=False)
        .str.replace(" ", "", regex=False)
        .str.replace(",", ".", regex=False)
    )
    cleaned = cleaned.replace({"": np.nan, "nan": np.nan, "None": np.nan})
    return pd.to_numeric(cleaned, errors="coerce")


def _to_boolean_numeric(series: pd.Series) -> tuple[pd.Series, bool]:
    """Пробует интерпретировать столбец как булевы признаки события.

    Возвращает (series_0_or_1, was_boolean). Распознаются: bool dtype,
    числовые столбцы со значениями только из {0, 1, NaN}, а также строки
    из множества {yes/no, y/n, true/false, да/нет, истина/ложь, 1/0, +/-}.
    """
    if series is None or len(series) == 0:
        return series, False
    if pd.api.types.is_bool_dtype(series):
        return series.astype(float), True
    if pd.api.types.is_numeric_dtype(series):
        numeric = series.astype(float)
        non_na = numeric.dropna()
        if not non_na.empty and set(non_na.unique()).issubset({0.0, 1.0}):
            return numeric, True
        return series, False
    normalized = series.astype(str).str.strip().str.lower()
    normalized = normalized.replace({"": np.nan, "nan": np.nan, "none": np.nan, "<na>": np.nan})
    non_na_tokens = set(normalized.dropna().unique())
    if not non_na_tokens:
        return series, False
    bool_tokens = _BOOL_TRUE_TOKENS | _BOOL_FALSE_TOKENS | {"1", "0", "1.0", "0.0"}
    if not non_na_tokens.issubset(bool_tokens):
        return series, False

    def _map(token):
        if isinstance(token, float) and pd.isna(token):
            return np.nan
        if token in _BOOL_TRUE_TOKENS or token in {"1", "1.0"}:
            return 1.0
        if token in _BOOL_FALSE_TOKENS or token in {"0", "0.0"}:
            return 0.0
        return np.nan

    return normalized.map(_map).astype(float), True


def _safe_divide(num: pd.Series, den: pd.Series) -> pd.Series:
    """Поэлементное деление с защитой от деления на 0 и от NaN."""
    den_safe = den.replace(0, np.nan)
    return num / den_safe


def clean_dataframe(df: pd.DataFrame, mapping: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Подготовка датасета: типизация по ролям, удаление мусорных строк."""
    df = df.copy()

    conv_col = mapping.get("conversions")
    if conv_col and conv_col in df.columns:
        converted, was_bool = _to_boolean_numeric(df[conv_col])
        if was_bool:
            df[conv_col] = converted
        else:
            df[conv_col] = _safe_to_numeric(df[conv_col])

    for role in NUMERIC_ROLES:
        if role == "conversions":
            continue
        col = mapping.get(role)
        if col and col in df.columns:
            df[col] = _safe_to_numeric(df[col])

    for role in ("channels", "campaigns", "month", "extra_category"):
        col = mapping.get(role)
        if col and col in df.columns:
            df[col] = df[col].astype("string").str.strip()
            df.loc[df[col].isin(["", "nan", "None", "<NA>"]), col] = pd.NA

    group_cols = [mapping[r] for r in ("channels", "campaigns") if mapping.get(r)]
    bad_mask = pd.Series(False, index=df.index)
    if group_cols:
        for col in group_cols:
            bad_mask |= df[col].isna()

    numeric_cols = [mapping[r] for r in NUMERIC_ROLES if mapping.get(r) and mapping[r] in df.columns]
    if numeric_cols:
        all_metrics_nan = df[numeric_cols].isna().all(axis=1)
        bad_mask |= all_metrics_nan

    dropped = df[bad_mask].copy()
    kept = df[~bad_mask].copy().reset_index(drop=True)
    return kept, dropped


# ---------- метрики ----------------------------------------------------------

METRIC_LABELS = {
    "displays": "Показы",
    "clicks": "Клики",
    "conversions": "Конверсии",
    "total_cost": "Общая стоимость",
    "placement_cost": "Стоимость размещения",
    "clicks_cost": "Стоимость за клики",
    "revenue": "Выручка",
    "CTR": "CTR, %",
    "CPC": "CPC",
    "CVR": "CVR, %",
    "CPA": "CPA",
    "cost_share": "Доля затрат, %",
    "conv_share": "Доля конверсий, %",
    "revenue_share": "Доля выручки, %",
    "value_per_conversion": "AOV",
    "ROAS": "ROAS",
}


def _compute_per_group(df: pd.DataFrame, mapping: dict, group_col: str) -> pd.DataFrame:
    """Агрегирует исходные метрики по группе и считает производные KPI."""
    aggregations = {}

    for role in (
        "displays",
        "clicks",
        "conversions",
        "total_cost",
        "placement_cost",
        "revenue",
        "clicks_cost",
    ):
        col = mapping.get(role)
        if col and col in df.columns:
            aggregations[role] = (col, "sum")

    if not aggregations:
        return pd.DataFrame()

    grouped = df.groupby(group_col, dropna=True).agg(**aggregations).reset_index()

    cost_total = None
    if "total_cost" in grouped:
        cost_total = grouped["total_cost"]
    elif "placement_cost" in grouped and "clicks_cost" in grouped:
        cost_total = grouped["placement_cost"] + grouped["clicks_cost"]
    elif "placement_cost" in grouped:
        cost_total = grouped["placement_cost"]
    elif "clicks_cost" in grouped:
        cost_total = grouped["clicks_cost"]

    if "clicks" in grouped and "displays" in grouped:
        grouped["CTR"] = _safe_divide(grouped["clicks"], grouped["displays"]) * 100

    if cost_total is not None and "clicks" in grouped:
        grouped["CPC"] = _safe_divide(cost_total, grouped["clicks"])

    if "conversions" in grouped and "clicks" in grouped:
        grouped["CVR"] = _safe_divide(grouped["conversions"], grouped["clicks"]) * 100

    if cost_total is not None and "conversions" in grouped:
        grouped["CPA"] = _safe_divide(cost_total, grouped["conversions"])

    if cost_total is not None:
        total_cost_sum = cost_total.sum(skipna=True)
        if total_cost_sum and not pd.isna(total_cost_sum) and total_cost_sum > 0:
            grouped["cost_share"] = cost_total / total_cost_sum * 100

    if "conversions" in grouped:
        total_conv = grouped["conversions"].sum(skipna=True)
        if total_conv and not pd.isna(total_conv) and total_conv > 0:
            grouped["conv_share"] = grouped["conversions"] / total_conv * 100

    if "revenue" in grouped and "conversions" in grouped:
        grouped["AOV"] = _safe_divide(grouped["revenue"], grouped["conversions"])

    if "revenue" in grouped:
        total_revenue = grouped["revenue"].sum(skipna=True)
        if total_revenue and not pd.isna(total_revenue) and total_revenue > 0:
            grouped["revenue_share"] = grouped["revenue"] / total_revenue * 100

        if cost_total is not None:
            grouped["ROAS"] = _safe_divide(grouped["revenue"], cost_total)

    return grouped


def _build_age_table(
    df: pd.DataFrame,
    mapping: dict,
    group_col: str,
    channel_col: str | None,
    campaign_col: str | None,
) -> tuple[pd.DataFrame | None, str | None]:
    age_col = mapping.get("age")
    if not age_col or age_col not in df.columns:
        return None, None

    age_series = _safe_to_numeric(df[age_col])
    if age_series.notna().sum() == 0:
        return None, None

    work = df.copy()
    work["__age_group__"] = pd.cut(
        age_series,
        bins=AGE_BINS,
        labels=AGE_LABELS,
        include_lowest=True,
    )

    group_fields = ["__age_group__"]

    if campaign_col and campaign_col in work.columns:
        group_fields.append(campaign_col)
    if channel_col and channel_col in work.columns:
        group_fields.append(channel_col)

    if len(group_fields) == 1 and group_col in work.columns:
        group_fields.append(group_col)

    aggregations: dict[str, tuple[str, str]] = {}

    for role in (
        "displays",
        "clicks",
        "conversions",
        "revenue",
        "total_cost",
        "placement_cost",
    ):
        col = mapping.get(role)
        if col and col in work.columns:
            aggregations[role] = (col, "sum")

    cpc_col = mapping.get("cpc")
    if cpc_col and cpc_col in work.columns:
        aggregations["cpc_avg"] = (cpc_col, "mean")

    if not aggregations:
        result = (
            work.groupby(group_fields, dropna=True, observed=True)
            .size()
            .reset_index(name="Записей")
        )
        result = result.rename(columns={"__age_group__": "Возрастная группа"})
        return result, None

    result = (
        work.groupby(group_fields, dropna=True, observed=True)
        .agg(**aggregations)
        .reset_index()
    )

    result = result.rename(columns={"__age_group__": "Возрастная группа"})

    if "clicks" in result.columns and "displays" in result.columns:
        result["CTR"] = _safe_divide(result["clicks"], result["displays"]) * 100

    cost_for_cpc = None
    if "total_cost" in result.columns:
        cost_for_cpc = result["total_cost"]
    elif "placement_cost" in result.columns:
        cost_for_cpc = result["placement_cost"]

    if cost_for_cpc is not None and "clicks" in result.columns:
        result["CPC"] = _safe_divide(cost_for_cpc, result["clicks"])
    elif "cpc_avg" in result.columns:
        result["CPC"] = result["cpc_avg"]

    if "conversions" in result.columns and "clicks" in result.columns:
        result["CVR"] = _safe_divide(result["conversions"], result["clicks"]) * 100

    cost_for_cpa = None
    if "total_cost" in result.columns:
        cost_for_cpa = result["total_cost"]
    elif "placement_cost" in result.columns:
        cost_for_cpa = result["placement_cost"]

    if cost_for_cpa is not None and "conversions" in result.columns:
        result["CPA"] = _safe_divide(cost_for_cpa, result["conversions"])

    if "revenue" in result.columns and "conversions" in result.columns:
        result["AOV"] = _safe_divide(result["revenue"], result["conversions"])

    if "revenue" in result.columns:
        if "total_cost" in result.columns:
            result["ROAS"] = _safe_divide(result["revenue"], result["total_cost"])
        elif "placement_cost" in result.columns:
            result["ROAS"] = _safe_divide(result["revenue"], result["placement_cost"])

    cost_total_col = None
    if "total_cost" in result.columns:
        cost_total_col = "total_cost"
    elif "placement_cost" in result.columns:
        cost_total_col = "placement_cost"

    if cost_total_col is not None:
        total_cost = result[cost_total_col].sum(skipna=True)
        if total_cost and not pd.isna(total_cost) and total_cost > 0:
            result["cost_share"] = result[cost_total_col] / total_cost * 100

    if "conversions" in result.columns:
        total_conv = result["conversions"].sum(skipna=True)
        if total_conv and not pd.isna(total_conv) and total_conv > 0:
            result["conv_share"] = result["conversions"] / total_conv * 100

    if "revenue" in result.columns:
        total_revenue = result["revenue"].sum(skipna=True)
        if total_revenue and not pd.isna(total_revenue) and total_revenue > 0:
            result["revenue_share"] = result["revenue"] / total_revenue * 100

    age_metric = None
    for metric in ("conversions", "revenue", "clicks", "displays"):
        if metric in result.columns and result[metric].notna().any():
            age_metric = metric
            break

    metric_order = [
        "Возрастная группа",
        campaign_col,
        channel_col,
        group_col,
        "displays",
        "clicks",
        "conversions",
        "revenue",
        "total_cost",
        "placement_cost",
        "cpc_avg",
        "CTR",
        "CPC",
        "CVR",
        "CPA",
        "AOV",
        "ROAS",
        "cost_share",
        "conv_share",
        "revenue_share",
        "Записей",
    ]
    metric_order = [c for c in metric_order if c and c in result.columns]
    other_cols = [c for c in result.columns if c not in metric_order]
    result = result[metric_order + other_cols]

    return result, age_metric

def _build_month_table(
    df: pd.DataFrame,
    mapping: dict,
    group_col: str,
    channel_col: str | None = None,
    campaign_col: str | None = None,
) -> pd.DataFrame | None:
    month_col = mapping.get("month")
    if not month_col or month_col not in df.columns:
        return None

    work = df.dropna(subset=[month_col]).copy()
    if work.empty:
        return None

    work["__month_label__"] = work[month_col].astype("string").str.strip()
    if work["__month_label__"].isna().all():
        return None

    aggregations = {}
    for role in ("displays", "clicks", "conversions", "total_cost", "placement_cost"):
        col = mapping.get(role)
        if col and col in work.columns:
            aggregations[role] = (col, "sum")

    cpc_col = mapping.get("cpc")
    if cpc_col and cpc_col in work.columns:
        aggregations["cpc_avg"] = (cpc_col, "mean")

    if not aggregations:
        return None

    group_fields = ["__month_label__", group_col]

    if campaign_col and campaign_col in work.columns and campaign_col != group_col:
        group_fields.append(campaign_col)

    if channel_col and channel_col in work.columns and channel_col != group_col:
        group_fields.append(channel_col)

    group_fields = list(dict.fromkeys(group_fields))

    grouped = (
        work.groupby(group_fields, dropna=True)
        .agg(**aggregations)
        .reset_index()
        .rename(columns={"__month_label__": "Месяц"})
    )

    if "clicks" in grouped.columns and "displays" in grouped.columns:
        grouped["CTR"] = _safe_divide(grouped["clicks"], grouped["displays"]) * 100

    cost_for_cpc = None
    if "total_cost" in grouped.columns:
        cost_for_cpc = grouped["total_cost"]
    elif "placement_cost" in grouped.columns:
        cost_for_cpc = grouped["placement_cost"]

    if cost_for_cpc is not None and "clicks" in grouped.columns:
        grouped["CPC"] = _safe_divide(cost_for_cpc, grouped["clicks"])
    elif "cpc_avg" in grouped.columns:
        grouped["CPC"] = grouped["cpc_avg"]

    if "conversions" in grouped.columns and "clicks" in grouped.columns:
        grouped["CVR"] = _safe_divide(grouped["conversions"], grouped["clicks"]) * 100

    cost_for_cpa = None
    if "total_cost" in grouped.columns:
        cost_for_cpa = grouped["total_cost"]
    elif "placement_cost" in grouped.columns:
        cost_for_cpa = grouped["placement_cost"]

    if cost_for_cpa is not None and "conversions" in grouped.columns:
        grouped["CPA"] = _safe_divide(cost_for_cpa, grouped["conversions"])

    month_order = {
        "январь": 1, "янв": 1, "january": 1, "jan": 1, "1": 1, "01": 1,
        "февраль": 2, "фев": 2, "february": 2, "feb": 2, "2": 2, "02": 2,
        "март": 3, "мар": 3, "march": 3, "mar": 3, "3": 3, "03": 3,
        "апрель": 4, "апр": 4, "april": 4, "apr": 4, "4": 4, "04": 4,
        "май": 5, "may": 5, "5": 5, "05": 5,
        "июнь": 6, "июн": 6, "june": 6, "jun": 6, "6": 6, "06": 6,
        "июль": 7, "июл": 7, "july": 7, "jul": 7, "7": 7, "07": 7,
        "август": 8, "авг": 8, "august": 8, "aug": 8, "8": 8, "08": 8,
        "сентябрь": 9, "сен": 9, "september": 9, "sep": 9, "9": 9, "09": 9,
        "октябрь": 10, "окт": 10, "october": 10, "oct": 10, "10": 10,
        "ноябрь": 11, "ноя": 11, "november": 11, "nov": 11, "11": 11,
        "декабрь": 12, "дек": 12, "december": 12, "dec": 12, "12": 12,
    }

    grouped["__month_sort__"] = (
        grouped["Месяц"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map(month_order)
    )

    sort_cols = ["__month_sort__", "Месяц"]
    if campaign_col and campaign_col in grouped.columns:
        sort_cols.append(campaign_col)
    if channel_col and channel_col in grouped.columns:
        sort_cols.append(channel_col)

    grouped = grouped.sort_values(sort_cols, na_position="last").reset_index(drop=True)
    return grouped.drop(columns=["__month_sort__"], errors="ignore")

def _pick_key_metric(mapping: dict) -> tuple[str | None, str | None]:
    """Возвращает (role, исходный_столбец) для главной метрики сводки.

    Порядок предпочтения: conversions → clicks → displays → total_cost → placement_cost.
    """
    for role in ("conversions", "clicks", "displays", "total_cost", "placement_cost"):
        col = mapping.get(role)
        if col:
            return role, col
    return None, None


def _build_extra_summary(
    df: pd.DataFrame,
    mapping: dict,
    extra_col: str | None,
) -> tuple[pd.DataFrame | None, str | None]:
    """Сводка по дополнительной категории без разбивки по каналам.

    В сводку попадают:
    - базовые суммы: displays, clicks, conversions, total_cost, placement_cost, revenue
    - KPI: CTR, CPC, CVR, CPA, AOV, ROAS
    - доли: cost_share, conv_share, revenue_share

    Возвращает:
    - summary: DataFrame по extra_col
    - extra_metric: ключевая метрика для extra-графика
      (приоритет: conversions -> revenue -> clicks -> displays)
    """
    if not extra_col or extra_col not in df.columns:
        return None, None

    work = df.dropna(subset=[extra_col]).copy()
    if work.empty:
        return None, None

    aggregations: dict[str, tuple[str, str]] = {}

    for role in (
        "displays",
        "clicks",
        "conversions",
        "total_cost",
        "placement_cost",
        "revenue",
    ):
        col = mapping.get(role)
        if col and col in work.columns:
            aggregations[role] = (col, "sum")

    cpc_col = mapping.get("cpc")
    if cpc_col and cpc_col in work.columns:
        aggregations["cpc_avg"] = (cpc_col, "mean")

    if not aggregations:
        return None, None

    summary = (
        work.groupby(extra_col, dropna=True)
        .agg(**aggregations)
        .reset_index()
    )

    if "clicks" in summary.columns and "displays" in summary.columns:
        summary["CTR"] = _safe_divide(summary["clicks"], summary["displays"]) * 100

    cost_for_cpc = None
    if "total_cost" in summary.columns:
        cost_for_cpc = summary["total_cost"]
    elif "placement_cost" in summary.columns:
        cost_for_cpc = summary["placement_cost"]

    if cost_for_cpc is not None and "clicks" in summary.columns:
        summary["CPC"] = _safe_divide(cost_for_cpc, summary["clicks"])
    elif "cpc_avg" in summary.columns:
        summary["CPC"] = summary["cpc_avg"]

    if "conversions" in summary.columns and "clicks" in summary.columns:
        summary["CVR"] = _safe_divide(summary["conversions"], summary["clicks"]) * 100

    cost_for_cpa = None
    if "total_cost" in summary.columns:
        cost_for_cpa = summary["total_cost"]
    elif "placement_cost" in summary.columns:
        cost_for_cpa = summary["placement_cost"]

    if cost_for_cpa is not None and "conversions" in summary.columns:
        summary["CPA"] = _safe_divide(cost_for_cpa, summary["conversions"])

    if "revenue" in summary.columns and "conversions" in summary.columns:
        summary["AOV"] = _safe_divide(summary["revenue"], summary["conversions"])

    if "revenue" in summary.columns:
        if "total_cost" in summary.columns:
            summary["ROAS"] = _safe_divide(summary["revenue"], summary["total_cost"])
        elif "placement_cost" in summary.columns:
            summary["ROAS"] = _safe_divide(summary["revenue"], summary["placement_cost"])

    cost_total_col = None
    if "total_cost" in summary.columns:
        cost_total_col = "total_cost"
    elif "placement_cost" in summary.columns:
        cost_total_col = "placement_cost"

    if cost_total_col is not None:
        total_cost = summary[cost_total_col].sum(skipna=True)
        if total_cost and not pd.isna(total_cost) and total_cost > 0:
            summary["cost_share"] = summary[cost_total_col] / total_cost * 100

    if "conversions" in summary.columns:
        total_conv = summary["conversions"].sum(skipna=True)
        if total_conv and not pd.isna(total_conv) and total_conv > 0:
            summary["conv_share"] = summary["conversions"] / total_conv * 100

    if "revenue" in summary.columns:
        total_revenue = summary["revenue"].sum(skipna=True)
        if total_revenue and not pd.isna(total_revenue) and total_revenue > 0:
            summary["revenue_share"] = summary["revenue"] / total_revenue * 100

    extra_metric = None
    for metric in ("conversions", "revenue", "clicks", "displays"):
        if metric in summary.columns and summary[metric].notna().any():
            extra_metric = metric
            break

    if extra_metric is None:
        return None, None

    summary = summary.sort_values(extra_metric, ascending=False).reset_index(drop=True)
    return summary, extra_metric


# ---------- рекомендации (DSS) ----------------------------------------------

def _format_value(name: str, value) -> str:
    if value is None or pd.isna(value):
        return "—"
    if name in ("CTR", "CVR", "cost_share", "conv_share"):
        return f"{value:.2f}%"
    if name in ("CPC", "CPA", "cpc_avg"):
        return f"{value:,.2f}".replace(",", " ")
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}".replace(",", " ")
    return f"{float(value):,.2f}".replace(",", " ")


def _top_bottom(metrics: pd.DataFrame, col: str, ascending=False):
    if col not in metrics.columns:
        return None, None
    valid = metrics.dropna(subset=[col])
    if valid.empty:
        return None, None
    sorted_df = valid.sort_values(col, ascending=ascending)
    return sorted_df.iloc[0], sorted_df.iloc[-1]


def build_recommendations(
    metrics: pd.DataFrame,
    mapping: dict,
    group_col: str,
    group_label: str,
) -> tuple[list[str], list[str]]:
    """Рекомендации и предупреждения на русском. Слово «группа» избегается:
    используется «сегмент» либо «канал/кампания» по контексту."""
    recs: list[str] = []
    warns: list[str] = []

    if metrics.empty:
        warns.append("Нет данных для построения рекомендаций.")
        return recs, warns

    if len(metrics) < 2:
        warns.append("В выборке только один сегмент — сравнительные выводы ограничены.")

    if "CTR" in metrics.columns and metrics["CTR"].notna().any():
        top, bottom = _top_bottom(metrics, "CTR", ascending=False)
        if top is not None:
            recs.append(
                f"Лучший CTR — у «{top[group_col]}» ({top['CTR']:.2f}%). "
                f"Креативы и таргетинг этого сегмента стоит изучить и переиспользовать."
            )
        if bottom is not None and bottom[group_col] != (top[group_col] if top is not None else None):
            recs.append(
                f"Самый низкий CTR — у «{bottom[group_col]}» ({bottom['CTR']:.2f}%). "
                f"Проверьте релевантность объявлений и качество аудитории."
            )

    if "CVR" in metrics.columns and metrics["CVR"].notna().any():
        top, bottom = _top_bottom(metrics, "CVR", ascending=False)
        if top is not None:
            recs.append(
                f"Самая высокая конверсия (CVR) — у «{top[group_col]}» ({top['CVR']:.2f}%). "
                f"Этот сегмент эффективно превращает трафик в действия — рассмотрите увеличение бюджета."
            )
        if bottom is not None and bottom[group_col] != (top[group_col] if top is not None else None):
            recs.append(
                f"Низкая конверсия у «{bottom[group_col]}» ({bottom['CVR']:.2f}%). "
                f"Стоит проверить посадочную страницу, оффер и сегментацию аудитории."
            )

    if "CPA" in metrics.columns and metrics["CPA"].notna().any():
        top, bottom = _top_bottom(metrics, "CPA", ascending=True)
        if top is not None:
            recs.append(
                f"Самая дешёвая конверсия (CPA) — у «{top[group_col]}» ({_format_value('CPA', top['CPA'])}). "
                f"Это приоритетный сегмент для масштабирования."
            )
        if bottom is not None and bottom[group_col] != (top[group_col] if top is not None else None):
            recs.append(
                f"Самая дорогая конверсия — у «{bottom[group_col]}» ({_format_value('CPA', bottom['CPA'])}). "
                f"Пересмотрите ставки, креативы или аудиторию; при отсутствии улучшений сократите бюджет."
            )

    if "CPC" in metrics.columns and metrics["CPC"].notna().any():
        top, bottom = _top_bottom(metrics, "CPC", ascending=True)
        if bottom is not None and top is not None and bottom[group_col] != top[group_col]:
            recs.append(
                f"Стоимость клика: дешевле всего у «{top[group_col]}» ({_format_value('CPC', top['CPC'])}), "
                f"дороже всего — у «{bottom[group_col]}» ({_format_value('CPC', bottom['CPC'])})."
            )

    if "cost_share" in metrics.columns and "conv_share" in metrics.columns:
        diff = metrics["cost_share"] - metrics["conv_share"]
        overspend = metrics[(diff > 10) & metrics["cost_share"].notna()]
        for _, row in overspend.iterrows():
            recs.append(
                f"«{row[group_col]}» расходует {row['cost_share']:.1f}% бюджета, "
                f"но приносит лишь {row['conv_share']:.1f}% конверсий — есть смысл перераспределить бюджет."
            )
        underfunded = metrics[(diff < -10) & metrics["conv_share"].notna()]
        for _, row in underfunded.iterrows():
            recs.append(
                f"«{row[group_col]}» приносит {row['conv_share']:.1f}% конверсий "
                f"всего при {row['cost_share']:.1f}% бюджета — кандидат на увеличение инвестиций."
            )

    if "CTR" in metrics.columns and metrics["CTR"].notna().sum() >= 3:
        mean_ctr = metrics["CTR"].mean()
        weak = metrics[metrics["CTR"] < mean_ctr * 0.5]
        for _, row in weak.iterrows():
            recs.append(
                f"CTR «{row[group_col]}» ({row['CTR']:.2f}%) более чем в 2 раза ниже среднего ({mean_ctr:.2f}%). "
                f"Стоит протестировать новые креативы."
            )

    needed = {
        "CTR": ("displays", "clicks"),
        "CVR": ("clicks", "conversions"),
        "CPA": ("conversions", "total_cost / placement_cost"),
        "CPC": ("clicks", "total_cost / placement_cost / cpc"),
    }
    available_cols = set(metrics.columns)
    for metric, sources in needed.items():
        if metric not in available_cols:
            warns.append(
                f"Метрика {metric} не рассчитана: не выбраны столбцы для ({', '.join(sources)})."
            )

    if not recs:
        recs.append(
            "Имеющихся данных недостаточно для содержательных рекомендаций. "
            "Сопоставьте больше столбцов (клики, конверсии, стоимость) и повторите анализ."
        )

    return recs, warns


# ---------- основная точка входа --------------------------------------------

def process_data(df: pd.DataFrame, mapping: dict) -> AnalysisResult:
    """Главный конвейер обработки."""
    channel_col = mapping.get("channels")
    campaign_col = mapping.get("campaigns")
    extra_col = mapping.get("extra_category")

    if not channel_col and not campaign_col:
        raise ValueError("Выберите столбец «Каналы» или «Кампании» для группировки.")

    work, dropped = clean_dataframe(df, mapping)

    if channel_col and campaign_col:
        group_col = _COMBO_KEY
        # Видимый сегмент — это «канал в кампании», без слова «группа».
        # Для подписей в графиках используются отдельные channel_col/campaign_col.
        work[group_col] = work[channel_col].astype(str) + " · " + work[campaign_col].astype(str)
        group_label = "Канал и кампания"
    elif channel_col:
        group_col = channel_col
        group_label = ROLE_LABELS_RU["channels"]
    else:
        group_col = campaign_col
        group_label = ROLE_LABELS_RU["campaigns"]

    metrics = _compute_per_group(work, mapping, group_col)

    if metrics.empty:
        raise ValueError(
            "Нет числовых данных для расчёта. "
            "Выберите хотя бы одно числовое поле: показы, клики, конверсии или стоимость."
        )

    # Для красивых многоуровневых меток в графиках сохраним столбцы канала и кампании
    # рядом с агрегированными метриками.
    if channel_col and campaign_col:
        labels_df = work[[group_col, channel_col, campaign_col]].drop_duplicates(group_col)
        metrics = metrics.merge(labels_df, on=group_col, how="left")

    for sort_col in ("conversions", "clicks", "displays", "total_cost"):
        if sort_col in metrics.columns and metrics[sort_col].notna().any():
            # При двух размерностях группируем по кампании, чтобы каналы кампании шли подряд
            if channel_col and campaign_col:
                metrics = metrics.sort_values(
                    [campaign_col, channel_col],
                    ascending=[True, True]
                ).reset_index(drop=True)
            else:
                metrics = metrics.sort_values(sort_col, ascending=False).reset_index(drop=True)
            break



    available = [
        c for c in metrics.columns
        if c not in {group_col, channel_col, campaign_col} and c is not None
    ]
    recommendations, warnings = build_recommendations(metrics, mapping, group_col, group_label)
    age_table, age_metric = _build_age_table(
        work,
        mapping,
        group_col,
        channel_col,
        campaign_col,
    )
    month_table = _build_month_table(
        work,
        mapping,
        group_col=group_col,
        channel_col=channel_col,
        campaign_col=campaign_col,
    )
    extra_summary, extra_metric = _build_extra_summary(work, mapping, extra_col)

    month_table = _build_month_table(work, mapping, group_col, channel_col, campaign_col)

    return AnalysisResult(
        mapping=mapping,
        group_col=group_col,
        group_label=group_label,
        channel_col=channel_col,
        campaign_col=campaign_col,
        cleaned=work,
        metrics=metrics,
        metric_labels=METRIC_LABELS,
        available_metrics=available,
        recommendations=recommendations,
        warnings=warnings,
        age_table=age_table,
        month_table=month_table,
        rows_loaded=len(work),
        rows_dropped=len(dropped),
        rows_original=len(work) + len(dropped),
        dropped_rows=dropped,
        extra_category_col=extra_col,
        extra_category_label=str(extra_col) if extra_col else None,
        extra_summary=extra_summary,
        extra_metric=extra_metric,
        age_metric=age_metric,
    )


def format_metrics_for_display(result: AnalysisResult) -> pd.DataFrame:
    """Возвращает копию таблицы метрик с человекочитаемыми названиями и форматированием.

    Если в результате есть отдельные каналы и кампании, в таблице тоже показываются
    они отдельно, без «склеенного» технического ключа.
    """
    df = result.metrics.copy()

    if result.channel_col and result.campaign_col:
        # Прячем технический объединённый ключ, оставляем человеческие столбцы.
        if result.group_col in df.columns:
            df = df.drop(columns=[result.group_col])
        front_cols = [c for c in (result.campaign_col, result.channel_col) if c in df.columns]
        other = [c for c in df.columns if c not in front_cols]
        df = df[front_cols + other]
        rename = {
            result.campaign_col: ROLE_LABELS_RU["campaigns"],
            result.channel_col: ROLE_LABELS_RU["channels"],
        }
    else:
        rename = {result.group_col: result.group_label}

    for raw, label in result.metric_labels.items():
        if raw in df.columns:
            rename[raw] = label
    df = df.rename(columns=rename)

    for raw, label in result.metric_labels.items():
        if label in df.columns:
            df[label] = df[label].apply(lambda v: _format_value(raw, v))
    return df
