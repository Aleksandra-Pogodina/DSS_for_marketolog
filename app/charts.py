"""Графики для DOCX-отчёта: matplotlib/seaborn -> PNG.

Каждый build_* возвращает объект Chart с заголовком, путём к PNG и описанием.
PNG-формат удобен для встраивания в DOCX без повторной отрисовки.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from app.processing import AnalysisResult, AGE_LABELS

sns.set_theme(style="whitegrid")
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.titlesize"] = 13
plt.rcParams["axes.labelsize"] = 11


@dataclass
class Chart:
    title: str
    path: str
    description: str = ""


def _new_path(prefix: str, tmpdir: str) -> str:
    fd, path = tempfile.mkstemp(prefix=f"{prefix}_", suffix=".png", dir=tmpdir)
    os.close(fd)
    return path


def _save(fig, path: str) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def _truncate(s, limit: int = 22) -> str:
    s = str(s)
    return s if len(s) <= limit else s[: limit - 1] + "…"


def _has_cols(df: pd.DataFrame, cols: list[str | None]) -> bool:
    real = [c for c in cols if c]
    return bool(real) and all(c in df.columns for c in real)


def _has_metric(df: pd.DataFrame, col: str) -> bool:
    return col in df.columns and df[col].notna().any()


def _segment_x_labels(result: AnalysisResult, df: pd.DataFrame) -> list[str]:
    """Для vertical bar: две строки (канал / кампания), если доступны оба поля."""
    if (
        result.channel_col
        and result.campaign_col
        and _has_cols(df, [result.channel_col, result.campaign_col])
    ):
        return [
            f"{_truncate(chan, 14)}\n{_truncate(camp, 14)}"
            for camp, chan in zip(
                df[result.campaign_col].astype(str),
                df[result.channel_col].astype(str),
            )
        ]
    return [_truncate(v, 18) for v in df[result.group_col].astype(str)]


def _campaign_group_annotations(result: AnalysisResult, df: pd.DataFrame) -> list[tuple[float, str]]:
    if not (
        result.channel_col
        and result.campaign_col
        and _has_cols(df, [result.channel_col, result.campaign_col])
        and not df.empty
    ):
        return []

    campaigns = df[result.campaign_col].astype(str).tolist()
    groups: list[tuple[int, int, str]] = []

    start = 0
    current = campaigns[0]
    for i, camp in enumerate(campaigns[1:], start=1):
        if camp != current:
            groups.append((start, i - 1, current))
            start = i
            current = camp
    groups.append((start, len(campaigns) - 1, current))

    return [((start + end) / 2, _truncate(camp, 18)) for start, end, camp in groups]


def _month_order_key(value: object) -> tuple[int, str]:
    text = str(value).strip().lower()

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

    return (month_order.get(text, 999), text)


def _line_by_month_channel(
    result: AnalysisResult,
    metric_cols: list[str],
    title: str,
    tmpdir: str,
) -> Chart | None:
    """
    Линейный график по месяцам для каждого канала отдельно.

    Строится только если:
    - выбран месяц (result.month_table не пустой),
    - выбраны и кампания, и канал,
    - в month_table есть колонки 'Месяц', campaign_col, channel_col.
    """
    mt = result.month_table
    if mt is None or mt.empty:
        return None

    if not (
        result.campaign_col
        and result.channel_col
        and _has_cols(mt, ["Месяц", result.campaign_col, result.channel_col])
    ):
        return None

    present = [c for c in metric_cols if _has_metric(mt, c)]
    if not present:
        return None

    work = mt.copy()
    work["__month_sort__"] = work["Месяц"].map(lambda x: _month_order_key(x)[0])

    campaigns = (
        work[result.campaign_col]
        .dropna()
        .astype(str)
        .drop_duplicates()
        .tolist()
    )
    if not campaigns:
        return None

    channels = (
        work[result.channel_col]
        .dropna()
        .astype(str)
        .drop_duplicates()
        .tolist()
    )
    if not channels:
        return None

    n_rows = len(present)
    fig, axes = plt.subplots(
        nrows=n_rows,
        ncols=1,
        figsize=(max(10, 0.85 * max(len(campaigns), 4) + 5), max(3.8 * n_rows, 4.4)),
        squeeze=False,
    )
    axes = axes.ravel()
    palette = sns.color_palette("tab10", n_colors=max(len(channels), 3))

    for ax, metric in zip(axes, present):
        metric_data = work.dropna(subset=[metric]).copy()
        if metric_data.empty:
            ax.set_visible(False)
            continue

        pivot = metric_data.pivot_table(
            index=result.campaign_col,
            columns=result.channel_col,
            values=metric,
            aggfunc="sum",
            fill_value=np.nan,
            observed=True,
        )

        if pivot.empty:
            ax.set_visible(False)
            continue

        pivot = pivot.reindex(campaigns)
        x = np.arange(len(pivot.index))

        for i, channel in enumerate(pivot.columns.astype(str)):
            y = pivot[channel].astype(float).values
            if np.all(pd.isna(y)):
                continue

            ax.plot(
                x,
                y,
                marker="o",
                linewidth=2.0,
                markersize=5,
                label=_truncate(channel, 18),
                color=palette[i % len(palette)],
            )

        month_labels = []
        for campaign in pivot.index.astype(str):
            subset = (
                metric_data[metric_data[result.campaign_col].astype(str) == campaign]
                .copy()
                .sort_values("__month_sort__")
            )
            month_vals = subset["Месяц"].dropna().astype(str).drop_duplicates().tolist()
            if month_vals:
                month_labels.append(_truncate(" / ".join(month_vals), 18))
            else:
                month_labels.append(_truncate(campaign, 18))

        ax.set_title(result.metric_labels.get(metric, metric), loc="left")
        ax.set_xticks(x)
        ax.set_xticklabels(month_labels, rotation=0, fontsize=9)
        ax.set_ylabel(result.metric_labels.get(metric, metric))
        ax.set_xlabel("Месяцы")
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(title="Канал", fontsize=8, title_fontsize=9, loc="best")

    fig.suptitle(title, fontsize=14, y=1.02)
    path = _new_path("line_month_channel", tmpdir)
    _save(fig, path)
    return Chart(
        title=title,
        path=path,
        description=(
            "Линейные графики построены по месяцам: для каждой метрики показаны линии каналов. "
            "График создаётся только если выбраны месяц, кампания и канал."
        ),
    )


def _bar_chart(
    result: AnalysisResult,
    value_col: str,
    title: str,
    ylabel: str,
    tmpdir: str,
    color: str = "steelblue",
) -> Chart | None:
    m = result.metrics
    if value_col not in m.columns:
        return None

    data = m.dropna(subset=[value_col]).copy()
    if data.empty:
        return None

    has_campaign_and_channel = (
        bool(result.channel_col)
        and bool(result.campaign_col)
        and _has_cols(data, [result.channel_col, result.campaign_col])
    )

    if has_campaign_and_channel:
        data = data.sort_values(
            [result.campaign_col, result.channel_col],
            ascending=[True, True],
        ).copy()
        x_labels = [_truncate(v, 14) for v in data[result.channel_col].astype(str)]
    else:
        data = data.sort_values(value_col, ascending=False).copy()
        x_labels = [_truncate(v, 18) for v in data[result.group_col].astype(str)]

    values = data[value_col].astype(float).fillna(0)
    x = np.arange(len(data))

    fig_width = max(9, 0.75 * len(data) + 3)
    fig, ax = plt.subplots(figsize=(fig_width, 6))
    bars = ax.bar(x, values, color=color, width=0.72)

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=0, fontsize=9)

    y_max = float(values.max()) if len(values) else 0.0
    top_pad = max(y_max * 0.12, 1.0)
    ax.set_ylim(0, y_max + top_pad)

    for rect, v in zip(bars, values):
        ax.text(
            rect.get_x() + rect.get_width() / 2,
            rect.get_height() + top_pad * 0.15,
            f"{v:,.2f}".replace(",", " "),
            ha="center",
            va="bottom",
            fontsize=9,
        )

    if has_campaign_and_channel:
        campaigns = data[result.campaign_col].astype(str).tolist()

        for center, camp_label in _campaign_group_annotations(result, data):
            ax.text(
                center,
                -0.18,
                camp_label,
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="top",
                fontsize=9,
                fontweight="bold",
            )

        for i in range(1, len(campaigns)):
            if campaigns[i] != campaigns[i - 1]:
                ax.axvline(
                    x=i - 0.5,
                    color="gray",
                    linewidth=1,
                    alpha=0.45,
                    zorder=0,
                )

        fig.subplots_adjust(bottom=0.28)

    path = _new_path(value_col, tmpdir)
    _save(fig, path)
    return Chart(
        title=title,
        path=path,
        description=f"Сегменты упорядочены по «{ylabel}».",
    )


def _share_chart(result: AnalysisResult, tmpdir: str) -> Chart | None:
    m = result.metrics
    cols = [c for c in ("cost_share", "revenue_share", "conv_share") if _has_metric(m, c)]
    if len(cols) < 2:
        return None

    data = m.dropna(subset=cols, how="all").copy()
    if data.empty:
        return None

    sort_metric = "conv_share" if "conv_share" in cols else cols[0]
    if result.channel_col and result.campaign_col and _has_cols(data, [result.channel_col, result.campaign_col]):
        data = data.sort_values([result.campaign_col, sort_metric], ascending=[True, False]).copy()
    else:
        data = data.sort_values(sort_metric, ascending=False).copy()

    labels = _segment_x_labels(result, data)
    n = len(labels)
    width = 0.24
    fig, ax = plt.subplots(figsize=(max(8, 0.8 * n + 3), 5.4))
    x = np.arange(n)

    pretty = {
        "cost_share": "Доля затрат, %",
        "revenue_share": "Доля выручки, %",
        "conv_share": "Доля конверсий, %",
    }
    colors = {
        "cost_share": "#CC6677",
        "revenue_share": "#4477AA",
        "conv_share": "#117733",
    }

    for i, c in enumerate(cols):
        offset = (i - (len(cols) - 1) / 2) * width
        ax.bar(
            x + offset,
            data[c].astype(float).fillna(0),
            width=width,
            label=pretty[c],
            color=colors[c],
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0, fontsize=9)
    ax.set_ylabel("%")
    ax.set_title("Сравнение долей: затраты, выручка, конверсии")
    ax.legend()

    path = _new_path("shares3", tmpdir)
    _save(fig, path)
    return Chart(
        title="Сравнение долей: затраты, выручка, конверсии",
        path=path,
        description="Сравнивает доли затрат, выручки и конверсий по каждому сегменту.",
    )


def _heatmap(result: AnalysisResult, cols: list[str], title: str, tmpdir: str) -> Chart | None:
    m = result.metrics
    present = [c for c in cols if _has_metric(m, c)]
    if len(present) < 2:
        return None

    base_cols = [result.group_col] + present
    if result.channel_col and result.channel_col in m.columns:
        base_cols.append(result.channel_col)
    if result.campaign_col and result.campaign_col in m.columns:
        base_cols.append(result.campaign_col)

    data = m[base_cols].copy().set_index(result.group_col)
    for c in present:
        data[c] = pd.to_numeric(data[c], errors="coerce")

    numeric = data[present].copy()
    if numeric.dropna(how="all").empty:
        return None

    z = numeric.copy()
    for c in present:
        col = numeric[c]
        mean = col.mean(skipna=True)
        std = col.std(skipna=True, ddof=0)
        if std and not pd.isna(std) and std > 0:
            z[c] = (col - mean) / std
        else:
            z[c] = 0.0

    abs_max = float(np.nanmax(np.abs(z.values))) if z.size else 1.0
    abs_max = max(abs_max, 0.5)

    fig, ax = plt.subplots(figsize=(1.7 * len(present) + 3, max(4, 0.5 * len(numeric))))
    sns.heatmap(
        z,
        annot=numeric.round(2),
        fmt=".2f",
        cmap="RdYlGn",
        center=0,
        vmin=-abs_max,
        vmax=abs_max,
        cbar_kws={"label": "", "shrink": 0.95},
        ax=ax,
        linewidths=0.5,
        linecolor="white",
    )

    ax.set_title(title)
    ax.set_ylabel("")
    ax.set_xticklabels([result.metric_labels.get(c, c) for c in present], rotation=30, ha="right")

    if (
        result.channel_col
        and result.campaign_col
        and _has_cols(m, [result.group_col, result.campaign_col, result.channel_col])
    ):
        info = (
            m[[result.group_col, result.campaign_col, result.channel_col]]
            .drop_duplicates(result.group_col)
            .set_index(result.group_col)
        )
        info = info.loc[numeric.index]

        campaigns = info[result.campaign_col].astype(str).tolist()
        channels = info[result.channel_col].astype(str).tolist()
        y_labels = [_truncate(ch, 14) for ch in channels]
    else:
        campaigns = []
        y_labels = [_truncate(v) for v in numeric.index.astype(str)]

    ax.set_yticklabels(y_labels, rotation=0)

    if campaigns:
        start = 0
        current = campaigns[0]

        for i, camp in enumerate(campaigns[1:], start=1):
            if camp != current:
                center = (start + i - 1) / 2 + 0.5
                ax.text(
                    -0.55,
                    center,
                    _truncate(current, 14),
                    transform=ax.get_yaxis_transform(),
                    ha="right",
                    va="center",
                    fontsize=9,
                    fontweight="bold",
                )
                start = i
                current = camp

        center = (start + len(campaigns) - 1) / 2 + 0.5
        ax.text(
            -0.55,
            center,
            _truncate(current, 14),
            transform=ax.get_yaxis_transform(),
            ha="right",
            va="center",
            fontsize=9,
            fontweight="bold",
        )

    path = _new_path("heatmap_kpi", tmpdir)
    _save(fig, path)
    return Chart(
        title=title,
        path=path,
        description=(
            "Цвет показывает отклонение сегмента от среднего значения метрики "
            "в стандартных отклонениях. Внутри ячеек указаны фактические значения."
        ),
    )


def _extra_category_chart(result: AnalysisResult, tmpdir: str) -> Chart | None:
    es = result.extra_summary
    if es is None or es.empty or not result.extra_metric:
        return None

    cat = result.extra_category_col
    metric = result.extra_metric
    if not cat or cat not in es.columns or metric not in es.columns:
        return None

    pretty = result.metric_labels.get(metric, metric)
    chan = result.channel_col if result.channel_col and result.channel_col in es.columns else None

    if chan:
        pivot = es.pivot_table(index=cat, columns=chan, values=metric, aggfunc="sum", fill_value=0, observed=True)
        if pivot.empty:
            return None

        pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=False).index]
        fig, ax = plt.subplots(figsize=(max(8, 0.85 * len(pivot.index) + 3), 5.4))

        n_cats = len(pivot.index)
        n_chans = len(pivot.columns)
        width = 0.8 / max(n_chans, 1)
        x = np.arange(n_cats)
        cmap = plt.get_cmap("tab10")

        for i, c in enumerate(pivot.columns):
            offset = (i - (n_chans - 1) / 2) * width
            ax.bar(
                x + offset,
                pivot[c].astype(float),
                width=width,
                label=_truncate(c, 20),
                color=cmap(i % 10),
            )

        ax.set_xticks(x)
        ax.set_xticklabels([_truncate(v, 18) for v in pivot.index.astype(str)], rotation=15, ha="right")
        ax.set_xlabel(str(cat))
        ax.set_ylabel(pretty)
        ax.set_title(f"{pretty}: {cat} × Каналы")
        ax.legend(title="Канал", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)

        desc = f"Сравнение значений «{pretty}» по дополнительной категории «{cat}» в разрезе каналов."
        chart_title = f"{pretty}: {cat} × Каналы"
    else:
        data = es.groupby(cat, dropna=True)[metric].sum().sort_values(ascending=False)
        if data.empty:
            return None

        fig, ax = plt.subplots(figsize=(max(7, 0.75 * len(data) + 3), 5))
        ax.bar(
            [_truncate(v, 20) for v in data.index.astype(str)],
            data.values.astype(float),
            color="#1C8F9E",
        )
        ax.set_ylabel(pretty)
        ax.set_xlabel(str(cat))
        ax.set_title(f"{pretty} по «{cat}»")
        plt.xticks(rotation=15, ha="right")

        desc = f"Суммарное значение «{pretty}» по дополнительной категории «{cat}»."
        chart_title = f"{pretty} по «{cat}»"

    path = _new_path("extra_cat", tmpdir)
    _save(fig, path)
    return Chart(title=chart_title, path=path, description=desc)


def _age_chart(result: AnalysisResult, tmpdir: str) -> Chart | None:
    age_table = result.age_table
    if age_table is None or age_table.empty:
        return None

    metric = getattr(result, "age_metric", None)
    if not metric or metric not in age_table.columns:
        for cand in ("conversions", "revenue", "clicks", "displays"):
            if cand in age_table.columns and age_table[cand].notna().any():
                metric = cand
                break

    if not metric or metric not in age_table.columns:
        return None

    age_col = "Возрастная группа"
    if age_col not in age_table.columns:
        return None

    pretty_metric = result.metric_labels.get(metric, metric)

    if (
        result.campaign_col
        and result.channel_col
        and result.campaign_col in age_table.columns
        and result.channel_col in age_table.columns
    ):
        first_channel_series = age_table[result.channel_col].dropna().astype(str)
        first_channel = first_channel_series.iloc[0] if not first_channel_series.empty else None
        if first_channel is None:
            return None

        filtered = age_table[age_table[result.channel_col].astype(str) == first_channel].copy()
        if filtered.empty:
            return None

        pivot = filtered.pivot_table(
            index=age_col,
            columns=result.campaign_col,
            values=metric,
            aggfunc="sum",
            fill_value=0,
            observed=True,
        )
        chart_title = f"{pretty_metric} по возрастным группам · канал: {first_channel}"
    else:
        column_key = result.group_col
        if column_key not in age_table.columns:
            if result.campaign_col and result.campaign_col in age_table.columns:
                column_key = result.campaign_col
            elif result.channel_col and result.channel_col in age_table.columns:
                column_key = result.channel_col
            else:
                return None

        pivot = age_table.pivot_table(
            index=age_col,
            columns=column_key,
            values=metric,
            aggfunc="sum",
            fill_value=0,
            observed=True,
        )
        chart_title = f"{pretty_metric} по возрастным группам"

    if pivot.empty:
        return None

    ordered_ages = [label for label in AGE_LABELS if label in pivot.index]
    if ordered_ages:
        pivot = pivot.reindex(ordered_ages).fillna(0)

    fig, ax = plt.subplots(figsize=(max(8, 0.8 * len(pivot.columns) + 3), 5.2))
    pivot.plot(kind="bar", ax=ax, colormap="tab20")

    ax.set_xlabel("Возрастная группа")
    ax.set_ylabel(pretty_metric)
    ax.set_title(chart_title)
    ax.legend(title="", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    plt.xticks(rotation=0)

    path = _new_path("age", tmpdir)
    _save(fig, path)
    return Chart(
        title=chart_title,
        path=path,
        description=f"Распределение по возрастным группам по метрике «{pretty_metric}».",
    )


def build_charts(result: AnalysisResult, tmpdir: str | None = None) -> list[Chart]:
    if tmpdir is None:
        tmpdir = tempfile.mkdtemp(prefix="mkt_charts_")

    charts: list[Chart] = []
    labels = result.metric_labels

    line_base = _line_by_month_channel(
        result,
        metric_cols=["displays", "clicks", "conversions"],
        title="Линейные графики по месяцам: показы, клики, конверсии",
        tmpdir=tmpdir,
    )
    if line_base:
        charts.append(line_base)

    line_kpi = _line_by_month_channel(
        result,
        metric_cols=["CPC", "CTR", "CVR", "CPA"],
        title="Линейные графики по месяцам: CPC, CTR, CVR, CPA",
        tmpdir=tmpdir,
    )
    if line_kpi:
        charts.append(line_kpi)

    for value_col, color in (
        ("displays", "#88CCEE"),
        ("clicks", "#4477AA"),
        ("conversions", "#117733"),
    ):
        if _has_metric(result.metrics, value_col):
            ch = _bar_chart(
                result,
                value_col,
                f"{labels.get(value_col, value_col)} — ранжирование",
                labels.get(value_col, value_col),
                tmpdir,
                color=color,
            )
            if ch:
                charts.append(ch)

    for value_col, color in (
        ("CPC", "#DDCC77"),
        ("CTR", "#882255"),
        ("CVR", "#AA4499"),
        ("CPA", "#999933"),
    ):
        if _has_metric(result.metrics, value_col):
            ch = _bar_chart(
                result,
                value_col,
                f"{labels.get(value_col, value_col)} — ранжирование",
                labels.get(value_col, value_col),
                tmpdir,
                color=color,
            )
            if ch:
                charts.append(ch)

    share = _share_chart(result, tmpdir)
    if share:
        charts.append(share)

    kpi_cols = [c for c in ("CTR", "CVR", "CPA", "CPC") if _has_metric(result.metrics, c)]
    if len(kpi_cols) >= 2:
        hm = _heatmap(result, kpi_cols, "Тепловая карта KPI с отклонением от среднего", tmpdir)
        if hm:
            charts.append(hm)

    extra = _extra_category_chart(result, tmpdir)
    if extra:
        charts.append(extra)

    age = _age_chart(result, tmpdir)
    if age:
        charts.append(age)

    return charts


def cleanup_charts(charts: list[Chart]) -> None:
    for ch in charts:
        try:
            if ch.path and os.path.exists(ch.path):
                os.remove(ch.path)
        except OSError:
            pass