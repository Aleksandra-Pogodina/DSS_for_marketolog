"""Графики для DOCX-отчёта: matplotlib/seaborn → PNG.

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

from app.processing import AnalysisResult

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


def _segment_y_labels(result: AnalysisResult, df: pd.DataFrame) -> list[str]:
    """Для horizontal bar: если есть и кампания, и канал — две строки в подписи."""
    if result.channel_col and result.campaign_col and {result.channel_col, result.campaign_col}.issubset(df.columns):
        return [f"{_truncate(camp, 18)}\n— {_truncate(chan, 18)}"
                for camp, chan in zip(df[result.campaign_col].astype(str),
                                      df[result.channel_col].astype(str))]
    return [_truncate(v) for v in df[result.group_col].astype(str)]


def _segment_x_labels(result: AnalysisResult, df: pd.DataFrame) -> list[str]:
    """Для vertical bar: две строки (канал / кампания)."""
    if result.channel_col and result.campaign_col and {result.channel_col, result.campaign_col}.issubset(df.columns):
        return [f"{_truncate(chan, 14)}\n{_truncate(camp, 14)}"
                for camp, chan in zip(df[result.campaign_col].astype(str),
                                      df[result.channel_col].astype(str))]
    return [_truncate(v) for v in df[result.group_col].astype(str)]


def _campaign_group_annotations(result: AnalysisResult, df: pd.DataFrame) -> list[tuple[float, str]]:
    """
    Возвращает список (x_center, campaign_label) для групповых подписей кампаний.
    """
    if not (
        result.channel_col
        and result.campaign_col
        and {result.channel_col, result.campaign_col}.issubset(df.columns)
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


def _bar_chart(result: AnalysisResult, value_col: str, title: str, ylabel: str,
               tmpdir: str, color: str = "steelblue") -> Chart | None:
    m = result.metrics
    if value_col not in m.columns:
        return None

    data = m.dropna(subset=[value_col]).copy()
    if data.empty:
        return None

    has_campaign_and_channel = (
        result.channel_col
        and result.campaign_col
        and {result.channel_col, result.campaign_col}.issubset(data.columns)
    )

    if has_campaign_and_channel:
        data = data.sort_values(
            [result.campaign_col, result.channel_col],
            ascending=[True, True]
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
            rotation=0,
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
        description=f"Сегменты упорядочены по «{ylabel}»."
    )


def _funnel_chart(result: AnalysisResult, tmpdir: str) -> Chart | None:
    """Показы / клики / конверсии — только из реально доступных."""
    cols_available = [c for c in ("displays", "clicks", "conversions") if c in result.metrics.columns]
    if len(cols_available) < 2:
        return None

    sort_metric = "conversions" if "conversions" in cols_available else cols_available[0]
    if result.channel_col and result.campaign_col:
        data = result.metrics.sort_values([result.campaign_col, sort_metric],
                                          ascending=[True, False]).copy()
    else:
        data = result.metrics.sort_values(sort_metric, ascending=False).copy()

    pretty = {"displays": "Показы", "clicks": "Клики", "conversions": "Конверсии"}
    colors = {"displays": "#88CCEE", "clicks": "#4477AA", "conversions": "#117733"}

    n = len(data)
    width = 0.8 / len(cols_available)
    fig, ax = plt.subplots(figsize=(max(8, 0.7 * n + 3), 5.5))
    x = np.arange(n)
    for i, col in enumerate(cols_available):
        offset = (i - (len(cols_available) - 1) / 2) * width
        ax.bar(x + offset, data[col].astype(float).fillna(0),
               width=width, label=pretty[col], color=colors.get(col))

    ax.set_xticks(x)
    ax.set_xticklabels(_segment_x_labels(result, data), rotation=0, fontsize=9)
    names = [pretty[c] for c in cols_available]
    if len(names) == 2:
        chart_title = f"{names[0]} и {names[1].lower()}"
    else:
        chart_title = f"{names[0]}, " + ", ".join(n.lower() for n in names[1:-1]) + f" и {names[-1].lower()}"
    ax.set_title(chart_title)
    ax.set_ylabel("Значение")
    ax.legend()

    path = _new_path("funnel", tmpdir)
    _save(fig, path)
    return Chart(title=chart_title, path=path,
                 description="Сравнение показов / кликов / конверсий по сегментам.")


def _heatmap(result: AnalysisResult, cols: list[str], title: str, tmpdir: str) -> Chart | None:
    """Тепловая карта KPI: цвет — отклонение от среднего в σ, в клетках — реальные значения."""
    m = result.metrics
    present = [c for c in cols if c in m.columns and m[c].notna().any()]
    if len(present) < 2:
        return None

    data = m[[result.group_col] + present].copy().set_index(result.group_col).astype(float)

    z = data.copy()
    for c in present:
        col = data[c]
        mean = col.mean(skipna=True)
        std = col.std(skipna=True, ddof=0)
        if std and not pd.isna(std) and std > 0:
            z[c] = (col - mean) / std
        else:
            z[c] = 0.0

    abs_max = float(np.nanmax(np.abs(z.values))) if z.size else 1.0
    abs_max = max(abs_max, 0.5)

    fig, ax = plt.subplots(figsize=(1.6 * len(present) + 3, max(4, 0.5 * len(data))))
    sns.heatmap(
        z, annot=data.round(2), fmt=".2f",
        cmap="RdYlGn", center=0, vmin=-abs_max, vmax=abs_max,
        cbar_kws={"label": "Отклонение от среднего, σ"},
        ax=ax, linewidths=0.5, linecolor="white",
    )
    ax.set_title(title)
    ax.set_ylabel("")
    ax.set_xticklabels([result.metric_labels.get(c, c) for c in present], rotation=30, ha="right")
    # Метки по y — если есть две размерности, показываем «кампания · канал»
    if result.channel_col and result.campaign_col:
        info = m[[result.group_col, result.campaign_col, result.channel_col]].drop_duplicates(result.group_col)
        info = info.set_index(result.group_col).loc[data.index]
        y_labels = [f"{_truncate(c, 14)} · {_truncate(ch, 14)}"
                    for c, ch in zip(info[result.campaign_col].astype(str),
                                     info[result.channel_col].astype(str))]
    else:
        y_labels = [_truncate(v) for v in data.index.astype(str)]
    ax.set_yticklabels(y_labels, rotation=0)

    path = _new_path("heatmap", tmpdir)
    _save(fig, path)
    return Chart(
        title=title, path=path,
        description=(
            "Цвет показывает, насколько сегмент выше (зелёный) или ниже (красный) среднего "
            "по этой метрике, в стандартных отклонениях. В клетках — фактические значения."
        ),
    )


def _correlation_heatmap(metrics: pd.DataFrame, labels: dict, tmpdir: str) -> Chart | None:
    numeric = metrics.select_dtypes(include=[np.number]).dropna(axis=1, how="all")
    if numeric.shape[1] < 3:
        return None

    corr = numeric.corr()
    fig, ax = plt.subplots(figsize=(0.7 * len(corr) + 3, 0.7 * len(corr) + 2))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                vmin=-1, vmax=1, ax=ax, linewidths=0.5, linecolor="white")
    display_labels = [labels.get(c, c) for c in corr.columns]
    ax.set_xticklabels(display_labels, rotation=30, ha="right")
    ax.set_yticklabels(display_labels, rotation=0)
    ax.set_title("Корреляции между показателями")

    path = _new_path("corr", tmpdir)
    _save(fig, path)
    return Chart(title="Корреляции между показателями", path=path,
                 description="Коэффициенты корреляции Пирсона.")


def _share_chart(result: AnalysisResult, tmpdir: str) -> Chart | None:
    m = result.metrics
    cols = [c for c in ("cost_share", "conv_share") if c in m.columns]
    if not cols:
        return None
    data = m.dropna(subset=cols, how="all").copy()
    if data.empty:
        return None

    sort_metric = "conv_share" if "conv_share" in cols else cols[0]
    if result.channel_col and result.campaign_col:
        data = data.sort_values([result.campaign_col, sort_metric], ascending=[True, False])
    else:
        data = data.sort_values(sort_metric, ascending=False)

    labels = _segment_x_labels(result, data)
    n = len(labels)
    width = 0.4
    fig, ax = plt.subplots(figsize=(max(7, 0.7 * n + 3), 5))
    x = np.arange(n)
    pretty = {"cost_share": "Доля затрат, %", "conv_share": "Доля конверсий, %"}
    colors = {"cost_share": "#cc6677", "conv_share": "#117733"}
    for i, c in enumerate(cols):
        ax.bar(x + (i - (len(cols) - 1) / 2) * width, data[c].astype(float).fillna(0),
               width=width, label=pretty[c], color=colors[c])
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0, fontsize=9)
    ax.set_ylabel("%")
    ax.set_title("Доли затрат и конверсий")
    ax.legend()

    path = _new_path("shares", tmpdir)
    _save(fig, path)
    return Chart(
        title="Доли затрат и конверсий", path=path,
        description="Сравнивает, сколько бюджета забирает сегмент и какую долю конверсий приносит.",
    )


def _age_chart(result: AnalysisResult, tmpdir: str) -> Chart | None:
    age_table = result.age_table
    if age_table is None or age_table.empty:
        return None
    value_col = None
    for cand in ("conversions", "clicks", "Записей"):
        if cand in age_table.columns:
            value_col = cand
            break
    if value_col is None:
        return None

    pivot = age_table.pivot_table(
        index="Возрастная группа", columns=result.group_col, values=value_col,
        aggfunc="sum", fill_value=0, observed=True,
    )
    if pivot.empty:
        return None

    fig, ax = plt.subplots(figsize=(max(7, 0.6 * len(pivot.columns) + 3), 5))
    pivot.plot(kind="bar", stacked=True, ax=ax, colormap="tab20")
    pretty = {"conversions": "Конверсии", "clicks": "Клики"}
    ax.set_ylabel(pretty.get(value_col, value_col))
    ax.set_xlabel("Возрастная группа")
    ax.set_title(f"{pretty.get(value_col, value_col)} по возрастным группам")
    ax.legend(title="Сегмент", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    plt.xticks(rotation=0)

    path = _new_path("age", tmpdir)
    _save(fig, path)
    return Chart(title="Распределение по возрасту", path=path,
                 description="Распределение ключевой метрики по возрастным группам.")


def _extra_category_chart(result: AnalysisResult, tmpdir: str) -> Chart | None:
    es = result.extra_summary
    if es is None or es.empty or not result.extra_metric:
        return None
    cat = result.extra_category_col
    metric = result.extra_metric
    if cat not in es.columns or metric not in es.columns:
        return None

    pretty = result.metric_labels.get(metric, metric)
    chan = result.channel_col if result.channel_col and result.channel_col in es.columns else None

    if chan:
        pivot = es.pivot_table(index=cat, columns=chan, values=metric, aggfunc="sum", fill_value=0)
        pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=False).index]
        fig, ax = plt.subplots(figsize=(max(8, 0.8 * len(pivot.index) + 3), 5))
        n_cats = len(pivot.index)
        n_chans = len(pivot.columns)
        width = 0.8 / max(n_chans, 1)
        x = np.arange(n_cats)
        cmap = plt.get_cmap("tab10")
        for i, c in enumerate(pivot.columns):
            offset = (i - (n_chans - 1) / 2) * width
            ax.bar(x + offset, pivot[c].astype(float),
                   width=width, label=_truncate(c, 20), color=cmap(i % 10))
        ax.set_xticks(x)
        ax.set_xticklabels([_truncate(v, 18) for v in pivot.index.astype(str)], rotation=15, ha="right")
        ax.set_xlabel(str(cat))
        ax.set_ylabel(pretty)
        ax.set_title(f"{pretty}: {cat} × Каналы")
        ax.legend(title="Канал", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
        desc = (
            f"Сравнение значений «{pretty}» по дополнительной категории «{cat}» в разрезе каналов."
        )
        chart_title = f"{pretty}: {cat} × Каналы"
    else:
        data = es.groupby(cat, dropna=True)[metric].sum().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(max(7, 0.7 * len(data) + 3), 5))
        ax.bar([_truncate(v, 20) for v in data.index.astype(str)],
               data.values.astype(float), color="#1c8f9e")
        ax.set_ylabel(pretty)
        ax.set_xlabel(str(cat))
        ax.set_title(f"{pretty} по «{cat}»")
        plt.xticks(rotation=15, ha="right")
        desc = f"Суммарное значение «{pretty}» по дополнительной категории «{cat}»."
        chart_title = f"{pretty} по «{cat}»"

    path = _new_path("extra_cat", tmpdir)
    _save(fig, path)
    return Chart(title=chart_title, path=path, description=desc)


def build_charts(result: AnalysisResult, tmpdir: str | None = None) -> list[Chart]:
    if tmpdir is None:
        tmpdir = tempfile.mkdtemp(prefix="mkt_charts_")

    charts: list[Chart] = []
    labels = result.metric_labels

    # сводный график по показам / кликам / конверсиям
    funnel = _funnel_chart(result, tmpdir)
    if funnel:
        charts.append(funnel)

    combined_funnel_metrics = {
        c for c in ("displays", "clicks", "conversions")
        if c in result.metrics.columns and result.metrics[c].notna().any()
    }
    has_combined_funnel = len(combined_funnel_metrics) >= 2

    # ranking bars
    for value_col, color in (
            ("conversions", "#117733"),
            ("clicks", "#4477AA"),
            ("displays", "#88CCEE"),
            ("total_cost", "#CC6677"),
            ("CTR", "#882255"),
            ("CVR", "#AA4499"),
            ("CPA", "#999933"),
            ("CPC", "#DDCC77"),
    ):
        if has_combined_funnel and value_col in {"displays", "clicks", "conversions"}:
            continue

        if value_col in result.metrics.columns:
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
        if value_col in result.metrics.columns:
            ch = _bar_chart(result, value_col,
                            f"{labels.get(value_col, value_col)} — ранжирование",
                            labels.get(value_col, value_col), tmpdir, color=color)
            if ch:
                charts.append(ch)

    # сравнение долей
    share = _share_chart(result, tmpdir)
    if share:
        charts.append(share)

    # heatmap KPI
    kpi_cols = [c for c in ("CTR", "CVR", "CPA", "CPC") if c in result.metrics.columns]
    if len(kpi_cols) >= 2:
        hm = _heatmap(result, kpi_cols, "Тепловая карта KPI", tmpdir)
        if hm:
            charts.append(hm)

    corr = _correlation_heatmap(result.metrics, labels, tmpdir)
    if corr:
        charts.append(corr)

    age = _age_chart(result, tmpdir)
    if age:
        charts.append(age)

    extra = _extra_category_chart(result, tmpdir)
    if extra:
        charts.append(extra)

    return charts


def cleanup_charts(charts: list[Chart]) -> None:
    for ch in charts:
        try:
            if ch.path and os.path.exists(ch.path):
                os.remove(ch.path)
        except OSError:
            pass
