"""Интерактивные графики Plotly для UI (QWebEngineView).

Каждый build_* возвращает объект PlotlySpec с заголовком, описанием и
HTML-строкой готовой к отображению. Набор графиков определяется по
доступным метрикам в AnalysisResult.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

from app.processing import AnalysisResult


@dataclass
class PlotlySpec:
    title: str
    html: str
    description: str = ""


_BASE_LAYOUT = dict(
    template="plotly_white",
    margin=dict(l=60, r=30, t=70, b=110),
    font=dict(family="DejaVu Sans, Arial, sans-serif", size=12),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
    hovermode="x unified",
)

_FUNNEL_LABELS = {"displays": "Показы", "clicks": "Клики", "conversions": "Конверсии"}


def _funnel_title(cols: list[str]) -> str:
    names = [_FUNNEL_LABELS[c] for c in cols]
    if len(names) == 1:
        return names[0]
    if len(names) == 2:
        return f"{names[0]} и {names[1].lower()}"
    # 3 ряда: «Показы, клики и конверсии»
    return f"{names[0]}, " + ", ".join(n.lower() for n in names[1:-1]) + f" и {names[-1].lower()}"


def _to_html(fig: go.Figure, title: str) -> str:
    """HTML-страница с прозрачным фоном — встраивается в любую тему Qt."""
    fig.update_layout(
        title=dict(text=title, x=0.02, xanchor="left"),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    inner = pio.to_html(
        fig, include_plotlyjs="cdn", full_html=False,
        config={
            "displaylogo": False, "responsive": True,
            "toImageButtonOptions": {"format": "png", "scale": 2},
            "modeBarButtonsToRemove": ["lasso2d", "select2d"],
        },
    )
    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
  html, body {{ margin: 0; padding: 0; height: 100%; background: transparent; }}
  .plotly-graph-div {{ width: 100% !important; height: 100% !important; }}
</style>
</head>
<body>
{inner}
</body>
</html>"""


def _truncate(s, n: int = 28) -> str:
    s = str(s)
    return s if len(s) <= n else s[: n - 1] + "…"


def _segment_x(result: AnalysisResult, source_df: pd.DataFrame):
    """Возвращает x для bar-чарта.

    Когда выбраны и канал, и кампания — используем двухуровневые подписи
    Plotly: первый уровень — кампания, второй — канал. Это решает требование
    «один раз подписать кампанию, каналы одной кампании рядом».

    Если выбрана одна размерность, возвращается простой список строк.
    """
    if result.channel_col and result.campaign_col and {result.channel_col, result.campaign_col}.issubset(source_df.columns):
        campaigns = source_df[result.campaign_col].astype(str).tolist()
        channels = source_df[result.channel_col].astype(str).tolist()
        return [campaigns, channels], "multi"

    labels = [_truncate(v) for v in source_df[result.group_col].astype(str)]
    return labels, "flat"


def _sorted_for_segments(result: AnalysisResult, df: pd.DataFrame, sort_metric: str | None) -> pd.DataFrame:
    """Каналы одной кампании должны идти подряд: сортируем по campaign, затем по метрике."""
    if df.empty:
        return df
    if result.channel_col and result.campaign_col and sort_metric and sort_metric in df.columns:
        return df.sort_values([result.campaign_col, sort_metric], ascending=[True, False])
    if sort_metric and sort_metric in df.columns:
        return df.sort_values(sort_metric, ascending=False, na_position="last")
    return df


def _apply_segment_xaxis(fig: go.Figure, kind: str) -> None:
    if kind == "multi":
        fig.update_xaxes(type="multicategory", tickangle=0)
    else:
        fig.update_xaxes(tickangle=-20)


# ---------- графики ---------------------------------------------------------

def _funnel_combo(result: AnalysisResult) -> PlotlySpec | None:
    """Сводный график: показы / клики / конверсии — только из доступных метрик."""
    cols_available = [c for c in ("displays", "clicks", "conversions") if c in result.metrics.columns]
    if len(cols_available) < 2:
        return None

    sort_metric = "conversions" if "conversions" in cols_available else cols_available[0]
    m = _sorted_for_segments(result, result.metrics, sort_metric)
    x, kind = _segment_x(result, m)
    colors = {"displays": "#88CCEE", "clicks": "#4477AA", "conversions": "#117733"}

    fig = go.Figure()
    for col in cols_available:
        fig.add_trace(go.Bar(
            x=x, y=m[col].astype(float),
            name=_FUNNEL_LABELS[col],
            marker_color=colors.get(col),
            hovertemplate="<b>%{x}</b><br>" + _FUNNEL_LABELS[col] + ": %{y:,.0f}<extra></extra>",
        ))

    fig.update_layout(**_BASE_LAYOUT, barmode="group", yaxis_title="Значение")
    _apply_segment_xaxis(fig, kind)

    chart_title = _funnel_title(cols_available)
    return PlotlySpec(
        title=chart_title,
        html=_to_html(fig, chart_title),
        description="Каждый ряд можно скрыть/показать кликом по легенде.",
    )


def _kpi_combo(result: AnalysisResult) -> PlotlySpec | None:
    available_pct = [c for c in ("CTR", "CVR") if c in result.metrics.columns]
    available_abs = [c for c in ("CPC", "CPA") if c in result.metrics.columns]
    if not available_pct and not available_abs:
        return None

    sort_metric = "conversions" if "conversions" in result.metrics.columns else None
    m = _sorted_for_segments(result, result.metrics, sort_metric)
    x, kind = _segment_x(result, m)
    labels = result.metric_labels

    fig = go.Figure()
    colors = {"CTR": "#882255", "CVR": "#AA4499", "CPC": "#DDCC77", "CPA": "#999933"}
    for col in available_pct:
        fig.add_trace(go.Scatter(
            x=x, y=m[col].astype(float),
            name=labels.get(col, col), mode="lines+markers",
            line=dict(color=colors.get(col), width=2),
            marker=dict(size=8),
            yaxis="y1",
            hovertemplate="<b>%{x}</b><br>" + labels.get(col, col) + ": %{y:.2f}%<extra></extra>",
        ))
    for col in available_abs:
        fig.add_trace(go.Scatter(
            x=x, y=m[col].astype(float),
            name=labels.get(col, col), mode="lines+markers",
            line=dict(color=colors.get(col), width=2, dash="dot"),
            marker=dict(size=8, symbol="diamond"),
            yaxis="y2",
            hovertemplate="<b>%{x}</b><br>" + labels.get(col, col) + ": %{y:,.2f}<extra></extra>",
        ))

    fig.update_layout(
        **_BASE_LAYOUT,
        yaxis=dict(title="%, CTR / CVR" if available_pct else None, rangemode="tozero"),
        yaxis2=dict(
            title="Стоимость, CPC / CPA" if available_abs else None,
            overlaying="y", side="right", rangemode="tozero", showgrid=False,
        ),
    )
    _apply_segment_xaxis(fig, kind)

    return PlotlySpec(
        title="KPI: CTR, CVR, CPC, CPA",
        html=_to_html(fig, "KPI — CTR, CVR (слева), CPC, CPA (справа)"),
        description="Левая ось — проценты (CTR, CVR), правая — стоимость (CPC, CPA). Скрывайте ряды через легенду.",
    )


def _shares(result: AnalysisResult) -> PlotlySpec | None:
    m = result.metrics
    cols = [c for c in ("cost_share", "conv_share") if c in m.columns]
    if not cols:
        return None
    data = m.dropna(subset=cols, how="all")
    if data.empty:
        return None

    data = _sorted_for_segments(result, data, "conv_share" if "conv_share" in cols else cols[0])
    x, kind = _segment_x(result, data)
    pretty = {"cost_share": "Доля затрат, %", "conv_share": "Доля конверсий, %"}
    colors = {"cost_share": "#CC6677", "conv_share": "#117733"}

    fig = go.Figure()
    for c in cols:
        fig.add_trace(go.Bar(
            x=x, y=data[c].astype(float),
            name=pretty[c], marker_color=colors[c],
            hovertemplate="<b>%{x}</b><br>" + pretty[c] + ": %{y:.2f}%<extra></extra>",
        ))
    fig.update_layout(**_BASE_LAYOUT, barmode="group", yaxis_title="%")
    _apply_segment_xaxis(fig, kind)

    return PlotlySpec(
        title="Доли затрат и конверсий",
        html=_to_html(fig, "Доли затрат и конверсий"),
        description=(
            "Сравнивает долю бюджета сегмента с долей конверсий. Если красный (затраты) "
            "сильно выше зелёного (конверсии) — кандидат на сокращение бюджета, и наоборот."
        ),
    )


def _ranking_bar(result: AnalysisResult, value_col: str, color: str) -> PlotlySpec | None:
    m = result.metrics
    if value_col not in m.columns:
        return None

    data = m.dropna(subset=[value_col]).copy()
    if data.empty:
        return None

    pretty_name = result.metric_labels.get(value_col, value_col)

    has_campaign_and_channel = (
        bool(result.channel_col)
        and bool(result.campaign_col)
        and {result.channel_col, result.campaign_col}.issubset(data.columns)
    )

    if has_campaign_and_channel:
        data = data.sort_values(
            [result.campaign_col, result.channel_col],
            ascending=[True, True]
        ).copy()

        x_labels = data[result.channel_col].astype(str).tolist()
        campaigns = data[result.campaign_col].astype(str).tolist()

        annotations = []
        shapes = []

        start = 0
        current = campaigns[0]
        for i, camp in enumerate(campaigns[1:], start=1):
            if camp != current:
                annotations.append(dict(
                    x=(start + i - 1) / 2,
                    y=-0.23,
                    xref="x",
                    yref="paper",
                    text=current,
                    showarrow=False,
                    xanchor="center",
                    yanchor="top",
                    font=dict(size=11)
                ))
                shapes.append(dict(
                    type="line",
                    xref="x",
                    yref="paper",
                    x0=i - 0.5,
                    x1=i - 0.5,
                    y0=0,
                    y1=1,
                    line=dict(color="rgba(120,120,120,0.45)", width=1)
                ))
                start = i
                current = camp

        annotations.append(dict(
            x=(start + len(campaigns) - 1) / 2,
            y=-0.23,
            xref="x",
            yref="paper",
            text=current,
            showarrow=False,
            xanchor="center",
            yanchor="top",
            font=dict(size=11)
        ))

        bottom_margin = 145
        customdata = np.column_stack([
            data[result.campaign_col].astype(str).to_numpy(),
            data[result.channel_col].astype(str).to_numpy()
        ])
        hovertemplate = (
            "Кампания: %{customdata[0]}<br>"
            "Канал: %{customdata[1]}<br>"
            f"{pretty_name}: %{{y:.2f}}<extra></extra>"
        )

    else:
        data = data.sort_values(value_col, ascending=False).copy()
        x_labels = data[result.group_col].astype(str).tolist()
        annotations = []
        shapes = []
        bottom_margin = 100
        customdata = np.column_stack([data[result.group_col].astype(str).to_numpy()])
        hovertemplate = (
            "Сегмент: %{customdata[0]}<br>"
            f"{pretty_name}: %{{y:.2f}}<extra></extra>"
        )

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=list(range(len(data))),
        y=data[value_col].astype(float).fillna(0),
        marker_color=color,
        customdata=customdata,
        hovertemplate=hovertemplate
    ))

    layout = dict(_BASE_LAYOUT)
    layout["margin"] = dict(l=60, r=30, t=70, b=bottom_margin)
    layout["title"] = dict(text=f"{pretty_name} — ранжирование", x=0.02, xanchor="left")
    layout["showlegend"] = False
    layout["annotations"] = annotations
    layout["shapes"] = shapes
    layout["xaxis"] = dict(
        tickmode="array",
        tickvals=list(range(len(data))),
        ticktext=x_labels,
        tickangle=0,
    )
    layout["yaxis"] = dict(title=pretty_name)

    fig.update_layout(**layout)

    return PlotlySpec(
        title=f"{pretty_name} — ранжирование",
        html=_to_html(fig, f"{pretty_name} — ранжирование"),
        description=f"Сегменты упорядочены по показателю «{pretty_name}».",
    )


def _kpi_zscore_heatmap(result: AnalysisResult) -> PlotlySpec | None:
    """Цвет — отклонение от среднего в σ, в клетках — фактические значения."""
    m = result.metrics
    cols = [c for c in ("CTR", "CVR", "CPA", "CPC") if c in m.columns and m[c].notna().any()]
    if len(cols) < 2:
        return None

    keep = [result.group_col] + cols
    data = m[keep].copy().set_index(result.group_col).astype(float)
    z = data.copy()
    for c in cols:
        col = data[c]
        mean = col.mean(skipna=True)
        std = col.std(skipna=True, ddof=0)
        z[c] = (col - mean) / std if (std and not pd.isna(std) and std > 0) else 0.0

    abs_max = float(np.nanmax(np.abs(z.values))) if z.size else 1.0
    abs_max = max(abs_max, 0.5)

    labels_map = result.metric_labels

    # Метки по y: если есть две размерности, склейка коротко
    if result.channel_col and result.campaign_col:
        info = m[[result.group_col, result.campaign_col, result.channel_col]].drop_duplicates(result.group_col)
        info = info.set_index(result.group_col).loc[data.index]
        y_labels = [f"{_truncate(c, 18)} · {_truncate(ch, 18)}"
                    for c, ch in zip(info[result.campaign_col].astype(str),
                                     info[result.channel_col].astype(str))]
    else:
        y_labels = [_truncate(v, 32) for v in data.index.astype(str)]

    fig = go.Figure(data=go.Heatmap(
        z=z.values,
        x=[labels_map.get(c, c) for c in cols],
        y=y_labels,
        text=data.round(2).values,
        texttemplate="%{text}",
        colorscale="RdYlGn",
        zmid=0, zmin=-abs_max, zmax=abs_max,
        colorbar=dict(title="Отклонение от среднего, σ"),
        hovertemplate="<b>%{y}</b><br>%{x}: %{text}<br>отклонение: %{z:.2f}σ<extra></extra>",
    ))
    fig.update_layout(
        **{**_BASE_LAYOUT, "legend": dict(visible=False)},
        xaxis_title="KPI",
        yaxis_title="",
        autosize=True,
    )
    return PlotlySpec(
        title="Тепловая карта KPI",
        html=_to_html(fig, "Тепловая карта KPI: цвет — отклонение от среднего, в клетках — реальные значения"),
        description=(
            "Цвет ячейки показывает, насколько сегмент выше (зелёный) или ниже (красный) "
            "среднего по этой метрике, в стандартных отклонениях. Текст — фактические "
            "значения метрик. Шкала симметрична относительно нуля."
        ),
    )


def _correlation(result: AnalysisResult) -> PlotlySpec | None:
    m = result.metrics
    numeric = m.select_dtypes(include=[np.number]).dropna(axis=1, how="all")
    if numeric.shape[1] < 3:
        return None
    corr = numeric.corr()
    labels = result.metric_labels
    text = [[f"{v:.2f}" for v in row] for row in corr.values]
    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=[labels.get(c, c) for c in corr.columns],
        y=[labels.get(c, c) for c in corr.index],
        text=text, texttemplate="%{text}",
        colorscale="RdBu_r",
        zmin=-1, zmax=1, zmid=0,
        colorbar=dict(title="Корреляция"),
    ))
    fig.update_layout(**{**_BASE_LAYOUT, "legend": dict(visible=False)}, autosize=True)
    return PlotlySpec(
        title="Корреляции между показателями",
        html=_to_html(fig, "Корреляции между показателями (Пирсон)"),
        description="Коэффициенты корреляции; помогает увидеть связи между метриками.",
    )


def _age_chart(result: AnalysisResult) -> PlotlySpec | None:
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
    pretty = {"conversions": "Конверсии", "clicks": "Клики", "Записей": "Количество записей"}
    fig = go.Figure()
    for seg in pivot.columns:
        fig.add_trace(go.Bar(
            x=pivot.index.astype(str),
            y=pivot[seg].astype(float),
            name=_truncate(seg, 28),
        ))
    fig.update_layout(
        **_BASE_LAYOUT, barmode="stack",
        xaxis_title="Возрастная группа",
        yaxis_title=pretty.get(value_col, value_col),
    )
    return PlotlySpec(
        title="Распределение по возрасту",
        html=_to_html(fig, f"{pretty.get(value_col, value_col)} по возрастным группам"),
        description="Сложенная диаграмма по возрастным группам; в легенде можно скрывать ряды.",
    )


def _extra_category_chart(result: AnalysisResult) -> PlotlySpec | None:
    """Графики по дополнительной категории (например, пол / категория покупки)."""
    es = result.extra_summary
    if es is None or es.empty or not result.extra_metric:
        return None
    cat_col = result.extra_category_col
    metric = result.extra_metric
    if cat_col not in es.columns or metric not in es.columns:
        return None

    pretty_metric = result.metric_labels.get(metric, metric)
    chan_col = result.channel_col if result.channel_col and result.channel_col in es.columns else None

    if chan_col:
        pivot = es.pivot_table(index=cat_col, columns=chan_col, values=metric,
                               aggfunc="sum", fill_value=0)
        # упорядочим категории по суммарному значению
        pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=False).index]
        fig = go.Figure()
        for ch in pivot.columns:
            fig.add_trace(go.Bar(
                x=pivot.index.astype(str),
                y=pivot[ch].astype(float),
                name=_truncate(ch, 28),
                hovertemplate=f"<b>%{{x}}</b><br>{ch}<br>{pretty_metric}: %{{y:,.0f}}<extra></extra>",
            ))
        fig.update_layout(
            **_BASE_LAYOUT, barmode="group",
            xaxis_title=str(cat_col), yaxis_title=pretty_metric,
        )
        title = f"{pretty_metric}: {cat_col} × Каналы"
        desc = (
            f"Сравнение значений «{pretty_metric}» по дополнительной категории "
            f"«{cat_col}» в разрезе каналов. Скрывайте каналы через легенду."
        )
    else:
        data = es.groupby(cat_col, dropna=True)[metric].sum().sort_values(ascending=False)
        fig = go.Figure(go.Bar(
            x=data.index.astype(str),
            y=data.values.astype(float),
            marker_color="#1c8f9e",
            hovertemplate="<b>%{x}</b><br>" + pretty_metric + ": %{y:,.0f}<extra></extra>",
        ))
        fig.update_layout(
            **{**_BASE_LAYOUT, "legend": dict(visible=False)},
            xaxis_title=str(cat_col), yaxis_title=pretty_metric,
        )
        title = f"{pretty_metric} по «{cat_col}»"
        desc = f"Суммарное значение «{pretty_metric}» по дополнительной категории «{cat_col}»."

    return PlotlySpec(title=title, html=_to_html(fig, title), description=desc)


def _heatmap_kpi(result: AnalysisResult) -> PlotlySpec | None:
    m = result.metrics
    cols = [c for c in ("CTR", "CVR", "CPA", "CPC") if c in m.columns and m[c].notna().any()]
    if len(cols) < 2:
        return None

    data = m[[result.group_col] + cols].copy().set_index(result.group_col)

    z = data.astype(float).copy()
    for c in cols:
        col = z[c]
        mean = col.mean(skipna=True)
        std = col.std(skipna=True, ddof=0)
        if std and not pd.isna(std) and std > 0:
            z[c] = (col - mean) / std
        else:
            z[c] = 0.0

    display_cols = [result.metric_labels.get(c, c) for c in cols]

    if result.channel_col and result.campaign_col and {result.channel_col, result.campaign_col}.issubset(m.columns):
        info = (
            m[[result.group_col, result.campaign_col, result.channel_col]]
            .drop_duplicates(result.group_col)
            .set_index(result.group_col)
            .loc[data.index]
        )
        y_labels = [
            f"{str(camp)} · {str(chan)}"
            for camp, chan in zip(
                info[result.campaign_col].astype(str),
                info[result.channel_col].astype(str)
            )
        ]
    else:
        y_labels = [str(v) for v in data.index]

    text_vals = data.round(2).values

    fig = go.Figure(
        data=go.Heatmap(
            z=z[cols].values,
            x=display_cols,
            y=y_labels,
            text=text_vals,
            texttemplate="%{text}",
            colorscale="RdYlGn",
            zmid=0,
            colorbar=dict(title="Отклонение от среднего, σ"),
            hovertemplate=(
                "Сегмент: %{y}<br>"
                "Метрика: %{x}<br>"
                "Значение: %{text}<br>"
                "Отклонение: %{z:.2f}σ<extra></extra>"
            ),
        )
    )

    layout = dict(_BASE_LAYOUT)
    layout["margin"] = dict(l=90, r=30, t=70, b=90)
    layout["xaxis"] = dict(tickangle=25)
    layout["yaxis"] = dict(automargin=True)
    layout["title"] = dict(text="Тепловая карта KPI", x=0.02, xanchor="left")

    fig.update_layout(**layout)

    return PlotlySpec(
        title="Тепловая карта KPI",
        html=_to_html(fig, "Тепловая карта KPI"),
        description=(
            "Цвет показывает, насколько сегмент выше или ниже среднего по KPI; "
            "в ячейках показаны фактические значения."
        ),
    )

# ---------- основной build --------------------------------------------------

def build_plotly_charts(result: AnalysisResult) -> list[PlotlySpec]:
    """Собирает все доступные Plotly-графики; порядок от обзорных к детальным."""
    specs: list[PlotlySpec] = []
    labels = result.metric_labels

    funnel = _funnel_combo(result)
    if funnel:
        specs.append(funnel)

    hm = _heatmap_kpi(result)
    if hm:
        specs.append(hm)

    combined_funnel_metrics = {
        c for c in ("displays", "clicks", "conversions")
        if c in result.metrics.columns and result.metrics[c].notna().any()
    }
    has_combined_funnel = len(combined_funnel_metrics) >= 2

    for col, color in (
            ("conversions", "#117733"),
            ("clicks", "#4477AA"),
            ("displays", "#88CCEE"),
            ("total_cost", "#CC6677"),
            ("CTR", "#882255"),
            ("CVR", "#AA4499"),
            ("CPA", "#999933"),
            ("CPC", "#DDCC77"),
    ):
        if has_combined_funnel and col in {"displays", "clicks", "conversions"}:
            continue

        if col in result.metrics.columns:
            spec = _ranking_bar(result, col, color)
            if spec:
                specs.append(spec)

    return specs

def _heatmap_kpi(result: AnalysisResult) -> PlotlySpec | None:
    m = result.metrics
    cols = [c for c in ("CTR", "CVR", "CPA", "CPC") if c in m.columns and m[c].notna().any()]
    if len(cols) < 2:
        return None

    data = m[[result.group_col] + cols].copy().set_index(result.group_col)

    z = data.astype(float).copy()
    for c in cols:
        col = z[c]
        mean = col.mean(skipna=True)
        std = col.std(skipna=True, ddof=0)
        if std and not pd.isna(std) and std > 0:
            z[c] = (col - mean) / std
        else:
            z[c] = 0.0

    abs_max = float(np.nanmax(np.abs(z.values))) if z.size else 1.0
    abs_max = max(abs_max, 0.5)

    display_cols = [result.metric_labels.get(c, c) for c in cols]

    if result.channel_col and result.campaign_col and {result.channel_col, result.campaign_col}.issubset(m.columns):
        info = (
            m[[result.group_col, result.campaign_col, result.channel_col]]
            .drop_duplicates(result.group_col)
            .set_index(result.group_col)
            .loc[data.index]
        )
        y_labels = [
            f"{str(camp)} · {str(chan)}"
            for camp, chan in zip(
                info[result.campaign_col].astype(str),
                info[result.channel_col].astype(str)
            )
        ]
    else:
        y_labels = [str(v) for v in data.index]

    fig = go.Figure(
        data=go.Heatmap(
            z=z[cols].values,
            x=display_cols,
            y=y_labels,
            text=data.round(2).values,
            texttemplate="%{text}",
            colorscale="RdYlGn",
            zmid=0,
            zmin=-abs_max,
            zmax=abs_max,
            colorbar=dict(title="Отклонение от среднего, σ"),
            hovertemplate=(
                "Сегмент: %{y}<br>"
                "Метрика: %{x}<br>"
                "Значение: %{text}<br>"
                "Отклонение: %{z:.2f}σ<extra></extra>"
            ),
        )
    )

    layout = dict(_BASE_LAYOUT)
    layout["margin"] = dict(l=90, r=30, t=70, b=90)
    layout["title"] = dict(text="Тепловая карта KPI", x=0.02, xanchor="left")
    layout["xaxis"] = dict(title="KPI", tickangle=25)
    layout["yaxis"] = dict(title=None, automargin=True)
    layout["showlegend"] = False

    fig.update_layout(**layout)

    return PlotlySpec(
        title="Тепловая карта KPI",
        html=_to_html(fig, "Тепловая карта KPI"),
        description=(
            "Цвет показывает, насколько сегмент выше или ниже среднего по KPI; "
            "в ячейках показаны фактические значения."
        ),
    )