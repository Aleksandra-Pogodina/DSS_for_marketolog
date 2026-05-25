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

from app.processing import AnalysisResult, AGE_LABELS


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

def _series_colors() -> tuple[tuple[str, str], ...]:
    return (
        ("displays", "#0F766E"),
        ("clicks", "#2563EB"),
        ("conversions", "#059669"),
        ("CTR", "#B45309"),
        ("CVR", "#7C3AED"),
        ("CPC", "#DC2626"),
        ("CPA", "#BE185D"),
    )


def _funnel_title(cols: list[str]) -> str:
    names = [_FUNNEL_LABELS[c] for c in cols]
    if len(names) == 1:
        return names[0]
    if len(names) == 2:
        return f"{names[0]} и {names[1].lower()}"
    # 3 ряда: «Показы, клики и конверсии»
    return f"{names[0]}, " + ", ".join(n.lower() for n in names[1:-1]) + f" и {names[-1].lower()}"

def month_funnel_combo(result: AnalysisResult) -> PlotlySpec | None:
    month_df = getattr(result, "month_table", None)
    if month_df is None or month_df.empty:
        return None

    cols_available = [c for c in ("displays", "clicks", "conversions") if c in month_df.columns]
    if not cols_available:
        return None

    pretty = {"displays": "Показы", "clicks": "Клики", "conversions": "Конверсии"}
    metric_colors = {"displays": "#88CCEE", "clicks": "#4477AA", "conversions": "#117733"}

    fig = go.Figure()

    for series_name, part in month_df.groupby(result.group_col, dropna=True):
        part = part.copy()
        x = part["Месяц"].astype(str).tolist()
        label_row = part.iloc[0]
        legend_name = _month_series_name(result, label_row)

        for col in cols_available:
            fig.add_trace(go.Scatter(
                x=x,
                y=part[col].astype(float),
                name=f"{pretty[col]} · {legend_name}",
                mode="lines+markers",
                line=dict(color=metric_colors[col], width=3),
                marker=dict(size=7),
                legendgroup=f"{col}",
                hovertemplate=(
                    f"<b>%{{x}}</b><br>"
                    f"Сегмент: {legend_name}<br>"
                    f"{pretty[col]}: %{{y:,.0f}}"
                    f"<extra></extra>"
                ),
            ))

    title = _funnel_title(cols_available) + " по месяцам"
    fig.update_layout(
        **_BASE_LAYOUT,
        xaxis=dict(title="Месяц"),
        yaxis=dict(title="Значение", rangemode="tozero"),
    )

    return PlotlySpec(
        title=title,
        html=_to_html(fig, title),
        description="Помесячная динамика по выбранным каналам, кампаниям или их сочетаниям.",
    )


def _to_html(fig: go.Figure, title: str) -> str:
    """HTML-страница с прозрачным фоном — встраивается в любую тему Qt."""
    fig.update_layout(
        title=None,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )

    inner = pio.to_html(
        fig,
        include_plotlyjs="cdn",
        full_html=False,
        config={
            "displaylogo": False,
            "responsive": True,
            "toImageButtonOptions": {"format": "png", "scale": 2},
            "modeBarButtonsToRemove": ["lasso2d", "select2d"],
        },
    )

    return f"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <style>
    html, body {{
      margin: 0;
      padding: 0;
      width: 100%;
      height: 100%;
      background: transparent;
      overflow: hidden;
    }}
    #wrap {{
      width: 100%;
      height: 100vh;
    }}
    .js-plotly-plot, .plot-container, .main-svg {{
      width: 100% !important;
      height: 100% !important;
    }}
  </style>
</head>
<body>
  <div id="wrap">{inner}</div>
</body>
</html>
"""

def _monthly_metric_specs_by_channel(result: AnalysisResult) -> list[PlotlySpec]:
    mapping = result.mapping
    month_col = mapping.get("month")
    channel_col = result.channel_col
    campaign_col = result.campaign_col

    if not month_col or not channel_col or not campaign_col:
        return []

    if month_col not in result.cleaned.columns:
        return []

    metric_defs = [
        ("displays", "Показы по месяцам"),
        ("clicks", "Клики по месяцам"),
        ("conversions", "Конверсии по месяцам"),
        ("CTR", "CTR по месяцам"),
        ("CVR", "CVR по месяцам"),
        ("CPC", "CPC по месяцам"),
        ("CPA", "CPA по месяцам"),
    ]

    work = result.cleaned.copy()
    work = work.dropna(subset=[month_col, channel_col, campaign_col])

    if work.empty:
        return []

    month_order = list(dict.fromkeys(work[month_col].astype(str).tolist()))
    channel_order = list(dict.fromkeys(work[channel_col].astype(str).tolist()))

    agg_map = {}
    for role in ("displays", "clicks", "conversions", "total_cost", "placement_cost"):
        col = mapping.get(role)
        if col and col in work.columns:
            agg_map[role] = (col, "sum")

    cpc_col = mapping.get("cpc")
    if cpc_col and cpc_col in work.columns:
        agg_map["cpc_avg"] = (cpc_col, "mean")

    if not agg_map:
        return []

    monthly = (
        work.groupby([channel_col, campaign_col, month_col], dropna=True)
        .agg(**agg_map)
        .reset_index()
    )

    if "clicks" in monthly.columns and "displays" in monthly.columns:
        monthly["CTR"] = (monthly["clicks"] / monthly["displays"].replace(0, np.nan)) * 100

    cost_for_cpc = None
    if "total_cost" in monthly.columns:
        cost_for_cpc = monthly["total_cost"]
    elif "placement_cost" in monthly.columns:
        cost_for_cpc = monthly["placement_cost"]

    if cost_for_cpc is not None and "clicks" in monthly.columns:
        monthly["CPC"] = cost_for_cpc / monthly["clicks"].replace(0, np.nan)
    elif "cpc_avg" in monthly.columns:
        monthly["CPC"] = monthly["cpc_avg"]

    if "conversions" in monthly.columns and "clicks" in monthly.columns:
        monthly["CVR"] = (monthly["conversions"] / monthly["clicks"].replace(0, np.nan)) * 100

    cost_for_cpa = None
    if "total_cost" in monthly.columns:
        cost_for_cpa = monthly["total_cost"]
    elif "placement_cost" in monthly.columns:
        cost_for_cpa = monthly["placement_cost"]

    if cost_for_cpa is not None and "conversions" in monthly.columns:
        monthly["CPA"] = cost_for_cpa / monthly["conversions"].replace(0, np.nan)

    specs: list[PlotlySpec] = []
    palette = _series_colors()

    for metric, title in metric_defs:
        if metric not in monthly.columns or monthly[metric].notna().sum() == 0:
            continue

        channel_figures: dict[str, go.Figure] = {}

        for channel_idx, channel in enumerate(channel_order):
            channel_df = monthly[monthly[channel_col].astype(str) == str(channel)].copy()
            if channel_df.empty:
                continue

            campaigns = list(dict.fromkeys(channel_df[campaign_col].astype(str).tolist()))
            fig = go.Figure()

            for i, campaign in enumerate(campaigns):
                part = channel_df[channel_df[campaign_col].astype(str) == str(campaign)].copy()
                if part.empty:
                    continue

                month_map = {m: idx for idx, m in enumerate(month_order)}
                part["__month_order__"] = part[month_col].astype(str).map(month_map)
                part = part.sort_values("__month_order__")

                fig.add_trace(
                    go.Scatter(
                        x=part[month_col].astype(str),
                        y=part[metric],
                        mode="lines+markers",
                        name=str(campaign),
                        line=dict(width=3, color=palette[i % len(palette)]),
                        marker=dict(size=7, color=palette[i % len(palette)]),
                        connectgaps=False,
                    )
                )

            fig.update_layout(
                **_BASE_LAYOUT,
                margin=dict(l=55, r=25, t=30, b=70),
                xaxis=dict(
                    title="Месяц",
                    type="category",
                    categoryorder="array",
                    categoryarray=month_order,
                    tickangle=0,
                ),
                yaxis=dict(title=result.metric_labels.get(metric, metric)),
            )

            channel_figures[str(channel)] = fig

        if channel_figures:
            specs.append(
                PlotlySpec(
                    title=title,
                    html=_to_html_with_channel_switch(channel_figures, title),
                    description=f"Сравнение кампаний внутри выбранного канала по показателю «{result.metric_labels.get(metric, metric)}».",
                )
            )

    return specs

def _to_html_with_channel_switch(figures: dict[str, go.Figure], title: str) -> str:
    """HTML с select-переключателем канала: один график на метрику, внутри — кампании выбранного канала."""
    prepared = {}
    for channel, fig in figures.items():
        fig.update_layout(
            title=None,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        prepared[channel] = fig.to_json()

    channels = list(figures.keys())
    if not channels:
        empty_fig = go.Figure()
        return _to_html(empty_fig, title)

    import json
    figures_json = json.dumps(prepared, ensure_ascii=False)
    channels_json = json.dumps(channels, ensure_ascii=False)

    return f"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    html, body {{
      margin: 0;
      padding: 0;
      width: 100%;
      height: 100%;
      background: transparent;
      overflow: hidden;
      font-family: DejaVu Sans, Arial, sans-serif;
    }}
    #root {{
      display: flex;
      flex-direction: column;
      width: 100%;
      height: 100vh;
      box-sizing: border-box;
    }}
    #toolbar {{
      display: flex;
      align-items: center;
      gap: 10px;
      padding: 10px 12px 0 12px;
      flex: 0 0 auto;
    }}
    #toolbar label {{
      font-size: 13px;
      font-weight: 600;
    }}
    #channelSelect {{
      min-width: 220px;
      max-width: 360px;
      padding: 6px 10px;
      font-size: 13px;
      border-radius: 6px;
      border: 1px solid #cfd6e2;
      background: #ffffff;
    }}
    #chart {{
      flex: 1 1 auto;
      min-height: 0;
      width: 100%;
    }}
  </style>
</head>
<body>
  <div id="root">
    <div id="toolbar">
      <label for="channelSelect">Канал:</label>
      <select id="channelSelect"></select>
    </div>
    <div id="chart"></div>
  </div>

  <script>
    const figures = {figures_json};
    const channels = {channels_json};
    const select = document.getElementById("channelSelect");
    const chart = document.getElementById("chart");

    channels.forEach(ch => {{
      const opt = document.createElement("option");
      opt.value = ch;
      opt.textContent = ch;
      select.appendChild(opt);
    }});

    function renderChannel(channel) {{
      const fig = JSON.parse(figures[channel]);
      Plotly.react("chart", fig.data, fig.layout, {{
        displaylogo: false,
        responsive: true,
        toImageButtonOptions: {{ format: "png", scale: 2 }},
        modeBarButtonsToRemove: ["lasso2d", "select2d"]
      }});
    }}

    select.addEventListener("change", () => renderChannel(select.value));

    if (channels.length > 0) {{
      select.value = channels[0];
      renderChannel(channels[0]);
    }}
  </script>
</body>
</html>
"""

def _month_labels(month_df: pd.DataFrame) -> list[str]:
    return month_df["Месяц"].astype(str).tolist()

def _month_series_name(result: AnalysisResult, row: pd.Series) -> str:
    if result.channel_col and result.campaign_col:
        camp = str(row.get(result.campaign_col, "")).strip()
        chan = str(row.get(result.channel_col, "")).strip()
        return f"{camp} — {chan}"
    if result.channel_col:
        return str(row.get(result.channel_col, "")).strip()
    if result.campaign_col:
        return str(row.get(result.campaign_col, "")).strip()
    return str(row.get(result.group_col, "")).strip()

def _month_top_groups(
    result: AnalysisResult,
    month_df: pd.DataFrame,
    metric_cols: list[str],
    top_n: int = 5,
) -> list[str]:
    if month_df is None or month_df.empty or result.group_col not in month_df.columns:
        return []

    month_df = month_df.copy()
    month_df[result.group_col] = month_df[result.group_col].astype(str)

    groups_all = month_df[result.group_col].dropna().drop_duplicates().tolist()

    if result.channel_col and not result.campaign_col:
        return groups_all

    if result.campaign_col and not result.channel_col:
        return groups_all

    if result.channel_col and result.campaign_col:
        present = [c for c in metric_cols if c in month_df.columns]
        if not present:
            return groups_all[:top_n]

        score_series = pd.Series(0.0, index=month_df[result.group_col].drop_duplicates())
        score_series.index = score_series.index.astype(str)

        for col in present:
            s = (
                month_df.groupby(result.group_col, dropna=True)[col]
                .sum(min_count=1)
                .fillna(0)
                .astype(float)
            )
            s.index = s.index.astype(str)
            score_series = score_series.add(s, fill_value=0)

        return score_series.sort_values(ascending=False).head(top_n).index.tolist()

    return groups_all

def _campaign_channel_style_map(result: AnalysisResult, data: pd.DataFrame) -> dict[str, dict]:
    campaign_palettes = [
        ["#1f77b4", "#4f9ed8", "#87c3eb", "#b7dcf6"],  # blue
        ["#2ca02c", "#5dbb63", "#8fd18f", "#bee6be"],  # green
        ["#ff7f0e", "#ff9f4a", "#ffc078", "#ffe0b2"],  # orange
        ["#d62728", "#e15759", "#f28e8e", "#f8bcbc"],  # red
        ["#9467bd", "#b08ad1", "#c9afe3", "#e0d2f1"],  # purple
        ["#17becf", "#56d2df", "#8fe3ea", "#c5f1f4"],  # cyan
    ]
    dash_cycle = ["solid", "dash", "dot", "dashdot"]

    style_map: dict[str, dict] = {}

    if result.channel_col and result.campaign_col:
        campaigns = [
            str(x) for x in data[result.campaign_col].dropna().astype(str).drop_duplicates().tolist()
        ]

        for i, campaign in enumerate(campaigns):
            campaign_df = data[data[result.campaign_col].astype(str) == campaign]
            channels = [
                str(x) for x in campaign_df[result.channel_col].dropna().astype(str).drop_duplicates().tolist()
            ]

            palette = campaign_palettes[i % len(campaign_palettes)]
            for j, channel in enumerate(channels):
                key = f"{channel} · {campaign}"
                style_map[key] = {
                    "color": palette[j % len(palette)],
                    "dash": dash_cycle[j % len(dash_cycle)],
                }
    else:
        default_colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728", "#9467bd", "#17becf"]
        groups = [str(x) for x in data[result.group_col].dropna().astype(str).drop_duplicates().tolist()]
        for i, g in enumerate(groups):
            style_map[g] = {
                "color": default_colors[i % len(default_colors)],
                "dash": "solid",
            }

    return style_map

def _month_single_metric_chart(
    result: AnalysisResult,
    metric: str,
    color: str,
) -> PlotlySpec | None:
    month_df = getattr(result, "month_table", None)
    if month_df is None or month_df.empty:
        return None
    if result.group_col not in month_df.columns or metric not in month_df.columns:
        return None

    data = month_df.copy()
    data[result.group_col] = data[result.group_col].astype(str)

    label = result.metric_labels.get(metric, metric)
    title = f"{label} по месяцам"

    # ---------- режим: и канал, и кампания ----------
    if result.channel_col and result.campaign_col:
        if result.channel_col not in data.columns or result.campaign_col not in data.columns:
            return None
        if "Месяц" not in data.columns:
            return None

        data[result.channel_col] = data[result.channel_col].astype(str)
        data[result.campaign_col] = data[result.campaign_col].astype(str)

        month_order = list(dict.fromkeys(data["Месяц"].astype(str).tolist()))
        channel_order = list(dict.fromkeys(data[result.channel_col].tolist()))
        style_map = _campaign_channel_style_map(result, data)

        figures_by_channel: dict[str, go.Figure] = {}

        for channel in channel_order:
            part_channel = data[data[result.channel_col] == channel].copy()

            if part_channel.empty:
                continue

            (data[[result.channel_col, result.campaign_col, "Месяц", metric]].to_string())
            fig = go.Figure()
            campaign_order = list(dict.fromkeys(part_channel[result.campaign_col].tolist()))

            for campaign in campaign_order:
                part = part_channel[part_channel[result.campaign_col] == campaign].copy()
                if part.empty:
                    continue

                part["__month_order__"] = (
                    part["Месяц"].astype(str).map({m: i for i, m in enumerate(month_order)})
                )
                part = part.sort_values("__month_order__")

                row0 = part.iloc[0]
                group_value = str(row0[result.group_col])
                series_name = str(campaign)

                if metric in ("CTR", "CVR"):
                    hover_tpl = (
                        f"<b>%{{x}}</b><br>{label}: %{{y:.2f}}%<br>"
                        f"Канал: {channel}<br>Кампания: {series_name}<extra></extra>"
                    )
                elif metric in ("CPC", "CPA"):
                    hover_tpl = (
                        f"<b>%{{x}}</b><br>{label}: %{{y:,.2f}}<br>"
                        f"Канал: {channel}<br>Кампания: {series_name}<extra></extra>"
                    )
                else:
                    hover_tpl = (
                        f"<b>%{{x}}</b><br>{label}: %{{y:,.0f}}<br>"
                        f"Канал: {channel}<br>Кампания: {series_name}<extra></extra>"
                    )

                style = style_map.get(group_value, {"color": color, "dash": "solid"})

                y_values = pd.to_numeric(part[metric], errors="coerce")
                non_na_count = int(y_values.notna().sum())
                is_single_point = non_na_count <= 1

                fig.add_trace(
                    go.Scatter(
                        x=part["Месяц"].astype(str).tolist(),
                        y=y_values.tolist(),
                        name=series_name,
                        mode="lines+markers",
                        line=dict(
                            color=style["color"],
                            width=2.5,
                            dash=style["dash"],
                        ),
                        marker=dict(
                            size=12 if is_single_point else 7,
                            color=style["color"],
                            symbol="diamond" if is_single_point else "circle",
                            line=dict(
                                color="white",
                                width=1.5 if is_single_point else 0.5,
                            ),
                        ),
                        hovertemplate=hover_tpl,
                        connectgaps=False,
                    )
                )

            layout = dict(_BASE_LAYOUT)
            layout["margin"] = dict(l=55, r=25, t=30, b=70)
            layout["xaxis"] = dict(
                title="Месяц",
                type="category",
                categoryorder="array",
                categoryarray=month_order,
            )
            layout["yaxis"] = dict(title=label, rangemode="tozero")

            fig.update_layout(**layout)

            figures_by_channel[channel] = fig

        if not figures_by_channel:
            return None

        return PlotlySpec(
            title=title,
            html=_to_html_with_channel_switch(figures_by_channel, title),
            description=f"Помесячная динамика метрики «{label}» с переключением по каналам.",
        )

    # ---------- обычный режим ----------
    style_map = _campaign_channel_style_map(result, data)
    fig = go.Figure()

    for group_value, part in data.groupby(result.group_col, dropna=True):
        part = part.copy()
        if part.empty:
            continue

        row0 = part.iloc[0]
        series_name = _month_series_name(result, row0)

        if metric in ("CTR", "CVR"):
            hover_tpl = (
                f"<b>%{{x}}</b><br>{label}: %{{y:.2f}}%<br>"
                f"Сегмент: {series_name}<extra></extra>"
            )
        elif metric in ("CPC", "CPA"):
            hover_tpl = (
                f"<b>%{{x}}</b><br>{label}: %{{y:,.2f}}<br>"
                f"Сегмент: {series_name}<extra></extra>"
            )
        else:
            hover_tpl = (
                f"<b>%{{x}}</b><br>{label}: %{{y:,.0f}}<br>"
                f"Сегмент: {series_name}<extra></extra>"
            )

        style = style_map.get(str(group_value), {"color": color, "dash": "solid"})

        fig.add_trace(
            go.Scatter(
                x=part["Месяц"].astype(str).tolist(),
                y=pd.to_numeric(part[metric], errors="coerce").tolist(),
                name=series_name,
                mode="lines+markers",
                line=dict(color=style["color"], width=2.5, dash=style["dash"]),
                marker=dict(size=7, color=style["color"]),
                hovertemplate=hover_tpl,
                connectgaps=False,
            )
        )

    fig.update_layout(
        **_BASE_LAYOUT,
        xaxis=dict(title="Месяц"),
        yaxis=dict(title=label, rangemode="tozero"),
    )

    return PlotlySpec(
        title=title,
        html=_to_html(fig, title),
        description=f"Помесячная динамика метрики «{label}».",
    )

def _build_monthly_split_specs(result: AnalysisResult) -> list[PlotlySpec]:
    specs: list[PlotlySpec] = []

    metric_colors = {
        "displays": "#88CCEE",
        "clicks": "#4477AA",
        "conversions": "#117733",
        "CTR": "#882255",
        "CVR": "#AA4499",
        "CPC": "#DDCC77",
        "CPA": "#999933",
    }

    month_df = getattr(result, "month_table", None)
    if month_df is None or month_df.empty:
        return specs

    ordered_metrics = ["displays", "clicks", "conversions", "CTR", "CVR", "CPC", "CPA"]
    for metric in ordered_metrics:
        if metric in month_df.columns:
            spec = _month_single_metric_chart(result, metric, metric_colors[metric])
            if spec is not None:
                specs.append(spec)

    return specs

def month_volume_combo_result(result: AnalysisResult) -> PlotlySpec | None:
    month_df = getattr(result, "month_table", None)
    if month_df is None or month_df.empty:
        return None

    if result.channel_col and result.campaign_col:
        metric_cols = [c for c in ("conversions", "clicks", "displays") if c in month_df.columns][:1]
    else:
        metric_cols = [c for c in ("displays", "clicks", "conversions") if c in month_df.columns]

    if not metric_cols or result.group_col not in month_df.columns:
        return None

    top_groups = _month_top_groups(result, month_df, metric_cols, top_n=5)
    if not top_groups:
        return None

    data = month_df.copy()
    data[result.group_col] = data[result.group_col].astype(str)
    data = data[data[result.group_col].isin(top_groups)].copy()

    print("TOP GROUPS VOLUME:", top_groups)
    print("VISIBLE GROUPS VOLUME:", data[result.group_col].drop_duplicates().tolist())

    if data.empty:
        return None

    pretty = {
        "displays": "Показы",
        "clicks": "Клики",
        "conversions": "Конверсии",
    }
    colors = {
        "displays": "#88CCEE",
        "clicks": "#4477AA",
        "conversions": "#117733",
    }

    fig = go.Figure()

    for group_value in top_groups:
        part = data[data[result.group_col] == str(group_value)].copy()
        if part.empty:
            continue

        row0 = part.iloc[0]
        series_name = _month_series_name(result, row0)

        for col in metric_cols:
            fig.add_trace(go.Scatter(
                x=part["Месяц"].astype(str).tolist(),
                y=pd.to_numeric(part[col], errors="coerce").fillna(0).tolist(),
                name=f"{pretty[col]} · {series_name}",
                mode="lines+markers",
                line=dict(color=colors[col], width=2.5),
                marker=dict(size=7),
                hovertemplate=(
                    f"<b>%{{x}}</b><br>"
                    f"{pretty[col]}: %{{y:,.0f}}<br>"
                    f"Сегмент: {series_name}"
                    f"<extra></extra>"
                ),
            ))

    title = "Показы, клики и конверсии по месяцам"
    fig.update_layout(
        **_BASE_LAYOUT,
        xaxis=dict(title="Месяц"),
        yaxis=dict(title="Значение", rangemode="tozero"),
    )

    return PlotlySpec(
        title=title,
        html=_to_html(fig, title),
        description=(
            "Помесячная динамика объёмных метрик. "
            "Если выбраны и кампании, и каналы, показаны только top-5 сочетаний."
        ),
    )

def month_kpi_combo_result(result: AnalysisResult) -> PlotlySpec | None:
    month_df = getattr(result, "month_table", None)
    if month_df is None or month_df.empty:
        return None

    pct_cols = [c for c in ("CTR", "CVR") if c in month_df.columns]
    abs_cols = [c for c in ("CPC", "CPA") if c in month_df.columns]
    metric_cols = pct_cols + abs_cols

    if not metric_cols or result.group_col not in month_df.columns:
        return None

    top_groups = _month_top_groups(result, month_df, metric_cols, top_n=5)
    if not top_groups:
        return None

    data = month_df.copy()
    data[result.group_col] = data[result.group_col].astype(str)
    data = data[data[result.group_col].isin(top_groups)].copy()

    print("TOP GROUPS KPI:", top_groups)
    print("VISIBLE GROUPS KPI:", data[result.group_col].drop_duplicates().tolist())

    if data.empty:
        return None

    labels = result.metric_labels
    colors = {
        "CTR": "#882255",
        "CVR": "#AA4499",
        "CPC": "#DDCC77",
        "CPA": "#999933",
    }

    fig = go.Figure()

    for group_value in top_groups:
        part = data[data[result.group_col] == str(group_value)].copy()
        if part.empty:
            continue

        row0 = part.iloc[0]
        series_name = _month_series_name(result, row0)

        for col in pct_cols:
            fig.add_trace(go.Scatter(
                x=part["Месяц"].astype(str).tolist(),
                y=pd.to_numeric(part[col], errors="coerce").fillna(0).tolist(),
                name=f"{labels.get(col, col)} · {series_name}",
                mode="lines+markers",
                line=dict(color=colors[col], width=2.5),
                marker=dict(size=7),
                yaxis="y1",
                hovertemplate=(
                    f"<b>%{{x}}</b><br>"
                    f"{labels.get(col, col)}: %{{y:.2f}}<br>"
                    f"Сегмент: {series_name}"
                    f"<extra></extra>"
                ),
            ))

        for col in abs_cols:
            fig.add_trace(go.Scatter(
                x=part["Месяц"].astype(str).tolist(),
                y=pd.to_numeric(part[col], errors="coerce").fillna(0).tolist(),
                name=f"{labels.get(col, col)} · {series_name}",
                mode="lines+markers",
                line=dict(color=colors[col], width=2.5, dash="dot"),
                marker=dict(size=7, symbol="diamond"),
                yaxis="y2",
                hovertemplate=(
                    f"<b>%{{x}}</b><br>"
                    f"{labels.get(col, col)}: %{{y:,.2f}}<br>"
                    f"Сегмент: {series_name}"
                    f"<extra></extra>"
                ),
            ))

    title = "CTR, CVR, CPC и CPA по месяцам"
    fig.update_layout(
        **_BASE_LAYOUT,
        xaxis=dict(title="Месяц"),
        yaxis=dict(
            title="CTR / CVR, %",
            rangemode="tozero",
        ),
        yaxis2=dict(
            title="CPC / CPA",
            overlaying="y",
            side="right",
            rangemode="tozero",
            showgrid=False,
        ),
    )

    return PlotlySpec(
        title=title,
        html=_to_html(fig, title),
        description=(
            "Помесячная динамика KPI. "
            "Если выбраны и кампании, и каналы, показаны только top-5 сочетаний."
        ),
    )

def _truncate(s, n: int = 28) -> str:
    s = str(s)
    return s if len(s) <= n else s[: n - 1] + "…"

def campaignseparatorshapes(result: AnalysisResult, df: pd.DataFrame) -> list[dict]:
    """
    Вертикальные разделители между группами кампаний.
    Линии рисуются только в области построения столбцов и не заходят на подписи.
    """
    if (
        df.empty
        or not result.campaign_col
        or result.campaign_col not in df.columns
    ):
        return []

    campaigns = df[result.campaign_col].astype(str).tolist()
    shapes: list[dict] = []

    for i in range(1, len(campaigns)):
        if campaigns[i] != campaigns[i - 1]:
            shapes.append(
                dict(
                    type="line",
                    xref="x",
                    yref="paper",
                    x0=i - 0.5,
                    x1=i - 0.5,
                    y0=0,
                    y1=1,
                    line=dict(color="rgba(120,120,120,0.35)", width=1),
                    layer="below",
                )
            )

    return shapes

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
        fig.update_xaxes(
            type="multicategory",
            tickangle=0,
            showdividers=False,
            dividercolor="rgba(0,0,0,0)",
            dividerwidth=0,
            automargin=True,
        )
    else:
        fig.update_xaxes(
            tickangle=-20,
            automargin=True,
        )


# ---------- графики ---------------------------------------------------------

def _funnel_combo(result: AnalysisResult) -> PlotlySpec | None:
    colsavailable = [c for c in ("displays", "clicks", "conversions") if c in result.metrics.columns]
    if len(colsavailable) < 2:
        return None

    sortmetric = "conversions" if "conversions" in colsavailable else colsavailable[0]
    m = _sorted_for_segments(result, result.metrics, sortmetric)
    x, kind = _segment_x(result, m)

    colors = {
        "displays": "#88CCEE",
        "clicks": "#4477AA",
        "conversions": "#117733",
    }

    fig = go.Figure()

    for col in colsavailable:
        fig.add_trace(
            go.Bar(
                x=x,
                y=m[col].astype(float),
                name=_FUNNEL_LABELS[col],
                marker_color=colors.get(col),
                hovertemplate=f"%{{x}}<br>{_FUNNEL_LABELS[col]}: %{{y:,.0f}}<extra></extra>",
            )
        )

    layout = dict(_BASE_LAYOUT)
    layout["barmode"] = "group"
    layout["yaxis"] = dict(title="")
    layout["margin"] = dict(l=60, r=30, t=70, b=140)

    if kind == "multi":
        layout["shapes"] = campaignseparatorshapes(result, m)

    fig.update_layout(layout)
    _apply_segment_xaxis(fig, kind)
    fig.update_xaxes(
        tickangle=-35,
        automargin=True,
    )

    charttitle = _funnel_title(colsavailable)
    return PlotlySpec(
        title=charttitle,
        html=_to_html(fig, charttitle),
        description="Показывает показы, клики и конверсии по сегментам.",
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

def month_kpi_combo(result: AnalysisResult) -> PlotlySpec | None:
    month_df = getattr(result, "month_table", None)
    if month_df is None or month_df.empty:
        return None

    available_pct = [c for c in ("CTR", "CVR") if c in month_df.columns]
    available_abs = [c for c in ("CPC", "CPA") if c in month_df.columns]

    if not available_pct and not available_abs:
        return None

    labels = result.metric_labels
    colors = {"CTR": "#882255", "CVR": "#AA4499", "CPC": "#DDCC77", "CPA": "#999933"}

    x = _month_labels(month_df)
    fig = go.Figure()

    for col in available_pct:
        fig.add_trace(go.Scatter(
            x=x,
            y=month_df[col].astype(float),
            name=labels.get(col, col),
            mode="lines+markers",
            line=dict(color=colors[col], width=3),
            marker=dict(size=8),
            yaxis="y1",
            hovertemplate=f"<b>%{{x}}</b><br>{labels.get(col, col)}: %{{y:.2f}}<extra></extra>",
        ))

    for col in available_abs:
        fig.add_trace(go.Scatter(
            x=x,
            y=month_df[col].astype(float),
            name=labels.get(col, col),
            mode="lines+markers",
            line=dict(color=colors[col], width=3, dash="dot"),
            marker=dict(size=8, symbol="diamond"),
            yaxis="y2",
            hovertemplate=f"<b>%{{x}}</b><br>{labels.get(col, col)}: %{{y:,.2f}}<extra></extra>",
        ))

    title = "CTR, CVR, CPC и CPA по месяцам"
    fig.update_layout(
        **_BASE_LAYOUT,
        xaxis=dict(title="Месяц"),
        yaxis=dict(
            title="CTR / CVR, %",
            rangemode="tozero",
        ),
        yaxis2=dict(
            title="CPC / CPA",
            overlaying="y",
            side="right",
            rangemode="tozero",
            showgrid=False,
        ),
    )

    return PlotlySpec(
        title=title,
        html=_to_html(fig, title),
        description="Динамика KPI по месяцам: CTR и CVR на левой оси, CPC и CPA на правой.",
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

        n = len(campaigns)
        start = 0
        current = campaigns[0]

        for i, camp in enumerate(campaigns[1:], start=1):
            if camp != current:
                annotations.append(dict(
                    x=(start + i - 1) / 2,
                    y=-0.30,
                    xref="x",
                    yref="paper",
                    text=current,
                    showarrow=False,
                    xanchor="center",
                    yanchor="top",
                    font=dict(size=11)
                ))

                # Верхний разделитель в paper-координатах
                x_paper = i / n
                shapes.append(
                    dict(
                        type="line",
                        xref="x",
                        yref="paper",
                        x0=i - 0.5,
                        x1=i - 0.5,
                        y0=0,
                        y1=1,
                        line=dict(color="rgba(120,120,120,0.35)", width=1),
                        layer="below",
                    )
                )

                start = i
                current = camp

        annotations.append(dict(
            x=(start + len(campaigns) - 1) / 2,
            y=-0.30,
            xref="x",
            yref="paper",
            text=current,
            showarrow=False,
            xanchor="center",
            yanchor="top",
            font=dict(size=11)
        ))

        top_margin = 95
        bottom_margin = 165

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
        top_margin = 70
        bottom_margin = 130

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
    layout["margin"] = dict(l=60, r=30, t=top_margin, b=bottom_margin)
    layout["title"] = dict(text=f"{pretty_name} — ранжирование", x=0.02, xanchor="left")
    layout["showlegend"] = False
    layout["annotations"] = annotations
    layout["shapes"] = shapes
    layout["xaxis"] = dict(
        tickmode="array",
        tickvals=list(range(len(data))),
        ticktext=x_labels,
        tickangle=-35,
        automargin=True,
    )
    layout["yaxis"] = dict(title=pretty_name)

    fig.update_layout(**layout)

    return PlotlySpec(
        title=f"{pretty_name} — ранжирование",
        html=_to_html(fig, f"{pretty_name} — ранжирование"),
        description=f"Сегменты упорядочены по показателю «{pretty_name}».",
    )




def _kpi_zscore_heatmap(result: AnalysisResult) -> PlotlySpec | None:
    m = result.metrics
    cols = [c for c in ("CTR", "CVR", "CPA", "CPC") if c in m.columns and m[c].notna().any()]
    if len(cols) < 2:
        return None

    keep = [result.group_col] + cols
    data = m[keep].copy().set_index(result.group_col)
    data = data.astype(float)

    z = data.copy()
    for c in cols:
        col = data[c]
        mean = col.mean(skipna=True)
        std = col.std(skipna=True, ddof=0)
        if std and not pd.isna(std) and std > 0:
            z[c] = (col - mean) / std
        else:
            z[c] = 0.0

    abs_max = float(np.nanmax(np.abs(z.values))) if z.size else 1.0
    abs_max = max(abs_max, 0.5)

    labels_map = result.metric_labels
    annotations = []

    if (
            result.channel_col
            and result.campaign_col
            and {result.channel_col, result.campaign_col}.issubset(m.columns)
    ):
        info = (
            m[[result.group_col, result.campaign_col, result.channel_col]]
            .drop_duplicates(result.group_col)
            .set_index(result.group_col)
            .loc[data.index]
        )

        campaigns = info[result.campaign_col].astype(str).tolist()
        channels = info[result.channel_col].astype(str).tolist()
        y_labels = [_truncate(ch, 18) for ch in channels]

        start = 0
        current = campaigns[0]
        for i, camp in enumerate(campaigns[1:], start=1):
            if camp != current:
                center = (start + i - 1) / 2
                annotations.append(
                    dict(
                        x=-0.34,
                        y=center,
                        xref="paper",
                        yref="y",
                        text=_truncate(current, 18),
                        showarrow=False,
                        xanchor="right",
                        yanchor="middle",
                        align="right",
                        font=dict(size=11),
                    )
                )
                start = i
                current = camp

        center = (start + len(campaigns) - 1) / 2
        annotations.append(
            dict(
                x=-0.34,
                y=center,
                xref="paper",
                yref="y",
                text=_truncate(current, 18),
                showarrow=False,
                xanchor="right",
                yanchor="middle",
                align="right",
                font=dict(size=11),
            )
        )
    else:
        y_labels = [_truncate(v, 32) for v in data.index.astype(str)]

    custom_text = data.round(2).astype(str).values

    fig = go.Figure(
        data=go.Heatmap(
            z=z.values,
            x=[labels_map.get(c, c) for c in cols],
            y=list(range(len(y_labels))),
            customdata=custom_text,
            colorscale="RdYlGn",
            zmid=0,
            zmin=-abs_max,
            zmax=abs_max,
            colorbar=dict(
                title=dict(
                    text="Отклонение от среднего, σ",
                    side="right"
                ),
                thickness=18,
                len=0.95,
                y=0.5,
                yanchor="middle",
            ),
            hovertemplate=(
                "Сегмент: %{y}<br>"
                "Метрика: %{x}<br>"
                "Значение: %{customdata}<br>"
                "Отклонение: %{z:.2f}σ<extra></extra>"
            ),
        )
    )

    layout = dict(_BASE_LAYOUT)
    layout["margin"] = dict(l=180, r=80, t=70, b=90)
    layout["showlegend"] = False
    layout["annotations"] = annotations
    layout["xaxis"] = dict(title="KPI", tickangle=20)
    layout["yaxis"] = dict(
        title=None,
        tickmode="array",
        tickvals=list(range(len(y_labels))),
        ticktext=y_labels,
        autorange="reversed",
        automargin=True,
    )

    fig.update_layout(**layout)

    return PlotlySpec(
        title="Тепловая карта KPI",
        html=_to_html(fig, "Тепловая карта KPI"),
        description=(
            "Цвет показывает, насколько сегмент выше или ниже среднего по KPI; "
            "подробные фактические значения видны при наведении."
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

    metric = result.age_metric
    if not metric or metric not in age_table.columns:
        return None

    age_col = "Возрастная группа"
    pretty_metric = result.metric_labels.get(metric, metric)

    has_campaign_channel = bool(result.campaign_col and result.channel_col)
    if has_campaign_channel and (
        result.campaign_col in age_table.columns and result.channel_col in age_table.columns
    ):
        channels = (
            age_table[result.channel_col]
            .dropna()
            .astype(str)
            .drop_duplicates()
            .tolist()
        )
        if not channels:
            return None

        fig = go.Figure()
        trace_channel_map: list[str] = []

        for ch in channels:
            ch_df = age_table[age_table[result.channel_col].astype(str) == ch].copy()
            campaigns = (
                ch_df[result.campaign_col]
                .dropna()
                .astype(str)
                .drop_duplicates()
                .tolist()
            )

            for camp in campaigns:
                sub = ch_df[ch_df[result.campaign_col].astype(str) == camp].copy()
                series = (
                    sub.groupby(age_col, dropna=True, observed=True)[metric]
                    .sum()
                    .reindex(AGE_LABELS, fill_value=0)
                )

                fig.add_trace(
                    go.Bar(
                        x=series.index.astype(str),
                        y=series.values.astype(float),
                        name=_truncate(camp, 28),
                        visible=(ch == channels[0]),
                        hovertemplate=(
                            f"Канал: {ch}<br>"
                            f"Кампания: {camp}<br>"
                            "Возраст: %{x}<br>"
                            f"{pretty_metric}: " +
                            ("%{y:.2f}" if metric in {"CTR", "CVR", "CPC", "CPA", "ROAS", "AOV"} else "%{y:,.0f}") +
                            "<extra></extra>"
                        ),
                    )
                )
                trace_channel_map.append(ch)

        buttons = []
        for ch in channels:
            visible = [trace_ch == ch for trace_ch in trace_channel_map]
            buttons.append(
                dict(
                    label=str(ch),
                    method="update",
                    args=[
                        {"visible": visible},
                        {}
                    ],
                )
            )

        fig.update_layout(
            _BASE_LAYOUT,
            barmode="group",
            xaxis_title="Возрастная группа",
            yaxis_title=pretty_metric,
            margin=dict(l=60, r=30, t=70, b=110),
            updatemenus=[
                dict(
                    buttons=buttons,
                    direction="down",
                    showactive=True,
                    x=0.0,
                    xanchor="left",
                    y=1.16,
                    yanchor="top",
                    pad=dict(r=8, t=0),
                )
            ],
        )

        return PlotlySpec(
            title=f"{pretty_metric} по возрастным группам",
            html=_to_html(fig, f"{pretty_metric} по возрастным группам"),
            description="Сравнение кампаний по возрастным группам с переключением по каналам.",
        )

    pivot = age_table.pivot_table(
        index=age_col,
        columns=result.group_col,
        values=metric,
        aggfunc="sum",
        fill_value=0,
        observed=True,
    )

    if pivot.empty:
        return None

    ordered_ages = [label for label in AGE_LABELS if label in pivot.index]
    pivot = pivot.reindex(ordered_ages)

    fig = go.Figure()
    for seg in pivot.columns:
        fig.add_trace(
            go.Bar(
                x=pivot.index.astype(str),
                y=pivot[seg].astype(float),
                name=_truncate(seg, 28),
            )
        )

    title = f"{pretty_metric} по возрастным группам"

    fig.update_layout(
        _BASE_LAYOUT,
        barmode="group",
        xaxis_title="Возрастная группа",
        yaxis_title=pretty_metric,
        title=dict(text=title, x=0.02, xanchor="left"),
    )

    return PlotlySpec(
        title=title,
        html=_to_html(fig, title),
        description=f"Распределение по возрастным группам по метрике «{pretty_metric}».",
    )


def _extra_category_chart(result: AnalysisResult) -> PlotlySpec | None:
    """График по дополнительной категории без разбивки по каналам."""
    es = result.extra_summary
    if es is None or es.empty or not result.extra_metric:
        return None

    cat_col = result.extra_category_col
    metric = result.extra_metric
    if cat_col not in es.columns or metric not in es.columns:
        return None

    pretty_metric = result.metric_labels.get(metric, metric)

    data = (
        es[[cat_col, metric]]
        .dropna(subset=[metric])
        .sort_values(metric, ascending=False)
        .copy()
    )
    if data.empty:
        return None

    fig = go.Figure(
        go.Bar(
            x=data[cat_col].astype(str),
            y=data[metric].astype(float),
            marker_color="#1c8f9e",
            hovertemplate=(
                "%{x}<br>"
                f"{pretty_metric}: " +
                ("%{y:.2f}" if metric in {"CTR", "CVR", "CPC", "CPA", "ROAS", "AOV"} else "%{y:,.0f}") +
                "<extra></extra>"
            ),
        )
    )

    title = f"{pretty_metric} по {cat_col}"

    layout = dict(_BASE_LAYOUT)
    layout["title"] = dict(text=title, x=0.02, xanchor="left")
    layout["showlegend"] = False
    layout["xaxis"] = dict(
        title=str(cat_col),
        tickangle=-35,
        automargin=True,
    )
    layout["yaxis"] = dict(
        title=pretty_metric,
        rangemode="tozero",
    )
    layout["margin"] = dict(l=60, r=30, t=70, b=120)

    fig.update_layout(layout)

    return PlotlySpec(
        title=title,
        html=_to_html(fig, title),
        description=f"Сравнение значений категории «{cat_col}» по метрике «{pretty_metric}».",
    )


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

def _kpi_heatmap_grouped(result: AnalysisResult) -> PlotlySpec | None:
    m = result.metrics
    cols = [c for c in ("CTR", "CVR", "CPA", "CPC") if c in m.columns and m[c].notna().any()]
    if len(cols) < 2:
        return None

    data = m[[result.group_col] + cols].copy().set_index(result.group_col).astype(float)

    z = data.copy()
    for c in cols:
        col = data[c]
        mean = col.mean(skipna=True)
        std = col.std(skipna=True, ddof=0)
        if std and not pd.isna(std) and std > 0:
            z[c] = (col - mean) / std
        else:
            z[c] = 0.0

    abs_max = float(np.nanmax(np.abs(z.values))) if z.size else 1.0
    abs_max = max(abs_max, 0.5)

    labels_map = result.metric_labels
    y_vals = list(range(len(data)))

    if (
        result.channel_col
        and result.campaign_col
        and {result.channel_col, result.campaign_col}.issubset(m.columns)
    ):
        info = (
            m[[result.group_col, result.campaign_col, result.channel_col]]
            .drop_duplicates(result.group_col)
            .set_index(result.group_col)
            .loc[data.index]
        )

        campaigns = info[result.campaign_col].astype(str).tolist()
        channels = info[result.channel_col].astype(str).tolist()

        y_labels = []
        prev_campaign = None
        for camp, ch in zip(campaigns, channels):
            camp_txt = _truncate(camp, 18)
            ch_txt = _truncate(ch, 18)
            if camp != prev_campaign:
                y_labels.append(f"<b>{camp_txt}</b><br>{ch_txt}")
            else:
                y_labels.append(f"<br>{ch_txt}")
            prev_campaign = camp
    else:
        y_labels = [_truncate(v, 32) for v in data.index.astype(str)]

    text_values = data[cols].round(2).astype(str).values

    fig = go.Figure(
        data=go.Heatmap(
            z=z[cols].values,
            x=[labels_map.get(c, c) for c in cols],
            y=y_vals,
            text=text_values,
            texttemplate="%{text}",
            textfont=dict(size=11),
            customdata=text_values,
            colorscale="RdYlGn",
            zmid=0,
            zmin=-abs_max,
            zmax=abs_max,
            colorbar=dict(
                title=dict(
                    text="Отклонение от среднего, σ",
                    side="right",
                ),
                thickness=18,
                len=0.98,
                y=0.5,
                yanchor="middle",
            ),
            hovertemplate=(
                "Строка: %{y}<br>"
                "Метрика: %{x}<br>"
                "Значение: %{customdata}<br>"
                "Отклонение: %{z:.2f}σ<extra></extra>"
            ),
        )
    )

    layout = dict(_BASE_LAYOUT)
    layout["margin"] = dict(l=70, r=50, t=50, b=70)
    layout["showlegend"] = False
    layout["xaxis"] = dict(title="KPI", tickangle=20)
    layout["yaxis"] = dict(
        title=None,
        tickmode="array",
        tickvals=y_vals,
        ticktext=y_labels,
        autorange="reversed",
        automargin=True,
        tickfont=dict(size=11),
    )
    layout["height"] = max(315, 30 * len(y_vals)) #layout["height"] = max(420, 38 * len(y_vals))

    fig.update_layout(**layout)

    return PlotlySpec(
        title="Тепловая карта KPI",
        html=_to_html(fig, "Тепловая карта KPI"),
        description=(
            "Цвет показывает отклонение KPI от среднего, строки сгруппированы по кампаниям, "
            "внутри группы показаны каналы, а в ячейках отображаются фактические значения."
        ),
    )

# ---------- основной build --------------------------------------------------
def build_plotly_charts(result: AnalysisResult) -> list[PlotlySpec]:
    """Собирает все доступные Plotly-графики; порядок от обзорных к детальным."""
    specs: list[PlotlySpec] = []

    # ---------- месячные графики ----------
    if result.channel_col:
        for metric, color in (
            ("displays", "#0F766E"),
            ("clicks", "#2563EB"),
            ("conversions", "#059669"),
            ("CTR", "#B45309"),
            ("CVR", "#7C3AED"),
            ("CPC", "#DC2626"),
            ("CPA", "#BE185D"),
        ):
            spec = _month_single_metric_chart(result, metric, color)
            if spec is not None:
                specs.append(spec)
    else:
        month_volume = month_volume_combo_result(result)
        if month_volume is not None:
            specs.append(month_volume)

        month_kpi = month_kpi_combo_result(result)
        if month_kpi is not None:
            specs.append(month_kpi)

    # ---------- обзорные графики ----------
    funnel = _funnel_combo(result)
    if funnel is not None:
        specs.append(funnel)

    hm = _kpi_heatmap_grouped(result)
    if hm is not None:
        specs.append(hm)

    share_compare = build_share_comparison_chart(result)
    if share_compare is not None:
        specs.append(share_compare)

    extra = _extra_category_chart(result)
    if extra is not None:
        specs.append(extra)

    age = _age_chart(result)
    if age is not None:
        specs.append(age)

    # Если уже есть общий график по базовым метрикам, не дублируем их ranking-графиками.
    combined_funnel_metrics = {
        col
        for col in ("displays", "clicks", "conversions")
        if col in result.metrics.columns and result.metrics[col].notna().any()
    }
    has_combined_funnel = len(combined_funnel_metrics) >= 2

    # ---------- ranking / detail ----------
    for col, color in (
        ("conversions", "#059669"),
        ("clicks", "#2563EB"),
        ("displays", "#0F766E"),
        ("total_cost", "#CC6677"),
        ("CTR", "#B45309"),
        ("CVR", "#7C3AED"),
        ("CPA", "#BE185D"),
        ("CPC", "#DC2626"),
    ):
        if col not in result.metrics.columns:
            continue
        if not result.metrics[col].notna().any():
            continue

        if has_combined_funnel and col in {"displays", "clicks", "conversions"}:
            continue

        spec = _ranking_bar(result, col, color)
        if spec is not None:
            specs.append(spec)

    return specs

def build_share_comparison_chart(result: AnalysisResult) -> PlotlySpec | None:
    m = result.metrics
    cols = [c for c in ("costshare", "convshare") if c in m.columns]
    if not cols:
        return None

    data = m.dropna(subset=cols, how="all")
    if data.empty:
        return None

    data = _sorted_for_segments(result, data, "convshare" if "convshare" in cols else cols[0])
    x, kind = _segment_x(result, data)

    pretty = {
        "costshare": "Доля расходов, %",
        "convshare": "Доля конверсий, %",
    }
    colors = {
        "costshare": "#CC6677",
        "convshare": "#117733",
    }

    fig = go.Figure()

    for c in cols:
        fig.add_trace(
            go.Bar(
                x=x,
                y=data[c].astype(float),
                name=pretty[c],
                marker_color=colors[c],
                hovertemplate=f"%{{x}}<br>{pretty[c]}: %{{y:.2f}}<extra></extra>",
            )
        )

    layout = dict(_BASE_LAYOUT)
    layout["barmode"] = "group"
    layout["yaxis"] = dict(title="")
    layout["margin"] = dict(l=60, r=30, t=70, b=140)

    if kind == "multi":
        layout["shapes"] = campaignseparatorshapes(result, data)

    fig.update_layout(layout)
    _apply_segment_xaxis(fig, kind)

    return PlotlySpec(
        title="Сравнение долей",
        html=_to_html(fig, "Сравнение долей"),
        description="Сравнивает долю расходов и долю конверсий по сегментам.",
    )

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

def month_metric_lines_result(result: AnalysisResult, value_col: str, color: str) -> PlotlySpec | None:
    month_df = getattr(result, "month_table", None)
    if month_df is None or month_df.empty or value_col not in month_df.columns:
        return None

    pretty = result.metric_labels.get(value_col, value_col)
    fig = go.Figure()

    for _, part in month_df.groupby(result.group_col, dropna=True):
        part = part.copy()
        row0 = part.iloc[0]
        legend_name = _month_series_name(result, row0)

        fig.add_trace(go.Scatter(
            x=part["Месяц"].astype(str),
            y=part[value_col].astype(float),
            name=legend_name,
            mode="lines+markers",
            line=dict(color=color, width=3),
            marker=dict(size=8),
            hovertemplate=(
                f"<b>%{{x}}</b><br>"
                f"{pretty}: %{{y:,.2f}}<br>"
                f"Сегмент: {legend_name}"
                f"<extra></extra>"
            ),
        ))

    title = f"{pretty} по месяцам"
    fig.update_layout(
        **_BASE_LAYOUT,
        xaxis=dict(title="Месяц"),
        yaxis=dict(title=pretty, rangemode="tozero"),
    )

    return PlotlySpec(
        title=title,
        html=_to_html(fig, title),
        description="Помесячная динамика с разбивкой по выбранным сегментам.",
    )

def build_share_comparison_chart(result: AnalysisResult) -> PlotlySpec | None:
    """Сравнение долей затрат, конверсий и выручки по сегментам."""
    df = result.metrics.copy()

    needed = ("cost_share", "conv_share", "revenue_share")
    available = [c for c in needed if c in df.columns and df[c].notna().any()]
    if len(available) < 2:
        return None

    df = df.dropna(subset=available, how="all").copy()
    if df.empty:
        return None

    if result.channel_col and result.campaign_col:
        if result.campaign_col not in df.columns or result.channel_col not in df.columns:
            return None
        df = df.sort_values(
            [result.campaign_col, result.channel_col],
            ascending=[True, True]
        ).reset_index(drop=True)
        x = [
            df[result.campaign_col].astype(str).tolist(),
            df[result.channel_col].astype(str).tolist(),
        ]
        kind = "multi"
    else:
        sort_col = "conv_share" if "conv_share" in available else available[0]
        df = df.sort_values(sort_col, ascending=False).reset_index(drop=True)
        x = df[result.group_col].astype(str).tolist()
        kind = "flat"

    labels = {
        "cost_share": "Доля затрат",
        "conv_share": "Доля конверсий",
        "revenue_share": "Доля выручки",
    }
    colors = {
        "cost_share": "#C65D3A",
        "conv_share": "#2E8B57",
        "revenue_share": "#2F6DB3",
    }

    fig = go.Figure()

    for col in available:
        fig.add_trace(
            go.Bar(
                x=x,
                y=pd.to_numeric(df[col], errors="coerce").fillna(0).tolist(),
                name=labels[col],
                marker_color=colors[col],
                text=[f"{v:.1f}%" if pd.notna(v) else "" for v in df[col]],
                textposition="outside",
                cliponaxis=False,
                hovertemplate=f"{labels[col]}: %{{y:.2f}}%<extra></extra>",
            )
        )

    layout = dict(_BASE_LAYOUT)
    layout["barmode"] = "group"
    layout["margin"] = dict(l=60, r=30, t=70, b=140)
    layout["yaxis"] = dict(
        title="Доля, %",
        ticksuffix="%",
        rangemode="tozero",
    )

    if result.channel_col and result.campaign_col:
        layout["shapes"] = campaignseparatorshapes(result, df)

    fig.update_layout(layout)

    _apply_segment_xaxis(fig, kind)
    fig.update_xaxes(tickangle=-35, automargin=True)

    title = "Сравнение долей"
    return PlotlySpec(
        title=title,
        html=_to_html(fig, title),
        description="Сравнение доли затрат, конверсий и выручки по сегментам.",
    )