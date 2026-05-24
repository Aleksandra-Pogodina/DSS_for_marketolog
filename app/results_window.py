"""Окно результатов: KPI-карточки, метрики, рекомендации, дашборд-графики, экспорт DOCX.

UI-графики строятся через Plotly и отображаются в одном растягиваемом
QWebEngineView с боковым списком названий: одновременно виден только один
график, который занимает всё доступное пространство.
"""

from __future__ import annotations

import os
import shutil
import tempfile

import pandas as pd
from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QFileDialog, QFrame, QHBoxLayout, QHeaderView, QLabel, QListWidget,
    QListWidgetItem, QMessageBox, QPushButton, QSizePolicy, QSplitter,
    QStackedWidget, QTabWidget, QTableWidget, QTableWidgetItem, QTextEdit,
    QVBoxLayout, QWidget,
)

try:
    from PySide6.QtWebEngineWidgets import QWebEngineView
    HAVE_WEBENGINE = True
except ImportError:  # pragma: no cover
    QWebEngineView = None
    HAVE_WEBENGINE = False

from app.charts import Chart, build_charts, cleanup_charts
from app.plotly_charts import PlotlySpec, build_plotly_charts
from app.processing import AnalysisResult, format_metrics_for_display
from app.report import export_docx
from app.theme import get_palette


def _format_compact(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "—"
    v = float(value)
    if abs(v) >= 1_000_000:
        return f"{v / 1_000_000:.2f}M"
    if abs(v) >= 1000:
        return f"{v:,.0f}".replace(",", " ")
    if v == int(v):
        return f"{int(v)}"
    return f"{v:.2f}"


def _card_frame() -> QFrame:
    f = QFrame()
    f.setObjectName("Card")
    return f


class _KpiCard(QFrame):
    """KPI-карточка с подписью и значением. Цвета подстраиваются под текущую тему."""

    def __init__(self, label: str, value: str, bg: str, fg: str):
        super().__init__()
        self.setObjectName("KpiCard")
        self._bg = bg
        self._fg = fg
        self.setStyleSheet(
            f"QFrame#KpiCard {{ background: {bg}; border: 1px solid {bg}; border-radius: 10px; }}"
        )
        lay = QVBoxLayout(self)
        lay.setContentsMargins(14, 12, 14, 12)
        lay.setSpacing(2)

        self.label = QLabel(label)
        self.label.setStyleSheet(f"color: {fg}; font-weight: 600; font-size: 12px;")
        self.value = QLabel(value)
        self.value.setStyleSheet(f"color: {fg}; font-size: 22px; font-weight: 700;")

        lay.addWidget(self.label)
        lay.addWidget(self.value)

    def apply_palette(self, bg: str, fg: str) -> None:
        self._bg, self._fg = bg, fg
        self.setStyleSheet(
            f"QFrame#KpiCard {{ background: {bg}; border: 1px solid {bg}; border-radius: 10px; }}"
        )
        self.label.setStyleSheet(f"color: {fg}; font-weight: 600; font-size: 12px;")
        self.value.setStyleSheet(f"color: {fg}; font-size: 22px; font-weight: 700;")


class ResultsWindow(QWidget):
    def __init__(self, result: AnalysisResult):
        super().__init__()
        self.result = result
        self.tmpdir = tempfile.mkdtemp(prefix="mkt_results_")
        self.charts: list[Chart] = []
        self.plotly_specs: list[PlotlySpec] = []

        self._chart_widgets: list[QWidget] = []
        self._chart_loaded: list[bool] = []
        self._kpi_cards: list[_KpiCard] = []

        self.setWindowTitle("Результаты анализа")
        self.resize(1600, 980)

        root = QVBoxLayout(self)
        root.setContentsMargins(24, 22, 24, 22)
        root.setSpacing(16)

        root.addLayout(self._build_header())
        root.addLayout(self._build_kpi_strip())
        root.addWidget(self._build_tabs(), 1)

    # ---------- шапка / KPI -------------------------------------------------

    def _build_header(self) -> QHBoxLayout:
        layout = QHBoxLayout()
        layout.setSpacing(10)

        text_block = QVBoxLayout()
        text_block.setSpacing(2)

        title = QLabel("Результаты анализа")
        title.setObjectName("AppTitle")
        text_block.addWidget(title)

        subtitle = QLabel(
            f"Группировка: <b>{self.result.group_label}</b> · "
            f"строк после очистки: {self.result.rows_loaded} из {self.result.rows_original} "
            f"(удалено {self.result.rows_dropped})"
        )
        subtitle.setObjectName("AppSubtitle")
        subtitle.setTextFormat(Qt.TextFormat.RichText)
        text_block.addWidget(subtitle)

        layout.addLayout(text_block)
        layout.addStretch()

        self.docx_btn = QPushButton("Сохранить отчёт DOCX")
        self.docx_btn.setProperty("primary", True)
        self.docx_btn.setMinimumWidth(220)
        self.docx_btn.setMinimumHeight(38)
        self.docx_btn.clicked.connect(self.export_docx)
        layout.addWidget(self.docx_btn, 0, Qt.AlignmentFlag.AlignVCenter)

        from app.main_window import ThemeToggleButton  # позднее, чтобы избежать циклов
        self.theme_btn = ThemeToggleButton(self)
        layout.addWidget(self.theme_btn, 0, Qt.AlignmentFlag.AlignVCenter)

        return layout

    def _build_kpi_strip(self) -> QHBoxLayout:
        m = self.result.metrics
        cards_data: list[tuple[str, str]] = []

        def add_sum(label: str, col: str) -> None:
            if col in m.columns and m[col].notna().any():
                cards_data.append((label, _format_compact(m[col].sum(skipna=True))))

        add_sum("Показы", "displays")
        add_sum("Клики", "clicks")
        add_sum("Конверсии", "conversions")
        add_sum("Затраты", "total_cost")

        def add_avg_pct(label: str, col: str) -> None:
            if col in m.columns and m[col].notna().any():
                cards_data.append((label, f"{m[col].mean(skipna=True):.2f}%"))

        add_avg_pct("Ср. CTR", "CTR")
        add_avg_pct("Ср. CVR", "CVR")

        row = QHBoxLayout()
        row.setSpacing(12)
        palette = get_palette().kpi
        for i, (label, value) in enumerate(cards_data):
            bg, fg = palette[i % len(palette)]
            card = _KpiCard(label, value, bg, fg)
            card.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            row.addWidget(card, 1)
            self._kpi_cards.append(card)
        return row

    # ---------- вкладки -----------------------------------------------------

    def _build_tabs(self) -> QTabWidget:
        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)

        self.tabs.addTab(self._make_metrics_tab(), "Метрики")
        self.tabs.addTab(self._make_recommendations_tab(), "Рекомендации (DSS)")

        self.plotly_specs = build_plotly_charts(self.result)
        self.charts = build_charts(self.result, tmpdir=self.tmpdir)

        if self.plotly_specs:
            self.tabs.addTab(self._make_dashboard_tab(self.plotly_specs),
                             f"Графики ({len(self.plotly_specs)})")

        if self.result.extra_summary is not None and not self.result.extra_summary.empty:
            self.tabs.addTab(self._make_extra_tab(),
                             f"Доп. категория · {self.result.extra_category_col}")

        if self.result.age_table is not None and not self.result.age_table.empty:
            self.tabs.addTab(self._make_dataframe_tab(self.result.age_table), "По возрасту")

        if not self.result.dropped_rows.empty:
            self.tabs.addTab(
                self._make_dataframe_tab(self.result.dropped_rows),
                f"Удалённые строки ({len(self.result.dropped_rows)})",
            )

        return self.tabs

    def _make_metrics_tab(self) -> QWidget:
        return self._make_dataframe_tab(format_metrics_for_display(self.result))

    def _make_extra_tab(self) -> QWidget:
        """Сводка по дополнительной категории."""
        widget = QWidget()
        lay = QVBoxLayout(widget)
        lay.setContentsMargins(14, 14, 14, 14)
        lay.setSpacing(10)

        cat = self.result.extra_category_col
        metric_pretty = self.result.metric_labels.get(
            self.result.extra_metric, self.result.extra_metric
        )
        hint = QLabel(
            f"Сводка по дополнительной категории <b>«{cat}»</b>. "
            f"Ключевая метрика — {metric_pretty}. Если выбран столбец «Каналы», "
            "значения показаны в разрезе каналов; иначе — итоги по категории."
        )
        hint.setWordWrap(True)
        hint.setTextFormat(Qt.TextFormat.RichText)
        hint.setObjectName("SectionHint")
        lay.addWidget(hint)

        df = self._format_extra_summary()
        lay.addWidget(self._df_to_table(df), 1)
        return widget

    def _format_extra_summary(self) -> pd.DataFrame:
        es = self.result.extra_summary.copy()
        print("EXTRA ORDER FIX APPLIED", list(es.columns))
        cat_col = self.result.extra_category_col

        metric_order = [
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
        ]

        front_cols = [c for c in [cat_col] if c and c in es.columns]
        ordered_metrics = [c for c in metric_order if c in es.columns]
        other_cols = [c for c in es.columns if c not in front_cols + ordered_metrics]

        es = es[front_cols + ordered_metrics + other_cols]

        rename = {}
        if cat_col and cat_col in es.columns:
            rename[cat_col] = str(cat_col)

        for raw, label in self.result.metric_labels.items():
            if raw in es.columns:
                rename[raw] = label

        es = es.rename(columns=rename)

        for c in es.columns:
            if pd.api.types.is_numeric_dtype(es[c]):
                es[c] = es[c].round(2)

        return es

    def _make_dataframe_tab(self, df: pd.DataFrame) -> QWidget:
        widget = QWidget()
        lay = QVBoxLayout(widget)
        lay.setContentsMargins(14, 14, 14, 14)
        lay.setSpacing(0)
        lay.addWidget(self._df_to_table(df))
        return widget

    def _df_to_table(self, df: pd.DataFrame) -> QTableWidget:
        """Таблица: столбцы по содержимому, без растягивания последнего на всю ширину.

        Если столбцы не помещаются — появляется горизонтальный скролл.
        """
        table = QTableWidget()
        table.setRowCount(len(df))
        table.setColumnCount(len(df.columns))
        table.setHorizontalHeaderLabels([str(c) for c in df.columns])
        table.verticalHeader().setVisible(False)
        table.setShowGrid(False)
        table.setAlternatingRowColors(True)
        table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        table.setHorizontalScrollMode(QTableWidget.ScrollMode.ScrollPerPixel)
        for r in range(len(df)):
            for c in range(len(df.columns)):
                val = df.iloc[r, c]
                item = QTableWidgetItem("" if pd.isna(val) else str(val))
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                table.setItem(r, c, item)
        header = table.horizontalHeader()
        # Каждый столбец — по содержимому, пользователь может расширить вручную.
        header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        header.setStretchLastSection(False)
        table.resizeColumnsToContents()
        # Ограничим максимальную ширину одного столбца, чтобы один длинный не сожрал всё
        for i in range(table.columnCount()):
            w = table.columnWidth(i)
            if w > 320:
                table.setColumnWidth(i, 320)
        return table

    def _make_recommendations_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(0)

        edit = QTextEdit()
        edit.setReadOnly(True)

        pal = get_palette()
        parts: list[str] = []
        if self.result.recommendations:
            parts.append(f"<h3 style='color:{pal.primary};margin-top:0'>Рекомендации</h3><ul>")
            for r in self.result.recommendations:
                parts.append(f"<li style='margin-bottom:8px;line-height:1.45'>{r}</li>")
            parts.append("</ul>")
        else:
            parts.append("<p>Рекомендации не сформированы.</p>")

        if self.result.warnings:
            parts.append(f"<h3 style='color:{pal.warning};margin-top:14px'>Замечания и предупреждения</h3><ul>")
            for w in self.result.warnings:
                parts.append(f"<li style='margin-bottom:6px;color:{pal.warning}'>{w}</li>")
            parts.append("</ul>")

        edit.setHtml("\n".join(parts))
        layout.addWidget(edit)
        return widget

    # ---------- dashboard-вкладка графиков ----------------------------------

    def _make_dashboard_tab(self, specs: list[PlotlySpec]) -> QWidget:
        container = QWidget()
        outer = QVBoxLayout(container)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(8)

        if not HAVE_WEBENGINE:
            outer.addWidget(self._webengine_missing_warning())

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.setHandleWidth(8)

        sidebar = QWidget()
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(4, 4, 4, 4)
        sidebar_layout.setSpacing(8)

        sidebar_title = QLabel("Графики")
        sidebar_title.setObjectName("SectionTitle")
        sidebar_layout.addWidget(sidebar_title)

        self.chart_list = QListWidget()
        self.chart_list.setObjectName("ChartList")
        self.chart_list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.chart_list.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        for spec in specs:
            QListWidgetItem(spec.title, self.chart_list)
        self.chart_list.currentRowChanged.connect(self._on_chart_changed)
        sidebar_layout.addWidget(self.chart_list, 1)

        nav_row = QHBoxLayout()
        nav_row.setSpacing(6)
        prev_btn = QPushButton("← Назад")
        prev_btn.setObjectName("GhostButton")
        next_btn = QPushButton("Далее →")
        next_btn.setObjectName("GhostButton")
        prev_btn.clicked.connect(lambda: self._step_chart(-1))
        next_btn.clicked.connect(lambda: self._step_chart(+1))
        nav_row.addWidget(prev_btn)
        nav_row.addWidget(next_btn)
        sidebar_layout.addLayout(nav_row)
        self._prev_btn = prev_btn
        self._next_btn = next_btn

        splitter.addWidget(sidebar)

        stage = _card_frame()
        stage_layout = QVBoxLayout(stage)
        stage_layout.setContentsMargins(16, 14, 16, 14)
        stage_layout.setSpacing(6)

        self.chart_title = QLabel("")
        self.chart_title.setObjectName("SectionTitle")
        self.chart_title.setWordWrap(True)
        stage_layout.addWidget(self.chart_title)

        self.chart_stack = QStackedWidget()
        self.chart_stack.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        stage_layout.addWidget(self.chart_stack, 1)

        self.chart_description = QLabel("")
        self.chart_description.setObjectName("SectionHint")
        self.chart_description.setWordWrap(True)
        stage_layout.addWidget(self.chart_description)

        splitter.addWidget(stage)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([260, 1340])

        outer.addWidget(splitter, 1)

        for _ in specs:
            placeholder = QWidget()
            placeholder.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            self.chart_stack.addWidget(placeholder)
            self._chart_widgets.append(placeholder)
            self._chart_loaded.append(False)

        if specs:
            self.chart_list.setCurrentRow(0)

        return container

    def _on_chart_changed(self, idx: int) -> None:
        if idx < 0 or idx >= len(self.plotly_specs):
            return
        if not self._chart_loaded[idx]:
            real = self._make_chart_widget(self.plotly_specs[idx])
            old = self._chart_widgets[idx]
            self.chart_stack.insertWidget(idx, real)
            self.chart_stack.removeWidget(old)
            old.deleteLater()
            self._chart_widgets[idx] = real
            self._chart_loaded[idx] = True

        self.chart_stack.setCurrentIndex(idx)
        spec = self.plotly_specs[idx]
        self.chart_title.setText(spec.title)
        self.chart_description.setText(spec.description or "")
        self._prev_btn.setEnabled(idx > 0)
        self._next_btn.setEnabled(idx < len(self.plotly_specs) - 1)

    def _step_chart(self, delta: int) -> None:
        if not self.plotly_specs:
            return
        new_idx = max(0, min(len(self.plotly_specs) - 1, self.chart_list.currentRow() + delta))
        self.chart_list.setCurrentRow(new_idx)

    def _make_chart_widget(self, spec: PlotlySpec) -> QWidget:
        if HAVE_WEBENGINE:
            view = QWebEngineView()
            view.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            view.setHtml(spec.html, QUrl("about:blank"))
            # Подложить цвет фона текущей темы под Plotly (HTML прозрачный)
            pal = get_palette()
            view.page().setBackgroundColor(Qt.GlobalColor.transparent)
            view.setStyleSheet(f"background: {pal.surface};")
            return view

        fallback = self._fallback_png_widget(spec.title)
        if fallback is not None:
            return fallback
        lbl = QLabel(
            "Для интерактивного графика установите PySide6-Addons (включает QtWebEngine). "
            "См. README."
        )
        lbl.setObjectName("SectionHint")
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl.setWordWrap(True)
        return lbl

    def _fallback_png_widget(self, plotly_title: str) -> QWidget | None:
        for ch in self.charts:
            if not ch.title:
                continue
            if plotly_title.lower() in ch.title.lower() or ch.title.lower() in plotly_title.lower():
                return _ScaledPixmapLabel(ch.path)
        return None

    def _webengine_missing_warning(self) -> QFrame:
        frame = QFrame()
        frame.setStyleSheet(
            "QFrame { background: rgba(208,140,30,0.15); border: 1px solid rgba(208,140,30,0.4); border-radius: 8px; }"
        )
        lay = QHBoxLayout(frame)
        lay.setContentsMargins(14, 10, 14, 10)
        lbl = QLabel(
            "<b>Интерактивные графики недоступны</b>: модуль "
            "<code>PySide6.QtWebEngineWidgets</code> не установлен. "
            "Графики показаны как статические изображения — установите "
            "<code>PySide6-Addons</code>, чтобы вернуть интерактивность."
        )
        lbl.setWordWrap(True)
        lay.addWidget(lbl)
        return frame

    # ---------- экспорт -----------------------------------------------------

    def export_docx(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Сохранить отчёт DOCX", "marketing_report.docx", "Word (*.docx)"
        )
        if not path:
            return
        if not path.lower().endswith(".docx"):
            path += ".docx"
        try:
            export_docx(self.result, self.charts, path)
            QMessageBox.information(self, "Отчёт сохранён", f"DOCX сохранён:\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка экспорта", f"{type(e).__name__}: {e}")

    # ---------- тема --------------------------------------------------------

    def on_theme_changed(self) -> None:
        """Перерисовать KPI-карточки и пересоздать Plotly-страницы под новую тему."""
        pal_kpi = get_palette().kpi
        for i, card in enumerate(self._kpi_cards):
            bg, fg = pal_kpi[i % len(pal_kpi)]
            card.apply_palette(bg, fg)

        # Перезаливаем фон у созданных WebEngineView; HTML страницы прозрачные,
        # так что менять контент не нужно — достаточно подложки.
        if HAVE_WEBENGINE:
            pal = get_palette()
            for w in self._chart_widgets:
                if isinstance(w, QWebEngineView):
                    w.setStyleSheet(f"background: {pal.surface};")
        self.style().polish(self)

    # ---------- очистка -----------------------------------------------------

    def closeEvent(self, event):
        try:
            cleanup_charts(self.charts)
            if os.path.isdir(self.tmpdir):
                shutil.rmtree(self.tmpdir, ignore_errors=True)
        finally:
            super().closeEvent(event)


class _ScaledPixmapLabel(QLabel):
    def __init__(self, path: str):
        super().__init__()
        self._pix_source = QPixmap(path)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(200, 200)

    def resizeEvent(self, event):
        if not self._pix_source.isNull():
            self.setPixmap(
                self._pix_source.scaled(
                    self.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )
        super().resizeEvent(event)
