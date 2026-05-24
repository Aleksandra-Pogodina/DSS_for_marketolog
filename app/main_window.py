"""Главное окно: загрузка файла, сопоставление столбцов, запуск анализа."""

from __future__ import annotations

import os

import pandas as pd
from PySide6.QtCore import Qt, QAbstractTableModel, QModelIndex
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import (
    QApplication, QComboBox, QFileDialog, QFrame, QGridLayout, QHBoxLayout,
    QHeaderView, QLabel, QMainWindow, QMessageBox, QPushButton, QSizePolicy,
    QTableView, QVBoxLayout, QWidget,
)

from app.data_mapping import (
    NUMERIC_ROLES, ROLE_LABELS_RU, ROLE_NAMES, auto_guess_mapping, classify_value,
)
from app.processing import process_data
from app.results_window import ResultsWindow
from app.theme import (
    THEME_DARK, THEME_LIGHT, apply_theme, current_theme_name, get_palette,
    save_theme, toggle_theme,
)


class PandasModel(QAbstractTableModel):
    """Модель для предпросмотра датасета с подсветкой подозрительных значений."""

    def __init__(self, df: pd.DataFrame):
        super().__init__()
        self._df = df.reset_index(drop=True)
        self.column_types = self._detect_column_types()
        self.suspicious_cells = self._find_suspicious_cells()

    def _detect_column_types(self) -> dict:
        types = {}
        for col in self._df.columns:
            values = self._df[col].dropna()
            if values.empty:
                types[col] = "empty"
                continue
            counts = {"numeric": 0, "date": 0, "text": 0}
            for v in values.head(200):
                t = classify_value(v)
                if t in counts:
                    counts[t] += 1
            types[col] = max(counts, key=counts.get)
        return types

    def _find_suspicious_cells(self) -> set:
        suspicious = set()
        for col_idx, col in enumerate(self._df.columns):
            dominant = self.column_types.get(col)
            if dominant in (None, "empty"):
                continue
            for row_idx, value in enumerate(self._df[col]):
                if pd.isna(value):
                    continue
                vtype = classify_value(value)
                if dominant == "numeric" and vtype != "numeric":
                    suspicious.add((row_idx, col_idx))
                elif dominant == "date" and vtype not in ("date", "numeric"):
                    suspicious.add((row_idx, col_idx))
        return suspicious

    def problem_columns(self):
        cols = []
        seen = set()
        for _, col_idx in self.suspicious_cells:
            name = self._df.columns[col_idx]
            if name not in seen:
                cols.append(str(name))
                seen.add(name)
        return cols

    def rowCount(self, parent=QModelIndex()):
        return len(self._df)

    def columnCount(self, parent=QModelIndex()):
        return len(self._df.columns)

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        value = self._df.iat[index.row(), index.column()]
        if role == Qt.ItemDataRole.DisplayRole:
            return "" if pd.isna(value) else str(value)
        if role == Qt.ItemDataRole.BackgroundRole:
            if (index.row(), index.column()) in self.suspicious_cells:
                pal = get_palette()
                return QBrush(QColor(pal.warning).lighter(180 if pal.name == "light" else 110))
        return None

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role != Qt.ItemDataRole.DisplayRole:
            return None
        if orientation == Qt.Orientation.Horizontal:
            return str(self._df.columns[section])
        return str(self._df.index[section])


def _card(child_layout: QVBoxLayout | None = None) -> QFrame:
    card = QFrame()
    card.setObjectName("Card")
    if child_layout is not None:
        card.setLayout(child_layout)
    return card


class ThemeToggleButton(QPushButton):
    """Кнопка переключения светлой/тёмной темы. Хранит выбор в QSettings."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("ThemeToggle")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setMinimumHeight(28)
        self.refresh_label()
        self.clicked.connect(self._on_click)

    def refresh_label(self) -> None:
        is_dark = current_theme_name() == THEME_DARK
        self.setText(("🌙 Тёмная" if is_dark else "☀ Светлая") + "  ⇄")
        self.setToolTip("Переключить тему (светлая / тёмная)")

    def _on_click(self) -> None:
        app = QApplication.instance()
        if app is None:
            return
        toggle_theme(app)
        # Уведомить всех слушателей: окно результатов обновит Plotly-страницы под тему.
        for w in app.topLevelWidgets():
            if hasattr(w, "on_theme_changed"):
                try:
                    w.on_theme_changed()
                except Exception:  # pragma: no cover
                    pass
        self.refresh_label()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Маркетинговая аналитика")
        self.resize(1480, 940)

        self.df: pd.DataFrame | None = None
        self.model: PandasModel | None = None
        self.results_window: ResultsWindow | None = None
        self.role_widgets: dict[str, QComboBox] = {}

        central = QWidget()
        central.setObjectName("RootSurface")
        self.setCentralWidget(central)

        root = QVBoxLayout(central)
        root.setContentsMargins(24, 22, 24, 22)
        root.setSpacing(16)

        root.addLayout(self._build_header())
        root.addWidget(self._build_upload_card())
        root.addWidget(self._build_mapping_card())
        root.addWidget(self._build_preview_card(), 1)
        root.addWidget(self._build_cta_card())

    # ---------- сборка макета -----------------------------------------------

    def _build_header(self) -> QHBoxLayout:
        layout = QHBoxLayout()
        layout.setSpacing(10)

        text_block = QVBoxLayout()
        text_block.setSpacing(2)
        title = QLabel("Маркетинговая аналитика")
        title.setObjectName("AppTitle")
        subtitle = QLabel(
            "Поддержка принятия решений на основе ваших данных"
        )
        subtitle.setObjectName("AppSubtitle")
        text_block.addWidget(title)
        text_block.addWidget(subtitle)
        layout.addLayout(text_block)
        layout.addStretch()

        self.header_status = QLabel("Файл не загружен")
        self.header_status.setObjectName("Badge")
        layout.addWidget(self.header_status, 0, Qt.AlignmentFlag.AlignVCenter)

        self.theme_btn = ThemeToggleButton(self)
        layout.addWidget(self.theme_btn, 0, Qt.AlignmentFlag.AlignVCenter)

        return layout

    def _build_upload_card(self) -> QFrame:
        card_layout = QVBoxLayout()
        card_layout.setContentsMargins(20, 16, 20, 16)
        card_layout.setSpacing(8)

        title_row = QHBoxLayout()
        title = QLabel("1. Загрузка данных")
        title.setObjectName("SectionTitle")
        title_row.addWidget(title)
        title_row.addStretch()

        self.load_btn = QPushButton("Выбрать файл (CSV / Excel)")
        self.load_btn.setProperty("primary", True)
        self.load_btn.setMinimumWidth(240)
        self.load_btn.clicked.connect(self.load_file)
        title_row.addWidget(self.load_btn)
        card_layout.addLayout(title_row)

        self.info_label = QLabel(
            "Загрузите Excel (.xlsx, .xls) или CSV файл. "
            "После загрузки сопоставьте столбцы с ролями и запустите анализ."
        )
        self.info_label.setObjectName("SectionHint")
        self.info_label.setWordWrap(True)
        card_layout.addWidget(self.info_label)

        return _card(card_layout)

    def _build_mapping_card(self) -> QFrame:
        """Все роли в одну сетку 3-в-ряд: без вертикальной прокрутки."""
        card_layout = QVBoxLayout()
        card_layout.setContentsMargins(20, 16, 20, 16)
        card_layout.setSpacing(10)

        head = QHBoxLayout()
        title = QLabel("2. Сопоставление столбцов")
        title.setObjectName("SectionTitle")
        head.addWidget(title)
        head.addStretch()
        self.mapping_status = QLabel("Сначала загрузите файл.")
        self.mapping_status.setObjectName("SectionHint")
        head.addWidget(self.mapping_status)
        card_layout.addLayout(head)

        hint = QLabel(
            "Обязательно: «Каналы» или «Кампании», плюс минимум одно числовое поле. "
            "Поле «Доп. категория» — любое категориальное поле."
        )
        hint.setObjectName("SectionHint")
        hint.setWordWrap(True)
        card_layout.addWidget(hint)

        self.mapping_grid = QGridLayout()
        self.mapping_grid.setHorizontalSpacing(18)
        self.mapping_grid.setVerticalSpacing(6)
        # Растягиваем все три колонки одинаково, чтобы карточка использовала всю ширину
        for col in (1, 3, 5):
            self.mapping_grid.setColumnStretch(col, 1)
        card_layout.addLayout(self.mapping_grid)

        return _card(card_layout)

    def _build_preview_card(self) -> QFrame:
        card_layout = QVBoxLayout()
        card_layout.setContentsMargins(20, 16, 20, 16)
        card_layout.setSpacing(8)

        head = QHBoxLayout()
        title = QLabel("3. Предпросмотр")
        title.setObjectName("SectionTitle")
        head.addWidget(title)
        head.addStretch()
        self.preview_hint = QLabel("файл не загружен")
        self.preview_hint.setObjectName("SectionHint")
        head.addWidget(self.preview_hint)
        card_layout.addLayout(head)

        self.table_view = QTableView()
        self.table_view.setAlternatingRowColors(True)
        self.table_view.setSelectionMode(QTableView.SelectionMode.NoSelection)
        self.table_view.setShowGrid(False)
        self.table_view.verticalHeader().setVisible(False)
        self.table_view.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        card_layout.addWidget(self.table_view, 1)

        return _card(card_layout)

    def _build_cta_card(self) -> QFrame:
        card_layout = QHBoxLayout()
        card_layout.setContentsMargins(20, 14, 20, 14)
        card_layout.setSpacing(14)

        text_block = QVBoxLayout()
        text_block.setSpacing(2)
        title = QLabel("4. Запустите анализ")
        title.setObjectName("SectionTitle")
        sub = QLabel(
            "Мы посчитаем метрики, подготовим рекомендации, графики и отчёт DOCX."
        )
        sub.setObjectName("SectionHint")
        sub.setWordWrap(True)
        text_block.addWidget(title)
        text_block.addWidget(sub)
        card_layout.addLayout(text_block, 1)

        self.next_btn = QPushButton("Запустить анализ →")
        self.next_btn.setProperty("primary", True)
        self.next_btn.setMinimumHeight(40)
        self.next_btn.setMinimumWidth(240)
        self.next_btn.setEnabled(False)
        self.next_btn.clicked.connect(self.run_analysis)
        card_layout.addWidget(self.next_btn, 0, Qt.AlignmentFlag.AlignVCenter)

        wrap = _card()
        wrap.setLayout(card_layout)
        return wrap

    # ---------- загрузка ----------------------------------------------------

    def load_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Выберите файл с данными", "",
            "Данные (*.xlsx *.xls *.csv);;Все файлы (*)",
        )
        if not file_path:
            return

        try:
            df = self._read_file(file_path)
        except Exception as e:
            QMessageBox.critical(self, "Ошибка загрузки", f"Не удалось прочитать файл:\n{e}")
            return

        if df.empty:
            QMessageBox.warning(self, "Пустой файл", "В файле нет данных для анализа.")
            return

        self.df = df
        self.model = PandasModel(df)
        self.table_view.setModel(self.model)
        self.table_view.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.table_view.horizontalHeader().setStretchLastSection(False)

        rows, cols = df.shape
        problem_cols = self.model.problem_columns()
        filename = os.path.basename(file_path)
        if problem_cols:
            self.info_label.setText(
                f"Файл «{filename}» загружен. Подозрительные значения в столбцах: "
                f"{', '.join(problem_cols)} (ячйки выделены цветом)  — они будут пропущены при подсчёте."
            )
            self.header_status.setObjectName("BadgeWarn")
            self.header_status.setText(f"{filename} · есть замечания")
        else:
            self.info_label.setText(
                f"Файл «{filename}» загружен. Сопоставьте столбцы ниже и запустите анализ."
            )
            self.header_status.setObjectName("Badge")
            self.header_status.setText(f"{filename} · готов к анализу")

        self.header_status.style().unpolish(self.header_status)
        self.header_status.style().polish(self.header_status)

        self.preview_hint.setText(f"{rows:,} строк · {cols} столбцов".replace(",", " "))
        self.load_btn.setText("Загрузить другой файл")
        self.load_btn.setProperty("primary", False)
        self.load_btn.setObjectName("GhostButton")
        self.load_btn.style().unpolish(self.load_btn)
        self.load_btn.style().polish(self.load_btn)

        self.build_mapping()

    def _read_file(self, path: str) -> pd.DataFrame:
        lower = path.lower()
        if lower.endswith((".xlsx", ".xls")):
            return pd.read_excel(path)
        if lower.endswith(".csv"):
            for sep in (",", ";", "\t"):
                try:
                    df = pd.read_csv(path, sep=sep)
                    if df.shape[1] > 1:
                        return df
                except Exception:
                    continue
            return pd.read_csv(path)
        raise ValueError("Поддерживаются только .xlsx, .xls, .csv")

    # ---------- сопоставление -----------------------------------------------

    _PLACEHOLDER = "— не выбрано —"
    _COLS_PER_ROW = 3

    def _clear_mapping_grid(self) -> None:
        while self.mapping_grid.count():
            item = self.mapping_grid.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)
                w.deleteLater()

    def build_mapping(self):
        self._clear_mapping_grid()
        self.role_widgets = {}

        items = [self._PLACEHOLDER] + [str(col) for col in self.df.columns]
        guesses = auto_guess_mapping(self.df.columns)
        guessed_any = False
        palette = get_palette()

        roles = list(ROLE_NAMES)

        rows_per_col = (len(roles) + self._COLS_PER_ROW - 1) // self._COLS_PER_ROW

        for idx, role in enumerate(ROLE_NAMES):
            row = idx % rows_per_col
            visual_col = idx // rows_per_col
            col_base = visual_col * 2

            label_text = ROLE_LABELS_RU[role]
            if role in ("channels", "campaigns"):
                label_text = f"<b>{label_text}</b> <span style='color:{palette.danger}'>*</span>"
            lbl = QLabel(label_text)
            lbl.setTextFormat(Qt.TextFormat.RichText)
            lbl.setMinimumWidth(140)

            combo = QComboBox()
            combo.addItems(items)
            guess = guesses.get(role)
            if guess and guess in items:
                combo.setCurrentText(guess)
                guessed_any = True
            else:
                combo.setCurrentIndex(0)
            combo.setMinimumWidth(160)
            combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
            combo.currentIndexChanged.connect(self._on_mapping_changed)

            self.mapping_grid.addWidget(lbl, row, col_base)
            self.mapping_grid.addWidget(combo, row, col_base + 1)
            self.role_widgets[role] = combo

        if guessed_any:
            self.mapping_status.setText(
                "Роли предложены автоматически по названиям столбцов — проверьте."
            )
        else:
            self.mapping_status.setText("Подсказка: «Каналы»/«Кампании» + минимум одна метрика.")

        self._on_mapping_changed()

    def get_mapping(self) -> dict:
        mapping = {}
        for role, widget in self.role_widgets.items():
            value = widget.currentText().strip()
            mapping[role] = None if value == self._PLACEHOLDER else value
        return mapping

    def _on_mapping_changed(self, *_args) -> None:
        if not self.role_widgets:
            self.next_btn.setEnabled(False)
            return

        mapping = self.get_mapping()
        problems: list[str] = []

        if not mapping.get("channels") and not mapping.get("campaigns"):
            problems.append("выберите «Каналы» или «Кампании»")

        metric_roles = ["displays", "clicks", "conversions", "total_cost", "placement_cost", "clicks_cost", "revenue"]
        if not any(mapping.get(r) for r in metric_roles):
            problems.append("выберите хотя бы одно числовое поле")

        chosen = [v for v in mapping.values() if v is not None]
        dups = sorted({v for v in chosen if chosen.count(v) > 1})
        if dups:
            problems.append("один столбец назначен на несколько ролей: " + ", ".join(dups))

        if problems:
            self.next_btn.setEnabled(False)
            self.mapping_status.setText("Нужно ещё: " + "; ".join(problems) + ".")
        else:
            self.next_btn.setEnabled(True)
            self.mapping_status.setText("Готово — можно запускать анализ.")

    def _validate_mapping(self, mapping: dict):
        if not mapping.get("channels") and not mapping.get("campaigns"):
            raise ValueError("Выберите столбец «Каналы» или «Кампании» для группировки.")

        metric_roles = ["displays", "clicks", "conversions", "total_cost", "placement_cost", "cpc", "revenue"]
        if not any(mapping.get(role) for role in metric_roles):
            raise ValueError(
                "Выберите хотя бы один числовой столбец: показы, клики, конверсии или стоимость."
            )

        chosen = [v for v in mapping.values() if v is not None]
        dups = sorted({v for v in chosen if chosen.count(v) > 1})
        if dups:
            raise ValueError(
                "Один столбец нельзя назначать на разные роли:\n" + "\n".join(dups)
            )

    # ---------- запуск ------------------------------------------------------

    def run_analysis(self):
        try:
            mapping = self.get_mapping()
            self._validate_mapping(mapping)
            result = process_data(self.df, mapping)
        except ValueError as e:
            QMessageBox.warning(self, "Не удалось запустить анализ", str(e))
            return
        except Exception as e:
            QMessageBox.critical(self, "Ошибка анализа", f"{type(e).__name__}: {e}")
            return

        self.results_window = ResultsWindow(result)
        self.results_window.showMaximized()

    # ---------- тема --------------------------------------------------------

    def on_theme_changed(self) -> None:
        """Вызывается ThemeToggleButton после смены темы. Перекрашивает виджеты."""
        self.style().polish(self)
        if hasattr(self, "header_status"):
            self.header_status.style().unpolish(self.header_status)
            self.header_status.style().polish(self.header_status)
