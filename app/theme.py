"""Темы оформления: светлая и тёмная.

Все цвета приложения собраны в Palette. apply_theme(app, name) применяет
выбранную тему глобально (Fusion + QPalette + QSS). current_theme() и
save_theme(name) хранят выбор пользователя в QSettings, чтобы он сохранялся
между запусками.
"""

from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import QSettings, Qt
from PySide6.QtGui import QColor, QPalette


THEME_LIGHT = "light"
THEME_DARK = "dark"
THEMES = (THEME_LIGHT, THEME_DARK)


@dataclass(frozen=True)
class PaletteSpec:
    name: str
    bg: str
    surface: str
    surface_alt: str
    border: str
    border_strong: str

    text: str
    text_muted: str
    text_subtle: str

    primary: str
    primary_hover: str
    primary_pressed: str
    primary_text: str
    primary_soft: str

    accent: str
    success: str
    warning: str
    danger: str

    # цвета для KPI-карточек (фон, цвет числа/подписи)
    kpi: tuple[tuple[str, str], ...]


_LIGHT = PaletteSpec(
    name=THEME_LIGHT,
    bg="#f4f6fa",
    surface="#ffffff",
    surface_alt="#f7f9fc",
    border="#e1e6ee",
    border_strong="#cfd6e2",
    text="#1f2a37",
    text_muted="#54627a",
    text_subtle="#7a8699",
    primary="#0f7a8f",
    primary_hover="#0c6679",
    primary_pressed="#0a5667",
    primary_text="#ffffff",
    primary_soft="#dff0f2",
    accent="#1c8f9e",
    success="#2f9e6c",
    warning="#d18b1a",
    danger="#c0584b",
    kpi=(
        ("#dff0f2", "#0a5667"),
        ("#e3f3e8", "#1f7a4f"),
        ("#fff1d6", "#9a6212"),
        ("#fbe6e1", "#a8473a"),
        ("#e9ecf6", "#2e3d63"),
        ("#efe1f3", "#6a3286"),
    ),
)


_DARK = PaletteSpec(
    name=THEME_DARK,
    bg="#0f141b",
    surface="#161d27",
    surface_alt="#1c2532",
    border="#28323f",
    border_strong="#39475a",
    text="#e6ecf3",
    text_muted="#9aa7ba",
    text_subtle="#71819a",
    primary="#2bb3c7",
    primary_hover="#37c4d8",
    primary_pressed="#1f95a7",
    primary_text="#03191d",
    primary_soft="#1d3a40",
    accent="#3ec6d6",
    success="#4ed09a",
    warning="#e7b25e",
    danger="#e0786a",
    kpi=(
        ("#162d33", "#7fd9e6"),
        ("#163026", "#7ddfb2"),
        ("#2e2616", "#e9c47a"),
        ("#311e1a", "#e89789"),
        ("#1a2034", "#a3b0d5"),
        ("#291a31", "#caa1de"),
    ),
)


_PALETTES = {THEME_LIGHT: _LIGHT, THEME_DARK: _DARK}

# Текущая палитра — доступна как Palette (см. внизу), обновляется при apply_theme.
_current_name: str = THEME_LIGHT


def _qss(p: PaletteSpec) -> str:
    return f"""
* {{
    font-family: "Segoe UI", "Inter", "DejaVu Sans", Arial, sans-serif;
    font-size: 13px;
    color: {p.text};
}}

QMainWindow, QWidget#RootSurface {{
    background: {p.bg};
}}

QLabel {{ background: transparent; }}

QLabel#AppTitle {{
    font-size: 22px;
    font-weight: 700;
    color: {p.text};
}}
QLabel#AppSubtitle {{
    color: {p.text_muted};
    font-size: 13px;
}}
QLabel#SectionTitle {{
    font-size: 15px;
    font-weight: 600;
    color: {p.text};
}}
QLabel#SectionHint {{
    color: {p.text_muted};
    font-size: 12px;
}}
QLabel#StatusOk {{ color: {p.success}; font-weight: 600; }}
QLabel#StatusWarn {{ color: {p.warning}; font-weight: 600; }}

/* Карточка-секция: применяется к QFrame с objectName='Card' */
QFrame#Card {{
    background: {p.surface};
    border: 1px solid {p.border};
    border-radius: 10px;
}}

/* Кнопки */
QPushButton {{
    background: {p.surface};
    color: {p.text};
    border: 1px solid {p.border_strong};
    border-radius: 6px;
    padding: 7px 14px;
}}
QPushButton:hover {{
    border-color: {p.primary};
    color: {p.primary};
}}
QPushButton:pressed {{
    background: {p.primary_soft};
}}
QPushButton:disabled {{
    color: {p.text_subtle};
    border-color: {p.border};
    background: {p.surface_alt};
}}

/* Primary button — обязательно с явным контрастным фоном и цветом текста. */
QPushButton#PrimaryButton, QPushButton[primary="true"] {{
    background-color: {p.primary};
    color: {p.primary_text};
    border: 1px solid {p.primary};
    padding: 9px 18px;
    font-weight: 600;
}}
QPushButton#PrimaryButton:hover, QPushButton[primary="true"]:hover {{
    background-color: {p.primary_hover};
    border-color: {p.primary_hover};
    color: {p.primary_text};
}}
QPushButton#PrimaryButton:pressed, QPushButton[primary="true"]:pressed {{
    background-color: {p.primary_pressed};
    border-color: {p.primary_pressed};
    color: {p.primary_text};
}}
QPushButton#PrimaryButton:disabled, QPushButton[primary="true"]:disabled {{
    background-color: {p.border};
    border-color: {p.border};
    color: {p.text_subtle};
}}

QPushButton#GhostButton {{
    background: transparent;
    border: 1px solid {p.border_strong};
    color: {p.text_muted};
}}
QPushButton#GhostButton:hover {{
    color: {p.primary};
    border-color: {p.primary};
}}

/* Переключатель темы — компактная иконка-кнопка в шапке */
QPushButton#ThemeToggle {{
    background: {p.surface};
    border: 1px solid {p.border_strong};
    border-radius: 14px;
    padding: 5px 12px;
    color: {p.text_muted};
    font-size: 12px;
    font-weight: 600;
}}
QPushButton#ThemeToggle:hover {{
    color: {p.primary};
    border-color: {p.primary};
}}

/* Поля ввода / комбобоксы */
QComboBox, QLineEdit {{
    background: {p.surface};
    border: 1px solid {p.border_strong};
    border-radius: 6px;
    padding: 5px 8px;
    color: {p.text};
    min-height: 22px;
}}
QComboBox:focus, QLineEdit:focus {{ border-color: {p.primary}; }}
QComboBox::drop-down {{ border: none; width: 22px; }}
QComboBox QAbstractItemView {{
    background: {p.surface};
    border: 1px solid {p.border_strong};
    selection-background-color: {p.primary_soft};
    selection-color: {p.text};
    padding: 4px;
    color: {p.text};
}}

/* Таблицы */
QTableView, QTableWidget {{
    background: {p.surface};
    border: 1px solid {p.border};
    border-radius: 8px;
    gridline-color: {p.border};
    color: {p.text};
    selection-background-color: {p.primary_soft};
    selection-color: {p.text};
    alternate-background-color: {p.surface_alt};
}}
QHeaderView::section {{
    background: {p.surface_alt};
    color: {p.text_muted};
    padding: 8px 10px;
    border: none;
    border-right: 1px solid {p.border};
    border-bottom: 1px solid {p.border};
    font-weight: 600;
}}
QHeaderView::section:last {{ border-right: none; }}
QTableView::item, QTableWidget::item {{ padding: 6px 8px; }}
QTableCornerButton::section {{
    background: {p.surface_alt};
    border: none;
    border-bottom: 1px solid {p.border};
    border-right: 1px solid {p.border};
}}

/* Вкладки */
QTabWidget::pane {{
    border: 1px solid {p.border};
    border-radius: 10px;
    background: {p.surface};
    top: -1px;
}}
QTabBar::tab {{
    background: transparent;
    color: {p.text_muted};
    padding: 9px 18px;
    margin-right: 4px;
    border: 1px solid transparent;
    border-bottom: none;
    border-top-left-radius: 8px;
    border-top-right-radius: 8px;
    font-weight: 500;
}}
QTabBar::tab:hover {{ color: {p.primary}; }}
QTabBar::tab:selected {{
    background: {p.surface};
    color: {p.primary};
    border: 1px solid {p.border};
    border-bottom-color: {p.surface};
    font-weight: 600;
}}

/* Список графиков (sidebar дашборда) */
QListWidget#ChartList {{
    background: {p.surface_alt};
    border: 1px solid {p.border};
    border-radius: 8px;
    padding: 6px;
    outline: none;
    color: {p.text};
}}
QListWidget#ChartList::item {{
    padding: 9px 12px;
    border-radius: 6px;
    color: {p.text};
    margin-bottom: 2px;
}}
QListWidget#ChartList::item:hover {{
    background: {p.primary_soft};
    color: {p.primary};
}}
QListWidget#ChartList::item:selected {{
    background: {p.primary};
    color: {p.primary_text};
    font-weight: 600;
}}

/* Скроллбары */
QScrollBar:vertical {{ background: transparent; width: 10px; margin: 4px; }}
QScrollBar::handle:vertical {{
    background: {p.border_strong};
    border-radius: 5px;
    min-height: 30px;
}}
QScrollBar::handle:vertical:hover {{ background: {p.text_subtle}; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0px; }}
QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 4px; }}
QScrollBar::handle:horizontal {{
    background: {p.border_strong};
    border-radius: 5px;
    min-width: 30px;
}}

/* GroupBox — карточка */
QGroupBox {{
    border: 1px solid {p.border};
    border-radius: 10px;
    margin-top: 14px;
    padding: 14px 14px 12px 14px;
    background: {p.surface};
    font-weight: 600;
    color: {p.text};
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 14px;
    top: -2px;
    padding: 0 6px;
    color: {p.primary};
    background: {p.surface};
}}

/* QTextEdit / QPlainTextEdit (рекомендации) */
QTextEdit, QPlainTextEdit {{
    background: {p.surface};
    border: 1px solid {p.border};
    border-radius: 8px;
    padding: 12px 14px;
    color: {p.text};
}}

/* KPI-карточки */
QFrame#KpiCard {{
    background: {p.surface};
    border: 1px solid {p.border};
    border-radius: 10px;
}}
QLabel#KpiLabel {{
    color: {p.text_muted};
    font-size: 12px;
    font-weight: 500;
}}
QLabel#KpiValue {{
    font-size: 22px;
    font-weight: 700;
    color: {p.text};
}}

/* Бейджи */
QLabel#Badge {{
    background: {p.primary_soft};
    color: {p.primary};
    border-radius: 10px;
    padding: 3px 10px;
    font-weight: 600;
    font-size: 11px;
}}
QLabel#BadgeWarn {{
    background: {p.surface_alt};
    color: {p.warning};
    border-radius: 10px;
    padding: 3px 10px;
    font-weight: 600;
    font-size: 11px;
}}

QMenu {{
    background: {p.surface};
    color: {p.text};
    border: 1px solid {p.border};
}}
QMenu::item:selected {{ background: {p.primary_soft}; color: {p.primary}; }}

QToolTip {{
    background: {p.surface_alt};
    color: {p.text};
    border: 1px solid {p.border_strong};
    padding: 4px 8px;
}}
"""


def _qpalette(p: PaletteSpec) -> QPalette:
    pal = QPalette()
    pal.setColor(QPalette.ColorRole.Window, QColor(p.bg))
    pal.setColor(QPalette.ColorRole.WindowText, QColor(p.text))
    pal.setColor(QPalette.ColorRole.Base, QColor(p.surface))
    pal.setColor(QPalette.ColorRole.AlternateBase, QColor(p.surface_alt))
    pal.setColor(QPalette.ColorRole.Text, QColor(p.text))
    pal.setColor(QPalette.ColorRole.PlaceholderText, QColor(p.text_subtle))
    pal.setColor(QPalette.ColorRole.Button, QColor(p.surface))
    pal.setColor(QPalette.ColorRole.ButtonText, QColor(p.text))
    pal.setColor(QPalette.ColorRole.ToolTipBase, QColor(p.surface_alt))
    pal.setColor(QPalette.ColorRole.ToolTipText, QColor(p.text))
    pal.setColor(QPalette.ColorRole.Highlight, QColor(p.primary))
    pal.setColor(QPalette.ColorRole.HighlightedText, QColor(p.primary_text))
    pal.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.WindowText, QColor(p.text_subtle))
    pal.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text, QColor(p.text_subtle))
    pal.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText, QColor(p.text_subtle))
    return pal


def get_palette(name: str | None = None) -> PaletteSpec:
    if name is None:
        return _PALETTES[_current_name]
    return _PALETTES.get(name, _LIGHT)


def current_theme_name() -> str:
    return _current_name


def _settings() -> QSettings:
    return QSettings("MarketingDSS", "App")


def load_saved_theme() -> str:
    val = _settings().value("ui/theme", THEME_LIGHT)
    if isinstance(val, str) and val in _PALETTES:
        return val
    return THEME_LIGHT


def save_theme(name: str) -> None:
    if name in _PALETTES:
        _settings().setValue("ui/theme", name)


def apply_theme(app, name: str | None = None) -> str:
    """Применяет тему ко всему приложению. Возвращает имя применённой темы."""
    global _current_name
    if name is None:
        name = load_saved_theme()
    if name not in _PALETTES:
        name = THEME_LIGHT
    _current_name = name
    p = _PALETTES[name]

    app.setStyle("Fusion")
    app.setPalette(_qpalette(p))
    app.setStyleSheet(_qss(p))
    # Перекрасить уже созданные виджеты, если есть
    for w in app.allWidgets():
        w.style().unpolish(w)
        w.style().polish(w)
    return name


def toggle_theme(app) -> str:
    """Переключает между светлой и тёмной темой, сохраняет выбор. Возвращает новое имя."""
    new = THEME_DARK if _current_name == THEME_LIGHT else THEME_LIGHT
    apply_theme(app, new)
    save_theme(new)
    return new


# Совместимость со старым кодом: `from app.theme import Palette` всё ещё работает,
# но теперь это прокси-объект, который читает поля из текущей PaletteSpec.
class _PaletteProxy:
    def __getattr__(self, item):
        return getattr(get_palette(), item)


Palette = _PaletteProxy()
