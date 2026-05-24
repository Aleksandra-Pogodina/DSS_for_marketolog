"""Описание ролей столбцов и вспомогательная классификация значений."""

from __future__ import annotations

import re

import pandas as pd


ROLE_NAMES = [
    "channels",
    "campaigns",
    "displays",
    "clicks",
    "conversions",
    "total_cost",
    "placement_cost",
    "clicks_cost",
    "age",
    "revenue",
    "month",
    "extra_category",
]

ROLE_LABELS_RU = {
    "channels": "Каналы",
    "campaigns": "Кампании",
    "displays": "Показы",
    "clicks": "Клики",
    "conversions": "Конверсии",
    "total_cost": "Общая стоимость",
    "placement_cost": "Стоимость размещения",
    "clicks_cost": "Стоимость за клики",
    "age": "Возраст",
    "revenue": "Выручка",
    "month": "Месяц",
    "extra_category": "Доп. категория",
}

NUMERIC_ROLES = {
    "displays",
    "clicks",
    "conversions",
    "total_cost",
    "placement_cost",
    "clicks_cost",
    "age",
    "revenue",
}

GROUPING_ROLES = ("channels", "campaigns")


# Признак ISO/американского порядка «год-месяц-день» (с опциональным временем).
# Сначала ровно 4-значный год, потом разделитель и две группы по 1–2 цифры.
_ISO_DATE_RE = re.compile(
    r"^\d{4}[-/.]\d{1,2}[-/.]\d{1,2}([ T]\d{1,2}:\d{2}(:\d{2})?)?$"
)

# Признак «день-месяц-год»: 1–2 цифры, разделитель, ещё одна группа 1–2,
# затем 2 или 4 цифры года (с опциональным временем).
_DAYFIRST_DATE_RE = re.compile(
    r"^\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}([ T]\d{1,2}:\d{2}(:\d{2})?)?$"
)


def _try_parse_date(text: str) -> bool:
    """Возвращает True, если строка распознаётся как дата.

    Чтобы pandas не выдавал UserWarning о несовпадении dayfirst и формата:
    - сначала пробуем явный ISO-шаблон (год впереди) с dayfirst=False,
    - и только если строка похожа на день-месяц-год — с dayfirst=True.
    Прочие неоднозначные строки в дату не превращаем.
    """
    if _ISO_DATE_RE.match(text):
        try:
            pd.to_datetime(text, errors="raise", dayfirst=False)
            return True
        except (ValueError, TypeError, OverflowError):
            return False

    m = _DAYFIRST_DATE_RE.match(text)
    if m:
        # Проверяем, что первая часть действительно может быть днём (1..31),
        # вторая — месяцем (1..12). Иначе pandas с dayfirst=True всё равно
        # переинтерпретирует и выдаст UserWarning — лучше сразу сказать «не дата».
        parts = re.split(r"[-/.]", text.split(" ")[0].split("T")[0])
        try:
            day, month = int(parts[0]), int(parts[1])
        except (ValueError, IndexError):
            return False
        if not (1 <= day <= 31 and 1 <= month <= 12):
            return False
        try:
            pd.to_datetime(text, errors="raise", dayfirst=True)
            return True
        except (ValueError, TypeError, OverflowError):
            return False

    return False


def classify_value(value) -> str:
    """Грубая классификация одиночного значения для подсветки подозрительных ячеек."""
    if value is None:
        return "empty"

    if isinstance(value, float) and pd.isna(value):
        return "empty"

    text = str(value).strip()
    if text == "":
        return "empty"

    try:
        float(text.replace(",", ".").replace(" ", ""))
        return "numeric"
    except ValueError:
        pass

    if _try_parse_date(text):
        return "date"

    return "text"


def auto_guess_mapping(columns) -> dict:
    """Эвристическая попытка угадать роли столбцов по их именам.

    Возвращает словарь role -> column_name (или None). Используется для
    предзаполнения комбобоксов в UI; пользователь может всё переопределить.
    """

    hints = {
        "channels": ["channel", "канал", "источник", "source", "placement"],
        "campaigns": ["campaign", "кампан"],
        "displays": ["impression", "display", "показ", "shows"],
        "clicks": ["click", "клик"],
        "conversions": ["conversion", "конверс", "order", "заказ"],
        "total_cost": ["total_cost", "totalcost", "общая", "spend", "затрат", "расход"],
        "placement_cost": ["placement", "размещен", "cost"],
        "clicks_cost": ["clicks_cost", "click_cost", "click_costs", "стоимость за клики", "затраты на клики"],
        #"age": ["age", "возраст"],
        "revenue": ["revenue", "sales_amount", "sales", "transaction_value", "value", "выруч", "доход"],
        "month": ["month", "месяц", "date", "дата"],
    }

    mapping = {role: None for role in ROLE_NAMES}
    used = set()

    for role, keywords in hints.items():
        for col in columns:
            if col in used:
                continue
            low = str(col).lower()
            if any(kw in low for kw in keywords):
                mapping[role] = str(col)
                used.add(col)
                break

    return mapping
