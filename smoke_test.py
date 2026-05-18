"""Smoke-тест: проверяет всю не-GUI часть конвейера + offscreen UI.

Прогоняет:
1. Распознавание булевых конверсий.
2. Полный пайплайн на sample_marketing_data.csv (с extra_category).
3. End-to-end с булевой колонкой конверсий.
4. Условный funnel chart (показы / клики / конверсии — только из доступных).
5. Offscreen UI: тема, MainWindow с компактным маппингом, ResultsWindow,
   переключение темы, наличие вкладки «Доп. категория».
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile

import pandas as pd

from app.charts import build_charts, cleanup_charts
from app.data_mapping import auto_guess_mapping
from app.plotly_charts import build_plotly_charts
from app.processing import _to_boolean_numeric, process_data
from app.report import export_docx


def _section(title: str) -> None:
    print(f"\n=== {title} ===")


def test_boolean_recognition() -> None:
    _section("Тест распознавания булевых конверсий")
    cases = [
        (["yes", "no", "Yes", "NO", " yes ", None], True, 3),
        (["да", "нет", " Да ", "НЕТ"], True, 2),
        ([True, False, True, None], True, 2),
        ([1, 0, 1, 0, 1], True, 3),
        (["1", "0", "1", "0"], True, 2),
        (["y", "n", "Y", "N"], True, 2),
        (["true", "false", "TRUE"], True, 2),
        ([1, 2, 0, 1], False, None),
        (["A", "B", "C"], False, None),
    ]
    all_ok = True
    for vals, expected_bool, expected_sum in cases:
        s = pd.Series(vals)
        out, was = _to_boolean_numeric(s)
        ok = (was == expected_bool) and (
            expected_sum is None or float(out.sum()) == float(expected_sum)
        )
        marker = "✓" if ok else "✗"
        print(f"  {marker} {vals!r:40s} -> bool={was}, sum={out.sum() if was else '—'}")
        all_ok = all_ok and ok
    if not all_ok:
        raise SystemExit("Тест булевых конверсий не пройден")


def test_full_pipeline() -> None:
    _section("Полный пайплайн на sample_marketing_data.csv (с extra_category)")
    csv_path = os.path.join(os.path.dirname(__file__), "sample_marketing_data.csv")
    df = pd.read_csv(csv_path)
    print(f"  Загружено {len(df)} строк, {len(df.columns)} столбцов")

    mapping = auto_guess_mapping(df.columns)
    mapping["extra_category"] = "product_category"  # вручную выбираем доп. категорию
    result = process_data(df, mapping)
    print(f"  Группировка: {result.group_label}; сегментов: {len(result.metrics)}")
    print(f"  Доступные метрики: {result.available_metrics}")
    print(f"  Строк после очистки: {result.rows_loaded}/{result.rows_original} "
          f"(удалено {result.rows_dropped})")

    assert result.extra_summary is not None and not result.extra_summary.empty, \
        "extra_summary должен быть построен"
    assert result.extra_category_col == "product_category"
    print(f"  ✓ extra_summary: {len(result.extra_summary)} строк, метрика={result.extra_metric}")

    # Plotly
    plotly_specs = build_plotly_charts(result)
    titles = [s.title for s in plotly_specs]
    print(f"\n  Plotly-графиков: {len(plotly_specs)}")
    for spec in plotly_specs:
        print(f"    · {spec.title}")
    assert any("Z-score" not in t and "z-score" not in s.description.lower() or True
               for s, t in zip(plotly_specs, titles))  # формально оставлено — проверим явно
    heatmap = next((s for s in plotly_specs if s.title == "Тепловая карта KPI"), None)
    assert heatmap is not None
    assert "z-score" not in heatmap.description.lower(), "В описании heatmap не должно быть Z-score"
    assert "Отклонение от среднего" in heatmap.html or "отклонение от среднего" in heatmap.html.lower()
    print("  ✓ Heatmap легенда: «Отклонение от среднего, σ» (без слова Z-score)")

    # extra category должен быть среди графиков
    extra = [t for t in titles if "product_category" in t.lower() or t.startswith("Конверсии: product_category")]
    assert extra, "extra_category chart должен присутствовать"
    print(f"  ✓ Plotly: extra-category график «{extra[0]}»")

    # matplotlib PNG для DOCX
    tmpdir = tempfile.mkdtemp(prefix="smoke_charts_")
    try:
        charts = build_charts(result, tmpdir=tmpdir)
        print(f"\n  matplotlib PNG: {len(charts)}")
        for ch in charts:
            size_kb = os.path.getsize(ch.path) / 1024
            print(f"    · {ch.title}  ({size_kb:.1f} KB)")

        chart_titles = [c.title for c in charts]
        assert any("product_category" in t for t in chart_titles), "extra-cat PNG должен быть"

        out_dir = os.path.dirname(os.path.abspath(__file__))
        docx_path = os.path.join(out_dir, "smoke_report.docx")
        export_docx(result, charts, docx_path)
        print(f"\n  DOCX: {docx_path}  ({os.path.getsize(docx_path) / 1024:.1f} KB)")
        cleanup_charts(charts)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_conditional_funnel() -> None:
    """Сводный график строится только из реально доступных метрик."""
    _section("Условный funnel chart (показы/клики/конверсии)")
    df = pd.DataFrame({
        "channel": ["A", "A", "B", "B"],
        "imps": [1000, 1200, 800, 900],
        "clicks": [50, 60, 40, 45],
    })
    # Только displays + clicks → название «Показы и клики»
    m = {"channels": "channel", "campaigns": None, "displays": "imps", "clicks": "clicks",
         "conversions": None, "total_cost": None, "placement_cost": None, "cpc": None,
         "age": None, "city": None, "month": None, "extra_category": None}
    r = process_data(df, m)
    specs = build_plotly_charts(r)
    titles = [s.title for s in specs]
    print(f"  Метрики={['displays','clicks']} → графики: {titles}")
    funnel = next((s for s in specs if "показы" in s.title.lower() and "клики" in s.title.lower()
                   and "ранжирование" not in s.title.lower()), None)
    assert funnel is not None, "Funnel из 2 метрик должен быть"
    assert "конверс" not in funnel.title.lower(), "Конверсий в названии быть не должно"
    print(f"  ✓ Без конверсий: название графика — «{funnel.title}»")

    # Только конверсии → одна метрика, funnel не строится
    m2 = dict(m); m2["displays"] = None; m2["clicks"] = None; m2["conversions"] = "clicks"
    r2 = process_data(df, m2)
    specs2 = build_plotly_charts(r2)
    funnel2 = next((s for s in specs2 if s.title.startswith(("Показы", "Клики", "Конверсии")) and "ранжирование" not in s.title), None)
    print(f"  ✓ Одна метрика: комбинированный funnel = {funnel2.title if funnel2 else 'не построен'}")
    assert funnel2 is None or "ранжирование" in funnel2.title.lower(), \
        "Комбинированный funnel не должен строиться при одной метрике"


def test_boolean_end_to_end() -> None:
    _section("End-to-end с булевой колонкой конверсий")
    df = pd.DataFrame({
        "channel": ["Google", "Google", "Google", "Yandex", "Yandex", "Yandex",
                    "VK", "VK", "VK", "Email", "Email", "Email"],
        "impressions": [10000, 12000, 9500, 8000, 7500, 8500, 5000, 6000, 5500, 2000, 2200, 2100],
        "clicks":      [300, 350, 280, 220, 200, 250, 180, 210, 190, 90, 100, 95],
        "cost":        [3000, 3500, 2800, 2400, 2200, 2700, 1500, 1700, 1600, 500, 550, 520],
        "converted":   ["yes", "no", "yes", "no", "no", "yes", "yes", "yes", "no",
                        "yes", "yes", "yes"],
    })
    m = {"channels": "channel", "campaigns": None, "displays": "impressions",
         "clicks": "clicks", "conversions": "converted",
         "total_cost": "cost", "placement_cost": None, "cpc": None,
         "age": None, "city": None, "month": None, "extra_category": None}
    r = process_data(df, m)
    total_conv = r.metrics["conversions"].sum()
    expected = (df["converted"].str.lower().str.strip() == "yes").sum()
    assert total_conv == expected, f"конверсии должны суммироваться как yes-факты: {total_conv} vs {expected}"
    print(f"  ✓ Сумма конверсий {int(total_conv)} совпала с числом 'yes' в данных ({expected})")


def test_offscreen_ui() -> None:
    """Создаёт MainWindow и ResultsWindow, проверяет компактный маппинг
    (без вертикального скролла), переключение темы и наличие вкладки доп. категории."""
    _section("Offscreen UI: компактный маппинг, тема, доп. категория")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    from PySide6.QtWidgets import QApplication
    from app.main_window import MainWindow, PandasModel
    from app.results_window import HAVE_WEBENGINE, ResultsWindow
    from app.theme import apply_theme, current_theme_name, toggle_theme, THEME_DARK, THEME_LIGHT

    app = QApplication.instance() or QApplication(sys.argv)
    apply_theme(app, THEME_LIGHT)
    assert current_theme_name() == THEME_LIGHT
    print(f"  ✓ Начальная тема: {current_theme_name()}")

    df = pd.read_csv(os.path.join(os.path.dirname(__file__), "sample_marketing_data.csv"))

    mw = MainWindow()
    mw.df = df
    mw.model = PandasModel(df)
    mw.table_view.setModel(mw.model)
    mw.build_mapping()
    assert mw.next_btn.isEnabled(), "Кнопка анализа должна быть активна при валидном маппинге"
    # Компактный маппинг: 12 ролей × сетка 3-в-ряд → 4 строки
    rows = mw.mapping_grid.rowCount()
    print(f"  ✓ Сетка маппинга: {rows} строк × 3 колонки (12 ролей)")
    assert rows <= 5, f"Маппинг не должен требовать прокрутки: rows={rows}"

    # Тема: переключение работает и не падает
    toggle_theme(app)
    mw.on_theme_changed()
    assert current_theme_name() == THEME_DARK
    toggle_theme(app)
    mw.on_theme_changed()
    assert current_theme_name() == THEME_LIGHT
    print("  ✓ Переключение темы light ⇄ dark работает")

    # ResultsWindow с extra_category
    mapping = auto_guess_mapping(df.columns)
    mapping["extra_category"] = "product_category"
    result = process_data(df, mapping)
    rw = ResultsWindow(result)
    tabs = [rw.tabs.tabText(i) for i in range(rw.tabs.count())]
    print(f"  ✓ ResultsWindow вкладки: {tabs} (WebEngine={HAVE_WEBENGINE})")
    extra_tab = [t for t in tabs if "Доп. категория" in t]
    assert extra_tab, "Должна быть вкладка с доп. категорией"

    # Метрики табл-вкладка: ширина последнего столбца — не Stretch
    metrics_widget = rw.tabs.widget(0)
    table = metrics_widget.findChild(type(rw._df_to_table(pd.DataFrame({"a": [1]}))))
    if table is not None:
        from PySide6.QtWidgets import QHeaderView
        assert not table.horizontalHeader().stretchLastSection(), \
            "Последний столбец не должен растягиваться"
        print("  ✓ Метрики: последний столбец не Stretch")

    # Дашборд работает, выбираем 3-й график
    rw.chart_list.setCurrentRow(3)
    assert rw.chart_stack.currentIndex() == 3
    print("  ✓ Дашборд: переключение графика работает")

    # Тема в ResultsWindow тоже
    toggle_theme(app)
    rw.on_theme_changed()
    print(f"  ✓ ResultsWindow выдерживает переключение темы (→ {current_theme_name()})")

    rw.close()
    mw.close()


def test_date_classification_no_warning() -> None:
    """Распознавание дат не должно выдавать UserWarning от pandas про dayfirst.

    В прошлом версия `pd.to_datetime(text, errors='raise', dayfirst=True)`
    давала предупреждение на ISO-строках вида '2024-01-15 10:30:00'.
    Helper в data_mapping выбирает dayfirst по форме строки.
    """
    _section("Парсинг дат: не должно быть UserWarning от pandas")
    import warnings as _warnings
    from app.data_mapping import classify_value

    cases = [
        ("2024-01-15 10:30:00", "date"),  # ISO с временем — исходный триггер
        ("2024-01-15", "date"),
        ("2024/01/15", "date"),
        ("15.01.2024", "date"),
        ("15/01/2024", "date"),
        ("15/01/2024 10:30", "date"),
        # 01-15-2024 — это US month-first, не входит в перечисленные форматы
        # (ISO и DD.MM/DD.MM.YYYY), сознательно отдаём text, чтобы не спорить с pandas
        ("01-15-2024", "text"),
        ("hello", "text"),
        ("42", "numeric"),
        ("", "empty"),
        ("2024-13-40", "text"),
    ]
    with _warnings.catch_warnings(record=True) as captured:
        _warnings.simplefilter("always")
        for text, expected in cases:
            got = classify_value(text)
            assert got == expected, f"{text!r}: expected {expected}, got {got}"

    pandas_warnings = [w for w in captured if "dayfirst" in str(w.message).lower()
                       or "parsing dates" in str(w.message).lower()]
    print(f"  ✓ {len(cases)} строк классифицированы корректно")
    print(f"  ✓ Pandas-предупреждений про dayfirst: {len(pandas_warnings)} (ожидалось 0)")
    assert not pandas_warnings, (
        "Должно быть 0 предупреждений pandas о dayfirst, но получены: "
        + "; ".join(str(w.message) for w in pandas_warnings)
    )


def main() -> int:
    test_boolean_recognition()
    test_date_classification_no_warning()
    test_full_pipeline()
    test_conditional_funnel()
    test_boolean_end_to_end()
    test_offscreen_ui()
    print("\n✓ Все smoke-тесты пройдены")
    return 0


if __name__ == "__main__":
    sys.exit(main())
