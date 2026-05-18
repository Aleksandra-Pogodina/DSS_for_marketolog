"""Точка входа. Запуск: python main.py"""

import sys

from PySide6.QtWidgets import QApplication

from app.main_window import MainWindow
from app.theme import apply_theme


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Маркетинговая аналитика")
    print("RUNNING TEST-XYZ")
    apply_theme(app)
    window = MainWindow()
    window.showMaximized()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
