# CloudLabeller — photogrammetric reconstruction and bidirectional 2D <-> 3D
# point-cloud labelling with U-Net label propagation.
# Copyright (C) 2026 Ítalo Gomes Gonçalves
#
# This program is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program. If not, see <https://www.gnu.org/licenses/>.
#
# Commercial licensing: this program is also available under a separate
# commercial license from the author — see README.md.

"""Application entry point: build the QApplication and show the main window."""

from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    """Launch the CloudLabeller desktop application."""
    # Imported lazily so that ``import cloudlabeller`` stays Qt-free and cheap.
    import logging

    from PySide6.QtWidgets import QApplication

    from cloudlabeller import __version__
    from cloudlabeller.config import AppConfig
    from cloudlabeller.logging_setup import DEFAULT_LOG_DIR, install_excepthook, setup_logging
    from cloudlabeller.ui.main_window import MainWindow

    logfile = setup_logging(DEFAULT_LOG_DIR / "cloudlabeller.log")
    install_excepthook()
    logging.getLogger("cloudlabeller").info("CloudLabeller %s starting (log: %s)",
                                            __version__, logfile)

    argv = list(sys.argv if argv is None else argv)

    # Give the process its own taskbar identity — otherwise Windows groups the
    # app under python.exe and shows the Python icon instead of ours.
    if sys.platform == "win32":
        import ctypes

        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("CloudLabeller.App")

    app = QApplication(argv)
    app.setApplicationName("CloudLabeller")
    app.setOrganizationName("CloudLabeller")

    # App icon (window title bars + taskbar), incl. the welcome dialog.
    from pathlib import Path

    from PySide6.QtGui import QIcon

    icon_path = Path(__file__).parent / "assets" / "icon.ico"
    if icon_path.exists():
        app.setWindowIcon(QIcon(str(icon_path)))

    config = AppConfig.load()

    # A project passed on the command line skips the welcome dialog.
    if len(argv) > 1:
        window = MainWindow(config=config)
        window.restore_layout()
        window.show()
        window.open_project(argv[1])
        return app.exec()

    # The main screen is useless without a project, so the welcome dialog runs
    # FIRST and only accepts once a project was created/opened (Quit rejects).
    from cloudlabeller.ui.welcome_dialog import WelcomeDialog

    welcome = WelcomeDialog(config.recent_projects)
    if welcome.exec() != WelcomeDialog.Accepted or welcome.project is None:
        return 0

    window = MainWindow(config=config)
    window.restore_layout()
    window.show()
    window.adopt_project(welcome.project)
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
