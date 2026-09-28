import os

# Qt renders the side panel offscreen in tests; no display or window server needed.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest


@pytest.fixture
def qt_app():
    """The panel draws text with Qt, which needs an application object to exist."""
    from PySide6.QtGui import QGuiApplication

    return QGuiApplication.instance() or QGuiApplication([])
