import os

# Qt renders the side panel offscreen in tests; no display or window server needed.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
