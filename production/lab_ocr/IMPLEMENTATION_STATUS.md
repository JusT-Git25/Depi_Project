# Implementation status

Changes in 0.2.0: Streamlit removed (`app.py` deleted, dependency dropped). Added a FastAPI backend with a static web page (`lab_ocr/api.py`, `lab_ocr/web/`), a command line (`python -m lab_ocr`), and a prediction layer (`lab_ocr/prediction/`) that runs a trained model on user-confirmed values.

Verified here: full test suite passes (OCR mocked), plus a live run with real Tesseract on a synthetic image: upload -> extract -> predict through the running server and through the CLI, using the synthetic DEMO model.

Not verified: your real model (feature names, units, order), accuracy on real lab reports, the browser UI in an actual browser (JS syntax checked, API tested), Windows setup, deployment security (no auth in this package).

The demo model is trained on made-up data and has no medical meaning.
