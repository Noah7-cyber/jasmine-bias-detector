from streamlit.testing.v1 import AppTest


def test_sample_to_analysis_to_results_flow():
    app=AppTest.from_file("app.py",default_timeout=30).run()
    assert not app.exception
    assert app.title[0].value=="AI Bias Detector"
    app.button[0].click().run()
    app.sidebar.radio[0].set_value("Analysis").run()
    assert not app.exception
    next(button for button in app.button if button.label=="Run analysis").click().run()
    assert not app.exception
    app.sidebar.radio[0].set_value("Results").run()
    assert not app.exception
    assert {metric.label for metric in app.metric} >= {"Rows analysed","DP difference","Impact ratio"}
    assert len(app.get("plotly_chart")) >= 1
    app.sidebar.radio[0].set_value("Analysis").run()
    next(widget for widget in app.number_input if widget.label=="DPD screen").set_value(0.20).run()
    assert not app.exception
    app.sidebar.radio[0].set_value("Results").run()
    assert not app.exception
    assert not app.metric
    assert any("Run an analysis first" in message.value for message in app.info)


def test_upload_screen_renders_with_pinned_streamlit():
    app = AppTest.from_file("app.py", default_timeout=30).run()
    next(widget for widget in app.radio if widget.label == "Data source").set_value("Upload a CSV").run()
    assert not app.exception
    assert len(app.get("file_uploader")) == 1
    assert any("25 MiB" in message.value for message in app.caption)


def test_continue_navigation_and_clear_dataset():
    app = AppTest.from_file("app.py", default_timeout=30).run()
    next(button for button in app.button if button.label == "Load sample").click().run()
    next(button for button in app.button if button.label == "Continue to analysis").click().run()
    assert app.sidebar.radio[0].value == "Analysis"
    assert not app.exception
    app.sidebar.radio[0].set_value("Dataset").run()
    next(button for button in app.button if button.label == "Clear current dataset and analysis").click().run()
    assert "audit_frame" not in app.session_state
    assert not app.exception
