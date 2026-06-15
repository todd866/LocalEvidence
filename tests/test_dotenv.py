import os
from localevidence import config


def test_load_dotenv_sets_missing_and_respects_existing(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text('# a comment\nLE_TEST_A=alpha\nLE_TEST_B="quoted"\nMALFORMED_LINE\nLE_TEST_C=fromfile\n')
    monkeypatch.delenv("LE_TEST_A", raising=False)
    monkeypatch.delenv("LE_TEST_B", raising=False)
    monkeypatch.setenv("LE_TEST_C", "exported-wins")  # a real env var must win
    config.load_dotenv(env)
    assert os.environ["LE_TEST_A"] == "alpha"
    assert os.environ["LE_TEST_B"] == "quoted"          # surrounding quotes stripped
    assert os.environ["LE_TEST_C"] == "exported-wins"   # setdefault did NOT override
    assert "MALFORMED_LINE" not in os.environ


def test_load_dotenv_missing_file_is_noop(tmp_path):
    config.load_dotenv(tmp_path / "does-not-exist.env")  # must not raise
