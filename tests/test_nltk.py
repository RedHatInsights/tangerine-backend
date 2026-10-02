import os
import stat

import pytest

import tangerine.nltk as tangerine_nltk


def _raise_lookup_error(_):
    raise LookupError


@pytest.mark.parametrize(
    ("umask", "expected_mode"),
    ((0o000, 0o755), (0o077, 0o700)),
)
def test_init_nltk_creates_data_directory_with_mode_capped_at_755(
    tmp_path, monkeypatch, umask, expected_mode
):
    data_dir = tmp_path / "nltk_data"
    monkeypatch.setattr(tangerine_nltk.cfg, "NLTK_DATA_DIR", str(data_dir))

    def download(*args, **kwargs):
        assert data_dir.is_dir()
        return True

    monkeypatch.setattr(tangerine_nltk, "find", _raise_lookup_error)
    monkeypatch.setattr(tangerine_nltk.nltk, "download", download)

    previous_umask = os.umask(umask)
    try:
        tangerine_nltk.init_nltk()
    finally:
        os.umask(previous_umask)

    assert stat.S_IMODE(data_dir.stat().st_mode) == expected_mode


def test_init_nltk_preserves_existing_data_directory_permissions(tmp_path, monkeypatch):
    data_dir = tmp_path / "nltk_data"
    data_dir.mkdir(mode=0o750)
    data_dir.chmod(0o750)
    monkeypatch.setattr(tangerine_nltk.cfg, "NLTK_DATA_DIR", str(data_dir))
    monkeypatch.setattr(tangerine_nltk, "find", _raise_lookup_error)
    monkeypatch.setattr(tangerine_nltk.nltk, "download", lambda *args, **kwargs: True)

    tangerine_nltk.init_nltk()
    assert stat.S_IMODE(data_dir.stat().st_mode) == 0o750


def test_init_nltk_does_not_create_directory_when_corpus_is_found(tmp_path, monkeypatch):
    data_dir = tmp_path / "nltk_data"
    monkeypatch.setattr(tangerine_nltk.cfg, "NLTK_DATA_DIR", str(data_dir))
    monkeypatch.setattr(tangerine_nltk, "find", lambda _: "corpora/words")
    monkeypatch.setattr(
        tangerine_nltk.nltk,
        "download",
        lambda *args, **kwargs: pytest.fail("download should not be called"),
    )

    tangerine_nltk.init_nltk()
    assert not data_dir.exists()
