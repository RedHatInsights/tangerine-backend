"""Regression tests for secret redaction in model-config logging."""

import logging

import tangerine.config as cfg


def test_redact_model_config_masks_api_key():
    raw = {
        "model": "mistral",
        "openai_api_base": "http://localhost:11434/v1",
        "openai_api_key": "sk-super-secret",
        "temperature": 0.7,
    }
    redacted = cfg.redact_model_config(raw)
    assert redacted["openai_api_key"] == "***"
    assert redacted["model"] == "mistral"
    assert redacted["openai_api_base"] == "http://localhost:11434/v1"
    # original must not be mutated
    assert raw["openai_api_key"] == "sk-super-secret"


def test_get_model_config_does_not_log_api_key(caplog):
    secret = "sk-leak-canary-12345"
    cfg.MODELS["_test_redact"] = {
        "model": "m",
        "openai_api_base": "http://localhost",
        "openai_api_key": secret,
        "temperature": 0.0,
    }
    try:
        with caplog.at_level(logging.INFO, logger="tangerine.config"):
            result = cfg.get_model_config("_test_redact")
        assert result["openai_api_key"] == secret  # caller still gets the real key
        for record in caplog.records:
            assert secret not in record.getMessage()
    finally:
        cfg.MODELS.pop("_test_redact", None)
