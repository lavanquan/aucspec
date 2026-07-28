from __future__ import annotations

try:
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING
except Exception:
    CONFIG_MAPPING = None
else:
    _original_register = CONFIG_MAPPING.register

    def _register_config_with_aimv2_compat(model_type, config, exist_ok=False):
        if model_type == "aimv2":
            exist_ok = True
        return _original_register(model_type, config, exist_ok=exist_ok)

    CONFIG_MAPPING.register = _register_config_with_aimv2_compat