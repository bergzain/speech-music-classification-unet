from dataclasses import dataclass, field
from typing import Dict, Any
from pathlib import Path

def get_model_registry() -> Dict[str, Any]:
    return {
        "U_Net": {
            "name": "unet_classifier",
            "stages": ["Development", "Staging", "Production"]
        },
        "R2U_Net": {
            "name": "r2unet_classifier",
            "stages": ["Development", "Staging", "Production"]
        },
        "R2AttU_Net": {
            "name": "r2attunet_classifier",
            "stages": ["Development", "Staging", "Production"]
        },
        "AttentionUNet": {
            "name": "attentionunet_classifier",
            "stages": ["Development", "Staging", "Production"]
        }
    }

def get_metrics_config() -> Dict[str, Dict[str, Any]]:
    return {
        "accuracy": {"higher_is_better": True, "threshold": 0.90},
        "loss": {"higher_is_better": False, "threshold": 0.1},
        "f1_score": {"higher_is_better": True, "threshold": 0.85}
    }

@dataclass
class MLflowConfig:
    TRACKING_URI: str = str(Path.home() / "mlflow")
    EXPERIMENT_BASE_NAME: str = "speech_music_classification"
    ARTIFACT_PATH: str = "models"
    MODEL_REGISTRY: Dict[str, Any] = field(default_factory=get_model_registry)
    METRICS_CONFIG: Dict[str, Dict[str, Any]] = field(default_factory=get_metrics_config)

    @classmethod
    def get_experiment_name(cls, model_type: str, feature_type: str) -> str:
        return f"{cls.EXPERIMENT_BASE_NAME}/{model_type}/{feature_type}"