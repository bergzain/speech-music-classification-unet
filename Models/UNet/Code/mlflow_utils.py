import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd
from typing import List, Dict, Optional
import torch


class MLflowExperimentManager:
    def __init__(self, tracking_uri: str):
        mlflow.set_tracking_uri(tracking_uri)
        self.client = MlflowClient()

    def create_nested_experiment(self, parent_name: str, child_name: str) -> str:
        """Create hierarchical experiments"""
        parent_exp = mlflow.get_experiment_by_name(parent_name)
        if not parent_exp:
            parent_id = mlflow.create_experiment(parent_name)
        else:
            parent_id = parent_exp.experiment_id
            
        return mlflow.create_experiment(f"{parent_name}/{child_name}")

    def get_best_run(self, experiment_name: str, metric: str = "val_accuracy") -> Optional[str]:
        """Get best performing run based on metric"""
        exp = mlflow.get_experiment_by_name(experiment_name)
        runs = self.client.search_runs(
            experiment_ids=[exp.experiment_id],
            order_by=[f"metrics.{metric} DESC"]
        )
        return runs[0].info.run_id if runs else None

    def register_best_model(self, run_id: str, model_name: str) -> None:
        """Register best model version"""
        result = self.client.create_model_version(
            name=model_name,
            source=f"runs:/{run_id}/model",
            run_id=run_id
        )
        self.client.transition_model_version_stage(
            name=model_name,
            version=result.version,
            stage="Production"
        )

    def compare_runs(self, experiment_name: str, metrics: List[str]) -> pd.DataFrame:
        """Compare runs within an experiment"""
        exp = mlflow.get_experiment_by_name(experiment_name)
        runs = self.client.search_runs([exp.experiment_id])
        
        results = []
        for run in runs:
            result = {
                "run_id": run.info.run_id,
                "status": run.info.status,
                **{k: v for k, v in run.data.params.items()},
                **{f"metric.{k}": v for k, v in run.data.metrics.items() if k in metrics}
            }
            results.append(result)
        
        return pd.DataFrame(results)

    def log_system_metrics(self, run_id: str) -> None:
        """Log system performance metrics"""
        import psutil
        import GPUtil

        with mlflow.start_run(run_id=run_id):
            mlflow.log_metrics({
                "cpu_percent": psutil.cpu_percent(),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_usage_percent": psutil.disk_usage('/').percent,
                **({"gpu_utilization": GPUtil.getGPUs()[0].load * 100} 
                   if GPUtil.getGPUs() else {})
            })

    def log_model_gradients(self, model: torch.nn.Module, step: int) -> None:
        """Log model gradient statistics"""
        grad_norms = {
            f"grad_norm.{name}": param.grad.norm().item()
            for name, param in model.named_parameters()
            if param.grad is not None
        }
        mlflow.log_metrics(grad_norms, step=step)