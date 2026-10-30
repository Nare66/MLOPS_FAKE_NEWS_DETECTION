import os
import json
import time
from typing import Dict, Any, Optional
from ml.drift_monitor import run_evidently_drift_monitoring
from backend.app.config import settings

class MonitoringService:
    @staticmethod
    def run_monitoring(current_csv: Optional[str] = None) -> Dict[str, Any]:
        os.makedirs(settings.REPORTS_DIR, exist_ok=True)
        timestamp = int(time.time())
        html_report = os.path.join(settings.REPORTS_DIR, f"evidently_drift_report_{timestamp}.html")
        json_report = os.path.join(settings.REPORTS_DIR, "latest_drift_metrics.json")
        
        ref_path = os.path.join(settings.REFERENCE_DIR, "reference_data.csv")
        
        result = run_evidently_drift_monitoring(
            reference_csv=ref_path,
            current_csv=current_csv,
            output_html_path=html_report,
            output_json_path=json_report
        )
        return result

    @staticmethod
    def get_latest_monitoring() -> Dict[str, Any]:
        json_path = os.path.join(settings.REPORTS_DIR, "latest_drift_metrics.json")
        if os.path.exists(json_path):
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        
        # If no report run yet, run one to initialize baseline
        return MonitoringService.run_monitoring()
