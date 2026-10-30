import sys
from backend.tests import test_api

tests = [
    ("test_health_check", test_api.test_health_check),
    ("test_model_info", test_api.test_model_info),
    ("test_predict_real_news", test_api.test_predict_real_news),
    ("test_predict_fake_news", test_api.test_predict_fake_news),
    ("test_predict_short_text_validation", test_api.test_predict_short_text_validation),
    ("test_prediction_history", test_api.test_prediction_history),
    ("test_metrics_endpoint", test_api.test_metrics_endpoint),
    ("test_upload_dataset_valid", test_api.test_upload_dataset_valid),
    ("test_upload_dataset_invalid_format", test_api.test_upload_dataset_invalid_format),
    ("test_retrain_unauthorized", test_api.test_retrain_unauthorized),
    ("test_retrain_authorized_status", test_api.test_retrain_authorized_status),
    ("test_monitoring_endpoint", test_api.test_monitoring_endpoint)
]

passed = 0
failed = 0
for name, func in tests:
    try:
        func()
        print(f"[PASS] {name}", flush=True)
        passed += 1
    except Exception as e:
        print(f"[FAIL] {name}: {e}", flush=True)
        failed += 1

print(f"\nFINAL TEST RESULT: {passed} passed, {failed} failed.", flush=True)
if failed > 0:
    sys.exit(1)
