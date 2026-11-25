# Forecast Service

Dịch vụ dự báo nhiệt độ trung bình ngày, dùng dữ liệu
[Jena Climate](https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip)
(2009–2016). Model: `Ridge` trên target delta (`y - lag_1`), dự báo đệ quy nhiều ngày.

## Kết quả trên holdout 2016

| Model | Test MAE | Test RMSE |
|---|---|---|
| Baseline persistence (`y_hat = y_{t-1}`) | 1.787 | 2.350 |
| **Ridge + delta (hiện tại)** | **1.681** | **2.165** |

## Cấu trúc

```
.
├── app.py                  # FastAPI: /health, /forecast
├── train.py                # Tải dữ liệu, đánh giá, refit full data, lưu model
├── src/
│   ├── data.py             # Download + cache, làm sạch, time_split
│   ├── features.py         # Lag + mùa vụ, feature cho inference
│   └── forecast.py         # Forecaster: load bundle, dự báo đệ quy
├── experiments/
│   ├── experiment.py       # So sánh cấu hình model trên holdout
│   └── predict_demo.py     # Demo dự báo từ model đã lưu
├── tests/
│   ├── test_pipeline.py    # Test edge cases + chống leakage
│   └── test_api.py         # Test /health và /forecast
├── .github/workflows/ci.yml  # CI: pytest trên Python 3.10 & 3.12
├── requirements.txt
└── Dockerfile
```

## Cài đặt

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Yêu cầu Python ≥ 3.10 (đã test trên 3.12).

## Huấn luyện

```bash
python train.py
```

Script sẽ:

1. Tải dataset (cache tại `data/`, không tải lại lần sau).
2. Đánh giá trên validation (90 ngày cuối train) và test (365 ngày cuối), kèm baseline persistence.
3. **Refit trên toàn bộ dữ liệu** rồi lưu `models/model.joblib` gồm `model`, `features`, `target_mode`, `trained_until`.

## Thử nghiệm model

```bash
# Cấu hình mặc định (Ridge + delta, 8 feature)
python experiments/experiment.py

# So sánh với HGB học trực tiếp y
python experiments/experiment.py --model hgb --target absolute

# Thêm rolling mean/std vào feature
python experiments/experiment.py --rolling 7 14 28

# Đổi alpha / horizon test
python experiments/experiment.py --alpha 10 --test-days 365
```

Output gồm MAE/RMSE trên holdout, so sánh với baseline persistence, và 5 dự báo mẫu.

Demo dự báo từ model đã lưu:

```bash
python experiments/predict_demo.py
python experiments/predict_demo.py --horizon 14
python experiments/predict_demo.py --as-of 2016-06-01 --horizon 7   # mô phỏng dự báo tại mốc thời gian
```

## Chạy API

```bash
uvicorn app:app --reload
```

- `GET /health` → `{"status": "ok", "model_loaded": true}`
- `GET /forecast?horizon=7` → dự báo 1–30 ngày kế tiếp

```bash
curl "http://127.0.0.1:8000/forecast?horizon=7"
```

Model và history được load **một lần** lúc startup (FastAPI `lifespan`), không load theo request.

## Test

```bash
pytest -q
```

Bộ test gồm `tests/test_pipeline.py` (feature, chia train/test, leakage, Forecaster)
và `tests/test_api.py` (dùng `fastapi.testclient`, tự tạo model tạm nên **không cần**
dataset hay `models/model.joblib`).

> Nếu môi trường có plugin pytest của ROS gây lỗi `No module named 'lark'`, chạy:
> `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q`

## Docker

```bash
python train.py                       # tạo models/model.joblib trước
docker build -t forecast-service .
docker run --rm -p 8000:8000 \
  -v "$PWD/models:/app/models:ro" \
  -v "$PWD/data:/app/data" \
  forecast-service
```

Image không chứa model; mount `models/` (và `data/` nếu muốn dùng cache) khi chạy.

## Ghi chú thiết kế

- **Không leakage**: mọi feature tại thời điểm `t` chỉ dùng dữ liệu `≤ t-1` (`shift(1)` trước `rolling`).
- **Target delta**: model học `y - lag_1`, dự báo cuối = `lag_1 + delta`. Cải thiện rõ so với học trực tiếp `y`.
- **Chia train/test theo mốc thời gian**, không shuffle; `time_split` validate `test_days > 0` và dữ liệu đủ dài.
- **Dự báo nhiều bước** dùng đệ quy: mỗi bước dự báo 1 ngày rồi nối vào history.
- **`trained_until`**: model production được refit trên toàn bộ dữ liệu nên bao phủ tới ngày cuối cùng.

## Dữ liệu & model

`data/` và `models/` nằm trong `.gitignore` (dataset ~13MB, model sinh ra khi train).
Sau khi clone, chạy `python train.py` để tạo `models/model.joblib`; dataset sẽ tự tải.

## License

MIT — xem [LICENSE](LICENSE).

