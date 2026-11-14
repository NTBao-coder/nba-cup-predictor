# 🏀 NBA Cup Predictor

Sản phẩm AI dự đoán **đội vô địch NBA Cup (Emirates NBA Cup) mùa 2026-27** bằng
**Logistic Regression**, dữ liệu lấy trực tiếp từ NBA Stats API (thư viện [`nba_api`](https://github.com/swar/nba_api)).

> Ý tưởng: Logistic Regression **không** đoán thẳng nhà vô địch (mỗi năm chỉ có 1 mẫu).
> Thay vào đó mô hình học **xác suất thắng của từng trận đấu** từ ~7.000 trận lịch sử,
> rồi **mô phỏng Monte Carlo** toàn bộ giải (vòng bảng còn lại → tứ kết → bán kết → chung kết)
> hàng chục nghìn lần để ra xác suất vô địch của mỗi đội.

---

## 📋 Kế hoạch dự án

| Giai đoạn | Nội dung | Trạng thái |
|---|---|---|
| 1. Thu thập dữ liệu | Lấy kết quả mùa giải 2020-21 → nay (`LeagueGameFinder`), bảng xếp hạng & lịch NBA Cup (`ISTStandings`). Cache ra CSV | ✅ Xong |
| 2. Tạo feature | Chỉ số của mỗi đội **trước** từng trận (không rò rỉ dữ liệu), kết hợp với mùa trước | ✅ Xong |
| 3. Huấn luyện mô hình | Logistic Regression đối xứng, có biến sân nhà / sân trung lập | ✅ Xong |
| 4. Kiểm định | Train các mùa cũ, test trên mùa 2025-26 (accuracy, log-loss, Brier) | ✅ Xong |
| 5. Mô phỏng giải | Monte Carlo đúng thể thức NBA Cup + luật tiebreak | ✅ Xong |
| 6. Cải thiện feature | Offensive/Defensive rating, pace, chấn thương, back-to-back, Elo | ⏳ Kế hoạch |
| 7. Giao diện | Dashboard web (Streamlit) hiển thị xác suất, cập nhật mỗi ngày | ⏳ Kế hoạch |
| 8. Tự động hoá | GitHub Actions chạy dự đoán hằng ngày trong mùa NBA Cup | ⏳ Kế hoạch |
| 9. Đánh giá sau giải | So sánh dự đoán với kết quả thực tế của NBA Cup 2026 | ⏳ Kế hoạch |

---

## 🧠 Phương pháp

### Feature (hiệu số giữa đội A và đội B)

| Feature | Ý nghĩa |
|---|---|
| `d_margin` | Chênh lệch hiệu số điểm trung bình/trận |
| `d_winpct` | Chênh lệch tỉ lệ thắng |
| `d_last10` | Chênh lệch phong độ 10 trận gần nhất |
| `d_rest` | Chênh lệch số ngày nghỉ |
| `LOC` | `1` = A đá sân nhà, `0` = sân trung lập (Las Vegas), `-1` = sân khách |

Đầu mùa chỉ số rất nhiễu, nên chỉ số mùa hiện tại được "co" về chỉ số mùa trước:

```
blended = (n · mùa_này + K · mùa_trước) / (n + K),   K = 12
```

### Mô hình

- `StandardScaler(with_mean=False)` + `LogisticRegression(fit_intercept=False)`
- Mỗi trận được đưa vào **2 lần** (góc nhìn chủ nhà và góc nhìn khách, đổi dấu feature),
  nên mô hình đối xứng: `P(A thắng B) = 1 − P(B thắng A)`. Nhờ vậy dự đoán được cả trận sân trung lập.

### Mô phỏng NBA Cup

- 6 bảng × 5 đội; nhất bảng (hạt giống 1-3) + 1 wild card (hạt giống 4) mỗi miền vào tứ kết.
- Tứ kết: 1 vs 4, 2 vs 3 trên sân của hạt giống cao hơn. Bán kết và chung kết đá ở sân trung lập.
- Xếp hạng bảng: số trận thắng → đối đầu → hiệu số → ngẫu nhiên.
- Cách biệt điểm mỗi trận mô phỏng được lấy mẫu từ `N(μ, 12.5)`, với μ chọn sao cho khớp xác suất
  thắng của mô hình (cần cho tiebreak theo hiệu số).

---

## 📊 Kết quả hiện tại (14/11/2026)

Kiểm định trên mùa 2025-26 (1.225 trận):

| Chỉ số | Mô hình | Baseline "chủ nhà luôn thắng" |
|---|---|---|
| Accuracy | **0.688** | 0.554 |
| Log-loss | 0.597 | — |
| Brier score | 0.205 | — |

Top ứng viên vô địch (vòng bảng: 28/60 trận đã đá):

| Đội | Bảng | Thành tích | Vào tứ kết | Vô địch |
|---|---|---|---|---|
| BOS | East C | 2-0 | 96% | **~26%** |
| SAS | West C | 2-0 | 87% | ~24% |
| DET | East A | 1-0 | 82% | ~14% |
| MIN | West B | 2-0 | 71% | ~12% |
| OKC | West B | 1-1 | 43% | ~4% |

*(Xác suất thay đổi mỗi ngày khi có kết quả mới. Chạy lại `predict.py` để cập nhật.)*

---

## 🚀 Cách chạy

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python predict.py                 # mặc định: mùa 2026-27, 20.000 lần mô phỏng
python predict.py --sims 50000    # mô phỏng nhiều hơn
python predict.py --refresh       # tải lại cả dữ liệu các mùa cũ
```

Kết quả được lưu ở `output/cup_prediction_2026-27.csv`.

## 📁 Cấu trúc

```
nbacup/
  data.py       # gọi NBA API, cache CSV, lấy lịch vòng bảng NBA Cup
  features.py   # tạo feature không rò rỉ dữ liệu
  model.py      # Logistic Regression, kiểm định, ma trận xác suất
  simulate.py   # mô phỏng Monte Carlo NBA Cup
predict.py      # chạy toàn bộ pipeline
```

## ⚠️ Hạn chế

- Chưa tính chấn thương và thay đổi đội hình giữa mùa.
- Vài trận lịch sử đá ở sân trung lập (Paris, Mexico City, Las Vegas) vẫn được tính là sân nhà.
- stats.nba.com đôi khi chặn hoặc phản hồi chậm; khi đó hãy chạy lại sau.
