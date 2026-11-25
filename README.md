# NBA Cup Predictor

Dự đoán nhà vô địch Emirates NBA Cup 2026-27 bằng mô hình logistic regression ở cấp độ trận đấu kết hợp mô phỏng Monte Carlo. Dữ liệu lấy từ NBA Stats API qua thư viện [`nba_api`](https://github.com/swar/nba_api).

## Phương pháp

Mỗi mùa giải chỉ có một nhà vô địch, quá ít mẫu để mô hình hoá trực tiếp. Thay vào đó, mô hình ước lượng xác suất một đội thắng đội kia trong từng trận, được huấn luyện trên khoảng 7.000 trận mùa giải chính thức từ 2020-21. Các xác suất này được dùng để mô phỏng phần còn lại của giải (vòng bảng, tứ kết, bán kết, chung kết) 20.000 lần. Xác suất vô địch của một đội là tỉ lệ số lần đội đó vô địch trong các lần mô phỏng.

## Mô hình

Mỗi trận được mô tả bằng chênh lệch giữa hai đội trên các đặc trưng sau, chỉ dùng thông tin có trước trận đấu:

| Đặc trưng | Mô tả |
|---|---|
| `d_margin` | Hiệu số điểm trung bình mỗi trận |
| `d_winpct` | Tỉ lệ thắng |
| `d_last10` | Hiệu số điểm trung bình 10 trận gần nhất |
| `d_rest` | Số ngày nghỉ |
| `LOC` | Địa điểm: 1 sân nhà, 0 sân trung lập, −1 sân khách |

Chỉ số đầu mùa nhiều nhiễu, nên giá trị mùa hiện tại được kéo về phía mùa trước:

```
blended = (n · mùa_này + K · mùa_trước) / (n + K),  K = 12
```

Bộ phân loại là `LogisticRegression(fit_intercept=False)` trên dữ liệu được chuẩn hoá không trừ trung bình. Mỗi trận xuất hiện hai lần, một lần từ góc nhìn của mỗi đội với dấu đặc trưng đảo ngược, nên mô hình đối xứng: P(A thắng B) = 1 − P(B thắng A). Cách này cũng cho phép dự đoán các trận ở sân trung lập.

## Mô phỏng giải đấu

- 30 đội chia thành 6 bảng, mỗi bảng 5 đội (3 bảng mỗi miền); mỗi đội đá 4 trận vòng bảng.
- Mỗi miền có 3 đội nhất bảng (hạt giống 1–3) và 1 đội wild card (hạt giống 4) vào tứ kết.
- Tứ kết: 1 gặp 4, 2 gặp 3, trên sân của hạt giống cao hơn. Bán kết và chung kết đá ở sân trung lập tại Las Vegas.
- Tiêu chí xếp hạng bảng: số trận thắng, đối đầu, hiệu số, ngẫu nhiên.
- Cách biệt điểm mỗi trận được lấy mẫu từ N(μ, 12.5), với μ chọn sao cho P(cách biệt > 0) bằng xác suất thắng của mô hình. Cách biệt cần cho tiêu chí hiệu số.

## Kết quả

Kiểm định ngoài mẫu: huấn luyện trên 2021-22 đến 2024-25, kiểm tra trên 2025-26 (1.225 trận):

| Chỉ số | Mô hình | Baseline (đội chủ nhà thắng) |
|---|---|---|
| Accuracy | 0.688 | 0.554 |
| Log loss | 0.597 | — |
| Brier score | 0.205 | — |

Xác suất vô địch tính đến ngày 14/11/2026, khi đã đá 28/60 trận vòng bảng:

| Đội | Bảng | Thành tích | Vào tứ kết | Vô địch |
|---|---|---|---|---|
| BOS | East C | 2-0 | 96% | 26% |
| SAS | West C | 2-0 | 87% | 24% |
| DET | East A | 1-0 | 82% | 14% |
| MIN | West B | 2-0 | 71% | 12% |
| OKC | West B | 1-1 | 43% | 4% |

Xác suất thay đổi khi có kết quả mới. Chạy `predict.py` để có dự đoán mới nhất.

## Cách sử dụng

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python predict.py
```

Tuỳ chọn:

```
--season    Mùa giải cần dự đoán (mặc định: 2026-27)
--sims      Số lần mô phỏng (mặc định: 20000)
--history   Số mùa dữ liệu huấn luyện (mặc định: 6)
--refresh   Tải lại dữ liệu các mùa cũ đã lưu
```

Kết quả được ghi vào `output/cup_prediction_<season>.csv`.

## Cấu trúc

```
nbacup/
  data.py       Lấy và lưu dữ liệu NBA API, dựng lịch vòng bảng
  features.py   Đặc trưng của đội trước mỗi trận
  model.py      Logistic regression, kiểm định, xác suất từng cặp đội
  simulate.py   Mô phỏng Monte Carlo giải đấu
predict.py      Chạy toàn bộ pipeline
```

## Lộ trình

| Giai đoạn | Trạng thái |
|---|---|
| Thu thập dữ liệu từ NBA Stats API | Hoàn thành |
| Xây dựng đặc trưng trước trận | Hoàn thành |
| Mô hình logistic regression | Hoàn thành |
| Kiểm định ngoài mẫu | Hoàn thành |
| Mô phỏng giải đấu | Hoàn thành |
| Bổ sung đặc trưng: offensive/defensive rating, pace, chấn thương, back-to-back, Elo | Dự kiến |
| Dashboard web cập nhật hằng ngày | Dự kiến |
| Chạy tự động hằng ngày bằng GitHub Actions | Dự kiến |
| Đánh giá sau giải so với kết quả thực tế | Dự kiến |

## Hạn chế

- Chưa mô hình hoá chấn thương và thay đổi đội hình giữa mùa.
- Một số trận lịch sử đá ở sân trung lập (Paris, Mexico City, Las Vegas) vẫn được tính là sân nhà.
- stats.nba.com đôi khi giới hạn hoặc phản hồi chậm; chạy lại nếu tải dữ liệu thất bại.
