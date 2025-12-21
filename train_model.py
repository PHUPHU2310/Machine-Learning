import pandas as pd
import cv2
import numpy as np
from skimage.feature import hog
from sklearn.neighbors import KNeighborsClassifier
import joblib
import os
from tqdm import tqdm

TRAIN_CSV = "Train.csv"
TEST_CSV = "Test.csv"
META_CSV = "Meta.csv"
IMG_DIR = "."

# ==== Load CSV ====
train_df = pd.read_csv(TRAIN_CSV, encoding="utf-8-sig")
test_df = pd.read_csv(TEST_CSV, encoding="utf-8-sig")
meta_df = pd.read_csv(META_CSV, encoding="utf-8-sig")

train_df.columns = train_df.columns.str.strip()
test_df.columns = test_df.columns.str.strip()
meta_df.columns = meta_df.columns.str.strip()

df = pd.concat([train_df, test_df], ignore_index=True)
df = df.merge(meta_df[["ClassId", "ShapeId", "ColorId"]], on="ClassId", how="left")

print("Tổng số ảnh:", len(df))
print("Ví dụ Path:", df["Path"].head().tolist())

# ==== Trích xuất HOG ====
def extract_hog(img_path, image_size=(64, 64)):
    full_path = os.path.join(IMG_DIR, img_path)
    if not os.path.exists(full_path):
        print("❌ Không tìm thấy:", full_path)
        return None
    img = cv2.imread(full_path)
    if img is None:
        print("❌ Không đọc được ảnh:", full_path)
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, image_size)
    features = hog(img,
                   orientations=9,
                   pixels_per_cell=(8, 8),
                   cells_per_block=(2, 2),
                   block_norm='L2-Hys')
    return features

# ==== Tạo dataset ====
X = []
y_class, y_shape, y_color = [], [], []

for _, row in tqdm(df.iterrows(), total=len(df)):
    feat = extract_hog(row["Path"])
    if feat is not None:
        X.append(feat)
        y_class.append(row["ClassId"])
        y_shape.append(row["ShapeId"])
        y_color.append(row["ColorId"])

X = np.array(X)
print("Số lượng mẫu:", len(X))
print("Kích thước X:", X.shape)

# ==== Train ====
if len(X) > 0:
    knn_class = KNeighborsClassifier(n_neighbors=5)
    knn_shape = KNeighborsClassifier(n_neighbors=5)
    knn_color = KNeighborsClassifier(n_neighbors=5)

    knn_class.fit(X, y_class)
    knn_shape.fit(X, y_shape)
    knn_color.fit(X, y_color)

    # ==== Lưu mô hình có nén ====
    joblib.dump(knn_class, "knn_class_model.joblib", compress=3)
    joblib.dump(knn_shape, "knn_shape_model.joblib", compress=3)
    joblib.dump(knn_color, "knn_color_model.joblib", compress=3)

    print("✅ Train xong, đã lưu 3 mô hình (có nén, file nhẹ hơn)")
else:
    print("❌ Không có dữ liệu hợp lệ để train!")
