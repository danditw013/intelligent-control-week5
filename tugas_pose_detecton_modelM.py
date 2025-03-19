from ultralytics import YOLO
import cv2
import numpy as np

# Load model YOLOv8 Medium Pose
model = YOLO("yolov8m-pose.pt")

# Inisialisasi kamera
cap = cv2.VideoCapture(0)

# Definisi titik wajah berdasarkan COCO keypoints
FACE_KEYPOINTS = [0, 1, 2, 3, 4]  # Kepala, mata kanan, mata kiri, telinga kanan, telinga kiri

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Deteksi pose
    results = model(frame)

    # Iterasi setiap hasil deteksi dalam satu frame
    for result in results:
        annotated_frame = result.plot()  # Anotasi default dari YOLO

        # Ambil keypoints (titik-titik pose)
        keypoints = result.keypoints.xy.cpu().numpy() if result.keypoints is not None else []

        for keypoint in keypoints:
            # Ambil hanya keypoint wajah
            face_points = keypoint[FACE_KEYPOINTS]

            # Gambar bounding box wajah
            x_min, y_min = np.min(face_points, axis=0)
            x_max, y_max = np.max(face_points, axis=0)

            # Pastikan nilai valid sebelum menggambar
            if x_min > 0 and y_min > 0 and x_max > 0 and y_max > 0:
                cv2.rectangle(annotated_frame, (int(x_min)-10, int(y_min)-10),
                              (int(x_max)+10, int(y_max)+10), (0, 0, 255), 2)

                # Tambahkan label "Wajah"
                cv2.putText(annotated_frame, "Wajah", (int(x_min), int(y_min)-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            # Gambar keypoint wajah lebih jelas
            for (x, y) in face_points:
                cv2.circle(annotated_frame, (int(x), int(y)), 5, (255, 0, 0), -1)

        cv2.imshow("YOLOv8 Pose Estimation (Detail Wajah)", annotated_frame)

    # Tekan 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Tutup kamera
cap.release()
cv2.destroyAllWindows()
