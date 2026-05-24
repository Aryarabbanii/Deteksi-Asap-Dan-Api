from ultralytics import YOLO
import cv2

# Load model
model = YOLO("best.pt")  # pastikan file best.pt ada di direktori yang sama

# Load image
img = cv2.imread("asap.jpg")

# Run inference
results = model(img)

# Tampilkan hasil
annotated_frame = results[0].plot()
cv2.imshow("Deteksi Asap dan Api", annotated_frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
