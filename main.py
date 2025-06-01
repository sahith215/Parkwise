import torch
import cv2
import pickle
import cvzone
import pandas as pd
import os

# --- Utility: Draw readable labels with transparency effect ---
def place_text_label(image, label, pos_x, pos_y, bg_color, size=0.5):
    (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, size, 1)
    overlay = image.copy()
    cv2.rectangle(overlay, (pos_x, pos_y - text_h - 10), (pos_x + text_w + 5, pos_y), bg_color, -1)
    cv2.addWeighted(overlay, 0.6, image, 0.4, 0, image)
    cv2.putText(image, label, (pos_x + 2, pos_y - 3), cv2.FONT_HERSHEY_SIMPLEX, size, (255, 255, 255), 1)

# --- Load YOLOv5 model (Ensure it's locally present) ---
try:
    detector = torch.hub.load('yolov5', 'yolov5s', source='local')
except Exception:
    print("❌ Failed to load YOLOv5. Please ensure it's cloned locally under 'yolov5'")
    exit()

# Configure detection settings
detector.conf = 0.4
detector.classes = [2, 3, 5, 7]  # Target: car, motorcycle, bus, truck

# --- Check if slot coordinates exist ---
if not os.path.exists('slot_positions'):
    print("❌ 'slot_positions' file is missing. Execute ParkingSpacePicker.py first.")
    exit()

with open('slot_positions', 'rb') as coord_file:
    predefined_slots = pickle.load(coord_file)

# --- Load the input image for processing ---
input_image = cv2.imread('InputImg.png')
if input_image is None:
    print("❌ 'InputImg.png' not found or corrupted.")
    exit()

# --- Perform object detection using YOLOv5 ---
prediction = detector(input_image)
boxes = prediction.xyxy[0]

# --- Capture bounding boxes of detected vehicles ---
vehicle_boxes = []
for *position, score, label in boxes:
    x_start, y_start, x_end, y_end = map(int, position)
    vehicle_boxes.append((x_start, y_start, x_end, y_end))
    cv2.rectangle(input_image, (x_start, y_start), (x_end, y_end), (255, 0, 0), 2)

# --- IoU Calculation Function ---
def calculate_overlap(rect1, rect2):
    x_left = max(rect1[0], rect2[0])
    y_top = max(rect1[1], rect2[1])
    x_right = min(rect1[2], rect2[2])
    y_bottom = min(rect1[3], rect2[3])

    inter_width = max(0, x_right - x_left)
    inter_height = max(0, y_bottom - y_top)
    inter_area = inter_width * inter_height

    area1 = (rect1[2] - rect1[0]) * (rect1[3] - rect1[1])
    area2 = (rect2[2] - rect2[0]) * (rect2[3] - rect2[1])
    total_area = area1 + area2 - inter_area

    return inter_area / total_area if total_area else 0

# --- Determine if a slot overlaps with any vehicle ---
def is_occupied(slot, vehicles, threshold=0.15):
    sx, sy, sw, sh = slot
    slot_rect = (sx, sy, sx + sw, sy + sh)

    for vx1, vy1, vx2, vy2 in vehicles:
        vehicle_rect = (vx1, vy1, vx2, vy2)
        if calculate_overlap(slot_rect, vehicle_rect) > threshold:
            return True

        # Fallback check: center point within slot
        cx = (vx1 + vx2) // 2
        cy = (vy1 + vy2) // 2
        if slot_rect[0] <= cx <= slot_rect[2] and slot_rect[1] <= cy <= slot_rect[3]:
            return True

    return False

# --- Analyze all parking slots and store results ---
slot_info = []
free_counter = 0

for idx, rect in enumerate(predefined_slots):
    status = "Occupied" if is_occupied(rect, vehicle_boxes) else "Free"
    color_code = (0, 0, 255) if status == "Occupied" else (0, 255, 0)
    if status == "Free":
        free_counter += 1

    x, y, w, h = rect
    cv2.rectangle(input_image, (x, y), (x + w, y + h), color_code, 2)
    place_text_label(input_image, f"Slot {idx + 1} - {status}", x, y, color_code)

    slot_info.append({'Slot': idx + 1, 'Status': status})

# --- Save status data to CSVs ---
status_df = pd.DataFrame(slot_info)
status_df.to_csv("ParkingStatus.csv", index=False)

summary_df = pd.DataFrame([{
    'Total Number of Slots': len(predefined_slots),
    'Occupied Slots': len(predefined_slots) - free_counter,
    'Available Slots': free_counter
}])
summary_df.to_csv("ParkingSummary.csv", index=False)

# --- Display summary overlay on image ---
summary_display = [
    "| Total Slots  | Occupied | Available |",
    f"| {len(predefined_slots):^11} | {len(predefined_slots) - free_counter:^8} | {free_counter:^9} |"
]

base_x, base_y = 40, 40
gap = 40
text_type = cv2.FONT_HERSHEY_SIMPLEX
text_size, text_thickness = 0.8, 2
text_fg, text_bg = (255, 255, 255), (0, 255, 0)

for line_num, line in enumerate(summary_display):
    pos_y = base_y + line_num * gap
    text_dim = cv2.getTextSize(line, text_type, text_size, text_thickness)[0]
    box_overlay = input_image.copy()
    cv2.rectangle(box_overlay, (base_x - 10, pos_y - 30), (base_x + text_dim[0] + 10, pos_y), text_bg, -1)
    cv2.addWeighted(box_overlay, 0.6, input_image, 0.4, 0, input_image)
    cv2.putText(input_image, line, (base_x, pos_y - 5), text_type, text_size, text_fg, text_thickness)

# --- Watermark at top-right ---
signature = "COMPUTER VISION PROJECT 2025 - Developed by P. Sahith, P. Jahnavi"
sig_font = cv2.FONT_HERSHEY_SIMPLEX
sig_scale = 0.6
sig_thickness = 1
sig_color = (255, 255, 255)

(sig_width, sig_height), _ = cv2.getTextSize(signature, sig_font, sig_scale, sig_thickness)
sig_x = input_image.shape[1] - sig_width - 20
sig_y = 30

watermark_layer = input_image.copy()
cv2.rectangle(watermark_layer, (sig_x - 10, sig_y - sig_height - 10), (sig_x + sig_width + 10, sig_y + 5), (0, 0, 0), -1)
cv2.addWeighted(watermark_layer, 0.4, input_image, 0.6, 0, input_image)
cv2.putText(input_image, signature, (sig_x, sig_y), sig_font, sig_scale, sig_color, sig_thickness, cv2.LINE_AA)

# --- Display output window ---
cv2.imshow("Parking Status", input_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
