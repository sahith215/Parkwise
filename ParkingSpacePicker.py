import cv2
import pickle
import os

# --------- Configuration Parameters ---------
IMAGE_PATH = 'InputImg.png'
DATA_FILE = 'slot_positions'
UI_WINDOW = "Draw Parking Zones"

# --------- Load Existing Zones or Start Fresh ---------
if os.path.isfile(DATA_FILE):
    with open(DATA_FILE, 'rb') as file:
        parking_zones = pickle.load(file)
else:
    parking_zones = []

drawing_mode = False
init_point = (-1, -1)
live_preview_box = None  # For temporary drawing

# --------- Mouse Event Handler ---------
def mouse_draw_callback(event, x, y, flags, param):
    global drawing_mode, init_point, live_preview_box, parking_zones

    # Start drawing on left-click
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing_mode = True
        init_point = (x, y)
        live_preview_box = None

    # Show live preview while dragging
    elif event == cv2.EVENT_MOUSEMOVE and drawing_mode:
        live_preview_box = (init_point[0], init_point[1], x, y)

    # Finalize rectangle on left-button release
    elif event == cv2.EVENT_LBUTTONUP:
        drawing_mode = False
        x_start, y_start = init_point
        x_end, y_end = x, y

        top_left_x = min(x_start, x_end)
        top_left_y = min(y_start, y_end)
        width = abs(x_end - x_start)
        height = abs(y_end - y_start)

        # Ignore very small areas
        if width > 10 and height > 10:
            parking_zones.append((top_left_x, top_left_y, width, height))
            with open(DATA_FILE, 'wb') as file:
                pickle.dump(parking_zones, file)
            print(f"✅ Zone #{len(parking_zones)} saved at ({top_left_x}, {top_left_y}, {width}, {height})")
        
        live_preview_box = None

    # Right-click to remove a marked zone
    elif event == cv2.EVENT_RBUTTONDOWN:
        for idx, (px, py, pw, ph) in enumerate(parking_zones):
            if px <= x <= px + pw and py <= y <= py + ph:
                print(f"🗑️ Removed zone #{idx + 1}")
                parking_zones.pop(idx)
                with open(DATA_FILE, 'wb') as file:
                    pickle.dump(parking_zones, file)
                break

# --------- Application Entry Point ---------
def run_parking_marker():
    global live_preview_box, parking_zones

    cv2.namedWindow(UI_WINDOW)
    cv2.setMouseCallback(UI_WINDOW, mouse_draw_callback)

    while True:
        img = cv2.imread(IMAGE_PATH)
        if img is None:
            print("❌ Could not load image. Check if 'InputImg.png' exists.")
            break

        # Draw saved zones
        for idx, (x, y, w, h) in enumerate(parking_zones):
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)
            cv2.putText(img, f"{idx + 1}", (x + 5, y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Show preview rectangle while drawing
        if live_preview_box:
            x0, y0, x1, y1 = live_preview_box
            cv2.rectangle(img, (x0, y0), (x1, y1), (0, 255, 0), 2)

        cv2.imshow(UI_WINDOW, img)
        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # ESC to quit
            break
        elif key == ord('r'):  # Reset all zones
            parking_zones = []
            live_preview_box = None
            if os.path.exists(DATA_FILE):
                os.remove(DATA_FILE)
            print("🔄 All zones cleared.")
        elif key == ord('s'):  # Save manually
            with open(DATA_FILE, 'wb') as file:
                pickle.dump(parking_zones, file)
            print(f"💾 {len(parking_zones)} zone(s) saved.")

    cv2.destroyAllWindows()

# --------- Launch Program ---------
if __name__ == "__main__":
    run_parking_marker()
