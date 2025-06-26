import cv2
import numpy as np
from util import get_parking_spots_bboxes, empty_or_not

def calc_diff(im1, im2):
    return np.abs(np.mean(im1) - np.mean(im2))

mask_path = r"C:\Users\altay\Desktop\Master's Degree\Bitirme Tezi Kodlar\otoparkalgi\parking-space-counter-master\mask_1920_1080.png"
video_path = r"C:\Users\altay\Desktop\Master's Degree\Bitirme Tezi Kodlar\otoparkalgi\parking-space-counter-master\parking_1920_1080_loop.mp4"
database_file = r"C:\Users\altay\Desktop\Master's Degree\Bitirme Tezi Kodlar\otoparkalgi\parking-space-counter-master\empty_slots.txt"

def save_parking_data(empty_slots):
    with open(database_file, 'w') as f:
        f.write("Empty Spots:\n")
        written_slots = set()   
        for slot in empty_slots:
            if slot not in written_slots:  
                f.write(f"{slot}\n")   
                written_slots.add(slot)   

def load_parking_data():
    try:
        with open(database_file, 'r') as f:
            return f.readlines()
    except FileNotFoundError:
        return []
    
mask = cv2.imread(mask_path, 0)
if mask is None:
    print(f"Maske dosyası okunamadı: {mask_path}")
    exit()

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"Video dosyası açılmadı: {video_path}")
    exit()

connected_components = cv2.connectedComponentsWithStats(mask, 4, cv2.CV_32S)
spots = sorted(get_parking_spots_bboxes(connected_components), key=lambda x: (x[1], x[0])) 

spots_status = [None for j in spots]
diffs = [None for j in spots]
selected_spot = None  

visible_spots = 20   
scroll_index = 0     

def empty_spots_click(event, x, y, flags, param):
    global selected_spot, empty_spots_list, scroll_index
    if event == cv2.EVENT_LBUTTONDOWN:
        index = (y - 30) // 20 + scroll_index
        if 0 <= index < len(empty_spots_list):
            selected_spot = empty_spots_list[index] - 1  

previous_frame = None
frame_nmr = 0
ret = True
step = 30

while ret:
    ret, frame = cap.read()

    if not ret or frame is None:
        print("Video okuma hatası, döngü sonlandırılıyor.")
        break

    if frame_nmr % step == 0 and previous_frame is not None:
        for spot_indx, spot in enumerate(spots):
            x1, y1, w, h = spot
            spot_crop = frame[y1:y1 + h, x1:x1 + w, :]
            diffs[spot_indx] = calc_diff(spot_crop, previous_frame[y1:y1 + h, x1:x1 + w, :])

    if frame_nmr % step == 0:
        if previous_frame is None:
            arr_ = range(len(spots))
        else:
            arr_ = [j for j in np.argsort(diffs) if diffs[j] / np.amax(diffs) > 0.4]

        for spot_indx in arr_:
            spot = spots[spot_indx]
            x1, y1, w, h = spot
            spot_crop = frame[y1:y1 + h, x1:x1 + w, :]
            spot_status = empty_or_not(spot_crop)
            spots_status[spot_indx] = spot_status

    if frame_nmr % step == 0:
        previous_frame = frame.copy()

    for spot_indx, spot in enumerate(spots):
        spot_status = spots_status[spot_indx]
        x1, y1, w, h = spots[spot_indx]

        if selected_spot == spot_indx:
            frame = cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (255, 255, 255), 10)
        else:
            color = (0, 255, 0) if spot_status else (0, 0, 255)
            frame = cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), color, 2)
        
        cv2.putText(frame, f"{spot_indx + 1}", (x1 + 5, y1 + 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)

    cv2.rectangle(frame, (80, 20), (550, 80), (0, 0, 0), -1)
    cv2.putText(frame, 'Available spots: {} / {}'.format(str(sum(spots_status)), str(len(spots_status))), 
                (100, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    empty_spots_list = [idx + 1 for idx, status in enumerate(spots_status) if status]
    save_parking_data(empty_spots_list)  

    visible_empty_spots = empty_spots_list[scroll_index:scroll_index + visible_spots]
    empty_spots_window = np.zeros((max(400, 40 + 20 * visible_spots), 300), dtype=np.uint8) 
    cv2.putText(empty_spots_window, "Empty Spots:", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    for i, spot_id in enumerate(visible_empty_spots):
        cv2.putText(empty_spots_window, f"Spot {spot_id}", (10, 30 + (i + 1) * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.namedWindow('Empty Spots')
    cv2.setMouseCallback('Empty Spots', empty_spots_click)
    cv2.imshow('Empty Spots', empty_spots_window)

    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
    cv2.imshow('frame', frame)

    key = cv2.waitKey(25) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('w') and scroll_index > 0:
        scroll_index -= 1  
    elif key == ord('s') and scroll_index + visible_spots < len(empty_spots_list):
        scroll_index += 1  

    frame_nmr += 1

cap.release()
cv2.destroyAllWindows()
