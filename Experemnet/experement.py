import cv2
import torch
from segment_anything import sam_model_registry, SamPredictor

device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"Using device: {device}")


sam_checkpoint = "sam_vit_b_01ec64.pth" 
model_type = "vit_b" 

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device)
predictor = SamPredictor(sam)


video_path = 'bridge_video.mp4'
cap = cv2.VideoCapture(video_path)

start_time = 14
end_time = 32

def process_frame(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    predictor.set_image(frame_rgb)

    masks, _, _ = predictor.predict_torch()

    return masks

def count_people_in_frame(masks):
    person_count = 0
    for mask in masks:
        
        if is_person_shape(mask):
            person_count += 1
    return person_count


def detect_people_in_video(start_time, end_time):
    cap.set(cv2.CAP_PROP_POS_MSEC, start_time * 1000) 
    total_people_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000


        if current_time > end_time:
            break
        masks = process_frame(frame)
        person_count = count_people_in_frame(masks)
        total_people_count += person_count
        for mask in masks:
            if is_person_shape(mask):
                frame[mask > 0] = [0, 255, 0] 

        cv2.imshow('Segmented Frame', frame)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return total_people_count

def is_person_shape(mask):
    bbox = cv2.boundingRect(mask)
    width, height = bbox[2], bbox[3]
    aspect_ratio = width / height
    if 0.5 < aspect_ratio < 1.8: 
        return True
    return False

total_people_count = detect_people_in_video(start_time, end_time)
print(f"Estimated number of people on the bridge: {total_people_count}")