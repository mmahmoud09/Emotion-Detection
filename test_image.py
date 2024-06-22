import pickle
import cv2
from utils import get_face_landmarks

emotions = ['HAPPY', 'SAD', 'angry']

with open('./model', 'rb') as f:
    model = pickle.load(f)


image_path = '2.jpg'
frame = cv2.imread(image_path)

face_landmarks = get_face_landmarks(frame, draw=True, static_image_mode=False)


output = model.predict([face_landmarks])


text = emotions[int(output[0])]
text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 3, 5)[0]
text_x = (frame.shape[1] - text_size[0])//3
text_y = 100 # Adjusted to position text at the top
cv2.putText(frame,
            text,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            3,
            (0, 255, 0),
            5)

# Display the image
cv2.imshow('image', frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
