import cv2 as cv
import mediapipe as mp

mp_draw = mp.solutions.drawing_utils # Function to Draw Landmarks over Hand
mp_hand = mp.solutions.hands # Hand Detection Function

# Capturing the Video from the Camera
video = cv.VideoCapture(0)

# Initializing the Hand Detection Function
hands = mp_hand.Hands(min_detection_confidence = 0.5, min_tracking_confidence = 0.5)

while True:
    success, image = video.read()

    # Converting the Image to RGB
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

    # Processing the Image for Hand Detection
    image.flags.writeable = False
    results = hands.process(image)
    image.flags.writeable = True

    # Converting the Image back to BGR
    image = cv.cvtColor(image, cv.COLOR_RGB2BGR)

    # List to store the Landmark's Coordinates
    landmarks_list = []

    # If Landmarks Detected i.e., Hand Detected Sucessfully
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[-1]

        for index, lm in enumerate(hand_landmarks.landmark):
            h, w, c = image.shape # Height, Width, Channels
            cx, cy = int(lm.x*w), int(lm.y*h)
            landmarks_list.append([index, cx, cy])

        # Drawing the Landmarks for only One Hand
        # Landmarks will be drawn for the Hand which was Detected First
        mp_draw.draw_landmarks(image, hand_landmarks, mp_hand.HAND_CONNECTIONS)

    # If Hand Detected
    if results.multi_hand_landmarks != None:
        if landmarks_list[2][1] > landmarks_list[17][1]:
            cv.rectangle(image, (20, 335), (200, 425), (0, 255, 0), cv.FILLED)
            cv.putText(image, "Right", (30, 395), cv.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 5)
        else:
            cv.rectangle(image, (20, 335), (200, 425), (0, 255, 0), cv.FILLED)
            cv.putText(image, "Left", (30, 395), cv.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 5)

    # Show the Video
    cv.imshow("Frame", image)
    
    # Close the Video if "q" key is pressed
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv.destroyAllWindows()