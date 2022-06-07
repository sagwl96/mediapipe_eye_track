import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

left_eye = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
right_eye = [33, 7, 163, 144, 153, 154, 133, 173, 157, 158, 159, 160, 161, 246]

left_iris = [474, 475, 476, 477]
right_iris = [469, 470, 471, 472]

# For image input:
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
image = cv2.imread("image_cal.jpg")
with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as face_mesh:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_h, img_w = image.shape[:2] #height and width of the image being taken from camera in pixels
    results = face_mesh.process(image)

    # Draw the face mesh annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_face_landmarks:
        #print(results.multi_face_landmarks[0].landmark)
        face_mesh_points = np.array([np.multiply([p.x, p.y], [img_w, img_h]).astype(int) for p in results.multi_face_landmarks[0].landmark])
        #print(face_mesh_points.shape)

        cv2.polylines(image, [face_mesh_points[left_eye]], True, (255, 0, 0), 1, cv2.LINE_AA) #drawing left eye
        cv2.polylines(image, [face_mesh_points[right_eye]], True, (255, 0, 0), 1, cv2.LINE_AA) #drawing right eye

        (l_cx, l_cy), l_radius = cv2.minEnclosingCircle(face_mesh_points[left_iris]) #calculating center of left iris
        center_left = np.array([l_cx, l_cy], dtype=np.int32) #converting it to integer

        (r_cx, r_cy), r_radius = cv2.minEnclosingCircle(face_mesh_points[right_iris]) #calculating center of right iris
        center_right = np.array([r_cx, r_cy], dtype=np.int32) #converting it to integer

        average_dia = l_radius+r_radius
        general_eye_dia = 11.77 # in cm
        focal_length = average_dia*40/general_eye_dia # focal length from a picture at a distance of 40cm from the camera

        file = open("calibration.txt", "w")
        file.write(str(focal_length))
        file.close()

        cv2.circle(image, center_left, int(l_radius), (0, 255, 0), 1, cv2.LINE_AA) #drawing the left iris
        cv2.circle(image, center_right, int(r_radius), (0, 255, 0), 1, cv2.LINE_AA) #drawing the right iris

    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Face Mesh', cv2.flip(image, 1))
    cv2.waitKey(0)

