# import required libraries
from vidgear.gears import VideoGear
import cv2
import serial
import time
import numpy as np
import onnxruntime
import mediapipe as mp
import Jetson.GPIO as GPIO
import jetson.inference 
import jetson.utils 


GPIO.setmode(GPIO.BOARD)
GPIO.setwarnings(False)
channel_buzzer = 11
channel_led = 18
channel_led1 = 21
GPIO.setup(channel_buzzer, GPIO.OUT)
GPIO.setup(channel_led, GPIO.OUT)
GPIO.setup(channel_led1, GPIO.OUT)
GPIO.output(channel_buzzer, GPIO.HIGH)
GPIO.output(channel_led, GPIO.HIGH)
GPIO.output(channel_led1, GPIO.HIGH)


# text font
font = cv2.FONT_HERSHEY_SIMPLEX 

# Load model
sess = onnxruntime.InferenceSession("models/cnn/model_cnn.onnx", providers=["CUDAExecutionProvider"])
net = jetson.inference.detectNet(argv=['--model=models/ssd/ssd-mobilenet.onnx', '--labels=models/ssd/labels.txt', '--input-blob=input_0', '--output-cvg=scores', '--output-bbox=boxes'], threshold=0.5)
# net = jetson.inference.detectNet("ssd-mobilenet-v2", threshold=0.5)

ser = serial.Serial('/dev/ttyACM0', baudrate=9600, timeout=.1)

# Input image size
image_width = 24
image_height = 24

# Face mediapipe
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection()

# Eyes haarcascade
r_eye_cascade = cv2.CascadeClassifier("haarcascades/haarcascade_lefteye_2splits.xml")
l_eye_cascade = cv2.CascadeClassifier("haarcascades/haarcascade_righteye_2splits.xml")

# Eyes state
labels = ' '

# Init value
# Eyes value
l_val = [99]
r_val = [99]
e_val = 0 
e_cnt = 0

# Face value
f_pos = 0
f_cnt = 0


stream1 = VideoGear(source=0, logging=True).start() 
stream2 = VideoGear(source=1, logging=True).start() 

def drowsiness_detection(frame):

    global labels, l_val, r_val, e_val, e_cnt, f_pos, f_cnt
    # frame_height, frame_width = frame.shape[:2]

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Face detection
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(image)
    # print(len(results.detections))
    if results.detections:
        face_exist = True
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, ic = frame.shape
            bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
            xcenter = int(bboxC.xmin * iw + (bboxC.width * iw)/2) 
            ycenter = int(bboxC.ymin * ih + (bboxC.height * ih)/2)
            cv2.rectangle(frame, bbox, (0, 0, 255), 2)
            # cv2.circle(frame, (xcenter, ycenter), 2, (0, 0, 0), 2)
            # print(len(results.detections))
    else:
        face_exist = False
        xcenter = None 
        ycenter = None
    # print(face_exist)

    # Detect eyes
    left_eye = l_eye_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))
    right_eye = r_eye_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    # eyes detection
    for (x, y, w, h) in left_eye:
        l_eye = frame[y:y+h, x:x+w]
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        l_eye = cv2.cvtColor(l_eye, cv2.COLOR_BGR2GRAY)
        l_eye = cv2.resize(l_eye, (image_width, image_height))
        l_eye = l_eye/255
        l_eye = l_eye.reshape(image_width, image_height, -1)
        l_eye = np.array(l_eye, dtype="float32")
        l_eye = np.expand_dims(l_eye, axis=0)
        l_eye = np.vstack([l_eye])
        l_val = sess.run(["dense_1"], {"input": l_eye})

    for (x, y, w, h) in right_eye:
        r_eye = frame[y:y+h,x:x+w]
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        r_eye = cv2.cvtColor(r_eye, cv2.COLOR_BGR2GRAY)
        r_eye = cv2.resize(r_eye, (image_width, image_height))
        r_eye = r_eye/255
        r_eye = r_eye.reshape(image_width, image_height, -1)
        r_eye = np.array(r_eye, dtype="float32")
        r_eye = np.expand_dims(r_eye, axis=0)
        r_eye = np.vstack([r_eye])
        r_val = sess.run(["dense_1"], {"input": r_eye})

    if len(left_eye) and len(right_eye):
        eyes_exist = True 
    else:
        eyes_exist = False

    e_val = ((l_val[0] + r_val[0])/2)
    f_pos = xcenter
    # print(f_pos)

    # Alert 
    # Case 1
    if face_exist and eyes_exist:
        GPIO.output(channel_led, GPIO.LOW)
        # cv2.putText(frame, "Truong hop 1", (50, 460), font, 2, (255, 255, 255), 2, cv2.LINE_AA)
        f_cnt = 0
        if e_val < 0.7:
            e_cnt += 1
            if e_cnt > 7:
                e_cnt = 7
                cv2.putText(frame, 'ALERT!!!', (50, 100), font, 2, (0, 0, 255), 2, cv2.LINE_AA)
                GPIO.output(channel_buzzer, GPIO.LOW)
                time.sleep(0.05)
                labels = 'Mat nham - Canh bao ngu ngat'
        else:
            e_cnt -= 5
            if e_cnt < 0:
                e_cnt = 0
                GPIO.output(channel_buzzer, GPIO.HIGH)
                time.sleep(0.05)
                labels = 'Mat mo - Binh Thuong'

    # Case 2
    elif face_exist and not eyes_exist:
        # cv2.putText(frame, "Truong hop 2", (50, 460), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
        e_cnt = 0
        if f_pos > 200 and f_pos < 400:
            f_cnt += 3
            if f_cnt > 10:
                f_cnt = 10
                cv2.putText(frame, 'ALERT!!!', (50, 100), font, 2, (0, 0, 255), 2, cv2.LINE_AA)
                GPIO.output(channel_buzzer, GPIO.LOW)
                time.sleep(0.05)
                labels = 'Khong phat hien mat - Canh bao ngu gat'
        else:
            f_cnt += 2
            if f_cnt > 20:
                f_cnt = 20
                cv2.putText(frame, 'ALERT!!!', (50, 100), font, 2, (0, 0, 255), 2, cv2.LINE_AA)
                GPIO.output(channel_buzzer, GPIO.LOW)
                time.sleep(0.05)
                labels = 'Khong phat hien mat - Canh bao mat tap trung'
    
    # Case 3
    elif not face_exist and not eyes_exist:
        GPIO.output(channel_buzzer, GPIO.HIGH)
        GPIO.output(channel_led, GPIO.HIGH)
        time.sleep(0.05)
        labels = ' '
        e_cnt = 0
        f_cnt = 0
    
    # print(f'eye count: {e_cnt}, face count: {f_cnt}')
    # print(face_exist, eyes_exist)
    
    cv2.rectangle(frame, (0, 0), (640, 50), (0, 0, 0), thickness=-1)
    cv2.rectangle(frame, (0, 430), (640, 480), (0, 0, 0), thickness=-1)
    cv2.putText(frame, labels, (30, 30), font, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, 'Eye counter: ' + str(e_cnt), (30, 460), font, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, 'Face counter: ' + str(f_cnt), (400, 460), font, 0.75, (255, 255, 255), 2, cv2.LINE_AA)

    
def blindspot_detection(frame):
    img_cuda = jetson.utils.cudaFromNumpy(frame)
    detections = net.Detect(img_cuda)
    if detections:
        GPIO.output(channel_led1, GPIO.LOW)
        time.sleep(0.05)
        GPIO.output(channel_led1, GPIO.HIGH)
        for d in detections:
            x1, y1, x2, y2 = int(d.Left), int(d.Top), int(d.Right), int(d.Bottom)
            class_name = net.GetClassDesc(d.ClassID)
            # print(class_name)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, class_name + " " + str(round(d.Confidence, 3)), (x1+5, y1+25), font, 0.75, (255, 0, 0), 2) 
            if class_name == "nguoi": 
                ser.write(bytes('person\n','utf-8'))
            elif class_name == "xemay":
                ser.write(bytes('motorbike\n','utf-8'))
            elif class_name == "oto":
                ser.write(bytes('car\n','utf-8')) 
    else:
        GPIO.output(channel_led1, GPIO.HIGH)
        time.sleep(0.05)  
        
while True:
    
    frameA = stream1.read()
    drowsiness_detection(frameA)
    frameB = stream2.read()
    blindspot_detection(frameB)

    
    # check if any of two frame is None
    if frameA is None or frameB is None:
        #if True break the infinite loop
        break
    
    frame = np.hstack((frameA, frameB))
    # do something with both frameA and frameB here
    # cv2.imshow("Output Frame1", frameA)
    # cv2.imshow("Output Frame2", frameB)
    cv2.imshow("Output", frame)

    # Show output window of stream1 and stream 2 seperately

    key = cv2.waitKey(1) & 0xFF
    # check for 'q' key-press
    if key == ord("q"):
        GPIO.cleanup()
        #if 'q' key-pressed break out
        break

cv2.destroyAllWindows()
# close output window

# safely close both video streams
stream1.stop()
stream2.stop()
