import cv2
import os
import time
import argparse

parser = argparse.ArgumentParser(
    description='Collecting data for drowsiness detection')

parser.add_argument('--c', '--camera-index', default=0, type=int,
                    help='Source camera 0, 1, 2,... (default: 0)')
parser.add_argument('--d', '--dataset-dir', default='data/dataset/',
                    help='Directory for saving dataset')
parser.add_argument('--n', '--number-image', default=500, type=int,
                    help='Number of image per state')

args = parser.parse_args()

def main():
    # Frame width, frame height
    width = 640
    height = 480

    # Load haarcascade
    r_eye_cascade = cv2.CascadeClassifier("haarcascades/haarcascade_lefteye_2splits.xml")
    l_eye_cascade = cv2.CascadeClassifier("haarcascades/haarcascade_righteye_2splits.xml")

    # Dataset path
    labels = ["open_eye", "close_eye"]
    DATASET_PATH = args.d
    number_images = args.n

    for label in labels:
        cap = cv2.VideoCapture(args.c + cv2.CAP_DSHOW)
        # Set width, height of frame
        cap.set(3, width)
        cap.set(4, height)
        os.mkdir(DATASET_PATH + label)
        print("Collecting images for " + label)
        time.sleep(3)
        for image in range(number_images):
            ret, frame = cap.read()

            # Convert frame to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            left_eye = l_eye_cascade.detectMultiScale(gray)
            right_eye = r_eye_cascade.detectMultiScale(gray)

            for (x, y, w, h) in left_eye:
                l_eye = frame[y:y+h, x:x+w]
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 0), thickness=1)
                cv2.imwrite(DATASET_PATH + label + "/" + str(image) + ".png", l_eye)
                
            for (x, y, w, h) in right_eye:
                r_eye = frame[y:y+h,x:x+w]
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 0), thickness=1)
                cv2.imwrite(DATASET_PATH + label + "/" + str(image) + ".png", r_eye)
            
            print("image " + str(image))
                
            cv2.imshow("Collect Images", frame)
            if cv2.waitKey(1) & 0xFF==ord("q"):
                break

    print('Collecting complete.....................')
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
	main()