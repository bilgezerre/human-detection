import cv2 #OpenCV Python wrapper
import imutils #To perform different transformations from our results
import argparse #To read commands from comman terminal inside the script
import numpy as np

# Function to detect people in the frame and to display the number of people in the capture
def detect_people(frame):

    boxes, _ = HOGCV.detectMultiScale(frame, winStride=(4, 4), padding=(8, 8), scale=1.05)
    human_count = 1
    # Iterate over each box in the list of boxes and draw rectangles around people
    for (x, y, w, h) in boxes:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f'person {human_count}', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        human_count += 1
        
    # Display status and number of people in the frame    
    cv2.putText(frame, 'Status: Detecting ', (40, 40), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 0, 0), 2)
    cv2.putText(frame, f'Number of People: {human_count - 1}', (40, 70), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 0, 0), 2)
    cv2.imshow('Output', frame)
    return frame, human_count

# Function to detect the number of people in the video
def detect_in_video(path, writer=None):

    # Open the video using path
    video = cv2.VideoCapture(path)
    if video.isOpened():
        print('Detecting people...')
        while True:
            # Read the frame from video
            check, frame = video.read()
            if not check:
                break
            # Resize the frame
            frame = imutils.resize(frame, width=min(800, frame.shape[1]))
            # Detect people in the frame
            frame = detect_people(frame)
            # Write the frame to the output video if writer is not None
            if writer is not None:
                writer.write(frame)
            # Exit the loop if "q" is pressed
            if cv2.waitKey(1) == ord('q'):
                break
        video.release()
        cv2.destroyAllWindows()
    else:
        raise Exception('Please enter a valid path for the video.')

# Function to detect the number of people using webcam
def detect_with_camera(writer=None):
    video = cv2.VideoCapture(0)
    if video.isOpened():
        print("Detecting people...")
        while True:
            # Read the frame from webcam
            check, frame = video.read()
            if not check:
                break
            # Resize the frame
            frame = detect_people(frame)
            # Write the frame to the output video if writer is not None
            if writer is not None:
                writer.write(frame)
            # Exit the loop if "q" is pressed
            if cv2.waitKey(1) == ord("q"):
                break
        video.release()
        cv2.destroyAllWindows()
    else:
        raise Exception('Failed to open the camera.')

# Function to detect humans in a given image
def detect_in_image(path, output_path=None):
    # read timage from path
    image = cv2.imread(path)

    # if the image could not be read, print an error message and return
    if image is None:
        print("Failed to read image from path:", path)
        return

    # resize the image to have a width of 800 or the original width, whichever is smaller
    image = imutils.resize(image, width=min(800, image.shape[1]))

    # call the detect_people function to detect humans in the image
    result_image = detect_people(image)
    
    # if an output path was provided, write the result image to the path
    if output_path:
        cv2.imwrite(output_path, result_image)

    # wait for a key press and then close all windows
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# function to handle human detection based on the input arguments
def human_detectors(args):
    # get the path of the image and video
    image_path = args["image"]
    video_path = args['video']

    # check if the camera flag is set to True, otherwise set it to False
    if str(args["camera"]) == 'true' : camera = True 
    else : camera = False

    # initialize the video writer as None
    writer = None

     # if an output path was provided and the image path is None, initialize the video writer
    if args['output'] is not None and image_path is None:
        writer = cv2.VideoWriter(args['output'],cv2.VideoWriter_fourcc(*'MJPG'), 10, (600,600))

    # if the camera flag is set to True, call the detect_with_camera function
    if camera:
        print('Human Detection from webcam has been started')
        detect_with_camera(writer)
    # if a video path is provided, call the detect_in_video function
    elif video_path is not None:
        print('Human Detection in video has been started')
        detect_in_video(video_path, writer)
    # if an image path is provided, call the detect_in_image function
    elif image_path is not None:
        print('Human Detection in image has been started')
        detect_in_image(image_path, args['output'])

# Function to parse the input arguments
def args_parser():
    # initialize the argument parser
    arg_parse = argparse.ArgumentParser()
    # add the arguments
    arg_parse.add_argument("-v", "--video", default=None, help="Insert the path of the video")
    arg_parse.add_argument("-i", "--image", default=None, help="Insert the path of the image")
    arg_parse.add_argument("-c", "--camera", default=False, help="Set true to enable the camera")
    arg_parse.add_argument("-o", "--output", type=str, help="Insert the path of the output that you want to save")
    args = vars(arg_parse.parse_args())

    return args

if __name__ == "__main__":
    HOGCV = cv2.HOGDescriptor()
    HOGCV.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    args = args_parser()
    human_detectors(args)

