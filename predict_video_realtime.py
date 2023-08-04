#python predict_video_realtime.py --model model/activity_gpu.model --label-bin model/lb.pickle --output output/shoplifting.avi --size 64
# python predict_video_realtime.py -m model.hdf5 -l label_bin -o output

# import the necessary packages
from keras.models import load_model
from imutils.video import VideoStream
from collections import deque
import numpy as np
import argparse
import time
import pickle
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
	help="path to trained serialized model")
ap.add_argument("-l", "--label-bin", required=True,
	help="path to  label binarizer")
ap.add_argument("-i", "--input", required=False,
	help="path to our input video")
ap.add_argument("-o", "--output", required=True,
	help="path to our output video")
ap.add_argument("-s", "--size", type=int, default=128,
	help="size of queue for averaging")
args = vars(ap.parse_args())

# load the trained model and label binarizer from disk
print("[INFO] loading model and label binarizer...")
model = load_model(args["model"])
lb = pickle.loads(open(args["label_bin"], "rb").read())

# initialize the image mean for mean subtraction along with the
# predictions queue
mean = np.array([123.68, 116.779, 103.939][::1], dtype="float32")
Q = deque(maxlen=args["size"])

# initialize the video stream, pointer to output video file, and
#------------------# frame dimensions
# initialize the video stream and pointer to output video file, then
# allow the camera sensor to warm up
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()

writer = None
time.sleep(2.0)
W, H = None, None
prelabel = ''

ok = 'Normal'
fi_label = []
framecount = 0
# loop over frames from the video file stream
while True:
	# read the next frame from the file
	frame = vs.read()

	# if the frame was not grabbed, then we have reached the end
	# of the stream

	# if the frame dimensions are empty, grab them
	if W is None or H is None:
		H, W = frame.shape[:2]

	# clone the output frame, then convert it from BGR to RGB
	# ordering, resize the frame to a fixed 224x224, and then
	# perform mean subtraction
	output = frame.copy()
	frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	frame = cv2.resize(frame, (224, 224)).astype("float32")
	frame -= mean

	# make predictions on the frame and then update the predictions
	# queue
	preds = model.predict(np.expand_dims(frame, axis=0))[0]
	prediction = preds.argmax(axis=0)
	Q.append(preds)

	# perform prediction averaging over the current history of
	# previous predictions
	results = np.array(Q).mean(axis=0)
	print('Results = ', results)
	maxprob = np.max(results)
	print('Maximum Probability = ', maxprob)
	i = np.argmax(results)
	label = lb[i]
	rest = 1 - maxprob

	diff = (maxprob) - (rest)
	print('Difference of prob ', diff)
	th = 100
	if diff > .80:
		th = diff

	if (preds[prediction]) < th:
		text = "Alert : {} - {:.2f}%".format((ok), 100 - (maxprob * 100))
		cv2.putText(output, text, (35, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (255, 255, 255), 5)
	else:
		fi_label = np.append(fi_label, label)
		text = "Alert : {} - {:.2f}%".format((label), maxprob * 100)
		cv2.putText(output, text, (35, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (255, 255, 255), 5)
		prelabel = label

	# check if the video writer is None
	if writer is None:
		# initialize our video writer
		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		writer = cv2.VideoWriter(args["output"], fourcc, 30,
			(W, H), True)

	# write the output frame to disk
	writer.write(output)

	# show the output image
	cv2.imshow("Output", output)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break
with open('prelabelreal.pickle', 'wb') as f:
    pickle.dump(prelabel, f)
# release the file pointers
print("[INFO] cleaning up...")
print(prelabel)
writer.release()
vs.stream.release()

