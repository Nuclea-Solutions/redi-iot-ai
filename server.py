from ultralytics import YOLO
import logging
import cv2
import supervision as sv
import socket
import numpy as np

class People:
	def __init__(self, id, position):
		self.id = id
		self.previous_position = None
		self.current_position = position

	def set_position(self, position):
		self.previous_position = self.current_position
		self.current_position = position
	
	def is_going_left(self):
		if self.previous_position is None:
			return False

		return self.current_position[0] < self.previous_position[0]
	
	def is_inside_of(self, start_box, end_box):
		x1, y1, x2, y2 = self.current_position.astype(int)
		# return x1 >= start_box[0] and y1 >= start_box[1] and x2 <= end_box[0] and y2 <= end_box[1]
		return x1 >= start_box[0] and x2 <= end_box[0]

server_ip = "127.0.0.1"
port = 8080

# camera is 1920x1080
safe_box_start, safe_box_end = (880, 90), (1540, 750)
safe_box_color, safe_box_thickness = (0, 255, 0), 3

annotator = sv.BoxAnnotator()
byte_tracker = sv.ByteTrack()

people_dict = {}

def load_model(path: str):
	return YOLO(path)

def main():
	FORMAT = '%(asctime)s %(clientip)-15s %(user)-8s %(message)s'
	logging.basicConfig(format=FORMAT)
	logger = logging.getLogger("server")

	model = load_model("yolov8l.pt")
	logger.info("yolov8 model loaded")

	server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
	server.bind((server_ip, port))
	server.listen(0)
	logger.info(f"listening on {server_ip}:{port}")

	video_capture = cv2.VideoCapture(0)
	logger.info("setting up ")

	if not video_capture.isOpened():
		logger.error("Error: Couldn't open camera.")
		exit()

	counter = 0

	while True:
		ret, frame = video_capture.read()
			
		if not ret:
			logger.error("Error: Can't receive frame (stream end?). Exiting ...")
			break

		predictions = model(frame)[0]
		detections = sv.Detections.from_yolov8(predictions)
		detections = detections[detections.class_id == 0]

		detections = byte_tracker.update_with_detections(detections)

		labels = []

		for xyxy, _, confidence, class_id, tracker_id in detections:
			if tracker_id in people_dict:
				people = people_dict[tracker_id]
				people.set_position(xyxy)
			else:
				people_dict[tracker_id] = People(tracker_id, xyxy)

		for id in people_dict:
			people = people_dict[id]

			if people.is_inside_of(safe_box_start, safe_box_end):
				print('thread thread thread')
				labels.append(f"#{id}: thread")
				print('\a')

		cv2.rectangle(frame, safe_box_start, safe_box_end, safe_box_color, safe_box_thickness)
		image_with_annotations = annotator.annotate(scene=frame.copy(), detections=detections, labels=labels)
		cv2.imshow('Prediction', image_with_annotations)

		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	video_capture.release()
	cv2.destroyAllWindows()


if __name__ == "__main__":
	main()