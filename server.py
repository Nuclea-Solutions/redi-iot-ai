from ultralytics import YOLO
import logging
import cv2
import supervision as sv
import socket
import requests
import os

class People:
	def __init__(self, id, position):
		self.id = id
		self.previous_position = None
		self.current_position = position
		self.has_warning_message_been_sent = False
		self.has_shot_being_fired_message_been_sent = False
		self.number_of_frames = 0

	def set_position(self, position):
		self.previous_position = self.current_position
		self.current_position = position
	
	def is_going_left(self):
		if self.previous_position is None:
			return False

		return self.current_position[0] < self.previous_position[0]
	
	def is_inside_of(self, start_box, end_box):
		x1, _, x2, _ = self.current_position.astype(int)
		is_inside = x1 >= start_box[0] and x2 <= end_box[0]

		if is_inside:
			self.number_of_frames += 1
		else:
			self.number_of_frames = 0
			self.has_message_been_sent = False

		return is_inside

	def have_n_frames_passed(self, n):
		return self.number_of_frames >= n

server_ip = "127.0.0.1"
port = 8080

# camera is 1920x1080
safe_box_start, safe_box_end = (150, 90), (600, 400)
# safe_box_start, safe_box_end = (880, 90), (1540, 750)
safe_box_color, safe_box_thickness = (0, 255, 0), 3

annotator = sv.BoxAnnotator()
byte_tracker = sv.ByteTrack()

people_dict = {}

def load_model(path: str):
	return YOLO(path)

def send_message_to_whatsapp_group(message: str):
	whatsapp_endpoint_base = os.environ["WHATSAPP_ENDPOINT_BASE"]
	requests.post(f"{whatsapp_endpoint_base}/chats/120363191426899928@g.us/messages",
		json = {
			"message": message
		}
	)


def main():
	FORMAT = '%(asctime)s %(message)s'
	logging.basicConfig(format=FORMAT, level=logging.DEBUG)
	logger = logging.getLogger("server")

	model = load_model("yolov8l.pt")
	logger.debug("yolov8 model loaded")

	server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
	server.bind((server_ip, port))
	server.listen(0)
	logger.info(f"listening on {server_ip}:{port}")

	video_capture = cv2.VideoCapture(0)
	logger.debug("setting up camera")

	if not video_capture.isOpened():
		logger.error("Error: Couldn't open camera.")
		exit()

	counter = 0
	frame_id = 0

	while True:
		logger.info(f"processing frame #{frame_id}")
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
				labels.append(f"#{id}: thread")

				if people.have_n_frames_passed(150) and not people.has_warning_message_been_sent:
					send_message_to_whatsapp_group("advertencia: se ha detectado un intruso")
					people.has_warning_message_been_sent = True

				if people.have_n_frames_passed(300) and not people.has_shot_being_fired_message_been_sent:
					send_message_to_whatsapp_group("objetivo ha sido eliminado")
					people.has_shot_being_fired_message_been_sent = True


		cv2.rectangle(frame, safe_box_start, safe_box_end, safe_box_color, safe_box_thickness)
		image_with_annotations = annotator.annotate(scene=frame.copy(), detections=detections, labels=labels)
		cv2.imshow('Prediction', image_with_annotations)

		key = cv2.waitKey(1) & 0xFF
		if key == ord("q"):
			break

		frame_id += 1

	video_capture.release()
	cv2.destroyAllWindows()


if __name__ == "__main__":
	main()
