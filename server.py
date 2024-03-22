from ultralytics import YOLO
from threading import Thread
from queue import Queue
import json
import logging
import time
import cv2
import supervision as sv
import socket
import requests
import os

class Event:
	def __init__(self, event_type, data):
		self.event_type = event_type
		self.data = data

	def __str__(self):
		return f"{self.event_type}: {self.data}"
	
	def encode(self):
		result = {
			"event_type": self.event_type,
		}
		
		if self.data is not None:
			result["value"] = self.data

		return json.dumps(result).encode()

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
			self.has_warning_message_been_sent = False
			self.has_shot_being_fired_message_been_sent = False

		return is_inside

	def have_n_frames_passed(self, n):
		return self.number_of_frames >= n

def compute_angle(x, width, frame_width):
    center_x = x + width / 2
    offset = center_x - frame_width / 2

    max_offset = frame_width / 2
    max_angle = 45

    angle = (offset / max_offset) * max_angle

    return angle

server_ip = "0.0.0.0"
port = 8080

# camera is 1920x1080
safe_box_start, safe_box_end = (150, 90), (600, 400)
# safe_box_start, safe_box_end = (880, 90), (1540, 750)
safe_box_color, safe_box_thickness = (0, 255, 0), 3

annotator = sv.BoxAnnotator()
byte_tracker = sv.ByteTrack()

people_dict = {}
conns = []

def load_model(path: str):
	return YOLO(path)

def send_message_to_whatsapp_group(message: str):
	whatsapp_endpoint_base = os.environ["WHATSAPP_ENDPOINT_BASE"]
	requests.post(f"{whatsapp_endpoint_base}/chats/120363191426899928@g.us/messages",
		json = {
			"message": message
		}
	)

def camera_and_processing_thread(events_queue, physical_events_queue):
	logger = logging.getLogger("camera_and_processing_thread")
	logger.info("starting camera and processing thread")

	model = load_model("yolov8l.pt")
	logger.debug("yolov8 model loaded")

	video_capture = cv2.VideoCapture(0)
	logger.debug("setting up camera")

	frame_id = -1

	while video_capture.isOpened():
		success, frame = video_capture.read()
		frame_id += 1
		logger.info(f"processing frame #{frame_id}")
			
		if not success:
			logger.error("Error: Can't receive frame (stream end?). Exiting ...")
			break

		if frame_id % 10 != 0:
			continue

		predictions = model(frame)[0]
		detections = sv.Detections.from_ultralytics(predictions)
		detections = detections[detections.class_id == 0]
		detections = byte_tracker.update_with_detections(detections)

		labels = []

		for xyxy, tracker_id in zip(detections.xyxy, detections.tracker_id):
			if tracker_id in people_dict:
				people = people_dict[tracker_id]
				people.set_position(xyxy)
			else:
				people_dict[tracker_id] = People(tracker_id, xyxy)

		for id in people_dict:
			people = people_dict[id]

			if people.is_inside_of(safe_box_start, safe_box_end):
				labels.append(f"#{id}: thread")

				angle = compute_angle(people.current_position[0], people.current_position[2] - people.current_position[0], frame.shape[1])
				events_queue.put(Event("move", f"{angle:.1f}"))

				if people.have_n_frames_passed(150) and not people.has_warning_message_been_sent:
					physical_events_queue.put(Event("notification", f"advertencia: se ha detectado un intruso"))
					people.has_warning_message_been_sent = True

				if people.have_n_frames_passed(300) and not people.has_shot_being_fired_message_been_sent:
					physical_events_queue.put(Event("notification", "objetivo ha sido eliminado"))
					events_queue.put(Event("shoot", None))
					people.has_shot_being_fired_message_been_sent = True


		cv2.rectangle(frame, safe_box_start, safe_box_end, safe_box_color, safe_box_thickness)
		image_with_annotations = annotator.annotate(scene=frame.copy(), detections=detections, labels=labels)
		cv2.imshow('Prediction', image_with_annotations)

		key = cv2.waitKey(1) & 0xFF
		if key == ord("q"):
			break

	# send signals to terminate threads
	events_queue.put(Event("stop", None))
	physical_events_queue.put(Event("stop", None))

	# clean up windows
	video_capture.release()
	cv2.destroyAllWindows()

def socket_thread_connections():
	logger = logging.getLogger("socket_thread")
	logger.info("starting socket thread")
	server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
	server.bind((server_ip, port))
	server.listen(0)
	logger.info(f"listening on {server_ip}:{port}")

	while True:
		conn, addr = server.accept()
		conns.append(conn)
		logger.info(f"accepted connection from {addr}")


def socket_thread_processing(events_queue):
	logger = logging.getLogger("socket_thread_processing")
	logger.info("starting socket thread processing")
	while True:
		time.sleep(.1)
		if not events_queue.empty():
			event = events_queue.get()
			logger.info(f"processing socket event {event}")
			if event.event_type == "stop":
				break

			for conn in conns:
				conn.sendall(event.encode())

def physical_thread(events_queue):
	logger = logging.getLogger("physical_thread")
	logger.info("starting physical thread")
	while True:
		time.sleep(1)
		if not events_queue.empty():
			event = events_queue.get()
			if event.event_type == "notification":
				send_message_to_whatsapp_group(event.data)
			else:
				break


def main():
	logger = logging.getLogger("main")
	# events that can happen at the device running this program
	physical_events_queue = Queue()
	# events that will communicate to the raspberry device through sockets
	socket_events_queue = Queue(maxsize=1)

	# thread running the socket server for connections
	stc = Thread(target=socket_thread_connections)

	# thread running the socket server for events to send to clients
	stp = Thread(target=socket_thread_processing, args=(socket_events_queue,))

	# thread running the physical events such as audio and alerts
	pt = Thread(target=physical_thread, args=(physical_events_queue,))

	stc.start()
	stp.start()
	pt.start()

	camera_and_processing_thread(events_queue=socket_events_queue, physical_events_queue=physical_events_queue)

	stc.join(timeout=1)
	logger.info("closing socket server")
	stp.join()
	logger.info("closing socket conns")
	pt.join()
	logger.info("closing physical events")

if __name__ == "__main__":
	FORMAT = '%(asctime)s %(message)s'
	logging.basicConfig(format=FORMAT, level=logging.DEBUG)
	main()
