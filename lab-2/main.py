from figure import Figure, Path
import imutils
import cv2


class Detection:

	def process_images(img_count):
		for i in range(img_count):
			image = cv2.imread(f"images/test{i}.jpg")
			thresh = cv2.threshold(cv2.GaussianBlur(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY),
													(5, 5), 0), 60, 255, cv2.THRESH_BINARY)[1]

			contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
			contours = imutils.grab_contours(contours)

			for ctr in contours:
				M = cv2.moments(ctr)
				cv2.drawContours(image, [ctr.astype("int")], -1, (255, 255, 255), 2)
				cv2.putText(image, Detection.find_figure(ctr),
							(int((M["m10"] / M["m00"])), int((M["m01"] / M["m00"]))),
							cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
			Path("tested_images").mkdir(parents=True, exist_ok=True)
			cv2.imwrite(f"tested_images/test{i}.jpg", image)

	def find_figure(contour):
		approx = cv2.approxPolyDP(contour, 0.04 * cv2.arcLength(contour, True), True)
		if len(approx) == 3:
			figure = "Triangle"
		elif len(approx) == 4:
			x, y, w, h = cv2.boundingRect(approx)
			figure = "Square" if 0.9 <= w / float(h) <= 1.1\
				else "Rectangle"
		elif len(approx) == 5:
			figure = "Pentagon"
		elif len(approx) == 6:
			figure = "Hexagon"
		else:
			figure = "Circle"
		return figure


if __name__ == '__main__':
	figure = Figure()
	N = int(input("Enter N: "))
	figure.generate(N)
	Detection.process_images(N)
