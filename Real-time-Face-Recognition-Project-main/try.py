import cv2

cap = cv2.VideoCapture(0)
ret, frame = cap.read()
if ret:
    cv2.imshow("Camera Test", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Camera not detected.")
cap.release()
