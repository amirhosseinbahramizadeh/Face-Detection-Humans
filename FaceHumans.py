import cv2

img = cv2.imread("humans.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

humanface_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')

human = humanface_cascade.detectMultiScale(gray, scaleFactor=1.07,minNeighbors=3)


for (x , y , w , h) in human:
    img = cv2.rectangle(img,(x,y),(x+w,y+h),(149, 29, 171),2)
    cv2.putText(img," ", (x,y),cv2.FONT_HERSHEY_SIMPLEX, 1, 255)
    
cv2.imwrite("human.png", img)
