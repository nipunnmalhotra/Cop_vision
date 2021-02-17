from imageai.Detection import ObjectDetection
import cv2
import numpy as np
import tensorflow as tf
import json
import requests
def getHelmets(frame):
    net = cv2.dnn.readNetFromDarknet("../static/models/yolov3-obj.cfg", "../static/models/yolov3-obj_2400.weights")
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    (H, W) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
    		swapRB=True, crop=False)
    net.setInput(blob)
    layerOutputs = net.forward(ln) 
    boxes = []
    confidences = []
    classIDs = []
    confThreshold=0.5
    	
    for output in layerOutputs:	
    	for detection in output:
    		scores = detection[5:]
    		classID = np.argmax(scores)
    		confidence = scores[classID]
    		if confidence > confThreshold:
    			box = detection[0:4] * np.array([W, H, W, H])
    			(centerX, centerY, width, height) = box.astype("int")
    			x = int(centerX - (width / 2))
    			y = int(centerY - (height / 2))
    			boxes.append([x, y, int(width), int(height)])
    			confidences.append(float(confidence))
    			classIDs.append(classID)
    
    if len(boxes)==0:
        return boxes
    def sortbyX1(temp):
        return temp[0]
    boxes=sorted(boxes,key=sortbyX1)
    new_boxes=[]
    for i in range(0,len(boxes)-1):
        temp1=boxes[i]
        temp2=boxes[i+1]
        
        if abs(temp1[0]-temp2[0])<=5 and abs(temp1[1]-temp2[1])<=5:
            continue
        
        new_boxes.append(boxes[i])
        
        
    new_boxes.append(boxes[len(boxes)-1])
    return new_boxes

def markItems(frame,numberOfHelmets,detections1):
    
    for i in numberOfHelmets:
        x1,y1,x2,y2=i
        frame=cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),3)
    
    for i in detections1:
        if i['name']=='person':
            x1,y1,x2,y2=i['box_points']
            frame=cv2.rectangle(frame,(x1,y1),(x2,y2),(150,120,0),3)
    return frame

def makeChallan(people,helmet):
    if people<helmet:
        return "Machine Error or image issue occured"
    triplingFault=False
    helmetFault=False
    
    if people>=3:
        triplingFault=True
    
    if helmet<people:
        helmetFault=True
    
    challan=str("")
    if helmetFault and triplingFault:
        challan="The faults are 1. Not wearing Helmet for all riders 2. Triple riding"
    elif helmetFault or triplingFault:
        challan=("The fault is 1. " + ("Not wearing Helmet for all riders" if helmetFault else "Triple riding"))
    else:
        challan="No challan"
    
    return challan

def getNumberPlate(image_path):   
    url = 'https://app.nanonets.com/api/v2/ObjectDetection/Model/' + credentials.model_id + '/LabelFile/'
    data = {'file': open(image_path, 'rb'),    'modelId': ('', credentials.model_id)}   
    response = requests.post(url, auth=requests.auth.HTTPBasicAuth(credentials.api_key, ''), files=data)
    result=list(json.loads(response.text)['result'])
    try:
        numberplate= result[0]['prediction'][0]['ocr_text']
        numberplate=numberplate.replace(' ','')
        numberplate=numberplate.replace('\n','')
        return numberplate
    except:
        return "Number not detected"

def isachallan(challan):
    if "The fault" in challan:
        return True
    else:
        return False


cap = cv2.VideoCapture('./videos/video.mp4') 
   

if (cap.isOpened()== False):  
  print("Error opening video  file") 
detector = ObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath("../static/models/resnet50_coco_best_v2.0.1.h5")
detector.loadModel()
model_id="1a61aa20-3298-48d2-ad39-5d62b6fd2768"
api_key='RkoauSinuSU7JTEMWPmkc8WZBXzDY4CR'

count=0
challans=[]
frmIncr=200
deltay1=-0.29
frmSize=(300,600)


while(cap.isOpened()):     
    ret, frame = cap.read() 
    if ret == True:
        frame=cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        image_name='frame{:d}.jpg'.format(count)
        input_path="./videos/frames/"+image_name
        output_path="./videos/output_frames/"+image_name
        cv2.imwrite(input_path, frame)
        detections = detector.detectObjectsFromImage(input_image=input_path, output_image_path= output_path)
        for i in detections:
            if i['name']=='motorcycle':
                x1,y1,x2,y2=i['box_points']
                y1=int(y1+deltay1*y1)
                frm=frame[y1:y2,x1:x2]
                frm=cv2.resize(frm,frmSize)
                numberOfHelmets=getHelmets(frm)
                inputfilepath="./videos/output/result{:d}.jpg".format(count)
                outputfilepath="./videos/output/suboutput{:d}.jpg".format(count)
                cv2.imwrite(inputfilepath,frm)
                detections1 = detector.detectObjectsFromImage(input_image=inputfilepath, 
                                                              output_image_path=outputfilepath)
                numberOfPeople=0
                for j in detections1:
                    if j['name']=='person':
                        numberOfPeople=numberOfPeople+1
                frm=markItems(frm,numberOfHelmets,detections1)
                resultpath="./videos/final/result{:d}.jpg".format(count)
                cv2.imwrite(resultpath,frm)
                challan=makeChallan(numberOfPeople,len(numberOfHelmets))
                if isachallan(challan):
                    numberplate=getNumberPlate(inputfilepath)
                    challans.append([inputfilepath,resultpath,numberplate,challan])
        count += frmIncr 
        cap.set(1, count)
    else:  
        break
cap.release() 
cv2.destroyAllWindows()                