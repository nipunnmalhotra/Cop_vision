from django.shortcuts import render,redirect,get_object_or_404
from django.contrib import auth
from django.contrib.auth import authenticate, login, logout
from django.http import HttpResponseRedirect, HttpResponse
from .models import *
from django.urls import reverse
from django.contrib import messages
from django.contrib.auth.decorators import login_required,permission_required
from .models import *
from datetime import date
from docx import *
from docx.shared import Inches
from io import StringIO
from io import BytesIO
from django.contrib.auth.models import User
from django.utils import timezone

# detection
import os
from imageai.Detection import ObjectDetection
import cv2
import numpy as np
import tensorflow as tf
import json
import requests

##pdf
from xhtml2pdf import pisa
from django.template.loader import get_template

serch=''

def render_to_pdf(template_src, context_dict={}):
    template = get_template(template_src)
    html  = template.render(context_dict)
    result = BytesIO()
    pdf = pisa.pisaDocument(BytesIO(html.encode("ISO-8859-1")), result)
    if not pdf.err:
        return HttpResponse(result.getvalue(), content_type='application/pdf')
    return None


def GeneratePdf(request):
    
    context={
        'challan':gen_challan.objects.filter(vehicle_number=serch).filter(approved=True)

    }    
    pdf = render_to_pdf('challan.html',context)
    return HttpResponse(pdf, content_type='application/pdf')

def getHelmets(frame):
    net = cv2.dnn.readNetFromDarknet("./static/models/yolov3-obj.cfg", "./static/models/yolov3-obj_2400.weights")
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
    challan=[]
    if helmetFault and triplingFault:
        challan.append('1. Not wearing Helmet for all riders')
        challan.append('2. Triple riding')
        # challan="The faults are 1. Not wearing Helmet for all riders 2. Triple riding"
    elif helmetFault or triplingFault:
        if helmetFault:
            challan.append('1. Not wearing Helmet for all riders') 
        else:
            challan.append('1. Triple riding')
        # challan=("The fault is 1. " + ("Not wearing Helmet for all riders" if helmetFault else "Triple riding"))
    else:
        challan.append("No challan")
    
    return challan

def getNumberPlate(image_path):   
    model_id="d839f991-462c-4557-a371-53678c9a43b2"
    api_key='mcGJdYIVR1QiDs5fOTB4fEZ7i7DHHqcX'
    # model_id="1a61aa20-3298-48d2-ad39-5d62b6fd2768"
    # api_key='RkoauSinuSU7JTEMWPmkc8WZBXzDY4CR'
    url = 'https://app.nanonets.com/api/v2/ObjectDetection/Model/' + model_id + '/LabelFile/'
    data = {'file': open(image_path, 'rb'),    'modelId': ('', model_id)}   
    response = requests.post(url, auth=requests.auth.HTTPBasicAuth(api_key, ''), files=data)
    # url = 'https://app.nanonets.com/api/v2/ObjectDetection/Model/' + credentials.model_id + '/LabelFile/'
    # data = {'file': open(image_path, 'rb'),    'modelId': ('', credentials.model_id)}   
    # response = requests.post(url, auth=requests.auth.HTTPBasicAuth(credentials.api_key, ''), files=data)
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

def count_fault(challan):
    if "are" in challan:
        return 2
    elif "No" in challan:
        return 0
    else:
        return 1        

def home(request):    
    return render(request,'homepage.html')

def login(request):
    if request.method=='POST':
        user=auth.authenticate(username=request.POST["username"],password=request.POST["password"])
        if user is not None:
            auth.login(request,user)
            
            return redirect('home')
        else:
            return render(request,'homepage.html',{'error':{
                'text':'Username or password is not correct'
            }})
    else:
        return render(request,'homepage.html')

def logout(request):
    if request.method=='POST':
        auth.logout(request)
        return redirect('home')    

def signup(request):
    if (request.method == 'GET'):
        #just want to view the sign up page
        return render(request,'signup.html')
    elif (request.method == 'POST'):
        #sign up data was filled and user is awaiting account creation
        if not request.POST["username"]:
            return render(request,'signup.html',{'error':{
                'text':'Username field cannot be left empty!'
            }})
        if (request.POST["password1"] == request.POST["password2"]):
            try:
                user = User.objects.get(username = request.POST["username"])
                return render(request,'signup.html',{'error':{
                    'text':'User already exists!'
                }})
            except User.DoesNotExist:
                user = User.objects.create_user(username=request.POST["username"],password=request.POST["password1"])
                auth.login(request,user)
                objct=employee()
                objct.user=request.user
                objct.save()

            return render(request,'details.html')
        else:
            return render(request,'signup.html',{'error':{
                'text':'Passwords don\'t match!'
            }})

@login_required
def details(request):
    if request.method == 'POST':
        if request.POST["nam"] and request.POST["email"] and request.POST["phone"] and request.POST["fnam"]:
            usersn=request.user
            obj=employee.objects.get(user=usersn)
            # obj.user = request.user
            obj.name = request.POST["nam"]
            obj.fname = request.POST["fnam"]
            obj.email=request.POST["email"]
            obj.phone=request.POST["phone"]         
            obj.save()
            return redirect('home')
        else:
                return render(request,'details.html',{'error':
                {
                'text':'Please fill all required information'
                }})
    else:
        return render(request,'details.html')           

def aboutus(request):        
    return render(request,'aboutus.html')

@login_required
def review(request):    
    context={
        'challan':gen_challan.objects.filter(approved=False).filter(rejected=False)        
    }
    return render(request,'review.html',context)

@login_required
def bin(request):    
    context={
        'challan':gen_challan.objects.filter(approved=False).filter(rejected=True)        
    }
    return render(request,'bin.html',context)

@login_required
def profile(request):    
	context = {
		'employee':employee.objects.filter(user=request.user)
		}
	return render(request,'profile.html',context)

@login_required
def approved(request):  
    context={
        'challan':gen_challan.objects.filter(approved=True).filter(rejected=False)        
    }  
    return render(request,'approved.html',context)    

def approve(request,ids):
    context={
        'challan':gen_challan.objects.filter(approved=True).filter(rejected=False)        
    }  
    challan=gen_challan.objects.filter(id=ids).get()
    if(challan.approved==True):
        challan.approved=False
    else:
        challan.approved=True
    challan.save()
    return redirect('home')

def reject(request,ids):
    context={
        'challan':gen_challan.objects.filter(approved=False).filter(rejected=True)        
    }
    challan=gen_challan.objects.filter(id=ids).get()
    if(challan.rejected==True):
        challan.rejected=False
    else:
        challan.rejected=True
    
    challan.save()
    return redirect('home')


def update(request,ids):
    context={
        'challan':gen_challan.objects.filter(approved=False).filter(rejected=True)        
    }
    challan=gen_challan.objects.filter(id=ids).get()
    if request.method == 'POST':
        if request.POST["vehicle_num"]:
            challan.vehicle_number=request.POST["vehicle_num"]
        else:
            return render(request,'homepage.html',{'error':
                {
                'text':'Please fill all required information'
                }})
    else:
        return render(request,'homepage.html')
    
    challan.save()
    return redirect('home')    

@login_required   
def video_upload(request):    
    return render(request,'video_upload.html')    
def blog(request):    
    return render(request,'blog.html')    

@login_required   
def image_upload(request):    
    return render(request,'image_upload.html')  

@login_required
def vid_upload(request): 
    if request.method == 'POST':
        if request.POST["title"] and request.POST["frames"]:
            vid=upload()
            vid.Video_Description = request.POST["title"]
            vid.numberofframes = request.POST["frames"]
            vid.datetime=timezone.datetime.now()            
            vid.videofile=request.FILES.get('video','')                                
            vid.save()
            conti=vid.videofile
            nishu=vid.numberofframes
            print(conti)
            print(nishu)            
            worker(nishu,str(conti))         
            return redirect('/video_upload')
        else:
                return render(request,'video_upload.html',{'error':
                {
                'text':'Please fill all required information'
                }})
    else:
        return render(request,'video_upload.html')  

@login_required
def img_upload(request): 
    if request.method == 'POST':
        if request.POST["title"]:
            vid=upload()
            vid.Video_Description = request.POST["title"]
            vid.numberofframes = 1
            vid.datetime=timezone.datetime.now()            
            vid.videofile=request.FILES.get('video','')                                
            vid.save()
            conti=vid.videofile
            nishu=vid.numberofframes
            print(conti)
            print(nishu)            
            work_image(str(conti))         
            return redirect('/video_upload')
        else:
                return render(request,'video_upload.html',{'error':
                {
                'text':'Please fill all required information'
                }})
    else:
        return render(request,'video_upload.html')          

def challan_amount(counts):
    if counts==1:
        return 2500
    elif counts==2:
        return 5000
    else:
        return 10000

def search(request):
    global serch
    if request.GET:
        search_term = request.GET['search_term']                
        serch=search_term
        
        context = {
            'search_term': search_term,
            'result': gen_challan.objects.filter(approved=True).filter(vehicle_number=search_term)    
        }

        return render(request,'search.html',context)
    else:
        return redirect('view_challan')

def view_challan(request):
    return render(request,'view_challan.html')

def worker(nishu,conti):
    dts=str(datetime.now())
    print(dts[0:4])
    print(dts[5:7])
    print(dts[8:10])
    cont=dts[0:4]+'/'+dts[5:7]+'/'+dts[8:10]
    print(cont)
    directory='./media/deploy/videos/'+cont+'/final'
    if not os.path.exists(directory):
        os.makedirs('./media/deploy/videos/'+cont+'/final')
        os.makedirs('./media/deploy/videos/'+cont+'/output')
        os.makedirs('./media/deploy/videos/'+cont+'/frames')
        os.makedirs('./media/deploy/videos/'+cont+'/output_frames')
    cap = cv2.VideoCapture('./media/'+conti)        
    if (cap.isOpened()== False):  
      print("Error opening video  file") 
    detector = ObjectDetection()
    detector.setModelTypeAsRetinaNet()
    detector.setModelPath("./static/models/resnet50_coco_best_v2.0.1.h5")
    detector.loadModel()
    model_id="d839f991-462c-4557-a371-53678c9a43b2"
    api_key='mcGJdYIVR1QiDs5fOTB4fEZ7i7DHHqcX'
    # model_id="1a61aa20-3298-48d2-ad39-5d62b6fd2768"
    # api_key='RkoauSinuSU7JTEMWPmkc8WZBXzDY4CR'
    count=0
    challans=[]
    frmIncr=200
    deltay1=-0.29
    frmSize=(300,600)
    while(cap.isOpened() and count<frmIncr*int(nishu)):     
        ret, frame = cap.read() 
        if ret == True:
            frame=cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            image_name='frame{:d}.jpg'.format(count)            
            input_path='./media/deploy/videos/'+cont+'/frames/'+image_name
            output_path='./media/deploy/videos/'+cont+'/output_frames/'+image_name            
            cv2.imwrite(input_path, frame)
            detections = detector.detectObjectsFromImage(input_image=input_path, output_image_path= output_path)
            for i in detections:
                if i['name']=='motorcycle':
                    x1,y1,x2,y2=i['box_points']
                    y1=int(y1+deltay1*y1)
                    frm=frame[y1:y2,x1:x2]
                    frm=cv2.resize(frm,frmSize)
                    numberOfHelmets=getHelmets(frm)
                    inputfilepath='./media/deploy/videos/'+cont+"/output/result{:d}.jpg".format(count)
                    outputfilepath='./media/deploy/videos/'+cont+"/output/suboutput{:d}.jpg".format(count)
                    cv2.imwrite(inputfilepath,frm)
                    detections1 = detector.detectObjectsFromImage(input_image=inputfilepath, 
                                                                  output_image_path=outputfilepath)
                    numberOfPeople=0
                    for j in detections1:
                        if j['name']=='person':
                            numberOfPeople=numberOfPeople+1
                    frm=markItems(frm,numberOfHelmets,detections1)                    
                    resultpath='./media/deploy/videos/'+cont+"/final/result{:d}.jpg".format(count)
                    cv2.imwrite(resultpath,frm)
                    challan=makeChallan(numberOfPeople,len(numberOfHelmets))
                    hoja=len(challan)
                    # fault_count=count_fault(challan)

                    # print(fault_count)

                    if(hoja>0 and challan[0]!='No challan'):
                        numberplate=getNumberPlate(inputfilepath)
                        # numberplate='no number detected'
                        ch=challan[0]+'\n'
                        if(hoja>1):
                            ch=ch+challan[1]
                        amt=challan_amount(hoja)
                        new_challan=gen_challan(vehicle_number=numberplate,fault=ch,faults=hoja,amount=amt,res_img=inputfilepath,frame_num=count)                        
                        new_challan.save()
                        challans.append([inputfilepath,resultpath,numberplate,challan])
            count += frmIncr 
            cap.set(1, count)
        else:  
            break
    cap.release() 
    cv2.destroyAllWindows()                
def work_image(conti):
    dts=str(datetime.now())
    print(dts[0:4])
    print(dts[5:7])
    print(dts[8:10])
    cont=dts[0:4]+'/'+dts[5:7]+'/'+dts[8:10]
    print(cont)
    directory='./media/deploy/videos/'+cont+'/final'
    if not os.path.exists(directory):
        os.makedirs('./media/deploy/videos/'+cont+'/final')
        os.makedirs('./media/deploy/videos/'+cont+'/output')
        os.makedirs('./media/deploy/videos/'+cont+'/frames')
        os.makedirs('./media/deploy/videos/'+cont+'/output_frames')
    
    detector = ObjectDetection()
    detector.setModelTypeAsRetinaNet()
    detector.setModelPath("./static/models/resnet50_coco_best_v2.0.1.h5")
    detector.loadModel()

    model_id="1a61aa20-3298-48d2-ad39-5d62b6fd2768"
    api_key='RkoauSinuSU7JTEMWPmkc8WZBXzDY4CR'
    
    count=1017
    challans=[]
    frmIncr=200
    deltay1=-0.29 #
    frmSize=(800,800)
    frame=cv2.imread('./media/'+conti)
    image_name='frame{:d}.jpg'.format(count)
    input_path='./media/deploy/videos/'+cont+'/frames/'+image_name
    output_path='./media/deploy/videos/'+cont+'/output_frames/'+image_name            
    cv2.imwrite(input_path, frame)            
    detections = detector.detectObjectsFromImage(input_image=input_path, output_image_path= output_path)
    for i in detections:
        if i['name']=='motorcycle':
            x1,y1,x2,y2=i['box_points']
            y1=int(y1+deltay1*y1)
            frm=frame[y1:y2,x1:x2]
            frm=cv2.resize(frm,frmSize)
            numberOfHelmets=getHelmets(frm)
            inputfilepath='./media/deploy/videos/'+cont+"/output/result{:d}.jpg".format(count)
            outputfilepath='./media/deploy/videos/'+cont+"/output/suboutput{:d}.jpg".format(count)
            cv2.imwrite(inputfilepath,frm)
            detections1 = detector.detectObjectsFromImage(input_image=inputfilepath, 
                                                          output_image_path=outputfilepath)
            numberOfPeople=0
            for j in detections1:
                if j['name']=='person':
                    numberOfPeople=numberOfPeople+1
            frm=markItems(frm,numberOfHelmets,detections1)                    
            resultpath='./media/deploy/videos/'+cont+"/final/result{:d}.jpg".format(count)
            cv2.imwrite(resultpath,frm)
            challan=makeChallan(numberOfPeople,len(numberOfHelmets))
            hoja=len(challan)
            # fault_count=count_fault(challan)

            # print(fault_count)

            if(hoja>0 and challan[0]!='No challan'):
                numberplate=getNumberPlate(inputfilepath)
                # numberplate='no number detected'
                ch=challan[0]+'\n'
                if(hoja>1):
                    ch=ch+challan[1]
                amt=challan_amount(hoja)
                new_challan=gen_challan(vehicle_number=numberplate,fault=ch,faults=hoja,amount=amt,res_img=inputfilepath,frame_num=count)                        
                new_challan.save()
                challans.append([inputfilepath,resultpath,numberplate,challan])    