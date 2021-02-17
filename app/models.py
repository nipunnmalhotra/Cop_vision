from django.db import models
from django.contrib.auth.models import User
from django.utils.timezone import datetime

# Create your models here.

class employee(models.Model):
	user = models.ForeignKey(User, on_delete=models.CASCADE)
	name = models.CharField(max_length=20,default='Employee')
	phone = models.IntegerField(default=0)
	fname = models.CharField(max_length=20,default='fathers name')
	email = models.EmailField(max_length=100,default='abc@gmail.com')
	
class gen_challan(models.Model):
	vehicle_number = models.CharField(max_length=50,default='Number not detected')
	amount = models.IntegerField(default=0)
	faults = models.IntegerField(default=0)
	fault = models.CharField(max_length=50,default='No fault')
	frame_num = models.CharField(max_length=50,default='0')	
	res_img = models.ImageField(upload_to='deploy/videos/%Y/%m/%d/',blank=True,null=True)
	approved = models.BooleanField(default=False)
	rejected = models.BooleanField(default=False)

class upload(models.Model):
	Video_Description= models.CharField(max_length=500)
	timestamp   = models.DateTimeField(auto_now_add=True,blank=True,null=True)
	videofile= models.FileField(upload_to='deploy/videos/%Y/%m/%d/', null=True, verbose_name="")
	numberofframes=models.IntegerField(default=0)
