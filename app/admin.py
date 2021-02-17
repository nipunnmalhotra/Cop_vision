from django.contrib import admin
from django.contrib.auth.models import Group
# Register your models here.

from . models import *

class employeeadmin(admin.ModelAdmin):
	list_display = ('user','name','email','phone','fname',)
	list_editable = ('name','email','phone')
	list_per_page = 10
	search_fields = ('user',)

class challanadmin(admin.ModelAdmin):
	list_display = ('id','vehicle_number','amount','faults','fault','frame_num','rejected','approved')
	list_editable = ('rejected','approved',)
	list_per_page = 10
	search_fields = ('number',)

class videoadmin(admin.ModelAdmin):
	list_display=('Video_Description','timestamp','videofile','numberofframes',)
	list_per_page = 10
	
admin.site.register(upload,videoadmin)
admin.site.register(employee,employeeadmin)
admin.site.register(gen_challan,challanadmin)
