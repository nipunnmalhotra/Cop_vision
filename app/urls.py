from django.conf.urls import url
from django.urls import path,include
from . import views
# from app.views import add_to_cart,mark_order_as_done
urlpatterns=[
    path('',views.home,name="home"),
    path('login',views.login,name='login'),
    path('logout',views.logout,name='logout'),
    path('signup',views.signup,name='signup'),
    path('details',views.details,name='details'),
    path('bin',views.bin,name='bin'),
    path('view_challan',views.view_challan,name='view_challan'),
    path('search', views.search, name="search"),
    path('review',views.review,name='review'),
    path('profile',views.profile,name='profile'),
    path('approved',views.approved,name='approved'),
    path('video_upload',views.video_upload,name='video_upload'),
    path('image_upload',views.image_upload,name='image_upload'),
    path('vid_upload',views.vid_upload,name='vid_upload'),
    path('img_upload',views.img_upload,name='img_upload'),
    path('reject/<int:ids>',views.reject,name='reject'),
    path('approve/<int:ids>',views.approve,name='approve'),
    path('update/<int:ids>',views.update,name='update'),
    path('pdf',views.GeneratePdf,name='pdf'),
    path('blog',views.blog,name='blog'),
    path('aboutus',views.aboutus,name='aboutus'),
]
