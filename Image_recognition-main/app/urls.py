
from django.urls import path, include
from .import views
from .import view
from rest_framework import routers
router=routers.DefaultRouter()
router.register('app',view.Note_image_recognition_view)
urlpatterns = [
 #path('', views.home, name="home"),
 #path('predict', views.predictdisease, name="predictdisease"),
 #path('team', views.team, name="team"),
 #path('predict', views.predict, name="predict"),
 #path('about', views.about, name="about"),
 path('form',view.myform,name='myform'),
 path('api',include(router.urls)),
 path('image_recognition',view.imagerecognition),

]