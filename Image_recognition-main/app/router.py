from app.models import Note_image
from rest_framework import routers
from .views import Note_imageview
router=routers.DefaultRouter()
router.register('menu', Note_imageview)