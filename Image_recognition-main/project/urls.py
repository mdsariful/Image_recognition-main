from django.contrib import admin
from django.urls import path, include
from django.conf.urls import handler404
from django.conf import settings
from django.conf.urls.static import static
from django.views.static import serve
from django.conf.urls import url
urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('app.urls')),
    
]+ static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)



