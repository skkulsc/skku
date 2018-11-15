from django.conf.urls import url
from django.urls import path
from . import views as core_views

urlpatterns = [
        url(r'^$', core_views.index),
        path('content/', core_views.show_content),
        path('scrap/', core_views.scrap),
        path('scrapNews/', core_views.show_scrapNews),
]
