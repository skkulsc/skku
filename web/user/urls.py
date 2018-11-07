from django.urls import path
from .views import *

urlpatterns = [
        path('', login.as_view()),
        path('signup', signup.as_view()),
        path('login', login.as_view()),
        path('logout', logout),
        ]
