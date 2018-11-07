"""recommendation URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/2.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""

import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from polls.views import index

from django.contrib import admin
from django.urls import path
from django.conf.urls import url, include

urlpatterns = [
    url(r'^', include('user.urls')),
    path('admin/', admin.site.urls),
    path('articles/', include('polls.urls')),
    path('user/', include('user.urls')),
]
