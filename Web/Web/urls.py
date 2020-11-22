"""Web URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.1/topics/http/urls/
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
from apps.Ope import views
from django.contrib import admin
from django.urls import path  # 从 django.urls 引入 include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('index/', views.index),
    path('login/', views.login),
    path('logout/', views.logout),
    path('register/', views.register),
    path('order/', views.order),
    path('fuzz/', views.fuzz),
    path('main/', views.demo, name="index"),
    path('setting/', views.setting),
    path('fuzzing/', views.fuzzing),
    path('result/', views.result),
    path('history/', views.history),
    path('list/', views.list),
    path('getFuzzList/', views.getFuzzList),
    path('getFuzzResult/', views.getFuzzResult),
    path('getChart/', views.getChart),
    path('getCrashList/', views.getCrashList),
    path('run/', views.run),
    path('example/', views.example),
    path('seedupload/', views.seedupload),
    path('seeddelete/',views.seeddelete)
]
