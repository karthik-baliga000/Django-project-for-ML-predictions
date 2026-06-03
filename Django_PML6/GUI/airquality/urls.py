from django.urls import path
from . import views

urlpatterns = [
    path('', views.aq_home, name='aq_home'),
    path('predict/', views.aq_predict, name='aq_predict'),
]