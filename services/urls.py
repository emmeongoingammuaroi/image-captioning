from django.urls import path

from . import views


urlpatterns = [
    path('', views.index, name='index'),
    path('evaluate/', views.evaluate_caption, name='evaluate_caption'),
]