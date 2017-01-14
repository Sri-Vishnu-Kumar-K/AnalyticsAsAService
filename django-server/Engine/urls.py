from django.conf.urls import url
from . import views

urlpatterns = [
    url(r'^getFeatures', views.getFeatures, name='getFeatures'),
    url(r'^pvals', views.pvals, name='pvals'),
    url(r'^buildModelClass', views.buildModelClass, name='buildModelClass'),
    url(r'^buildModelRegress', views.buildModelRegression, name='buildModelRegression'),
    url(r'^runModel', views.runModel, name='runModel'),
]
