from django.conf.urls import url
from . import views

urlpatterns = [
    url(r'^getModels', views.getModels, name='getModels'),
    url(r'^dirExists', views.dirExists, name='dirExists'),
    url(r'^getFeatures', views.getFeatures, name='getFeatures'),
    url(r'^getColumns', views.getColumns, name='getColumns'),
    url(r'^pvals', views.pvals, name='pvals'),
    url(r'^fvals', views.fvals, name='fvals'),
    url(r'^buildModelClass', views.buildModelClass, name='buildModelClass'),
    url(r'^buildModelRegress', views.buildModelRegression, name='buildModelRegression'),
    url(r'^runModel', views.runModel, name='runModel'),
]
