from django.urls import path
from . import views as v

urlpatterns = [
    path('search/', v.SearchView.as_view(), name='search'),
]
