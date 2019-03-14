from django.urls import path
from . import views as v
from django.views.decorators.csrf import csrf_exempt

urlpatterns = [
    path('', v.test, name='test'),
    path('search', csrf_exempt(v.SearchView.as_view()), name='search'),
    path('simwords', csrf_exempt(v.SimwordsView.as_view()), name='simwords'),
    path('simbrands', csrf_exempt(v.SimbrandsView.as_view()), name='simbrands')
]
