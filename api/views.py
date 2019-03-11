from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from django.conf import settings
from django.contrib.staticfiles.storage import staticfiles_storage
from django.contrib.staticfiles.templatetags.staticfiles import static
from gensim.models import Word2Vec
import os

# Create your views here.

def search(request):
    # model_path = os.path.join(settings.STATIC_URL, 'word2vec.model')
    #model_path = staticfiles_storage.url('word2vec.model')
    model_path = static('word2vec.model')
    w2v = Word2Vec.load(model_path)
    result = w2v.wv.most_similar(positive=['화제'], topn=20)
    return JsonResponse(result, safe=False)
    # return HttpResponse(model_path)
