from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from django.conf import settings
from django.views.generic import View
# from django.contrib.staticfiles.storage import staticfiles_storage
# from django.contrib.staticfiles.templatetags.staticfiles import static
from gensim.models import Word2Vec
import os
import time

# Create your views here.

model_path = os.path.join(settings.BASE_DIR, 'word2vec.model')
w2v = Word2Vec.load(model_path)


class SearchView(View):
    def get(self, request):
        qry = request.GET.get('q', None)
        bnames = request.GET.get('b', None)

        if (qry is None) | (bnames is None):
            return JsonResponse({})

        else:
            qry = qry.split(' ')
            bnames = bnames.split(' ')
            sims = {}

            for bname in bnames:
                try:
                    sims[bname] = float(w2v.wv.n_similarity([bname], qry))
                except:
                    pass

            return JsonResponse(sims)


class SimwordsView(View):
    def get(self, request):
        bname = request.GET.get('b', None)

        if bname is None:
            return JsonResponse({})

        else:
            simwords = {k:v for k,v in w2v.wv.most_similar(bname, topn=100) if v>0.5}
            return JsonResponse(simwords, safe=False)
