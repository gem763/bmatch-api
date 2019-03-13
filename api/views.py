from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from django.conf import settings
from django.views.generic import View
# from django.contrib.staticfiles.storage import staticfiles_storage
# from django.contrib.staticfiles.templatetags.staticfiles import static
from gensim.models import Word2Vec
import os
import time
import json

# Create your views here.

model_path = os.path.join(settings.BASE_DIR, 'word2vec.model')
w2v = Word2Vec.load(model_path)


def test(request):
    return HttpResponse(model_path)


class SearchView(View):
    def get(self, request):
        q = request.GET.get('q', None)

        if q is not None:
            q = json.loads(q)
            qry = q.pop('qry')
            sims = {}

            for bname, keywords in q.items():
                try:
                    sims[bname] = float(w2v.wv.n_similarity(keywords, qry))
                except:
                    _keywords = [k for k in keywords if k in w2v.wv.vocab]
                    _qry = [k for k in qry if k in w2v.wv.vocab]
                    if len(_keywords)*len(_qry) != 0:
                        sims[bname] = float(w2v.wv.n_similarity(_keywords, _qry))

            return JsonResponse(sims)


class SearchView2(View):
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
        words = request.GET.get('b', None)
        topn = request.GET.get('topn', 100)
        min = request.GET.get('min', 0.5)

        if words is None:
            return JsonResponse({})

        else:
            simwords = {k:v for k,v in w2v.wv.most_similar(words.split(' '), topn=topn) if v>min}
            return JsonResponse(simwords, safe=False)
