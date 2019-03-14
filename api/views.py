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
    def post(self, request):
        qry = request.POST.get('qry', None)
        brands = request.POST.get('brands', None)

        if (qry is None) | (brands is None):
            return JsonResponse({})

        else:
            brands = json.loads(brands)
            qry = qry.split(' ')
            sims = {}

            for bname, keywords in brands.items():
                try:
                    sims[bname] = float(w2v.wv.n_similarity(keywords, qry))
                except:
                    _keywords = [k for k in keywords if k in w2v.wv.vocab]
                    _qry = [k for k in qry if k in w2v.wv.vocab]
                    if len(_keywords)*len(_qry) != 0:
                        sims[bname] = float(w2v.wv.n_similarity(_keywords, _qry))

            return JsonResponse(sims)


class SearchView_old(View):
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
    def post(self, request):
        words = request.POST.get('w', None)
        topn = request.POST.get('topn', 100)
        min = request.POST.get('min', 0.5)

        if words is None:
            return JsonResponse({})

        else:
            words = words.split(' ')

            try:
                simwords = {k:v for k,v in w2v.wv.most_similar(words, topn=int(topn)) if v>float(min)}
                return JsonResponse(simwords)

            except:
                _words = [k for k in words if k in w2v.wv.vocab]
                if len(_words) != 0:
                    simwords = {k:v for k,v in w2v.wv.most_similar(_words, topn=int(topn)) if v>float(min)}
                    return JsonResponse(simwords)

                else:
                    return JsonResponse({})


class SimbrandsView(View):
    def post(self, request):
        my = request.POST.get('my', None)
        brands = request.POST.get('brands', None)
        # topn = request.GET.get('topn', 30)
        # min = request.GET.get('min', 0.2)

        if (my is None) | (brands is None):
            return JsonResponse({})

        else:
            brands = json.loads(brands)
            mykeywords = brands.pop(my)
            sims = {}

            for bname, keywords in brands.items():
                try:
                    sims[bname] = float(w2v.wv.n_similarity(keywords, mykeywords))
                except:
                    _keywords = [k for k in keywords if k in w2v.wv.vocab]
                    _mykeywords = [k for k in mykeywords if k in w2v.wv.vocab]
                    if len(_keywords)*len(_mykeywords) != 0:
                        sims[bname] = float(w2v.wv.n_similarity(_keywords, _mykeywords))

            return JsonResponse(sims)
