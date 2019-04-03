from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from django.conf import settings
from django.views.generic import View
from gensim.models import Doc2Vec #, Word2Vec
import os
import time
import json
import re
import numpy as np

# Create your views here.

model_path = os.path.join(settings.BASE_DIR, 'doc2vec.model')
d2v = Doc2Vec.load(model_path)


def test(request):
    return HttpResponse(model_path)


# class SearchView(View):
#     def post(self, request):
#         qry = request.POST.get('qry', None)
#         brands = request.POST.get('brands', None)
#
#         if (qry is None) | (brands is None):
#             return JsonResponse({})
#
#         else:
#             brands = json.loads(brands)
#             qry = qry.split(' ')
#             sims = {}
#
#             for bname, keywords in brands.items():
#                 try:
#                     sims[bname] = float(w2v.wv.n_similarity(keywords, qry))
#                 except:
#                     _keywords = [k for k in keywords if k in w2v.wv.vocab]
#                     _qry = [k for k in qry if k in w2v.wv.vocab]
#                     if len(_keywords)*len(_qry) != 0:
#                         sims[bname] = float(w2v.wv.n_similarity(_keywords, _qry))
#
#             return JsonResponse(sims)
#
#
# class SearchView_old(View):
#     def get(self, request):
#         qry = request.GET.get('q', None)
#         bnames = request.GET.get('b', None)
#
#         if (qry is None) | (bnames is None):
#             return JsonResponse({})
#
#         else:
#             qry = qry.split(' ')
#             bnames = bnames.split(' ')
#             sims = {}
#
#             for bname in bnames:
#                 try:
#                     sims[bname] = float(w2v.wv.n_similarity([bname], qry))
#                 except:
#                     pass
#
#             return JsonResponse(sims)



# class SimwordsView_old(View):
#     def post(self, request):
#         words = request.POST.get('w', None)
#         topn = request.POST.get('topn', 100)
#         min = request.POST.get('min', 0.5)
#
#         if words is None:
#             return JsonResponse({})
#
#         else:
#             words = words.split(' ')
#
#             try:
#                 simwords = {k:v for k,v in w2v.wv.most_similar(words, topn=int(topn)) if v > float(min)}
#                 return JsonResponse(simwords)
#
#             except:
#                 _words = [k for k in words if k in w2v.wv.vocab]
#                 if len(_words) != 0:
#                     simwords = {k:v for k,v in w2v.wv.most_similar(_words, topn=int(topn)) if v > float(min)}
#                     return JsonResponse(simwords)
#
#                 else:
#                     return JsonResponse({})


class SimwordsView(View):
    def post(self, request):
        bname = request.POST.get('bname', None)
        topn = request.POST.get('topn', 100)
        min = request.POST.get('min', 0.5)

        if bname is None:
            return JsonResponse({})

        else:
            simwords = d2v.wv.most_similar(positive=[d2v.docvecs[bname]], topn=int(topn))
            return JsonResponse({k:v for k,v in simwords if v > float(min)})


class SimbrandsView(View):
    def post(self, request):
        qry = request.POST.get('qry', None)
        bname = request.POST.get('bname', None)
        topn = len(d2v.docvecs) + 10

        if (qry is None) & (bname is not None):
            sims = d2v.docvecs.most_similar(positive=[bname], topn=topn)
            return JsonResponse(dict(sims))

        elif (qry is not None) & (bname is None):
            qry = [w for w in re.split('\W+', qry) if w!='']
            qry_vec = d2v.infer_vector(qry, epochs=500)
            sims = d2v.docvecs.most_similar(positive=[qry_vec], topn=topn)
            return JsonResponse(dict(sims))

        else:
            return JsonResponse({})


# class SimbrandsView_old(View):
#     def _simbrands(self, mykeywords, keywords_dict):
#         sims = {}
#         for bname, keywords in keywords_dict.items():
#             try:
#                 sims[bname] = float(w2v.wv.n_similarity(keywords, mykeywords))
#             except:
#                 _keywords = [k for k in keywords if k in w2v.wv.vocab]
#                 _mykeywords = [k for k in mykeywords if k in w2v.wv.vocab]
#                 if len(_keywords)*len(_mykeywords) != 0:
#                     sims[bname] = float(w2v.wv.n_similarity(_keywords, _mykeywords))
#
#         return sims
#
#     def post(self, request):
#         qry = request.POST.get('qry', None)
#         mybname = request.POST.get('mybname', None)
#         brands = request.POST.get('brands', None)
#
#         if (qry is None) & (mybname is not None) & (brands is not None):
#             keywords_dict = json.loads(brands)
#             mykeywords = keywords_dict[mybname]
#             sims = self._simbrands(mykeywords, keywords_dict)
#             return JsonResponse(sims)
#
#         elif (qry is not None) & (mybname is None) & (brands is not None):
#             keywords_dict = json.loads(brands)
#             mykeywords = qry.split(' ')
#             sims = self._simbrands(mykeywords, keywords_dict)
#             return JsonResponse(sims)
#
#         else:
#             return JsonResponse({})


def minmax_scale(dic, max=100, min=0):
    keys = dic.keys()
    x = np.array(list(dic.values()))
    x = np.interp(x, (x.min(), x.max()), (min, max))
    # return dict(zip(keys, x))
    return {k:int(v) for k,v in zip(keys, x)}


class IdentityView(View):
    def post(self, request):
        bname = request.POST.get('bname', None)
        idwords = request.POST.get('idwords', None)

        if (bname is None) | (idwords is None):
            return JsonResponse({})

        else:
            idwords = json.loads(idwords)
            brand_vec = d2v.docvecs[bname]
            idty = {}

            for _idwords in idwords:
                _idty = {}
                for k,v in _idwords.items():
                    word_vec = d2v.infer_vector([w.strip() for w in v.split(' ')], epochs=500)
                    _idty[k] = float(d2v.wv.cosine_similarities(brand_vec, [word_vec])[0])

                _idty_sum = sum(_idty.values())
                _idty = {k:v/_idty_sum for k,v in _idty.items()}
                idty.update(_idty)

            idty = minmax_scale(idty, max=100, min=30)
            return JsonResponse(idty)
