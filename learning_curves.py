#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 20:33:51 2019

@author: juhoi
"""
import sys
import annif
from annif.cli import get_project, open_documents
from annif.suggestion import SuggestionFilter


app = annif.create_app(config_name='annif.default_config.ProductionConfig')

#project_id = sys.argv[1]
#paths = sys.argv[2]
#evals = int(sys.argv[3])
project_id = 'tfidf-fi'
paths = '../Annif-corpora/training/2019/yso-cicero-finna-fi_01.tsv.gz'

limit = 10
threshold = 0

with app.app_context():
    proj = get_project(project_id)
documents = open_documents(paths)
print(documents)
proj.train(documents)

hit_filter = SuggestionFilter(limit=limit, threshold=threshold)
eval_batch = annif.eval.EvaluationBatch(proj.subjects)


docs = open_documents(paths)
for doc in docs.documents:
    results = proj.suggest(doc.tex)
    hits = hit_filter(results)
    eval_batch.evaluate(hits,
                        annif.corpus.SubjectSet((doc.uris, doc.labels)))

template = "{0:<20}\t{1}"
for metric, score in eval_batch.results().items():
    print(template.format(metric + ":", score))
