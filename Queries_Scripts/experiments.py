import argparse
import json
import math
import multiprocessing
import pickle
import statistics
import subprocess
import time
from functools import partial

import regex
import yake

import requests
import urllib3
from tqdm import tqdm

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def runQuery(query, index="wapo"):
    try_again = True
    try_again_counter = 0
    while try_again:
        headers = {"Content-Type": "application/json", "charset": "utf-8"}
        response = requests.get(url=f"https://localhost:9200/{index}/_search", data=json.dumps(query),
                                headers=headers, auth=('admin', 'admin'), verify=False)
        try_again = False
        if response.status_code == 200:
            json_output = json.loads(response.text)
            return json_output["hits"]["hits"]
        elif response.status_code == 429:
            try_again_counter += 1
            if try_again_counter < 2:
                print("\nError 429, sleeping a bit")
                time.sleep(60)
                try_again = True
            else:
                print(f"\n{response.status_code}")
                print(response.text)
                exit(1)
        else:
            print(f"\n{response.status_code}")
            print(response.text)
            exit(1)


def runQueryExplanation(query, document_id):
    try_again = True
    try_again_counter = 0
    while try_again:
        headers = {"Content-Type": "application/json", "charset": "utf-8"}
        response = requests.get(url=f"https://localhost:9200/wapo/_explain/{document_id}", data=json.dumps(query),
                                headers=headers, auth=('admin', 'admin'), verify=False)
        try_again = False
        if response.status_code == 200:
            json_output = json.loads(response.text)
            return json_output["explanation"]["value"]
        elif response.status_code == 429:
            try_again_counter += 1
            if try_again_counter < 2:
                print("\nError 429, sleeping a bit")
                time.sleep(60)
                try_again = True
            else:
                print(f"\n{response.status_code}")
                print(response.text)
                exit(1)
        else:
            print(f"\n{response.status_code}")
            print(response.text)
            exit(1)


def requestByID(topic_id, return_source=False, index="wapo"):
    query = {
        "query": {
            "ids": {
                "values": topic_id
            }
        }
    }
    document_info = runQuery(query, index=index)
    data = None
    if len(document_info) != 0:
        if return_source:
            data = document_info[0]
        else:
            data = document_info[0]["_source"]
    return data


def searchByVectorExactExplain(vector, search_in, explain_document):
    query = {
        "query": {
                "script_score": {
                    "query": {
                        "match_all": {}
                    },
                    "script": {
                        "source": "knn_score",
                        "lang": "knn",
                        "params": {
                            "field": f"{search_in}_vector",
                            "query_value": vector,
                            "space_type": "cosinesimil"
                        }
                    }
                }
        }
    }
    results = runQueryExplanation(query, explain_document)
    return results


def searchByVectorExact(vector, vector2, search_in, search_in2, limit_year, *text_values):
    query = {
        "size": 200,
        "query": {
            "boosting": {
                "positive": {
                    "bool": {
                        "should": [
                            {"script_score": {
                                        "query": {
                                            "script_score": {
                                                "query": {
                                                    "match_all": {}
                                                },
                                                "script": {
                                                    "source": " 1 + cosineSimilarity(params.query_value, doc[params.field])",
                                                    #"source": "knn_score",
                                                    #"lang": "knn",
                                                    "params": {
                                                        "field": f"{search_in}_vector",
                                                        "query_value": vector,
                                                        #"space_type": "cosinesimil"
                                                    }
                                                }
                                            }
                                        },
                                        "script": {
                                            "source": "_score < 1.0 ? 0.0 : 250 * (_score-1)"
                                        }
                                    }
                            },
                            {
                                #"match": {
                                #    "title": text_values[0]
                                #}
                                "query_string": {
                                    "query": text_values[0],
                                    "default_field": "title"
                                }
                            },
                            {
                                "query_string": {
                                    "query": text_values[1],
                                    "default_field": "body"
                                }
                            }
                        ],
                        "filter": {
                            "range": {
                                "date": {
                                    "lt": f"{limit_year}-01-01T00:00:00Z"
                                }
                            }
                        }
                    }

                },
                "negative": {
                    "bool": {
                        "should": [
                            {
                                "script_score": {
                                    "query": {
                                        "match_all": {}
                                    },
                                    "script": {
                                        "source": "knn_score",
                                        "lang": "knn",
                                        "params": {
                                            "field": f"{search_in2}_vector",
                                            "query_value": vector2,
                                            "space_type": "cosinesimil"
                                        }
                                    }
                                }

                            },
                            {
                                "terms": {
                                    "kicker": ["Opinions", "Opinion", "Letters to the Editor", "The Post's View", "Global Opinions", "All Opinions Are Local", "Local Opinions"]
                                }
                            }
                        ]
                    }

                },
                "negative_boost": 0.80
            }
        },
        "fields": [],
        "_source": False
    }
    return runQuery(query)

#"kicker": ["Opinions", "Opinion", "Letters to the Editor", "The Post's View", "Global Opinions", "All Opinions Are Local", "Local Opinions"]

# def processRecommendations(recommendations, original_id):
#     final_recommendations = []
#     final_scores = []
#     for recommendation in recommendations:
#         document_id = recommendation["_id"]
#         score = recommendation["_score"]
#         if document_id == original_id:
#             continue
#         final_recommendations.append(document_id)
#         final_scores.append(score)
#
#     return final_recommendations, final_scores


def processRecommendations(recommendations, original_id):
    final_recommendations = {}
    for recommendation in recommendations:
        document_id = recommendation["_id"]
        score = recommendation["_score"]
        if document_id == original_id:
            continue
        final_recommendations[document_id] = score

    return final_recommendations

def printPredictions(lambdamart_topic_scores, year, eval=False):

    with open(f"{results_path}{experiment}.txt", 'w') as output_file:
        for topic_id in lambdamart_topic_scores.keys():
            for recommendation_id, (document_id, score) in enumerate(
                    sorted(lambdamart_topic_scores[topic_id].items(), key=lambda item: item[1], reverse=True)):
                if recommendation_id >= 100:
                    break
                output_file.write(f"{topic_id}\tQ0\t{document_id}\t{recommendation_id}\t{score}\t{experiment}\n")

    if eval:
        ndcg = runEvaluation(year, f"{experiment}")
        ndcg_5 = runEvaluation(year, f"{experiment}", cut=5)
        ndcg_10 = runEvaluation(year, f"{experiment}", cut=10)
        print(f"& {ndcg} & {ndcg_5} & {ndcg_10} \\\\")

        ndcg = runEvaluation(year, f"{experiment}", median=True)
        ndcg_5 = runEvaluation(year, f"{experiment}", cut=5, median=True)
        ndcg_10 = runEvaluation(year, f"{experiment}", cut=10, median=True)
        print(f"Median")
        print(f"& {ndcg} & {ndcg_5} & {ndcg_10} \\\\")
        print("")



# def printRecommendations(file_pointer, topic_id, recommendations, scores, run_name):
#     for recommendation_id, (document_id, score) in enumerate(zip(recommendations, scores)):
#         file_pointer.write(f"{topic_id}\tQ0\t{document_id}\t{recommendation_id}\t{score}\t{run_name}\n")


def runEvaluation(year, experiment, cut=0, median=False):
    if cut == 0:
        metric = "ndcg"
    else:
        metric = f"ndcg_cut.{cut}"
    addtional_param = ""
    if median:
        addtional_param = "-q"
    results = subprocess.check_output(f"{trec_eval_path}trec_eval {addtional_param} -m {metric}"
                            f" {gs_path}{year}.txt"
                            f" {results_path}{experiment}.txt", shell=True,
                            universal_newlines=True)
    if median:
        final_results = []
        for topic_result in results.split("\n"):
            if topic_result == "":
                continue
            _, topic, score = topic_result.split("\t")
            if topic != "all":
                final_results.append(float(score))
        ndcg_result = statistics.median(final_results)
    else:
        ndcg_result = results.split("\t")[2].rstrip()
    return ndcg_result


def extractKewords(text, to_lower=True):
    if to_lower:
        text = text.lower()
    keywords = dict(keyword_extractor.extract_keywords(text))
    #processed_keywords = " ".join(keywords.keys())
    processed_keywords = {}
    for (keyword, score) in keywords.items():
        #score = 2-score
        score = -math.log(score)
        if score <= 0.0:
            continue
        processed_keywords[keyword] = score
    return processed_keywords


def keywordsToQuery(keywords):
    processed_keywords = []
    for (keyword, score) in keywords.items():
        # split_keywords = keyword.split(" ")
        processed_keywords.append(f"({keyword})^{score}")
        # processed_keywords.append(f"({' AND '.join(split_keywords)})^{score}")
    return " OR ".join(processed_keywords)


def searchKeywords(text, keywords):
    processed_keywords = []
    #tokens = dict(keyword_extractor.extract_keywords(text.lower()))
    tokens = regex.split("\s", text.lower())
    #for token in tokens.keys():
    for token in tokens:
        if token in keywords:
            processed_keywords.append(f"({token})^{keywords[token]}")
        #else:
        #    processed_keywords.append(f"({token})^{0.8}")
    return " OR ".join(processed_keywords)

class Rescorer:

    def __init__(self, topic_scores, year, field, printing_path, print_field=True):
        self.__topic_scores = topic_scores
        self.__year = year
        self.__field = field
        self.__new_scores = None
        self.__printing_path = printing_path
        self.__print_field = print_field
        self.__extra_vectors = None

    def retrieveNewScore(self, document_id, from_vector, shared_dict):
        shared_dict[document_id] = searchByVectorExactExplain(self.__extra_vectors[from_vector], "body", document_id) - 1

    def rescoreBy(self, save=False, load=False, cache_extension="K300"):
        if save:
            self.__new_scores = {}
            print("Retrieving addtional information")
            for topic_id in tqdm(self.__topic_scores.keys()):
                self.__extra_vectors = requestByID(f"{topic_id}", index=f"test_{self.__year}")[f"{self.__field}_vector"]
                if self.__field != "subtopic":
                    self.__extra_vectors = [self.__extra_vectors]
                self.__new_scores[topic_id] = []
                for subfield_id in range(len(self.__extra_vectors)):
                    self.__new_scores[topic_id].append({})
                    temporal_dict = multiprocessing.Manager().dict()
                    with multiprocessing.Pool(processes=10) as pool:
                        pool.map(partial(self.retrieveNewScore, shared_dict=temporal_dict, from_vector=subfield_id),
                                    self.__topic_scores[topic_id].keys())
                    for document_id in temporal_dict.keys():
                        self.__new_scores[topic_id][subfield_id][document_id] = temporal_dict[document_id]
            with open(f"./cache_recommendations_{self.__year}_{self.__field}{cache_extension}.pkl", 'wb') as saved_file:
                pickle.dump(self.__new_scores, saved_file, -1)
        if load:
            with open(f"./cache_recommendations_{self.__year}_{self.__field}{cache_extension}.pkl", 'rb') as saved_file:
                self.__new_scores = pickle.load(saved_file)

        self.__rescore()

        print("Printing scores")
        with open(f"{self.__printing_path}_{self.__field}.txt", 'w') as output_file:
            for topic_id in tqdm(self.__new_scores.keys()):
                for subfield_id, subfield in enumerate(self.__new_scores[topic_id]):
                    if self.__print_field:
                        extra_printing = f".{subfield_id}"
                    for recommendation_id, (document_id, score) in enumerate(
                            sorted(subfield.items(), key=lambda item: item[1], reverse=True)):
                        if recommendation_id >= 100:
                            break
                        output_file.write(
                            f"{topic_id}{extra_printing}\tQ0\t{document_id}\t{recommendation_id}\t{score}\t{experiment}\n")

    def __rescore(self):
        beta_squared = 2.25
        for topic_id in self.__new_scores.keys():
            ranking_topic_scores = {}
            for ranking, (document_id, _) in enumerate(sorted(self.__topic_scores[topic_id].items(), key=lambda item: item[1], reverse=True)):
                ranking_topic_scores[document_id] = 1/(ranking+1)

            for subfield_id, subfield in enumerate(self.__new_scores[topic_id]):

                for ranking, (document_id, _) in enumerate(sorted(subfield.items(), key=lambda item: item[1], reverse=True)):

                    rank_topic_score = ranking_topic_scores[document_id]
                    rank_new_score = 1/(ranking+1)
                    final_score = (1 + beta_squared) * (rank_topic_score * rank_new_score) / (
                                (beta_squared * rank_topic_score) + rank_new_score)
                    self.__new_scores[topic_id][subfield_id][document_id] = final_score

    def getNewScores(self):
        return self.__new_scores


trec_eval_path = "/home/lcabrera/Programs/trec_eval-9.0.7/"
gs_path = "/home/lcabrera/TREC_Results/GS/"
results_path = "/home/lcabrera/TREC_Results/Results/"

parser = argparse.ArgumentParser(description='Index WashingtonPost docs to ElasticSearch')
parser.add_argument('experiment')
parser.add_argument('vector_from')
parser.add_argument('vector_to')
parser.add_argument("year")
parser.add_argument("--rescore_by", default="")

args = parser.parse_args()

keyword_extractor = yake.KeywordExtractor(lan="en", n=1, top=300)

experiment = args.experiment
vector_from = args.vector_from
vector_to = args.vector_to
year = int(args.year)

with open(f"./test_{year}.pkl", 'rb') as test_file:
    test_documents = pickle.load(test_file)

topics_recommendations = {}
for topic_id in tqdm(test_documents):
    data_document = requestByID(test_documents[topic_id])
    #recommendations = searchByVectorApprox(data_document[f"{vector_from}_vector"], vector_to)
    document_keywords = extractKewords(data_document["title"]+"\n"+data_document["body"]+"\n"+data_document["captions"])
    document_keywords_query = keywordsToQuery(document_keywords)
    title_keywords_query = searchKeywords(data_document["title"], document_keywords)
    #title_keywords = extractKewords(data_document["title"]+"\n"+data_document["lead"])
    #title_keywords_query = keywordsToQuery(title_keywords)
    recommendations = searchByVectorExact(data_document[f"{vector_from}_vector"], data_document[f"title_vector"],  vector_to, "title", year,  title_keywords_query, document_keywords_query)
    #final_recommendations, scores = processRecommendations(recommendations, test_documents[topic_id])
    #printRecommendations(output_file, topic_id, final_recommendations, scores, experiment)
    topics_recommendations[topic_id] = processRecommendations(recommendations, test_documents[topic_id])
if year == 2020:
    printPredictions(topics_recommendations, year, eval=True)
else:
    results_path += "/Submit/"
    printPredictions(topics_recommendations, year)
if args.rescore_by != "":
    rescorer = Rescorer(topics_recommendations, year, args.rescore_by, f"/home/lcabrera/TREC_Results/Results/Submit/{experiment}_{args.rescore_by}.txt")
    rescorer.rescoreBy(save=True, cache_extension="_K300")

