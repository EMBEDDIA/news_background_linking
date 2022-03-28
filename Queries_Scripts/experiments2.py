import argparse
import datetime
import json
import math
import multiprocessing
import pickle
import statistics
import subprocess
import time
from functools import partial
from pathlib import Path

import regex
import yake

import requests
import urllib3
from tqdm import tqdm

from LambdaInterpolation import LambdaInterpolation
from LambdaMart import LambdaMart

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
            return json_output["hits"]["hits"], json_output["hits"]["max_score"]
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
    document_info, _ = runQuery(query, index=index)
    data = None
    if len(document_info) != 0:
        if return_source:
            data = document_info[0]
        else:
            data = document_info[0]["_source"]
    return data


def searchByVectorApprox(vector, search_in):
    query = {
                "size": 101,
                "query": {
                    "knn": {
                        f"{search_in}_vector": {
                            "vector": vector,
                            "k": 101
                        }
                    }
                },
                "fields": [
                    "title",
                    "lead",
                    "type",
                    "source"
                ],
                "_source": False
            }
    return runQuery(query)


def searchByKeywords(keywords, field, year, explain_document=None):
    query = {
        "query": {
            "bool": {
                "should": {
                    "query_string": {
                        "query": keywords,
                        "default_field": field
                    }
                 },
                "must_not": {
                    "terms": {
                        "kicker": ["Opinions", "Opinion", "Letters to the Editor", "The Post's View", "Global Opinions",
                                   "All Opinions Are Local", "Local Opinions", "Editorial Board"]
                    }
                },
                "filter": {
                    "range": {
                        "date": {
                            "lt": f"{year}-01-01T00:00:00Z"
                        }
                    }
                }
            }
        }
    }
    if explain_document is not None:
        results = runQueryExplanation(query, explain_document)
    else:
        query["size"] = 101
        hits, max_score = runQuery(query)
        results = {"type": "keywords", "recommendations": hits, "max_score": max_score}
    return results


def searchByVectorExact(vector, search_in, year, explain_document=None):
    query = {
        "query": {
            "bool": {
                "should": {
                    #"script_score": {
                        #"query": {
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
                        #},
                        #"script": {
                        #    "source": f"_score < 1.0 ? 0.0 : {weight} (_score-1)"
                        #}
                    #}
                },
                "must_not": {
                    "terms": {
                        "kicker": ["Opinions", "Opinion", "Letters to the Editor", "The Post's View", "Global Opinions",
                                   "All Opinions Are Local", "Local Opinions", "Editorial Board"]
                    }
                },
                "filter": {
                    "range": {
                        "date": {
                            "lt": f"{year}-01-01T00:00:00Z"
                        }
                    }
                }
            }

        }
    }
    if explain_document is not None:
        results = runQueryExplanation(query, explain_document)
    else:
        query["size"] = 101
        hits, max_score = runQuery(query)
        results = {"type": "keywords", "recommendations": hits, "max_score": max_score}
    return results


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
    tokens = dict(keyword_extractor.extract_keywords(text.lower()))
    #tokens = regex.split("\s", text.lower())
    for token in tokens.keys():
    #for token in tokens:
        if token in keywords:
            processed_keywords.append(f"({token})^{keywords[token]}")
        #else:
        #    processed_keywords.append(f"({token})^{0.8}")
    return " OR ".join(processed_keywords)


class Recommender:

    def __init__(self, gs_path):
        self.__kicker_dictionary = {"": -10.0}
        self.__check_emptiness = {"title": [0, 2, 5, 7, 8, 13], "body": [1, 3, 4, 6, 9, 10, 12, 14], "captions": [11, 15]}
        self.__gs_path = gs_path

    def setKickerDictionary(self, kicker_dictionary):
        self.__kicker_dictionary = kicker_dictionary

    def getOpinionKicker(self):
        return self.__kicker_dictionary["Opinion"]

    def getRecommendations(self, year, max_year, training=False):
        topic_joint_recommendations = {}
        print(f"\nProcessing year: {year}")
        with open(f"./test_{year}.pkl", 'rb') as test_file:
            test_documents = pickle.load(test_file)
        known_recommendations = {}
        if training:
            with open(f"{self.__gs_path}/{year}.txt") as file_reader:
                for file_line in file_reader:
                    file_line = regex.sub("\n|\r", "", file_line)
                    data = regex.split("\s+", file_line)
                    data[0] = int(data[0])
                    if year not in known_recommendations:
                        known_recommendations[year] = {}
                    if data[0] not in known_recommendations[year]:
                        known_recommendations[year][data[0]] = {}
                    known_recommendations[year][data[0]][data[2]] = 1

        no_captions = 0
        no_intersected_documents = 0
        for topic_number, topic_id in enumerate(tqdm(test_documents)):
            #if topic_number < 2:
            #    continue
            data_document = requestByID(test_documents[topic_id])
            if data_document is None:
                print(f"\nMissing topic: {topic_id}")
                continue
            document_keywords = extractKewords(
                data_document["title"] + "\n" + data_document["body"] + "\n" + data_document["captions"])
            document_keywords_query = keywordsToQuery(document_keywords)
            title_keywords_query = searchKeywords(data_document["title"], document_keywords)
            lead_keywords_query = searchKeywords(data_document["lead"], document_keywords)
            captions_keywords_query = searchKeywords(data_document["captions"], document_keywords)
            recommendations = {}
            recommendations["kw_title"] = searchByKeywords(title_keywords_query, "title", max_year)                             # 0
            recommendations["kw_body"] = searchByKeywords(document_keywords_query, "body", max_year)                            # 1
            recommendations["kw_lead2title"] = searchByKeywords(lead_keywords_query, "title", max_year)                         # 2
            recommendations["kw_lead2body"] = searchByKeywords(lead_keywords_query, "body", max_year)                           # 3
            recommendations[f"body2body"] = searchByVectorExact(data_document[f"body_vector"], "body", max_year)                # 4
            recommendations[f"title2title"] = searchByVectorExact(data_document[f"title_vector"], "title", max_year)            # 5
            recommendations[f"lead2body"] = searchByVectorExact(data_document[f"lead_vector"], "body", max_year)                # 6
            recommendations[f"lead2title"] = searchByVectorExact(data_document[f"lead_vector"], "title", max_year)              # 7
            if len(data_document["captions"]) > 0:
                if len(captions_keywords_query) > 0:
                    recommendations["kw_captions2title"] = searchByKeywords(captions_keywords_query, "title", max_year)         # 8
                    recommendations["kw_captions2body"] = searchByKeywords(captions_keywords_query, "body", max_year)           # 9
                    recommendations["kw_captions2lead"] = searchByKeywords(captions_keywords_query, "lead", max_year)           # 10
                    recommendations["kw_captions2captions"] = searchByKeywords(captions_keywords_query, "captions", max_year)   # 11
                else:
                    recommendations["kw_captions2title"] = None
                    recommendations["kw_captions2body"] = None
                    recommendations["kw_captions2lead"] = None
                    recommendations["kw_captions2captions"] = None

                recommendations[f"captions2body"] = searchByVectorExact(data_document[f"captions_vector"], "body", max_year)            # 12
                recommendations[f"captions2title"] = searchByVectorExact(data_document[f"captions_vector"], "title", max_year)          # 13
                recommendations[f"captions2lead"] = searchByVectorExact(data_document[f"captions_vector"], "lead", max_year)            # 14
                recommendations[f"captions2captions"] = searchByVectorExact(data_document[f"captions_vector"], "captions", max_year)    # 15
            else:
                no_captions += 1
                recommendations["kw_captions2title"] = None
                recommendations["kw_captions2body"] = None
                recommendations["kw_captions2lead"] = None
                recommendations["kw_captions2captions"] = None
                recommendations[f"captions2body"] = None
                recommendations[f"captions2title"] = None
                recommendations[f"captions2lead"] = None
                recommendations[f"captions2captions"] = None

            #original_date = data_document["date"]
            #if original_date != "2000-01-01T00:00:00Z":
            #    original_date = datetime.datetime.strptime(original_date, "%Y-%m-%dT%H:%M:%SZ")
            #else:
            #    original_date = None

            joint_recommendations, max_scores = self.__extractRecommendations(test_documents[topic_id],
                                                                  data_document["kicker"],
                                                                  *recommendations.values())

            if training:
                number_scores = len(max_scores) + 3
                for document_id in known_recommendations[year][topic_id].keys():
                    if document_id == test_documents[topic_id]:
                        continue
                    if document_id not in joint_recommendations:
                        additional_recommendation = requestByID(document_id, return_source=True)
                        if additional_recommendation is None:
                            print(f"\nDocument ID {document_id} not found in ES")
                            continue
                        if additional_recommendation["_source"]["kicker"] in ["Opinions", "Opinion", "Letters to the Editor",
                                                                   "The Post's View", "Global Opinions",
                                                   "All Opinions Are Local", "Local Opinions", "Editorial Board"]:
                            continue
                        no_intersected_documents += 1
                        joint_recommendations[document_id] = self.__createJointRecommendation(additional_recommendation,
                                                                                              data_document["kicker"],
                                                                                              number_scores)

            shared_joint_recommendations = multiprocessing.Manager().dict(joint_recommendations)

            if training:
                expander = self.Expander(document_keywords_query, title_keywords_query,
                                             lead_keywords_query, captions_keywords_query,
                                             data_document, max_year,
                                             max_scores,
                                             known_recommendations=known_recommendations[year][topic_id],
                                             opinion_kicker=self.getOpinionKicker())

            else:
                expander = self.Expander(document_keywords_query, title_keywords_query,
                                         lead_keywords_query, captions_keywords_query,
                                         data_document, max_year,
                                         max_scores)

            with multiprocessing.Pool(processes=10) as pool:
                #with tqdm(total=len(joint_recommendations), position=1, leave=False) as progress:
                #    for i in pool.imap(partial(expander.expandRecommendations,
                #                            shared_joint_recommendations=shared_joint_recommendations),
                #                            joint_recommendations.keys()):
                #        progress.update()
                pool.map(partial(expander.expandRecommendations,
                                 shared_joint_recommendations=shared_joint_recommendations),
                                 joint_recommendations.keys())

            for key, value in shared_joint_recommendations.items():
                joint_recommendations[key] = value

            #for document_id in tqdm(joint_recommendations.keys(), leave=False):
            #    expander.expandRecommendations(document_id, joint_recommendations)



            topic_joint_recommendations[topic_id] = joint_recommendations

        if training:
            print(f"{no_intersected_documents} documents were added to joint_recommendation")
        print(f"\nNo captions in {no_captions} topics")
        return topic_joint_recommendations

    class Expander:

        def __init__(self, document_keywords_query,
                     title_keywords_query,
                     lead_keywords_query, captions_keywords_query,
                     data_document, max_year,
                     max_scores,
                     known_recommendations=None,
                     opinion_kicker=None):
            self.__document_keywords_query = document_keywords_query
            self.__title_keywords_query = title_keywords_query
            self.__lead_keywords_query = lead_keywords_query
            self.__captions_keywords_query = captions_keywords_query
            self.__data_document = data_document
            self.__max_year = max_year
            self.__max_scores = max_scores
            self.__known_recommendations = known_recommendations
            self.__opinion_kicker = opinion_kicker

        def expandRecommendations(self, document_id, shared_joint_recommendations):
            #print(document_id)
            joint_recommendations = shared_joint_recommendations[document_id]
            #if self.__known_recommendations is not None and document_id not in self.__known_recommendations:
            #    return
            expanded_scores = [-10.0] * len(self.__max_scores) #We only care about those that we have actual scores
            for field_id in range(len(self.__max_scores)):
                if joint_recommendations[field_id] != -10.0:
                    continue
                if field_id == 0 and self.__max_scores[field_id] > 0:
                    expanded_scores[field_id] = searchByKeywords(self.__title_keywords_query, "title", self.__max_year, explain_document=document_id)  # 0
                elif field_id == 1 and self.__max_scores[field_id] > 0:
                    expanded_scores[field_id] = searchByKeywords(self.__document_keywords_query, "body", self.__max_year, explain_document=document_id)  # 1
                elif field_id == 2 and self.__max_scores[field_id] > 0:
                    expanded_scores[field_id] = searchByKeywords(self.__lead_keywords_query, "title", self.__max_year, explain_document=document_id)  # 2
                elif field_id == 3 and self.__max_scores[field_id] > 0:
                    expanded_scores[field_id] = searchByKeywords(self.__lead_keywords_query, "body", self.__max_year, explain_document=document_id)  # 3
                elif field_id == 4 and self.__max_scores[field_id] > 0:
                    expanded_scores[field_id] = searchByVectorExact(self.__data_document[f"body_vector"], "body", self.__max_year, explain_document=document_id)  # 4
                elif field_id == 5 and self.__max_scores[field_id] > 0:
                    expanded_scores[field_id] = searchByVectorExact(self.__data_document[f"title_vector"], "title", self.__max_year, explain_document=document_id)  # 5
                elif field_id == 6 and self.__max_scores[field_id] > 0:
                    expanded_scores[field_id] = searchByVectorExact(self.__data_document[f"lead_vector"], "body", self.__max_year, explain_document=document_id)  # 6
                elif field_id == 7 and self.__max_scores[field_id] > 0:
                    expanded_scores[field_id] = searchByVectorExact(self.__data_document[f"lead_vector"], "title", self.__max_year, explain_document=document_id)  # 7
                elif len(self.__data_document["captions"]) > 0:
                    if len(self.__captions_keywords_query) > 0 and field_id < 12:
                        if field_id == 8 and self.__max_scores[field_id] > 0:
                            expanded_scores[field_id] = searchByKeywords(self.__captions_keywords_query, "title", self.__max_year, explain_document=document_id)  # 8
                        elif field_id == 9 and self.__max_scores[field_id] > 0:
                            expanded_scores[field_id] = searchByKeywords(self.__captions_keywords_query, "body", self.__max_year, explain_document=document_id)  # 9
                        elif field_id == 10 and self.__max_scores[field_id] > 0:
                            expanded_scores[field_id] = searchByKeywords(self.__captions_keywords_query, "lead", self.__max_year, explain_document=document_id)  # 10
                        elif field_id == 11 and self.__max_scores[field_id] > 0:
                            expanded_scores[field_id] = searchByKeywords(self.__captions_keywords_query, "captions", self.__max_year, explain_document=document_id)  # 11
                    elif field_id > 11:
                        if field_id == 12 and self.__max_scores[field_id] > 0:
                            expanded_scores[field_id] = searchByVectorExact(self.__data_document[f"captions_vector"], "body", self.__max_year, explain_document=document_id)  # 12
                        elif field_id == 13 and self.__max_scores[field_id] > 0:
                            expanded_scores[field_id] = searchByVectorExact(self.__data_document[f"captions_vector"], "title", self.__max_year, explain_document=document_id)  # 13
                        elif field_id == 14 and self.__max_scores[field_id] > 0:
                            expanded_scores[field_id] = searchByVectorExact(self.__data_document[f"captions_vector"], "lead", self.__max_year, explain_document=document_id)  # 14
                        elif field_id == 15 and self.__max_scores[field_id] > 0:
                            expanded_scores[field_id] = searchByVectorExact(self.__data_document[f"captions_vector"], "captions", self.__max_year, explain_document=document_id)  # 15

            new_scores = []
            for field_id, score in enumerate(expanded_scores):
                if score != -10.0:
                    if field_id in [4, 5, 6, 7, 12, 13, 14, 15]:
                        score -= 1
                    else:
                        score /= self.__max_scores[field_id]
                    #local_joint_recommendations[field_id] = score
                    new_scores.append(score)
                elif self.__max_scores[field_id] == 0.0:
                    #local_joint_recommendations[field_id] = 0.0
                    new_scores.append(0.0)
                else:
                    new_scores.append(joint_recommendations[field_id])

            for field_id in range(len(expanded_scores), len(joint_recommendations)):
                new_scores.append(joint_recommendations[field_id])

            shared_joint_recommendations[document_id] = new_scores
            #return joint_recommendations

    def __createJointRecommendation(self, recommendation, original_kicker, number_scores):
        document_recommended = [-10.0] * number_scores
        # Kicker
        # document_kicker_id = -1
        # original_kicker_id = -1
        #if recommendation["_source"]["kicker"] in ["Opinions", "Opinion", "Letters to the Editor", "The Post's View", "Global Opinions",
        #                           "All Opinions Are Local", "Local Opinions", "Editorial Board"]:
        #
        #     document_kicker_id = self.__kicker_dictionary[recommendation["_source"]["kicker"]]
        # if original_kicker in self.__kicker_dictionary:
        #     original_kicker_id = self.__kicker_dictionary[original_kicker]
        #
        # if document_kicker_id == self.getOpinionKicker():
        #     document_recommended[-1] = 1
        # elif document_kicker_id != -10:
        #     document_recommended[-1] = 0
        #
        # if document_kicker_id == original_kicker_id:
        #     document_recommended[-2] = 1
        # else:
        #     document_recommended[-2] = 0


        for field in self.__check_emptiness.keys():
            if len(recommendation["_source"][field]) == 0:
                if field == "title":
                    document_recommended[-1] = 0
                elif field == "body":
                    document_recommended[-2] = 0
                elif field == "captions":
                    document_recommended[-3] = 0

                for field_position in self.__check_emptiness[field]:
                    document_recommended[field_position] = 0
            else:
                if field == "title":
                    document_recommended[-1] = 1
                elif field == "body":
                    document_recommended[-2] = 1
                elif field == "captions":
                    document_recommended[-3] = 1

        return document_recommended

    def __extractRecommendations(self, original_id, original_kicker, *recommendations_collection):

        joint_recommendations = {}
        number_collection = len(recommendations_collection)
        number_scores = number_collection + 3
        max_scores = []
        for collection_id, collection in enumerate(recommendations_collection):
            if collection is None:
                max_scores.append(-1.0)
                continue
            collection_type = collection["type"]
            recommendations = collection["recommendations"]
            max_score = collection["max_score"]
            max_scores.append(max_score)
            if max_score == 0.0 and collection_type != "vector":
                print(f"\nScore recommendation zero: {collection_id}")
                continue
            if collection_type == "vector":
                for recommendation in recommendations:
                    recommendation["_score"] -= 1
            else:
                for recommendation in recommendations:
                    recommendation["_score"] /= max_score
            for recommendation in recommendations:
                document_id = recommendation["_id"]
                score = recommendation["_score"]
                if document_id == original_id:
                    continue
                if document_id not in joint_recommendations:
                    joint_recommendations[document_id] = self.__createJointRecommendation(recommendation, original_kicker, number_scores)
                joint_recommendations[document_id][collection_id] = score

        return joint_recommendations, max_scores


def trainLambdaMart(recommender, lambdamart, previous_years, max_previous_years, print_plot=False, extended_gs=False, maximize=None):
    topic_year_collection = {}
    for previous_year, max_previous_year in zip(previous_years.keys(), max_previous_years):
        if previous_years[previous_year]:
            topic_year_collection[previous_year] = recommender.getRecommendations(previous_year, max_previous_year, training=True)
            with open(f"./cache_recommendations_{previous_year}.pkl", 'wb') as test_file:
                pickle.dump(topic_year_collection[previous_year], test_file, -1)
        else:
            with open(f"./cache_recommendations_{previous_year}.pkl", 'rb') as test_file:
                topic_year_collection[previous_year] = pickle.load(test_file)

    lambdamart.train(topic_year_collection, gs_path, previous_years, recommender.getOpinionKicker(), print_plot=print_plot, extended_gs=extended_gs,
                     maximize=maximize)


def processYear(recommender, year, max_year, load=False, save=True):
    if load:
        with open(f"./cache_recommendations_test_{year}.pkl", 'rb') as test_file:
            current_year_recommendations = pickle.load(test_file)
    else:
        current_year_recommendations = recommender.getRecommendations(year, max_year)
        if save:
            with open(f"./cache_recommendations_test_{year}.pkl", 'wb') as test_file:
                pickle.dump(current_year_recommendations, test_file, -1)
    return current_year_recommendations


def printPredictions(lambdamart_topic_scores, year, filter_by_gs=False, eval=False):
    known_recommendations = {}
    files_extra_name = [""]
    if filter_by_gs:
        files_extra_name.append("_filtered")
        with open(f"{gs_path}/{year}.txt") as file_reader:
            for file_line in file_reader:
                file_line = regex.sub("\n|\r", "", file_line)
                data = regex.split("\s+", file_line)
                known_recommendations[f"{year}_{data[0]}_{data[2]}"] = 1
    for extra_name in files_extra_name:
        with open(f"{results_path}{experiment}{extra_name}.txt", 'w') as output_file:
            for topic_id in lambdamart_topic_scores.keys():
                unknown_skipped = 0
                for recommendation_id, (document_id, score) in enumerate(
                        sorted(lambdamart_topic_scores[topic_id].items(), key=lambda item: item[1], reverse=True)):
                    if extra_name == "_filtered" and f"{year}_{topic_id}_{document_id}" not in known_recommendations:
                        unknown_skipped += 1
                        continue
                    recommendation_id -= unknown_skipped
                    if recommendation_id >= 100:
                        break
                    output_file.write(f"{topic_id}\tQ0\t{document_id}\t{recommendation_id}\t{score}\t{experiment}\n")

        if eval:
            ndcg = runEvaluation(year, f"{experiment}{extra_name}")
            ndcg_5 = runEvaluation(year, f"{experiment}{extra_name}", cut=5)
            ndcg_10 = runEvaluation(year, f"{experiment}{extra_name}", cut=10)
            print(f"{extra_name}")
            print(f"& {ndcg} & {ndcg_5} & {ndcg_10} \\\\")

            ndcg = runEvaluation(year, f"{experiment}{extra_name}", median=True)
            ndcg_5 = runEvaluation(year, f"{experiment}{extra_name}", cut=5, median=True)
            ndcg_10 = runEvaluation(year, f"{experiment}{extra_name}", cut=10, median=True)
            print(f"Median {extra_name}")
            print(f"& {ndcg} & {ndcg_5} & {ndcg_10} \\\\")
            print("")


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
        shared_dict[document_id] = searchByVectorExact(self.__extra_vectors[from_vector], "body", year, explain_document=document_id) - 1

    def rescoreBy(self, save=False, load=False, cache_extension=""):
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
        extra_printing = ""
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
                            f"{topic_id}{extra_printing}\tQ0\t{document_id}\t{recommendation_id}\t{score}\t{experiment}_{self.__field}\n")

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


def readKickers(kicker_path):
    kickers = {"": -10.0}
    for kicker_id, full_path in enumerate(sorted(Path(kicker_path).glob("*.txt"))):
        with open(full_path, 'r') as input_file:
            for line in input_file:
                line = regex.sub("\n|\r", "", line)
                if line == "":
                    continue
                kickers[line] = kicker_id
    return kickers


trec_eval_path = "/home/lcabrera/Programs/trec_eval-9.0.7/"
gs_path = "/home/lcabrera/TREC_Results/GS/"
results_path = "/home/lcabrera/TREC_Results/Results/"
kicker_path = "./Kickers"

parser = argparse.ArgumentParser(description='Index WashingtonPost docs to ElasticSearch')
parser.add_argument('experiment')
parser.add_argument('vector_from')
parser.add_argument('vector_to')
parser.add_argument("year")
parser.add_argument("--load_model", default="")
parser.add_argument("--save_model", default="")
parser.add_argument("--rescore_by", default="")

args = parser.parse_args()

keyword_extractor = yake.KeywordExtractor(lan="en", n=1, top=300)

recommender = Recommender(gs_path)
recommender.setKickerDictionary(readKickers(kicker_path))

experiment = args.experiment
vector_from = args.vector_from
vector_to = args.vector_to
year = int(args.year)


#lambdamart = LambdaMart()
lambdamart = LambdaInterpolation(input_size=8, skip_columns=[8,9,10,11,12,13,14,15,16,17,18], a0=False)
#lambdamart = LambdaInterpolation(input_size=16, skip_columns=[16,17,18])

if args.load_model != "":
    lambdamart.loadModel(args.load_model)
else:
    if year == 2020:
        training_years = {2018: False, 2019: False}
        limit_years = [2018, 2018]
    else:
        training_years = {2018: False, 2019: False, 2020: False}
        limit_years = [2018, 2018, 2020]

    trainLambdaMart(recommender, lambdamart, training_years, limit_years, print_plot=True, extended_gs=True,
                        maximize=[-2.9960321254867717, 9.731360260770366, 0.23285254487602458, 1.2961047475432395, 9.905579302201307, 0.27990005570771714, -2.4215895340988247, -0.5715339058266125])
    lambdamart.saveModel(args.save_model)

if year != 2021:
    current_year_recommendations = processYear(recommender, year, year-1, load=True)
else:
    current_year_recommendations = processYear(recommender, year, 2022, load=True)

lambdamart_topic_scores = lambdamart.predict(current_year_recommendations)

if args.rescore_by != "":

    if args.rescore_by == "narr":
        rescorer = Rescorer(lambdamart_topic_scores, year, args.rescore_by,
                            f"/home/lcabrera/TREC_Results/Results/Submit/{experiment}", print_field=False)
    else:
        rescorer = Rescorer(lambdamart_topic_scores, year, args.rescore_by,
                            f"/home/lcabrera/TREC_Results/Results/Submit/{experiment}")
    rescorer.rescoreBy(load=True)
else:
    if year == 2020:
        printPredictions(lambdamart_topic_scores, year, filter_by_gs=True, eval=True)
    else:
        printPredictions(lambdamart_topic_scores, year)
