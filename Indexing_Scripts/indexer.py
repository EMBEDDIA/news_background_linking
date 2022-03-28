#Fork from the code provided by TREC
import logging
import subprocess
import time
from pathlib import Path

import regex
from elasticsearch import helpers
from elasticsearch import Elasticsearch, TransportError
import argparse
import json
import sys

from elasticsearch.helpers.errors import BulkIndexError
from tqdm import tqdm
import copy
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def doc_generator(f, num_docs, index_name ):
    for line in tqdm(f, total=num_docs):
        document_id, js = line.split("\t")
        js = json.loads(js)
        change_name = {}
        for key in js.keys():
            if regex.search("_vector$", key) and len(js[key]) == 0:
                    js[key] = [0.0] * 768
            if regex.search("titie", key):
                change_name[key] = regex.sub("titie", "title", key)
        if "date" in js and (js["date"] is None or js["date"] == ""):
            js["date"] = "2000-01-01T00:00:00Z"
        for key in change_name:
            js[change_name[key]] = js.pop(key)
        if not args.flat_subtopics:
            data_dict = {
                "_index": index_name,
                "_type": '_doc',
                "_id": document_id,
                "_source": js
            }
            if args.only_create:
                data_dict["_op_type"] = "create"
            yield data_dict
        else:
            for subtopic_id in range(len(js["subtopic"])):
                new_js = copy.deepcopy(js)
                new_js["subtopic"] = new_js["subtopic"][subtopic_id]
                new_js["subtopic_vector"] = new_js["subtopic_vector"][subtopic_id]
                new_js["subtopic_lemma"] = new_js["subtopic_lemma"][subtopic_id]
                new_js["title_subtopic_vector"] = new_js["title_subtopic_vector"][subtopic_id]
                new_js["title_desc_subtopic_vector"] = new_js["title_desc_subtopic_vector"][subtopic_id]
                new_js["title_narr_subtopic_vector"] = new_js["title_narr_subtopic_vector"][subtopic_id]
                new_js["title_desc_narr_subtopic_vector"] = new_js["title_desc_narr_subtopic_vector"][subtopic_id]
                data_dict = {
                    "_index": index_name,
                    "_type": '_doc',
                    "_id": f"{document_id}.{subtopic_id}",
                    "_source": new_js
                }
                yield data_dict


def countDocuments(file_path):
    print("Counting documents")
    with open(file_path, 'r') as f:
        total_lines = 0
        for _ in f:
            total_lines += 1
    return total_lines


def createIndex(index_name, settings_path):
    if es.indices.exists(index=index_name):
        es.indices.delete(index=index_name)
    try:
        with open(settings_path, 'r') as f:
            settings_file = ""
            for line in f:
                settings_file += f"{line}\n"
        settings = json.loads(settings_file)
        es.indices.create(index=index_name, body=settings)
    except TransportError as e:
        print(e.info)
        sys.exit(-1)


parser = argparse.ArgumentParser(description='Index WashingtonPost docs to ElasticSearch')
parser.add_argument('bundle', help='WaPo bundle to index')
parser.add_argument('--host', default='localhost', help='Host for ElasticSearch endpoint')
parser.add_argument('--port', default='9200', help='Port for ElasticSearch endpoint')
parser.add_argument('--index_name', default='wapo', help='index name')
parser.add_argument('--create', action='store_true')
parser.add_argument('--settings', default="settings.json", help="settings json file")
parser.add_argument("--extension", default="lacd", help="Extension of the files to index")
parser.add_argument('--flat_subtopics', action='store_true')
parser.add_argument('--start_from', default=0)
parser.add_argument("--only_create", action="store_true")

args = parser.parse_args()

es = Elasticsearch(hosts=[{"host": args.host, "port": args.port}],
                   retry_on_timeout=True, max_retries=10,
                   http_auth=("admin", "admin"),
                   use_ssl=True, verify_certs=False)

logging.basicConfig(filename=f"/data/lcabrera/WashingtonPost.v4/data/{args.index_name}.log", level=logging.INFO)

es_tracer = logging.getLogger('elasticsearch')
es_tracer.setLevel(logging.WARNING)

if args.create: #or not es.indices.exists(index=args.index_name):
    createIndex(args.index_name, args.settings)

start_from = int(args.start_from)
for file_id, file_path in enumerate(sorted(Path(args.bundle).glob(f"*{args.extension}"))):
    if file_id < start_from:
        continue
    print(f"Processing file {file_id}: {file_path}")
    logging.info(f"Processing: {file_id}\t{file_path}")
    #total_lines = countDocuments(file_path)
    total_lines = subprocess.check_output(f"wc -l {file_path}", shell=True, universal_newlines=True)
    total_lines = int(total_lines.split(" ")[0])
    retry = True
    counter_retry = 0
    while retry:
        with open(file_path, 'r') as f:
            print("Indexing")
            try:
                #helpers.bulk(es, doc_generator(f, total_lines, args.index_name), request_timeout=30)
                for success, info in helpers.parallel_bulk(es, doc_generator(f, total_lines, args.index_name), request_timeout=30, thread_count=2, queue_size=200):
                    pass
                retry = False
            except TransportError as e:
                logging.error(f"Error while indexing {file_id}\t{file_path}: \n {e}")
                if counter_retry == 0:
                    retry = True
                    counter_retry += 1
                    logging.info(f"Waiting for 60")
                    time.sleep(60)
                else:
                    logging.error(f"Not solved after 60s")
                    retry = False
            except BulkIndexError as e:
                logging.warning(f"Some documents exist already")
                retry = False


es.indices.put_settings(index=args.index_name,
                        body={'index': {'refresh_interval': '1s',
                                        'number_of_replicas': '1',
                                        }})
