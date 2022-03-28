import json
import regex
import torch
import requests
from sentence_transformers import SentenceTransformer
import logging
import sys

from tqdm import tqdm


def processContent(json_contents, json_title):
	title = ""
	text_paragraphs = []
	date = ""
	kicker = ""
	captions = []
	links = set()
	for json_element in json_contents:
		if json_element is not None and "type" in json_element:
			if json_element["type"] == "title" and "content" in json_element and json_element["content"] is not None:
				title = json_element["content"]
			elif json_element["type"] == "date" and "content" in json_element and json_element["content"] is not None:
				date = json_element["content"]
			elif json_element["type"] == "sanitized_html" and "content" in json_element and json_element["content"] is not None:
				paragraph = json_element["content"]
				links.update(regex.findall('href="([^"]*)"', paragraph))
				paragraph = regex.sub('<.*?>', ' ', paragraph)
				text_paragraphs.append(paragraph)
			elif json_element["type"] == "kicker" and "content" in json_element and json_element["content"] is not None:
				kicker = json_element["content"]
			elif json_element["type"] == "image" and "fullcaption" in json_element and json_element["fullcaption"] is not None:
				captions.append(json_element["fullcaption"])

	if title == "":
		if json_title is not None and json_title != "":
			title = json_title

	title = processText(title, join=False)
	body = processText(text_paragraphs)
	if len(text_paragraphs) > 0:
		lead = processText(text_paragraphs[0], join=False)
	else:
		lead = body
	captions = processText(captions)
	return title, date, kicker, body, lead, captions, list(links)


def processParsedText(parsed_text):
	sentences = []
	lemmas = []
	sentence = []
	sentence_lemma = []
	for parsed_line in parsed_text.split("\n"):
		if parsed_line.startswith("#") or parsed_line == "":
			if len(sentence) > 0:
				sentence = "".join(sentence)
				sentence = regex.sub("\s\s+", " ", sentence)
				sentence = regex.sub("(?:^ | $)", "", sentence)

				sentence_lemma = " ".join(sentence_lemma)
				sentence_lemma = regex.sub("\s\s+", " ", sentence_lemma)
				sentence_lemma = regex.sub("(?:^ | $)", "", sentence_lemma)

				sentences.append(sentence)
				lemmas.append(sentence_lemma)
				sentence = []
				sentence_lemma = []
		else:
			columns = parsed_line.split("\t")
			columns[1] = regex.sub("’", "'", columns[1])
			columns[1] = regex.sub("(?:“|”)", '"', columns[1])
			sentence.append(columns[1])
			if columns[9] != "SpaceAfter=No":
				sentence.append(" ")
			sentence_lemma.append(columns[2])

	if len(sentence) > 0:
		sentence = "".join(sentence)
		sentence = regex.sub("\s\s+", " ", sentence)
		sentence = regex.sub("(?:^ | $)", "", sentence)

		sentence_lemma = " ".join(sentence_lemma)
		sentence_lemma = regex.sub("\s\s+", " ", sentence_lemma)
		sentence_lemma = regex.sub("(?:^ | $)", "", sentence_lemma)

		sentences.append(sentence)
		lemmas.append(sentence_lemma)

	return sentences, "\n".join(lemmas)


def parseText(text):
	headers = {"Content-Type": "text/plain", "charset": "utf-8"}
	text = text.encode('utf-8')
	parsed_text = requests.post(f"http://localhost:{server_port}", data=text, headers=headers)
	if parsed_text.status_code == 200:
		sentences, lemmas = processParsedText(parsed_text.text)
	else:
		raise Exception("Error with the parser")
	return sentences, lemmas


def vectorize(sentences):
	embeddings_list = sbert.encode(sentences, convert_to_tensor=True, output_value="token_embeddings")
	tokens = 0
	document_embedding = None
	for embeddings in embeddings_list:
		sentence_embedding = torch.sum(embeddings, dim=0)
		if document_embedding is None:
			document_embedding = sentence_embedding
		else:
			document_embedding = document_embedding.add(sentence_embedding)
		tokens += embeddings.size(0)
	return document_embedding, tokens


def processText(text, join=True):
	if join:
		text = "\n".join(text)
	vector = None
	lemmatized_text = ""
	tokens = 0
	if text != "":
		sentences, lemmatized_text = parseText(text)
		text = "\n".join(sentences)
		vector, tokens = vectorize(sentences)
	elements = {"text": text, "vector": vector, "tokens": tokens, "lemma": lemmatized_text}
	return elements


def convertVector(*vectors, divide_by=0):
	vector_as_list = []
	final_vector = None
	for vector in vectors:
		if vector is None:
			continue
		if final_vector is None:
			final_vector = vector
		else:
			final_vector = final_vector.add(vector)
	if divide_by > 0 and final_vector is not None:
		final_vector = torch.div(final_vector, divide_by)
	if final_vector is not None:
		vector_as_list = final_vector.tolist()
	return vector_as_list


def processLine(file_line):
	json_line = json.loads(file_line)
	document_id = json_line["id"]
	if document_id in missing_documents:
		title, date, kicker, body, lead, captions, links = processContent(json_line["contents"], json_line["title"])
		article_type = ""
		article_source = ""
		if "type" in json_line and json_line["type"] is not None:
			article_type = json_line["type"]
		if "source" in json_line and json_line["source"] is not None:
			article_source = json_line["source"]
		document = {
			"type": article_type,
			"source": article_source,
			"url": json_line['article_url'],
			"title": title["text"],
			"title_vector": convertVector(title["vector"], divide_by=title["tokens"]),
			"title_lemma": title["lemma"],
			"date": date,
			"kicker": kicker,
			"body": body["text"],
			"body_vector": convertVector(body["vector"], divide_by=body["tokens"]),
			"body_lemma": body["lemma"],
			"lead": lead["text"],
			"lead_vector": convertVector(lead["vector"], divide_by=lead["tokens"]),
			"lead_lemma": lead["lemma"],
			"captions": captions["text"],
			"captions_vector": convertVector(captions["vector"], divide_by=captions["tokens"]),
			"captions_lemma": captions["lemma"],
			"title_lead_vector": convertVector(title["vector"], lead["vector"], divide_by=title["tokens"]+lead["tokens"]),
			"title_body_vector": convertVector(title["vector"], body["vector"], divide_by=title["tokens"] + body["tokens"]),
			"title_body_captions_vector": convertVector(title["vector"], body["vector"], captions["vector"], divide_by=title["tokens"] + body["tokens"] + captions["tokens"]),
			"links": links
		}
		return document_id, document
	else:
		return None, None


def readFile(file_path, output_file, start_from=0, stop_at=-1):
	if stop_at != -1 and start_from > stop_at:
		return
	if start_from > 728626:
		return
	with open(file_path) as file_reader:
		file_writer = open(f"{output_file}.missing", mode="a")
		for line_id, file_line in enumerate(tqdm(file_reader, total=728626)):
			if line_id + 1 < start_from:
				continue
			try:
				document_id, document = processLine(file_line)
				if document_id is not None:
					file_writer.write(f"{document_id}\t{json.dumps(document)}\n")
			except Exception as e:
				logging.exception(f"Error while processing line {line_id} (start from zero)\n")
				file_writer.close()
				raise e
			if line_id + 1 == stop_at:
				break
		file_writer.close()

missing_documents = ["9171debc316e5e2782e0d2404ca7d09d", "cad56e871cd0bca6cc77e97ffe246258"]
arguments = sys.argv[1].split("_")
server_port = sys.argv[2]
sbert = SentenceTransformer('stsb-mpnet-base-v2')
#logging.basicConfig(filename=f"/data/lcabrera/WashingtonPost.v4/data/wp_v4.log", level=logging.INFO)
#readFile("/data/lcabrera/WashingtonPost.v4/data/TREC_Washington_Post_collection.v4.jl", "/data/lcabrera/WashingtonPost.v4/data/wp_processed", start_from=2993, stop_at=50000)
logging.basicConfig(filename=f"/data/lcabrera/WashingtonPost.v4/data/wp_v4_missing_{arguments[0]}.log", level=logging.INFO)
readFile("/data/lcabrera/WashingtonPost.v4/data/TREC_Washington_Post_collection.v4.jl", f"/data/lcabrera/WashingtonPost.v4/data/wp_processed_{arguments[0]}")
