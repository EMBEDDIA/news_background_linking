import math
import statistics
import string
import subprocess

import regex
from bayes_opt import BayesianOptimization
from scipy.stats.mstats import mquantiles


class Optimization:

    def __init__(self, input_size, training_data, topics_without_relevant, cut=10, a0=False):
        self.__training_data = training_data
        self.__best_score = [-1]
        self.__best_central = [-1]
        self.__best_deviation = [-1]
        self.__best_param = {}
        self.__cut = cut
        self.__input_size = input_size
        self.__topics_without_relevant = topics_without_relevant
        self.__trec_eval_path = "/home/lcabrera/Programs/trec_eval-9.0.7/"
        self.__gs_path = "/home/lcabrera/TREC_Results/GS/"
        self.__results_path = "/home/lcabrera/TREC_Results/Results/"
        self.__a0 = a0

    def bayesianOptimization(self):
        param_bound = {}
        letters = list(string.ascii_lowercase)
        a0 = 0
        if self.__a0:
            a0 = 1
        for alpha in range(self.__input_size + a0):
            param_bound[f"{letters[alpha]}"] = (-10.0, 10.0)
        optimizer = BayesianOptimization(self.__objectiveFunction, param_bound, random_state=12)
        #Practical scorer
        if self.__a0:

            optimizer.probe(
                params=[4.94, 10.0, 0.3628, 2.474, 4.026, 2.932, 1.258, 4.978, 3.698],
                lazy=True
            )
            optimizer.probe(
                params=[5.135, 8.275, -6.338, 6.127, 10.00, 5.784, -2.341, 2.829, -6.75],
                lazy=True
            )
            optimizer.probe(
                params=[3.774, 10.00, -2.45, 3.239, 10.00, 2.876, .5584, 1.285, -10.00],
                lazy=True
            )
            optimizer.probe(
                params=[3.938155680897926, 10.00, -2.6081353266331007, 4.377092546084274, 7.454459553303228, 3.785366550487547, -1.5091437018196089, -0.09985499856494899, 6.7121259581523],
                lazy=True
            )
            optimizer.probe(
                params=[7.612, 8.87, 8.296, .1557, 8.922, 8.52, 3.655, -3.062, -5.108],
                lazy=True
            )
            optimizer.probe(
                params=[3.551, 9.89, 2.388, -.7407, 9.127, 7.306, 7.156, 3.102, -6.002],
                lazy=True
            )

            optimizer.probe(
                params=[1.6582781783649436, 9.997986074783114, 0.21472448060033456, 0.006893667195373432,
                        3.1510612189529805, 0.3122629971152935, 0.3591995006860538, 0.24805603344267624, 0.0],
                lazy=True
            )

            optimizer.probe(
                params=[2.035185150859349, 10.0, -0.6877427668961589, 1.4573203043343252, 10.0, 0.9470492935031377,
                        -2.320053015578459, -2.051928952647437, 0.0],
                lazy=True
            )

            optimizer.probe(
                params=[0.5123924482105198, 10.0, -1.2350150992249094, 1.4947062710196088, 10.0, -1.3092396199320315,
                        -2.107874738811746, 0.011675602577151556, 0.0],
                lazy=True
            )

            optimizer.probe(
                params=[3.6463633190958644, 10.0, -1.8773672836928867, 0.02583968115204241, 10.0, -2.0954716570946226,
                        -1.1346160438150195, -0.9566142530127371, 0.0],
                lazy=True
            )

            optimizer.probe(
                params=[1.9023702140437273, 10.0, 0.20397572803352837, -1.5970328111552923, 10.0, -0.45631575420135273,
                        -2.595178289674247, -0.08100447998228567, 0.0],
                lazy=True
            )

            optimizer.probe(
                params=[-1.3866726571563355, 10.0, -1.2778370998218895, 2.6709058069899894, 10.0, -0.2678868442723595,
                        -3.7551308488421133, 0.42342885189796814, 0.0],
                lazy=True
            )

            optimizer.probe(
                params=[5.7419504034578095, 10.0, -6.367115363774636, 3.7863479674680045, 10.0, -2.253884002686588, -1.5178809034523935, -2.8586002707429503, 5.972824553248542],
                lazy=True
            )

        else:
            optimizer.probe(
                params=[1.6582781783649436, 9.997986074783114, 0.21472448060033456, 0.006893667195373432, 3.1510612189529805, 0.3122629971152935, 0.3591995006860538, 0.24805603344267624],
                lazy=True
            )

            optimizer.probe(
                params=[2.035185150859349, 10.0, -0.6877427668961589, 1.4573203043343252, 10.0, 0.9470492935031377, -2.320053015578459, -2.051928952647437],
                lazy=True
            )

            optimizer.probe(
                params=[0.5123924482105198, 10.0, -1.2350150992249094, 1.4947062710196088, 10.0, -1.3092396199320315, -2.107874738811746, 0.011675602577151556],
                lazy=True
            )

            optimizer.probe(
                params=[3.6463633190958644, 10.0, -1.8773672836928867, 0.02583968115204241, 10.0, -2.0954716570946226, -1.1346160438150195, -0.9566142530127371],
                lazy=True
            )

            optimizer.probe(
                params=[1.9023702140437273, 10.0, 0.20397572803352837, -1.5970328111552923, 10.0, -0.45631575420135273, -2.595178289674247, -0.08100447998228567],
                lazy=True
            )

            optimizer.probe(
                params=[-1.3866726571563355, 10.0, -1.2778370998218895, 2.6709058069899894, 10.0, -0.2678868442723595, -3.7551308488421133, 0.42342885189796814],
                lazy=True
            )
            optimizer.probe(
                params=[-2.9960321254867717, 9.731360260770366, 0.23285254487602458, 1.2961047475432395, 9.905579302201307, 0.27990005570771714, -2.4215895340988247, -0.5715339058266125],
                lazy=True
            )

            optimizer.probe(
                params=[2.108641170737216, 10.0, -0.48348598777355895, 0.7269624769733463, 7.868115406177424, 0.013213339413243194, -2.229638309394156, -0.9648610207508962],
                lazy=True
            )


        #optimizer.maximize(init_points=300, n_iter=400, n_restarts_optimizer=100, acq='ei')
        optimizer.maximize(init_points=50, n_iter=100, n_restarts_optimizer=25, acq='ei')
        #optimizer.maximize(init_points=2, n_iter=5, n_restarts_optimizer=1, acq='ei')

        print(optimizer.max["target"])
        print("\n")
        for score, mean, std in zip(self.__best_score, self.__best_central, self.__best_deviation):
            print(f"{score}\t{mean}\t{std}")
        print("\n")
        print(f"Best param saved:\n{self.__best_param}")
        return self.__best_param

    def __produceScore(self, scores, param):
        #mean = statistics.mean(scores)
        #std = statistics.stdev(scores)
        #if std == 0:
        #    std3 = 100000
        #else:
        #    std3 = 1.0/(3.0*std)
        # central = statistics.median(scores)
        # deviation = scipy.stats.median_abs_deviation(scores)
        # if deviation == 0:
        #     inv_deviation = 100000000
        # else:
        #     inv_deviation = 1.0/deviation
        # beta_squared = 2.25
        # harmonic_score = (1+beta_squared)*(central * inv_deviation) / ((beta_squared*central) + inv_deviation)
        quantiles = mquantiles(scores)
        if quantiles[0] == quantiles[1] == quantiles[2] == 0:
            harmonic_score = 0
        else:
            median = quantiles[1]
            q1 = quantiles[0]
            q3 = quantiles[2]
            #iqr = 1.5*(q3 - q1)
            #iqr = q3 - q1
            #lower_bound = q1 - iqr
            #upper_bound = q3 + iqr
            #maximum_score = 0
            #minimum_score = 1
            maximum_score = q3
            minimum_score = q1
            #We remove outliers
            # for score in scores:
            #     #if lower_bound <= score <= upper_bound:
            #     if score <= upper_bound:
            #         if score > maximum_score:
            #             maximum_score = score
            #if maximum_score != 1.0:
            #    print("let's see")
            if minimum_score == 0:
                minimum_score = 0.01
            #harmonic_score = (3.5*quantiles[0]*quantiles[1]*quantiles[2])/((quantiles[0]*quantiles[1])+(quantiles[0]*quantiles[2]*1.5)+(quantiles[1]*quantiles[2]))
            harmonic_score = (5 * minimum_score * median * maximum_score) / (
                                (median*maximum_score) +                #Nothing for minimum
                                (minimum_score*maximum_score*2.5) +     #Boost median
                                (minimum_score*median*1.5)              #Boost maximum
                             )
        if math.isnan(harmonic_score):
            harmonic_score = 0
        if harmonic_score > self.__best_score[-1]:
            self.__best_param = param
            self.__best_score.append(harmonic_score)
            self.__best_central.append(quantiles[1])
            self.__best_deviation.append(f"{quantiles[0]}\t{quantiles[2]}")
        return harmonic_score

    def __objectiveFunction(self, **alphas):
        alphas = [alphas[value] for value in sorted(alphas.keys())]
        ndcg = []
        for year in self.__training_data.keys():
            with open(f"{self.__results_path}/optimization_{year}.txt", 'w') as output_file:
                for topic in self.__training_data[year].keys():
                    final_scores = {}
                    for recommendation_id in self.__training_data[year][topic].keys():
                        score = 0.0
                        coordination = 0.0
                        for alpha_id, alpha in enumerate(alphas):
                            if self.__a0 and alpha_id == len(alphas)-1:
                                score += alpha
                            else:
                                local_score = self.__training_data[year][topic][recommendation_id][alpha_id]
                                coordination += local_score
                                if local_score > 0:
                                    score += local_score*alpha
                        final_scores[recommendation_id] = coordination*score
                    for recommendation_id, (document_id, score) in enumerate(
                        sorted(final_scores.items(), key=lambda item: item[1], reverse=True)):
                        if recommendation_id >= 100:
                            break
                        output_file.write(f"{topic}\tQ0\t{document_id}\t{recommendation_id}\t{score}\toptimization\n")

            ndcg.extend(self.__evaluate(year))
        return self.__produceScore(ndcg, alphas)

    def __evaluate(self, year):
        if self.__cut == 0:
            metric = "ndcg"
        else:
            metric = f"ndcg_cut.{self.__cut}"
        results = subprocess.check_output(f"{self.__trec_eval_path}trec_eval -q -m {metric}"
                                          f" {self.__gs_path}{year}_extended_clean.txt"
                                          f" {self.__results_path}/optimization_{year}.txt", shell=True,
                                          universal_newlines=True)
        final_results = []
        for topic_result in results.split("\n"):
            if topic_result == "":
                continue
            _, topic, score = topic_result.split("\t")
            #We skip those topics for which we know there are no relevant documents
            if topic in self.__topics_without_relevant[year]:
                continue
            if topic != "all":
                final_results.append(float(score))
        return final_results


class LambdaInterpolation:

    def __init__(self, input_size=0, skip_columns=None, a0=False):
        self.__model = None
        self.__skip_columns = skip_columns
        self.__input_size = input_size
        self.__topics_without_relevant = {}
        self.__a0 = a0

    def __processTrainData(self, data, gold_standard):

        for year in data.keys():
            found = 0
            for topic in data[year].keys():
                skip_recommendation = []
                for recommendation in data[year][topic].keys():
                    if f"{year}_{topic}_{recommendation}" in gold_standard:
                        if self.__skip_columns is not None:
                            for column in sorted(self.__skip_columns, reverse=True):
                                del (data[year][topic][recommendation][column])
                        for column, value in enumerate(data[year][topic][recommendation]):
                            if value == -10.0:
                                data[year][topic][recommendation][column] = 0
                    else:
                        skip_recommendation.append(recommendation)
                for recommendation in skip_recommendation:
                    del(data[year][topic][recommendation])

            print(f"Number of recommendations with an evaluation in {year}: {found}")

        return data

    def __processGoldStandard(self, gs_path, years, extended_gs=False, print_clean=True, cleaned_data=None):
        gold_standard = {}
        extended = ""
        if extended_gs:
            extended = "_extended"
        for year in years:
            clean_gs = {}
            with_relevant = {}
            with open(f"{gs_path}/{year}{extended}.txt") as file_reader:
                for file_line in file_reader:
                    file_line = regex.sub("\n|\r", "", file_line)
                    data = regex.split("\s+", file_line)
                    score = int(data[3])
                    topic = int(data[0])
                    document_id = data[2]
                    if cleaned_data is not None and document_id not in cleaned_data[year][topic]:
                        score = 0.0
                    if topic not in clean_gs:
                        clean_gs[topic] = {}
                        with_relevant[topic] = 0
                    if f"{year}_{topic}_{document_id}" in gold_standard and score > gold_standard[f"{year}_{topic}_{document_id}"]:
                        gold_standard[f"{year}_{topic}_{document_id}"] = score
                        clean_gs[topic][document_id] = score
                        if score > 0:
                            with_relevant[topic] += 1
                    else:
                        gold_standard[f"{year}_{topic}_{document_id}"] = score
                        clean_gs[topic][document_id] = score
                        if score > 0:
                            with_relevant[topic] += 1
            self.__topics_without_relevant[year] = []
            for topic in with_relevant.keys():
                if with_relevant[topic] == 0:
                    self.__topics_without_relevant[year].append(topic)
                    print(f"Topic {topic} in {year} has no relevants")
            if print_clean:
                with open(f"{gs_path}/{year}{extended}_clean.txt", "w") as file_writer:
                    for topic in clean_gs.keys():
                        for document_id in clean_gs[topic].keys():
                            file_writer.write(f"{topic} 0 {document_id} {clean_gs[topic][document_id]}\n")
        return gold_standard

    def train(self, data, gs_path, years, inverse_kicker, print_plot=False, extended_gs=False, maximize=None):
        if maximize is not None:
            self.__model = maximize
        else:
            gold_standard = self.__processGoldStandard(gs_path, years, extended_gs=extended_gs, cleaned_data=data)
            training_data = self.__processTrainData(data, gold_standard)

            optimization = Optimization(self.__input_size, training_data, self.__topics_without_relevant, a0=self.__a0)

            self.__model = optimization.bayesianOptimization()


    def saveModel(self, path):
        print("Nothing")
        #self.__model.save_model(path)

    def loadModel(self, path):
        print("Nothing")
        #self.__model = xgb.Booster()
        #self.__model.load_model(path)

    def predict(self, data):
        for topic in data.keys():
            for recommendation in data[topic].keys():
                if self.__skip_columns is not None:
                    for column in sorted(self.__skip_columns, reverse=True):
                        del (data[topic][recommendation][column])
                score = 0.0
                coordination = 0.0
                for alpha_id, alpha in enumerate(self.__model):
                    if self.__a0 and alpha_id == len(self.__model)-1:
                        score += alpha
                    else:
                        local_score = data[topic][recommendation][alpha_id]
                        coordination += local_score
                        if local_score > 0:
                            score += local_score * alpha
                data[topic][recommendation] = coordination * score
        return data







