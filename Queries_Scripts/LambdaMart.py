import numpy as np
import regex
import xgboost as xgb
from bayes_opt import BayesianOptimization


class Optimization:

    def __init__(self, training_data, base_param):
        self.__training_data = training_data
        self.__best_score = [-1]
        self.__best_mean = [-1]
        self.__best_std = [-1]
        self.__best_param = {}
        self.__base_param = base_param
        self.__feature_selector = ["cyclic", "shuffle", "random", "greedy", "greedy"]

    def bayesianOptimization(self, booster):
        param_bound = {}
        optimizer = None
        if booster == "gbtree":
            param_bound = {"max_depth": (3, 20), "eta": (0, 0.3), "max_delta_step": (0, 10),
                           "subsample": (0.5, 1), "iterations": (10, 200), #"max_leaves": (3, 25),
                           "gamma": (0, 5), "colsample_bytree": (0.1, 1), "min_child_weight": (0, 50)}
            optimizer = BayesianOptimization(self.__cvTrainingGbtree, param_bound, random_state=12)
        elif booster == "gblinear":
            param_bound = {"alpha": (0, 10), "feature_selector": (0, 4)}
            optimizer = BayesianOptimization(self.__cvTrainingGblinear, param_bound, random_state=12)
        optimizer.maximize(init_points=20 * len(param_bound), n_iter=150, n_restarts_optimizer=20, acq='ei')
        iterations, final_params = self.__normalizeParams(optimizer.max["params"])

        print(optimizer.max["target"])
        print("\n")
        for score, mean, std in zip(self.__best_score, self.__best_mean, self.__best_std):
            print(f"{score}\t{mean}\t{std}")
        print("\n")
        print(f"Best param saved:\n{self.__best_param}")
        print(iterations)
        print(f"Best param obtained:\n{final_params}")
        return final_params, iterations

    @staticmethod
    def __rounder2(value):
        value = round(value)
        return value*10

    @staticmethod
    def __rounder(value, max_multiplied_by):
        value *= 10
        int_value = int(value)
        multiplied_by = 10
        while int_value != 0 and multiplied_by < max_multiplied_by:
            value *= 10
            multiplied_by *= 10
            int_value = int(value)
        return int_value / multiplied_by

    def __normalizeParams(self, params):
        final_params = {}
        iterations = 100
        for param in params.keys():
            value = params[param]
            #if param in ["eta"]:
            #    value = self.__rounder(value, 1000)
            #elif param in ["subsample", "colsample_bytree", "gamma"]:
            #    value = self.__rounder(value, 10)
            #elif param in ["iterations", "max_leaves"]:
            if param in ["iterations", "max_leaves"]:
                value = self.__rounder2(value)
            #elif param in ["max_delta_step", "min_child_weight"]:
            #    value = round(value)
            elif param in ["max_depth", "feature_selector"]:
                value = int(value)

            if param == "feature_selector":
                final_params[param] = self.__feature_selector[value]
            elif param == "iterations":
                iterations = value
            else:
                final_params[param] = value

        return iterations, final_params

    def __produceScore(self, mean, std, param):
        if std == 0:
            std3 = 100000
        else:
            std3 = 1.0/(3.0*std)
        beta_squared = 2.25
        harmonic_score = (1+beta_squared)*(mean * std3) / ((beta_squared*mean) + std3)
        if harmonic_score > self.__best_score[-1]:
            self.__best_param = param
            self.__best_score.append(harmonic_score)
            self.__best_mean.append(mean)
            self.__best_std.append(std)
        return harmonic_score

    def __cvTrainingGblinear(self, alpha, feature_selector, ndcg_at="15-"):
        param = {"alpha": alpha, "feature_selector": feature_selector}
        param.update(self.__base_param)
        iterations, param = self.__normalizeParams(param)
        results = xgb.cv(param, self.__training_data, num_boost_round=iterations, seed=12, metrics=f"ndcg@{ndcg_at}", shuffle=True, nfold=5)
        mean = results[f"test-ndcg@{ndcg_at}-mean"][-1]
        std = results[f"test-ndcg@{ndcg_at}-std"][-1]
        return self.__produceScore(mean, std, param)

    def __cvTrainingGbtree(self, max_depth, eta, subsample, colsample_bytree, iterations, max_delta_step, gamma, min_child_weight, ndcg_at="@10-"):
        param = {"max_depth": max_depth, "eta": eta, "colsample_bytree": colsample_bytree, "max_delta_step": max_delta_step,
                 "subsample": subsample, "gamma": gamma, "iterations": iterations, "min_child_weight": min_child_weight}
        param.update(self.__base_param)
        iterations, param = self.__normalizeParams(param)
        results = xgb.cv(param, self.__training_data, num_boost_round=iterations, seed=12, metrics=f"ndcg{ndcg_at}", shuffle=True, nfold=5)
        mean = results[f"test-ndcg{ndcg_at}-mean"][-1]
        std = results[f"test-ndcg{ndcg_at}-std"][-1]
        return self.__produceScore(mean, std, param)


class LambdaMart:

    def __init__(self, booster="gbtree", skip_zero_relevant=False):
        self.__model = None
        self.__booster = booster
        self.__skip_zero_relevant = skip_zero_relevant
        self.__zero_relevant = {}

    def __processTrainData(self, data, gold_standard, opinion_kicker):
        relevance = []
        features = []
        qid = []
        qid_counter = 0

        for year in data.keys():
            not_found = 0
            relevant_but_kicker = 0
            for topic in data[year].keys():
                qid_counter += 1
                for recommendation in data[year][topic].keys():
                    if f"{year}_{topic}_{recommendation}" in gold_standard:
                        #if data[year][topic][recommendation][-1] == 1:
                        #    continue
                        #features.append(data[year][topic][recommendation])
                        features.append(data[year][topic][recommendation])
                        qid.append(qid_counter)
                        #if gold_standard[f"{year}_{topic}_{recommendation}"] > 0 and\
                        #        data[year][topic][recommendation][-1] == 1:
                        #    relevance.append(0)
                        #    relevant_but_kicker += 1
                        #else:
                        #    relevance.append(gold_standard[f"{year}_{topic}_{recommendation}"])
                        relevance.append(gold_standard[f"{year}_{topic}_{recommendation}"])
                    #elif data[year][topic][recommendation][-1] == -10:
                    #   not_found += 1
                    #elif data[year][topic][recommendation][-1] == opinion_kicker:    #Should not recommend these
                    #   features.append(data[year][topic][recommendation])
                    #   qid.append(qid_counter)
                    #   relevance.append(0)
                    else:
                        not_found += 1
            print(f"Number of recommendations without an evaluation in {year}: {not_found}")
            print(f"Number of recommendations with forbidden kicker but relevant {year}: {relevant_but_kicker}")
        print(f"Total number of documents with relevance: {len(relevance)}")

        relevance = np.array(relevance)
        qid = np.array(qid)
        features = np.array(features)

        #return qid, features, relevance
        return xgb.DMatrix(data=features, qid=qid, label=relevance, missing=-10.0)

    @staticmethod
    def __processGoldStandard(gs_path, years, extended_gs=False):
        gold_standard = {}
        extended = ""
        if extended_gs:
            extended = "_extended"
        for year in years:
            with open(f"{gs_path}/{year}{extended}.txt") as file_reader:
                for file_line in file_reader:
                    file_line = regex.sub("\n|\r", "", file_line)
                    data = regex.split("\s+", file_line)
                    score = int(data[3])
                    # if score == 2:
                    #     score = 1
                    # elif score == 4:
                    #     score = 2
                    # elif score == 8:
                    #     score = 3
                    # elif score == 16:
                    #     score = 4
                    if f"{year}_{data[0]}_{data[2]}" in gold_standard and score > gold_standard[f"{year}_{data[0]}_{data[2]}"]:
                        gold_standard[f"{year}_{data[0]}_{data[2]}"] = score
                    else:
                        gold_standard[f"{year}_{data[0]}_{data[2]}"] = score
        return gold_standard

    def train(self, data, gs_path, years, inverse_kicker, print_plot=False, extended_gs=False, maximize=None):
        gold_standard = self.__processGoldStandard(gs_path, years, extended_gs=extended_gs)
        training_data = self.__processTrainData(data, gold_standard, inverse_kicker)
        param = {}
        if self.__booster == "gbtree":
            param = {'objective': 'rank:ndcg', 'verbosity': 1, "nthread": 10, "tree_method": "auto", "booster": self.__booster,
                     "sampling_method": "uniform", "grow_policy": "depthwise"}  # "sampling_method": "gradient_based"}
        elif self.__booster == "gblinear":
            param = {'objective': 'rank:ndcg', 'verbosity': 1, "nthread": 10, "booster": self.__booster, "updater": "coord_descent"}
        if maximize is None:
            optimized_param, iterations = self.__bayesianOptimization(training_data, param)
        else:
            print("Training model on:")
            print(maximize)
            iterations = maximize["iterations"]
            del (maximize["iterations"])
            optimized_param = maximize

        param.update(optimized_param)
        self.__model = xgb.train(param, training_data, num_boost_round=iterations)
        if print_plot:
            ax = xgb.plot_importance(self.__model, color='red')
            fig = ax.figure
            fig.set_size_inches(20, 20)
            fig.show()
            print("Plot should be visible")

    def saveModel(self, path):
        self.__model.save_model(path)

    def loadModel(self, path):
        self.__model = xgb.Booster()
        self.__model.load_model(path)

    def __bayesianOptimization(self, training_data, base_param):
        optimization = Optimization(training_data, base_param)
        return optimization.bayesianOptimization(self.__booster)

    def predict(self, data):

        for topic in data.keys():
            features = []
            recommendations = []
            for recommendation in data[topic].keys():
                features.append(data[topic][recommendation])
                recommendations.append(recommendation)
            features = np.array(features)
            features = xgb.DMatrix(features, missing=-10.0)
            predictions = self.__model.predict(features)
            for recommendation_id, recommendation in enumerate(recommendations):
                data[topic][recommendation] = predictions[recommendation_id]

        return data







