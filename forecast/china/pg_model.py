import os

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
from tqdm import tqdm


class NAIFCModel:
    def __init__(
            self,
            dates,
            n_clusters=None,
            n_jobs=-1,
            ts_cols=None,
            original_features=None,
            dates_features=None,
            lag_features=None,
            all_others_cols=None,
            total_features=None,
            tsfresh_features=None
    ):
        self.dates = dates
        self.n_jobs = n_jobs

        if not os.path.exists("output"):
            os.makedirs("output")

        self.encoders = {}
        self.models = {}
        self.clusters = {}
        self.drop_cols = None
        self.datas = {}
        if n_clusters is None:
            n_clusters = [5, 6, 7, 8, 9, 10]
        self.n_clusters = n_clusters

        self.ts_cols = ts_cols
        self.original_features = original_features
        self.dates_features = dates_features
        self.lag_features = lag_features
        self.all_others_cols = all_others_cols
        self.total_features = total_features
        self.tsfresh_features = tsfresh_features

    def fit(self, X, y=None, drop_cols=None):
        if drop_cols is None:
            drop_cols = [
                "year", "year", "basic_material", "material_description", "is_month_start", "is_month_end",
                "is_quarter_start", "is_quarter_end", "is_year_start", "is_year_end"
            ]

        self.drop_cols = drop_cols
        f = X.copy()

        if self.ts_cols is None:
            self.ts_cols = ["id", "ts", "y"]
            self.original_features = [
                "price", "list_price", "basic_material", "channel", "company", "customer", "customer_group", "material",
                "material_description", "prod_brand", "prod_category"
            ]
            self.dates_features = [
                "year", "month", "quarter", "woy", "is_month_start", "is_month_end", "is_quarter_start",
                "is_quarter_end", "is_year_start", "is_year_end"
            ]
            self.lag_features = [x for x in f.columns if "y_lag_" in x]
            self.all_others_cols = self.ts_cols + self.original_features + self.dates_features + self.lag_features
            self.total_features = [x for x in f.drop(columns=self.all_others_cols).columns[-66:]]
            self.tsfresh_features = [x for x in f.drop(columns=self.all_others_cols + self.total_features).columns]
        else:
            self.all_others_cols = self.ts_cols + self.original_features + self.dates_features + self.lag_features

        train = f.loc[f.ts < np.min(self.dates["test"])].reset_index(drop=True)
        test = f.loc[f.ts.isin(self.dates["test"])].reset_index(drop=True)
        predict = f.loc[f.ts.isin(self.dates["forecast"])].reset_index(drop=True)

        train["y"] = train["y"].astype(float)
        test["y"] = test["y"].astype(float)
        predict["y"] = np.nan

        train_dates = train["ts"]
        x_train, y_train = train.drop(columns=["y", "ts"]), train["y"]
        x_test, y_test = test.drop(columns=["y", "ts"]), test["y"]
        x_predict, y_predict = predict.drop(columns=["y", "ts"]), predict["y"]

        for c in tqdm(x_train.select_dtypes(include=["number", "bool_"]).columns):
            x_train[c] = np.nan_to_num(x_train[c], nan=-9999, posinf=-9999, neginf=-9999)
            x_test[c] = np.nan_to_num(x_test[c], nan=-9999, posinf=-9999, neginf=-9999)
            x_predict[c] = np.nan_to_num(x_predict[c], nan=-9999, posinf=-9999, neginf=-9999)

        enc_cols = x_train.select_dtypes(exclude="number").columns.to_list()
        self.encoders = {}
        for col in enc_cols:
            enc = LabelEncoder()
            x_train[col] = enc.fit_transform(x_train[col])
            x_test[col] = enc.transform(x_test[col])
            x_predict[col] = enc.transform(x_predict[col])
            self.encoders.update({col: enc})

        x_train.drop(columns=drop_cols, inplace=True)
        x_test.drop(columns=drop_cols, inplace=True)
        x_predict.drop(columns=drop_cols, inplace=True)

        self.datas = {
            "train": train,
            "test": test,
            "predict": predict,
            "x_train": x_train,
            "y_train": y_train,
            "x_test": x_test,
            "y_test": y_test,
            "x_predict": x_predict,
            "y_predict": y_predict,
            "train_dates": train_dates,
        }

        self.models = {}

        xx_train = x_train[self.total_features].values
        # remove the -9999
        xx_train = np.where(xx_train == -9999, 0, xx_train)

        # do PCA
        pca = PCA()
        xx_train = pca.fit_transform(xx_train)

        for clust in self.n_clusters:
            print("N Cluster: {}".format(clust))

            kmeans = KMeans(
                init="k-means++",
                n_clusters=clust,
                random_state=0
            ).fit(xx_train)
            cluster_labels = kmeans.labels_

            _model = {}
            for label in np.unique(cluster_labels):
                idx = cluster_labels == label
                _x_train = x_train.loc[idx].reset_index(drop=True)
                _y_train = y_train.loc[idx].reset_index(drop=True)

                model = RandomForestRegressor(
                    n_estimators=1000,
                    criterion="mse",
                    max_features="sqrt",
                    random_state=0,
                    max_samples=0.9,
                    n_jobs=-1,
                    verbose=1,
                )
                if len(drop_cols) == 0:
                    model.fit(
                        _x_train.drop(columns=self.total_features, errors="ignore"),
                        _y_train
                    )
                else:
                    model.fit(
                        _x_train.drop(columns=drop_cols + self.total_features, errors="ignore"),
                        _y_train
                    )

                _model.update({label: model})
            self.models.update({clust: _model})
            self.clusters.update({
                clust: {
                    "pca": pca,
                    "labels": cluster_labels,
                    "kmeans": kmeans
                }}
            )

    def predict(self, X=None, return_all_att=False):
        if X is None:
            x_train = self.datas["x_train"]
            x_test = self.datas["x_test"]
            x_predict = self.datas["x_predict"]

            xx_train = x_train[self.total_features].values
            xx_test = x_test[self.total_features].values
            xx_predict = x_predict[self.total_features].values

            # remove the -9999
            xx_train = np.where(xx_train == -9999, 0, xx_train)
            xx_test = np.where(xx_test == -9999, 0, xx_test)
            xx_predict = np.where(xx_predict == -9999, 0, xx_predict)

            yhat = np.zeros(x_train.shape[0])
            yhat_test = np.zeros(x_test.shape[0])
            yhat_predict = np.zeros(x_predict.shape[0])

            for clust in self.n_clusters:
                pca = self.clusters[clust]["pca"]
                xxx_train = pca.transform(xx_train)
                xxx_test = pca.transform(xx_test)
                xxx_predict = pca.transform(xx_predict)

                kmeans = self.clusters[clust]["kmeans"]
                labels = kmeans.predict(xxx_train)
                labels_test = kmeans.predict(xxx_test)
                labels_predict = kmeans.predict(xxx_predict)

                _model = self.models[clust]
                for label in np.unique(self.clusters[clust]["labels"]):
                    idx1 = labels == label
                    idx2 = labels_test == label
                    idx3 = labels_predict == label

                    model = _model[label]
                    model.set_params(**{"verbose": 0})

                    _x_train = x_train.loc[idx1].copy()
                    _x_test = x_test.loc[idx2].copy()
                    _x_predict = x_predict.loc[idx3].copy()

                    if len(self.drop_cols) > 0:
                        _x_train.drop(columns=self.drop_cols + self.total_features, inplace=True, errors="ignore")
                        _x_test.drop(columns=self.drop_cols + self.total_features, inplace=True, errors="ignore")
                        _x_predict.drop(columns=self.drop_cols + self.total_features, inplace=True, errors="ignore")
                    else:
                        _x_train.drop(columns=self.total_features, inplace=True, errors="ignore")
                        _x_test.drop(columns=self.total_features, inplace=True, errors="ignore")
                        _x_predict.drop(columns=self.total_features, inplace=True, errors="ignore")

                    yhat[idx1] += model.predict(_x_train)
                    if len(_x_test) > 0:
                        yhat_test[idx2] += model.predict(_x_test)
                    if len(_x_predict) > 0:
                        yhat_predict[idx3] += model.predict(_x_predict)

            yhat /= len(self.n_clusters)
            yhat_test /= len(self.n_clusters)

            predict_df = pd.concat([
                self.datas["train"],
                self.datas["test"],
                self.datas["predict"],
            ]).reset_index(drop=True)
            predict_df["yhat"] = np.concatenate((yhat, yhat_test, yhat_predict))
            predict_df["yhat"] = np.clip(np.round(predict_df["yhat"]), 0, None)
            predict_df["model"] = "NAI_clustering_RF"

            if return_all_att:
                return predict_df
            else:
                return predict_df["yhat"]
        else:
            xx = X[self.total_features].values

            # remove the -9999
            xx = np.where(xx == -9999, 0, xx)

            yhat = np.zeros(X.shape[0])
            for clust in self.n_clusters:
                pca = self.clusters[clust]["pca"]
                xxx = pca.transform(xx)

                kmeans = self.clusters[clust]["kmeans"]
                labels = kmeans.predict(xxx)

                _model = self.models[clust]
                for label in np.unique(self.clusters[clust]["labels"]):
                    idx1 = labels == label

                    model = _model[label]
                    model.set_params(**{"verbose": 0})

                    _x = X.loc[idx1].copy()

                    if len(self.drop_cols) > 0:
                        _x.drop(columns=self.drop_cols + self.total_features, inplace=True, errors="ignore")
                    else:
                        _x.drop(columns=self.total_features, inplace=True, errors="ignore")

                    if len(_x) > 0:
                        yhat[idx1] += model.predict(_x)

            yhat /= len(self.n_clusters)
            predict_df = X.copy()
            predict_df["yhat"] = yhat
            predict_df["yhat"] = np.clip(np.round(predict_df["yhat"]), 0, None)
            predict_df["model"] = "NAI_clustering_RF"

            if return_all_att:
                return predict_df
            else:
                return predict_df["yhat"]
