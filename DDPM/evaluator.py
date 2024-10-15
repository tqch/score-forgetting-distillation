"""
Modified from https://github.com/openai/guided-diffusion/blob/main/evaluations/evaluator.py.
"""
import os
import glob

try:
    cuda_path = glob.glob("/usr/local/cuda*")[0]
    os.environ["PATH"] = f"{cuda_path}/bin:{os.environ['PATH']}"
    os.environ[
        "LD_LIBRARY_PATH"
    ] = f"{cuda_path}/lib64:{os.environ.get('LD_LIBRARY_PATH', '')}"
except IndexError:
    pass

import argparse
import io
import random
import warnings
import zipfile
from abc import ABC, abstractmethod
from contextlib import contextmanager
from functools import partial
from multiprocessing import cpu_count
from multiprocessing.pool import ThreadPool
from typing import Iterable, Optional, Tuple

import csv
try:
    import cv2
    def imread(path):
        return cv2.imread(path)[:,:,-1::-1]  # BGR -> RGB
except ImportError:
    import PIL.Image
    imread = PIL.Image.open  # RGB

import numpy as np
import requests
import tensorflow.compat.v1 as tf
from scipy import linalg
from tqdm.auto import tqdm

INCEPTION_V3_URL = "https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/classify_image_graph_def.pb"
INCEPTION_V3_PATH = "classify_image_graph_def.pb"

FID_POOL_NAME = "pool_3:0"
FID_SPATIAL_NAME = "mixed_6/conv:0"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ref_batch", help="path to reference folder")
    parser.add_argument("sample_batch", help="path to sample folder")
    parser.add_argument("exclude_regex", nargs="?", default=None, help="regular expression for exclusion")
    args = parser.parse_args()

    ref_arr = read_images_folder(args.ref_batch, exclude_regex=args.exclude_regex)
    sample_arr = read_images_folder(args.sample_batch, exclude_regex=args.exclude_regex)

    config = tf.ConfigProto(
        allow_soft_placement=True  # allows DecodeJpeg to run on CPU in Inception graph
    )
    config.gpu_options.allow_growth = True
    evaluator = Evaluator(tf.Session(config=config))

    print("warming up TensorFlow...")
    # This will cause TF to print a bunch of verbose stuff now rather
    # than after the next print(), to help prevent confusion.
    evaluator.warmup()

    print("computing reference batch activations...")
    ref_acts = evaluator.read_activations(ref_arr)
    print("computing/reading reference batch statistics...")
    ref_stats, ref_stats_spatial = evaluator.read_statistics(ref_acts)

    print("computing sample batch activations...")
    sample_acts = evaluator.read_activations(sample_arr)
    print("computing/reading sample batch statistics...")
    sample_stats, sample_stats_spatial = evaluator.read_statistics(sample_acts)

    print("Computing evaluations...")
    print("Inception Score:", evaluator.compute_inception_score(sample_acts[0]))
    print("FID:", sample_stats.frechet_distance(ref_stats))
    print("sFID:", sample_stats_spatial.frechet_distance(ref_stats_spatial))
    prec, recall = evaluator.compute_prec_recall(ref_acts[0], sample_acts[0])
    print("Precision:", prec)
    print("Recall:", recall)

    # Check if the CSV file exists
    csv_file_path = "results/result.csv"
    headers = ["Index", "Inception Score", "FID", "sFID", "Precision", "Recall"]
    csv_data = [headers]
    indices = []
    skip_row = True
    if os.path.exists(csv_file_path):
        with open(csv_file_path) as csv_file:
            reader = csv.reader(csv_file, delimiter=",", quotechar="\"")
            for data in reader:
                if skip_row:
                    skip_row = False
                    continue
                csv_data.append(data)
                indices.append(data[0])

    name = "/".join(args.ref_batch.split("/")[-3: -1])
    result = {
        "Inception Score": evaluator.compute_inception_score(sample_acts[0]),
        "FID": sample_stats.frechet_distance(ref_stats),
        "sFID": sample_stats_spatial.frechet_distance(ref_stats_spatial),
        "Precision": prec,
        "Recall": recall,
    }

    if name not in indices:
        csv_data.append([name, ] + [result[metric] for metric in headers[1:]])
    else:
        k = indices.index(name)
        for i, metric in enumerate(headers[1:]):
            csv_data[k][i] = result[metric]
    print(csv_data)
    # Save the updated csv
    with open(csv_file_path, "w", newline="") as csv_file:
        writer = csv.writer(csv_file, delimiter=",", quotechar="\"")
        writer.writerows(csv_data)


class InvalidFIDException(Exception):
    pass


class FIDStatistics:
    def __init__(self, mu: np.ndarray, sigma: np.ndarray):
        self.mu = mu
        self.sigma = sigma

    def frechet_distance(self, other, eps=1e-6):
        """
        Compute the Frechet distance between two sets of statistics.
        """
        # https://github.com/bioinf-jku/TTUR/blob/73ab375cdf952a12686d9aa7978567771084da42/fid.py#L132
        mu1, sigma1 = self.mu, self.sigma
        mu2, sigma2 = other.mu, other.sigma

        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)

        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)

        assert (
            mu1.shape == mu2.shape
        ), f"Training and test mean vectors have different lengths: {mu1.shape}, {mu2.shape}"
        assert (
            sigma1.shape == sigma2.shape
        ), f"Training and test covariances have different dimensions: {sigma1.shape}, {sigma2.shape}"

        diff = mu1 - mu2

        # product might be almost singular
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            msg = (
                "fid calculation produces singular product; adding %s to diagonal of cov estimates"
                % eps
            )
            warnings.warn(msg)
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        # numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError("Imaginary component {}".format(m))
            covmean = covmean.real

        tr_covmean = np.trace(covmean)

        return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


class Evaluator:
    def __init__(
        self,
        session,
        batch_size=64,
        softmax_batch_size=512,
    ):
        self.sess = session
        self.batch_size = batch_size
        self.softmax_batch_size = softmax_batch_size
        self.manifold_estimator = ManifoldEstimator(session)
        with self.sess.graph.as_default():
            self.image_input = tf.placeholder(tf.float32, shape=[None, None, None, 3])
            self.softmax_input = tf.placeholder(tf.float32, shape=[None, 2048])
            self.pool_features, self.spatial_features = _create_feature_graph(
                self.image_input
            )
            self.softmax = _create_softmax_graph(self.softmax_input)

    def warmup(self):
        self.compute_activations(np.zeros([1, 8, 64, 64, 3]))

    # def read_activations(self, npz_path: str) -> Tuple[np.ndarray, np.ndarray]:
    #     with open_npz_array(npz_path, "arr_0") as reader:
    #         return self.compute_activations(reader.read_batches(self.batch_size))

    def read_activations(self, np_arr: str) -> Tuple[np.ndarray, np.ndarray]:
        reader = MemoryNpzArrayReader(np_arr)
        return self.compute_activations(reader.read_batches(self.batch_size))

    def compute_activations(
        self, batches: Iterable[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute image features for downstream evals.

        :param batches: a iterator over NHWC numpy arrays in [0, 255].
        :return: a tuple of numpy arrays of shape [N x X], where X is a feature
                 dimension. The tuple is (pool_3, spatial).
        """
        preds = []
        spatial_preds = []
        for batch in tqdm(batches):
            batch = batch.astype(np.float32)
            pred, spatial_pred = self.sess.run(
                [self.pool_features, self.spatial_features], {self.image_input: batch}
            )
            preds.append(pred.reshape([pred.shape[0], -1]))
            spatial_preds.append(spatial_pred.reshape([spatial_pred.shape[0], -1]))
        return (
            np.concatenate(preds, axis=0),
            np.concatenate(spatial_preds, axis=0),
        )

    def read_statistics(
        self, activations: Tuple[np.ndarray, np.ndarray]
    ) -> Tuple[FIDStatistics, FIDStatistics]:
        # obj = np.load(npz_path)
        # if "mu" in list(obj.keys()):
        #     return FIDStatistics(obj["mu"], obj["sigma"]), FIDStatistics(
        #         obj["mu_s"], obj["sigma_s"]
        #     )
        return tuple(self.compute_statistics(x) for x in activations)

    def compute_statistics(self, activations: np.ndarray) -> FIDStatistics:
        mu = np.mean(activations, axis=0)
        sigma = np.cov(activations, rowvar=False)
        return FIDStatistics(mu, sigma)

    def compute_inception_score(
        self, activations: np.ndarray, split_size: int = 5000
    ) -> float:
        softmax_out = []
        for i in range(0, len(activations), self.softmax_batch_size):
            acts = activations[i : i + self.softmax_batch_size]
            softmax_out.append(
                self.sess.run(self.softmax, feed_dict={self.softmax_input: acts})
            )
        preds = np.concatenate(softmax_out, axis=0)
        # https://github.com/openai/improved-gan/blob/4f5d1ec5c16a7eceb206f42bfc652693601e1d5c/inception_score/model.py#L46
        scores = []
        for i in range(0, len(preds), split_size):
            part = preds[i : i + split_size]
            kl = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
            kl = np.mean(np.sum(kl, 1))
            scores.append(np.exp(kl))
        return float(np.mean(scores))

    def compute_prec_recall(
        self, activations_ref: np.ndarray, activations_sample: np.ndarray
    ) -> Tuple[float, float]:
        radii_1 = self.manifold_estimator.manifold_radii(activations_ref)
        radii_2 = self.manifold_estimator.manifold_radii(activations_sample)
        pr = self.manifold_estimator.evaluate_pr(
            activations_ref, radii_1, activations_sample, radii_2
        )
        return (float(pr[0][0]), float(pr[1][0]))


class ManifoldEstimator:
    """
    A helper for comparing manifolds of feature vectors.

    Adapted from https://github.com/kynkaat/improved-precision-and-recall-metric/blob/f60f25e5ad933a79135c783fcda53de30f42c9b9/precision_recall.py#L57
    """

    def __init__(
        self,
        session,
        row_batch_size=10000,
        col_batch_size=10000,
        nhood_sizes=(3,),
        clamp_to_percentile=None,
        eps=1e-5,
    ):
        """
        Estimate the manifold of given feature vectors.

        :param session: the TensorFlow session.
        :param row_batch_size: row batch size to compute pairwise distances
                               (parameter to trade-off between memory usage and performance).
        :param col_batch_size: column batch size to compute pairwise distances.
        :param nhood_sizes: number of neighbors used to estimate the manifold.
        :param clamp_to_percentile: prune hyperspheres that have radius larger than
                                    the given percentile.
        :param eps: small number for numerical stability.
        """
        self.distance_block = DistanceBlock(session)
        self.row_batch_size = row_batch_size
        self.col_batch_size = col_batch_size
        self.nhood_sizes = nhood_sizes
        self.num_nhoods = len(nhood_sizes)
        self.clamp_to_percentile = clamp_to_percentile
        self.eps = eps

    def warmup(self):
        feats, radii = (
            np.zeros([1, 2048], dtype=np.float32),
            np.zeros([1, 1], dtype=np.float32),
        )
        self.evaluate_pr(feats, radii, feats, radii)

    def manifold_radii(self, features: np.ndarray) -> np.ndarray:
        num_images = len(features)

        # Estimate manifold of features by calculating distances to k-NN of each sample.
        radii = np.zeros([num_images, self.num_nhoods], dtype=np.float32)
        distance_batch = np.zeros([self.row_batch_size, num_images], dtype=np.float32)
        seq = np.arange(max(self.nhood_sizes) + 1, dtype=np.int32)

        for begin1 in range(0, num_images, self.row_batch_size):
            end1 = min(begin1 + self.row_batch_size, num_images)
            row_batch = features[begin1:end1]

            for begin2 in range(0, num_images, self.col_batch_size):
                end2 = min(begin2 + self.col_batch_size, num_images)
                col_batch = features[begin2:end2]

                # Compute distances between batches.
                distance_batch[
                    0 : end1 - begin1, begin2:end2
                ] = self.distance_block.pairwise_distances(row_batch, col_batch)

            # Find the k-nearest neighbor from the current batch.
            radii[begin1:end1, :] = np.concatenate(
                [
                    x[:, self.nhood_sizes]
                    for x in _numpy_partition(
                        distance_batch[0 : end1 - begin1, :], seq, axis=1
                    )
                ],
                axis=0,
            )

        if self.clamp_to_percentile is not None:
            max_distances = np.percentile(radii, self.clamp_to_percentile, axis=0)
            radii[radii > max_distances] = 0
        return radii

    def evaluate(
        self, features: np.ndarray, radii: np.ndarray, eval_features: np.ndarray
    ):
        """
        Evaluate if new feature vectors are at the manifold.
        """
        num_eval_images = eval_features.shape[0]
        num_ref_images = radii.shape[0]
        distance_batch = np.zeros(
            [self.row_batch_size, num_ref_images], dtype=np.float32
        )
        batch_predictions = np.zeros([num_eval_images, self.num_nhoods], dtype=np.int32)
        max_realism_score = np.zeros([num_eval_images], dtype=np.float32)
        nearest_indices = np.zeros([num_eval_images], dtype=np.int32)

        for begin1 in range(0, num_eval_images, self.row_batch_size):
            end1 = min(begin1 + self.row_batch_size, num_eval_images)
            feature_batch = eval_features[begin1:end1]

            for begin2 in range(0, num_ref_images, self.col_batch_size):
                end2 = min(begin2 + self.col_batch_size, num_ref_images)
                ref_batch = features[begin2:end2]

                distance_batch[
                    0 : end1 - begin1, begin2:end2
                ] = self.distance_block.pairwise_distances(feature_batch, ref_batch)

            # From the minibatch of new feature vectors, determine if they are in the estimated manifold.
            # If a feature vector is inside a hypersphere of some reference sample, then
            # the new sample lies at the estimated manifold.
            # The radii of the hyperspheres are determined from distances of neighborhood size k.
            samples_in_manifold = distance_batch[0 : end1 - begin1, :, None] <= radii
            batch_predictions[begin1:end1] = np.any(samples_in_manifold, axis=1).astype(
                np.int32
            )

            max_realism_score[begin1:end1] = np.max(
                radii[:, 0] / (distance_batch[0 : end1 - begin1, :] + self.eps), axis=1
            )
            nearest_indices[begin1:end1] = np.argmin(
                distance_batch[0 : end1 - begin1, :], axis=1
            )

        return {
            "fraction": float(np.mean(batch_predictions)),
            "batch_predictions": batch_predictions,
            "max_realisim_score": max_realism_score,
            "nearest_indices": nearest_indices,
        }

    def evaluate_pr(
        self,
        features_1: np.ndarray,
        radii_1: np.ndarray,
        features_2: np.ndarray,
        radii_2: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Evaluate precision and recall efficiently.

        :param features_1: [N1 x D] feature vectors for reference batch.
        :param radii_1: [N1 x K1] radii for reference vectors.
        :param features_2: [N2 x D] feature vectors for the other batch.
        :param radii_2: [N x K2] radii for other vectors.
        :return: a tuple of arrays for (precision, recall):
                 - precision: an np.ndarray of length K1
                 - recall: an np.ndarray of length K2
        """
        features_1_status = np.zeros([len(features_1), radii_2.shape[1]], dtype=bool)
        features_2_status = np.zeros([len(features_2), radii_1.shape[1]], dtype=bool)
        for begin_1 in range(0, len(features_1), self.row_batch_size):
            end_1 = begin_1 + self.row_batch_size
            batch_1 = features_1[begin_1:end_1]
            for begin_2 in range(0, len(features_2), self.col_batch_size):
                end_2 = begin_2 + self.col_batch_size
                batch_2 = features_2[begin_2:end_2]
                batch_1_in, batch_2_in = self.distance_block.less_thans(
                    batch_1, radii_1[begin_1:end_1], batch_2, radii_2[begin_2:end_2]
                )
                features_1_status[begin_1:end_1] |= batch_1_in
                features_2_status[begin_2:end_2] |= batch_2_in
        return (
            np.mean(features_2_status.astype(np.float64), axis=0),
            np.mean(features_1_status.astype(np.float64), axis=0),
        )


class DistanceBlock:
    """
    Calculate pairwise distances between vectors.

    Adapted from https://github.com/kynkaat/improved-precision-and-recall-metric/blob/f60f25e5ad933a79135c783fcda53de30f42c9b9/precision_recall.py#L34
    """

    def __init__(self, session):
        self.session = session

        # Initialize TF graph to calculate pairwise distances.
        with session.graph.as_default():
            self._features_batch1 = tf.placeholder(tf.float32, shape=[None, None])
            self._features_batch2 = tf.placeholder(tf.float32, shape=[None, None])
            distance_block_16 = _batch_pairwise_distances(
                tf.cast(self._features_batch1, tf.float16),
                tf.cast(self._features_batch2, tf.float16),
            )
            self.distance_block = tf.cond(
                tf.reduce_all(tf.math.is_finite(distance_block_16)),
                lambda: tf.cast(distance_block_16, tf.float32),
                lambda: _batch_pairwise_distances(
                    self._features_batch1, self._features_batch2
                ),
            )

            # Extra logic for less thans.
            self._radii1 = tf.placeholder(tf.float32, shape=[None, None])
            self._radii2 = tf.placeholder(tf.float32, shape=[None, None])
            dist32 = tf.cast(self.distance_block, tf.float32)[..., None]
            self._batch_1_in = tf.math.reduce_any(dist32 <= self._radii2, axis=1)
            self._batch_2_in = tf.math.reduce_any(
                dist32 <= self._radii1[:, None], axis=0
            )

    def pairwise_distances(self, U, V):
        """
        Evaluate pairwise distances between two batches of feature vectors.
        """
        return self.session.run(
            self.distance_block,
            feed_dict={self._features_batch1: U, self._features_batch2: V},
        )

    def less_thans(self, batch_1, radii_1, batch_2, radii_2):
        return self.session.run(
            [self._batch_1_in, self._batch_2_in],
            feed_dict={
                self._features_batch1: batch_1,
                self._features_batch2: batch_2,
                self._radii1: radii_1,
                self._radii2: radii_2,
            },
        )


def _batch_pairwise_distances(U, V):
    """
    Compute pairwise distances between two batches of feature vectors.
    """
    with tf.variable_scope("pairwise_dist_block"):
        # Squared norms of each row in U and V.
        norm_u = tf.reduce_sum(tf.square(U), 1)
        norm_v = tf.reduce_sum(tf.square(V), 1)

        # norm_u as a column and norm_v as a row vectors.
        norm_u = tf.reshape(norm_u, [-1, 1])
        norm_v = tf.reshape(norm_v, [1, -1])

        # Pairwise squared Euclidean distances.
        D = tf.maximum(norm_u - 2 * tf.matmul(U, V, False, True) + norm_v, 0.0)

    return D


class NpzArrayReader(ABC):
    @abstractmethod
    def read_batch(self, batch_size: int) -> Optional[np.ndarray]:
        pass

    @abstractmethod
    def remaining(self) -> int:
        pass

    def read_batches(self, batch_size: int) -> Iterable[np.ndarray]:
        def gen_fn():
            while True:
                batch = self.read_batch(batch_size)
                if batch is None:
                    break
                yield batch

        rem = self.remaining()
        num_batches = rem // batch_size + int(rem % batch_size != 0)
        return BatchIterator(gen_fn, num_batches)


class BatchIterator:
    def __init__(self, gen_fn, length):
        self.gen_fn = gen_fn
        self.length = length

    def __len__(self):
        return self.length

    def __iter__(self):
        return self.gen_fn()


class StreamingNpzArrayReader(NpzArrayReader):
    def __init__(self, arr_f, shape, dtype):
        self.arr_f = arr_f
        self.shape = shape
        self.dtype = dtype
        self.idx = 0

    def read_batch(self, batch_size: int) -> Optional[np.ndarray]:
        if self.idx >= self.shape[0]:
            return None

        bs = min(batch_size, self.shape[0] - self.idx)
        self.idx += bs

        if self.dtype.itemsize == 0:
            return np.ndarray([bs, *self.shape[1:]], dtype=self.dtype)

        read_count = bs * np.prod(self.shape[1:])
        read_size = int(read_count * self.dtype.itemsize)
        data = _read_bytes(self.arr_f, read_size, "array data")
        return np.frombuffer(data, dtype=self.dtype).reshape([bs, *self.shape[1:]])

    def remaining(self) -> int:
        return max(0, self.shape[0] - self.idx)


class MemoryNpzArrayReader(NpzArrayReader):
    def __init__(self, arr):
        self.arr = arr
        self.idx = 0

    @classmethod
    def load(cls, path: str, arr_name: str):
        with open(path, "rb") as f:
            arr = np.load(f)[arr_name]
        return cls(arr)

    def read_batch(self, batch_size: int) -> Optional[np.ndarray]:
        if self.idx >= self.arr.shape[0]:
            return None

        res = self.arr[self.idx : self.idx + batch_size]
        self.idx += batch_size
        return res

    def remaining(self) -> int:
        return max(0, self.arr.shape[0] - self.idx)


@contextmanager
def open_npz_array(path: str, arr_name: str) -> NpzArrayReader:
    with _open_npy_file(path, arr_name) as arr_f:
        version = np.lib.format.read_magic(arr_f)
        if version == (1, 0):
            header = np.lib.format.read_array_header_1_0(arr_f)
        elif version == (2, 0):
            header = np.lib.format.read_array_header_2_0(arr_f)
        else:
            yield MemoryNpzArrayReader.load(path, arr_name)
            return
        shape, fortran, dtype = header
        if fortran or dtype.hasobject:
            yield MemoryNpzArrayReader.load(path, arr_name)
        else:
            yield StreamingNpzArrayReader(arr_f, shape, dtype)


def _read_bytes(fp, size, error_template="ran out of data"):
    """
    Copied from: https://github.com/numpy/numpy/blob/fb215c76967739268de71aa4bda55dd1b062bc2e/numpy/lib/format.py#L788-L886

    Read from file-like object until size bytes are read.
    Raises ValueError if not EOF is encountered before size bytes are read.
    Non-blocking objects only supported if they derive from io objects.
    Required as e.g. ZipExtFile in python 2.6 can return less data than
    requested.
    """
    data = bytes()
    while True:
        # io files (default in python3) return None or raise on
        # would-block, python2 file will truncate, probably nothing can be
        # done about that.  note that regular files can't be non-blocking
        try:
            r = fp.read(size - len(data))
            data += r
            if len(r) == 0 or len(data) == size:
                break
        except io.BlockingIOError:
            pass
    if len(data) != size:
        msg = "EOF: reading %s, expected %d bytes got %d"
        raise ValueError(msg % (error_template, size, len(data)))
    else:
        return data


@contextmanager
def _open_npy_file(path: str, arr_name: str):
    with open(path, "rb") as f:
        with zipfile.ZipFile(f, "r") as zip_f:
            if f"{arr_name}.npy" not in zip_f.namelist():
                raise ValueError(f"missing {arr_name} in npz file")
            with zip_f.open(f"{arr_name}.npy", "r") as arr_f:
                yield arr_f


def _download_inception_model():
    if os.path.exists(INCEPTION_V3_PATH):
        return
    print("downloading InceptionV3 model...")
    with requests.get(INCEPTION_V3_URL, stream=True) as r:
        r.raise_for_status()
        tmp_path = INCEPTION_V3_PATH + ".tmp"
        with open(tmp_path, "wb") as f:
            for chunk in tqdm(r.iter_content(chunk_size=8192)):
                f.write(chunk)
        os.rename(tmp_path, INCEPTION_V3_PATH)


def _create_feature_graph(input_batch):
    _download_inception_model()
    prefix = f"{random.randrange(2**32)}_{random.randrange(2**32)}"
    with open(INCEPTION_V3_PATH, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    pool3, spatial = tf.import_graph_def(
        graph_def,
        input_map={f"ExpandDims:0": input_batch},
        return_elements=[FID_POOL_NAME, FID_SPATIAL_NAME],
        name=prefix,
    )
    _update_shapes(pool3)
    spatial = spatial[..., :7]
    return pool3, spatial


def _create_softmax_graph(input_batch):
    _download_inception_model()
    prefix = f"{random.randrange(2**32)}_{random.randrange(2**32)}"
    with open(INCEPTION_V3_PATH, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    (matmul,) = tf.import_graph_def(
        graph_def, return_elements=[f"softmax/logits/MatMul"], name=prefix
    )
    w = matmul.inputs[1]
    logits = tf.matmul(input_batch, w)
    return tf.nn.softmax(logits)


def _update_shapes(pool3):
    # https://github.com/bioinf-jku/TTUR/blob/73ab375cdf952a12686d9aa7978567771084da42/fid.py#L50-L63
    ops = pool3.graph.get_operations()
    for op in ops:
        for o in op.outputs:
            shape = o.get_shape()
            if shape._dims is not None:  # pylint: disable=protected-access
                # shape = [s.value for s in shape] TF 1.x
                shape = [s for s in shape]  # TF 2.x
                new_shape = []
                for j, s in enumerate(shape):
                    if s == 1 and j == 0:
                        new_shape.append(None)
                    else:
                        new_shape.append(s)
                o.__dict__["_shape_val"] = tf.TensorShape(new_shape)
    return pool3


def _numpy_partition(arr, kth, **kwargs):
    num_workers = min(cpu_count(), len(arr))
    chunk_size = len(arr) // num_workers
    extra = len(arr) % num_workers

    start_idx = 0
    batches = []
    for i in range(num_workers):
        size = chunk_size + (1 if i < extra else 0)
        batches.append(arr[start_idx : start_idx + size])
        start_idx += size

    with ThreadPool(num_workers) as pool:
        return list(pool.map(partial(np.partition, kth=kth, **kwargs), batches))


def read_images_folder(folder_path, exclude_regex=None):
    """
    Reads all images in a folder and stores them into a numpy array.

    Args:
        folder_path (str): Path to the folder containing the images.

    Returns:
        np.ndarray: A numpy array containing all images from the folder.
    """
    import re

    images = []
    filenames = glob.glob(f"{folder_path}/**/*.png", recursive=True)
    if exclude_regex is not None:
        filenames = [filename for filename in filenames if re.search(exclude_regex, filename) is None]
    for filename in tqdm(filenames, desc=f"Reading files from {folder_path} into numpy array."):
        img_path = filename
        img = imread(img_path)
        images.append(img)
    images = np.array(images)

    return images


if __name__ == "__main__":
    main()


# LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES="0,1,2,3" python evaluator.py ./results/cifar10/2024_05_09_002924/fid_samples_guidance_0.0 cifar10_all
# Inception Score: 8.399567604064941
# FID: 5.154592177122993
# sFID: 4.803673987587672
# Precision: 0.55498
# Recall: 0.66798

# LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES="0,1,2,3" python evaluator.py ./results/cifar10/2024_05_09_002924/fid_samples_guidance_2.0 cifar10_all
# Inception Score: 8.3995943069458
# FID: 9.71754947487085
# sFID: 7.465944760590446
# Precision: 0.41098
# Recall: 0.76352


# LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES="4,5,6,7" python evaluator.py ./results/cifar10/2024_06_11_004401/fid_samples_guidance_2.0 cifar10_all
# Inception Score: 8.399556159973145
# FID: 9.027708055264952
# sFID: 7.634165064751073
# Precision: 0.41068
# Recall: 0.78452

# LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES="4,5,6,7" python evaluator.py ./results/cifar10/2024_06_11_004401/fid_samples_guidance_0.0 cifar10_all
# Inception Score: 8.399548530578613
# FID: 4.618358221790459
# sFID: 4.633412555775067
# Precision: 0.55102
# Recall: 0.68948

# LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES="3,5,6,7" python evaluator.py ./results/stl10/2024_06_25_012831/fid_samples_guidance_2.0 stl10_without_label_0 "000_\d{5}\.png"
# stl10_esd.yml
# Inception Score: 12.705127716064453
# FID: 42.88119238318211
# sFID: 38.703155827967635
# sFID: 38.703155827967635
# Precision: 0.25623931623931623
# Recall: 0.5554700854700855

# LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES="4,5,6,7" python evaluator.py ./results/stl10/2024_06_27_155957/fid_samples_guidance_2.0 stl10_without_label_0 "000_\d{5}\.png"
# stl10_retrain.yml (w/ ema)
# Inception Score: 12.557376861572266
# FID: 17.664962080003875
# sFID: 30.44923487670428
# Precision: 0.307008547008547
# Recall: 0.7217094017094017

# LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES="6,7" python evaluator.py ./results/stl10/2024_06_16_173346/fid_samples_guidance_2.0 stl10_without_label_0 "000_\d{5}\.png"
# stl10_retrain.yml (w/o ema)
# Inception Score: 12.705161094665527
# FID: 76.47496862200518
# sFID: 92.18109791219263
# Precision: 0.2152136752136752
# Recall: 0.4896055437100213

# LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES="4,5,6,7" python evaluator.py ./results/stl10/2024_06_15_035238/fid_samples_guidance_0.0 stl10_without_label_0 "000_\d{5}\.png"
# stl10_sid_forget_alpha1.2_sglrx3.yml (w/ ema)
# Inception Score: 12.557475090026855
# FID: 49.28102822937626
# sFID: 37.201969036818355
# Precision: 0.3211965811965812
# Recall: 0.3425641025641026

# LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES="6,7" python evaluator.py ./results/stl10/2024_06_15_035238/fid_samples_guidance_0.0 stl10_without_label_0 "000_\d{5}\.png"
# Inception Score: 12.705291748046875
# FID: 173.4913014341182
# sFID: 124.84489690853115
# Precision: 0.0841025641025641
# Recall: 0.17914529914529914

# LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES="6,7" python evaluator.py "/data/tqchen/Projects/Unlearn-Saliency/DDPM/results/stl10/forget/rl/0.001_full/2024_05_21_071906/fid_samples_guidance_2.0" "stl10_without_label_0" "000_\d{5}\.png"
# Inception Score: 12.705390930175781
# FID: 19.04230730736765
# sFID: 32.80809852068978
# Precision: 0.348034188034188
# Recall: 0.7111111111111111

# rsync -avtr tqchen@10.158.19.8:/data/tqchen/Projects/Unlearn-Saliency/DDPM/results/stl10/2024_06_15_035238/ /data/tqchen/Projects/Unlearn-Saliency/DDPM/results/stl10/2024_06_15_035238/

# LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES="6,7" python evaluator.py ./results/stl10/2024_06_15_035238/fid_samples_guidance_0.0_ema stl10_without_label_0 "000_\d{5}\.png"
# Inception Score: 12.705199241638184
# FID: 49.280770907989165
# sFID: 37.202116474127024
# Precision: 0.3206837606837607
# Recall: 0.34307692307692306

# LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES="6,7" python evaluator.py ./results/stl10/2024_06_04_041704/fid_samples_guidance_0.0 stl10_without_label_0 "000_\d{5}\.png"
# Inception Score: 12.705168724060059
# FID: 41.04703644691841
# sFID: 33.42519075159714
# Precision: 0.3464102564102564
# Recall: 0.4188034188034188

# LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES="6,7" python evaluator.py ./results/stl10/2024_05_12_224610/fid_samples_guidance_0.0 stl10_without_label_0 "000_\d{5}\.png"
# Inception Score: 12.55732250213623
# FID: 14.333271563421306
# sFID: 25.06416879392509
# Precision: 0.5357264957264958
# Recall: 0.655982905982906

# LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES="6,7" python evaluator.py ./results/stl10/2024_06_30_215841/fid_samples_guidance_0.0 stl10_without_label_0 "000_\d{5}\.png"
# distill only
# Inception Score: 12.55733871459961
# FID: 40.916161683919086
# sFID: 33.15333102642751
# Precision: 0.3481196581196581
# Recall: 0.4323931623931624

# LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES="4,5,6,7" python evaluator.py ./results/stl10/2024_07_01_170650/fid_samples_guidance_0.0_ckpt_49999 stl10_without_label_0 "000_\d{5}\.png" > ./results/stl10/2024_07_01_170650/logs/49999.txt
# LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES="4,5,6,7" python evaluator.py ./results/stl10/2024_07_01_170650/fid_samples_guidance_0.0_ckpt_99999 stl10_without_label_0 "000_\d{5}\.png" > ./results/stl10/2024_07_01_170650/logs/99999.txt
# LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES="4,5,6,7" python evaluator.py ./results/stl10/2024_07_01_170650/fid_samples_guidance_0.0_ckpt_149999 stl10_without_label_0 "000_\d{5}\.png" > ./results/stl10/2024_07_01_170650/logs/149999.txt
# LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES="4,5,6,7" python evaluator.py ./results/stl10/2024_07_01_170650/fid_samples_guidance_0.0_ckpt_199999 stl10_without_label_0 "000_\d{5}\.png" > ./results/stl10/2024_07_01_170650/logs/199999.txt
# LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES="4,5,6,7" python evaluator.py ./results/stl10/2024_07_01_170650/fid_samples_guidance_0.0_ckpt_249999 stl10_without_label_0 "000_\d{5}\.png" > ./results/stl10/2024_07_01_170650/logs/249999.txt
# Step-49999
# Inception Score: 12.557351112365723

# FID: 40.600541040464975
# sFID: 33.120474243907324
# Precision: 0.35384615384615387
# Recall: 0.42914529914529914
# Step-99999
# Inception Score: 12.557557106018066
# FID: 46.29778533815863
# sFID: 35.48869393435666
# Precision: 0.32803418803418805
# Recall: 0.3751282051282051
# Step-149999
# Inception Score: 12.557387351989746
# FID: 47.748488654699145
# sFID: 35.63793185799375
# Precision: 0.34376068376068375
# Recall: 0.3360683760683761
# Step-199999
# Inception Score: 12.557421684265137
# FID: 48.653686779928364
# sFID: 36.7704442179471
# Precision: 0.3466666666666667
# Recall: 0.29555555555555557
# Step-249999
# Inception Score: 12.557293891906738
# FID: 49.4588156000878
# sFID: 35.18958572156657
# Precision: 0.3358119658119658
# Recall: 0.3333333333333333

# LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES="4,5,6,7" python evaluator.py ./results/stl10/2024_06_27_155957/fid_samples_guidance_0.0 stl10_without_label_0 "000_\d{5}\.png"
# Inception Score: 12.557296752929688
# FID: 26.311940152601267
# sFID: 29.726963879349796
# Precision: 0.4829059829059829
# Recall: 0.5835897435897436

# LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES="4,5,6,7" python evaluator.py ./results/stl10/2024_06_15_035238/fid_samples_guidance_0.5 stl10_without_label_0 "000_\d{5}\.png"
# Inception Score: 12.766864776611328
# FID: 51.21219755566801
# sFID: 37.20468894218811
# Precision: 0.305982905982906
# Recall: 0.3205982905982906

# LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES="4,5,6,7" python evaluator.py ./results/stl10/2024_06_15_035238/fid_samples_guidance_0.0_ckpt_9999 stl10_without_label_0 "000_\d{5}\.png"

# LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES="4,5,6,7" python evaluator.py ./results/stl10/2024_07_03_123622/fid_samples_guidance_0.0 stl10_without_label_0 "000_\d{5}\.png"
# Inception Score: 12.766685485839844
# FID: 40.73052413959044
# sFID: 31.692057028952718
# Precision: 0.3611111111111111
# Recall: 0.4395726495726496


# LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES="4,5,6,7" python evaluator.py ./results/cifar10/2024_07_15_221705/fid_samples_guidance_0.0 cifar10_without_label_0 "000_\d{5}\.png"


# LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES="4,5,6,7" python evaluator.py ./results/cifar10/2024_07_21_183729/fid_samples_guidance_0.0 cifar10_without_label_0 "000_\d{5}\.png"


# LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES="4,5,6,7" python evaluator.py ./results/cifar10/2024_05_09_002924/fid_samples_guidance_0.0 cifar10_without_label_0 "000_\d{5}\.png"
# Inception Score: 8.197938919067383
# FID: 5.1312658347428055
# sFID: 5.08630016608879
# Precision: 0.5556666666666666
# Recall: 0.6736444444444445


# python save_base_dataset.py --dataset cifar10 --label_to_forget 0 --num_samples_per_class 5000

# LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES="4,5,6,7" python evaluator.py ./cifar10_without_label_0 ./results/cifar10/2024_07_21_183729/fid_samples_guidance_0.0 "000_\d{5}\.png"

# LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES="4,5,6,7" python evaluator.py ./results/cifar10_debug cifar10_without_label_0 "000_\d{5}\.png"


# LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES="4,5,6,7" python evaluator.py cifar10_all ./results/cifar10/2024_05_09_002924/fid_samples_guidance_0.0

# LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES="4,5,6,7" python evaluator.py ./cifar10_without_label_0 ./results/cifar10/2024_07_21_183729/fid_samples_guidance_0.0_excluded_class_0
# more samples (almost no diff)
# Inception Score: 7.308907985687256
# FID: 13.914010668437868
# sFID: 9.226368418801371
# Precision: 0.6288686868686869
# Recall: 0.4700666666666667

# LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES="4,5,6,7" python evaluator.py ./cifar10_without_label_0 ./results/cifar10/2024_07_21_183729/fid_samples_guidance_0.0_seed123_excluded_class_0
# more samples (almost no diff)
# Inception Score: 7.308907985687256
# FID: 13.914010668437868
# sFID: 9.226368418801371
# Precision: 0.6288686868686869
# Recall: 0.4700666666666667

# LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES="4" python evaluator.py ./cifar10_without_label_0 ./results/cifar10/2024_07_21_183729/fid_samples_guidance_0.0 "000_\d{5}\.png"
# Inception Score: 9.139715194702148
# FID: 5.674606732936468
# sFID: 5.056453424536812
# Precision: 0.65788
# Recall: 0.58296

# LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES="4,5,6,7" python evaluator.py ./cifar10_without_label_0 ./results/cifar10/2024_07_21_183729/fid_samples_guidance_0.0 "000_\d{5}\.png"
# Inception Score: 9.302041053771973
# FID: 11.203927203785895
# sFID: 9.50684669109944
# Precision: 0.6312222222222222
# Recall: 0.4998666666666667


# LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES="4,5,6,7" python evaluator.py cifar10_without_label_0 ./results/cifar10_debug
# Inception Score: 9.33872127532959
# FID: 5.31688462344448
# sFID: 7.510807510788595
# Precision: 0.6589074488491049
# Recall: 0.5469333333333334

# LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES="4,5,6,7" python evaluator.py cifar10_without_label_0 ./results/cifar10/2024_07_21_183729/fid_samples_guidance_0.0 "000_\d{5}\.png"
# Image.save()
# Inception Score: 9.302335739135742
# FID: 11.203111370282954
# sFID: 9.507485341895176
# Precision: 0.6313111111111112
# Recall: 0.4999111111111111

# LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES="4,5,6,7" python evaluator.py cifar10_without_label_0 ./results/cifar10/2024_07_21_183729/fid_samples_guidance_0.0 "000_\d{5}\.png"
# 5500 per class
# Inception Score: 9.248886108398438
# FID: 10.788009776202216
# sFID: 9.337035426783586
# Precision: 0.6328686868686869
# Recall: 0.49844444444444447

# isn't about the order

# 440
# tensor(0.3711, device='cuda:0')
# tensor(0.9286, device='cuda:0')
# tensor(2.5023, device='cuda:0')
# {
#   "init_t": 440,
#   "init_a": 0.3710930347442627,
#   "init_b": 0.9285957217216492,
#   "init_sigma": 2.5023259257926713
# }

# LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES="4,5,6,7" python evaluator.py cifar10_without_label_0 ./results/cifar10/2024_07_21_183729/fid_samples_guidance_0.0 "000_\d{5}\.png"
# Inception Score: 9.554271697998047
# FID: 5.389408530128605
# sFID: 7.7236534369463925
# Precision: 0.6607777777777778
# Recall: 0.5456444444444445

# LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES="4,5,6,7" python evaluator.py stl10_without_label_0 ./results/stl10/2024_06_15_035238/fid_samples_guidance_0.0 "000_\d{5}\.png"
# stl10_sid_forget_alpha1.2_sglrx3.yml
# Inception Score: 10.103686332702637
# FID: 41.199654638809534
# sFID: 36.60200807243916
# Precision: 0.3847863247863248
# Recall: 0.4066666666666667

# LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES="4,5,6,7" python evaluator.py stl10_without_label_0 ./results/stl10/2024_06_04_041704/fid_samples_guidance_0.0 "000_\d{5}\.png"
# stl10_sid_forget_alpha1.2.yml
# Inception Score: 10.338909149169922
# FID: 34.14374398989844
# sFID: 32.891179150489734
# Precision: 0.4605982905982906
# Recall: 0.4354700854700855


# LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES="4,5,6,7" python evaluator.py stl10_without_label_0 ./results/stl10/forget/rl/0.001_full/2024_06_29_183825/fid_samples_guidance_0.0_ckpt_19999 "000_\d{5}\.png"

# LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES="4,5,6,7" python evaluator.py cifar10_without_label_0 ./results/cifar10/2024_05_09_002924/fid_samples_guidance_0.0 "000_\d{5}\.png"
# Inception Score: 9.073158264160156
# FID: 4.841240295510886
# sFID: 6.373423136171823
# Precision: 0.6923555555555555
# Recall: 0.5525555555555556

# LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES="4,5,6,7" python evaluator.py cifar10_without_label_0 ./results/cifar10/2024_05_09_002924/fid_samples_guidance_2.0 "000_\d{5}\.png"
# Inception Score: 9.438050270080566
# FID: 12.59937644328545
# sFID: 9.684805506585008
# Precision: 0.8046888888888889
# Recall: 0.36646666666666666

# LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES="4,5,6,7" python evaluator.py cifar10_without_label_0 ./results/cifar10/2024_07_21_183729/fid_samples_guidance_0.0 "000_\d{5}\.png"
# cifar10_sid_forget_alpha1.2_sglrx3_ema.yml
# Inception Score: 9.534753799438477
# FID: 5.324725926494693
# sFID: 7.696774859268089
# Precision: 0.6587111111111111
# Recall: 0.5470888888888888

# LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES="4,5,6,7" python evaluator.py stl10_without_label_0 ./results/stl10/2024_06_25_012831/fid_samples_guidance_2.0_ckpt_49999 "000_\d{5}\.png"
# stl10_esd.yml
# Inception Score: 5.610227108001709
# FID: 139.6841139365239
# sFID: 71.15282666822839
# Precision: 0.1576923076923077
# Recall: 0.18914529914529915

# LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES="4,5,6,7" python evaluator.py stl10_without_label_0 ./results/stl10/2024_06_25_012831/fid_samples_guidance_2.0_ckpt_19999 "000_\d{5}\.png"
# stl10_esd.yml
# Inception Score: 8.77419376373291
# FID: 75.34796469920315
# sFID: 57.0493642537449
# Precision: 0.32188034188034187
# Recall: 0.2760683760683761

# LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES="4,5,6,7" python evaluator.py stl10_without_label_0 ./results/stl10/2024_06_25_012831/fid_samples_guidance_2.0 "000_\d{5}\.png"
# stl10_esd.yml 9999 (old buggy ver.)
# Inception Score: 10.170247077941895
# FID: 39.37165030286667
# sFID: 39.27293755434607
# Precision: 0.5314529914529914
# Recall: 0.288034188034188

# LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES="4,5,6,7" python evaluator.py stl10_without_label_0 ./results/stl10/2024_06_25_012831/fid_samples_guidance_0.0_ckpt_9999 "000_\d{5}\.png"
# stl10_esd.yml 9999 (old buggy ver.)
# Inception Score: 6.524508476257324
# FID: 97.37787940866042
# sFID: 45.232968133598206
# Precision: 0.33493333333333336
# Recall: 0.2963247863247863

# LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES="4,5,6,7" python evaluator.py stl10_without_label_0 ./results/stl10/forget/rl/0.001_full/2024_06_29_183825/fid_samples_guidance_0.0_ckpt_19999 "000_\d{5}\.png"
# stl10_saliency_unlearn.yml (19999)
# Inception Score: 10.695719718933105
# FID: 11.020921463131458
# sFID: 18.347670398936884
# Precision: 0.6413777777777778
# Recall: 0.5277777777777778

# LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES="6,7" python evaluator.py "stl10_without_label_0" "./results/stl10/forget/rl/0.001_full/2024_06_29_183825/fid_samples_guidance_2.0_ckpt_19999" "000_\d{5}\.png"

# LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES="6,7" python evaluator.py stl10_without_label_0 ./results/stl10/2024_06_25_012831/fid_samples_guidance_0.0_ckpt_9999 "000_\d{5}\.png"
# stl10_esd.yml (9999/0.0)
# Inception Score: 6.5244927406311035
# FID: 97.37778777843232
# sFID: 45.232959382240665
# Precision: 0.3348888888888889
# Recall: 0.29572649572649573

# LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES="6,7" python evaluator.py stl10_without_label_0 ./results/stl10/2024_06_25_012831/fid_samples_guidance_2.0_ckpt_9999 "000_\d{5}\.png"
# # stl10_esd.yml (9999/2.0)
# Inception Score: 10.163376808166504
# FID: 39.32349332359769
# sFID: 41.11249815931865
# Precision: 0.5229059829059829
# Recall: 0.2898290598290598

# LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES="4,5,6,7" python evaluator.py stl10_without_label_0 ./results/stl10/2024_06_25_012831/fid_samples_guidance_0.0_ckpt_19999 "000_\d{5}\.png"
# stl10_esd.yml (19999/0.0)
# Inception Score: 4.675250053405762
# FID: 143.69660383725824
# sFID: 72.55174341958389
# Precision: 0.2071111111111111
# Recall: 0.2592307692307692

# LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES="6,7" python evaluator.py stl10_without_label_0 ./results/stl10/2024_06_25_012831/fid_samples_guidance_2.0_ckpt_19999 "000_\d{5}\.png"
# stl10_esd.yml (19999/2.0)
# Inception Score: 8.755633354187012
# FID: 74.16204655790142
# sFID: 57.765735044038365
# Precision: 0.33931623931623933
# Recall: 0.26871794871794874

# LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES="6,7" python evaluator.py stl10_without_label_0 ./results/stl10/2024_06_25_012831/fid_samples_guidance_0.0_ckpt_49999 "000_\d{5}\.png"
# stl10_esd.yml (49999/0.0)
# Inception Score: 2.9437739849090576
# FID: 209.69964003463633
# sFID: 89.69944460364786
# Precision: 0.13837777777777777
# Recall: 0.13384615384615384

# LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES="6,7" python evaluator.py stl10_without_label_0 ./results/stl10/2024_06_25_012831/fid_samples_guidance_2.0_ckpt_49999 "000_\d{5}\.png"
# stl10_esd.yml
# Inception Score: 5.1842942237854
# FID: 143.95468091289405
# sFID: 70.44358506348067
# Precision: 0.16197131026323572
# Recall: 0.1552991452991453

# LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES="4,5,6,7" python evaluator.py "stl10_without_label_0" "./results/stl10/2024_05_12_224610/fid_samples_guidance_0.0" "000_\d{5}\.png"
# pretrained (cond_scale 0.0)
# Inception Score: 10.554375648498535
# FID: 10.076642058260973
# sFID: 18.61724640740283
# Precision: 0.666
# Recall: 0.5064957264957265

# LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES="4,5,6,7" python evaluator.py "stl10_without_label_0" "./results/stl10/2024_05_12_224610/fid_samples_guidance_2.0" "000_\d{5}\.png"
# pretrained (cond_scale 2.0)
# Inception Score: 11.132503509521484
# FID: 16.660119117249735
# sFID: 24.304269323364224
# Precision: 0.8156444444444444
# Recall: 0.2839316239316239

# LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES="4,5,6,7" python evaluator.py "stl10_without_label_0" "./results/stl10/2024_05_12_224610/fid_samples_guidance_2.0" "000_\d{5}\.png"
# pretrained

# LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES="4,5,6,7" python evaluator.py stl10_without_label_0 ./results/stl10/forget/rl/0.001_full/2024_06_29_183825/fid_samples_guidance_0.0 "000_\d{5}\.png"
# stl10_saliency_unlearn.yml (step-49999 cond_scale 0.0)
# Inception Score: 10.808262825012207
# FID: 11.522320233770188
# sFID: 18.519952149449182
# Precision: 0.6364
# Recall: 0.5222222222222223

# LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES="4,5,6,7" python evaluator.py stl10_without_label_0 ./results/stl10/forget/rl/0.001_full/2024_06_29_183825/fid_samples_guidance_0.0 "000_\d{5}\.png"
# stl10_saliency_unlearn.yml (step-49999 cond_scale 2.0)
# Inception Score: 11.709370613098145
# FID: 14.518809695007064
# sFID: 24.23877998326782
# Precision: 0.7999111111111111
# Recall: 0.3294017094017094

# LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES="4,5,6,7" python evaluator.py stl10_without_label_0 ./results/stl10/2024_07_14_102349/fid_samples_guidance_0.0 "000_\d{5}\.png"
# stl10_sid_forget_alpha1.2_sglrx3_lsg.yml
# Inception Score: 11.538372993469238
# FID: 16.650475669739592
# sFID: 28.8630773218606
# Precision: 0.5921367521367521
# Recall: 0.3943589743589744
# (1300 -> 5000)
# Inception Score: 11.458431243896484
# FID: 15.439691276363817
# sFID: 21.396015533149352
# Precision: 0.5983111111111111
# Recall: 0.35512820512820514

# LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES="4,5,6,7" python evaluator.py stl10_without_label_0 ./results/stl10/2024_08_03_021143/fid_samples_guidance_0.0 "000_\d{5}\.png"
# stl10_sid_forget_alpha1.2_sglrx3_ema.yml
# Inception Score: 10.85281753540039
# FID: 19.740916324126886
# sFID: 28.02161526380371
# Precision: 0.5635042735042735
# Recall: 0.4481196581196581

# LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES="0,1,2,3" python evaluator.py stl10_without_label_0 ./results/stl10/2024_06_27_155957/fid_samples_guidance_0.0 "000_\d{5}\.png"
# stl10_retrain
# Inception Score: 8.303309440612793
# FID: 26.515639013463215
# sFID: 38.03729597869665
# Precision: 0.5572649572649573
# Recall: 0.4526495726495727

# LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES="6,7" python evaluator.py stl10_without_label_0 ./results/stl10/2024_06_27_155957/fid_samples_guidance_0.0_noema "000_\d{5}\.png"
# stl10_retrain (no_ema)
# Inception Score: 8.554924011230469
# FID: 50.48934753037531
# sFID: 43.09561809578827
# Precision: 0.41504273504273503
# Recall: 0.46384615384615385

# LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES="4,5,6,7" python evaluator.py stl10_without_label_0 ./results/stl10/forget/rl/0.001_full/2024_08_02_214225/fid_samples_guidance_0.0_19999 "000_\d{5}\.png"
# stl10_saliency_unlearn_noema.yml (step-19999 cond_scale 0.0)
# Inception Score: 10.429718017578125
# FID: 20.70528971577744
# sFID: 29.221537304864228
# Precision: 0.5647008547008547
# Recall: 0.5368376068376068

# LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES="0,1,2,3" python evaluator.py stl10_without_label_0 ./results/stl10/forget/rl/0.001_full/2024_08_02_214225/fid_samples_guidance_0.0_noema "000_\d{5}\.png"
# stl10_saliency_unlearn_noema.yml (step-49999 cond_scale 0.0)
# Inception Score: 10.88608455657959
# FID: 20.78496139916831
# sFID: 30.620101448445666
# Precision: 0.5712820512820512
# Recall: 0.5414529914529914

# LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES="0,1,2,3" python evaluator.py stl10_without_label_0 ./results/stl10/forget/rl/0.001_full/2024_08_02_214225/fid_samples_guidance_2.0_noema "000_\d{5}\.png"

# LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES="4,5,6,7" python evaluator.py stl10_without_label_0 ./results/stl10/forget/rl/0.001_full/2024_08_02_214225/fid_samples_guidance_2.0_noema "000_\d{5}\.png"
# stl10_saliency_unlearn_noema.yml (step-49999 cond_scale 2.0)
# Inception Score: 10.434341430664062
# FID: 27.065545007284754
# sFID: 37.71363838103605
# Precision: 0.7375824175824176
# Recall: 0.3487179487179487


# CUDA_VISIBLE_DEVICES="4,5,6,7" python evaluator.py ./image_logs/exp_unseen_fid/baseline/inv_normal/original_images ./image_logs/exp_unseen_fid/baseline/inv_normal/generated_images

# LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES="0,1,2,3" python evaluator.py ./cifar10_without_label_0 ./results/cifar10/2024_05_09_002924/_fid_samples_guidance_0.0 "000_\d{5}\.png"
# Inception Score: 9.027972221374512
# FID: 5.66329312005621
# sFID: 5.315983375115366
# Precision: 0.6631555555555556
# Recall: 0.5842888888888889

# LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES="0,1,2,3" python evaluator.py ./cifar10_without_label_0 ./results/cifar10/2024_05_09_002924/_fid_samples_guidance_0.0 "000_\d{5}\.png"
# Inception Score: 9.71778392791748
# FID: 10.383779701044375
# sFID: 7.502929614714617
# Precision: 0.7832444444444444
# Recall: 0.4081111111111111

# LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES="0,1,2,3" python evaluator.py cifar10_without_label_0 ./results/cifar10/2024_05_09_002924/fid_samples_guidance_0.0_noema "000_\d{5}\.png"
# Inception Score: 8.959577560424805
# FID: 5.598052735364092
# sFID: 6.571520590005662
# Precision: 0.6663555555555556
# Recall: 0.5699333333333333

# LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES="0,1,2,3" python evaluator.py cifar10_without_label_0 ./results/cifar10/2024_05_09_002924/fid_samples_guidance_0.0 "000_\d{5}\.png"
# FID: 4.891432131329566
# sFID: 6.461894537895546
# Precision: 0.6916888888888889
# Recall: 0.5527111111111112

# LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES="0,1,2,3" python evaluator.py cifar10_without_label_0 ./results/cifar10/forget/rl/0.0_full/2024_05_10_075057/fid_samples_guidance_2.0_excluded_class_0 "000_\d{5}\.png"

# LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES="0,1,2,3" python evaluator.py cifar10_without_label_0 results/cifar10/forget/rl/0.001_full/2024_09_28_131547/fid_samples_guidance_2.0 "000_\d{5}\.png"
# SalUn 1000 CIFAR10 reproduction
# Inception Score: 9.406073570251465
# FID: 11.250104997496805
# sFID: 11.61882089817766
# Precision: 0.7805555555555556
# Recall: 0.3176

# LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES="0,1,2,3" python evaluator.py cifar10_without_label_0 results/cifar10/2024_09_29_125342/fid_samples_guidance_2.0 "000_\d{5}\.png"
# ESD CIFAR-10
# Inception Score: 9.775150299072266
# FID: 12.6830467181818
# sFID: 7.821285669386498
# Precision: 0.7709333333333334
# Recall: 0.38482222222222223

# LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES="0,1,2,3" python evaluator.py cifar10_without_label_0 results/cifar10/2024_09_29_212743/fid_samples_guidance_0.0 "000_\d{5}\.png"
# Retrain CIFAR-10
# Inception Score: 8.343452453613281
# FID: 7.937922210696797
# sFID: 7.214987013696373
# Precision: 0.6417555555555555
# Recall: 0.5203111111111111

# LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES="0,1,2,3" python evaluator.py stl10_without_label_0 results/stl10/2024_08_03_021143/fid_samples_guidance_0.0 "000_\d{5}\.png"
# Ours STL-10
# Inception Score: 10.852901458740234
# FID: 19.74088587958647
# sFID: 28.021630306655766
# Precision: 0.563076923076923
# Recall: 0.44666666666666666
# -> 5000 per class
# Inception Score: 10.92637825012207
# FID: 18.67988657860451
# sFID: 20.380385562595052
# Precision: 0.5543333333333333
# Recall: 0.4053846153846154

