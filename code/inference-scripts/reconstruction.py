import os
import numpy as np
from pathlib import Path
from copy import deepcopy
import pycolmap
import pandas as pd

import sqlite3
from PIL import Image, ExifTags
import h5py
from tqdm import tqdm
import warnings
import pickle
import json

# import sys
# sys.path.append("/mnt/2ndHDD/kaggle/IMC2024/datas/input/colmap-db-import")
# from database import *
# from h5_to_db import *

# def import_into_colmap(
#     path: Path,
#     feature_dir: Path,
#     database_path: str = "colmap.db",
# ) -> None:
#     """Adds keypoints into colmap"""
#     db = COLMAPDatabase.connect(database_path)
#     db.create_tables()
#     single_camera = False
#     fname_to_id = add_keypoints(db, feature_dir, path, "", "simple-pinhole", single_camera)
#     add_matches(
#         db,
#         feature_dir,
#         fname_to_id,
#     )
#     db.commit()

def arr_to_str(a):
    """Returns ;-separated string representing the input"""
    return ";".join([str(x) for x in a.reshape(-1)])

MAX_IMAGE_ID = 2**31 - 1


CREATE_CAMERAS_TABLE = """CREATE TABLE IF NOT EXISTS cameras (
    camera_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
    model INTEGER NOT NULL,
    width INTEGER NOT NULL,
    height INTEGER NOT NULL,
    params BLOB,
    prior_focal_length INTEGER NOT NULL)"""


CREATE_DESCRIPTORS_TABLE = """CREATE TABLE IF NOT EXISTS descriptors (
    image_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB,
    FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE)"""


CREATE_IMAGES_TABLE = """CREATE TABLE IF NOT EXISTS images (
    image_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
    name TEXT NOT NULL UNIQUE,
    camera_id INTEGER NOT NULL,
    prior_qw REAL,
    prior_qx REAL,
    prior_qy REAL,
    prior_qz REAL,
    prior_tx REAL,
    prior_ty REAL,
    prior_tz REAL,
    CONSTRAINT image_id_check CHECK(image_id >= 0 and image_id < {}),
    FOREIGN KEY(camera_id) REFERENCES cameras(camera_id))
""".format(MAX_IMAGE_ID)


CREATE_TWO_VIEW_GEOMETRIES_TABLE = """
CREATE TABLE IF NOT EXISTS two_view_geometries (
    pair_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB,
    config INTEGER NOT NULL,
    F BLOB,
    E BLOB,
    H BLOB)
"""


CREATE_KEYPOINTS_TABLE = """CREATE TABLE IF NOT EXISTS keypoints (
    image_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB,
    FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE)
"""


CREATE_MATCHES_TABLE = """CREATE TABLE IF NOT EXISTS matches (
    pair_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB)"""


CREATE_NAME_INDEX = \
    "CREATE UNIQUE INDEX IF NOT EXISTS index_name ON images(name)"


CREATE_ALL = "; ".join([
    CREATE_CAMERAS_TABLE,
    CREATE_IMAGES_TABLE,
    CREATE_KEYPOINTS_TABLE,
    CREATE_DESCRIPTORS_TABLE,
    CREATE_MATCHES_TABLE,
    CREATE_TWO_VIEW_GEOMETRIES_TABLE,
    CREATE_NAME_INDEX
])


def image_ids_to_pair_id(image_id1, image_id2):
    if image_id1 > image_id2:
        image_id1, image_id2 = image_id2, image_id1
    return image_id1 * MAX_IMAGE_ID + image_id2


def array_to_blob(array):
    return array.tostring()


class COLMAPDatabase(sqlite3.Connection):

    @staticmethod
    def connect(database_path):
        return sqlite3.connect(database_path, factory=COLMAPDatabase)

    def __init__(self, *args, **kwargs):
        super(COLMAPDatabase, self).__init__(*args, **kwargs)

        self.create_tables = lambda: self.executescript(CREATE_ALL)
        self.create_cameras_table = \
            lambda: self.executescript(CREATE_CAMERAS_TABLE)
        self.create_descriptors_table = \
            lambda: self.executescript(CREATE_DESCRIPTORS_TABLE)
        self.create_images_table = \
            lambda: self.executescript(CREATE_IMAGES_TABLE)
        self.create_two_view_geometries_table = \
            lambda: self.executescript(CREATE_TWO_VIEW_GEOMETRIES_TABLE)
        self.create_keypoints_table = \
            lambda: self.executescript(CREATE_KEYPOINTS_TABLE)
        self.create_matches_table = \
            lambda: self.executescript(CREATE_MATCHES_TABLE)
        self.create_name_index = lambda: self.executescript(CREATE_NAME_INDEX)

    def add_camera(self, model, width, height, params,
                   prior_focal_length=0, camera_id=None):
        params = np.asarray(params, np.float64)
        cursor = self.execute(
            "INSERT INTO cameras VALUES (?, ?, ?, ?, ?, ?)",
            (camera_id, model, width, height, array_to_blob(params),
             prior_focal_length))
        return cursor.lastrowid

    def add_image(self, name, camera_id,
                  prior_q=np.zeros(4), prior_t=np.zeros(3), image_id=None):
        cursor = self.execute(
            "INSERT INTO images VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (image_id, name, camera_id, prior_q[0], prior_q[1], prior_q[2],
             prior_q[3], prior_t[0], prior_t[1], prior_t[2]))
        return cursor.lastrowid

    def add_keypoints(self, image_id, keypoints):
        assert(len(keypoints.shape) == 2)
        assert(keypoints.shape[1] in [2, 4, 6])

        keypoints = np.asarray(keypoints, np.float32)
        self.execute(
            "INSERT INTO keypoints VALUES (?, ?, ?, ?)",
            (image_id,) + keypoints.shape + (array_to_blob(keypoints),))

    def add_matches(self, image_id1, image_id2, matches):
        assert(len(matches.shape) == 2)
        assert(matches.shape[1] == 2)
        if image_id1 > image_id2:
            matches = matches[:,::-1]
        pair_id = image_ids_to_pair_id(image_id1, image_id2)
        matches = np.asarray(matches, np.uint32)
        self.execute(
            "INSERT INTO matches VALUES (?, ?, ?, ?)",
            (pair_id,) + matches.shape + (array_to_blob(matches),))

    def add_two_view_geometry(self, image_id1, image_id2, matches, F=np.eye(3), E=np.eye(3), H=np.eye(3), config=2):
        assert(len(matches.shape) == 2)
        assert(matches.shape[1] == 2)
        if image_id1 > image_id2:
            matches = matches[:,::-1]
        pair_id = image_ids_to_pair_id(image_id1, image_id2)
        matches = np.asarray(matches, np.uint32)
        F = np.asarray(F, dtype=np.float64)
        E = np.asarray(E, dtype=np.float64)
        H = np.asarray(H, dtype=np.float64)
        self.execute(
            "INSERT INTO two_view_geometries VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (pair_id,) + matches.shape + (array_to_blob(matches), config,
             array_to_blob(F), array_to_blob(E), array_to_blob(H)))

        
def get_focal(height, width, exif):
    max_size = max(height, width)
    focal_found, focal = False, None
    if exif is not None:
        focal_35mm = None
        for tag, value in exif.items():
            focal_35mm = None
            if ExifTags.TAGS.get(tag, None) == 'FocalLengthIn35mmFilm':
                focal_35mm = float(value)
                break
        if focal_35mm is not None:
            focal_found = True
            focal = focal_35mm / 35. * max_size
            print(f"Focal found: {focal}")
    if focal is None:
        FOCAL_PRIOR = 1.2
        focal = FOCAL_PRIOR * max_size
    return focal_found, focal


def create_camera(db, height, width, exif, camera_model):
    focal_found, focal = get_focal(height, width, exif)
    if camera_model == 'simple-pinhole':
        model = 0 # simple pinhole
        param_arr = np.array([focal, width / 2, height / 2])
    if camera_model == 'pinhole':
        model = 1 # pinhole
        param_arr = np.array([focal, focal, width / 2, height / 2])
    elif camera_model == 'simple-radial':
        model = 2 # simple radial
        param_arr = np.array([focal, width / 2, height / 2, 0.1])
    elif camera_model == 'radial':
        model = 3 # radial
        param_arr = np.array([focal, width / 2, height / 2, 0., 0.])
    elif camera_model == 'opencv':
        model = 4 # opencv
        param_arr = np.array([focal, focal, width / 2, height / 2, 0., 0., 0., 0.])
    return db.add_camera(model, width, height, param_arr, prior_focal_length=int(focal_found))


def add_keypoints(db, feature_dir, h_w_exif, camera_model, single_camera=False):
    keypoint_f = h5py.File(os.path.join(feature_dir, 'keypoints.h5'), 'r')
    camera_id = None
    fname_to_id = {}
    for filename in tqdm(list(keypoint_f.keys())):
        keypoints = keypoint_f[filename][()]
        if camera_id is None or not single_camera:
            height = h_w_exif[filename]['h']
            width = h_w_exif[filename]['w']
            exif = h_w_exif[filename]['exif']
            camera_id = create_camera(db, height, width, exif, camera_model)
        image_id = db.add_image(filename, camera_id)
        fname_to_id[filename] = image_id
        db.add_keypoints(image_id, keypoints)
    return fname_to_id


def add_matches_and_fms(db, feature_dir, fname_to_id, fms):
    match_file = h5py.File(os.path.join(feature_dir, 'matches.h5'), 'r')
    added = set()
    for key_1 in match_file.keys():
        group = match_file[key_1]
        for key_2 in group.keys():
            id_1 = fname_to_id[key_1]
            id_2 = fname_to_id[key_2]
            pair_id = (id_1, id_2)
            if pair_id in added:
                warnings.warn(f'Pair {pair_id} ({id_1}, {id_2}) already added!')
                continue
            added.add(pair_id)
            matches = group[key_2][()]
            db.add_matches(id_1, id_2, matches)
            db.add_two_view_geometry(id_1, id_2, matches, fms[(key_1, key_2)])


def import_into_colmap(feature_dir, h_w_exif, fms):
    db = COLMAPDatabase.connect(f"{feature_dir}/colmap.db")
    db.create_tables()
    fname_to_id = add_keypoints(db, feature_dir, h_w_exif, camera_model='simple-radial', single_camera=False)
    add_matches_and_fms(db, feature_dir, fname_to_id, fms)
    db.commit()
    db.close()




import math
_EPS = np.finfo(float).eps * 4.0

def quaternion_matrix(quaternion):
    """Return homogeneous rotation matrix from quaternion."""
    q = np.array(quaternion, dtype=np.float64, copy=True)
    n = np.dot(q, q)
    if n < _EPS:
        # print("special case")
        return np.identity(4)
    q *= math.sqrt(2.0 / n)
    q = np.outer(q, q)
    return np.array(
        [
            [
                1.0 - q[2, 2] - q[3, 3],
                q[1, 2] - q[3, 0],
                q[1, 3] + q[2, 0],
                0.0,
            ],
            [
                q[1, 2] + q[3, 0],
                1.0 - q[1, 1] - q[3, 3],
                q[2, 3] - q[1, 0],
                0.0,
            ],
            [
                q[1, 3] - q[2, 0],
                q[2, 3] + q[1, 0],
                1.0 - q[1, 1] - q[2, 2],
                0.0,
            ],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


def vector_norm(data, axis=None, out=None):
    """Return length, i.e. Euclidean norm, of ndarray along axis."""
    data = np.array(data, dtype=np.float64, copy=True)
    if out is None:
        if data.ndim == 1:
            return math.sqrt(np.dot(data, data))
        data *= data
        out = np.atleast_1d(np.sum(data, axis=axis))
        np.sqrt(out, out)
        return out
    data *= data
    np.sum(data, axis=axis, out=out)
    np.sqrt(out, out)
    return None

def affine_matrix_from_points(v0, v1, shear=False, scale=True, usesvd=True):
    """Return affine transform matrix to register two point sets.
    v0 and v1 are shape (ndims, -1) arrays of at least ndims non-homogeneous
    coordinates, where ndims is the dimensionality of the coordinate space.
    If shear is False, a similarity transformation matrix is returned.
    If also scale is False, a rigid/Euclidean traffansformation matrix
    is returned.
    By default the algorithm by Hartley and Zissermann [15] is used.
    If usesvd is True, similarity and Euclidean transformation matrices
    are calculated by minimizing the weighted sum of squared deviations
    (RMSD) according to the algorithm by Kabsch [8].
    Otherwise, and if ndims is 3, the quaternion based algorithm by Horn [9]
    is used, which is slower when using this Python implementation.
    The returned matrix performs rotation, translation and uniform scaling
    (if specified)."""

    v0 = np.array(v0, dtype=np.float64, copy=True)
    v1 = np.array(v1, dtype=np.float64, copy=True)

    ndims = v0.shape[0]
    if ndims < 2 or v0.shape[1] < ndims or v0.shape != v1.shape:
        raise ValueError("input arrays are of wrong shape or type")

    # move centroids to origin
    t0 = -np.mean(v0, axis=1)
    M0 = np.identity(ndims + 1)
    M0[:ndims, ndims] = t0
    v0 += t0.reshape(ndims, 1)
    t1 = -np.mean(v1, axis=1)
    M1 = np.identity(ndims + 1)
    M1[:ndims, ndims] = t1
    v1 += t1.reshape(ndims, 1)

    if shear:
        # Affine transformation
        A = np.concatenate((v0, v1), axis=0)
        u, s, vh = np.linalg.svd(A.T)
        vh = vh[:ndims].T
        B = vh[:ndims]
        C = vh[ndims : 2 * ndims]
        t = np.dot(C, np.linalg.pinv(B))
        t = np.concatenate((t, np.zeros((ndims, 1))), axis=1)
        M = np.vstack((t, ((0.0,) * ndims) + (1.0,)))
    elif usesvd or ndims != 3:
        # Rigid transformation via SVD of covariance matrix
        u, s, vh = np.linalg.svd(np.dot(v1, v0.T))
        # rotation matrix from SVD orthonormal bases
        R = np.dot(u, vh)
        if np.linalg.det(R) < 0.0:
            # R does not constitute right handed system
            R -= np.outer(u[:, ndims - 1], vh[ndims - 1, :] * 2.0)
            s[-1] *= -1.0
        # homogeneous transformation matrix
        M = np.identity(ndims + 1)
        M[:ndims, :ndims] = R
    else:
        # Rigid transformation matrix via quaternion
        # compute symmetric matrix N
        xx, yy, zz = np.sum(v0 * v1, axis=1)
        xy, yz, zx = np.sum(v0 * np.roll(v1, -1, axis=0), axis=1)
        xz, yx, zy = np.sum(v0 * np.roll(v1, -2, axis=0), axis=1)
        N = [
            [xx + yy + zz, 0.0, 0.0, 0.0],
            [yz - zy, xx - yy - zz, 0.0, 0.0],
            [zx - xz, xy + yx, yy - xx - zz, 0.0],
            [xy - yx, zx + xz, yz + zy, zz - xx - yy],
        ]
        # quaternion: eigenvector corresponding to most positive eigenvalue
        w, V = np.linalg.eigh(N)
        q = V[:, np.argmax(w)]
        # print (vector_norm(q), np.linalg.norm(q))
        q /= vector_norm(q)  # unit quaternion
        # homogeneous transformation matrix
        M = quaternion_matrix(q)

    if scale and not shear:
        # Affine transformation; scale is ratio of RMS deviations from centroid
        v0 *= v0
        v1 *= v1
        M[:ndims, :ndims] *= math.sqrt(np.sum(v1) / np.sum(v0))

    # move centroids back
    M = np.dot(np.linalg.inv(M1), np.dot(M, M0))
    M /= M[ndims, ndims]
    return M



def register_by_Horn(ev_coord, gt_coord, ransac_threshold, inl_cf, strict_cf):
    """Return the best similarity transforms T that registers 3D points pt_ev in <ev_coord> to
    the corresponding ones pt_gt in <gt_coord> according to a RANSAC-like approach for each
    threshold value th in <ransac_threshold>.

    Given th, each triplet of 3D correspondences is examined if not already present as strict inlier,
    a correspondence is a strict inlier if <strict_cf> * err_best < th, where err_best is the registration
    error for the best model so far.
    The minimal model given by the triplet is then refined using also its inliers if their total is greater
    than <inl_cf> * ninl_best, where ninl_best is th number of inliers for the best model so far. Inliers
    are 3D correspondences (pt_ev, pt_gt) for which the Euclidean distance |pt_gt-T*pt_ev| is less than th.
    """

    # remove invalid cameras, the index is returned
    idx_cams = np.all(np.isfinite(ev_coord), axis=0)
    ev_coord = ev_coord[:, idx_cams]
    gt_coord = gt_coord[:, idx_cams]

    # initialization
    n = ev_coord.shape[1]
    r = ransac_threshold.shape[0]
    ransac_threshold = np.expand_dims(ransac_threshold, axis=0)
    ransac_threshold2 = ransac_threshold**2
    ev_coord_1 = np.vstack((ev_coord, np.ones(n)))
    max_no_inl = np.zeros((1, r))
    best_inl_err = np.full(r, np.Inf)
    best_transf_matrix = np.zeros((r, 4, 4))
    best_err = np.full((n, r), np.Inf)
    strict_inl = np.full((n, r), False)
    triplets_used = np.zeros((3, r))

    # run on camera triplets
    for ii in range(n - 2):
        for jj in range(ii + 1, n - 1):
            for kk in range(jj + 1, n):
                i = [ii, jj, kk]
                triplets_used_now = np.full((n), False)
                triplets_used_now[i] = True
                # if both ii, jj, kk are strict inliers for the best current model just skip
                if np.all(strict_inl[i]):
                    continue
                # get transformation T by Horn on the triplet camera center correspondences
                transf_matrix = affine_matrix_from_points(
                    ev_coord[:, i], gt_coord[:, i], usesvd=False
                )
                # apply transformation T to test camera centres
                rotranslated = np.matmul(transf_matrix[:3], ev_coord_1)
                # compute error and inliers
                err = np.sum((rotranslated - gt_coord) ** 2, axis=0)
                inl = np.expand_dims(err, axis=1) < ransac_threshold2
                no_inl = np.sum(inl, axis=0)
                # if the number of inliers is close to that of the best model so far, go for refinement
                to_ref = np.squeeze(
                    ((no_inl > 2) & (no_inl > max_no_inl * inl_cf)), axis=0
                )
                for q in np.argwhere(to_ref):
                    qq = q[0]
                    if np.any(
                        np.all(
                            (np.expand_dims(inl[:, qq], axis=1) == inl[:, :qq]), axis=0
                        )
                    ):
                        # already done for this set of inliers
                        continue
                    # get transformation T by Horn on the inlier camera center correspondences
                    transf_matrix = affine_matrix_from_points(
                        ev_coord[:, inl[:, qq]], gt_coord[:, inl[:, qq]]
                    )
                    # apply transformation T to test camera centres
                    rotranslated = np.matmul(transf_matrix[:3], ev_coord_1)
                    # compute error and inliers
                    err_ref = np.sum((rotranslated - gt_coord) ** 2, axis=0)
                    err_ref_sum = np.sum(err_ref, axis=0)
                    err_ref = np.expand_dims(err_ref, axis=1)
                    inl_ref = err_ref < ransac_threshold2
                    no_inl_ref = np.sum(inl_ref, axis=0)
                    # update the model if better for each threshold
                    to_update = np.squeeze(
                        (no_inl_ref > max_no_inl)
                        | ((no_inl_ref == max_no_inl) & (err_ref_sum < best_inl_err)),
                        axis=0,
                    )
                    if np.any(to_update):
                        triplets_used[0, to_update] = ii
                        triplets_used[1, to_update] = jj
                        triplets_used[2, to_update] = kk
                        max_no_inl[:, to_update] = no_inl_ref[to_update]
                        best_err[:, to_update] = np.sqrt(err_ref)
                        best_inl_err[to_update] = err_ref_sum
                        strict_inl[:, to_update] = (
                            best_err[:, to_update]
                            < strict_cf * ransac_threshold[:, to_update]
                        )
                        best_transf_matrix[to_update] = transf_matrix
    
    for i in range(r):
        print(
            f"Registered cameras {int(max_no_inl[0, i])}/{n} for threshold {ransac_threshold[0, i]}"
        )

    best_model = {
        "valid_cams": idx_cams,
        "no_inl": max_no_inl,
        "err": best_err,
        "triplets_used": triplets_used,
        "transf_matrix": best_transf_matrix,
    }    
    return best_model


def reconstruction(data_dict, dataset, scene, base_path, work_dir, colmap_mapper_options):
    # Import keypoint distances of matches into colmap for RANSAC 
    images_dir = data_dict[dataset][scene][0].parent
    with open(os.path.join(work_dir, "h_w_exif.json"), "r") as f:
        h_w_exif = json.load(f)
    
    with open(os.path.join(work_dir, "fms.pkl"), "rb") as f:
        fms = pickle.load(f)
    import_into_colmap(feature_dir=work_dir, h_w_exif=h_w_exif, fms=fms)

    database_path = f"{work_dir}/colmap.db"
    mapper_options = pycolmap.IncrementalPipelineOptions(**colmap_mapper_options)
    output_path = f"{work_dir}/colmap_rec_aliked"
    os.makedirs(output_path, exist_ok=True)

    db = COLMAPDatabase.connect(database_path)
    cursor = db.execute("SELECT image_id, name from images")
    db_data = cursor.fetchall()
    image_ids = [int(x[0]) for x in db_data]
    names = [str(x[1]) for x in db_data]
    db.close()

    df = pd.read_csv(f"{work_dir}/image_pair.csv")
    images_matches = {}
    for name in names:
        images_matches[name] = [0, 0]
    for i, row in df.iterrows():
        key1 = row["key1"]
        key2 = row["key2"]
        match_num = row["match_num"]
        images_matches[key1][0] += 1
        images_matches[key1][1] += match_num
        images_matches[key2][0] += 1
        images_matches[key2][1] += match_num
    sorted_images_matches = sorted(images_matches.items(), key=lambda item: (item[1][0], item[1][1]), reverse=True)
    init_image_name1 = sorted_images_matches[0][0]
    init_image_id1 = image_ids[names.index(init_image_name1)]
    mapper_options.init_image_id1 = init_image_id1

    maps = pycolmap.incremental_mapping(database_path=database_path, image_path=images_dir, output_path=output_path, options=mapper_options)
    print(maps)

    # 2. Look for the best reconstruction: The incremental mapping offered by 
    # pycolmap attempts to reconstruct multiple models, we must pick the best one
    images_registered  = 0
    best_idx = None
    
    print ("Looking for the best reconstruction")

    if isinstance(maps, dict):
        for idx1, rec in maps.items():
            print(idx1, rec.summary())
            try:
                if len(rec.images) > images_registered:
                    images_registered = len(rec.images)
                    best_idx = idx1
            except Exception:
                continue

    # Parse the reconstruction object to get the rotation matrix and translation vector
    # obtained for each image in the reconstruction
    results = {}
    camid_im_map = {}
    if best_idx is not None:
        for k, im in maps[best_idx].images.items():
            key = os.path.join(images_dir, im.name)
            results[key] = {}
            results[key]["R"] = deepcopy(im.cam_from_world.rotation.matrix())
            results[key]["t"] = deepcopy(np.array(im.cam_from_world.translation))

            camid_im_map[im.camera_id] = im.name

    try:
        if best_idx is not None:
            for idx1, rec in maps.items():
                u_cameras = []
                g_cameras = []
                if idx1 != best_idx:
                    for k, im in rec.images.items():
                        key = os.path.join(images_dir, im.name)
                        if key in results:
                            g_R = deepcopy(results[key]["R"])
                            g_t = deepcopy(results[key]["t"])
                            g_C = -g_R.T @ g_t

                            u_R = deepcopy(im.cam_from_world.rotation.matrix())
                            u_t = deepcopy(np.array(im.cam_from_world.translation))
                            u_C = -u_R.T @ u_t
                            g_cameras.append(g_C.reshape(3, 1))
                            u_cameras.append(u_C.reshape(3, 1))
                    if len(g_cameras) < 3:
                        continue
                    g_cameras = np.array(g_cameras).reshape(3, -1)
                    u_cameras = np.array(u_cameras).reshape(3, -1)
                    inl_cf = 0
                    strict_cf = -1
                    thresholds = np.array([0.025, 0.05, 0.1, 0.2, 0.5, 1.0])
                    model = register_by_Horn(
                        u_cameras, g_cameras,
                        np.asarray(thresholds), inl_cf, strict_cf
                    )
                    T = np.squeeze(model["transf_matrix"][-1])
                    # print(T)
                    # print(T[:3].shape)
                    for k, im in rec.images.items():
                        key = os.path.join(images_dir, im.name)
                        if key not in results:
                            Tcw2 = np.eye(4)
                            Tcw2[:3, :3] = deepcopy(im.cam_from_world.rotation.matrix())
                            Tcw2[:3, 3] = deepcopy(np.array(im.cam_from_world.translation))
                            Tw2c = np.linalg.inv(Tcw2)
                            Tw1c = np.matmul(T, Tw2c)
                            Tcw1 = np.linalg.inv(Tw1c)
                            results[key]["R"] = deepcopy(Tcw1[:3, :3])
                            results[key]["t"] = deepcopy(Tcw1[:3, 3])
    except:
        pass
    # with open(os.path.join(work_dir, "camid_im_map.json"), "w") as f:
    #     json.dump(camid_im_map, f, indent=2)
    
    print(f"Registered: {dataset} / {scene} -> {len(results)} images")
    print(f"Total: {dataset} / {scene} -> {len(data_dict[dataset][scene])} images")

    # Create Submission
    submission = {
        "image_path": [],
        "dataset": [],
        "scene": [],
        "rotation_matrix": [],
        "translation_vector": []
    }
    for image in data_dict[dataset][scene]:
        if str(image) in results:
            print(image)
            R = results[str(image)]["R"].reshape(-1)
            T = results[str(image)]["t"].reshape(-1)
        else:
            R = np.eye(3).reshape(-1)
            T = np.zeros((3))
        image_path = str(image.relative_to(base_path))
        
        submission["image_path"].append(image_path)
        submission["dataset"].append(dataset)
        submission["scene"].append(scene)
        submission["rotation_matrix"].append(arr_to_str(R))
        submission["translation_vector"].append(arr_to_str(T))
    
    submission_df = pd.DataFrame.from_dict(submission)
    submission_path = os.path.join(work_dir, "submission.csv")
    print(f"Save to {submission_path}")
    submission_df.to_csv(submission_path, index=False)
    return submission_path
