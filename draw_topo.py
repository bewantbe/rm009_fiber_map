# draw topological map

import os
import sys
import glob
import logging

from os.path import (
    basename,
    splitext
)
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)
"""
logger.debug('This message should go to the log file')
logger.info('So should this')
logger.warning('And this, too')
logger.error('And non-ASCII stuff, too, like Øresund and Malmö')
"""

import joblib

logger_m = logging.getLogger('matplotlib')
logger_m.setLevel(logging.WARNING)
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import (
    figure,
    scatter
)

import trimesh

import pyvista as pv

sys.path.insert(0, '/home/xyy/code/py/vtk_test')
sys.path.insert(0, '/home/xyy/code/py/fiber-inspector')
sys.path.insert(0, '/home/xyy/code/py/neurite_walker')
sys.path.insert(0, '../SimpleVolumeViewer')
sys.path.insert(0, '../neurite_walker')
sys.path.insert(0, '../fiber-inspector')

from neu3dviewer.data_loader import (
    LoadSWCTree,
    SplitSWCTree,
    SWCDFSSort,
    dtype_id,
    dtype_coor
)

from neu3dviewer.utils import (
    get_num_in_str,
    contain_int,
    Struct,
    ArrayfyList
)

# mostly copy from the ObjTranslator.obj_swc.LoadRawSwc 
# in "cg_translators.py".
def LoadRawSwc(file_path):
    ntree = LoadSWCTree(file_path)
    processes = SplitSWCTree(ntree)
    ntree, processes = SWCDFSSort(ntree, processes)
    
    raw_points = ntree[1][:,0:3].astype(dtype_coor, copy=False)

    # extract neuron ID
    # e.g. filename "neuron#96.lyp.swc", the ID is 96
    swc_name = splitext(basename(file_path))[0]
    if contain_int(swc_name):
        neu_id = str(get_num_in_str(swc_name))  # neu_id always str
    else:
        neu_id = swc_name

    swc_ext = Struct(
        neu_id     = neu_id,
        ntree      = ntree,
        processes  = processes,
        raw_points = raw_points
    )

    return swc_ext

def BatchLoadSwc(swc_pathes):
    results = [None] * len(swc_pathes)
    for i, f in enumerate(swc_pathes):
        logger.debug(f'Loading "{f}" ...')
        results[i] = LoadRawSwc(f)
    return results

def LoadSwcDir(swc_dir, parallel_lib = 'jobliba'):
    if not os.path.isdir(swc_dir):
        logger.error(f'"{swc_dir}" is not a directory!')
        raise FileNotFoundError(f'"{swc_dir}" is not a directory!')
    fn_s = glob.glob(swc_dir + '/*.swc')
    if len(fn_s) == 0:
        logger.error(f'"{swc_dir}" do not contain SWC file!')
        return

    # split the file pathes into batched list
    swc_batch_list = []
    batch_size = 1
    k = 0
    while k < len(fn_s):
        swc_batch_list.append(fn_s[k : k + batch_size])
        k += batch_size

    # run in parallel or serial
    if parallel_lib == 'joblib':
        results_batch = joblib.Parallel(n_jobs = 4) \
            (joblib.delayed(BatchLoadSwc)(j)
                for j in swc_batch_list)
        # unpack the batch
        results = []
        for b in results_batch:
            results.extend(b)
    else:
        results = BatchLoadSwc(fn_s)

    print('len(results):', len(results))

    return results

def SortSwcsList(swcs_ext):
    # sort according to neu_id
    swcs_ext = sorted(swcs_ext, key =                                  \
            lambda x: str(type(x.neu_id)) + str(x.neu_id).zfill(10)
        )

    return swcs_ext

def CheckDuplicateName(swcs_ext):
    # https://stackoverflow.com/questions/9835762/how-do-i-find-the-duplicates-in-a-list-and-create-another-list-with-them
    seen = set()
    dupes = [x for x in swcs_a.neu_id if x in seen or seen.add(x)]
    if len(dupes) > 0:
        logger.error('Duplication found!')

def LoadLGNMesh():
    mesh_dir = './rm009_mesh/region/LGN/LGN_Layers/'
    mesh_fn_list = ['../LGNright120.obj', 'lgn_1.obj', 'lgn_2.obj', 'lgn_3.obj', 'lgn_4.obj', 'lgn_5.obj', 'lgn_6.obj']
    mesh_list = []
    for fn in mesh_fn_list:
        fn_path = os.path.join(mesh_dir, fn)
        mesh = trimesh.load_mesh(fn_path)
        mesh.vertices *= 10.0
        mesh_list.append(mesh)
    return mesh_list

def ShowLGNPlt(ax, show_layers = [0]):
    mesh_list = LoadLGNMesh()
    for idx_l in show_layers:
        mesh = mesh_list[idx_l]
        x, y, z = mesh.vertices.T
        ax.plot_trisurf(x, y, z, triangles=mesh.faces,
            color='grey', alpha=0.1)

def ShowLGNPv(plotter, show_layers = [0]):
    mesh_list = LoadLGNMesh()
    for idx_l in show_layers:
        mesh = mesh_list[idx_l]
        pv_mesh = pv.PolyData(mesh.vertices,
                    np.c_[[[3]]*len(mesh.faces),mesh.faces])
        plotter.add_mesh(pv_mesh, smooth_shading=True,
                        opacity = 0.1)

if __name__ == '__main__':
    swc_dir = './rm009_swcs'
    swcs_ext = LoadSwcDir(swc_dir)

    swcs_ext = SortSwcsList(swcs_ext)

    swcs_a = ArrayfyList(swcs_ext, index_list = 
            [s.neu_id for s in swcs_ext]
        )
    CheckDuplicateName(swcs_a)

    # locate soma
    pos_soma = np.array([s.ntree[1][0, 0:3] for s in swcs_a])

    # filter these in LGN
    idx_valid = pos_soma[:, 2] < 40000

    swcs_a = swcs_a[idx_valid]
    pos_soma = pos_soma[idx_valid]

    # locate soma
    pos_soma2 = np.array([s.ntree[1][0, 0:3] for s in swcs_a])
    assert not np.any(pos_soma2 - pos_soma)

    plot_mode = 'pyvista'
    if plot_mode == 'plt':
        # plot soma position
        plt.ion()
        fig = plt.figure(1)
        fig.clf()
        ax = fig.add_subplot(111, projection='3d')
        ax.cla()
        ax.scatter(pos_soma[:, 0], pos_soma[:, 1], pos_soma[:, 2])
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ShowLGNPlt(ax)
    elif plot_mode == 'pyvista':
        plotter = pv.Plotter()
        # plot soma
        cloud = pv.PolyData(pos_soma)
        plotter.add_mesh(cloud, color="red", point_size = 10.0,
                        render_points_as_spheres = True)
        # plot LGN
        ShowLGNPv(plotter)
        plotter.show_axes()
        plotter.show()       # Press 'q' for quit

    mesh_list = LoadLGNMesh()
    idx_neu_l1 = mesh_list[1].contains(pos_soma)




