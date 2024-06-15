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

def GetLgnSoma2DCoordinate(pos_soma, lgn_mesh_list):
    # define manifold that represent projected layer of a LGN layer
    # for simplicity, here use a planer mesh
    idx_layer = 3
    # define the origin of the manifold in the world coordinate
    origin_pos = np.mean(lgn_mesh_list[idx_layer].vertices, axis = 0)
    # define coordinate frame at the origin of the manifold (plane)
    origin_normal = np.array([0, 1, 0])
    origin_x_direction = np.array([1, 0, 0])
    origin_y_direction = np.cross(origin_normal, origin_x_direction)
    # get projection of the soma position on the manifold
    pos_soma_2d = (pos_soma - origin_pos).dot(
        np.c_[origin_x_direction, origin_y_direction])

    plt.figure(10)
    plt.plot(pos_soma_2d[:, 0], pos_soma_2d[:, 1], 'o')
    #plt.show()
    OutputFigure('soma_lgn_map.png')
    plt.cla()
    plt.clf()

    frame_manifold = Struct(
        type = 'plane',
        origin = origin_pos,
        normal = origin_normal,
        x_axis = origin_x_direction,
        y_axis = origin_y_direction,
        x_range = [-10000, 10000],
        y_range = [-10000, 10000],
    )

    return frame_manifold

def PlotInMatplotlib(swcs_a, pos_soma, lgn_mesh_s):
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

def PlotInPyvista(swcs_a, pos_soma, lgn_mesh_s):
    plotter = pv.Plotter()
    # plot soma
    cloud = pv.PolyData(pos_soma)
    plotter.add_mesh(cloud, color="red", point_size = 10.0,
                    render_points_as_spheres = True)
    # plot LGN
    ShowLGNPv(plotter)
    plotter.show_axes()

    # plot 2D map of soma on LGN layer(s)
    frame_manifold = GetLgnSoma2DCoordinate(pos_soma, lgn_mesh_s)

    # plot reference manifold

    plotter.show()       # Press 'q' for quit

OutputFigure = lambda fn: plt.savefig(os.path.join('./pic', fn))

if __name__ == '__main__':
    swc_dir = './rm009_swcs'

    ## load and check SWCs
    swcs_ext = LoadSwcDir(swc_dir)
    swcs_ext = SortSwcsList(swcs_ext)
    swcs_a = ArrayfyList(swcs_ext, index_list = 
            [s.neu_id for s in swcs_ext]
        )
    CheckDuplicateName(swcs_a)

    ## filter soma in LGN
    # locate soma
    pos_soma = np.array([s.ntree[1][0, 0:3] for s in swcs_a])
    idx_valid = pos_soma[:, 2] < 40000
    swcs_a = swcs_a[idx_valid]
    pos_soma = pos_soma[idx_valid]
    # location of soma
    pos_soma2 = np.array([s.ntree[1][0, 0:3] for s in swcs_a])
    assert not np.any(pos_soma2 - pos_soma)

    ## load LGN mesh
    lgn_mesh_s = LoadLGNMesh()
    idx_neu_l1 = lgn_mesh_s[1].contains(pos_soma)

    ## plat soma with LGN mesh
    plot_mode = 'pyvista'
    if plot_mode == 'plt':
        PlotInMatplotlib(swcs_a, pos_soma, lgn_mesh_s)
    elif plot_mode == 'pyvista':
        PlotInPyvista(swcs_a, pos_soma, lgn_mesh_s)
    else:
        pass




