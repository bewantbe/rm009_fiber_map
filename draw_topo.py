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

logger_m = logging.getLogger('trimesh')
logger_m.setLevel(logging.WARNING)
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

def LoadSwcDir(swc_dir, parallel_lib = 'joblib'):
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
        logger.info('Loading SWC files in parallel ...')
        results_batch = joblib.Parallel(n_jobs = 4, verbose=2) \
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

def ShowMeshBatch(plotter, mesh_list):
    for mesh in mesh_list:
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

def ReadErwinData():
    dir_path = 'ref/Erwin1999'
    volume_dims = (240, 280, 320)  # medial-lateral, dorsal-ventral, anterior-posterior
    voxel_size_um = (25, 25, 25)
    # Horsley-Clarke position of the origin
    origin_mm = (8.5, -1.5, 3.5)  # lateral, dorsal, anterior
    # data format: binary little-endian int16 array
    # read ECC.DAT, eccentricity map
    with open(os.path.join(dir_path, 'ECC.DAT'), 'rb') as f:
        data = np.fromfile(f, dtype=np.int16)
        ecc = data.reshape(volume_dims, order='F').astype(np.float32)
        # Rounded to the nearest 0.1°, then multiplied by 10
        # Extralaminar space is coded 999
        ecc[ecc!=999] /= 10
        ecc[ecc==999] = np.nan
        #ecc[ecc==999] = 0
    # read INCL.DAT, inclination map
    with open(os.path.join(dir_path, 'INCL.DAT'), 'rb') as f:
        data = np.fromfile(f, dtype=np.int16)
        incl = data.reshape(volume_dims, order='F').astype(np.float32)
        # rounded to the nearest degree
        # Extralaminar space is coded 999
    # read LAYERS.DAT, laminar map, 8-bit integers
    with open(os.path.join(dir_path, 'LAYERS.DAT'), 'rb') as f:
        data = np.fromfile(f, dtype=np.uint8)
        layers = data.reshape(volume_dims, order='F')
        # 1 = contra magno; 2 = ipsi magno; 3 = ipsi parvo; 4 = contra parvo
        # Extralaminar space is coded zero
    # read CELLS.DAT, cell density map, 16-bit integers
    with open(os.path.join(dir_path, 'CELLS.DAT'), 'rb') as f:
        data = np.fromfile(f, dtype=np.int16)
        cells = data.reshape(volume_dims, order='F').astype(np.float32)
        # cells/voxel multiplied by 1000
        cells = cells / 1000
    erwin_data = Struct(
        ecc = ecc,
        incl = incl,
        layers = layers,
        cells = cells,
        origin_mm = origin_mm,
        voxel_size_um = voxel_size_um
    )
    return erwin_data

def PlotInPyvista(swcs_a, pos_soma, lgn_mesh_s, soma_layer_i):
    """Main ploting function"""
    plotter = pv.Plotter()
    # plot soma
    cloud = pv.PolyData(pos_soma)
    cloud['layer'] = soma_layer_i
    plotter.add_mesh(cloud, color="red", point_size = 10.0,
                    render_points_as_spheres = True,
                    scalars = 'layer', cmap = 'viridis')
    # plot LGN
    ShowMeshBatch(plotter, lgn_mesh_s[1:2])
    plotter.show_axes()

    # plot 2D map of soma on LGN layer(s)
    frame_manifold = GetLgnSoma2DCoordinate(pos_soma, lgn_mesh_s)

    # plot reference manifold

    #plotter.show()       # Press 'q' for quit

    erwin_data = ReadErwinData()

    ml_idx = 120
    plt.figure(10)
    plt.imshow(erwin_data.ecc[ml_idx, :, :])
    plt.colorbar()
    plt.title(f'eccentricity, medial-lateral idx={ml_idx}')
    plt.xlabel('anterior-posterior')
    plt.ylabel('dorsal-ventral')
    OutputFigure(f'ecc_map_ml{ml_idx}.png')

    ap_idx = 120
    plt.figure(11)
    plt.imshow(erwin_data.ecc[:, :, ap_idx])
    plt.colorbar()
    plt.title(f'inclination, anterior-posterior idx={ap_idx}')
    plt.xlabel('medial-lateral')
    plt.ylabel('dorsal-ventral')
    OutputFigure(f'ecc_map_ap{ap_idx}.png')

    ap_idx = 120
    plt.figure(21)
    plt.imshow(erwin_data.layers[:, :, ap_idx])
    plt.colorbar()
    plt.title(f'laminar type, anterior-posterior idx={ap_idx}')
    plt.xlabel('medial-lateral')
    plt.ylabel('dorsal-ventral')
    OutputFigure(f'layers_map_ap{ap_idx}.png')

    # TODO:  we need to plot 3D map in 2D with a slider for slicing
    # maybe use pyvista or PyQtGraph
    # python -m pyqtgraph.examples
    #   find the data slicing example

    map_render_mode = 'pyvista'
    if map_render_mode == 'pyqtgraph':
        # the pyqtgraph way
        import pyqtgraph as pg
        from pyqtgraph.Qt import QtWidgets

        app = pg.mkQApp("Data Slicing Example")
        win = QtWidgets.QMainWindow()
        win.resize(800,800)
        # TODO: add a slider for slicing

    elif map_render_mode == 'pyvista':
        # the pyvista way
        img_pv = pv.wrap(erwin_data.ecc)
        ss = img_pv.slice_along_axis(n=7, axis="z")
        ss.plot(cmap="viridis", opacity=0.75)
    else:
        logger.warning('No map rendering mode specified!')


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

    ## load LGN mesh, 0:LGN, 1-6:LGN layers 1-6
    lgn_mesh_s = LoadLGNMesh()
    soma_layer_i = np.zeros(len(pos_soma), dtype=np.int32)
    for i_layer in range(len(lgn_mesh_s)):
        bidx = lgn_mesh_s[i_layer].contains(pos_soma)
        # note that we must put LGN to the first mesh,
        # to be overwritten by detailed layers
        soma_layer_i[bidx] = i_layer

    ## plat soma with LGN mesh
    plot_mode = 'pyvista'
    if plot_mode == 'plt':
        PlotInMatplotlib(swcs_a, pos_soma, lgn_mesh_s, soma_layer_i)
    elif plot_mode == 'pyvista':
        PlotInPyvista(swcs_a, pos_soma, lgn_mesh_s, soma_layer_i)
    else:
        pass




