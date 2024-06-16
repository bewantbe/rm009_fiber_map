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
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import (
    figure,
    scatter,
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
    SimplifyTreeWithDepth,
    dtype_id,
    dtype_coor,
)

from neu3dviewer.utils import (
    get_num_in_str,
    contain_int,
    Struct,
    ArrayfyList,
    VecNorm,
)

logger_m = logging.getLogger('numcodecs')
logger_m.setLevel(logging.WARNING)
from neu_walk import (
    NTreeOps,
    exec_filter_string,
)

# mostly copy from the ObjTranslator.obj_swc.LoadRawSwc 
# in "cg_translators.py".
def LoadRawSwc(file_path):
    ntree_ext = NTreeOps(file_path, sort_proc = True)
    
    raw_points = ntree_ext.ntree[1][:,0:3].astype(dtype_coor, copy=False)

    # extract neuron ID
    # e.g. filename "neuron#96.lyp.swc", the ID is 96
    swc_name = splitext(basename(file_path))[0]
    if contain_int(swc_name):
        neu_id = str(get_num_in_str(swc_name))  # neu_id always str
    else:
        neu_id = swc_name

    swc_ext = Struct(
        file_path  = file_path,
        neu_id     = neu_id,
        ntree_ext  = ntree_ext,
        ntree      = ntree_ext.ntree,
        processes  = ntree_ext.processes,
        raw_points = raw_points,
    )

    return swc_ext

def BatchLoadSwc(swc_pathes):
    results = [None] * len(swc_pathes)
    for i, f in enumerate(swc_pathes):
        logger.debug(f'Loading "{f}" ...')
        results[i] = LoadRawSwc(f)
    return results

def is_interactive():
    #try:
    #    shell = get_ipython().__class__.__name__
    #    if shell == 'ZMQInteractiveShell':
    #        return True   # Jupyter notebook or qtconsole
    #    elif shell == 'TerminalInteractiveShell':
    #        return False  # Terminal running IPython
    #    else:
    #        return False  # Other type (?)
    #except NameError:
    #    return False      # Probably standard Python interpreter
    return hasattr(__builtins__,'__IPYTHON__')

def LoadSwcDir(swc_dir, parallel_lib = 'auto'):
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

    if parallel_lib == 'auto':
        if is_interactive():
            parallel_lib = 'serial'
        else:
            parallel_lib = 'joblib'

    # run in parallel or serial
    if parallel_lib == 'joblib':
        logger.info('Loading SWC files in parallel ...')
        results_batch = joblib.Parallel(n_jobs = 8, verbose=2) \
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

def LoadV1Mesh():
    mesh_dir = './rm009_mesh/region/V1/'
    mesh_fn = '3.obj'
    mesh_fn_path = os.path.join(mesh_dir, mesh_fn)
    mesh = trimesh.load_mesh(mesh_fn_path)
    mesh.vertices *= 10.0
    return mesh

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

def Normalize(vector):
    norm = np.linalg.norm(vector)
    if norm == 0: 
        return vector  # Return the original vector if its norm is 0
    return vector / norm

def ProjectNormal(vec, normal):
    normal = Normalize(normal)
    return vec - np.dot(vec, normal) * normal

def GetLgnSoma2DCoordinate(pos_soma, lgn_mesh_list):
    # define manifold that represent projected layer of a LGN layer
    # for simplicity, here use a planer mesh
    idx_layer = 3
    # define the origin of the manifold in the world coordinate
    origin_pos = np.mean(lgn_mesh_list[idx_layer].vertices, axis = 0)
    # define coordinate frame at the origin of the manifold (plane)
    origin_normal = Normalize(np.array([-0.7, 1, 0.9]))
    origin_x_direction = np.array([1, 0, -0.3])
    origin_x_direction = Normalize(ProjectNormal(origin_x_direction, origin_normal))
    origin_y_direction = np.cross(origin_normal, origin_x_direction)
    # get projection of the soma position on the manifold
    pos_soma_2d = (pos_soma - origin_pos).dot(
        np.c_[origin_x_direction, origin_y_direction])

    ## draw the 2D map
    plt.figure(10)
    plt.plot(pos_soma_2d[:, 0], pos_soma_2d[:, 1], 'o')
    plt.xlabel('Inclination')
    plt.ylabel('Eccentricity')
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
        x_range = [-5000, 5000],
        y_range = [-5000, 5000],
    )

    return frame_manifold, pos_soma_2d

def PlotInMatplotlib(swcs_a, pos_soma, lgn_mesh_s, v1_mesh):
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

def PlotInPyvista(swcs_a, pos_soma, lgn_mesh_s, soma_layer_i, v1_mesh):
    """Main ploting function"""
    plotter = pv.Plotter()

    ## plot soma
    cloud = pv.PolyData(pos_soma)
    cloud['layer'] = soma_layer_i
    cmap = matplotlib.colormaps['Paired']
    cmap = matplotlib.colors.ListedColormap(cmap.colors[:7])
    sargs = dict(
        title_font_size=20,
        label_font_size=16,
        shadow=True,
        n_labels=7,
        italic=False,
        fmt="%.1g",
        font_family="arial",
    )
    plotter.add_mesh(cloud, color="red", point_size = 20.0,
                    render_points_as_spheres = True,
                    scalars = 'layer', clim = [0, 6.1],
                    cmap = cmap, scalar_bar_args=sargs)

    ## plot LGN layer mesh
    ShowMeshBatch(plotter, lgn_mesh_s[1:2])
    plotter.show_axes()

    ## plot 2D map reference of LGN layer(s)
    frame_manifold, pos_soma_2d = GetLgnSoma2DCoordinate(pos_soma, lgn_mesh_s)
    # plot reference manifold
    frame_geo = pv.Plane(
            center = frame_manifold.origin,
            direction = frame_manifold.normal,
            i_size = frame_manifold.x_range[1],
            j_size = frame_manifold.y_range[1])
    plotter.add_mesh(frame_geo, opacity=0.7)
    # plot coordinate frame
    arrow_scale = frame_manifold.x_range[1] * 0.35
    frame_geo_normal = pv.Arrow(
            start = frame_manifold.origin,
            direction = frame_manifold.normal,
            scale = arrow_scale)
    frame_geo_x = pv.Arrow(
            start = frame_manifold.origin,
            direction = frame_manifold.x_axis,
            scale = arrow_scale)
    frame_geo_y = pv.Arrow(
            start = frame_manifold.origin,
            direction = frame_manifold.y_axis,
            scale = arrow_scale)
    plotter.add_mesh(frame_geo_normal, color='blue')
    plotter.add_mesh(frame_geo_x, color='red')
    plotter.add_mesh(frame_geo_y, color='green')

    plotter.show()       # Press 'q' for quit

    ## plot V1 and swc terminal positions
    plotter = pv.Plotter()
    ShowMeshBatch(plotter, [v1_mesh])
    point_set = []
    for ntr in swcs_a.ntree_ext:
        b_leaves = exec_filter_string('path_length_to_root(leaves) > 0', ntr)
        point_set.append(ntr.position_of_node(ntr.leaves[b_leaves]))
    point_set = np.concatenate(point_set, axis = 0)
    cloud = pv.PolyData(point_set)
    plotter.add_mesh(cloud, color="red", point_size = 20.0,
                    render_points_as_spheres = True)
    ShowMeshBatch(plotter, lgn_mesh_s[1:2])
    plotter.show_axes()
    plotter.show()

    ## LGN reference map
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

    map_render_mode = 'none'
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
    elif map_render_mode == 'none':
        pass
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

    ## load V1 mesh
    v1_mesh = LoadV1Mesh()

    ## plat soma with LGN mesh
    plot_mode = 'pyvista'
    if plot_mode == 'plt':
        PlotInMatplotlib(swcs_a, pos_soma, lgn_mesh_s, soma_layer_i, v1_mesh)
    elif plot_mode == 'pyvista':
        PlotInPyvista(swcs_a, pos_soma, lgn_mesh_s, soma_layer_i, v1_mesh)
    else:
        pass




