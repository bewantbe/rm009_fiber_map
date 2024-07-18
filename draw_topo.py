# Draw topological map for LGN-V1 tracing

"""
# Usage:
# output figure
python draw_topy.py output_figure
"""

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
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return True  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter
    #return hasattr(__builtins__,'__IPYTHON__')
    #return True

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
    dupes = [x for x in swcs_ext.neu_id if x in seen or seen.add(x)]
    if len(dupes) > 0:
        logger.error('Duplication found!')

def LoadLGNMesh():
    """ load LGN mesh, 0:LGN, 1-6:LGN layers 1-6 """
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

    # simplified version
    mesh_fn = '3_f800.obj'
    mesh_fn_path = os.path.join(mesh_dir, mesh_fn)
    v1_s_mesh = trimesh.load_mesh(mesh_fn_path)
    v1_s_mesh.vertices *= 10.0

    return mesh, v1_s_mesh

def Normalize(vector):
    norm = np.linalg.norm(vector)
    if norm == 0: 
        return vector  # Return the original vector if its norm is 0
    return vector / norm

def ProjectionResidual(vec, normal):
    # normal: normal of the (hyper-)plane
    normal = Normalize(normal)
    return vec - np.dot(vec, normal) * normal

def GetLgnSoma2DMap(pos_soma, lgn_mesh_list):
    # define manifold that represent projected layer of a LGN layer
    # for simplicity, here use a planer mesh
    idx_layer = 3
    # define the origin of the manifold in the world coordinate
    origin_pos = np.mean(lgn_mesh_list[idx_layer].vertices, axis = 0)
    # define coordinate frame at the origin of the manifold (plane)
    origin_normal = Normalize(np.array([0.7, -1.0, -0.9]))
    # approximate Inclination
    origin_y_direction = np.array([1, 0, -0.3])
    origin_y_direction = Normalize(ProjectionResidual(origin_y_direction, origin_normal))
    # approximate Eccentricity
    origin_x_direction = np.cross(origin_normal, -origin_y_direction)
    # get projection of the soma position on the manifold
    pos_soma_2d = (pos_soma - origin_pos).dot(
        np.c_[origin_x_direction, origin_y_direction])

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

def GetV1Terminal2DMap(pos_terminal, v1_mesh):
    # define manifold that represent projected layer of a LGN layer
    # for simplicity, here use a planer mesh
    idx_layer = 3
    # note
    # blender	obj
    # x	  	x
    # y		-z
    # z		y
    ## define the origin of the manifold in the world coordinate
    #origin_pos = np.array([4871.33, -5835.53, 2509.96]) * 10.0
    ## define coordinate frame at the origin of the manifold (plane)
    #px = np.array([5254.99, -5188.76, 2095.69]) * 10.0
    #po = np.array([4296.63, -5901.01, 1875.48]) * 10.0
    #py = np.array([4487.67, -6482.3 , 2924.23]) * 10.0
    # define the origin of the manifold in the world coordinate
    origin_pos = np.array([4871.33, 2509.96, 5835.53]) * 10.0
    # define coordinate frame at the origin of the manifold (plane)
    px = np.array([5254.99, 2095.69, 5188.76]) * 10.0
    po = np.array([4296.63, 1875.48, 5901.01]) * 10.0
    py = np.array([4487.67, 2924.23, 6482.3 ]) * 10.0
    # define the origin of the manifold in the world coordinate
    #origin_pos = np.array([2509.96, -5835.53, 4871.33]) * 10.0
    # define coordinate frame at the origin of the manifold (plane)
    #px = np.array([2095.69, -5188.76, 5254.99]) * 10.0
    #po = np.array([1875.48, -5901.01, 4296.63]) * 10.0
    #py = np.array([2924.23, -6482.3 , 4487.67]) * 10.0
    origin_x_direction = Normalize(px - po)
    origin_y_direction = Normalize(py - po) 
    origin_normal = np.cross(origin_x_direction, origin_y_direction)
    # get projection of the soma position on the manifold
    pos_terminal_2d = (pos_terminal - origin_pos).dot(
        np.c_[origin_x_direction, origin_y_direction])

    frame_manifold = Struct(
        type = 'plane',
        origin = origin_pos,
        normal = origin_normal,
        x_axis = origin_x_direction,
        y_axis = origin_y_direction,
        x_range = np.array([-1, 1]) * np.linalg.norm(px-po),
        y_range = np.array([-1, 1]) * np.linalg.norm(py-po),
    )

    return frame_manifold, pos_terminal_2d

def GetTopoMapSiteWithColor(swcs_a, color_scalar):
    """ get terminal points """
    terminal_set = []
    terminal_segment_len = np.zeros(len(swcs_a.ntree_ext), dtype = np.int32)
    term_plan = 2
    for j, ntr in enumerate(swcs_a.ntree_ext):
        if term_plan == 1:
            tip_filter = '(path_length_to_root(leaves) > 30000) & (branch_depth(leaves) >= 5)'
            b_leaves = exec_filter_string(tip_filter, ntr)
            pos = ntr.position_of_node(ntr.leaves[b_leaves])
        elif term_plan == 2:
            proc_filter = '(path_length_to_root(end_point(processes)) > 30000) & (branch_depth(processes) == 7)'
            b_proc = exec_filter_string(proc_filter, ntr)
            pos = ntr.position_of_node(ntr.end_point(ntr.processes)[b_proc])
        terminal_set.append(pos)
        terminal_segment_len[j] = len(pos)
        #bidx = v1_mesh.contains(pos)   # very slow, >1 hours
        #terminal_set.append(pos[bidx])
    terminal_set = np.concatenate(terminal_set, axis = 0)
    return terminal_set, terminal_segment_len

def GetTerminalColorScalar(terminal_segment_len, color_scalar):
    """ Used with GetTopoMapSiteWithColor() """
    #terminal_color_scalar = np.concatenate(
    #    [np.full(l, v) for l, v in zip(terminal_segment_len, color_scalar)]
    #)
    terminal_color_scalar = np.repeat(color_scalar, terminal_segment_len)
    return terminal_color_scalar

def GetTerminalStat(pos_terminal_2d, terminal_segment_len):
    import warnings
    l = len(terminal_segment_len)
    terminal_centers = np.zeros((l, 2))
    rg = np.cumsum(np.hstack([0, terminal_segment_len]))
    with np.errstate(divide='ignore', invalid='ignore'):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            # if empty, output nan, without warning
            for k in range(l):
                pos = np.mean(pos_terminal_2d[rg[k]:rg[k+1], :], axis = 0)
                terminal_centers[k] = pos
    return terminal_centers

def LoadSWC(swc_dir):
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
    idx_valid[swcs_a.neu_id.index('505')] = False  # filter out specific
    swcs_a = swcs_a[idx_valid]
    pos_soma = pos_soma[idx_valid]
    # location of soma
    pos_soma2 = np.array([s.ntree[1][0, 0:3] for s in swcs_a])
    assert not np.any(pos_soma2 - pos_soma)

    return swcs_a, pos_soma

def GetSomaLayer(pos_soma, lgn_mesh_s):
    soma_layer_i = np.zeros(len(pos_soma), dtype=np.int32)
    for i_layer in range(len(lgn_mesh_s)):
        bidx = lgn_mesh_s[i_layer].contains(pos_soma)
        # note that we must put LGN to the first mesh,
        # to be overwritten by detailed layers
        soma_layer_i[bidx] = i_layer
    return soma_layer_i

def LoadAndAnalyze(swc_dir):
    """ Main function of 'analyzer' """
    swcs_a, pos_soma = LoadSWC(swc_dir)
    lgn_mesh_s = LoadLGNMesh()
    v1_mesh, v1_s_mesh = LoadV1Mesh()

    soma_layer_i = GetSomaLayer(pos_soma, lgn_mesh_s)

    ## Prepare LGN data
    # get 2D map coordinate of soma in LGN layer(s)
    lgn_frame_manifold, pos_soma_2d = GetLgnSoma2DMap(pos_soma, lgn_mesh_s)

    ## parpare V1 data
    terminal_set, terminal_segment_len = \
            GetTopoMapSiteWithColor(swcs_a, soma_layer_i)
    v1_frame_manifold, pos_terminal_2d = \
            GetV1Terminal2DMap(terminal_set, v1_mesh)
    
    lgn_v1_data = Struct(
        swcs_a     = swcs_a,
        lgn_mesh_s = lgn_mesh_s,
        v1_mesh    = v1_mesh,
        v1_s_mesh  = v1_s_mesh,
        # LGN
        pos_soma           = pos_soma,
        soma_layer_i       = soma_layer_i,
        lgn_frame_manifold = lgn_frame_manifold,
        pos_soma_2d        = pos_soma_2d,
        # V1
        terminal_set         = terminal_set,
        terminal_segment_len = terminal_segment_len,
        v1_frame_manifold    = v1_frame_manifold,
        pos_terminal_2d      = pos_terminal_2d,
    )
    return lgn_v1_data

def ShowLGNPlt(ax, show_layers = [0]):
    mesh_list = LoadLGNMesh()
    for idx_l in show_layers:
        mesh = mesh_list[idx_l]
        x, y, z = mesh.vertices.T
        ax.plot_trisurf(x, y, z, triangles=mesh.faces,
            color='grey', alpha=0.1)

def DrawMeshBatch(plotter, mesh_list):
    for mesh in mesh_list:
        pv_mesh = pv.PolyData(mesh.vertices,
                    np.c_[[[3]]*len(mesh.faces),mesh.faces])
        plotter.add_mesh(pv_mesh, smooth_shading=True,
                        opacity = 0.1)

def PlotInMatplotlib(lgn_v1_data):
    # swcs_a, pos_soma, lgn_mesh_s, v1_mesh
    pos_soma = lgn_v1_data.pos_soma
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

def PlotErwinData(erwin_data, ml_idx = 120, ap_idx = 120):
    plt.figure(30)
    plt.imshow(erwin_data.ecc[ml_idx, :, :])
    plt.colorbar()
    plt.title(f'eccentricity, medial-lateral idx={ml_idx}')
    plt.xlabel('anterior-posterior')
    plt.ylabel('dorsal-ventral')
    SaveCurrentFigure(f'erwin_ecc_map_ml{ml_idx}.png')

    plt.figure(31)
    plt.imshow(erwin_data.ecc[:, :, ap_idx])
    plt.colorbar()
    plt.title(f'inclination, anterior-posterior idx={ap_idx}')
    plt.xlabel('medial-lateral')
    plt.ylabel('dorsal-ventral')
    SaveCurrentFigure(f'erwin_ecc_map_ap{ap_idx}.png')

    plt.figure(32)
    plt.imshow(erwin_data.layers[:, :, ap_idx])
    plt.colorbar()
    plt.title(f'laminar type, anterior-posterior idx={ap_idx}')
    plt.xlabel('medial-lateral')
    plt.ylabel('dorsal-ventral')
    SaveCurrentFigure(f'erwin_layers_map_ap{ap_idx}.png')

def DrawErwin3Views():
    """ Show Erwin LGN map"""

    ## LGN reference map
    erwin_data = ReadErwinData()
    PlotErwinData(erwin_data)

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

def DrawDotsWithColor(plotter, pos, color_scalar, point_size):
    cloud = pv.PolyData(pos)
    cloud['color_scalar'] = color_scalar
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
    plotter.add_mesh(cloud, color="red", point_size = point_size,
                    render_points_as_spheres = True,
                    scalars = 'color_scalar', clim = [0, 6.1],
                    cmap = cmap, scalar_bar_args=sargs)

def DrawCoordinateFrame(plotter, frame_manifold, arrow_scaling):
    # plot reference manifold
    frame_geo = pv.Plane(
            center = frame_manifold.origin,
            direction = frame_manifold.normal,
            i_size = frame_manifold.x_range[1],
            j_size = frame_manifold.y_range[1])
    plotter.add_mesh(frame_geo, opacity=0.7)
    # plot coordinate frame
    arrow_scale = frame_manifold.x_range[1] * arrow_scaling
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

def Plot2DLabels(pos_2d, swcs_a, **kwargs):
    # if kwargs is empty, use defaults
    kwargs0 = {
        'horizontalalignment': 'left',
        'verticalalignment': 'top',
        'fontsize': 4,
        'color': 'black',
        'alpha': 0.4
    }
    kwargs0.update(kwargs)

    labels = [s.neu_id for s in swcs_a]
    for i, txt in enumerate(labels):
        plt.text(pos_2d[i, 0], pos_2d[i, 1], txt, **kwargs0)

def Plot2DMapWithColor(pos_2d, color_scalar, point_size,
                       title, labels = ['x','y'], cmap = None, **kwargs):
    X = pos_2d[:, 0]
    Y = pos_2d[:, 1]
    if len(color_scalar.shape) == 2:
        c = color_scalar[:, 0]
        d = color_scalar[:, 1]
        dx, dy = np.cos(d), np.sin(d)
        plt.quiver(X, Y, dx, dy, headlength=3.0, alpha=0.5)
    else:
        c = color_scalar
    if cmap is None:
        cmap = 'viridis'
    plt.scatter(X, Y, c=c, cmap=cmap, s = point_size, **kwargs)
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    plt.title(title)
    SaveCurrentFigure(f'{title}.png')

def Plot2DMapWithColorDirection(pos_2d, color_scalar, point_size, title, labels = ['x','y']):
    plt.scatter(pos_2d[:, 0], pos_2d[:, 1],
                c=color_scalar, cmap='viridis', s = point_size)
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    plt.title(title)
    SaveCurrentFigure(f'{title}.png')

def GetColorScalarArray(lgn_v1_data, color_mode):
    """ return color scalar array and color map. """
    soma_layer_i = lgn_v1_data.soma_layer_i
    pos_soma_2d = lgn_v1_data.pos_soma_2d

    if color_mode == 'layer':
        color_scalar = soma_layer_i
    elif color_mode == 'lgn_x':
        color_scalar = (pos_soma_2d[:, 0] - 0) * 1
    elif color_mode == 'lgn_y':
        color_scalar = (pos_soma_2d[:, 1] - 0) * 1
    elif color_mode == 'lgn_xy':
        c1 = (pos_soma_2d[:, 0] - 0) * 1
        c2 = (pos_soma_2d[:, 1] - 0) * 1
        c2 = ((c2 - c2.min()) / (c2.max() - c2.min()) - 0.5) * np.pi
        color_scalar = np.vstack([c1, c2]).T
    elif color_mode == 'lgn_yx':
        c1 = (pos_soma_2d[:, 0] - 0) * 1
        c2 = (pos_soma_2d[:, 1] - 0) * 1
        c1 = ((c1 - c1.min()) / (c1.max() - c1.min())-0.5) * np.pi
        color_scalar = np.vstack([c2, c1]).T
    elif color_mode == 'left_right_eye':
        ll = soma_layer_i
        bias = np.random.rand(len(ll)) * 0.01
        color_scalar = 1 * ((ll == 6) | (ll == 4) | (ll == 1)) + bias
    elif color_mode == 'fiber_len':
        color_scalar = [] # tree or processes length in total
    
    return color_scalar

def Draw3DLgnV1(lgn_v1_data):
    """Main ploting function"""
    swcs_a       = lgn_v1_data.swcs_a
    lgn_mesh_s   = lgn_v1_data.lgn_mesh_s
    v1_mesh      = lgn_v1_data.v1_mesh
    v1_s_mesh    = lgn_v1_data.v1_s_mesh
    # LGN
    pos_soma     = lgn_v1_data.pos_soma
    soma_layer_i = lgn_v1_data.soma_layer_i
    lgn_frame_manifold = lgn_v1_data.lgn_frame_manifold
    pos_soma_2d        = lgn_v1_data.pos_soma_2d
    # V1
    terminal_set          = lgn_v1_data.terminal_set
    terminal_segment_len  = lgn_v1_data.terminal_segment_len
    v1_frame_manifold = lgn_v1_data.v1_frame_manifold
    pos_terminal_2d   = lgn_v1_data.pos_terminal_2d

    ## Draw LGN and soma
    plotter = pv.Plotter()
    plotter.show_axes()
    # draw LGN layer mesh
    DrawMeshBatch(plotter, lgn_mesh_s[1:2])
    # draw soma
    DrawDotsWithColor(plotter, pos_soma, soma_layer_i, 20.0)
    # draw projecting manifold
    DrawCoordinateFrame(plotter, lgn_frame_manifold, 0.35)
    plotter.show()       # Press 'q' for quit

    ## Draw V1 and terminal
    plotter = pv.Plotter()
    plotter.show_axes()
    # draw V1
    DrawMeshBatch(plotter, [v1_mesh])
    #DrawMeshBatch(plotter, [v1_s_mesh])  # use simplified mesh for now
    # plot terminal points
    cloud = pv.PolyData(terminal_set)
    plotter.add_mesh(cloud, color="red", point_size = 5.0,
                     render_points_as_spheres = True)
    # plot reference manifold
    DrawCoordinateFrame(plotter, v1_frame_manifold, 0.35)
    # plot also LGN
    DrawMeshBatch(plotter, lgn_mesh_s[1:2])
    plotter.show()

SaveCurrentFigure = lambda fn: plt.savefig(os.path.join('./pic', fn), dpi=300)

def GenerateFigures(lgn_v1_data):
    swcs_a = lgn_v1_data.swcs_a
    # LGN
    pos_soma_2d  = lgn_v1_data.pos_soma_2d
    # V1
    terminal_segment_len  = lgn_v1_data.terminal_segment_len
    pos_terminal_2d       = lgn_v1_data.pos_terminal_2d
    
    ## output the LGN soma 2D map
    # x
    plt.figure(10)
    c = GetColorScalarArray(lgn_v1_data, 'lgn_x')
    Plot2DMapWithColor(pos_soma_2d, c, 50.0, 'LGN_soma_2d_map_x',
                       labels=['Eccentricity', 'Inclination'])
    # y
    plt.figure(11)
    c = GetColorScalarArray(lgn_v1_data, 'lgn_y')
    Plot2DMapWithColor(pos_soma_2d, c, 50.0, 'LGN_soma_2d_map_y',
                       labels=['Eccentricity', 'Inclination'])
    # xy
    plt.figure(19); plt.cla()
    cc = GetColorScalarArray(lgn_v1_data, 'lgn_xy')
    Plot2DMapWithColor(pos_soma_2d, cc, 35.0, 'LGN_soma_2d_map_xy',
                       labels=['Eccentricity', 'Inclination'])


    ## draw the 2D map
    # V1 by lgn x
    plt.figure(20)
    c = GetColorScalarArray(lgn_v1_data, 'lgn_x')
    terminal_segment_color = GetTerminalColorScalar(terminal_segment_len, c)
    Plot2DMapWithColor(pos_terminal_2d, terminal_segment_color, 0.2,
                       'V1_term_2d_map_lgn_x',
                       labels=['x (um)', 'y (um)'])
    plt.figure(20); plt.cla()
    terminal_centers = GetTerminalStat(pos_terminal_2d, terminal_segment_len)
    Plot2DMapWithColor(terminal_centers, cc[:,0], 20,
                       'V1_termc_2d_map_lgn_x',
                       labels=['x (um)', 'y (um)'])
    # V1 by lgn y
    plt.figure(21)
    c = GetColorScalarArray(lgn_v1_data, 'lgn_y')
    terminal_segment_color = GetTerminalColorScalar(terminal_segment_len, c)
    Plot2DMapWithColor(pos_terminal_2d, terminal_segment_color, 0.2,
                       'V1_term_2d_map_lgn_y',
                       labels=['x (um)', 'y (um)'])
    plt.figure(21); plt.cla()
    Plot2DMapWithColor(terminal_centers, cc[:,1], 20,
                       'V1_termc_2d_map_lgn_y',
                       labels=['x (um)', 'y (um)'])

    # V1 by lgn x and y
    plt.figure(29); plt.cla()
    terminal_centers = GetTerminalStat(pos_terminal_2d, terminal_segment_len)
    Plot2DMapWithColor(terminal_centers, cc, 30.0,
                       'V1_termc_2d_map_lgn_xy',
                       labels=['x (um)', 'y (um)'])

    # V1 by lgn x and y, with neuron id
    plt.figure(28); plt.cla()
    # labels as the swc name
    terminal_centers = GetTerminalStat(pos_terminal_2d, terminal_segment_len)
    Plot2DLabels(terminal_centers, swcs_a)
    Plot2DMapWithColor(terminal_centers, cc, 30.0,
                       'V1_term_2d_map_lgn_xy_dbg',
                       labels=['x (um)', 'y (um)'])

    # V1 by lgn left-right eye
    plt.figure(25); plt.cla()
    c = GetColorScalarArray(lgn_v1_data, 'left_right_eye')
    cm = {'cmap': 'hsv', 'vmin': -0.1, 'vmax': 3.0, 'alpha': 0.3}
    #cm = {'cmap': 'RdYlGn', 'vmin': -0.2, 'vmax': 1.2, 'alpha': 0.1}
    terminal_segment_color = GetTerminalColorScalar(terminal_segment_len, c)
    ipermute = np.random.permutation(len(terminal_segment_color))
    Plot2DMapWithColor(pos_terminal_2d[ipermute],
                       terminal_segment_color[ipermute],
                       0.2, 'V1_term_2d_map_left_right_eye',
                       labels=['x (um)', 'y (um)'], **cm)

if __name__ == '__main__':
    if len(sys.argv) > 1:
        plot_mode = sys.argv[1]
    else:
        plot_mode = 'output_figure'

    if plot_mode != 'erwin':
        #old
        #swc_dir = '/mnt/data_ext/swc_collect/RM009_v1.6.6/1.6.6_complete/'
        # new
        swc_dir = './rm009_swcs'
        lgn_v1_data = LoadAndAnalyze(swc_dir)

    if plot_mode == 'plt':
        PlotInMatplotlib(lgn_v1_data)
    elif plot_mode == 'pyvista':
        Draw3DLgnV1(lgn_v1_data)
    elif plot_mode == 'output_figure':
        GenerateFigures(lgn_v1_data)
    elif plot_mode == 'erwin':
        DrawErwin3Views()
    else:
        raise ValueError(f'Unknown plot mode: {plot_mode}')
