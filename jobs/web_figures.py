# Copyright (c) 2026 University of Connecticut
# Created by B.L. DuBois
# SPDX-License-Identifier: MIT
# See the LICENSE file for details
"""
Make web figures.
"""

import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import colors, patches
import numpy as np
import torch

from iris import arepo_processing_write as ap
from iris import hyper as hp
from iris import observation as ob
from iris import reversion as rv


FULL_CONE_DATASET_DIR = '/path/to/full_cone_data'
AREPO_SNAPSHOT_PATH = '/path/to/snapshot.hdf5'
WEB_OBSERVATION_DATASET_DIR = '~/IRIS/data/web_figure_data'
REVERTER_CHECKPOINT_PATH = '~/IRIS/models/reverter_1/chp_32.pt'

COLOR_SCHEME = 'black-bright'  # Options: 'white-bright', 'black-bright', 'glow-mode'.
DRAW_LABELS = False
DPI = 600

OUTPUT_DIR = '~/IRIS/output'
VERTICAL_REVERSION_PATH = 'web_reversions_vertical.png'
HORIZONTAL_REVERSION_PATH = 'web_reversions_horizontal.png'
SOLO_REVERSION_PATH = 'web_reversion_solo.png'
OBSERVATION_LB_PATH = 'web_observation_lb.png'

REVERSION_PANEL_WIDTH_IN = 8.0
REVERSION_PANEL_LV_MARGIN_X_IN = 0.40
REVERSION_PANEL_MARGIN_Y_IN = 0.18
REVERSION_PANEL_LV_WIDTH_IN = REVERSION_PANEL_WIDTH_IN - 2.0 * REVERSION_PANEL_LV_MARGIN_X_IN
REVERSION_PANEL_LV_HEIGHT_IN = REVERSION_PANEL_LV_WIDTH_IN / 3.0
REVERSION_PANEL_ARROW_HEIGHT_IN = 0.70
REVERSION_PANEL_TOP_ARROW_GAP_IN = 0.10
REVERSION_PANEL_ARROW_LV_GAP_IN = 0.08
REVERSION_PANEL_TOP_DOWN_GAP_IN = 0.22
REVERSION_STACK_GAP_IN = 0.20
REVERSION_FIGURE_MARGIN_IN = 0.12


def figure_style() -> tuple[object, str, str]:
    if COLOR_SCHEME == 'white-bright':
        return 'gray', 'black', 'white'
    if COLOR_SCHEME == 'black-bright':
        return 'gray_r', 'white', 'black'
    if COLOR_SCHEME == 'glow-mode':
        cmap = colors.LinearSegmentedColormap.from_list(
            'glow_mode',
            ['black', '#fffade'])
        return cmap, 'black', '#fffade'
    raise ValueError("COLOR_SCHEME must be 'white-bright', 'black-bright', or 'glow-mode'.")


def border_color() -> str:
    if COLOR_SCHEME == 'glow-mode':
        return 'white'
    _cmap, _dark, bright = figure_style()
    return bright


def arrow_color_map() -> object:
    if COLOR_SCHEME == 'glow-mode':
        return colors.LinearSegmentedColormap.from_list(
            'glow_mode_arrows',
            ['black', 'white'])
    cmap, _dark, _bright = figure_style()
    return cmap


def label_color() -> str:
    if COLOR_SCHEME == 'glow-mode':
        return 'white'
    _cmap, _dark, bright = figure_style()
    return bright


def density_unit(hyper: hp.Hyper) -> float:
    mass = hyper.dataset_hyper._mass_iris_per_SI
    length = hyper.dataset_hyper._length_iris_per_SI
    volume = length * length * length
    density = mass / volume
    solar_mass = 1.988e30
    parsec = hyper.dataset_hyper.meters_per_parsec
    return density * solar_mass / parsec / parsec / parsec


def asinh_norm(data: np.ndarray,
               color_scale: float | None = None,
               color_scale_center: float = 0.0,
               color_norm_min: float | None = None,
               color_norm_max: float | None = None) -> colors.FuncNorm:
    finite = np.asarray(data)[np.isfinite(data)]
    if color_norm_min is None:
        color_norm_min = float(finite.min())
    if color_norm_max is None:
        color_norm_max = float(finite.max())
    if color_scale is None:
        mean = float(np.mean(np.abs(finite)))
        color_scale = 1.0 / mean if mean > 0 else 1.0

    forward = lambda x: np.asinh(color_scale * (x - color_scale_center))
    inverse = lambda y: np.sinh(y) / color_scale + color_scale_center
    return colors.FuncNorm((forward, inverse), vmin=color_norm_min, vmax=color_norm_max)


def decorate_axis(ax: plt.Axes,
                  label: str | None = None,
                  rectangular_border: bool = True) -> None:
    _cmap, panel_face, _label_color = figure_style()
    ax.set_facecolor(panel_face)
    ax.grid(False)
    ax.set_axis_off()
    if rectangular_border:
        ax.add_patch(patches.Rectangle((0, 0), 1, 1,
                                       transform=ax.transAxes,
                                       fill=False,
                                       edgecolor=border_color(),
                                       linewidth=2.0,
                                       zorder=20,
                                       clip_on=False))
    if DRAW_LABELS and label is not None:
        ax.text(0.04, 0.94, label,
                transform=ax.transAxes,
                color=label_color(),
                fontsize='large',
                fontweight='bold',
                ha='left',
                va='top')


def top_down_edges(hyper: hp.Hyper,
                   shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    lon_min = np.deg2rad(hyper.coordinate_hyper.lon_min)
    lon_max = np.deg2rad(hyper.coordinate_hyper.lon_max)
    r_steps, lon_steps = shape
    theta_edges = np.linspace(lon_min, lon_max, lon_steps + 1)

    r_min = hyper.coordinate_hyper.r_min / 1000
    r_max = hyper.coordinate_hyper.r_max / 1000
    r_crop_min_index = hyper.coordinate_hyper.r_crop_min_index
    r_crop_max_index = hyper.coordinate_hyper.r_crop_max_index
    r_steps = hyper.coordinate_hyper.r_steps
    r_crop_min = r_min + (r_max - r_min) * r_crop_min_index / r_steps
    r_crop_max = r_min + (r_max - r_min) * r_crop_max_index / r_steps
    r_edges = np.linspace(r_crop_min, r_crop_max, shape[0] + 1)

    theta_grid, r_grid = np.meshgrid(theta_edges, r_edges)
    x_edges = r_grid * np.sin(theta_grid)
    y_edges = r_grid * np.cos(theta_grid)
    return x_edges, y_edges, theta_edges, r_edges


def top_down_wedge_extent(hyper: hp.Hyper,
                          shape: tuple[int, int]) -> tuple[float, float, float, float]:
    x_edges, y_edges, _theta_edges, _r_edges = top_down_edges(hyper, shape)
    return (
        float(np.min(x_edges)),
        float(np.min(y_edges)),
        float(np.max(x_edges)),
        float(np.max(y_edges)),
    )


def top_down_height_in(hyper: hp.Hyper,
                       shape: tuple[int, int],
                       width_in: float) -> float:
    x_min, y_min, x_max, y_max = top_down_wedge_extent(hyper, shape)
    data_width = x_max - x_min
    data_height = y_max - y_min
    if data_width <= 0 or data_height <= 0:
        raise RuntimeError('Could not measure positive top-down wedge dimensions.')
    return width_in * data_height / data_width


def reversion_panel_height_in(hyper: hp.Hyper,
                              top_down_shape: tuple[int, int]) -> float:
    top_down_width_in = 0.5 * (REVERSION_PANEL_LV_WIDTH_IN - REVERSION_PANEL_TOP_DOWN_GAP_IN)
    return (
        2.0 * REVERSION_PANEL_MARGIN_Y_IN
        + REVERSION_PANEL_LV_HEIGHT_IN
        + REVERSION_PANEL_ARROW_LV_GAP_IN
        + REVERSION_PANEL_ARROW_HEIGHT_IN
        + REVERSION_PANEL_TOP_ARROW_GAP_IN
        + top_down_height_in(hyper, top_down_shape, top_down_width_in))


def inches_to_figure_rect(fig: plt.Figure,
                          left: float,
                          bottom: float,
                          width: float,
                          height: float) -> list[float]:
    fig_width, fig_height = fig.get_size_inches()
    return [
        left / fig_width,
        bottom / fig_height,
        width / fig_width,
        height / fig_height,
    ]


def plot_top_down(ax: plt.Axes,
                  density: np.ndarray,
                  hyper: hp.Hyper,
                  norm: colors.FuncNorm,
                  label: str | None = None) -> None:
    cmap, _panel_face, _label_color = figure_style()
    x_edges, y_edges, _theta_edges, _r_edges = top_down_edges(hyper, density.shape)
    ax.pcolormesh(x_edges, y_edges, density, norm=norm, cmap=cmap, shading='auto')
    ax.plot(x_edges[0, :], y_edges[0, :],
            color=border_color(), linewidth=2.0, zorder=20, clip_on=False)
    ax.plot(x_edges[-1, :], y_edges[-1, :],
            color=border_color(), linewidth=2.0, zorder=20, clip_on=False)
    ax.plot(x_edges[:, 0], y_edges[:, 0],
            color=border_color(), linewidth=2.0, zorder=20, clip_on=False)
    ax.plot(x_edges[:, -1], y_edges[:, -1],
            color=border_color(), linewidth=2.0, zorder=20, clip_on=False)
    ax.set_xlim(float(np.min(x_edges)), float(np.max(x_edges)))
    ax.set_ylim(float(np.min(y_edges)), float(np.max(y_edges)))
    ax.set_aspect('equal')
    decorate_axis(ax, label, rectangular_border=False)


def plot_lv(ax: plt.Axes,
            observed_lv: np.ndarray,
            hyper: hp.Hyper,
            norm: colors.FuncNorm,
            label: str | None = None) -> None:
    cmap, _panel_face, _label_color = figure_style()
    lon_min = hyper.coordinate_hyper.lon_min
    lon_max = hyper.coordinate_hyper.lon_max
    v_min = hyper.cube_hyper.v_min
    v_max = hyper.cube_hyper.v_max
    lon_edges = np.linspace(lon_min, lon_max, observed_lv.shape[0] + 1)
    v_edges = np.linspace(v_min, v_max, observed_lv.shape[1] + 1)

    ax.pcolormesh(lon_edges, v_edges, observed_lv.transpose(), cmap=cmap, norm=norm, shading='auto')
    ax.invert_xaxis()
    ax.set_aspect('auto')
    decorate_axis(ax, label)


def plot_lb(ax: plt.Axes,
            observed_lb: np.ndarray,
            hyper: hp.Hyper,
            norm: colors.FuncNorm,
            label: str | None = None,
            border: bool = True) -> None:
    cmap, _panel_face, _label_color = figure_style()
    lon_min = hyper.coordinate_hyper.lon_min
    lon_max = hyper.coordinate_hyper.lon_max
    lat_min = hyper.coordinate_hyper.lat_min
    lat_max = hyper.coordinate_hyper.lat_max
    lon_edges = np.linspace(lon_min, lon_max, observed_lb.shape[0] + 1)
    lat_edges = np.linspace(lat_min, lat_max, observed_lb.shape[1] + 1)

    ax.pcolormesh(lon_edges, lat_edges, observed_lb.transpose(), cmap=cmap, norm=norm, shading='auto')
    ax.invert_xaxis()
    ax.set_aspect('equal')
    decorate_axis(ax, label, rectangular_border=border)


def compute_reversions(reverter: rv.Reverter,
                       dataset: ap.CPUBatchObservedDataset) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    hyper = dataset.hyper
    density = density_unit(hyper)
    temperature = hyper.dataset_hyper._temperature_iris_per_SI

    top_down, observed_lv = dataset.sample(n=2, numpy=False, validation=False)
    reverter.cuda()
    reverter.eval()

    with torch.no_grad():
        observed_lv = observed_lv.cuda()
        reverted = reverter.multi_unit_call(observed_lv,
                                            in_units=hyper,
                                            out_units=hyper,
                                            reduce=False)

    top_down = top_down.detach().cpu().numpy()[:, 0] / density
    observed_lv = observed_lv.detach().cpu().numpy()[:, 0] / temperature
    reverted = reverted.detach().cpu().numpy()[:, 0] / density
    reverter.cpu()
    return [(top_down[i], reverted[i], observed_lv[i]) for i in range(2)]


def draw_reversion_panel(fig: plt.Figure,
                         sample: tuple[np.ndarray, np.ndarray, np.ndarray],
                         hyper: hp.Hyper,
                         left_in: float = 0.0,
                         bottom_in: float = 0.0) -> None:
    _cmap, figure_face, _label_color = figure_style()
    fig.set_facecolor(figure_face)
    arepo_top_down, reverted_top_down, observed_lv = sample
    top_down_shape = arepo_top_down.shape

    lv_left_in = left_in + REVERSION_PANEL_LV_MARGIN_X_IN
    lv_bottom_in = bottom_in + REVERSION_PANEL_MARGIN_Y_IN
    lv_rect = inches_to_figure_rect(fig,
                                    lv_left_in,
                                    lv_bottom_in,
                                    REVERSION_PANEL_LV_WIDTH_IN,
                                    REVERSION_PANEL_LV_HEIGHT_IN)

    arrow_bottom_in = lv_bottom_in + REVERSION_PANEL_LV_HEIGHT_IN + REVERSION_PANEL_ARROW_LV_GAP_IN
    arrow_rect = inches_to_figure_rect(fig,
                                       lv_left_in,
                                       arrow_bottom_in,
                                       REVERSION_PANEL_LV_WIDTH_IN,
                                       REVERSION_PANEL_ARROW_HEIGHT_IN)

    top_bottom_in = arrow_bottom_in + REVERSION_PANEL_ARROW_HEIGHT_IN + REVERSION_PANEL_TOP_ARROW_GAP_IN
    top_down_wedge_width_in = 0.5 * (REVERSION_PANEL_LV_WIDTH_IN - REVERSION_PANEL_TOP_DOWN_GAP_IN)
    top_down_h_in = top_down_height_in(hyper, top_down_shape, top_down_wedge_width_in)

    arepo_initial_left_in = lv_left_in
    reverted_initial_left_in = lv_left_in + top_down_wedge_width_in + REVERSION_PANEL_TOP_DOWN_GAP_IN
    arepo_ax = fig.add_axes(inches_to_figure_rect(fig,
                                                  arepo_initial_left_in,
                                                  top_bottom_in,
                                                  top_down_wedge_width_in,
                                                  top_down_h_in))
    reverted_ax = fig.add_axes(inches_to_figure_rect(fig,
                                                     reverted_initial_left_in,
                                                     top_bottom_in,
                                                     top_down_wedge_width_in,
                                                     top_down_h_in))
    arrow_ax = fig.add_axes(arrow_rect)
    observed_ax = fig.add_axes(lv_rect)
    arrow_ax.set_facecolor(figure_face)
    arrow_ax.set_axis_off()

    top_down_norm = asinh_norm(np.stack((arepo_top_down, reverted_top_down)),
                               color_scale=10.0,
                               color_norm_min=0,
                               color_norm_max=100.0)
    lv_norm = asinh_norm(observed_lv,
                         color_scale=1e2,
                         color_scale_center=0.035,
                         color_norm_min=0,
                         color_norm_max=2.0)

    plot_top_down(arepo_ax, arepo_top_down, hyper, top_down_norm, r'AREPO H$_2$')
    plot_top_down(reverted_ax, reverted_top_down, hyper, top_down_norm, 'IRIS Reversion')
    plot_lv(observed_ax, observed_lv, hyper, lv_norm, r'Synthetic $^{13}$CO')
    add_reversion_arrows(arrow_ax)


def add_reversion_arrows(ax: plt.Axes) -> None:
    cmap = arrow_color_map()
    arrow_cmap = plt.get_cmap(cmap) if isinstance(cmap, str) else cmap
    dark_arrow_color = arrow_cmap(0.2)
    bright_arrow_color = arrow_cmap(0.8)
    arrow_style = 'simple,head_length=0.4,head_width=1.2,tail_width=0.6'
    arrow_kwargs = dict(transform=ax.transAxes,
                        arrowstyle=arrow_style,
                        mutation_scale=39,
                        zorder=30)
    arrow_down_outer = patches.FancyArrowPatch((0.25, 0.90),
                                               (0.25, 0.10),
                                               facecolor=dark_arrow_color,
                                               edgecolor=bright_arrow_color,
                                               linewidth=4.0,
                                               **arrow_kwargs)
    arrow_down = patches.FancyArrowPatch((0.25, 0.90),
                                         (0.25, 0.10),
                                         facecolor=dark_arrow_color,
                                         edgecolor=bright_arrow_color,
                                         linewidth=2.0,
                                         zorder=31,
                                         transform=ax.transAxes,
                                         arrowstyle=arrow_style,
                                         mutation_scale=39)
    arrow_up_outer = patches.FancyArrowPatch((0.75, 0.10),
                                             (0.75, 0.90),
                                             facecolor=bright_arrow_color,
                                             edgecolor=bright_arrow_color,
                                             linewidth=4.0,
                                             **arrow_kwargs)
    arrow_up = patches.FancyArrowPatch((0.75, 0.10),
                                       (0.75, 0.90),
                                       facecolor=bright_arrow_color,
                                       edgecolor=dark_arrow_color,
                                       linewidth=2.0,
                                       zorder=31,
                                       transform=ax.transAxes,
                                       arrowstyle=arrow_style,
                                       mutation_scale=39)
    ax.add_artist(arrow_down_outer)
    ax.add_artist(arrow_down)
    ax.add_artist(arrow_up_outer)
    ax.add_artist(arrow_up)


def save_reversion_stack(samples: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
                         hyper: hp.Hyper,
                         orientation: str,
                         path: Path) -> None:
    _cmap, figure_face, _label_color = figure_style()
    panel_height = reversion_panel_height_in(hyper, samples[0][0].shape)
    if orientation == 'vertical':
        fig_width = REVERSION_PANEL_WIDTH_IN + 2.0 * REVERSION_FIGURE_MARGIN_IN
        fig_height = (
            2.0 * panel_height
            + REVERSION_STACK_GAP_IN
            + 2.0 * REVERSION_FIGURE_MARGIN_IN)
        fig = plt.figure(figsize=(fig_width, fig_height), facecolor=figure_face)
        panel_origins = [
            (REVERSION_FIGURE_MARGIN_IN,
             REVERSION_FIGURE_MARGIN_IN + panel_height + REVERSION_STACK_GAP_IN),
            (REVERSION_FIGURE_MARGIN_IN, REVERSION_FIGURE_MARGIN_IN),
        ]
    elif orientation == 'horizontal':
        fig_width = (
            2.0 * REVERSION_PANEL_WIDTH_IN
            + REVERSION_STACK_GAP_IN
            + 2.0 * REVERSION_FIGURE_MARGIN_IN)
        fig_height = panel_height + 2.0 * REVERSION_FIGURE_MARGIN_IN
        fig = plt.figure(figsize=(fig_width, fig_height), facecolor=figure_face)
        panel_origins = [
            (REVERSION_FIGURE_MARGIN_IN, REVERSION_FIGURE_MARGIN_IN),
            (REVERSION_FIGURE_MARGIN_IN + REVERSION_PANEL_WIDTH_IN + REVERSION_STACK_GAP_IN,
             REVERSION_FIGURE_MARGIN_IN),
        ]
    else:
        raise ValueError("orientation must be 'vertical' or 'horizontal'.")

    for (left_in, bottom_in), sample in zip(panel_origins, samples):
        draw_reversion_panel(fig, sample, hyper, left_in=left_in, bottom_in=bottom_in)

    fig.savefig(path, dpi=DPI, facecolor=figure_face)
    plt.close(fig)


def save_solo_reversion(sample: tuple[np.ndarray, np.ndarray, np.ndarray],
                        hyper: hp.Hyper,
                        path: Path) -> None:
    _cmap, figure_face, _label_color = figure_style()
    panel_height = reversion_panel_height_in(hyper, sample[0].shape)
    fig_width = REVERSION_PANEL_WIDTH_IN + 2.0 * REVERSION_FIGURE_MARGIN_IN
    fig_height = panel_height + 2.0 * REVERSION_FIGURE_MARGIN_IN
    fig = plt.figure(figsize=(fig_width, fig_height), facecolor=figure_face)
    draw_reversion_panel(fig,
                         sample,
                         hyper,
                         left_in=REVERSION_FIGURE_MARGIN_IN,
                         bottom_in=REVERSION_FIGURE_MARGIN_IN)
    fig.savefig(path, dpi=DPI, facecolor=figure_face)
    plt.close(fig)


def make_observation_hyper() -> hp.Hyper:
    hyper = hp.SEDIGISM_13C16O()
    hyper.writer_hyper.points_per_snapshot = 1
    hyper.dataset_hyper.CMZ_scale_factor = None
    hyper.dataset_hyper.CMZ_scale_range = None
    hyper.coordinate_hyper.theta_zero = 270.
    hyper.coordinate_hyper.r_steps = 512
    hyper.coordinate_hyper.r_crop_max_index = 512
    hyper.coordinate_hyper.lon_steps = 8192
    hyper.coordinate_hyper.lat_steps = 2048
    hyper.coordinate_hyper.r_pieces = 64
    hyper.observer_hyper.lon_pieces = 64
    hyper.observer_hyper.lat_pieces = 256
    hyper.observer_hyper.blur_inputs = True
    hyper.observer_hyper.blur_kernel_r = 3
    hyper.observer_hyper.blur_kernel_lon = 81
    hyper.observer_hyper.blur_kernel_lat = 21
    hyper.observer_hyper.out_blur_fwhm = None
    hyper.validate()
    return hyper


def observe_lb(dataset: ap.StandardDataset, hyper: hp.Hyper) -> np.ndarray:
    temperature = hyper.dataset_hyper._temperature_iris_per_SI
    arepo = dataset.sample(1, validation=False)
    observer = ob.IteratedSyntheticObserver(hyper=hyper, cpu_batch=True)
    observer.eval()
    observer.cuda()
    observed = observer(arepo)

    observed = observed.detach().cpu().numpy()[0, 0] / temperature
    if hyper.cube_hyper.reduction == 'mean':
        return np.mean(observed, axis=2)
    if hyper.cube_hyper.reduction == 'max':
        return np.max(observed, axis=2)
    raise ValueError("Cube reduction must be one of: 'mean', 'max'.")


def save_lb(observed_lb: np.ndarray, hyper: hp.Hyper, path: Path) -> None:
    _cmap, figure_face, _label_color = figure_style()
    fig = plt.figure(figsize=(16, 4), facecolor=figure_face)
    ax = fig.add_subplot(111)
    norm = asinh_norm(observed_lb,
                      color_scale=1e2,
                      color_scale_center=0.035,
                      color_norm_min=0,
                      color_norm_max=2.0)
    plot_lb(ax, observed_lb, hyper, norm, r'Synthetic $^{13}$CO Observation', border=False)
    fig.savefig(path, dpi=DPI, bbox_inches='tight', pad_inches=0.02, facecolor=figure_face)
    plt.close(fig)

units_reader = ap.Reader(path=FULL_CONE_DATASET_DIR,
                         dataset_type=ap.CPUBatchObservedDataset)
observation_hyper = make_observation_hyper()
try:
    reader = ap.Reader(path=WEB_OBSERVATION_DATASET_DIR,
                       dataset_type=ap.StandardDataset,
                       hyper=observation_hyper)
    dataset = reader.dataset
except Exception:
    writer = ap.Writer(path=WEB_OBSERVATION_DATASET_DIR,
                       snapshot_paths=[AREPO_SNAPSHOT_PATH],
                       hyper=observation_hyper,
                       dataset_type=ap.StandardDataset,
                       units_from=units_reader.dataset,
                       gpu_interpolate=True,
                       gpu_normalize=False,
                       verbose=True)
    dataset = writer.dataset

if ap.MPI.COMM_WORLD.Get_rank() == 0:
    output_dir = Path(os.path.expanduser(OUTPUT_DIR))
    output_dir.mkdir(parents=True, exist_ok=True)

    full_cone_reader = ap.Reader(path=FULL_CONE_DATASET_DIR,
                                 dataset_type=ap.CPUBatchObservedDataset)
    full_cone_dataset = full_cone_reader.dataset
    reverter = rv.Reverter(hyper=full_cone_dataset.hyper)
    reverter.load_state_dict(torch.load(os.path.expanduser(REVERTER_CHECKPOINT_PATH),
                                        weights_only=True,
                                        map_location='cpu'))

    observed_lb = observe_lb(dataset, observation_hyper)
    save_lb(observed_lb, observation_hyper, output_dir / OBSERVATION_LB_PATH)

    samples = compute_reversions(reverter, full_cone_dataset)
    save_reversion_stack(samples,
                         full_cone_dataset.hyper,
                         'vertical',
                         output_dir / VERTICAL_REVERSION_PATH)
    save_reversion_stack(samples,
                         full_cone_dataset.hyper,
                         'horizontal',
                         output_dir / HORIZONTAL_REVERSION_PATH)
    save_solo_reversion(samples[0],
                        full_cone_dataset.hyper,
                        output_dir / SOLO_REVERSION_PATH)
