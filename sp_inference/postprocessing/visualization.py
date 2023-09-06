#====================================== Preliminary Commands =======================================
import os
from dataclasses import dataclass
from typing import final, Iterable, Optional

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from multimethod import multimethod

from . import data_types as dt

sns.set_theme(style="ticks")


#========================================== Color Control ==========================================
class ColorControl:

    _color_palette = ('royalblue', 'darkorange', 'forestgreen', 'firebrick')

    def __init__(self) -> None:
        self._num_colors = len(self._color_palette)
        self._counter = 0

    def get_color(self, next: Optional[bool]=True) -> str:
        current_index = self._counter % self._num_colors
        if next:
            color = self._color_palette[current_index]
            self._counter += 1
        else:
            color = self._color_palette[current_index-1]
        return color


#======================================== Custom MPL Figure ========================================
class MatplotlibFigure:
    def __init__(self, file_type: Optional[str] = "pdf") -> None:
        self.fig, self.axis = plt.subplots(figsize=(5, 5))
        self.title = {"prefix": None, "component": None, "timestamp": None, "suffix": None}
        self.file_name = {"prefix": None, "component": None, "timestamp": None, "suffix": None}
        self.title_is_set = False
        self.file_type = file_type

    def make_axis_title(self) -> None:
        if not self.title_is_set:
            title = ""
            if prefix := self.title['prefix']: title += f"{prefix}, "
            if component := self.title['component']: title += f"component {component}, "
            if timestamp := self.title['timestamp']: title += f"time = {timestamp}, "
            if suffix := self.title['suffix']: title += f"{suffix}"
            self.axis.set_title(title, wrap=True)
            self.title_is_set = True

    def prune_and_save(self) -> None:
        if not self.axis.has_data():
            self.fig.delaxes(self.axis)
        else:
            file_name = ""
            if prefix := self.file_name['prefix']: file_name += f"{prefix}_"
            if component := self.file_name['component']: file_name += f"c{component}_"
            if timestamp := self.file_name['timestamp']: file_name += f"t{timestamp}_"
            if suffix := self.file_name['suffix']: file_name += f"{suffix}"
            self.fig.savefig(f"{file_name}.{self.file_type}")
    
#---------------------------------------------------------------------------------------------------
@final
class Pyplot1DFigure(MatplotlibFigure):
    def __init__(self):
        super().__init__()
        self.color_control = ColorControl()

#---------------------------------------------------------------------------------------------------
@final
class Pyplot2DFigure(MatplotlibFigure):
    def __init__(self):
        super().__init__()
        self.colormap = 'inferno'

    def make_axis_legend(self, label: str, x_lims: Iterable[int], y_lims: Iterable[int]) -> None:
        text_pos_x = x_lims[0] + 0.8 * (x_lims[1] - x_lims[0])
        text_pos_y = y_lims[0] + 0.9 * (y_lims[1] - y_lims[0])
        text_box_props = dict(boxstyle='round', facecolor='white', alpha=0.75)
        self.axis.text(text_pos_x, text_pos_y, label, bbox=text_box_props)

#---------------------------------------------------------------------------------------------------
@dataclass 
class FigureCollection:
    num_components: int
    num_timestamps: int
    num_sub_inds: int
    figure_handles: Iterable

    def __post_init__(self):
        if not len(self.figure_handles) == self.num_components:
            raise ValueError(f"Number of components ({self.num_components}) does not match "
                             f"provided figure handle dimension ({len(self.figure_handles)}).")
        for i in range(self.num_components):
            if not len(self.figure_handles[i]) == self.num_timestamps:
                raise ValueError(f"Number of figure handles ({len(self.figure_handles[i])}) "
                                 f"for component {i} does not match "
                                 f"number of timestamps ({self.num_timestamps}).")
            for j in range(self.num_timestamps):
                if not len(self.figure_handles[i][j]) == self.num_sub_inds:
                    raise ValueError(f"Number of figure handles ({len(self.figure_handles[i][j])}) "
                                     f"for component {i} and timestamp {j} does not match "
                                     f"number of sub indices ({self.num_sub_inds}).")

    def prune_and_save(self) -> None:
        for i in range(self.num_components):
            for j in range(self.num_timestamps):
                for k in range(self.num_sub_inds):
                    self.figure_handles[i][j][k].prune_and_save()
    

#======================================== Figure Generator =========================================
class FigureGenerator:

    _num_figs_for_time_dependent = 3
    def __init__(self,
                 inference_data_list: Iterable[dt.InferenceData]) -> None:
        self._data_list = inference_data_list
        self._num_data_structs, self._num_components \
            = check_inference_data_list(inference_data_list)
        self._field_template = inference_data_list[0].field
        self._domain_template = self._field_template.components[0].domain
        self._num_timestamps = None
        self._num_sub_inds = None

    #-----------------------------------------------------------------------------------------------
    def build_figure_collection(self) -> Iterable:
        figure_handles = []
        for _ in range(self._num_components):
            sub_handles = self._build_figures_component(self._field_template)
            figure_handles.append(sub_handles)

        figure_collection = FigureCollection(self._num_components,
                                             self._num_timestamps,
                                             self._num_sub_inds,
                                             figure_handles)
        return figure_collection
    
    #-----------------------------------------------------------------------------------------------
    @multimethod
    def _build_figures_component(self, field: dt.StationaryField) -> Iterable:
        self._num_timestamps = 1
        figure_handles = []
        sub_handles = self._build_figure_component_snapshot(self._domain_template)
        figure_handles.append(sub_handles)

        return figure_handles

    #-----------------------------------------------------------------------------------------------
    @multimethod
    def _build_figures_component(self, field: dt.TimeDependentField) -> Iterable:
        self._num_timestamps = self._num_figs_for_time_dependent
        figure_handles = []
        for _ in range(self._num_timestamps):
            sub_handles = self._build_figure_component_snapshot(self._domain_template)
            figure_handles.append(sub_handles)

        return figure_handles

    #-----------------------------------------------------------------------------------------------
    @multimethod
    def _build_figure_component_snapshot(self, domain: dt.Grid1D) -> Iterable:
        self._num_sub_inds = 1
        figure_handles = []
        handle = Pyplot1DFigure()
        figure_handles.append(handle)

        return figure_handles

    #-----------------------------------------------------------------------------------------------
    @multimethod
    def _build_figure_component_snapshot(self, domain: dt.Grid2D) -> Iterable:
        self._num_sub_inds = 2 * self._num_data_structs
        figure_handles = []
        for _ in range(self._num_sub_inds):
            handle = Pyplot2DFigure()
            figure_handles.append(handle)

        return figure_handles


#=========================================== Visualizer ============================================
class Visualizer:

    _interval_scale = 1.96
    _num_time_inds_to_plot = 3
    _time_inds_to_plot = staticmethod(lambda time_array:
                                      (0, int(time_array.size/2), time_array.size-1))

    def __init__(self):
        self._num_data_structs = None
        self._num_components = None
        self._figure_handles = None

    #============================================ API ==============================================
    def plot_inference_data(self,
                            *inference_data_list: Iterable[dt.InferenceData],
                            path_name: str) -> None:
        figure_generator = FigureGenerator(inference_data_list)
        self._figure_collection = figure_generator.build_figure_collection()
        self._num_data_structs, self._num_components \
            = check_inference_data_list(inference_data_list)
        
        for component in range(self._num_components):
            figs_per_component = self._figure_collection.figure_handles[component]
            for figs_per_timestamp in figs_per_component:
                for fig in figs_per_timestamp:
                    fig.file_name["prefix"] = path_name
                    fig.file_name["component"] = f"{component+1}"
                    fig.title["prefix"] = os.path.basename(path_name)
                    fig.title["component"] = f"{component+1}"

            for list_ind, inference_data in enumerate(inference_data_list):
                self._plot_component((list_ind, component),
                                     inference_data.field,
                                     inference_data.label,
                                     inference_data.is_point_data)
                if inference_data.variance_field:
                    self._plot_component_interval((list_ind, component),
                                                  inference_data.field,
                                                  inference_data.variance_field,
                                                  inference_data.label,
                                                  inference_data.is_point_data)
        self._figure_collection.prune_and_save()

    #================================== Field Level Functions ======================================
    @multimethod
    def _plot_component(self,
                        indices: Iterable[int],
                        field: dt.StationaryField,
                        label:str,
                        is_point_data: bool, /) -> None:
        if not len(indices) == 2:
            raise ValueError("Method needs two indices (param_ind, component)")
        list_ind, component = indices
        domain = field.components[component].domain
        values = field.components[component].values

        self._plot_component_snapshot((list_ind, component, 0),
                                      domain,
                                      values,
                                      label,
                                      is_point_data)

    #-----------------------------------------------------------------------------------------------
    @multimethod 
    def _plot_component(self,
                        indices: Iterable[int],
                        field: dt.TimeDependentField,
                        label:str,
                        is_point_data: bool, /) -> None:
        if not len(indices) == 2:
            raise ValueError("Method needs two indices (param_ind, component)")
        list_ind, component = indices
        domain = field.components[component].domain
        times = field.components[component].times
        time_inds_to_plot = self._time_inds_to_plot(times)

        for i, time_ind in enumerate(time_inds_to_plot):
            values = field.components[component].values[time_ind]
            for handle in self._figure_collection.figure_handles[component][i]:
                handle.file_name["timestamp"] = f"{i+1}"
                handle.title["timestamp"] = f"{times[time_ind]: .3e}"

            self._plot_component_snapshot((list_ind, component, i),
                                          domain,
                                          values,
                                          label,
                                          is_point_data)

    #-----------------------------------------------------------------------------------------------
    @multimethod
    def _plot_component_interval(self,
                                 indices: Iterable[int],
                                 field: dt.StationaryField,
                                 var_field: dt.StationaryField,
                                 label:str,
                                 is_point_data: bool, /) -> None:
        if not len(indices) == 2:
            raise ValueError("Method needs two indices (param_ind, component)")
        list_ind, component = indices
        domain = field.components[component].domain
        mean_values = field.components[component].values
        var_values = var_field.components[component].values

        self._plot_component_interval_snapshot((list_ind, component, 0),
                                               domain,
                                               mean_values,
                                               var_values,
                                               label,
                                               is_point_data)

    #-----------------------------------------------------------------------------------------------
    @multimethod
    def _plot_component_interval(self,
                                 indices: Iterable[int],
                                 field: dt.TimeDependentField,
                                 var_field: dt.TimeDependentField,
                                 label:str,
                                 is_point_data: bool, /) -> None:
        if not len(indices) == 2:
            raise ValueError("Method needs two indices (param_ind, component)")
        list_ind, component = indices
        domain = field.components[component].domain
        times = field.components[component].times
        time_inds_to_plot = self._time_inds_to_plot(times)

        for i, time_ind in enumerate(time_inds_to_plot):
            mean_values = field.components[component].values[time_ind]
            var_values = var_field.components[component].values[time_ind]
            for handle in self.self._figure_collection.figure_handles[component][i][:]:
                handle.file_name["timestamp"] = str(i)
                handle.title["timestamp"] = f"{times[time_ind]: .3e}"
            self._plot_component_interval_snapshot((list_ind, component, 0),
                                                   domain,
                                                   mean_values,
                                                   var_values,
                                                   label,
                                                   is_point_data) 

    #================================== Domain Level Functions =====================================
    @multimethod
    def _plot_component_snapshot(self,
                                 indices,
                                 domain: dt.Grid1D,
                                 values: np.ndarray,
                                 label: str,
                                 is_point_data: bool, /) -> None:
        if not len(indices) == 3:
            raise ValueError("Method needs three indices (param_ind, component, timestamp)")
        _, component, timestamp = indices         
        x_values = domain.locations
        y_values = values
        handle = self._figure_collection.figure_handles[component][timestamp][0]
        handle.file_name["suffix"] = "mean_ci"
        handle.title["suffix"] = "mean and opt. 95% CI" 
        handle.make_axis_title()

        if is_point_data:
            PlottingFunctions.plot_1D_points(handle, x_values, y_values, label)
        else:
            PlottingFunctions.plot_1D_line(handle, x_values, y_values, label)

    #-----------------------------------------------------------------------------------------------
    @multimethod
    def _plot_component_interval_snapshot(self,
                                          indices,
                                          domain: dt.Grid1D,
                                          mean_values: np.ndarray,
                                          var_values: np.ndarray,
                                          label: str,
                                          is_point_data: bool, /) -> None:
        if not len(indices) == 3:
            raise ValueError("Method needs three indices (param_ind, component, timestamp)") 
        _, component, timestamp = indices 
        x_values = domain.locations
        y_mean_values = mean_values
        y_var_values = var_values
        handle = self._figure_collection.figure_handles[component][timestamp][0]

        if is_point_data:
            PlottingFunctions.plot_1D_errorbars(handle,
                                                x_values,
                                                y_mean_values,
                                                y_var_values,
                                                self._interval_scale)
        else:
            PlottingFunctions.plot_1D_interval(handle,
                                               x_values,
                                               y_mean_values,
                                               y_var_values,
                                               self._interval_scale)
    
    #-----------------------------------------------------------------------------------------------
    @multimethod
    def _plot_component_snapshot(self,
                                 indices,
                                 domain: dt.Grid2D,
                                 values: np.ndarray,
                                 label: str,
                                 is_point_data: bool, /) -> None:
        if not len(indices) == 3:
            raise ValueError("Method needs three indices (param_ind, component, timestamp)")
        list_ind, component, timestamp = indices
        x_values, y_values = domain.locations
        z_values = values
        handle = self._figure_collection.figure_handles[component][timestamp][2*list_ind]
        handle.file_name["suffix"] = "mean"
        handle.title["suffix"] = "mean"
        handle.make_axis_title()
        
        if is_point_data:
            PlottingFunctions.plot_2D_grid_points(handle, x_values, y_values, z_values, label)
        else:
            PlottingFunctions.plot_2D_grid_contour(handle, x_values, y_values, z_values, label)

    #-----------------------------------------------------------------------------------------------
    @multimethod
    def _plot_component_interval_snapshot(self,
                                          indices,
                                          domain: dt.Grid2D,
                                          mean_values: np.ndarray,
                                          var_values: np.ndarray,
                                          label: str,
                                          is_point_data: bool, /) -> None:
        if not len(indices) == 3:
            raise ValueError("Method needs three indices (param_ind, component, timestamp)") 
        list_ind, component, timestamp = indices
        x_values, y_values = domain.locations
        z_values = 2 * 1.96 * mean_values
        handle = self._figure_collection.figure_handles[component][timestamp][1+2*list_ind]
        handle.file_name["suffix"] = "variance"
        handle.title["suffix"] = "variance"
        handle.make_axis_title()
        
        if is_point_data:
            PlottingFunctions.plot_2D_grid_points(handle, x_values, y_values, z_values, label)
        else:
            PlottingFunctions.plot_2D_grid_contour(handle, x_values, y_values, z_values, label)


#=================================== Actual Plotting Functions =====================================
class PlottingFunctions:

    #-----------------------------------------------------------------------------------------------
    @staticmethod
    def plot_1D_line(handle: Pyplot1DFigure,
                     x_values: np.ndarray,
                     y_values: np.ndarray,
                     label: str):
        handle.axis.set_xlabel("x")
        plot_color = handle.color_control.get_color()
        handle.axis.set_xlim(np.min(x_values), np.max(x_values))
        handle.axis.plot(x_values, y_values, label=label, color=plot_color)
        handle.axis.legend()

    #-----------------------------------------------------------------------------------------------
    @staticmethod
    def plot_1D_interval(handle: Pyplot1DFigure,
                         x_values: np.ndarray,
                         y_mean_values: np.ndarray,
                         y_var_values: np.ndarray,
                         interval_scale: int):
        plot_color = handle.color_control.get_color(next=False)
        handle.axis.set_xlim(np.min(x_values), np.max(x_values))
        handle.axis.fill_between(x_values,
                                 y_mean_values - interval_scale * np.sqrt(y_var_values),
                                 y_mean_values + interval_scale * np.sqrt(y_var_values),
                                 color=plot_color,
                                 alpha=0.25)

    #-----------------------------------------------------------------------------------------------
    @staticmethod
    def plot_1D_points(handle: Pyplot1DFigure,
                       x_values: np.ndarray,
                       y_values: np.ndarray,
                       label: str):
        handle.axis.set_xlabel("x")
        plot_color = handle.color_control.get_color()
        handle.axis.scatter(x_values, y_values, label=label, color=plot_color)
        handle.axis.legend()

    #-----------------------------------------------------------------------------------------------
    @staticmethod
    def plot_1D_errorbars(handle: Pyplot1DFigure,
                          x_values: np.ndarray,
                          y_mean_values: np.ndarray,
                          y_var_values: np.ndarray,
                          interval_scale: int):
        plot_color = handle.color_control.get_color(next=False)
        error_bar_size = 2 * interval_scale * np.sqrt(y_var_values)
        handle.axis.errorbar(x_values,
                             y_mean_values,
                             yerr=error_bar_size,
                             color=plot_color,
                             fmt=" ",
                             alpha=0.5)
    
    #-----------------------------------------------------------------------------------------------
    @staticmethod
    def plot_2D_grid_contour(handle: Pyplot2DFigure,
                             x_values: np.ndarray,
                             y_values: np.ndarray,
                             z_values: np.ndarray,
                             label: str):
        mg_x_values, mg_y_values = np.meshgrid(x_values, y_values)
        colormap = handle.colormap
        handle.axis.set_xlabel("x")
        handle.axis.set_ylabel("y")
        handle.axis.set_xlim(np.min(x_values), np.max(x_values))
        handle.axis.set_ylim(np.min(y_values), np.max(y_values))
        plot2D = handle.axis.contourf(mg_x_values, mg_y_values, z_values, cmap=colormap)
        handle.fig.colorbar(plot2D)
        handle.make_axis_legend(label,
                               (np.min(x_values), np.max(x_values)),
                               (np.min(y_values), np.max(y_values)))    

    #-----------------------------------------------------------------------------------------------
    @staticmethod
    def plot_2D_grid_points(handle: Pyplot2DFigure,
                            x_values: np.ndarray,
                            y_values: np.ndarray,
                            z_values: np.ndarray,
                            label: str):
        mg_x_values, mg_y_values = np.meshgrid(x_values, y_values)
        stacked_x_values = np.column_stack(np.hsplit(mg_x_values, mg_x_values.shape[1]))
        stacked_y_values = np.column_stack(np.hsplit(mg_y_values, mg_y_values.shape[1]))
        stacked_z_values = np.column_stack(np.hsplit(z_values, z_values.shape[1]))
        colormap = handle.colormap
        handle.axis.set_xlabel("x")
        handle.axis.set_ylabel("y")
        plot2D = handle.axis.scatter(stacked_x_values,
                                     stacked_y_values,
                                     c=stacked_z_values,
                                     cmap=colormap)
        handle.fig.colorbar(plot2D)
        handle.make_axis_legend(label,
                               (np.min(x_values), np.max(x_values)),
                               (np.min(y_values), np.max(y_values)))


#======================================= Plot Hessian Data =========================================
def plot_hessian_data(eigenvalues: np.ndarray, path_name: str) -> None:
    x_values = np.indices(eigenvalues) + 1
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_title('Hessian eigenvalues')
    ax.xet_xlabel(r'i')
    ax.xet_ylabel(r'\lambda (i)')
    ax.scatter(x_values, eigenvalues, color='firebrick')
    ax.axhline(y=1)
    fig.savefig(os.path.join(path_name, 'hessian_eigenvalues.pdf'))


#=========================================== Utilities =============================================
def check_inference_data_list(inference_data_list: Iterable[dt.InferenceData]) -> tuple[int]:
    for i, inference_data in enumerate(inference_data_list[1:]):
            if not inference_data_list[0].is_compatible(inference_data):
                raise TypeError(f"Fields 1 and {i+2} are incompatible.")
            
    num_data_structs = len(inference_data_list)
    num_components = inference_data_list[0].field.num_components
    return num_data_structs, num_components