"""Visualization helpers for the kitscenes dataset.

Public API::

    from kitscenes.visualization import (
        # scene-level (ego / cameras)
        render_scene_bev,
        render_scene_animation,
        render_surround_view,
        plot_map_interactive,
        # HD map (ML converter — tested pipeline)
        render_map_top_down,
        render_map_top_down_for_frame,
        MapProjector,
        create_map_projector,
        project_scene_maps,
        generate_map_projection_video,
        # frame-level (sensors)
        render_lidar_bev,
        render_lidar_on_camera,
        render_radar_bev,
        render_radar_on_camera,
        render_frame_overview,
    )
"""

from kitscenes.visualization.scene_viz import (
    render_scene_animation,
    render_scene_bev,
    render_surround_view,
)
from kitscenes.visualization.frame_viz import (
    render_frame_overview,
    render_lidar_bev,
    render_lidar_on_camera,
    render_radar_bev,
    render_radar_on_camera,
)

__all__ = [
    # scene-level
    "render_scene_bev",
    "render_scene_animation",
    "render_surround_view",
    # HD map projection pipeline
    "MapProjector",
    "create_map_projector",
    "project_scene_maps",
    "render_map_top_down",
    "render_map_top_down_for_frame",
    "generate_map_labels_for_scene",
    "get_map_data_for_ego_pose",
    "get_map_data_for_frame",
    "generate_map_projection_video",
    "create_composite_frame",
    "get_projection_frame_indices",
    # frame-level sensors
    "render_lidar_bev",
    "render_lidar_on_camera",
    "render_radar_bev",
    "render_radar_on_camera",
    "render_frame_overview",
    # ML converter utilities
    "get_map_data",
    "plot_map_data",
    "TYPE_GROUPINGS",
]

# plot_map_interactive is imported lazily (requires plotly)
try:
    from kitscenes.visualization.scene_viz import plot_map_interactive

    __all__.append("plot_map_interactive")
except ImportError:
    pass

# HD map / projection modules (optional — cv2, lanelet2 ml_converter, …)
try:
    from kitscenes.visualization.map_projection import MapProjector
    from kitscenes.visualization.map_viz import (
        create_map_projector,
        generate_map_labels_for_scene,
        get_map_data_for_ego_pose,
        get_map_data_for_frame,
        project_scene_maps,
        render_map_top_down,
        render_map_top_down_for_frame,
    )
    from kitscenes.visualization.ml_converter_vis_utils import (
        TYPE_GROUPINGS,
        get_map_data,
        plot_map_data,
    )
    from kitscenes.visualization.video_generation import (
        create_composite_frame,
        generate_map_projection_video,
        get_frame_indices as get_projection_frame_indices,
    )
except ImportError:
    MapProjector = None  # type: ignore[misc, assignment]
    create_map_projector = None  # type: ignore[misc, assignment]
    project_scene_maps = None  # type: ignore[misc, assignment]
    render_map_top_down = None  # type: ignore[misc, assignment]
    render_map_top_down_for_frame = None  # type: ignore[misc, assignment]
    generate_map_labels_for_scene = None  # type: ignore[misc, assignment]
    get_map_data_for_ego_pose = None  # type: ignore[misc, assignment]
    get_map_data_for_frame = None  # type: ignore[misc, assignment]
    generate_map_projection_video = None  # type: ignore[misc, assignment]
    create_composite_frame = None  # type: ignore[misc, assignment]
    get_projection_frame_indices = None  # type: ignore[misc, assignment]
    get_map_data = None  # type: ignore[misc, assignment]
    plot_map_data = None  # type: ignore[misc, assignment]
    TYPE_GROUPINGS = {}  # type: ignore[misc, assignment]
