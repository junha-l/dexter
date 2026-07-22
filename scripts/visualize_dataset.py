"""Interactive visualization for the DexGYS / Dexonomy datasets.

Renders ground-truth grasps over the object (mesh or point cloud) with optional
contact points. The Shadow Hand can be posed with either the analytic backend
(default) or the MuJoCo kinematics backend (``--use-mujoco``).

An optional **Partial Observation** mode simulates a partial RGB-D view of the
object (camera frustums, visibility masking, sensor noise) for debugging
``dexter.data.rgbd_simulation``.
"""

from typing import Optional

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

from dexter.data.dexgys_tools import DexGYSVisualizationDataset
from dexter.data.dexonomy import DexonomyDataset
from dexter.data.rgbd_simulation import (
    compute_ego_and_thirdperson_cameras,
    compute_eye_in_hand_camera,
    compute_fixed_multiview_cameras,
    compute_grasp_aware_cameras,
    compute_random_viewpoint_cameras,
    compute_single_view_camera,
    simulate_partial_rgbd_observation,
)
from dexter.utils.viz import (
    BaseVisualization,
    Colors,
    Settings,
    pointcloud_colors,
)

# Partial-observation camera modes (see dexter.data.rgbd_simulation)
CAMERA_MODES = [
    "ego_thirdperson",
    "hemisphere",
    "fixed_multiview",
    "random",
    "front",
    "side",
    "overhead",
    "front_elevated",
    "eye_in_hand",
]
# Defaults for the niche sub-parameters we don't expose as sliders.
DEFAULT_FIXED_VIEWS = ["front", "left", "overhead"]
DEFAULT_DISTANCE_RANGE = (0.3, 0.6)
DEFAULT_ELEVATION_RANGE = (0.1, 0.5)

CAMERA_FRUSTUM_COLOR = (1.0, 0.0, 0.0)
ORIGINAL_PC_COLOR = (0.5, 0.5, 0.5)


def compute_look_at_quaternion(camera_pos: np.ndarray, target_pos: np.ndarray) -> np.ndarray:
    """Compute the wxyz quaternion for a camera at ``camera_pos`` looking at ``target_pos``."""
    forward = target_pos - camera_pos
    forward = forward / (np.linalg.norm(forward) + 1e-8)

    world_up = np.array([0.0, 0.0, 1.0])
    right = np.cross(forward, world_up)
    if np.linalg.norm(right) < 1e-6:
        # Looking straight up/down: pick a different reference up.
        world_up = np.array([0.0, 1.0, 0.0])
        right = np.cross(forward, world_up)
    right = right / (np.linalg.norm(right) + 1e-8)

    up = np.cross(right, forward)
    up = up / (np.linalg.norm(up) + 1e-8)

    # viser convention: +Z is the look direction.
    rot_matrix = np.stack([right, -up, forward], axis=1)
    quat_xyzw = Rotation.from_matrix(rot_matrix).as_quat()
    return np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])


def compute_cameras(
    camera_mode: str,
    grasp_pose: np.ndarray,
    pointcloud: np.ndarray,
    num_views: int,
    camera_radius: float,
    seed: int,
) -> np.ndarray:
    """Return camera positions for ``camera_mode`` (for frustum rendering)."""
    if camera_mode == "hemisphere":
        return compute_grasp_aware_cameras(
            grasp_pose, pointcloud, num_views=num_views, camera_radius=camera_radius, seed=seed
        )
    if camera_mode == "fixed_multiview":
        return compute_fixed_multiview_cameras(
            pointcloud,
            camera_distance=camera_radius,
            elevation=camera_radius * 0.6,
            views=DEFAULT_FIXED_VIEWS,
        )
    if camera_mode == "random":
        return compute_random_viewpoint_cameras(
            pointcloud,
            num_views=num_views,
            camera_distance_range=DEFAULT_DISTANCE_RANGE,
            elevation_range=DEFAULT_ELEVATION_RANGE,
            seed=seed,
        )
    if camera_mode in ("front", "side", "overhead", "front_elevated"):
        return compute_single_view_camera(
            pointcloud,
            view_type=camera_mode,
            camera_distance=camera_radius,
            elevation=camera_radius * 0.4,
        )
    if camera_mode == "eye_in_hand":
        return compute_eye_in_hand_camera(grasp_pose, pointcloud, offset=camera_radius)
    # Default: ego + third-person
    return compute_ego_and_thirdperson_cameras(grasp_pose, pointcloud, ego_offset=camera_radius)


class DatasetVisualization(BaseVisualization):
    """Interactive viser visualization for a grasp dataset."""

    def __init__(
        self,
        dataset,
        port: int = 8080,
        device: str = "cuda",
        use_mujoco: bool = False,
    ) -> None:
        super().__init__(port=port, device=device, use_mujoco=use_mujoco)
        self.dataset = dataset
        self._scene_index = self._build_scene_index()

        self.current_sample: Optional[dict] = None
        self.current_hand_mesh: Optional[dict] = None
        self.current_object_mesh: Optional[trimesh.Trimesh] = None
        self._last_idx: int = -1

        self._setup_gui()

    def _setup_gui(self) -> None:
        """Setup viser GUI elements."""
        backend = "MuJoCo" if self.use_mujoco else "analytic"
        self._create_server("DexGYS Dataset Viewer")

        self.server.gui.add_markdown(f"# Dataset Visualization ({backend})")
        self.server.gui.add_markdown(
            f"**Split:** {self.dataset.split} | **Samples:** {len(self.dataset)}"
        )

        with self.server.gui.add_folder("Sample Navigation"):
            self.prev_button = self.server.gui.add_button("Previous Sample")
            self.next_button = self.server.gui.add_button("Next Sample")
            self.sample_slider = self.server.gui.add_slider(
                "Sample Index", min=0, max=len(self.dataset) - 1, step=1, initial_value=0
            )

        with self.server.gui.add_folder("Find by Scene"):
            first_scene = self._scene_names[0]
            self.scene_dropdown = self.server.gui.add_dropdown(
                "Scene Name", options=self._scene_names, initial_value=first_scene
            )
            self.grasp_in_scene = self.server.gui.add_slider(
                "Grasp in Scene",
                min=0,
                max=max(0, len(self._scene_index[first_scene]) - 1),
                step=1,
                initial_value=0,
            )
            self.scene_search_result = self.server.gui.add_markdown("")

        with self.server.gui.add_folder("Display Options"):
            self.show_rgb = self.server.gui.add_checkbox("Show RGB Colors", initial_value=True)
            self.show_mask = self.server.gui.add_checkbox(
                "Show Segmentation Mask", initial_value=False
            )
            self.show_hand = self.server.gui.add_checkbox("Show Hand", initial_value=True)
            self.show_contact_points = self.server.gui.add_checkbox(
                "Show Contact Points", initial_value=True
            )
            self.hand_opacity = self.server.gui.add_slider(
                "Hand Opacity",
                min=0.0,
                max=1.0,
                step=0.05,
                initial_value=Settings.HAND_OPACITY_DEFAULT,
            )
            self.point_size = self.server.gui.add_slider(
                "Point Size", min=0.001, max=0.02, step=0.001, initial_value=0.005
            )

        self._setup_partial_obs_gui()

        self.text_display = self.server.gui.add_markdown("")
        self.info_display = self.server.gui.add_markdown("")

        self._register_callbacks()

    def _setup_partial_obs_gui(self) -> None:
        """Setup the partial RGB-D observation controls."""
        with self.server.gui.add_folder("Partial Observation"):
            self.enable_partial_obs = self.server.gui.add_checkbox(
                "Enable Partial Obs", initial_value=False
            )
            self.partial_obs_mode = self.server.gui.add_dropdown(
                "Camera Mode", options=CAMERA_MODES, initial_value=Settings.PARTIAL_OBS_MODE
            )
            self.num_views = self.server.gui.add_slider(
                "Num Views (hemisphere/random)", min=1, max=8, step=1, initial_value=3
            )
            self.camera_radius = self.server.gui.add_slider(
                "Camera Radius / Ego Offset",
                min=0.1,
                max=1.0,
                step=0.05,
                initial_value=Settings.PARTIAL_OBS_CAMERA_RADIUS,
            )
            self.seed = self.server.gui.add_number("Random Seed", initial_value=Settings.PARTIAL_OBS_SEED, step=1)
            self.show_cameras = self.server.gui.add_checkbox("Show Cameras", initial_value=True)
            self.show_original = self.server.gui.add_checkbox(
                "Show Original PC (gray)", initial_value=True
            )
            self.enable_noise = self.server.gui.add_checkbox(
                "Add Sensor Noise", initial_value=False
            )
            self.depth_noise = self.server.gui.add_slider(
                "Depth Noise (mm)",
                min=0.0,
                max=10.0,
                step=0.5,
                initial_value=Settings.PARTIAL_OBS_DEPTH_NOISE,
            )
            self.lateral_noise = self.server.gui.add_slider(
                "Lateral Noise (mm)",
                min=0.0,
                max=5.0,
                step=0.25,
                initial_value=Settings.PARTIAL_OBS_LATERAL_NOISE,
            )
            self.outlier_ratio = self.server.gui.add_slider(
                "Outlier Ratio (%)",
                min=0.0,
                max=5.0,
                step=0.5,
                initial_value=Settings.PARTIAL_OBS_OUTLIER_RATIO,
            )

    def _register_callbacks(self) -> None:
        """Register UI callbacks for interactive controls."""
        self.prev_button.on_click(lambda _: self._navigate_sample(-1))
        self.next_button.on_click(lambda _: self._navigate_sample(1))
        self.scene_dropdown.on_update(lambda _: self._on_scene_select())
        self.grasp_in_scene.on_update(lambda _: self._on_grasp_select())

        for control in [
            self.sample_slider,
            self.show_rgb,
            self.show_mask,
            self.show_hand,
            self.show_contact_points,
            self.hand_opacity,
            self.point_size,
            self.enable_partial_obs,
            self.partial_obs_mode,
            self.num_views,
            self.camera_radius,
            self.seed,
            self.show_cameras,
            self.show_original,
            self.enable_noise,
            self.depth_noise,
            self.lateral_noise,
            self.outlier_ratio,
        ]:
            control.on_update(lambda _: self._update_visualization())

    def _navigate_sample(self, delta: int) -> None:
        """Navigate to the previous or next sample."""
        new_idx = max(0, min(len(self.dataset) - 1, self.sample_slider.value + delta))
        self.sample_slider.value = new_idx

    def _build_scene_index(self) -> dict[str, list[int]]:
        """Map ``scene_name`` -> sample indices, derived cheaply from dataset internals.

        DexGYS datasets expose ``(scene_name, query_index)`` pairs and Dexonomy
        stores one directory per sample, so scene names can be recovered without
        loading every sample. Falls back to loading each sample if neither is
        available.
        """
        dataset = self.dataset

        if hasattr(dataset, "flatten_indices"):
            scene_names = [scene_name for scene_name, _ in dataset.flatten_indices]
        elif hasattr(dataset, "files"):
            # Path structure: .../taxonomy/object_id/floating/scale_id
            scene_names = [
                f"{d.parent.parent.parent.name}/{d.parent.parent.name}/{d.name}"
                for d in dataset.files
            ]
        else:
            scene_names = [dataset[i]["scene_name"] for i in range(len(dataset))]

        scene_index: dict[str, list[int]] = {}
        for idx, name in enumerate(scene_names):
            scene_index.setdefault(name, []).append(idx)
        self._scene_names = sorted(scene_index.keys())
        return scene_index

    def _on_scene_select(self) -> None:
        """Point the grasp slider at the selected scene and jump to its first grasp."""
        indices = self._scene_index[self.scene_dropdown.value]
        self.grasp_in_scene.max = len(indices) - 1
        # Setting value fires _on_grasp_select, which updates the sample slider.
        if self.grasp_in_scene.value == 0:
            self._on_grasp_select()
        else:
            self.grasp_in_scene.value = 0

    def _on_grasp_select(self) -> None:
        """Jump to the selected grasp within the currently selected scene."""
        scene_name = self.scene_dropdown.value
        indices = self._scene_index[scene_name]
        grasp_idx = max(0, min(len(indices) - 1, int(self.grasp_in_scene.value)))
        self.sample_slider.value = indices[grasp_idx]  # triggers _update_visualization
        self._update_markdown(
            self.scene_search_result,
            f"**{scene_name}** — grasp {grasp_idx + 1}/{len(indices)} "
            f"(sample {indices[grasp_idx] + 1}).",
        )

    def _load_sample(self, idx: int) -> None:
        """Load sample from dataset and compute the hand mesh."""
        print(f"Loading sample {idx}...")
        self.current_sample = self.dataset[idx]

        self.current_object_mesh = None
        mesh_path = self.current_sample.get("mesh_path")
        if mesh_path is not None and mesh_path.exists():
            self.current_object_mesh = trimesh.load(mesh_path)

        hand_out = self.hand.compute_meshes(
            self.current_sample["actions"], parse_28d=self.use_mujoco
        )
        self.current_hand_mesh = {
            "vertices": hand_out["vertices"][0],
            "faces": hand_out["faces"],
        }

        print(f"Sample loaded: {len(self.current_sample['pointcloud'])} points")

    def _update_visualization(self) -> None:
        """Update the visualization based on current settings."""
        idx = self.sample_slider.value

        if self.current_sample is None or idx != self._last_idx:
            self._load_sample(idx)
            self._last_idx = idx

        self._clear_scene(["/object", "/partial_pc", "/cameras", "/hand", "/contact_points"])

        if self.enable_partial_obs.value:
            self._render_partial_observation()
        else:
            self._render_object()

        if self.show_hand.value:
            self._render_hand_mesh(
                "/hand",
                self.current_hand_mesh["vertices"],
                self.current_hand_mesh["faces"],
                Colors.HAND,
                self.hand_opacity.value,
            )

        if self.show_contact_points.value:
            self._render_contact_points()

        self._update_info()

    def _render_object(self) -> None:
        """Render the object as a mesh or point cloud."""
        if self.current_object_mesh is not None:
            self.server.scene.add_mesh_trimesh("/object", mesh=self.current_object_mesh)
            return

        pointcloud = self.current_sample["pointcloud"]
        colors = pointcloud_colors(
            pointcloud,
            mask=self.current_sample["mask"],
            show_mask=self.show_mask.value,
            show_rgb=self.show_rgb.value,
        )
        self.server.scene.add_point_cloud(
            "/object",
            points=pointcloud[:, :3],
            colors=colors,
            point_size=self.point_size.value,
        )

    def _render_partial_observation(self) -> None:
        """Simulate and render a partial RGB-D observation of the object."""
        pointcloud = self.current_sample["pointcloud"]
        grasp_pose = self.current_sample["actions"]
        mode = self.partial_obs_mode.value

        partial_pc, mask, stats = simulate_partial_rgbd_observation(
            pointcloud,
            grasp_pose,
            num_views=int(self.num_views.value),
            camera_radius=float(self.camera_radius.value),
            seed=int(self.seed.value),
            camera_mode=mode,
            add_noise=self.enable_noise.value,
            depth_noise_std=self.depth_noise.value / 1000.0,  # mm to meters
            lateral_noise_std=self.lateral_noise.value / 1000.0,  # mm to meters
            outlier_ratio=self.outlier_ratio.value / 100.0,  # % to ratio
            fixed_views=DEFAULT_FIXED_VIEWS,
            camera_distance_range=DEFAULT_DISTANCE_RANGE,
            elevation_range=DEFAULT_ELEVATION_RANGE,
        )
        self._partial_obs_stats = stats

        # Original cloud (gray, behind the partial one)
        if self.show_original.value:
            self.server.scene.add_point_cloud(
                "/object",
                points=pointcloud[:, :3],
                colors=np.full((len(pointcloud), 3), ORIGINAL_PC_COLOR),
                point_size=self.point_size.value * 0.5,
            )

        # Visible (partial) cloud, colored by mask/RGB like the full view
        colors = pointcloud_colors(
            pointcloud[mask],
            mask=self.current_sample["mask"][mask],
            show_mask=self.show_mask.value,
            show_rgb=self.show_rgb.value,
        )
        self.server.scene.add_point_cloud(
            "/partial_pc",
            points=partial_pc[:, :3],
            colors=colors,
            point_size=self.point_size.value,
        )

        # Camera frustums
        if self.show_cameras.value:
            obj_center = pointcloud[:, :3].mean(axis=0)
            cameras = compute_cameras(
                mode,
                grasp_pose,
                pointcloud,
                num_views=int(self.num_views.value),
                camera_radius=float(self.camera_radius.value),
                seed=int(self.seed.value),
            )
            for i, cam_pos in enumerate(cameras):
                quat_wxyz = compute_look_at_quaternion(cam_pos, obj_center)
                self.server.scene.add_camera_frustum(
                    f"/cameras/cam_{i}",
                    fov=np.pi / 3,
                    aspect=1.0,
                    scale=0.05,
                    wxyz=tuple(quat_wxyz),
                    position=tuple(cam_pos),
                    color=CAMERA_FRUSTUM_COLOR,
                )

    def _render_contact_points(self) -> None:
        """Render contact points with joint name labels if available."""
        contact_data = self.current_sample.get("contact")
        if not contact_data:
            return

        for joint_name, positions in contact_data.items():
            positions = np.asarray(positions).reshape(-1)
            self.server.scene.add_icosphere(
                f"/contact_points/sphere/{joint_name}",
                radius=Settings.CONTACT_POINT_RADIUS,
                color=Colors.CONTACT_POINT,
                position=positions,
            )
            self.server.scene.add_label(
                f"/contact_points/label/{joint_name}",
                text=joint_name,
                position=positions,
            )

    def _update_info(self) -> None:
        """Update the info text displays."""
        idx = self.sample_slider.value
        sample = self.current_sample

        self._update_markdown(self.text_display, f"**Text Prompt:** {sample['prompt']}")

        num_contact_points = 0
        if sample.get("contact"):
            for positions in sample["contact"].values():
                num_contact_points += len(positions) if np.ndim(positions) > 1 else 1

        info = (
            f"**Sample:** {idx + 1}/{len(self.dataset)}  \n"
            f"**Points:** {len(sample['pointcloud'])}  \n"
            f"**Mask Ratio:** {sample['mask'].mean():.2%}  \n"
            f"**Contact Points:** {num_contact_points}  \n"
        )
        if self.enable_partial_obs.value and getattr(self, "_partial_obs_stats", None):
            stats = self._partial_obs_stats
            info += (
                f"**Partial Obs:** {stats['original_points']} → {stats['visible_points']} pts "
                f"({stats['visibility_ratio']:.1%}), mode={self.partial_obs_mode.value}  \n"
            )
        self._update_markdown(self.info_display, info)

    def run(self) -> None:
        """Run the visualization server."""
        self._update_visualization()
        self.run_server_loop()


def _create_dataset(data_path: str, split: str, subsets: list[str]):
    """Instantiate the dataset implied by ``data_path``."""
    if "dexonomy" in data_path:
        return DexonomyDataset(data_path=data_path, subsets=subsets, split=split)
    elif "dexgys" in data_path:
        return DexGYSVisualizationDataset(data_path=data_path, split=split)
    raise ValueError(f"Unsupported dataset path: {data_path}")


def main(
    data_path: str,
    split: str = "train",
    subsets: list[str] = ["1_Large_Diameter"],
    port: int = 8080,
    device: str = "cuda",
    use_mujoco: bool = False,
) -> None:
    """Main entry point for dataset visualization.

    Args:
        data_path: Path to dataset directory (selects DexGYS vs Dexonomy).
        split: Dataset split ('train' or 'test').
        subsets: Dataset subsets for Dexonomy.
        port: Port for the viser server.
        device: Device for hand model computation.
        use_mujoco: Pose the hand with the MuJoCo backend instead of the analytic one.
    """
    dataset = _create_dataset(data_path, split, subsets)
    print(f"Dataset loaded: {len(dataset)} samples")

    viz = DatasetVisualization(dataset=dataset, port=port, device=device, use_mujoco=use_mujoco)
    viz.run()


if __name__ == "__main__":
    import fire

    fire.Fire(main)
