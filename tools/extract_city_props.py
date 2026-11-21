import shutil
from pathlib import Path

import numpy as np
import trimesh
from isaaclab.app import AppLauncher


GLB_PATH = Path(r"C:\robotics\droneRL\Collected_litterRoad\SubUSDs\textures\city_props_collection_volume_1.glb")
ASSET_DIR = Path(r"C:\robotics\droneRL\assets\props")
TEXTURE_DIR = ASSET_DIR / "textures"
TMP_DIR = ASSET_DIR / "_tmp_glb"

PROPS = {
    "TrashCanLarge": "trash_can_large",
    "PostBarrier": "post_barrier",
    "Crossing": "crossing",
    "Dumpster": "dumpster",
    "Light": "light_pole",
    "FireHydrant": "fire_hydrant",
    "ConcreteBench": "concrete_bench",
    "BusStop": "bus_stop",
    "Utility": "utility",
    "Tarp": "tarp",
    "pallet1": "pallet1",
}


def descendant_geometry_nodes(scene, root_name):
    stack = [root_name]
    found = []
    while stack:
        node = stack.pop()
        if node in scene.graph.nodes_geometry:
            found.append(node)
        stack.extend(scene.graph.transforms.children.get(node, []))
    return found


def safe_name(name):
    return "".join(c if c.isalnum() or c == "_" else "_" for c in name)


def material_color(geometry):
    material = getattr(getattr(geometry, "visual", None), "material", None)
    color = getattr(material, "main_color", None)
    if color is None:
        return Gf.Vec3f(0.5, 0.5, 0.5)
    return Gf.Vec3f(float(color[0]) / 255.0, float(color[1]) / 255.0, float(color[2]) / 255.0)


def save_texture(image, path):
    if image is None:
        return None
    if image.mode not in ("RGB", "RGBA"):
        image = image.convert("RGBA")
    image.save(path)
    return f"textures/{path.name}"


def create_material(stage, path, geometry, texture_stem):
    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f"{path}/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.7)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)

    source_material = getattr(getattr(geometry, "visual", None), "material", None)
    texture_image = getattr(source_material, "baseColorTexture", None)
    texture_path = save_texture(texture_image, TEXTURE_DIR / f"{texture_stem}_base_color.png")

    if texture_path is None:
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(material_color(geometry))
    else:
        uv_reader = UsdShade.Shader.Define(stage, f"{path}/UvReader")
        uv_reader.CreateIdAttr("UsdPrimvarReader_float2")
        uv_reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
        uv_reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)

        texture = UsdShade.Shader.Define(stage, f"{path}/BaseColorTexture")
        texture.CreateIdAttr("UsdUVTexture")
        texture.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(texture_path))
        texture.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(uv_reader.ConnectableAPI(), "result")
        texture.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)

        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
            texture.ConnectableAPI(), "rgb"
        )

    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material


def write_prop_usd(scene, root_name, usd_path):
    nodes = descendant_geometry_nodes(scene, root_name)
    if not nodes:
        raise RuntimeError(f"No geometry found under GLB node: {root_name}")

    transformed = []
    all_vertices = []
    for node in nodes:
        world_tf, geometry_name = scene.graph[node]
        geometry = scene.geometry[geometry_name].copy()
        geometry.apply_transform(world_tf)
        geometry.vertices = geometry.vertices[:, [0, 2, 1]]
        transformed.append((node, geometry))
        all_vertices.append(geometry.vertices)

    vertices = np.vstack(all_vertices)
    bounds_min = vertices.min(axis=0)
    bounds_max = vertices.max(axis=0)
    offset = np.array([
        -(bounds_min[0] + bounds_max[0]) * 0.5,
        -(bounds_min[1] + bounds_max[1]) * 0.5,
        -bounds_min[2],
    ])

    stage = Usd.Stage.CreateNew(str(usd_path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(root.GetPrim())

    for node, geometry in transformed:
        geometry.apply_translation(offset)
        mesh_path = f"/World/{safe_name(node)}"
        mesh = UsdGeom.Mesh.Define(stage, mesh_path)
        mesh.CreatePointsAttr([Gf.Vec3f(float(x), float(y), float(z)) for x, y, z in geometry.vertices])
        mesh.CreateFaceVertexCountsAttr([3] * len(geometry.faces))
        mesh.CreateFaceVertexIndicesAttr([int(i) for i in geometry.faces.reshape(-1)])
        mesh.CreateSubdivisionSchemeAttr("none")

        uv = getattr(geometry.visual, "uv", None)
        if uv is not None:
            st = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
                "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex
            )
            st.Set([Gf.Vec2f(float(u), float(v)) for u, v in uv])

        material_path = f"/World/Looks/{safe_name(node)}_material"
        material = create_material(stage, material_path, geometry, safe_name(node))
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)

    stage.GetRootLayer().Save()


def export_prop_usds():
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    TEXTURE_DIR.mkdir(parents=True, exist_ok=True)

    scene = trimesh.load(GLB_PATH, force="scene")

    for root_name, file_stem in PROPS.items():
        usd_path = ASSET_DIR / f"{file_stem}.usd"
        write_prop_usd(scene, root_name, usd_path)
        print(usd_path)


app_launcher = AppLauncher(headless=True, device="cuda:0")
simulation_app = app_launcher.app

from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade

try:
    if TMP_DIR.exists():
        shutil.rmtree(TMP_DIR)
    export_prop_usds()
finally:
    simulation_app.close()
