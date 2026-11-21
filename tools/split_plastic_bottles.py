from collections import defaultdict, deque
from pathlib import Path

from isaaclab.app import AppLauncher


simulation_app = AppLauncher(headless=True).app


from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade


SRC_USD = Path(r"C:\robotics\droneRL\assets\props\plastic_bottles.usd")
OUT_DIR = Path(r"C:\robotics\droneRL\assets\props")
TEXTURE_PATH = "../../Collected_litterRoad/SubUSDs/textures/PlasticBottles_Plastic_Bottles_AlbedoTrans.png"


def face_ranges(face_counts):
    start = 0
    ranges = []
    for count in face_counts:
        end = start + count
        ranges.append((start, end))
        start = end
    return ranges


def connected_face_components(face_counts, face_indices):
    vertex_to_faces = defaultdict(list)
    ranges = face_ranges(face_counts)

    for face_id, (start, end) in enumerate(ranges):
        for vertex_id in face_indices[start:end]:
            vertex_to_faces[int(vertex_id)].append(face_id)

    face_neighbors = [[] for _ in face_counts]
    for faces in vertex_to_faces.values():
        for face_id in faces:
            face_neighbors[face_id].extend(other for other in faces if other != face_id)

    components = []
    seen = set()
    for face_id in range(len(face_counts)):
        if face_id in seen:
            continue
        queue = deque([face_id])
        seen.add(face_id)
        component = []

        while queue:
            current = queue.popleft()
            component.append(current)
            for neighbor in face_neighbors[current]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    queue.append(neighbor)

        components.append(sorted(component))

    return components


def make_material(stage, root_path):
    material = UsdShade.Material.Define(stage, f"{root_path}/Looks/plastic_bottle_material")

    shader = UsdShade.Shader.Define(stage, f"{root_path}/Looks/plastic_bottle_material/preview")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.6)

    uv_reader = UsdShade.Shader.Define(stage, f"{root_path}/Looks/plastic_bottle_material/uv_reader")
    uv_reader.CreateIdAttr("UsdPrimvarReader_float2")
    uv_reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")

    texture = UsdShade.Shader.Define(stage, f"{root_path}/Looks/plastic_bottle_material/diffuse_texture")
    texture.CreateIdAttr("UsdUVTexture")
    texture.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(TEXTURE_PATH)
    texture.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(uv_reader.ConnectableAPI(), "result")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(texture.ConnectableAPI(), "rgb")

    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material


def write_component(name, points, face_counts, face_indices, st_values, st_interp, face_ids):
    ranges = face_ranges(face_counts)
    used_vertices = []
    for face_id in face_ids:
        start, end = ranges[face_id]
        used_vertices.extend(int(idx) for idx in face_indices[start:end])

    old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(sorted(set(used_vertices)))}
    new_points = [points[old_idx] for old_idx in old_to_new.keys()]

    center = Gf.Vec3f(0.0, 0.0, 0.0)
    if new_points:
        mins = [min(point[i] for point in new_points) for i in range(3)]
        maxs = [max(point[i] for point in new_points) for i in range(3)]
        center = Gf.Vec3f(*[(mins[i] + maxs[i]) * 0.5 for i in range(3)])
        new_points = [point - center for point in new_points]

    new_counts = []
    new_indices = []
    new_st = []
    for face_id in face_ids:
        start, end = ranges[face_id]
        new_counts.append(face_counts[face_id])
        new_indices.extend(old_to_new[int(idx)] for idx in face_indices[start:end])
        if st_values and st_interp == UsdGeom.Tokens.faceVarying:
            new_st.extend(st_values[start:end])

    out_path = OUT_DIR / f"{name}.usd"
    root_path = f"/{name}"
    stage = Usd.Stage.CreateNew(str(out_path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    root = stage.DefinePrim(root_path, "Xform")
    stage.SetDefaultPrim(root)

    mesh = UsdGeom.Mesh.Define(stage, f"{root_path}/mesh")
    mesh.CreatePointsAttr(new_points)
    mesh.CreateFaceVertexCountsAttr(new_counts)
    mesh.CreateFaceVertexIndicesAttr(new_indices)
    mesh.CreateSubdivisionSchemeAttr("none")

    if new_st:
        st = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying)
        st.Set(new_st)

    material = make_material(stage, root_path)
    UsdShade.MaterialBindingAPI(mesh).Bind(material)
    stage.Save()

    print(f"wrote: {out_path} faces={len(face_ids)} vertices={len(new_points)} center={tuple(center)}", flush=True)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    src_stage = Usd.Stage.Open(str(SRC_USD))
    mesh_prim = next(prim for prim in src_stage.Traverse() if prim.IsA(UsdGeom.Mesh))
    mesh = UsdGeom.Mesh(mesh_prim)

    points = mesh.GetPointsAttr().Get()
    face_counts = list(mesh.GetFaceVertexCountsAttr().Get())
    face_indices = list(mesh.GetFaceVertexIndicesAttr().Get())

    st_primvar = UsdGeom.PrimvarsAPI(mesh).GetPrimvar("st")
    st_values = st_primvar.Get() if st_primvar else []
    st_values = list(st_values) if st_values is not None else []
    st_interp = st_primvar.GetInterpolation() if st_primvar else None

    components = connected_face_components(face_counts, face_indices)
    components = sorted(components, key=len, reverse=True)
    print(f"component_count: {len(components)}", flush=True)

    for i, component in enumerate(components, start=1):
        write_component(f"plastic_bottle_{i}", points, face_counts, face_indices, st_values, st_interp, component)

    simulation_app.close()


if __name__ == "__main__":
    main()
