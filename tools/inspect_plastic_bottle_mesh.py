from isaaclab.app import AppLauncher


simulation_app = AppLauncher(headless=True).app


from pxr import Usd, UsdGeom


USD_PATH = r"C:\robotics\droneRL\assets\props\plastic_bottles.usd"


def main():
    stage = Usd.Stage.Open(USD_PATH)
    mesh_prim = next(prim for prim in stage.Traverse() if prim.IsA(UsdGeom.Mesh))
    mesh = UsdGeom.Mesh(mesh_prim)

    print(f"mesh_path: {mesh_prim.GetPath()}", flush=True)
    print(f"points: {len(mesh.GetPointsAttr().Get())}", flush=True)
    print(f"faces: {len(mesh.GetFaceVertexCountsAttr().Get())}", flush=True)

    attrs = [attr.GetName() for attr in mesh_prim.GetAttributes()]
    for attr in attrs:
        print(f"attr: {attr}", flush=True)

    rels = [rel.GetName() for rel in mesh_prim.GetRelationships()]
    for rel in rels:
        print(f"rel: {rel}", flush=True)

    simulation_app.close()


if __name__ == "__main__":
    main()
