from pathlib import Path

from isaaclab.app import AppLauncher


simulation_app = AppLauncher(headless=True).app


from isaaclab.sim.converters import MeshConverter, MeshConverterCfg
from pxr import Usd, UsdGeom


FBX_PATH = Path(r"C:\robotics\droneRL\Collected_litterRoad\SubUSDs\textures\Plastic_Bottles.fbx")
OUT_DIR = Path(r"C:\robotics\droneRL\assets\props")
OUT_NAME = "plastic_bottles.usd"


def print_meshes(usd_path):
    stage = Usd.Stage.Open(str(usd_path))
    meshes = [prim.GetPath().pathString for prim in stage.Traverse() if prim.IsA(UsdGeom.Mesh)]
    print(f"converted: {usd_path}", flush=True)
    print(f"mesh_count: {len(meshes)}", flush=True)
    for mesh_path in meshes:
        print(f"mesh: {mesh_path}", flush=True)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg = MeshConverterCfg(
        asset_path=str(FBX_PATH),
        usd_dir=str(OUT_DIR),
        usd_file_name=OUT_NAME,
        force_usd_conversion=True,
        make_instanceable=False,
    )
    converter = MeshConverter(cfg)
    print_meshes(Path(converter.usd_path))
    simulation_app.close()


if __name__ == "__main__":
    main()
