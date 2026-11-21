from pathlib import Path

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})

from pxr import Usd, UsdGeom, UsdPhysics


PROP_USDS = [
    Path(r"C:\robotics\droneRL\assets\props\post_barrier.usd"),
    Path(r"C:\robotics\droneRL\assets\props\light_pole.usd"),
    Path(r"C:\robotics\droneRL\assets\props\bus_stop.usd"),
]


def add_mesh_collisions(usd_path, approximation="convexDecomposition"):
    stage = Usd.Stage.Open(str(usd_path))
    if stage is None:
        raise RuntimeError(f"Could not open USD: {usd_path}")

    mesh_count = 0
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue

        UsdPhysics.CollisionAPI.Apply(prim)
        mesh_collision = UsdPhysics.MeshCollisionAPI.Apply(prim)
        mesh_collision.CreateApproximationAttr().Set(approximation)
        mesh_count += 1

    stage.GetRootLayer().Save()
    return mesh_count


def main():
    for usd_path in PROP_USDS:
        mesh_count = add_mesh_collisions(usd_path)
        print(f"{usd_path.name}: added collision to {mesh_count} mesh prims", flush=True)

    simulation_app.close()


if __name__ == "__main__":
    main()
