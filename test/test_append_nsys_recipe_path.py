from ..load_nsight_report import get_nsys_recipe_package_path
import sys

if __name__ == "__main__":
    print(get_nsys_recipe_package_path())

    sys.path.append(get_nsys_recipe_package_path())
    import nsys_recipe

    print(nsys_recipe)

    from ..retrieve_nsys_sm_metrics import GpuMetricUtilReport
