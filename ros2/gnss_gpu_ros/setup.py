from setuptools import find_packages, setup

package_name = "gnss_gpu_ros"

setup(
    name=package_name,
    version="0.2.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", ["launch/robust_navsat_filter.launch.py"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Ryohei Sasaki",
    maintainer_email="rsasaki0109@gmail.com",
    description=(
        "Robust GNSS fix filtering for outdoor robots: Hampel spike gate + "
        "constant-velocity Kalman filter on NavSatFix."
    ),
    license="Apache License 2.0",
    extras_require={"test": ["pytest"]},
    entry_points={
        "console_scripts": [
            "robust_navsat_filter = gnss_gpu_ros.robust_navsat_filter_node:main",
        ],
    },
)
