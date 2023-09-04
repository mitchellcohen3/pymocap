from setuptools import setup, find_packages

setup(
    name="pymocap",
    version="0.0.1",
    description="Various tools for working with mocap data",
    packages=find_packages(),
    install_requires = [
        "csaps",
        "numpy",
        "rosbags",
        "scipy",
        "pymlg @ git+https://github.com/decargroup/pymlg@main",
    ]
)
