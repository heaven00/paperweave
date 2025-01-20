from setuptools import setup, find_packages


def read_requirements():
    with open("requirements.txt", "r") as req:
        return req.read().splitlines()


setup(
    name="paperweave",
    version="0.0.1",
    description="use for creating podcast from article",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=read_requirements(),
    python_requires=">=3.10.0",
)
