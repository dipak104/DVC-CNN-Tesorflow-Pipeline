from setuptools import setup

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

## edit below variables as per your requirements -
REPO_NAME = "DEV-CNN-Tensorflow-Pipeline"
AUTHOR_USER_NAME = "dipak104"
SRC_REPO = "src"
LIST_OF_REQUIREMENTS = [
    "tqdm==4.62.3",
    "dvc==2.7.2",
    "black==21.7b0",
    "tensorflow==2.5.0"
]


setup(
    name=SRC_REPO,
    version="0.0.2",
    author="Dipak Tripathi",
    description="A small package for DVC",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
    author_email="dipaktripathi10@gmail.com",
    packages=[SRC_REPO],
    license="MIT",
    python_requires=">=3.6",
    install_requires=LIST_OF_REQUIREMENTS
)
