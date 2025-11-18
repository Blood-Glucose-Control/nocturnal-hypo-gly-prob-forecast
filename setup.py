from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="nocturnal-forecast",
    version="0.1.0",
    author="Blood Glucose Control Team",
    description="Nocturnal hypoglycemia and glycemic forecasting",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        # Core dependencies will be read from requirements.txt
    ],
)
