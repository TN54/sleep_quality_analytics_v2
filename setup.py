from setuptools import setup, find_packages

setup(
    name="sleep_quality_analytics",
    version="1.0",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "numpy",
        "scikit-learn",
        "matplotlib",
        "seaborn",
        "streamlit",
        "fastapi",
        "pydantic",
        "uvicorn",
        "Pillow",
        "requests"
    ],
    python_requires='>=3.8',
)
