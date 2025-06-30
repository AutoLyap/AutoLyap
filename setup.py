from pathlib import Path
from setuptools import setup, find_packages

here = Path(__file__).parent
long_description = (here / "README.md").read_text(encoding="utf-8")

setup(
    name="autolyap",
    version="0.1.0",
    author="Manu Upadhyaya",
    author_email="manu.upadhyaya.42@gmail.com",
    description="Automatic Lyapunov analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://autolyap.github.io",
    project_urls={
        "Documentation": "https://autolyap.github.io/",
        # later, add "Source": "https://github.com/…"
    },
    license="MIT",
    python_requires=">=3.9",
    packages=find_packages(include=["autolyap", "autolyap.*"]),
    install_requires=[
        "numpy>=1.24,<2.0",
        "mosek>=10.0"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Operating System :: OS Independent",
    ],
)
