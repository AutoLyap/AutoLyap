from pathlib import Path
from setuptools import setup, find_packages

here = Path(__file__).parent
long_description = (here / "README.md").read_text(encoding="utf-8")
version = (here / "VERSION").read_text(encoding="utf-8").strip()

requirements = []
for line in (here / "requirements.txt").read_text(encoding="utf-8").splitlines():
    line = line.strip()
    if line and not line.startswith("#"):
        requirements.append(line)

setup(
    name="autolyap",
    version=version,
    author="Manu Upadhyaya",
    author_email="manu.upadhyaya.42@gmail.com",
    description="Automatic Lyapunov analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://autolyap.github.io",
    project_urls={
        "Documentation": "https://autolyap.github.io/",
        "Source": "https://github.com/AutoLyap/AutoLyap",
    },
    license="MIT",
    python_requires=">=3.9",
    packages=find_packages(include=["autolyap", "autolyap.*"]),
    install_requires=requirements,
    extras_require={
        "mosek": [
            "mosek",
        ],
        "test": [
            "pytest>=7.4",
            "pytest-cov>=4.1",
            "ruff",
            "mypy",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Operating System :: OS Independent",
    ],
)
