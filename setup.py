from setuptools import setup, find_packages

setup(
    name="tokenizer-fingerprint",
    version="0.1.0",
    description="Tokenizer fingerprint detection for black-box LLM API attribution and shell detection",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "pyyaml>=6.0",
        "httpx>=0.24.0",
        "click>=8.1.0",
    ],
    extras_require={
        "full": [
            "openai>=1.0.0",
            "anthropic>=0.20.0",
            "scikit-learn>=1.3.0",
            "tqdm>=4.65.0",
            "rich>=13.0.0",
            "pandas>=2.0.0",
            "matplotlib>=3.7.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "tkfp=tokenizer_fingerprint.cli:main",
        ],
    },
)
