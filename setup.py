from setuptools import find_packages, setup

with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name='fastapi_redis_vet1ments',
    description="redis 쉽게",
    long_description=long_description,
    long_description_content_type='text/markdown',
    version='{{VERSION_PLACEHOLDER}}',
    python_requires='>=3.11.0',
    author="no hong seok",
    author_email="vet1ments@naver.com",
    maintainer="no hong seok",
    maintainer_email="vet1ments@naver.com",
    project_urls={
        "Repository": "https://github.com/vet1ments/fastapi_redis"
    },
    license="MIT",
    packages=find_packages(),
    install_requires=[
        "redis[hiredis] >= 5.0.0",
        "fastapi[all] >= 0.111.0"
    ],
    tests_require=[
        "uvicorn[standard] >= 0.29.0",
        "fastapi[all] >= 0.111.0"
    ]

)