from setuptools import setup

with open("requirements.txt") as f:
    install_requires = f.read()

setup(
    name="sql_to_text",
    version="1.0",
    install_requires=install_requires,
)
