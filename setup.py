from setuptools import setup, find_packages


setup(name="news_classification",
      version="0.0.1",
      description="Classifies news articles to seperate classes",
      author="Chaitanya Lakhchaura",
      license="MIT",
      packages= find_packages(),
      install_requires=[
          "spacy>=3.6.0",
          "pandas>=2.0.3",
          "scikit-learn>=1.3.0",
          "matplotlib>=3.7.2"
      ])