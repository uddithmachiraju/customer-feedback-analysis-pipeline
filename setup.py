from setuptools import setup, find_packages

setup(
    name = "Customer-Feedback-Analysis-Pipeline",
    version = 0.1,
    description = "This is A machine learning pipeline that analyzes customer " \
    "feedback data to extract actionable insights. The system handle data ingestion, " \
    "preprocessing, model training, and deployment for real-time predictions.",
    packages = find_packages() 
)