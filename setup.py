from setuptools import setup, find_packages

setup(
    name="efficientstreamod",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "opencv-python",
        "numpy",
    ],
    author="Ruby Abdullah",
    author_email="rubyabdullah14@gmail.com",
    description="Efficient Object Detection on video streams using OpenCV",
    license="MIT",
    keywords="efficient object detection opencv video stream",
    url="https://github.com/rubythalib33/EfficientStreamOD",
)
