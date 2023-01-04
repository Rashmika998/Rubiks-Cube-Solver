from setuptools import setup, find_packages

setup(
    name="rubiksolver",
    version="0.0",
    description="An end-to-end Rubik's Cube Solver (from loading input images of the cube to outputting a step-by-step solution).",
    author="Paul Kepley",
    author_email="pakepley@gmail.com",
    url="https://github.com/pkepley/rubiksolver",
    #package_dir = {'' : 'src'},    
    packages=find_packages('src'),
    package_dir={'' : 'src'},
    package_data={
        'rubiksolver' : ['training_data/*.npy']
    },
    include_package_data=True   
)
