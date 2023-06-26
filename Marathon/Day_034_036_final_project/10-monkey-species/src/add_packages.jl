using Pkg

# add required packages
used_packages = [
    "Flux", 
    "DataFrames", 
    "Images", 
    "Plots", 
    "Parameters",
    "TensorBoardLogger",
    "ProgressMeter",
    "DrWatson",
    "BSON"
]
Pkg.add(used_packages)
Pkg.installed()