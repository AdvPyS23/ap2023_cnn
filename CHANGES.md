Rearranged the entire project structure:
Three different modules for:   

* Data preprocessing tasks (preprocessing.py) 
* Model training tasks (./scripts/training.py)
* Data visualization tasks (./scripts/visualization.py)

The main script should simply call functions from these modules in an easily readable and understandable way.

As visualizations are necessary at each step of the analysis, the other two modules will depend on the visualization.py. However, there are no other dependencies. 

The preprocessing step was divided into two tasks (one function per task), with the second one including visualization.
Any important parameters can be set from the main script.   
The training step will work in a similar way once completed.
