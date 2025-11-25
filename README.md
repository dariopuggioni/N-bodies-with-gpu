This project is a simulation of the n bodies prolem. 
Two serial codes are written in C (one with boundary periodic conditions and one without periodic conditions).
Then a parallelized code for gpu is written in CUDA (configurable in the first lines of the code).
Compile the cuda code, naming "nbodies" the executable file, with 
nvcc -O3 nbodies.cu -o nbodies -lm -ccbin g++-10
In the final part of the code a part can be (un)commented in orded to generate an output file (csv or xyz) for the trajectories and energy. 
Another version of the presentation (with gif animated) is at https://www.canva.com/design/DAG5LKavp6k/YCNqhAQ2R_qQsHHDf4Uu5g/edit
