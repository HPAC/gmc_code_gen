# Code Generator for Generalized Matrix Chains

This project is the artifact accompanying the paper titled *Compilation of Generalized Matrix Chains with Symbolic Sizes* accepted for publication in **The IEEE/ACM International Symposium on Code Generation and Optimization (CGO) 2026**.
Generalized Matrix Chains (GMC) are ubiquitous linear algebra expressions that are commonly evaluated directly or indirectly through the invocation of high-performant kernels (such as those in BLAS/LAPACK).
A GMC has the following form: $\rm{op}(M_1) \rm{op}(M_2) \cdots \rm{op}(M_n)$, where $M_i$ has size $q_{i-1} \times q_i$, may exhibit different features (e.g., symmetric, triangular, positive-definite), and can be subjected to unary operators, such as transposition and inversion.

The central part of the artifact is the code generator itself.
The code generator is designed to function in the following way: First, at compile-time, it takes a GMC with symbolic sizes and generates code that *provably performs well* for any combination of matrix sizes in the chain; Second, at run-time, the produced code is used by the (prospective) user-code to evaluate a particular *instance* (combination of operand sizes) of the GMC that was given as input to the code generator.
If you are a user of the code generator, the workflow is the following:

1. Feed the code generator with your GMC to evaluate and produce the corresponding code.
2. Take the output code and integrate it into your codebase using the corresponding interface functions.

The generated code comes in the form of a set of *variants* and some small functions that enable the selection at run-time of the most suitable variant for the sizes at hand.
In our codebase, variants are represented by the class `Algorithm` and the generated code contains one function per selected variant at compile-time.
Each such function is a sequence of invocations to kernels in BLAS/LAPACK and in the extended set of kernels that comes with this repository.

## Requirements

* A compiler that supports C++17. The system has been successfully built on both MacOS and Linux with the GCC-13.3.0 C++ compiler.
* CMake 3.15.
* OpenMP support (should come with any relatively modern compiler with C++17 support).
* OpenBLAS. OpenBLAS 0.3.27 was used in our experiments, but any other version should suffice.
* The C++ library `fmt`. This library can be obtained [here](https://fmt.dev/12.0/get-started/#installation).

If the second experiment in the paper is to be reproduced, then Armadillo is also a dependency. Specifically, Armadillo 14.6.1 was used to produce the results in the paper.

If the figures in the paper are to be replicated, the following are also required:
* Python 3.9 and above
* Matplotlib
* Numpy
* Seaborn

## Setup and Compilation

Clone the repository using:

```bash
git clone git@github.com:HPAC/gmc_code_gen.git
```

The parts of the project are the following:
- An extended set of kernels written in C++ for dense linear algebra operations that complements the functionality offered by BLAS and LAPACK. These are found in the directory `cblas_extended/`. The kernels come in two layers (similar to common implementation of BLAS and LAPACK), a base (found in `cblas_extended/base/`) and a slightly more user-friendly interface (in `cblas_extended/`).

- A code generator that generates C++ code for the evaluation of an input GMC with symbolic sizes. The code generator itself resides in the directory `src/`. The code generator uses abstract representation of kernels, which are located in `src/kernels/`, and some basic performance models, located in `src/models/`. The code generator comes with a frontend, whose implementation is in `src/frontend/`.

- A number of tests to check the functionality of the code generator works as expected, located in `test/`.

- A set of experiments, corresponding to the results presented in the paper. The computation of raw results is done by executables located in `experiments/`. The experiments are thought to be run in a cluster; the corresponding sbatch scripts are found in `jobs/`. Finally, to process the raw data and create figures, we provide Python scripts, to be found in `plotting/`.

When generating code, we recommend the input scripts to be located in the directory `inputs/`.

Use the commands below to build the system if you are going to run on the same machine in which you are compiling:
```bash
cd gmc_code_gen
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release -S .. -B .
cmake --build .
```

If the experiments are not to be replicated, you can disable its compilation from the `CMakeLists.txt` file found in the root of the project, by disabling the option `WITH_EXPERIMENTS`.

If experiments are to be replicated, we assume you have access to a cluster. 
Therefore, we recommend you to compile directly in one of the computing nodes.
To this end, you can find the script `jobs/compile.sbatch` which is to be run from the `build/` directory.
Make sure to edit the script to include common things such as your account number, the target computing nodes, etc (these are inside curly braces).
Once you have edited the sbatch script, use the following after cloning the repository:

```bash
cd gmc_code_gen
mkdir build
cd build
sbatch ../jobs/compile.sbatch
```

Before running any test, experiment, or generating code, make sure to create (if it does not exist) the environment variable `OMP_NUM_THREADS` with the following (where `{NUM_THREADS}` is the number of cores used when kernels are invoked):
```bash
export OMP_NUM_THREADS={NUM_THREADS}
```
We recommend the number of threads is set to the number of physical cores in the processor where programs will be executed.
If you do not know the number of cores in your machine, we recommend you familiarize yourself with `lstopo` and [wikichip](https://en.wikichip.org/wiki/WikiChip).

Once the system is built, we recommend you run the tests in `build/test/` to make sure everything is as it should.
Now, you should generate the performance models that will be used during the expansion of the fanning-out sets of variants.
If you are **not** running on a cluster, do the following from the `build/` directory:

```bash
export OMP_PROC_BIND=true
./experiment/generate_models 20
```
Doing this might take up to an hour.
The argument passed to `generate_models` is the number of repetitions for each measurement.
Feel free to diminish this value to, for example, $10$.

If you are running on a cluster, edit the script `jobs/generate_models.sbatch`.
Once that is done, do the following from the `build/` directory:

```bash
sbatch ../jobs/generate_models.sbatch
```

Once this is done, the models will have been generated and placed in `build/models/`.

## Generating Code for GMCs

We provide an executable as the main entry point for code generation.
The executable is `code_generator/main.cpp` and has 5 positional arguments:

1. `<fname_input>`: String. Path to the file with the input GMC according to the grammar defined in the paper.
2. `<K>`: Positive integer. Maximum number of variants to generate. Should be less than the maximum number of variants for the shape.
3. `<N_t>`: Positive integer. Number of instances to use while choosing the generated sets.
4. `<flops/models>`: String. Whether to use FLOPs or performance models for set expansion.
5. `id_F`: Positive integer. Objective function to minimize while expanding the set of variants. 0 for maximum penalty; 1 for average penalty.

We provide some exemplary inputs to the code generator in `inputs/`.
An example of a full command line use of the code generator is (from the `build/` directory):
```bash
./code_generator/main ../inputs/mc5_w_lower.txt 7 10000 models 1
```

The generated code is found in `build/generated_code/`. 
Two files will be there, both called `GMC_code.cpp/hpp`.
As expected, the `.cpp` file contains the actual code, while the `.hpp` file is there to be `#include`'d by the user code.
The generated code uses an in-house matrix class as of now, whose implementation can be found in `src/utils/dMatrix.cpp`.

## Replicating Experiments

> **Note**: We assume readers of this section are familiar with the contents of the paper, in particular with the experimental section. If you want to check out the paper, please go [here]() (link to be inserted when paper is published)

In the paper, we presented results for two experiments.
The experiments are time-consuming.
Therefore, we strongly suggest you use a cluster/supercomputer if possible.
The execution of both experiments took around $100,000$ core-hours in our experimental setup, which had multiple nodes, each with an Intel Xeon 6132 and 192 GB of RAM.

In the following we specify how to replicate the experiments, the amount of data that is expected as output, and how to process the results into figures.
We have crafted sbatch files (in the directory `jobs`) that are to be used as templates for launching the computation of raw data to be later processed.
You will have to modify the parts left between curly braces in these scripts, since they are dependant on where you launch the executions.

### Experiment with FLOPs 

* **Time consumption**: around $6,000$ core-hours.
* **Amount of output data**: around $90$ GB.

Make sure to be in the directory `build`.
Three major executions have to be launched for this experiment: One for each value of $n$ in the paper.

```bash
sbatch ../jobs/experiment1_n5.sbatch
sbatch ../jobs/experiment1_n6.sbatch
sbatch ../jobs/experiment1_n7.sbatch
```

Each job will launch an array of tasks that will output results to `results/experiment1/n${n}/`.
Each task in the array will generate two output files: one in binary format, named `instances_proc_{ID}.bin` with ratios over optimal in FLOPs for randomly sampled instances; the other one in plain text, named `summary_proc_{ID}.txt` with some metrics that measure the quality of the considered sets in different ways. 
Only the former is needed to reproduce the results in the paper.
The `ID` in the naming corresponds to the task identifier in the array of tasks.

Around $90$ GB of data should have been generated by now.
Now we are going to put the output files together and do some processing.
Being located in the `build/` directory, launch the following sbatch script:

```bash
sbatch ../jobs/process_exp1.sbatch
```
Make sure the previous script loads Python and Numpy before launching the Python process.
Also, since more than $80$ GB of data will be in RAM, make sure enough cores are allocated to the execution of the script (usually the amount of allocated memory is directly proportional to the number of cores allocated to the job).

Once this is done, three files called `trimmed_instances_n{n}.npz` should have been created in `results/experiment1/`.
Finally, let's plot the results.
To do this, we used `matplotlib` and `seaborn`.
A similar plot can be obtained just with the former, but you will have to manually compute the ECDF.
Assuming you have both and Numpy in your environment, do the following from the `build/` directory:

```bash
python3 ../plotting/figure_exp1.py
```

### Experiment with Execution Time
* **Time consumption**: around $90,000$ core-hours.
* **Amount of output data**: around $15$ GB.

For this experiment we provide the shapes that were used to produce our results in `tmp/expr_exp2.txt`.
In that file, each row is a number that represents a specific GMC with symbolic sizes.
During the compilation, this file is copied to `build/tmp/` and also the corresponding C++ code that uses Armadillo to evaluate the expressions therein (`arma_code.cpp/hpp`) is generated and placed in the same directory.

First, we generate the execution times for all the variants for a number of shapes with 7 matrices and a number of instances.
This what consumes the most time out of the whole set of experiments (it took us around $80,000$ core-hours).
For this, we will launch the sbatch script in `jobs/generate_times.sbatch`.
Please, edit the script before, making sure to add the basic information, allocating whole nodes to measure execution time, and loading the appropriate modules.
Most importantly, make sure to fix the variable `OMP_NUM_THREADS` to the number of physical cores in the computing nodes.
From the `build/` directory, launch the execution with the following

```bash
sbatch ../jobs/generate_times.sbatch
```
the generated times will be placed in the directory `build/times/vars/`.

Now, we generate the execution times for Armadillo.
Edit the script `jobs/times_arma.sbatch` same as the previous one. 
From the `build/` directory, do

```bash
sbatch ../jobs/times_arma.sbatch
```

We will generate and evaluate the sets of variants now.
**Important**: the following requires you have built the models.
You can find information about how to build them at the end of the section "Setup and Compilation".
Do the following from the `build/` directory:

```bash
sbatch ../jobs/experiment2.sbatch
```

At the end of the execution you will have four resulting files in `results/experiment2/`.
To generate the figure, we use the two named `ratios_optimal_sets_w`.
From the `build` directory, do the following:

```bash
python3 ../plotting/figure_exp2.py
```
If you are working through ssh, modify the Python script to directly save the figure.
