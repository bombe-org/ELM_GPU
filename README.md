# ELM_GPU

The function of this project is to use GPU to execute the ELM algorithm.

## build

our project dependent on the **cuda** and **cula** lib.
first you need to download and install both of them before build it in VS2010.

Then you need to setup the project, as follows:
- c/c++ -> 常规 -> 附加包含目录：C:\ProgramData\NVIDIA Corporation\CUDA Samples\v9.1\common\inc;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.1\include;C:\Program Files\CULA\R17\include
- c/c++ -> 语言 -> OpenMP 支持 ：是
- 链接器-> 常规 -> 附加库目录： C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.1\lib\x64\;C:\Program Files\CULA\R17\lib64
- 链接器-> 输入 -> 附加依赖项：cublas.lib;cudart_static.lib;cula_lapack.lib;

you MUST build the project execute file into x64 MODE.

