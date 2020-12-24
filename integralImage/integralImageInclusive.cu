#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>

#define WIDTH  7
#define HEIGHT 3
#define WIDTH_OUT  8
#define HEIGHT_OUT 4
#define THREADS_PER_BLOCK 32

#define CUDA_CHECK(err) if(err != cudaSuccess)\
    {\
      printf("cudaMalloc returned error %s (code %d) (file %s) (line %d)\n", cudaGetErrorString(err), err, __FILE__, __LINE__);\
    }\

__global__ void transpose(int *output, int *input, int width, int height)
{

    __shared__ int temp[THREADS_PER_BLOCK][THREADS_PER_BLOCK];
    
    int xIndex = blockIdx.x*blockDim.x + threadIdx.x;
    int yIndex = blockIdx.y*blockDim.y + threadIdx.y;
    
    if((xIndex < width) && (yIndex < height)) {
        int id_in = yIndex * width + xIndex;
        temp[threadIdx.y][threadIdx.x] = input[id_in];
    }

    __syncthreads();

    xIndex = blockIdx.y * blockDim.y + threadIdx.x;
    yIndex = blockIdx.x * blockDim.x + threadIdx.y;

    if((xIndex < height) && (yIndex < width)) {
        int id_out = yIndex * height + xIndex;
        output[id_out] = temp[threadIdx.x][threadIdx.y];
    }
}
inline __device__
void PrefixSum(int* output, int* input, int w, int nextpow2)
{
    extern __shared__ int temp[];

    const int tdx = threadIdx.x;
    int offset = 1;
    const int tdx2 = 2*tdx;
    const int tdx2p = tdx2 + 1;

    temp[tdx2] =  tdx2 < w ? input[tdx2] : 0;
    temp[tdx2p] = tdx2p < w ? input[tdx2p] : 0;

    for(int d = nextpow2>>1; d > 0; d >>= 1) {
        __syncthreads();
        if(tdx < d)
        {
            int ai = offset*(tdx2p)-1;
            int bi = offset*(tdx2+2)-1;
            temp[bi] += temp[ai];
        }
        offset *= 2;
    }

    int last = temp[nextpow2 - 1];
    if(tdx == 0) temp[nextpow2 - 1] = 0;

    for(int d = 1; d < nextpow2; d *= 2) {
        offset >>= 1;

        __syncthreads();

        if(tdx < d )
        {
            int ai = offset*(tdx2p)-1;
            int bi = offset*(tdx2+2)-1;
            int t  = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
    }

    __syncthreads();

    if(tdx2 < w)  output[tdx2 - 1] = temp[tdx2];
    if(tdx2p < w) output[tdx2p - 1] = temp[tdx2p];
    if(tdx2p < w) output[w - 1] = last;
}

__global__ void KernPrefixSumRows(int *out, int *in, int height, int width)
{
    const int row = blockIdx.y;
    PrefixSum(out+row*width, in+row*width, width, 2*blockDim.x );
}

__global__ void KernPrefixSumRowsTrans(int *out, int *in, int height, int width)
{
    const int row = blockIdx.y;
    // PrefixSum(out+row*(width+1)+(width+1), in+row*width, width, 2*blockDim.x );
    PrefixSum(out+row*(width+1), in+row*width, width, 2*blockDim.x );
}

void PrefixSumRows(int *out, int *in, int *outT, int height, int width)
{
    dim3 blockDim = dim3( 1, 1);
    // dim3 blockDim = dim3( 16, 1);
    printf("\nceil(width/2.0f) = %f\n", ceil(width/2.0f));
    while(blockDim.x < ceil(width/2.0f)) blockDim.x <<= 1;
    printf("\nblockDim.x = %d\n", blockDim.x);
    dim3 gridDim =  dim3( 1, height );
    KernPrefixSumRows<<<gridDim,blockDim,2*sizeof(int)*blockDim.x>>>(out,in,height,width);
    cudaDeviceSynchronize();

    dim3 gridSize, blockSize;
    gridSize.x   = (int)((width + THREADS_PER_BLOCK - 1)/THREADS_PER_BLOCK);
    gridSize.y   = (int)((height + THREADS_PER_BLOCK - 1)/THREADS_PER_BLOCK);
    blockSize.x  = THREADS_PER_BLOCK;
    blockSize.y  = THREADS_PER_BLOCK;

    transpose<<<gridSize, blockSize>>>(outT, out, width, height);
    cudaDeviceSynchronize();

    memset(out, 0, (HEIGHT+1)*sizeof(int));
    blockDim = dim3( 1, 1);
    while(blockDim.x < ceil((height)/2.0f)) blockDim.x <<= 1;
    gridDim =  dim3( 1, width );
    KernPrefixSumRowsTrans<<<gridDim,blockDim,2*sizeof(int)*blockDim.x>>>(out,outT,width,height);
    cudaDeviceSynchronize();

    gridSize.x   = (int)((height+1 + THREADS_PER_BLOCK - 1)/THREADS_PER_BLOCK);
    gridSize.y   = (int)((width+1 + THREADS_PER_BLOCK - 1)/THREADS_PER_BLOCK);
    blockSize.x  = THREADS_PER_BLOCK;
    blockSize.y  = THREADS_PER_BLOCK;
    transpose<<<gridSize, blockSize>>>(outT, out, height+1, width+1);
    cudaDeviceSynchronize();
}
void ComputeIntegrals(const unsigned char *Img, int *Integral) {

  const int SUM_WIDTH_STEP = WIDTH_OUT;

#define SUM_TYPE int

  int iW     = WIDTH;   // image dimensions
  int iH     = HEIGHT;
  int sW     = WIDTH_OUT;   // sum dimensions

  unsigned char *ImgPtr   = 0;
  SUM_TYPE      *IntegPtr = 0;

  // write zeros to first row
  memset(Integral, 0, WIDTH_OUT*sizeof(int));

//#if WITH_CUDA
//  CudaComputeIntegralImages(Img, Integral, TiltedIntegral, SUM_WIDTH_STEP, cudaComputeStream);
//#else
  {
    int yy=1;
    ImgPtr   = (unsigned char *)(Img + WIDTH*(yy-1));
    IntegPtr = (SUM_TYPE *)(Integral + SUM_WIDTH_STEP*yy);

    SUM_TYPE *IntegPtrA = IntegPtr - 1;
    SUM_TYPE *IntegPtrB = IntegPtr - sW - 1;
    SUM_TYPE *IntegPtrC = IntegPtr - sW;

    *IntegPtr++ = (SUM_TYPE)0.0;

    IntegPtrA++;
    IntegPtrB++;
    IntegPtrC++;

    for(int xx=1; xx<iW; xx++){

      SUM_TYPE fTemp = (SUM_TYPE)*(ImgPtr++);
            
      *IntegPtr++ = fTemp
        + *IntegPtrA++
        - *IntegPtrB++
        + *IntegPtrC++;
    }

    SUM_TYPE fTemp = (SUM_TYPE)*(ImgPtr);

    *IntegPtr = fTemp
      + *IntegPtrA
      - *IntegPtrB
      + *IntegPtrC;
  }

  // compute regular integral and first pass of tilted
  for(int yy=2; yy<=iH; yy++){
    ImgPtr   = (unsigned char *)(Img + WIDTH*(yy-1));
    IntegPtr = (SUM_TYPE *)(Integral + SUM_WIDTH_STEP*yy);

    SUM_TYPE *IntegPtrA = IntegPtr - 1;
    SUM_TYPE *IntegPtrB = IntegPtr - sW - 1;
    SUM_TYPE *IntegPtrC = IntegPtr - sW;

    *IntegPtr++ = (SUM_TYPE)0.0;

    IntegPtrA++;
    IntegPtrB++;
    IntegPtrC++;

    for(int xx=1; xx<iW; xx++){

        SUM_TYPE fTemp = (SUM_TYPE)*(ImgPtr++);

      *IntegPtr++ = fTemp
        + *IntegPtrA++
        - *IntegPtrB++
        + *IntegPtrC++;
    }

    SUM_TYPE fTemp = (SUM_TYPE)*(ImgPtr);

    *IntegPtr = fTemp
      + *IntegPtrA
      - *IntegPtrB
      + *IntegPtrC;

  }
    printf("\n\n");
//#endif
}

int main() {

    unsigned char *Img=0;
    int *ImgInt=0;
    int *Integral=0;
    int *IntegralTransposed=0;
    clock_t start, end;

    CUDA_CHECK( cudaMallocManaged((void **) &Img, WIDTH*HEIGHT) );
    CUDA_CHECK( cudaMallocManaged((void **) &ImgInt, WIDTH*HEIGHT*sizeof(int)) );
    CUDA_CHECK( cudaMallocManaged((void **) &Integral, WIDTH_OUT*HEIGHT_OUT*sizeof(int)) );
    CUDA_CHECK( cudaMallocManaged((void **) &IntegralTransposed, WIDTH_OUT*HEIGHT_OUT*sizeof(int)) );

    for (int i=0; i<WIDTH*HEIGHT; i++)   Img[i] = 1;
    for (int i=0; i<WIDTH*HEIGHT; i++)   ImgInt[i] = 1;
    for (int i=0; i<WIDTH_OUT*HEIGHT_OUT; i++) Integral[i] = 1;
    for (int i=0; i<WIDTH_OUT*HEIGHT_OUT; i++) IntegralTransposed[i] = 1;

   start = clock();
    ComputeIntegrals(Img, Integral);
   end = clock();
   printf("CPU Time Taken: %f\n", ((double)(end-start))/CLOCKS_PER_SEC);
   int  *IntegPtr;
   unsigned char *ImgPtr;

   
   // input
   printf("Input\n\n");
   for (int i=0; i<HEIGHT; i++) {
       for (int j=0; j<WIDTH; j++) {
          
          ImgPtr = Img + i * WIDTH + j;
          printf("%d ", *ImgPtr);
       }
       printf("\n");
    }
    printf("\n\n");
    printf("Output CPU");
    printf("\n\n");
    for (int i=0; i<HEIGHT_OUT; i++) {
       for (int j=0; j<WIDTH_OUT; j++) {
          
          IntegPtr = Integral + i * WIDTH_OUT + j;
          printf("%d ", *IntegPtr);
       }
       printf("\n");
    }
    printf("\n\n");
    printf("OUTPUT GPU");
    printf("\n\n"); 
    for (int i=0; i<WIDTH_OUT*HEIGHT_OUT; i++) Integral[i] = 0;

    //CudaComputeIntegralImages(ImgInt, Integral, IntegralTransposed);
    start = clock();
    PrefixSumRows(Integral, ImgInt, IntegralTransposed, HEIGHT, WIDTH);
    end = clock();
    printf("GPU Time Taken: %f\n", ((double)(end-start))/CLOCKS_PER_SEC);

    for (int i=0; i<HEIGHT_OUT; i++) {
        for (int j=0; j<WIDTH_OUT; j++) {
           
           IntegPtr = IntegralTransposed + i * WIDTH_OUT + j;
           printf("%d ", *IntegPtr);
        }
 
        printf("\n");
 
     }

    cudaFree(Img);
    cudaFree(ImgInt);
    cudaFree(Integral);
    cudaFree(IntegralTransposed);
    return 0;
}