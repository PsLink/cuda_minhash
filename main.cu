#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <math.h>
#include <time.h>

#define HUGE_PRIME 1000033
//#define HUGE_PRIME 233

__global__ void signReduce(
		int bucketSize,
		int sigSize,
		unsigned int *tmpRes,
		int *d_sig) {

	int tid = blockDim.x*blockIdx.x + threadIdx.x;

	if (tid<sigSize) {
		int tmp = HUGE_PRIME;

		for (int i=tid; i<(bucketSize-1)*sigSize; i+=sigSize)
			if (tmpRes[i] < tmp)
				tmp = tmpRes[i];

		d_sig[tid] = tmp;
	}
}

__global__ void signCurrent(
    int bucketSize,
    int sigSize,
    const int* coeffA,
    const int* coeffB,
    const int *bucket,
    unsigned int *tmpRes) {

    int tid = blockDim.x*blockIdx.x + threadIdx.x;

    if (tid<sigSize*(bucketSize-1)) {
        int pointID = bucket[(tid/sigSize)+1];
        int sigID = tid % sigSize;
        long long t1 = (coeffA[sigID] * (long long)pointID) % HUGE_PRIME;
        int tmp = (t1 + coeffB[sigID]) % HUGE_PRIME;

        tmpRes[tid] = tmp;
    }

}

__global__ void sigCompare(
    int numOfbucket,
    int sigSize,
    const int curbucket,
    const int* d_signatures,
    unsigned int *tmpRes) {

    int tid = blockDim.x*blockIdx.x + threadIdx.x;

    if (tid<numOfbucket*sigSize) {
        //int bucketID = tid % numOfbucket;
        int j = tid / numOfbucket;
        if (d_signatures[curbucket+j*numOfbucket] == d_signatures[tid]){
           tmpRes[tid] = 1; 
        //printf("%d %d %d %d %d\n",bucketID,j,tid,d_signatures[curbucket+j*numOfbucket],d_signatures[tid]);
        }
    }
}


__global__ void sigComReduce(
    int numOfbucket,
    int sigSize,
    int* count,
    unsigned int *tmpRes) {

    int tid = blockDim.x*blockIdx.x + threadIdx.x;

    if (tid<numOfbucket) {
        for (int i=tid; i<sigSize*numOfbucket; i+=numOfbucket)
            count[tid] = count[tid] + tmpRes[i];
    }
}



int main(int argc, char **argv) {
    FILE * fin=fopen("bucket.txt","r");
    FILE * fhash=fopen("hashcoeff.txt","r");
    FILE * fout=fopen("output.txt","w");
    // sscanf(argv[1],"%d",&k);
    clock_t start,end;

    int sigSize = 50;
    int numOfBucket;

    //sscanf(argv[1],"%d",&sigSize);
    //printf("sigSize=%d\n",sigSize);

    int **Bucket,*coeffA,*coeffB,tmp,*signatures;

    // srand((unsigned)time(NULL));


    coeffA = (int *)calloc(sigSize,sizeof(int));
    coeffB = (int *)calloc(sigSize,sizeof(int));

    // data = (float *)calloc(nums*dim,sizeof(float));

    for (int i=0; i<sigSize; i++) {
        fscanf(fhash,"%d",&tmp);
        coeffA[i] = tmp;
    }

    for (int i=0; i<sigSize; i++) {
        fscanf(fhash,"%d",&tmp);
        coeffB[i] = tmp;
    }

    fscanf(fin,"%d",&numOfBucket);

    Bucket = (int **)calloc(numOfBucket,sizeof(int*));

    for (int i=0; i<numOfBucket; i++) {
        fscanf(fin,"%d",&tmp);
        Bucket[i] = (int *)calloc(tmp+1,sizeof(int));
        Bucket[i][0] = tmp+1;
        for (int j=1; j<Bucket[i][0]; j++) {
            fscanf(fin,"%d",&tmp);
            Bucket[i][j] = tmp;
        }
    }

//  Caculate the signatures
    signatures = (int *)calloc(sigSize*numOfBucket,sizeof(int));
    memset(signatures,0,sizeof(signatures));

    int *d_bucket,*d_cA,*d_cB,*d_sig,*tmpSig;
    unsigned int *tmpRes;

    tmpSig = (int *)calloc(sigSize,sizeof(int));

    cudaMalloc((void**)&d_sig, sigSize*sizeof(int));
    cudaMalloc((void**)&d_cA, sigSize*sizeof(int));
    cudaMalloc((void**)&d_cB, sigSize*sizeof(int));
    cudaMemcpy(d_cA, coeffA, sigSize*sizeof(int), cudaMemcpyDefault);
    cudaMemcpy(d_cB, coeffB, sigSize*sizeof(int), cudaMemcpyDefault);

    start = clock();

    for (int i=0; i<numOfBucket; i++) {

       // printf("current bucket: %d %d \n",i,Bucket[i][0]);

        cudaMemset(d_sig, 0, sigSize*sizeof(int));

       // printf("test 1 %d %d \n",i,Bucket[i][0]);

        cudaMalloc((void**)&tmpRes, sigSize*(Bucket[i][0]-1)*sizeof(int));

        cudaMalloc((void**)&d_bucket, Bucket[i][0]*sizeof(int));
        cudaMemcpy(d_bucket, Bucket[i], Bucket[i][0]*sizeof(int), cudaMemcpyDefault);

      //  printf("test 2 %d %d \n",i,Bucket[i][0]);

        int blockSize = 256;
        int gridSize = ((Bucket[i][0]-1)*sigSize+blockSize-1)/blockSize;

        signCurrent<<<gridSize,blockSize>>>(Bucket[i][0],sigSize,d_cA,d_cB,d_bucket,tmpRes);
        signReduce<<<1,blockSize>>>(Bucket[i][0],sigSize,tmpRes,d_sig);

       // printf("test 3 %d %d \n",i,Bucket[i][0]);

        cudaMemcpy(tmpSig, d_sig, sigSize*sizeof(int), cudaMemcpyDeviceToHost);
        //cudaDeviceSynchronize();

        for (int j=0; j<sigSize; j++) {
            fprintf(fout,"%d ",tmpSig[j]);
            signatures[i+j*numOfBucket] = tmpSig[j];
        }
        fprintf(fout,"\n");

        // do something with tmpSig

        cudaFree(d_bucket);
        cudaFree(tmpRes);
    }
    cudaFree(d_cA);
    cudaFree(d_cB);
    cudaFree(d_sig);
    
    end = clock();

    printf("running time for signatures: %.2f\n", (double)(end-start)/CLOCKS_PER_SEC);

    fprintf(fout,"running time for signatures: %.2f\n", (double)(end-start)/CLOCKS_PER_SEC);
  
    printf("\n\n");
    // for (int i=0; i<sigSize*numOfBucket; i++) 
    //     printf("%d ",signatures[i]);
    // printf("\n\n");


//  Compare the signatures
    int *d_signatures,*count,*d_count;

    count = (int *)calloc(numOfBucket,sizeof(int));

    cudaMalloc((void**)&d_signatures, sigSize*numOfBucket*sizeof(int));
    cudaMemcpy(d_signatures, signatures, sigSize*numOfBucket*sizeof(int), cudaMemcpyDefault);
    cudaMalloc((void**)&tmpRes, sigSize*numOfBucket*sizeof(int));
    cudaMalloc((void**)&d_count, numOfBucket*sizeof(int));

    for (int i=0; i<numOfBucket; i++) {
        // printf("\n");
        // for (int j=0; j<sigSize; j++) {
        //     printf("%d ",signatures[i+j*numOfBucket]);
        // }
        // printf("\n");

//        cudaMemcpy(d_sig, tmpSig, sigSize*sizeof(int), cudaMemcpyDefault);
        cudaMemset(tmpRes, 0, sigSize*numOfBucket*sizeof(int));
        cudaMemset(d_count, 0, numOfBucket*sizeof(int));

        int blockSize = 256;
        int gridSize = (numOfBucket*sigSize+blockSize-1)/blockSize;
        //printf("CudaINFO:\n");
        sigCompare<<<gridSize,blockSize>>>(numOfBucket,sigSize,i,d_signatures,tmpRes);
        
        gridSize = (numOfBucket+blockSize-1)/blockSize;
        sigComReduce<<<gridSize,blockSize>>>(numOfBucket,sigSize,d_count,tmpRes);

        cudaMemcpy(count, d_count, numOfBucket*sizeof(int), cudaMemcpyDefault);
        //printf("CountINFO:\n");
        for (int j=i+1; j<numOfBucket; j++)
            if (count[j]>10)
                printf("%d %d %d \n",i,j,count[j]);

    }
    cudaFree(tmpRes);
    cudaFree(d_sig);
   // cudaFree(d_signatures);


    // free(countM);
    free(coeffA);
    free(coeffB);

    fclose(fin);
    fclose(fhash);
    fclose(fout);
    return 0;
}

