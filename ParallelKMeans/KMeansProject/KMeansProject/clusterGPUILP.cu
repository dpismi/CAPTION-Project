// GPU implementation of Big Data Clustering
// Adhi Prahara. Universitas Ahmad Dahlan. 2017

#pragma warning(disable:4996)

#include "clusterCPUILP.h"
#include "clusterGPUILP.cuh"

// GPUMiner CODE /////////////////////

// set variable to array
void BenMemset(void *buf, int c, size_t size)
{
	memset(buf, c, size);
}

// memory allocation with zeros
void *BenMalloc(size_t size)
{
	void *buf = malloc(size);
	BenMemset(buf, 0, size);
	return buf;
}

// read input data
void getInput(char *fileName, int *n, int *m, int *k, int *dim, float *threshold, float **h_datapt)
{
	FILE* fin = fopen( fileName, "r");
	fscanf(fin, "%d%d%d", n, k, dim);
	fscanf(fin, "%d%f", m, threshold);
	float *tmppt = (float*) BenMalloc(sizeof(float)*(*n)*(*dim));
	
	//printf("#object:%d, #k:%d, #dim:%d\n", *n, *k, *dim);

	char buf[1024];
	size_t cur = 0;
	fgets(buf, 1024, fin);
	while (fgets(buf, 1024, fin) != NULL)
	{
		 char * pch;
		 pch = strtok (buf," ");
		 while (pch != NULL)
		{
			//printf("%s", pch);
			float num = (float)atof(pch);
			memcpy(tmppt+cur, &num, sizeof(float));
			//printf("%f\n", tmppt[cur]);		
			pch = strtok (NULL, " ");
			cur++;
		}
	}

	*h_datapt = tmppt;
	fclose(fin);
}
// GPUMiner CODE ///////////////////////

// get device information
void getDeviceInfo()
{
	// get device properties and print to console
	cudaDeviceProp prop;
	checkCudaErrors(cudaGetDeviceProperties(&prop, 0));
	printf("Device name : %s \n", prop.name);
	printf("Global memory : %d \n", prop.totalGlobalMem);
	printf("Compute capability : %d.%d \n", prop.major, prop.minor);
	printf("Multiprocessor count : %d \n\n", prop.multiProcessorCount);
	printf("Warp size : %d \n", prop.warpSize);
	printf("Max threads per block : %d \n", prop.maxThreadsPerBlock);
	printf("Max threads dim : [%d, %d, %d] \n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
	printf("Max grid size : [%d, %d, %d] \n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
}

// perform k-means clustering
void kmeansGPU(
	float *data,		// data
	float *centers,		// centers
	int *labels,		// labels
	int N,				// number of data
	int D,				// number of data dimension
	int K,				// number of cluster
	float threshold,	// distance threshold
	int &niter,			// iteration
	float &timer		// timer (ms)
	)
{
	// cuda event timer
	cudaEvent_t start, stop;
	checkCudaErrors(cudaEventCreate(&start));
	checkCudaErrors(cudaEventCreate(&stop));

	// start timer 
	checkCudaErrors(cudaEventRecord(start, 0));

	// use the first cuda device
	checkCudaErrors(cudaSetDevice(0));

	// transfer data from host to device memory
	int *glabels;				// labels
	float *gdata, *gcenters;	// data, centers
	checkCudaErrors(cudaMalloc((void **)&glabels, N*sizeof(int)));
	checkCudaErrors(cudaMalloc((void **)&gdata, N*D*sizeof(float)));
	checkCudaErrors(cudaMalloc((void **)&gcenters, K*D*sizeof(float)));
	//
	checkCudaErrors(cudaMemcpy(gdata, data, N*D*sizeof(float), cudaMemcpyHostToDevice));		// data
	checkCudaErrors(cudaMemcpy(glabels, labels, N*sizeof(int), cudaMemcpyHostToDevice));		// labels
	checkCudaErrors(cudaMemcpy(gcenters, centers, K*D*sizeof(float), cudaMemcpyHostToDevice));	// centers

	float *gncenters;	// number of data from each cluster
	float *gtcenters;	// sum of data from each cluster
	checkCudaErrors(cudaMalloc((void **)&gncenters, K*sizeof(float)));
	checkCudaErrors(cudaMalloc((void **)&gtcenters, K*D*sizeof(float)));

	int iter = 0;
	bool sflag = true;
	int gridC = 1 + (K >> LOG_BLOCKDIM);
	int gridD = 1 + (N >> LOG_BLOCKDIM);
	gridC = gridC > MAXBLOCKDIM ? MAXBLOCKDIM : gridC;
	gridD = gridD > MAXBLOCKDIM ? MAXBLOCKDIM : gridD;
	while (sflag == true && iter < niter)
	{
		// reset global mem to zero
		sflag = false;
		checkCudaErrors(cudaMemcpyToSymbol(gflag, &sflag, sizeof(bool), 0U, cudaMemcpyHostToDevice));

		checkCudaErrors(cudaMemset(gncenters, 0, K*sizeof(float)));
		checkCudaErrors(cudaMemset(gtcenters, 0, K*D*sizeof(float)));

		// perform kmeans clustering
		computeClosestDistance<<<gridD, BLOCKDIM, (1+2*D)*K*sizeof(float)>>>(gdata, gcenters, gtcenters, gncenters, glabels, N, D, K);
		computeCentroids<<<gridC, BLOCKDIM>>>(gdata, gcenters, gtcenters, gncenters, N, D, K, threshold);

		// check convergent
		checkCudaErrors(cudaMemcpyFromSymbol(&sflag, gflag, sizeof(bool), 0U, cudaMemcpyDeviceToHost));

		iter++;
	}
	niter = iter;

	// transfer the result to host
	checkCudaErrors(cudaMemcpy(labels, glabels, N*sizeof(int), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(centers, gcenters, K*D*sizeof(float), cudaMemcpyDeviceToHost));

	// calculate elapsed time
	checkCudaErrors(cudaEventRecord(stop));
	checkCudaErrors(cudaEventSynchronize(stop));
	checkCudaErrors(cudaEventElapsedTime(&timer, start, stop));

	// release everything
	checkCudaErrors(cudaEventDestroy(start));
	checkCudaErrors(cudaEventDestroy(stop));
	if (gdata != NULL) checkCudaErrors(cudaFree(gdata));
	if (glabels != NULL) checkCudaErrors(cudaFree(glabels));
	if (gcenters != NULL) checkCudaErrors(cudaFree(gcenters));
	if (gtcenters != NULL) checkCudaErrors(cudaFree(gtcenters));
	if (gncenters != NULL) checkCudaErrors(cudaFree(gncenters));
}

// main function
int main(int argc, char **argv) 
{
	// timer for CPU
	clock_t start, end;
	int niter = 10000;
	int iterKMeans = niter;
	double timeKMeansCPU = 0; 

	//// get data dimension
	//int K = 10;
	//int rows = 0, cols = 0;
	//string path = "fcmdata2.dat";
	//getFileDimension(path, rows, cols);

	//// read data from file
	//float *data = new float[ndata];
	//readFile(path, data, '\t', rows, cols);
	
	float *data;
	float threshold;
	int K;
	int rows = 0, cols = 0;
	char *fileName = "inputfiles/50K";
	getInput(fileName, &rows, &niter, &K, &cols, &threshold, &data);
	iterKMeans = niter;

	// show device information
	getDeviceInfo();

	// clustering variables
	int *labels = new int[rows];
	float *centerKMeans, *initCenters;
	
	// initialize cluster centers
	int *fcent = new int[K];
	initCenters = new float[K*cols];
	getRandomCenters(data, initCenters, fcent, rows, cols, K);
	
	// assign initial cluster centers
	centerKMeans = new float[K*cols];
	for (int k=0; k<K; k++)
	{
		for (int i=0; i<cols; i++)
			centerKMeans[i + k*cols] = data[i + k*cols];//initCenters[i + k*cols];//
	}
	
	// initialize labels
	for (int i=0; i<rows; i++)
		labels[i] = 0;

	// perform kmeans on CPU
	start = std::clock();
	
	iterKMeans = 2;
	kmeansCPU(data, centerKMeans, labels, rows, cols, K, threshold, iterKMeans);

	end = std::clock();
	timeKMeansCPU = (end - start) / (double)CLOCKS_PER_SEC;

	// show clustering performance
	printf("\nPerformance Information:\n");
	printf("Clustering with K = %d on %d x %d data\n\n", K, rows, cols);
	printf("KMeans CPU finished on %.6f seconds with %d iterations\n", timeKMeansCPU, iterKMeans);

	//check data
	int *sumlabels = new int[K];
	for (int k=0; k<K; k++) sumlabels[k] = 0;
	for (int i=0; i<rows; i++) sumlabels[labels[i]]++;

	string filename = "testCPU.txt";
	ofstream savefile;
	savefile.open(filename);
	savefile << "Performance Information:\n";
	savefile << "Clustering with K = " << K << " on " << rows << " x " << cols << "\n\n";
	savefile << "KMeans CPU finished on " << timeKMeansCPU << " seconds with " << iterKMeans << " iterations\n\n";
	for (int i=0; i<K; i++)
	{
		savefile << sumlabels[i] << " ";
		if (i==K-1) savefile << "\n\n";
	}
	for (int i=0; i<K; i++)
	{
		for (int j=0; j<cols; j++)
			savefile << centerKMeans[j + i*cols] << " ";
		if (i<K-1) savefile << "\n";
	}
	savefile.close();

	///////////////////////////////////////////////////////////////////////////////////////
	// GPU implementation
	iterKMeans = 2;
	float timeKMeansGPU = 0;
	
	// initialize cluster centers
	for (int k=0; k<K; k++)
	{
		for (int i=0; i<cols; i++)
			centerKMeans[i + k*cols] = data[i + k*cols];//initCenters[i + k*cols];
	}
	
	// initialize labels
	for (int i=0; i<rows; i++)
		labels[i] = 0;

	// perform k-means on GPU 
	kmeansGPU(data, centerKMeans, labels, rows, cols, K, threshold, iterKMeans, timeKMeansGPU);
	
	// show clustering performance
	printf("KMeans GPU finished on %.6f seconds with %d iterations\n", timeKMeansGPU/1000, iterKMeans);

	//check data
	for (int k=0; k<K; k++) sumlabels[k] = 0;
	for (int i=0; i<rows; i++) sumlabels[labels[i]]++;

	// check data
	filename = "testGPU.txt";
	savefile.open(filename);
	savefile << "Performance Information:\n";
	savefile << "Clustering with K = " << K << " on " << rows << " x " << cols << "\n\n";
	savefile << "KMeans GPU finished on " << timeKMeansGPU/1000.0f << " seconds with " << iterKMeans << " iterations\n\n";
	for (int i=0; i<K; i++)
	{
		savefile << sumlabels[i] << " ";
		if (i==K-1) savefile << "\n\n";
	}
	for (int i=0; i<K; i++)
	{
		for (int j=0; j<cols; j++)
			savefile << centerKMeans[j + i*cols] << " ";
		if (i<K-1) savefile << "\n";
	}
	savefile.close();
	
	return 0;
}