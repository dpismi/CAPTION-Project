// GPU implementation of k-means clustering algorithm
// DIKTI-PEKERTI 2017. Universitas Ahmad Dahlan - Institut Teknologi Bandung.

#pragma once

#ifndef __KMEANSGPU_H
#define __KMEANSGPU_H

// CUDA header
#include <helper_cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// CUDA kernel parameters
#define GRID_SIZE 256		// grid size
#define BLOCK_SIZE 32		// block size
#define LOG_HALF_WARP 4
#define LOG_NUM_BANKS 4
#define LOG_BLOCKDIM 5
#define LOG_MAXBLOCKDIM 8
#define BLOCKDIM (1<<LOG_BLOCKDIM)
#define MAXBLOCKDIM (1<<LOG_MAXBLOCKDIM)
#define ILP_LEVEL 2		// ILP level

// global variable for termination criterion
__device__ bool gflag;

// kernel to compute distance
// this code is optimized with ILP
__device__ float computeDistance(
	float *data,		// data
	float *centers,		// centroids
	int kid,			// index of centroids
	int xid,			// index of data
	int D,				// number of data dimension
	int tD,				// division of data dimension
	int remD			// remainder after division
	)
{
	//float sumDist = 0;
	//for (int j=0; j<D; j++)
	//{
	//	float diff = (data[j + xid*D] - centers[j + kid*D]);
	//	sumDist += diff*diff;
	//}

	float sumDist = 0;
	float temp[ILP_LEVEL];
	// ILP part
	for (int j=0; j<tD; j++)
	{
		for (int r=0; r<ILP_LEVEL; r++)
		{
			temp[r] = (data[j + r*tD + xid*D] - centers[j + r*tD + kid*D]) * (data[j + r*tD + xid*D] - centers[j + r*tD + kid*D]);			
		}
		for (int r=0; r<ILP_LEVEL; r++)
			sumDist += temp[r];
	}
	// remainder part
	for (int r=0; r<remD; r++)
	{
		sumDist += (data[r + ILP_LEVEL*tD + xid*D] - centers[r + ILP_LEVEL*tD + kid*D])*(data[r + ILP_LEVEL*tD + xid*D] - centers[r + ILP_LEVEL*tD + kid*D]);
	
	}		

	return sumDist;
}

// perform parallelized k-means
// part 1: compute the closest centroid
__global__ void computeClosestDistance(
	float *data,		// data
	float *centers,		// centroids
	float *tcenters,	// sum of data in each cluster
	float *ncenters,	// number of data in each cluster
	int *labels,		// labels
	int N,				// number of data
	int D,				// number of feature/dimension
	int K				// number of cluster
	)
{
	// create a shared memory allocation
	extern __shared__ float smem[];		// dynamic shared memory allocation
	float *scent = &smem[0];			// smem for sum of data in each cluster
	float *tcent = &smem[K*D];			// smem for temporary cluster centers	
	float *snum = &smem[2*K*D];			// smem for number of data in each cluster

	// initialization
	// initialize smem to zero
	// copy centroids to smem
	int tid = threadIdx.x; // index for thread in a block
	while (tid < K)
	{
		snum[tid] = 0;
		tid += blockDim.x;
	}
	//__syncthreads();

	// no need to synchronize because the threads is within a warp
	tid = threadIdx.x; // index for thread in a block
	while (tid < K*D)
	{
		scent[tid] = 0;
		tcent[tid] = centers[tid];
		tid += blockDim.x;
	}
	//__syncthreads();

	//
	int tD = D/ILP_LEVEL;		// division of number of dimension
	int remD = D % ILP_LEVEL;	// remaninder after division
	float temp[ILP_LEVEL];

	// index for thread across block
	int xid = threadIdx.x + blockIdx.x * blockDim.x;
	int xstride = blockDim.x * gridDim.x;
	
	// the threads already given more works
	// the ILP is naturally done	
	while (xid < N)
	{
		int minIndex = 0;
		float minValue = FLT_MAX;
		
		// find the closest distance between a data and centroids 
		// cannot do ILP for this part
		// because the comparation instruction has data dependency
		for (int k=0; k<K; k++) // serial in K
		{
			// compute distance optimized with ILP
			float sumDist = computeDistance(data, tcent, k, xid, D, tD, remD);

			// find the closest distance and label
			minIndex = sumDist < minValue ? k : minIndex;
			minValue = sumDist < minValue ? sumDist : minValue;
		}

		// assign the closest centroid to label
		labels[xid] = minIndex;

		// partial calculation of number of data in each cluster
		atomicAdd(&(snum[minIndex]), 1);

		//for (int j=0; j<D; j++)
		//{
		//	atomicAdd(&(scent[j + minIndex*D]), data[j + xid*D]);
		//}

		// partial calculation of sum of data in each cluster
		// optimized with ILP
		for (int j=0; j<tD; j++)
		{
			for (int r=0; r<ILP_LEVEL; r++)
				atomicAdd(&(scent[j + r*tD + minIndex*D]), data[j + r*tD + xid*D]);
		}
		// remainder part
		for (int r=0; r<remD; r++)
			atomicAdd(&(scent[r + ILP_LEVEL*tD + minIndex*D]), data[r + ILP_LEVEL*tD + xid*D]);

		xid += xstride;	// assign more works to thread
	}
	//__syncthreads();

	// no need to synchronize because the threads is within a warp
	// merge the partial number of data in each cluster to global mem
	tid = threadIdx.x; // index for thread in a block
	while (tid < K)
	{ 
		atomicAdd(&(ncenters[tid]), snum[tid]);
		tid += blockDim.x;	// assign more works to thread
	}
	//__syncthreads();
	
	// no need to synchronize because the threads is within a warp
	// merge the partial sum of data in each cluster to global mem
	tid = threadIdx.x; // index for thread in a block
	while (tid < K*D)
	{ 
		atomicAdd(&(tcenters[tid]), scent[tid]);
		tid += blockDim.x;	// assign more works to thread
	}
}

// perform k-means parallel
// part 2 : new centroids calculation
__global__ void computeCentroids(
	float *data,			// data
	float *centers,			// centers
	float *tcenters,		// sum of data from each cluster
	float *ncenters,		// number of data from each cluster
	int N,					// number of data
	int D,					// number of dimension
	int K,					// number of cluster
	float threshold			// distance threshold
	)
{
	// index for thread across block
	int xid = threadIdx.x + blockIdx.x * blockDim.x;
	int xstride = blockDim.x * gridDim.x;

	int tD = D/ILP_LEVEL;		// number of data dimension after division
	int remD = D % ILP_LEVEL;	// remainder after division

	// the threads already given more works
	// the ILP is naturally done	
	while (xid < K)
	{
		// calculate new centroids
		for (int j=0; j<D; j++)
			tcenters[j + xid*D] = tcenters[j + xid*D] / ncenters[xid];

		// calculate distance between cluster
		float sumDist = computeDistance(centers, tcenters, xid, xid, D, tD, remD);
		sumDist = sqrtf(sumDist);

		// if distance more than threshold
		float thresh = threshold*N;
		if (sumDist > thresh)
		{
			gflag = true;
			for (int j=0; j<D; j++)
				centers[j + xid*D] = tcenters[j + xid*D];
		}

		xid += xstride;	// assign more works on thread
	}
}

#endif