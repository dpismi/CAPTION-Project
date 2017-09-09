// CPU implementation of Big Data Clustering
// Adhi Prahara. Universitas Ahmad Dahlan. 2017
// 1. k-means
// 2. k-medoids
// 3. fuzzy c-means

#pragma once

#ifndef __CLUSTERCPU_H
#define __CLUSTERCPU_H

// standard header for I/O
#include <time.h>
#include <ctime>
#include <string>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <iostream>
#include <iterator>
#include <algorithm>

using namespace std;

typedef unsigned int uint;
typedef unsigned char uchar;
typedef unsigned long ulong;

#define EPSILON 0.00001		// small value
#define KILOBYTE 1024					// kilobyte
#define MEGABYTE 1024*1024				// megabyte
#define GIGABYTE 1024*1024*1024			// gigabyte
#define TERABYTE 1024*1024*1024*1024	// terabyte

// get file dimension
void getFileDimension(
	string path,				// input data location
	int &rows, int &cols		// output data dimension
	)
{
	rows = 0; cols = 0;
	// read number of rows
    ifstream file;
	file.open(path);
	file.unsetf(std::ios_base::skipws);
	rows = count(istream_iterator<char>(file), istream_iterator<char>(), '\n') + 1;
	// return back to the first line
	file.clear();
	file.seekg(0, ios::beg);
	// read number of column
	string line;
	getline(file, line);
	stringstream stream(line);
	cols = distance(std::istream_iterator<std::string>(stream), std::istream_iterator<std::string>());
	// return back to the first line
	file.clear();
	file.seekg(0, ios::beg);
}

// read data from file
void readFile(
	string path,			// input data location
	float *data,			// output data
	char delimiter,			// data delimiter
	int rows, int cols	// data dimension
	)
{
	// read file line by line
	int i = 0; // index of rows
    string line;
    ifstream file;
    file.open(path);
    while (getline(file, line))
	{
        // convert each line into double array
		int j = 0;	// index of cols
		string subline;
		stringstream stream(line);
		while(stream.good())
		{
			getline(stream, subline, delimiter);
			data[j + i*cols] = stof(subline.c_str());
			j++;
		}
		i++;
	}
	// return back to the first line
	file.clear();
	file.seekg(0, ios::beg);
}

// random cluster centers initialization
void getRandomCenters(
	float *data,			// input data
	float *centers,			// output cluster centers
	int *index,			// output centers index
	int rows, int cols,	// data dimension
	int K					// number of cluster
	)
{
	srand(time(NULL));

	// get random index from data
	index = new int[K];
	index[0] = rand() % rows;
	for (int k=1; k<K; k++)
	{
		bool stop = false;
		while (stop == false)
		{
			int i = 0;
			bool status = false;
			while (i<k && status == false)
			{
				index[k] = rand() % rows;
				float subs = 0;
				for (int j=0; j<cols; j++)
					subs += abs(data[j + index[k]*cols] - data[j + index[i]*cols]);
				status = subs == 0 ? true : false;
				i++;
			}
			stop = status == false ? true : false;
		}
	}
	// assign random value from data as initial cluster centers
	for (int k=0; k<K; k++)
	{
		for (int i=0; i<cols; i++)
			centers[i + k*cols] = data[i + index[k]*cols];
	}
}

// perform kmeans clustering on CPU
void kmeansCPU(
	float *data,			// input data
	float *centers,			// output cluster centers
	int *label,			// output cluster labels
	int N, 
	int D,	// data dimension	
	int K,					// number of cluster	
	float threshold,
	int &niter			// number of iteration
	)
{
	// initialize variables
	int *index = new int[N];
	float *nums = new float[K];
	float *sums = new float[K*D];
	
	// perform clustering
	int iter = 0;
	bool stop = true;
	while (stop == true && iter < niter)
	{		
		// reset the temp
		stop = false;
		for (int k=0; k<K; k++)
		{
			nums[k] = 0;
			for (int j=0; j<D; j++)
				sums[j + k*D] = 0;
		}
		// assign data to clusters
		for (int i=0; i<N; i++)
		{
			int minlabel = 0;
			float mindist = FLT_MAX;
			for (int k=0; k<K; k++)
			{
				// calculate distance
				float tdist = 0;
				for (int j=0; j<D; j++)
					tdist = tdist + (data[j + i*D] - centers[j + k*D])*(data[j + i*D] - centers[j + k*D]);
				tdist = sqrt(tdist);
				minlabel = tdist < mindist ? k : minlabel;
				mindist = tdist < mindist ? tdist : mindist;
			}
			// find minimum distance to cluster centers
			label[i] = minlabel;

			// fill the temp
			for (int j=0; j<D; j++)
				sums[j + label[i]*D] = sums[j + label[i]*D] + data[j + i*D];		
			nums[label[i]]++;
		}

		// calculate new centers
		for (int k=0; k<K; k++)
		{
			for (int j=0; j<D; j++)
				sums[j + k*D] = sums[j + k*D] / nums[k];
			//
			float nsums = 0;
			for (int j=0; j<D; j++)
				nsums = nsums + (sums[j + k*D] - centers[j + k*D])*(sums[j + k*D] - centers[j + k*D]);
			nsums = sqrt(nsums);
			float thresh = threshold * N;
			if (nsums > thresh)
			{
				for (int j=0; j<D; j++)
					centers[j + k*D] = sums[j + k*D];
				stop = true;
			}
		}
		//
		iter++;
	}

	niter = iter;
	// cleaning up
	delete(nums); delete(index); delete(sums);
}

#endif