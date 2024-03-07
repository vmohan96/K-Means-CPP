#ifndef CENTROID_H
#define CENTROID_H

#include <vector>
#include <mutex>
#include <iostream>
#include "dataPoint.h"
#include "distance.h"

template<typename T>
struct UpdateStrategy {
    virtual void updateCentroids(std::vector<DataPoint<T>>& centroids, 
                                 const std::vector<DataPoint<T>>& dataPoints, 
                                 const std::vector<int>& assignments, 
                                 int k) = 0;
};

template<typename T>
struct LloydsUpdateStrategy : UpdateStrategy<T> {
    void updateCentroids(std::vector<DataPoint<T>>& centroids, 
                         const std::vector<DataPoint<T>>& dataPoints, 
                         const std::vector<int>& assignments, 
                         int k) override {
        std::vector<std::vector<T>> sums(k, std::vector<T>(dataPoints[0].getCoordinates().size(), 0));
        std::vector<int> counts(k, 0);

        for (int i = 0; i < assignments.size(); ++i) {
            int cluster = assignments[i];
            counts[cluster] ++;
            for (int j = 0; j < dataPoints[i].getCoordinates().size(); ++j) {
                sums[cluster][j] += dataPoints[i].getCoordinates()[j];
            }
        }

        for (int i = 0; i < k; ++i) {
            if (counts[i] == 0) continue;
            for (int j = 0; j < sums[i].size(); ++j) {
                sums[i][j] /= counts[i];
            }
            centroids[i].setCoordinates(sums[i]);
        }
    }
};


template<typename T>
struct MedianUpdateStrategy : UpdateStrategy<T> {
    void updateCentroids(std::vector<DataPoint<T>>& centroids, 
                         const std::vector<DataPoint<T>>& dataPoints, 
                         const std::vector<int>& assignments, 
                         int k) override {
        for (int clusterIndex = 0; clusterIndex < k; ++clusterIndex) {
            std::vector<std::vector<T>> dimensionValues(dataPoints[0].getCoordinates().size());

            // Collect all points assigned to the current cluster
            for (int i = 0; i < assignments.size(); ++i) {
                if (assignments[i] == clusterIndex) {
                    for (int dim = 0; dim < dataPoints[i].getCoordinates().size(); ++dim) {
                        dimensionValues[dim].push_back(dataPoints[i].getCoordinates()[dim]);
                    }
                }
            }

            std::vector<T> newCoordinates(dimensionValues.size());
            // Calculate median for each dimension and update the centroid
            for (int dim = 0; dim < dimensionValues.size(); ++dim) {
                std::sort(dimensionValues[dim].begin(), dimensionValues[dim].end());
                size_t midIndex = dimensionValues[dim].size() / 2;
                T median;
                if (dimensionValues[dim].size() % 2 == 0) {
                    median = (dimensionValues[dim][midIndex - 1] + dimensionValues[dim][midIndex]) / 2;
                } else {
                    median = dimensionValues[dim][midIndex];
                }
                newCoordinates[dim] = median;
            }
            centroids[clusterIndex].setCoordinates(newCoordinates);
        }
    }
};



#endif // CENTROID_H
