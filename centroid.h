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


template<typename T, typename Distance>
struct KMeansPlusPlusUpdateStrategy : UpdateStrategy<T> {
    void updateCentroids(std::vector<DataPoint<T>>& centroids, 
                         const std::vector<DataPoint<T>>& dataPoints, 
                         const std::vector<int>& assignments, 
                         int k) override {
        std::random_device rd;
        std::mt19937 gen(rd());
        for (int clusterIdx = 0; clusterIdx < k; ++clusterIdx) {
            std::vector<T> distances(dataPoints.size());
            for (int i = 0; i < assignments.size(); ++i) {
                if (assignments[i] == clusterIdx) {
                    T dist = Distance::distance(dataPoints[i], centroids[clusterIdx]);
                    distances[i] = dist * dist;
                }
            }
            T totalDistance = std::accumulate(distances.begin(), distances.end(), static_cast<T>(0));
            std::uniform_real_distribution<> distrib(0, totalDistance);
            T rnd = distrib(gen);
            
            for (int i = 0; i < assignments.size(); ++i) {
                if (assignments[i] == clusterIdx) {
                    if ((rnd -= distances[i]) > 0) continue;
                    centroids[clusterIdx] = dataPoints[i];
                    break;
                }
            }
        }
    }
};


#endif // CENTROID_H
