#ifndef KMEANS_H
#define KMEANS_H

#include <vector>
#include <random>
#include <limits>
#include <algorithm>
#include <thread>
#include <mutex>
#include <iostream>

#include "dataPoint.h"
#include "distance.h"
#include "centroid.h"

// pointer to method as template parameter
template<typename T, typename Distance = EuclideanDistance<T>, typename UpdateStrategy = LloydsUpdateStrategy<T>>
class KMeans {
private:
    int k, maxIterations;
    std::vector<DataPoint<T>> centroids;
    UpdateStrategy updateStrategy;

    // Initialize centroids
    void initializeCentroids(const std::vector<DataPoint<T>>& dataPoints) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> distrib(0, dataPoints.size() - 1);

        for(int i = 0; i < k; ++i) {
            int rand_idx = distrib(gen);
            centroids.push_back(dataPoints[rand_idx]);
        }
    }

    // Assign Clusters
    std::vector<int> assignClusters(const std::vector<DataPoint<T>>& dataPoints, int n_threads) {
        std::vector<int> assignments(dataPoints.size());
        std::mutex assignmentsMutex;

        // Assign Chunk of Clusters
        auto assignClusterChunk = [&](int start, int end) {
            for(int i = start; i < end; ++i) {
                T minDistance = std::numeric_limits<T>::max();
                int closestCentroid = 0;
                for(int j = 0; j < k; ++j) {
                    T dist = Distance::distance(dataPoints[i], centroids[j]);
                    if(dist < minDistance) {
                        minDistance = dist;
                        closestCentroid = j;
                    }
                }
                std::lock_guard<std::mutex> guard(assignmentsMutex);
                assignments[i] = closestCentroid;
            }
        };

        // Parallelize cluster assignment by chunking all data
        int numThreads = n_threads;
        std::vector<std::thread> threads;

        int chunkSize = dataPoints.size() / numThreads;

        for(int i = 0; i < numThreads; ++i) {
            int chunkStart = chunkSize * i;
            if (chunkStart > dataPoints.size()) {
                chunkStart = dataPoints.size();
            }
            int chunkEnd = chunkStart + chunkSize;
            if (chunkEnd > dataPoints.size()) {
                chunkEnd = dataPoints.size();
            }

            threads.emplace_back(assignClusterChunk, chunkStart, chunkEnd);
        }

        for(auto& t : threads) {
            t.join();
        }

        return assignments;
    }

    void updateCentroids(const std::vector<DataPoint<T>>& dataPoints, const std::vector<int>& assignments) {
        updateStrategy.updateCentroids(centroids, dataPoints, assignments, k);
    }



public:
    KMeans(int k, int maxIterations) : k(k), maxIterations(maxIterations) {}

    std::vector<int> cluster(const std::vector<DataPoint<T>>& dataPoints, int n_threads) {
        initializeCentroids(dataPoints);

        std::vector<int> assignments;
        for (int iteration = 0; iteration < maxIterations; ++iteration) {
            assignments = assignClusters(dataPoints, n_threads);
            updateCentroids(dataPoints, assignments);
        }

        return assignments;
    }

    void writeToOutput(const std::vector<DataPoint<T>>& dataPoints, const std::vector<int>& assignments, std::string filepath, bool writeToFile) {
        std::unique_ptr<std::ofstream> fileStream;
        std::ostream* output_location;

        if (writeToFile) {
            fileStream = std::make_unique<std::ofstream>(filepath, std::ios::app);
            output_location = fileStream.get();
        } else {
            output_location = &std::cout;
            *output_location << "Result:" << std::endl;
        }

        for (int i = 0; i < k; ++i) {
            // *output_location << "Cluster " << i + 1 << " (Centroid: ";
            // for (const auto& coord : centroids[i].getCoordinates()) {
            //     *output_location << coord << " ";
            // }
            // *output_location << "):" << std::endl;

            for (size_t j = 0; j < assignments.size(); ++j) {
                if (assignments[j] == i) {
                    if (!dataPoints[j].getLabel().empty()) {
                        *output_location << dataPoints[j].getLabel() << ", ";
                    }
                    for (const auto& coord : dataPoints[j].getCoordinates()) {
                        *output_location << coord << ", ";
                    }
                    *output_location << i + 1;
                    *output_location << std::endl;
                }
            }
        }
    }

};

#endif // KMEANS_H
