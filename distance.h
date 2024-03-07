#ifndef DISTANCE_H
#define DISTANCE_H

#include <cmath>
#include <numeric>

// Distance strategy interface
template<typename T>
struct DistanceStrategy {
    static T distance(const DataPoint<T>& a, const DataPoint<T>& b);
};

// Euclidean distance strategy
template<typename T>
struct EuclideanDistance : DistanceStrategy<T> {
    static T distance(const DataPoint<T>& a, const DataPoint<T>& b) {
        T sum = 0;
        for (size_t i = 0; i < a.getCoordinates().size(); ++i) {
            sum += std::pow(a.getCoordinates()[i] - b.getCoordinates()[i], 2);
        }
        return std::sqrt(sum);
    }
};

template<typename T>
struct CosineDistance : DistanceStrategy<T> {
    static T distance(const DataPoint<T>& a, const DataPoint<T>& b) {
        T dotProduct = 0;
        T normA = 0;
        T normB = 0;
        for (size_t i = 0; i < a.getCoordinates().size(); ++i) {
            dotProduct += a.getCoordinates()[i] * b.getCoordinates()[i];
            normA += std::pow(a.getCoordinates()[i], 2);
            normB += std::pow(b.getCoordinates()[i], 2);
        }
        normA = std::sqrt(normA);
        normB = std::sqrt(normB);
        return 1 - (dotProduct / (normA * normB)); // 1 - similarity for distance
    }
};

#endif // DISTANCE_H