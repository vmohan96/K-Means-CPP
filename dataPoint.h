#ifndef DATAPOINT_H
#define DATAPOINT_H

#include <vector>
#include <cmath>
#include <string>

template<typename T>
class DataPoint {
private:
    std::vector<T> coordinates;
    std::string label;

public:
    DataPoint(std::vector<T> coords, std::string label = "") : coordinates(std::move(coords)), label(std::move(label)) {}

    // Returns read-only ref to the coords of current DataPoint, faster than a copy
    const std::vector<T>& getCoordinates() const {
        return coordinates;
    }

    void setCoordinates(const std::vector<T>& newCoordinates) {
        coordinates = newCoordinates;
    }

    const std::string& getLabel() const {
        return label;
    }

    // Making static so we can call this function separately on instances of DataPoint
    static T distance(const DataPoint<T>& a, const DataPoint<T>& b) {
        T sum = 0;
        for(size_t i = 0; i < a.coordinates.size(); ++i) {
            sum += std::pow(a.coordinates[i] - b.coordinates[i], 2);
        }
        return std::sqrt(sum);
    }

};

#endif // DATAPOINT_H
