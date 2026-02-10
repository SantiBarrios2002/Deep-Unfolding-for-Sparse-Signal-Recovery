#pragma once

#include <fstream>
#include <string>
#include <vector>
#include <sstream>

namespace unfolding {

class CsvWriter {
public:
    explicit CsvWriter(const std::string& filepath)
        : file_(filepath) {}

    ~CsvWriter() { if (file_.is_open()) file_.close(); }

    bool is_open() const { return file_.is_open(); }

    void write_header(const std::vector<std::string>& columns) {
        write_row_impl(columns);
    }

    void write_row(const std::vector<double>& values) {
        std::vector<std::string> strs;
        strs.reserve(values.size());
        for (double v : values) {
            std::ostringstream oss;
            oss << v;
            strs.push_back(oss.str());
        }
        write_row_impl(strs);
    }

    void write_row(const std::vector<std::string>& values) {
        write_row_impl(values);
    }

private:
    std::ofstream file_;

    void write_row_impl(const std::vector<std::string>& values) {
        for (size_t i = 0; i < values.size(); ++i) {
            if (i > 0) file_ << ",";
            file_ << values[i];
        }
        file_ << "\n";
    }
};

}  // namespace unfolding
