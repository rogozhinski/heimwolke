#include <filesystem>
#include <string>
#include <fstream>
#include <iostream>
#include <map>
#include <algorithm>
#include <random>

#include "data_prep.h"


DataPrep::DataPrep (std::string data_dir_arg, float test_share_arg) {
    data_dir = data_dir_arg;
    test_share = test_share_arg;
    this->collect_mapping();
};

void DataPrep::collect_mapping(){
    std::ifstream label_file;
    label_file.open(this->data_dir + "labels.txt");
    if (!label_file.is_open()) {
        std::cout << "Could not open file";
    }
    else {
        std::string line;
        std::string::size_type delimeter_index;
        while (std::getline(label_file, line)) {
            delimeter_index = line.find(" = ");
            this->class_map->insert(
                {line.substr(0, delimeter_index), line.substr(delimeter_index + 3)}
            );
        };  
    };
    label_file.close();
};

void DataPrep::write_pairs(std::vector<std::pair<std::string, std::string>> &pairs, std::string name) {
    std::ofstream pairs_file;
    pairs_file.open(this->data_dir + name + ".csv");
    if (!pairs_file.is_open()) {
        std::cout << "Could not open file";
    }
    else {
        std::vector<std::pair<std::string, std::string>>::iterator iter = pairs.begin();
        while (iter != pairs.end()) {
            pairs_file << (*iter).first + "," + (*iter).second << std::endl;
            iter++;
        };  
    };
    pairs_file.close();
};

void DataPrep::annotate(std::string anno_dir) {
    std::vector<std::string> class_files {};
    std::vector<std::pair<std::string, std::string>> train {};
    std::vector<std::pair<std::string, std::string>> test {};
    std::default_random_engine random_engine (42);
    int threshold_index;
    int counter;
    for (
        std::pair<std::string, std::string> const &class_label_name
         : *(this->class_map)
    ) {
        class_files.clear();
        for (
            auto const &file : std::filesystem::directory_iterator(
                this->data_dir + class_label_name.first
                )
            ) {
            class_files.push_back(file.path().string());
        };
        threshold_index = (int)(class_files.size() * (1.0 - this->test_share));
        std::shuffle(std::begin(class_files), std::end(class_files), random_engine);
        counter = 0;
        for (std::string const &class_file_path : class_files){
            if (counter <= threshold_index) {
                train.push_back(std::make_pair(class_file_path, class_label_name.second));
            }
            else {
                test.push_back(std::make_pair(class_file_path, class_label_name.second));
            }
            counter++;
        };
        class_files.clear();
    };
    this->write_pairs(train, "train");
    this->write_pairs(test, "test");
};

