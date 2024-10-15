#include <iostream>  
#include <fstream>  
#include <sstream>  
#include <vector>  
#include <string>  
  
using namespace std;  
  
vector<int> read_csv() {  
    ifstream inFile("IRIS.csv");  
    string lineStr;  
    vector<int> firstColumnData; // 用于存储第一列的数据  
  
    if (!inFile.is_open()) {  
        cerr << "无法打开文件" << endl;   
		return firstColumnData;
    }  
  
    // 逐行读取文件  
    while (getline(inFile, lineStr)) {  
        stringstream ss(lineStr);  
        string str;  
  
        // 读取第一列的数据  
        if (getline(ss, str, ',')) { // 假设第一列之后有逗号  
            try {  
                int num = stod(str); // 尝试将字符串转换为double  
                firstColumnData.push_back(num); // 存储到vector中  
            } catch (const std::invalid_argument& e) {  
                cerr << "转换错误: " << e.what() << endl;  
                continue; // 忽略当前行，继续下一行  
            } catch (const std::out_of_range& e) {  
                cerr << "范围错误: " << e.what() << endl;  
                continue; // 忽略当前行，继续下一行  
            }  
        }  
    }  
  
    inFile.close(); // 关闭文件  
  
    // 输出第一列的数据（可选）  
    for (double num : firstColumnData) {  
        cout << num << endl;  
    }  

}