
#include <gtest/gtest.h>
#include <fstream>
#include <vector>
#include <chrono>
#include <filesystem>
#include <cmath>
#include <iostream>

int add(int a, int b) {
  return a + b;
}
 
// Google Test 测试用例

//多个测试用例不需要数据共享
TEST(TEST_ADD, UNSIGNED_INT_VALUE) {  
  int result = add(100, 200);  
  EXPECT_EQ(result, 300);  
  result = add(200, 300);  
  EXPECT_NE(result, 400);
}

//多个测试用例需要数据共享
class Student {
public:
  Student(int id, std::string name): id_(id), name_(name) {};
  ~Student() = default;
  void SetAge(int age) { age_ = age; }
  int GetAge() const { return this->age_; }
  void SetScore(int score) { score_ = score; }
  int GetScore() const { return this->score_; }
private:
  int id_;
  std::string name_;
  int age_;
  int score_;
};
class StudentTest : public testing::Test {
protected:
  void SetUp() override {
    student = new Student(1234, "Tom");
  }
  void TearDown() override {
    delete student;
  }
  Student* student;
};
TEST_F(StudentTest, SET_AGE_TEST) {
  student->SetAge(16);
  int age = student->GetAge();
  EXPECT_EQ(age, 16);
}
TEST_F(StudentTest, SET_SCORE_TEST) {
  student->SetScore(99);
  int score = student->GetScore();
  ASSERT_EQ(score, 99);
}

//行为取决于参数，暂时没啥用
// struct TestData {
//   int  a;
//   int  b;
//   int  result;
//   char type;
// };
// class CalculateTest : public ::testing::TestWithParam<TestData> {
// protected:
//   void checkData() {
//     int a = GetParam().a;
//     int b = GetParam().b;
//     int result = GetParam().result;
//     switch (GetParam().type) {
//       case '+':
//         EXPECT_EQ(a + b, result);
//         break;
//       case '-':
//         EXPECT_EQ(a - b, result);
//         break;
//       case '*':
//         EXPECT_EQ(a * b, result);
//         break;
//       case '/':
//         EXPECT_EQ(a / b, result);
//         break;
//       default:
//         break;
//     }
//   }
// };
// TEST_P(CalculateTest, Test) {
//   checkData();
// }
// INSTANTIATE_TEST_SUITE_P(TestMyClassParams,
//                          CalculateTest,
//                          ::testing::Values(
//                            TestData{100, 200, 300, '+'},
//                            TestData{20, 5, 15, '-'},
//                            TestData{5, 6, 30, '*'},
//                            TestData{8, 2, 3, '/'}
//                          ));


int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}