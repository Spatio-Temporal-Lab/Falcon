import csv

def extract_float_values(input_file):
    float_values = []  # 存储所有浮点数值的列表
    
    with open(input_file, 'r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # 跳过标题行
        
        for row in reader:
            for value in row:
                # 跳过空值并转换有效数字
                if value.strip():  # 确保非空字符串
                    try:
                        num = float(value)
                        float_values.append(num)
                    except ValueError:
                        # 忽略无法转换为浮点数的值
                        continue
    return float_values

if __name__ == "__main__":
    input_filename = "new_tsbs//float_data_complete.csv"  # 替换为你的CSV文件名
    output_filename = "new_tsbs//chunk.csv"
    
    # 提取所有浮点值
    extracted_data = extract_float_values(input_filename)
    
    # 保存为单列CSV文件
    with open(output_filename, 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(["floats"])  # 列标题
        for value in extracted_data:
            writer.writerow([value])
    
    print(f"完成! 共提取 {len(extracted_data)} 个浮点数")
    print(f"结果已保存到: {output_filename}")